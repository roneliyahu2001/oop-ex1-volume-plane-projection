"""Projection of a volume ``M`` onto a :class:`~volproj.plane.Plane` (see docs/ALGORITHM.md)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .plane import Plane
from .volume import Volume

Affine = Tuple[np.ndarray, np.ndarray]  # (matrix, offset): index = matrix @ p + offset


class PlaneProjector:
    """Samples a :class:`Volume` on planes, producing 2D images ``E``.

    The output image is a NumPy array ``E[y, x]`` (row-major image convention; the
    assignment's pseudo-code writes ``E[x][y]``) of the volume's own dtype:

    * row ``y`` (0 = apex / top edge) lies at distance ``y`` from the top along both legs,
    * column ``x`` covers the base width ``Dbc``; each row's segment ``a1(y) -> a2(y)`` is
      centred horizontally, exactly like ``x = Dbc/2 - d12/2 + x1`` in the pseudo-code,
    * every sample takes the value of its *closest voxel*; samples outside ``M`` and
      pixels outside the triangle / trapezoid are 0.

    The implementation is fully vectorised: instead of iterating ``(y, x1)`` and
    scattering into ``E`` (which can leave holes / collisions after rounding) it inverts
    the mapping and computes, for every output pixel, the 3D point it samples.
    """

    def __init__(self, volume: Volume):
        self._volume = volume
        sx, sy, sz = volume.shape
        self._sizes = np.array([sx, sy, sz], dtype=np.uint32)
        self._strides = (1, sx, sx * sy)  # x-fastest flat buffer
        self._buffer = volume.flat_buffer
        self._sentinel = volume.sentinel_index

    @property
    def volume(self) -> Volume:
        return self._volume

    def project(self, plane: Plane, to_index: Optional[Affine] = None) -> np.ndarray:
        """Return the 2D image of the volume on ``plane``.

        ``to_index`` optionally maps the plane's coordinate space (e.g. world / mm) to
        voxel indices; sampling then advances one unit *of the plane's space* per pixel.
        """
        rows, cols = plane.output_shape
        base = plane.base_length

        # --- per-row geometry (rows x 3): the segment a1(y) -> a2(y) -------------------
        y = np.arange(rows, dtype=np.float64)[:, None]
        left_dir = _unit(plane.bottom_left - plane.top_left)
        right_dir = _unit(plane.bottom_right - plane.top_right)
        a1 = plane.top_left + y * left_dir
        a2 = plane.top_right + y * right_dir
        segment = a2 - a1
        d12 = np.linalg.norm(segment, axis=1)
        dp = np.divide(segment, d12[:, None], out=np.zeros_like(segment), where=d12[:, None] > 1e-12)
        offset = (base - d12) / 2.0  # column where x1 = 0

        if to_index is not None:
            matrix, shift = (np.asarray(v, np.float64) for v in to_index)
            a1 = a1 @ matrix.T + shift
            dp = dp @ matrix.T

        # --- per-pixel position along the row segment (rows x cols) --------------------
        x1 = np.arange(cols, dtype=np.float32)[None, :] - offset.astype(np.float32)[:, None]
        d12_col = d12.astype(np.float32)[:, None]
        # a pixel belongs to the shape if its centre is within half a pixel of the segment
        inside = (x1 >= -0.5) & (x1 <= d12_col + 0.5)
        np.clip(x1, 0.0, d12_col, out=x1)

        # --- closest voxel: floor(p + 0.5), per axis, accumulated into a flat index ----
        a1 = (a1 + 0.5).astype(np.float32)
        dp = dp.astype(np.float32)
        flat = np.zeros((rows, cols), dtype=np.int64)
        valid = inside
        coord = np.empty((rows, cols), dtype=np.float32)
        for axis in range(3):
            np.multiply(x1, dp[:, axis:axis + 1], out=coord)
            coord += a1[:, axis:axis + 1]
            np.floor(coord, out=coord)
            index = coord.astype(np.int32)
            # negative values wrap to huge unsigned numbers -> one comparison checks both bounds
            valid &= index.view(np.uint32) < self._sizes[axis]
            if self._strides[axis] == 1:
                flat += index
            else:
                flat += index.astype(np.int64) * self._strides[axis]
        flat[~valid] = self._sentinel
        return self._buffer.take(flat)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else np.zeros(3)
