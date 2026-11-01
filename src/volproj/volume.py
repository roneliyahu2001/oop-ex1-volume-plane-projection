"""The 3D volume ``M`` and its (optional) index <-> world transform."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .nrrd import NrrdError, read_nrrd

MAX_AXIS_SIZE = 2 ** 10  # spec: 1 <= sx, sy, sz < 2^10


class Volume:
    """A 3D scalar raster ``M[x][y][z]`` with 0 <= x < sx, 0 <= y < sy, 0 <= z < sz.

    Internally the voxels are kept in one flat, x-fastest buffer followed by a single
    zero "sentinel" voxel. Any sample that falls outside of ``M`` is redirected to the
    sentinel, which lets the projector gather a whole frame with one ``np.take`` call
    and no masking (see docs/ALGORITHM.md, section "Implementation").
    """

    def __init__(self, data: np.ndarray, space_directions: Optional[np.ndarray] = None,
                 space_origin: Optional[np.ndarray] = None, enforce_size_limit: bool = True):
        data = np.asarray(data)
        if data.ndim != 3:
            raise ValueError(f"volume must be 3D, got shape {data.shape}")
        if data.dtype.kind not in "uif":
            raise ValueError(f"unsupported voxel type {data.dtype}")
        if enforce_size_limit and any(s >= MAX_AXIS_SIZE for s in data.shape):
            raise ValueError(f"each axis must be smaller than {MAX_AXIS_SIZE}, got {data.shape}")
        if data.dtype.byteorder not in ("=", "|"):
            data = data.astype(data.dtype.newbyteorder("="))

        self._shape: Tuple[int, int, int] = tuple(int(s) for s in data.shape)  # type: ignore
        count = data.size
        self._buffer = np.empty(count + 1, dtype=data.dtype)
        self._buffer[:count] = data.ravel(order="F")
        self._buffer[count] = 0
        self._sentinel = count

        self._directions = None if space_directions is None else np.asarray(space_directions, float)
        self._origin = np.zeros(3) if space_origin is None else np.asarray(space_origin, float)

    # --- construction ---------------------------------------------------
    @classmethod
    def from_nrrd(cls, path: str) -> "Volume":
        image = read_nrrd(path)
        if image.data.ndim != 3:
            raise NrrdError(f"expected a 3D volume, got dimension {image.data.ndim}")
        return cls(image.data, image.header.space_directions, image.header.space_origin)

    # --- properties -----------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._buffer.dtype

    @property
    def flat_buffer(self) -> np.ndarray:
        """x-fastest voxel buffer of length sx*sy*sz + 1 (last element is the zero sentinel)."""
        return self._buffer

    @property
    def sentinel_index(self) -> int:
        return self._sentinel

    @property
    def data(self) -> np.ndarray:
        """Read-only view indexed ``[x, y, z]``."""
        view = self._buffer[:self._sentinel].reshape(self._shape, order="F")
        view.flags.writeable = False
        return view

    @property
    def has_world_transform(self) -> bool:
        return self._directions is not None

    # --- voxel access ---------------------------------------------------
    def value_at(self, x: int, y: int, z: int):
        """``M[x][y][z]``, or 0 when the index lies outside of ``M``."""
        sx, sy, sz = self._shape
        if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
            return self._buffer[x + sx * (y + sy * z)]
        return self._buffer.dtype.type(0)

    # --- coordinates ----------------------------------------------------
    def world_to_index_affine(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (A, t) such that ``index = A @ world + t``."""
        if self._directions is None:
            raise ValueError("this volume has no 'space directions'; world coordinates unavailable")
        inverse = np.linalg.inv(self._directions)
        return inverse, -inverse @ self._origin

    def index_to_world(self, index) -> np.ndarray:
        if self._directions is None:
            raise ValueError("this volume has no 'space directions'; world coordinates unavailable")
        return self._directions @ np.asarray(index, float) + self._origin

    def intensity_range(self, low_percentile: float = 0.5, high_percentile: float = 99.5,
                        max_samples: int = 2_000_000) -> Tuple[float, float]:
        """Robust display window computed on a strided subsample of the voxels."""
        voxels = self._buffer[:self._sentinel]
        step = max(1, voxels.size // max_samples)
        sample = voxels[::step]
        low, high = np.percentile(sample, [low_percentile, high_percentile])
        if high <= low:
            low, high = float(sample.min()), float(sample.max())
        if high <= low:
            high = low + 1.0
        return float(low), float(high)
