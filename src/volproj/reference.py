"""A literal, loop-based transcription of the assignment's pseudo-code.

It is slow (pure Python) and is used only as an oracle in the tests and for the
comparison in docs/EXPERIMENT.md. The production implementation is
:class:`volproj.projector.PlaneProjector`.
"""

from __future__ import annotations

import math

import numpy as np

from .plane import Plane
from .volume import Volume


def reference_project(volume: Volume, plane: Plane) -> np.ndarray:
    a1 = plane.top_left.astype(float).copy()
    a2 = plane.top_right.astype(float).copy()
    d_ab = plane.leg_length
    d_bc = plane.base_length
    v_ab = (plane.bottom_left - plane.top_left) / np.linalg.norm(plane.bottom_left - plane.top_left)
    v_ac = (plane.bottom_right - plane.top_right) / np.linalg.norm(plane.bottom_right - plane.top_right)
    rows, cols = plane.output_shape
    image = np.zeros((rows, cols), dtype=volume.dtype)
    for y in range(rows):  # y = 0 .. floor(Dab): the apex row and the base row are included
        d12 = float(np.linalg.norm(a2 - a1))
        dp = (a2 - a1) / d12 if d12 > 1e-12 else np.zeros(3)
        p = a1.copy()
        for x1 in range(int(math.floor(d12 + 1e-9)) + 1):
            voxel = np.floor(p + 0.5).astype(int)  # closest voxel
            x = int(math.floor(d_bc / 2 - d12 / 2 + x1 + 0.5))
            if 0 <= x < cols:
                image[y, x] = volume.value_at(*voxel)
            p = p + dp
        a1 = a1 + v_ab
        a2 = a2 + v_ac
    return image
