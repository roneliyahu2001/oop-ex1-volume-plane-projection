"""Shared test utilities. Importing this module puts ``src/`` on ``sys.path``."""

import os
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def coordinate_volume(shape=(40, 30, 20), dtype=np.uint32):
    """Volume whose value encodes its own index: M[x][y][z] = x + 1000*y + 1000000*z."""
    x, y, z = np.meshgrid(*(np.arange(s) for s in shape), indexing="ij")
    return (x + 1000 * y + 1000000 * z).astype(dtype)


def decode_coordinate(value):
    value = int(value)
    return value % 1000, (value // 1000) % 1000, value // 1000000
