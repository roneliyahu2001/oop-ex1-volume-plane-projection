"""Plane definitions: an isosceles triangle (3 points) or an isosceles trapezoid (4 points).

Both shapes are represented by one class, :class:`Plane`, as a quad
``(top_left, top_right, bottom_left, bottom_right)``:

* triangle ``(a, b, c)``      -> ``top_left = top_right = a``, ``bottom_left = b``, ``bottom_right = c``
* trapezoid ``(a1, a2, b, c)`` -> legs ``a1 -> b`` and ``a2 -> c``

so the projection algorithm has a single code path (a triangle is a trapezoid with a
zero-length top edge).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

DEFAULT_EPS = 1e-3


class InvalidPlaneError(ValueError):
    """Raised when a set of points does not define a valid projection plane."""


def _dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(q - p))


@dataclass(frozen=True)
class Plane:
    top_left: np.ndarray
    top_right: np.ndarray
    bottom_left: np.ndarray
    bottom_right: np.ndarray
    is_triangle: bool

    # --- factories ------------------------------------------------------
    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]], eps: float = DEFAULT_EPS) -> "Plane":
        """Build and validate a plane from 3 points ``(a, b, c)`` or 4 points ``(a1, a2, b, c)``.

        Raises :class:`InvalidPlaneError` if the points violate the definition (see SRS FR-2).
        """
        try:
            pts = np.array(points, dtype=np.float64)
        except (TypeError, ValueError):
            raise InvalidPlaneError("points must be numeric") from None
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise InvalidPlaneError(f"expected a list of 3D points, got shape {pts.shape}")
        if not np.all(np.isfinite(pts)):
            raise InvalidPlaneError("points must be finite numbers (no NaN / inf)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if len(pts) == 3:
            return cls._triangle(pts[0], pts[1], pts[2], eps)
        if len(pts) == 4:
            return cls._trapezoid(pts[0], pts[1], pts[2], pts[3], eps)
        raise InvalidPlaneError(f"a plane needs 3 or 4 points, got {len(pts)}")

    @classmethod
    def _triangle(cls, a, b, c, eps) -> "Plane":
        d_ab, d_ac, d_bc = _dist(a, b), _dist(a, c), _dist(b, c)
        if abs(d_ab - d_ac) >= eps:
            raise InvalidPlaneError(
                f"not isosceles: |dist(a,b) - dist(a,c)| = {abs(d_ab - d_ac):.6g} >= eps ({eps:g})")
        if d_ab < eps:
            raise InvalidPlaneError("degenerate triangle: a coincides with b and c")
        if d_bc < eps:
            raise InvalidPlaneError("degenerate triangle: b coincides with c (zero-width base)")
        # a, b, c collinear <=> the triangle has (almost) no height
        height = np.linalg.norm(np.cross(b - a, c - a)) / d_bc
        if height < eps:
            raise InvalidPlaneError("degenerate triangle: a, b, c are collinear and span no plane")
        return cls(a, a.copy(), b, c, is_triangle=True)

    @classmethod
    def _trapezoid(cls, a1, a2, b, c, eps) -> "Plane":
        if _dist(a1, a2) < eps:  # the top edge collapsed to a point: it is a triangle
            return cls._triangle((a1 + a2) / 2, b, c, eps)
        d_left, d_right = _dist(a1, b), _dist(a2, c)
        d_top, d_bottom = _dist(a1, a2), _dist(b, c)
        if abs(d_left - d_right) >= eps:
            raise InvalidPlaneError(
                f"legs differ: |dist(a1,b) - dist(a2,c)| = {abs(d_left - d_right):.6g} >= eps ({eps:g})")
        if d_left < eps:
            raise InvalidPlaneError("degenerate trapezoid: zero-length legs")
        if d_top > d_bottom + eps:
            raise InvalidPlaneError(
                f"top edge longer than base: dist(a1,a2) = {d_top:.6g} > dist(b,c) = {d_bottom:.6g}")
        # coplanarity: distance of the 4th point from the plane spanned by the other three
        normal = np.cross(b - a1, c - a1)
        normal_len = np.linalg.norm(normal)
        if normal_len < eps * max(d_left, d_bottom, 1.0):
            normal = np.cross(b - a1, a2 - a1)
            normal_len = np.linalg.norm(normal)
            if normal_len < eps * max(d_left, d_top, 1.0):
                raise InvalidPlaneError("degenerate trapezoid: all points are collinear")
            off_plane = abs(np.dot(c - a1, normal)) / normal_len
        else:
            off_plane = abs(np.dot(a2 - a1, normal)) / normal_len
        if off_plane >= eps:
            raise InvalidPlaneError(f"points are not coplanar (off-plane distance {off_plane:.6g})")
        return cls(a1, a2, b, c, is_triangle=False)

    # --- geometry -------------------------------------------------------
    @property
    def leg_length(self) -> float:
        """``Dab`` - the (common) length of the two legs."""
        return 0.5 * (_dist(self.top_left, self.bottom_left) + _dist(self.top_right, self.bottom_right))

    @property
    def base_length(self) -> float:
        """``Dbc`` - the length of the base edge."""
        return _dist(self.bottom_left, self.bottom_right)

    @property
    def output_shape(self) -> Tuple[int, int]:
        """(rows, columns) of the projected image: rows sample y = 0..floor(Dab), columns x = 0..floor(Dbc)."""
        return int(np.floor(self.leg_length + 1e-9)) + 1, int(np.floor(self.base_length + 1e-9)) + 1

    def points(self) -> np.ndarray:
        if self.is_triangle:
            return np.stack([self.top_left, self.bottom_left, self.bottom_right])
        return np.stack([self.top_left, self.top_right, self.bottom_left, self.bottom_right])
