import unittest

import numpy as np

from tests import helpers  # noqa: F401
from volproj.plane import InvalidPlaneError, Plane


class TrianglePlaneTest(unittest.TestCase):
    def test_valid_triangle(self):
        plane = Plane.from_points([[5, 0, 0], [0, 10, 0], [10, 10, 0]])
        self.assertTrue(plane.is_triangle)
        self.assertAlmostEqual(plane.leg_length, np.sqrt(125))
        self.assertAlmostEqual(plane.base_length, 10)
        self.assertEqual(plane.output_shape, (12, 11))  # floor(11.18)+1, floor(10)+1

    def test_isosceles_within_eps_is_accepted(self):
        Plane.from_points([[5, 0, 0], [0, 10, 0], [10.0004, 10, 0]])

    def test_not_isosceles_raises(self):
        with self.assertRaisesRegex(InvalidPlaneError, "isosceles"):
            Plane.from_points([[5, 0, 0], [0, 10, 0], [10.01, 10, 0]])

    def test_eps_is_configurable(self):
        Plane.from_points([[5, 0, 0], [0, 10, 0], [10.01, 10, 0]], eps=0.1)

    def test_degenerate_triangles_raise(self):
        cases = {
            "all equal": [[1, 1, 1]] * 3,
            "b == c": [[0, 0, 0], [3, 4, 0], [3, 4, 0]],
            "collinear (a is midpoint of bc)": [[0, 0, 0], [-5, 0, 0], [5, 0, 0]],
        }
        for name, pts in cases.items():
            with self.subTest(name), self.assertRaises(InvalidPlaneError):
                Plane.from_points(pts)

    def test_bad_input_raises(self):
        cases = {
            "two points": [[0, 0, 0], [1, 1, 1]],
            "five points": [[0, 0, 0]] * 5,
            "2D points": [[0, 0], [1, 1], [2, 2]],
            "nan": [[0, 0, float("nan")], [0, 10, 0], [10, 10, 0]],
            "inf": [[0, 0, float("inf")], [0, 10, 0], [10, 10, 0]],
            "text": [["a", 0, 0], [0, 10, 0], [10, 10, 0]],
        }
        for name, pts in cases.items():
            with self.subTest(name), self.assertRaises(InvalidPlaneError):
                Plane.from_points(pts)


class TrapezoidPlaneTest(unittest.TestCase):
    def test_valid_isosceles_trapezoid(self):
        plane = Plane.from_points([[3, 0, 0], [7, 0, 0], [0, 10, 0], [10, 10, 0]])
        self.assertFalse(plane.is_triangle)
        self.assertEqual(plane.output_shape, (11, 11))

    def test_rectangle_is_valid(self):
        Plane.from_points([[0, 0, 5], [20, 0, 5], [0, 30, 5], [20, 30, 5]])

    def test_oblique_trapezoid_in_3d(self):
        u = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        w = np.array([0.0, 0.0, 1.0])
        o = np.array([10.0, 10.0, 10.0])
        pts = [o + 2 * u, o + 6 * u, o + 15 * w, o + 8 * u + 15 * w]
        Plane.from_points(pts)

    def test_coincident_top_points_become_triangle(self):
        plane = Plane.from_points([[5, 0, 0], [5, 0, 0], [0, 10, 0], [10, 10, 0]])
        self.assertTrue(plane.is_triangle)

    def test_unequal_legs_raise(self):
        with self.assertRaisesRegex(InvalidPlaneError, "legs"):
            Plane.from_points([[3, 0, 0], [7, 0, 0], [0, 10, 0], [10, 11, 0]])

    def test_top_longer_than_base_raises(self):
        with self.assertRaisesRegex(InvalidPlaneError, "top edge"):
            Plane.from_points([[0, 0, 0], [10, 0, 0], [3, 10, 0], [7, 10, 0]])

    def test_non_coplanar_raises(self):
        with self.assertRaisesRegex(InvalidPlaneError, "coplanar"):
            Plane.from_points([[3, 0, 0], [7, 0, 1], [0, 10, 0], [10, 10, 1]])

    def test_collinear_raises(self):
        with self.assertRaises(InvalidPlaneError):
            Plane.from_points([[0, 0, 0], [1, 0, 0], [-5, 0, 0], [6, 0, 0]])


if __name__ == "__main__":
    unittest.main()
