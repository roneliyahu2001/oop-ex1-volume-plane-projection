import time
import unittest

import numpy as np

from tests.helpers import coordinate_volume, decode_coordinate
from volproj.plane import Plane
from volproj.projector import PlaneProjector
from volproj.reference import reference_project
from volproj.volume import Volume


class ProjectorTest(unittest.TestCase):
    def setUp(self):
        self.volume = Volume(coordinate_volume((40, 30, 20)))
        self.projector = PlaneProjector(self.volume)

    def test_rectangle_reproduces_an_axial_slice(self):
        # trapezoid with a1->b and a2->c parallel to y: an axis-aligned slice at z = 7
        plane = Plane.from_points([[0, 0, 7], [39, 0, 7], [0, 29, 7], [39, 29, 7]])
        image = self.projector.project(plane)
        self.assertEqual(image.shape, (30, 40))
        np.testing.assert_array_equal(image, self.volume.data[:, :, 7].T)

    def test_output_dtype_is_preserved(self):
        for dtype in (np.uint8, np.uint16, np.uint32, np.int16, np.float32):
            with self.subTest(dtype=dtype):
                volume = Volume(np.ones((8, 8, 8), dtype=dtype))
                image = PlaneProjector(volume).project(Plane.from_points([[3.5, 0, 4], [0, 7, 4], [7, 7, 4]]))
                self.assertEqual(image.dtype, np.dtype(dtype))

    def test_triangle_mask_apex_and_base(self):
        volume = Volume(np.full((50, 50, 3), 9, dtype=np.uint8))
        plane = Plane.from_points([[20, 0, 1], [0, 40, 1], [40, 40, 1]])
        image = PlaneProjector(volume).project(plane)
        rows, cols = image.shape
        self.assertEqual(cols, 41)
        # apex row: exactly one pixel in the middle; base row: fully covered
        self.assertEqual(np.flatnonzero(image[0]).tolist(), [20])
        self.assertTrue(np.all(image[-1] == 9))
        # the covered width grows monotonically and stays centred
        widths = (image > 0).sum(axis=1)
        self.assertTrue(np.all(np.diff(widths) >= 0))
        for row in image:
            on = np.flatnonzero(row)
            self.assertLessEqual(abs((on[0] + on[-1]) / 2 - 20), 0.5)

    def test_samples_follow_the_plane_geometry(self):
        # every non-zero pixel must decode to a voxel within 0.5 (rounding) of the plane
        a, b, c = np.array([20.0, 1, 2]), np.array([3.0, 25, 15]), np.array([35.0, 25, 15])
        b[0] = 20 - 17
        c[0] = 20 + 17
        plane = Plane.from_points([a, b, c])
        image = self.projector.project(plane)
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        for value in image[image > 0]:
            voxel = np.array(decode_coordinate(value), float)
            self.assertLessEqual(abs(np.dot(voxel - a, normal)), np.sqrt(3) / 2 + 1e-9)

    def test_closest_voxel_rounding(self):
        # a plane at z = 2.4 samples z = 2, at z = 2.6 samples z = 3
        for z, expected in ((2.4, 2), (2.6, 3)):
            plane = Plane.from_points([[0, 0, z], [10, 0, z], [0, 10, z], [10, 10, z]])
            image = self.projector.project(plane)
            self.assertEqual(decode_coordinate(image[5, 5])[2], expected)

    def test_partially_outside_volume_is_zero(self):
        plane = Plane.from_points([[0, -10, 5], [39, -10, 5], [0, 19, 5], [39, 19, 5]])
        image = self.projector.project(plane)
        self.assertTrue(np.all(image[:10] == 0))  # y < 0 -> outside M
        np.testing.assert_array_equal(image[10:], self.volume.data[:, :20, 5].T)

    def test_entirely_outside_volume_is_black(self):
        plane = Plane.from_points([[500, 500, 500], [480, 520, 500], [520, 520, 500]])
        self.assertFalse(np.any(self.projector.project(plane)))

    def test_agrees_with_literal_pseudocode(self):
        # The pseudo-code scatters integer steps x1 into rounded columns, we gather at the
        # exact pixel centre, so a pixel may pick a neighbouring voxel - but never further
        # than one voxel away, and the footprints must coincide.
        rng = np.random.default_rng(7)
        volume = Volume(coordinate_volume((60, 60, 60)) + 1)  # +1: no voxel decodes to 0
        projector = PlaneProjector(volume)
        for trial in range(10):
            with self.subTest(trial=trial):
                a = rng.uniform(10, 50, 3)
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                side = np.cross(direction, rng.normal(size=3))
                side /= np.linalg.norm(side)
                height, half = rng.uniform(15, 40), rng.uniform(8, 25)
                b = a + height * direction - half * side
                c = a + height * direction + half * side
                plane = Plane.from_points([a, b, c])
                fast = projector.project(plane)
                slow = reference_project(volume, plane)
                both = (fast > 0) & (slow > 0)
                self.assertGreater(both.sum() / max((slow > 0).sum(), 1), 0.95)
                fast_vox = np.array([decode_coordinate(v - 1) for v in fast[both]])
                slow_vox = np.array([decode_coordinate(v - 1) for v in slow[both]])
                self.assertLessEqual(np.abs(fast_vox - slow_vox).max(), 1)

    def test_world_coordinates_transform(self):
        # spacing 2 along x: one world unit = half a voxel
        volume = Volume(coordinate_volume((40, 30, 20)), space_directions=np.diag([2.0, 1.0, 1.0]),
                        space_origin=np.array([100.0, 0.0, 0.0]))
        plane = Plane.from_points([[100, 0, 3], [178, 0, 3], [100, 29, 3], [178, 29, 3]])
        image = PlaneProjector(volume).project(plane, volume.world_to_index_affine())
        self.assertEqual(image.shape, (30, 79))  # 78 mm wide -> 79 columns
        self.assertEqual(decode_coordinate(image[4, 0]), (0, 4, 3))
        self.assertEqual(decode_coordinate(image[4, 78]), (39, 4, 3))


class PerformanceTest(unittest.TestCase):
    def test_frame_rate_on_256_cube(self):
        rng = np.random.default_rng(0)
        volume = Volume(rng.integers(0, 255, size=(256, 256, 256), dtype=np.uint8))
        projector = PlaneProjector(volume)
        planes = [Plane.from_points([[127.5, 0, 128], [0, 255, 128 + i], [255, 255, 128 - i]])
                  for i in range(60)]
        start = time.perf_counter()
        for plane in planes:
            projector.project(plane)
        fps = len(planes) / (time.perf_counter() - start)
        self.assertGreater(fps, 30, f"only {fps:.1f} fps")


if __name__ == "__main__":
    unittest.main()
