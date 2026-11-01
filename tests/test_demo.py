"""Integration tests for the demo (``python Ex0.py file.nrrd file.csv file.mp4``)."""

import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from tests import helpers  # noqa: F401
from volproj.demo import DemoError, DemoOptions, run_demo
from volproj.nrrd import write_nrrd
from volproj.video import FrameRenderer, IntensityWindow

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_video(path):
    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


class DemoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        volume = np.zeros((64, 64, 64), dtype=np.uint16)
        volume[16:48, 16:48, 16:48] = 1000  # bright cube
        self.nrrd = self.path("cube.nrrd")
        write_nrrd(self.nrrd, volume, space_directions=np.diag([2.0, 2.0, 2.0]),
                   space_origin=np.zeros(3))
        self.csv = self.path("planes.csv")
        with open(self.csv, "w") as f:
            f.write("# header comment\n"
                    "32,0,32, 0,63,32, 64,63,32\n"        # valid triangle
                    "1,2,3\n"                             # invalid: one point
                    "0,0,32, 63,0,32, 0,63,32, 63,63,32\n"  # valid rectangle
                    "32,0,32, 0,63,32, 70,63,32\n")       # invalid: not isosceles
        self.mp4 = self.path("out.mp4")

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_video_has_one_frame_per_plane_including_error_frames(self):
        report = run_demo(self.nrrd, self.csv, self.mp4, DemoOptions(frames_dir=self.path("frames")))
        self.assertEqual(report.frames, 4)
        self.assertEqual(report.valid_frames, 2)
        self.assertEqual([line for line, _ in report.errors], [3, 5])
        frames = read_video(self.mp4)
        self.assertEqual(len(frames), 4)
        self.assertEqual(len(os.listdir(self.path("frames"))), 4)

        exact = [cv2.imread(self.path(f"frames/frame_{i:04d}.png")) for i in range(4)]
        # error frames are dominated by red (BGR: R channel high, B/G low) on black
        for index in (1, 3):
            red = (exact[index][:, :, 2] > 150) & (exact[index][:, :, 0] < 100)
            self.assertGreater(red.mean(), 0.01, f"frame {index} is not an error frame")
        # valid frames show the bright cube in grey levels (no red)
        for index in (0, 2):
            frame = exact[index]
            self.assertGreater((frame[:, :, 0] > 200).mean(), 0.05)
            self.assertTrue(np.array_equal(frame[:-40, :, 0], frame[:-40, :, 2]))

    def test_world_coordinates(self):
        with open(self.csv, "w") as f:  # spacing 2: world 0..126 covers the whole volume
            f.write("0,0,64, 126,0,64, 0,126,64, 126,126,64\n")
        report = run_demo(self.nrrd, self.csv, self.mp4, DemoOptions(coordinates="world"))
        self.assertEqual(report.valid_frames, 1)

    def test_only_invalid_planes_still_produces_a_video(self):
        with open(self.csv, "w") as f:
            f.write("1,2,3\n4,5,6\n")
        report = run_demo(self.nrrd, self.csv, self.mp4)
        self.assertEqual((report.frames, report.valid_frames), (2, 0))
        self.assertEqual(len(read_video(self.mp4)), 2)

    def test_fatal_input_errors(self):
        with open(self.path("empty.csv"), "w") as f:
            f.write("# nothing\n")
        with open(self.path("bad.nrrd"), "wb") as f:
            f.write(b"not an nrrd")
        cases = [(self.path("missing.nrrd"), self.csv, self.mp4),
                 (self.nrrd, self.path("missing.csv"), self.mp4),
                 (self.nrrd, self.path("empty.csv"), self.mp4),
                 (self.path("bad.nrrd"), self.csv, self.mp4),
                 (self.nrrd, self.csv, self.path("no/such/dir/out.mp4"))]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(DemoError):
                run_demo(*args)

    def test_command_line(self):
        result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "Ex0.py"), self.nrrd,
                                 self.csv, self.mp4], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("4 frames (2 valid, 2 error)", result.stdout)
        self.assertTrue(os.path.getsize(self.mp4) > 0)

        result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "Ex0.py"),
                                 self.path("missing.nrrd"), self.csv, self.mp4],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("not found", result.stderr)


class RenderingTest(unittest.TestCase):
    def test_intensity_window(self):
        window = IntensityWindow(100, 200)
        np.testing.assert_array_equal(window.to_uint8(np.array([50, 100, 150, 200, 900])),
                                      [0, 0, 127, 255, 255])
        with self.assertRaises(ValueError):
            IntensityWindow(5, 5)

    def test_renderer_canvas_is_constant_and_even(self):
        renderer = FrameRenderer([(101, 57), (33, 77)], scale=1)
        self.assertEqual(renderer.frame_size[0] % 2, 0)
        self.assertEqual(renderer.frame_size[1] % 2, 0)
        for shape in ((101, 57), (33, 77), (5000, 10)):
            frame = renderer.render_image(np.zeros(shape, np.uint8), "caption")
            self.assertEqual((frame.shape[1], frame.shape[0]), renderer.frame_size)
        error = renderer.render_error("title", "a long message " * 20)
        self.assertEqual((error.shape[1], error.shape[0]), renderer.frame_size)


if __name__ == "__main__":
    unittest.main()
