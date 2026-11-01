import gzip
import os
import tempfile
import unittest

import numpy as np

from tests import helpers  # noqa: F401  (sets up sys.path)
from volproj.nrrd import NrrdError, read_nrrd, write_nrrd
from volproj.volume import Volume

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MRHEAD = os.path.join(REPO_ROOT, "data", "MRHead.nrrd")


class NrrdRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_round_trip_all_types_and_encodings(self):
        rng = np.random.default_rng(1)
        for dtype in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32,
                      np.float64):
            for encoding in ("raw", "gzip", "bzip2", "ascii"):
                for endian in ("little", "big"):
                    with self.subTest(dtype=dtype, encoding=encoding, endian=endian):
                        data = (rng.random((5, 4, 3)) * 100).astype(dtype)
                        write_nrrd(self.path("v.nrrd"), data, encoding=encoding, endian=endian)
                        image = read_nrrd(self.path("v.nrrd"))
                        self.assertEqual(image.data.shape, (5, 4, 3))
                        self.assertEqual(image.data.dtype, np.dtype(dtype))
                        np.testing.assert_array_equal(image.data, data)

    def test_axis_order_first_axis_is_fastest(self):
        # value = x + 10*y + 100*z, so a wrong axis order is immediately visible
        x, y, z = np.meshgrid(np.arange(4), np.arange(3), np.arange(2), indexing="ij")
        data = (x + 10 * y + 100 * z).astype(np.uint16)
        write_nrrd(self.path("o.nrrd"), data, encoding="raw")
        with open(self.path("o.nrrd"), "rb") as f:
            payload = f.read().split(b"\n\n", 1)[1]
        first_values = np.frombuffer(payload, dtype="<u2")[:5].tolist()
        self.assertEqual(first_values, [0, 1, 2, 3, 10])  # x runs fastest on disk
        volume = Volume.from_nrrd(self.path("o.nrrd"))
        self.assertEqual(volume.value_at(3, 2, 1), 123)

    def test_space_fields(self):
        directions = np.diag([0.5, 1.0, 2.0])
        origin = np.array([10.0, -5.0, 3.0])
        write_nrrd(self.path("s.nrrd"), np.zeros((2, 2, 2), np.uint8), space_directions=directions,
                   space_origin=origin)
        image = read_nrrd(self.path("s.nrrd"))
        np.testing.assert_allclose(image.header.space_directions, directions)
        np.testing.assert_allclose(image.header.space_origin, origin)

    def test_detached_data_file(self):
        data = np.arange(24, dtype=np.uint8).reshape((2, 3, 4), order="F")
        with open(self.path("v.raw.gz"), "wb") as f:
            f.write(gzip.compress(data.ravel(order="F").tobytes()))
        with open(self.path("v.nhdr"), "w") as f:
            f.write("NRRD0004\ntype: uchar\ndimension: 3\nsizes: 2 3 4\nencoding: gzip\n"
                    "data file: v.raw.gz\n")
        np.testing.assert_array_equal(read_nrrd(self.path("v.nhdr")).data, data)

    def test_malformed_files_raise(self):
        cases = {
            "bad magic": b"PNG\n\n",
            "missing sizes": b"NRRD0004\ntype: uchar\ndimension: 3\nencoding: raw\n\n",
            "sizes/dimension mismatch": b"NRRD0004\ntype: uchar\ndimension: 3\nsizes: 2 2\n"
                                        b"encoding: raw\n\n\x00\x00\x00\x00",
            "unknown type": b"NRRD0004\ntype: complex\ndimension: 3\nsizes: 1 1 1\n"
                            b"encoding: raw\n\n\x00",
            "truncated data": b"NRRD0004\ntype: uchar\ndimension: 3\nsizes: 2 2 2\n"
                              b"encoding: raw\n\n\x00\x00",
            "bad encoding": b"NRRD0004\ntype: uchar\ndimension: 3\nsizes: 1 1 1\n"
                            b"encoding: zip\n\n\x00",
            "corrupt gzip": b"NRRD0004\ntype: uchar\ndimension: 3\nsizes: 1 1 1\n"
                            b"encoding: gzip\n\nnot gzip",
        }
        for name, content in cases.items():
            with self.subTest(name):
                with open(self.path("bad.nrrd"), "wb") as f:
                    f.write(content)
                with self.assertRaises(NrrdError):
                    read_nrrd(self.path("bad.nrrd"))

    def test_volume_rejects_non_3d(self):
        write_nrrd(self.path("2d.nrrd"), np.zeros((3, 3), np.uint8))
        with self.assertRaises(NrrdError):
            Volume.from_nrrd(self.path("2d.nrrd"))


@unittest.skipUnless(os.path.exists(MRHEAD), "sample MRI not available")
class SampleDataTest(unittest.TestCase):
    def test_mrhead_loads(self):
        volume = Volume.from_nrrd(MRHEAD)
        self.assertEqual(volume.shape, (256, 256, 130))
        self.assertEqual(volume.dtype, np.int16)
        self.assertTrue(volume.has_world_transform)
        # index -> world -> index is the identity
        affine, offset = volume.world_to_index_affine()
        world = volume.index_to_world([10, 20, 30])
        np.testing.assert_allclose(affine @ world + offset, [10, 20, 30], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
