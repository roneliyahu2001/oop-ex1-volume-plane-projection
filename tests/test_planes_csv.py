import os
import tempfile
import unittest

from tests import helpers  # noqa: F401
from volproj.planes_csv import parse_planes_csv


class PlanesCsvTest(unittest.TestCase):
    def parse(self, text, **kwargs):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(text)
        self.addCleanup(os.remove, f.name)
        return parse_planes_csv(f.name, **kwargs)

    def test_triangles_and_trapezoids(self):
        records = self.parse("5,0,0,0,10,0,10,10,0\n3,0,0,7,0,0,0,10,0,10,10,0\n")
        self.assertEqual([r.is_valid for r in records], [True, True])
        self.assertTrue(records[0].plane.is_triangle)
        self.assertFalse(records[1].plane.is_triangle)

    def test_header_comments_and_blank_lines_are_skipped(self):
        records = self.parse("ax,ay,az,bx,by,bz,cx,cy,cz\n# a comment\n\n5,0,0,0,10,0,10,10,0\n")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].line_number, 4)

    def test_brackets_and_spaces_are_tolerated(self):
        records = self.parse("(5, 0, 0), (0, 10, 0), (10, 10, 0)\n")
        self.assertTrue(records[0].is_valid, records[0].error)

    def test_invalid_lines_become_error_records_in_order(self):
        records = self.parse("5,0,0,0,10,0,10,10,0\n"
                             "1,2,3,4,5,6\n"
                             "5,0,0,x,10,0,10,10,0\n"
                             "5,0,0,0,10,0,13,10,0\n"
                             "5,0,0,0,10,0,10,10,0\n")
        self.assertEqual([r.is_valid for r in records], [True, False, False, False, True])
        self.assertIn("got 6", records[1].error)
        self.assertIn("non-numeric", records[2].error)
        self.assertIn("isosceles", records[3].error)
        self.assertEqual([r.line_number for r in records], [1, 2, 3, 4, 5])

    def test_non_numeric_first_line_with_numbers_is_data_not_header(self):
        records = self.parse("5,0,0,0,10,0,10,10,zz\n")
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].is_valid)

    def test_eps_is_forwarded(self):
        self.assertFalse(self.parse("5,0,0,0,10,0,10.01,10,0\n")[0].is_valid)
        self.assertTrue(self.parse("5,0,0,0,10,0,10.01,10,0\n", eps=0.1)[0].is_valid)


if __name__ == "__main__":
    unittest.main()
