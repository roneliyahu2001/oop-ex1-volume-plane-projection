# Testing strategy

## Levels

| Level | What | Where |
|---|---|---|
| Unit | NRRD reader, plane validation, projection, CSV parsing, rendering | `tests/test_nrrd.py`, `test_plane.py`, `test_projector.py`, `test_planes_csv.py`, `test_demo.py::RenderingTest` |
| Oracle | Vectorised projector vs. literal transcription of the assignment pseudo-code | `test_projector.py::test_agrees_with_literal_pseudocode` |
| Integration | Whole pipeline NRRD + CSV → mp4, read back with OpenCV | `test_demo.py::DemoTest` |
| System / CLI | `python Ex0.py …` as a subprocess: exit codes, stdout, output file | `test_demo.py::test_command_line` |
| Performance | ≥ 30 fps on `256³` in the test suite; full `500³` benchmark | `test_projector.py::PerformanceTest`, `benchmarks/benchmark.py` |
| Visual / experiment | Videos on real MRI and CT data, checked by eye | `results/`, [EXPERIMENT.md](EXPERIMENT.md) |

Run everything (standard library `unittest`, no extra tools needed; `pytest` also works):

```bash
python -m unittest discover -s tests -t . -v
```

## Test design techniques

* **Self-describing data.** `coordinate_volume()` stores `x + 1000·y + 10⁶·z` in each voxel,
  so every pixel of a projection tells which voxel it came from. This lets tests check the
  geometry directly (e.g. every sampled voxel lies within `√3/2` of the plane).
* **Exact expectations.** An axis-aligned rectangle must reproduce `M[:, :, z]` exactly;
  a plane at `z = 2.4` / `2.6` must sample `z = 2` / `3` (closest voxel).
* **Equivalence classes and boundaries** for validation: within / outside EPS (`10.0004` vs
  `10.01`), configurable EPS, degenerate shapes (coincident, collinear, zero base), 2 / 5 points,
  2D points, NaN / inf / text, top edge longer than base, non-coplanar 4th point, `a1 = a2`.
* **Round trips** for I/O: write → read for 8 types × 4 encodings × 2 byte orders; on-disk axis order
  checked byte by byte; detached headers; 7 kinds of malformed files must raise `NrrdError`.
* **Oracle testing** against the pseudo-code on 10 random oblique triangles: footprints overlap
  ≥ 95 %, and every pixel samples a voxel at most 1 index away from the pseudo-code's choice
  (the two differ only in where along the row the sample is taken, see ALGORITHM.md §1).

## The demo (requirement 7) — what is tested

| Scenario | Expected | Test |
|---|---|---|
| Mixed valid / invalid lines | One frame per line, in order; invalid lines → red error frames; valid frames are grey images | `test_video_has_one_frame_per_plane_including_error_frames` |
| Only invalid lines | Video with only error frames, exit 0 | `test_only_invalid_planes_still_produces_a_video` |
| World coordinates | Valid frame using the NRRD affine | `test_world_coordinates` |
| Missing NRRD / CSV, empty CSV, corrupt NRRD, missing output dir | `DemoError` / exit code 1, message on stderr | `test_fatal_input_errors`, `test_command_line` |
| Frame size | Constant and even (H.264), any image size fits | `test_renderer_canvas_is_constant_and_even` |
| Frame count in the mp4 | Read back with `cv2.VideoCapture` | `read_video()` |

**Error handling policy** (Q&A #3): `Plane.from_points` raises `InvalidPlaneError` for an invalid
point set. The demo catches it per line, so one bad plane never affects the other frames; the
error frame shows the CSV line number and the reason, and the console summary lists them as well.
Problems that make the whole run impossible (no volume, no planes) stop the demo with exit code 1.

## Current result

```
Ran 44 tests in ~0.6s
OK
```
