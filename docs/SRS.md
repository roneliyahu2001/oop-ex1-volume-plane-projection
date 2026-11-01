# Software Requirements Specification (SRS)

**Project:** Ex1 – Plane projection of a 3D volume
**Version:** 1.0

Structure loosely follows IEEE 830 / ISO/IEC/IEEE 29148.

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for a program that computes a 2D image of a 3D raster
volume sampled on an arbitrary plane, and for a demo that turns a list of planes into a video.

### 1.2 Scope
In scope:

* reading 3D scalar volumes in NRRD format (as produced by 3D Slicer),
* defining a plane by 3 points (isosceles triangle) or 4 points (isosceles trapezoid), and validating it,
* projecting (sampling) the volume on the plane with the algorithm in [ALGORITHM.md](ALGORITHM.md),
* a command-line demo `python Ex0.py file.nrrd file.csv file.mp4`,
* documentation, a dataset, tests, a benchmark and an experiment.

Out of scope: a GUI, interpolation beyond the closest voxel ("super-resolution"), multi-channel
(vector / RGB / DTI) volumes, 4D sequences, and writing NRRD files other than for tests.

### 1.3 Definitions

| Term | Meaning |
|---|---|
| `M` | The 3D matrix (volume) of size `sx × sy × sz`; `M[x][y][z]` is one voxel. |
| Voxel / index space | Integer coordinates `0 ≤ x < sx`, `0 ≤ y < sy`, `0 ≤ z < sz`. Voxel *centres* lie on integer coordinates. |
| World space | Physical coordinates from the NRRD header: `world = D · index + origin` (`space directions`, `space origin`). |
| Plane `p` | 3 points `(a, b, c)` or 4 points `(a1, a2, b, c)`. |
| `Dab`, `Dbc` | Leg length `dist(a, b)` and base length `dist(b, c)`. |
| `E` | The output 2D image (matrix). |
| EPS | Validation tolerance, default `0.001`. |
| Frame | One image of the output video (one per CSV line). |
| Error frame | A frame that marks an invalid plane instead of showing an image. |

### 1.4 References
* Assignment text, "Ex1: Algorithm: Implementing a plane projection of 3D volume" (incl. Q&A).
* NRRD format: <http://teem.sourceforge.net/nrrd/format.html>
* 3D Slicer sample data: <https://www.slicer.org/>

---

## 2. Overall description

### 2.1 Product perspective
A standalone Python 3 package (`src/volproj`) plus a CLI script (`Ex0.py`). Dependencies are
deliberately minimal: NumPy (computation) and OpenCV (`opencv-python`, video encoding and
text rendering only). The NRRD reader is implemented in the project (no `pynrrd` dependency).

### 2.2 Product functions
1. Load a volume from an NRRD file.
2. Parse a CSV list of planes.
3. Validate each plane (raise an exception for an invalid point set).
4. Project the volume on each valid plane.
5. Render one frame per plane (error frame for invalid planes) and write an mp4.
6. Report what happened (frames, errors, timing).

### 2.3 Users
Students / graders (run the demo, read the code), and developers who use `PlaneProjector` as a
library from Python.

### 2.4 Constraints
* Language: Python (one of Java / Python / C / C++, as allowed).
* `1 ≤ sx, sy, sz < 2^10`; voxel values are unsigned 8/16/32-bit integers (other NRRD scalar types
  are also accepted, see FR-1).
* Performance: ≥ 30 fps is mandatory, ≥ 100 fps expected, for a `500^3` volume.

### 2.5 Assumptions
* CSV coordinates are voxel indices by default (`M[x][y][z]` ↔ point `(x, y, z)`); world
  coordinates are available as an option (`--coordinates world`).
* The image rows advance from the apex / top edge (`a`, or `a1 a2`) toward the base (`b c`),
  matching the pseudo-code (`y` from `0` "up" to `Dab`).

---

## 3. Functional requirements

### FR-1 Volume input
* FR-1.1 Read NRRD versions 1–5 with attached data (`.nrrd`) or a single detached data file (`.nhdr`).
* FR-1.2 Encodings: `raw`, `gzip`, `bzip2`, `ascii`/`text`, `hex`; byte order `little` / `big`.
* FR-1.3 Types: all NRRD scalar types (`int8 … uint64`, `float`, `double`); the required
  `uint8/16/32` are covered by tests.
* FR-1.4 Only 3D volumes; each size must be `< 1024`. Other dimensions → error.
* FR-1.5 Axis order: the first NRRD axis (fastest on disk) is `x`.
* FR-1.6 A malformed or unsupported file raises `NrrdError` with a descriptive message.

### FR-2 Plane definition and validation
A plane is created by `Plane.from_points(points, eps=0.001)`, which raises `InvalidPlaneError`
when the points are invalid:

* FR-2.1 **3 points `(a, b, c)`**: valid iff `|dist(a,b) − dist(a,c)| < EPS`, and the triangle is not
  degenerate (`dist(a,b) ≥ EPS`, `dist(b,c) ≥ EPS`, `a b c` not collinear).
* FR-2.2 **4 points `(a1, a2, b, c)`**: valid iff
  1. `|dist(a1,b) − dist(a2,c)| < EPS`,
  2. `dist(a1,a2) ≤ dist(b,c)` (with EPS tolerance),
  3. the four points are coplanar (distance of the 4th point from the plane of the other three `< EPS`),
  4. not degenerate (non-zero legs, not all collinear).
  If `dist(a1,a2) < EPS` the shape is treated as the triangle `((a1+a2)/2, b, c)`.
* FR-2.3 Points must be finite numbers (NaN / inf rejected); exactly 3 or 4 points.
* FR-2.4 EPS is configurable (`--eps`).

### FR-3 Projection
* FR-3.1 Output `E` has `floor(Dab)+1` rows and `floor(Dbc)+1` columns, with the volume's dtype.
* FR-3.2 Row `y` samples the segment `a1(y) → a2(y)` where `a1(y) = a1 + y·v(a1,b)` and
  `a2(y) = a2 + y·v(a2,c)`; the segment is centred: column `x` corresponds to
  `x1 = x − (Dbc − d12)/2` along it.
* FR-3.3 Each sample takes the value of the **closest voxel** (`floor(p + 0.5)` per axis).
* FR-3.4 Samples outside `M` and pixels outside the triangle / trapezoid have value `0`.
* FR-3.5 Optional world coordinates: the plane is defined and stepped in world units (e.g. mm),
  each sample is mapped to index space with the inverse NRRD affine.

### FR-4 CSV input
* FR-4.1 One plane per line: 9 numbers (`ax,ay,az,bx,by,bz,cx,cy,cz`) or 12 numbers
  (`a1x,…,a2x,…,bx,…,cx,…`).
* FR-4.2 Blank lines and lines starting with `#` are ignored; a first line containing no numbers
  is a header. Parentheses / brackets around coordinates are tolerated.
* FR-4.3 A malformed line (wrong count, non-numeric, invalid plane) does **not** stop processing;
  it becomes an error record with the line number and reason.

### FR-5 Demo / video output
* FR-5.1 `python Ex0.py file.nrrd file.csv file.mp4` writes an mp4 with exactly one frame per CSV
  plane, in CSV order.
* FR-5.2 Invalid planes produce an **error frame**: black, red border and X, the CSV line number
  and the reason. All valid planes produce correct frames.
* FR-5.3 All frames share one size (the largest projected image, centred; small volumes are
  up-scaled by an integer factor, `--scale` overrides).
* FR-5.4 Grey levels: one linear intensity window for the whole video (0.5–99.5 percentile of the
  volume by default, `--window LOW HIGH` overrides).
* FR-5.5 Options: `--fps`, `--eps`, `--coordinates {index,world}`, `--scale`, `--window`,
  `--no-captions`, `--frames-dir` (save PNG of every frame).
* FR-5.6 Fatal input problems (missing / unreadable NRRD, missing / empty CSV, missing output
  directory) print `error: …` to stderr and exit with code 1 (2 for bad options). Otherwise exit 0
  and print a summary (frames, errors with line numbers, projection fps).

---

## 4. Non-functional requirements

| ID | Requirement | Verification |
|---|---|---|
| NFR-1 Performance | ≥ 100 fps (≥ 30 mandatory) projection for a `500^3` volume, all voxel types | `benchmarks/benchmark.py`, `PerformanceTest` |
| NFR-2 Correctness | Agreement with a literal implementation of the pseudo-code; exact results on axis-aligned planes | `tests/test_projector.py` |
| NFR-3 Robustness | No crash on any CSV content; invalid planes isolated to their frame | `tests/test_planes_csv.py`, `tests/test_demo.py` |
| NFR-4 Portability | Python ≥ 3.9 on macOS / Linux / Windows; only NumPy + OpenCV | `requirements.txt` |
| NFR-5 Maintainability | Separate modules for I/O, geometry, projection, CSV, video, CLI; docstrings; unit tests per module | code review |
| NFR-6 Memory | One copy of the volume (+1 sentinel voxel) and `O(frame)` working memory | design |

---

## 5. Interfaces

### 5.1 Command line
```
python Ex0.py file.nrrd file.csv file.mp4 [--fps 10] [--eps 0.001] [--coordinates index|world]
              [--scale N] [--window LOW HIGH] [--no-captions] [--frames-dir DIR]
```

### 5.2 Library
```python
from volproj.volume import Volume
from volproj.plane import Plane, InvalidPlaneError
from volproj.projector import PlaneProjector

volume = Volume.from_nrrd("data/MRHead.nrrd")
plane = Plane.from_points([(127.5, 0, 65), (0, 255, 65), (255, 255, 65)])  # may raise InvalidPlaneError
image = PlaneProjector(volume).project(plane)                              # 2D numpy array E[y, x]
```
