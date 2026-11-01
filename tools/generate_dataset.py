#!/usr/bin/env python3
"""Generate the synthetic part of the dataset (phantom volumes + plane CSV files).

    python tools/generate_dataset.py

Writes into ``data/``:
  phantom_u8.nrrd, phantom_u16.nrrd, phantom_u32.nrrd  - 128^3 analytic phantoms (sphere, cube,
                                                         cylinder) in the three required voxel types
  planes/*.csv                                         - plane lists used by the demo and experiment
                                                         (mrhead_world.csv is derived from MRHead's header)
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from volproj.nrrd import write_nrrd  # noqa: E402
from volproj.volume import Volume  # noqa: E402

DATA = os.path.join(ROOT, "data")
PLANES = os.path.join(DATA, "planes")


def phantom(size=128) -> np.ndarray:
    """A sphere (bright), a cube (mid grey) and a thin cylinder (darker) on a faint gradient."""
    x, y, z = np.meshgrid(*(np.arange(size, dtype=np.float32),) * 3, indexing="ij")
    volume = 10 + 20 * z / size  # background gradient along z
    sphere = (x - 64) ** 2 + (y - 64) ** 2 + (z - 64) ** 2 <= 40 ** 2
    shell = sphere & ((x - 64) ** 2 + (y - 64) ** 2 + (z - 64) ** 2 >= 34 ** 2)
    cube = (np.abs(x - 64) <= 15) & (np.abs(y - 64) <= 15) & (np.abs(z - 64) <= 15)
    cylinder = ((x - 30) ** 2 + (z - 96) ** 2 <= 6 ** 2) & (y > 10) & (y < 118)
    volume[shell] = 255
    volume[cube] = 140
    volume[cylinder] = 80
    return volume


def fmt(points) -> str:
    return ",".join(f"{v:.4f}".rstrip("0").rstrip(".") for p in points for v in p)


def rotating_rectangles(center, half_width, half_height, steps, down=(0, 1, 0),
                        e1=(1, 0, 0), e2=(0, 0, 1)):
    """Rectangles (4-point planes) rotating about the ``down`` axis through ``center``.

    Image rows advance along ``down``; the horizontal direction turns from ``e1`` to ``e2``.
    """
    c = np.array(center, float)
    v = np.array(down, float)
    for angle in np.linspace(0, np.pi, steps, endpoint=False):
        u = np.cos(angle) * np.array(e1, float) + np.sin(angle) * np.array(e2, float)
        yield [c - half_width * u - half_height * v, c + half_width * u - half_height * v,
               c - half_width * u + half_height * v, c + half_width * u + half_height * v]


def sweeping_triangles(z_values, apex_x, apex_y, height, half_base):
    """Isosceles triangles (apex on top) in planes z = const."""
    for z in z_values:
        a = (apex_x, apex_y, z)
        yield [a, (apex_x - half_base, apex_y + height, z), (apex_x + half_base, apex_y + height, z)]


def tilting_trapezoids(center, steps, half_top, half_base, half_height, max_tilt_deg=60):
    """Isosceles trapezoids tilted forward/backward around the horizontal x axis."""
    c = np.array(center, float)
    u = np.array([1.0, 0.0, 0.0])
    for tilt in np.radians(np.linspace(-max_tilt_deg, max_tilt_deg, steps)):
        v = np.array([0.0, np.cos(tilt), np.sin(tilt)])
        yield [c - half_top * u - half_height * v, c + half_top * u - half_height * v,
               c - half_base * u + half_height * v, c + half_base * u + half_height * v]


INVALID_ROWS = [
    "# --- deliberately invalid planes (each must become an error frame) ---",
    "10,0,60, 0,100,60, 30,100,60            # not isosceles",
    "1,2,3, 4,5,6                            # only two points",
    "10,0,60, abc,100,60, 20,100,60          # non-numeric value",
    "40,0,60, 80,0,70, 0,100,60, 120,100,70  # 4 points, not coplanar",
]


def write_csv(name, header, rows):
    path = os.path.join(PLANES, name)
    with open(path, "w") as f:
        f.write(header.rstrip() + "\n")
        for row in rows:
            f.write((row if isinstance(row, str) else fmt(row)) + "\n")
    print("wrote", os.path.relpath(path, ROOT))


def strip_inline_comment(row):
    return row if row.startswith("#") else row.split("#")[0].strip()


def main():
    os.makedirs(PLANES, exist_ok=True)
    volume = phantom()
    for dtype, scale, name in ((np.uint8, 1, "u8"), (np.uint16, 200, "u16"), (np.uint32, 1_000_000, "u32")):
        path = os.path.join(DATA, f"phantom_{name}.nrrd")
        write_nrrd(path, (volume * scale).astype(dtype), encoding="gzip")
        print("wrote", os.path.relpath(path, ROOT))

    invalid = [strip_inline_comment(r) for r in INVALID_ROWS]

    # MRHead (256 x 256 x 130 voxels; index y runs superior -> inferior, so "up" is -y)
    center = (127.5, 127.5, 64.5)
    mr_rows = ["# 1) rectangle rotating about the vertical axis (sagittal -> coronal -> sagittal)"]
    mr_rows += list(rotating_rectangles(center, 125, 125, 36))
    mr_rows += [invalid[0], invalid[1]]
    mr_rows += ["# 2) triangles sweeping left -> right through the head (sagittal planes)"]
    mr_rows += list(sweeping_triangles(np.linspace(15, 115, 26), 127.5, 10, 236, 118))
    mr_rows += [invalid[2], invalid[3], invalid[4]]
    mr_rows += ["# 3) trapezoids tilting around the left-right axis"]
    mr_rows += list(tilting_trapezoids((127.5, 127.5, 64.5), 25, 70, 120, 110))
    write_csv("mrhead_experiment.csv",
              "# MRHead experiment - points are voxel indices (x, y, z); see docs/EXPERIMENT.md", mr_rows)

    write_csv("mrhead_basic.csv", "ax,ay,az,bx,by,bz,cx,cy,cz",
              [[(127.5, 0, 65), (0, 255, 65), (255, 255, 65)],
               [(0, 0, 65), (255, 0, 65), (0, 255, 65), (255, 255, 65)],
               [(127.5, 100, 0), (0, 100, 129), (255, 100, 129)]])

    # the same rotation in world (LPS, mm) coordinates: anatomically correct aspect ratio
    mr = Volume.from_nrrd(os.path.join(DATA, "MRHead.nrrd"))
    world_center = mr.index_to_world(center)
    world_rows = ["# rectangle rotating about the superior-inferior axis, LPS millimetres"]
    world_rows += list(rotating_rectangles(world_center, 110, 120, 36, down=(0, 0, -1),
                                           e1=(0, -1, 0), e2=(-1, 0, 0)))
    write_csv("mrhead_world.csv",
              "# MRHead in world coordinates - run with --coordinates world", world_rows)

    ph_rows = ["# rotating rectangles through the phantom centre"]
    ph_rows += list(rotating_rectangles((64, 64, 64), 62, 62, 24))
    ph_rows += ["# axial triangles"]
    ph_rows += list(sweeping_triangles(np.linspace(20, 108, 12), 64, 2, 124, 62))
    ph_rows += invalid[1:]
    write_csv("phantom.csv", "# phantom planes - voxel indices", ph_rows)

    write_csv("invalid_only.csv", "# every line is invalid", invalid[1:])


if __name__ == "__main__":
    main()
