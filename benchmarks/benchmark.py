#!/usr/bin/env python3
"""Projection throughput benchmark (frames per second, projection only - no video encoding).

    python benchmarks/benchmark.py [--size 500] [--frames 300] [--dtypes uint8 uint16 uint32]

For every voxel type a random ``size^3`` volume is created and three plane families are
projected; the median of 5 repetitions is reported as a Markdown table.
"""

import argparse
import os
import platform
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from volproj.plane import Plane  # noqa: E402
from volproj.projector import PlaneProjector  # noqa: E402
from volproj.reference import reference_project  # noqa: E402
from volproj.volume import Volume  # noqa: E402


def axial_rectangles(n, size):
    last = size - 1
    return [Plane.from_points([[0, 0, z], [last, 0, z], [0, last, z], [last, last, z]])
            for z in np.linspace(0, last, n)]


def oblique_triangles(n, size):
    half, last = (size - 1) / 2, size - 1
    return [Plane.from_points([[half, 0, half], [0, last, half + t], [last, last, half - t]])
            for t in np.linspace(-half, half, n)]


def random_trapezoids(n, size, seed=0):
    rng = np.random.default_rng(seed)
    planes = []
    while len(planes) < n:
        center = rng.uniform(0.3, 0.7, 3) * size
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        v = np.cross(u, rng.normal(size=3))
        v /= np.linalg.norm(v)
        top, base, height = rng.uniform(0.1, 0.3) * size, rng.uniform(0.3, 0.5) * size, 0.45 * size
        planes.append(Plane.from_points([center - top * u - height * v, center + top * u - height * v,
                                         center - base * u + height * v, center + base * u + height * v]))
    return planes


FAMILIES = {"axial rectangle": axial_rectangles, "oblique triangle": oblique_triangles,
            "random trapezoid": random_trapezoids}


def measure(projector, planes, repeats=5):
    projector.project(planes[0])  # warm-up
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        for plane in planes:
            projector.project(plane)
        runs.append(len(planes) / (time.perf_counter() - start))
    return statistics.median(runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--dtypes", nargs="+", default=["uint8", "uint16", "uint32"])
    parser.add_argument("--reference", action="store_true",
                        help="also time the literal pseudo-code (slow) on one frame")
    args = parser.parse_args()

    print(f"machine: {platform.machine()} / {platform.processor() or platform.system()}, "
          f"python {platform.python_version()}, numpy {np.__version__}")
    print(f"volume: {args.size}^3, {args.frames} planes per family, median of 5 runs\n")
    print("| voxel type | plane family | output size (typ.) | fps |")
    print("|---|---|---|---|")
    rng = np.random.default_rng(1)
    for dtype in args.dtypes:
        info = np.iinfo(dtype)
        data = rng.integers(0, min(info.max, 2 ** 31 - 1), size=(args.size,) * 3, dtype=dtype)
        volume = Volume(data)
        del data
        projector = PlaneProjector(volume)
        for name, family in FAMILIES.items():
            planes = family(args.frames, args.size)
            rows, cols = planes[len(planes) // 2].output_shape
            print(f"| {dtype} | {name} | {cols}x{rows} | {measure(projector, planes):.0f} |", flush=True)
        if args.reference:
            plane = oblique_triangles(3, args.size)[1]
            start = time.perf_counter()
            reference_project(volume, plane)
            print(f"| {dtype} | literal pseudo-code (pure Python), 1 frame | "
                  f"{plane.output_shape[1]}x{plane.output_shape[0]} | "
                  f"{1 / (time.perf_counter() - start):.2f} |", flush=True)
        del volume, projector


if __name__ == "__main__":
    main()
