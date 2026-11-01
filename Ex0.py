#!/usr/bin/env python3
"""Ex1 demo: project a 3D NRRD volume onto a list of planes and write an mp4.

Usage:
    python Ex0.py file.nrrd file.csv file.mp4 [--fps 10] [--eps 0.001]
                  [--coordinates index|world] [--scale N] [--window LOW HIGH]
                  [--no-captions] [--frames-dir DIR]

Each CSV line is one plane: 9 numbers (triangle a,b,c) or 12 numbers (trapezoid a1,a2,b,c).
Invalid planes produce a red "error frame"; all other frames are rendered normally.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from volproj.demo import DemoError, DemoOptions, run_demo  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("nrrd", help="input 3D volume (.nrrd / .nhdr)")
    parser.add_argument("csv", help="planes, one per line: 9 or 12 coordinates")
    parser.add_argument("mp4", help="output video")
    parser.add_argument("--fps", type=float, default=10.0, help="video frame rate (default 10)")
    parser.add_argument("--eps", type=float, default=0.001, help="validation tolerance (default 0.001)")
    parser.add_argument("--coordinates", choices=("index", "world"), default="index",
                        help="CSV points are voxel indices (default) or NRRD world coordinates")
    parser.add_argument("--scale", type=int, default=None,
                        help="integer up-scaling of frames (default: automatic)")
    parser.add_argument("--window", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="intensity window mapped to black..white (default: 0.5-99.5 percentile)")
    parser.add_argument("--no-captions", action="store_true", help="do not print frame captions")
    parser.add_argument("--frames-dir", help="also save each frame as a PNG in this directory")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.fps <= 0 or args.eps <= 0 or (args.scale is not None and args.scale < 1):
        print("error: --fps, --eps and --scale must be positive", file=sys.stderr)
        return 2
    options = DemoOptions(fps=args.fps, eps=args.eps, coordinates=args.coordinates, scale=args.scale,
                          captions=not args.no_captions,
                          window=tuple(args.window) if args.window else None,
                          frames_dir=args.frames_dir)
    try:
        report = run_demo(args.nrrd, args.csv, args.mp4, options)
    except DemoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {report.output_path}: {report.frames} frames "
          f"({report.valid_frames} valid, {len(report.errors)} error), "
          f"{report.frame_size[0]}x{report.frame_size[1]}, codec {report.codec}")
    for line, message in report.errors:
        print(f"  invalid plane at csv line {line}: {message}")
    if report.valid_frames:
        print(f"projection: {report.projection_seconds * 1000:.1f} ms total, "
              f"{report.projection_fps:.0f} fps; total run time {report.total_seconds:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
