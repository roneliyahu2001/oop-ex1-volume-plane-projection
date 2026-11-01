"""The demo pipeline: NRRD + CSV of planes -> mp4 with one frame per plane."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2

from .plane import DEFAULT_EPS
from .planes_csv import PlaneRecord, parse_planes_csv
from .projector import PlaneProjector
from .video import FrameRenderer, IntensityWindow, Mp4Writer
from .volume import Volume


class DemoError(Exception):
    """A fatal problem with the demo inputs (as opposed to a single invalid plane)."""


@dataclass
class DemoOptions:
    fps: float = 10.0
    eps: float = DEFAULT_EPS
    coordinates: str = "index"  # "index" (voxel indices) or "world" (NRRD space, e.g. mm)
    scale: Optional[int] = None  # None = automatic
    captions: bool = True
    window: Optional[Tuple[float, float]] = None  # None = robust percentile window
    frames_dir: Optional[str] = None  # also save every frame as PNG


@dataclass
class DemoReport:
    output_path: str
    frames: int = 0
    valid_frames: int = 0
    errors: List[Tuple[int, str]] = field(default_factory=list)  # (csv line, message)
    projection_seconds: float = 0.0
    total_seconds: float = 0.0
    frame_size: Tuple[int, int] = (0, 0)
    codec: str = ""

    @property
    def projection_fps(self) -> float:
        return self.valid_frames / self.projection_seconds if self.projection_seconds > 0 else 0.0


def run_demo(nrrd_path: str, csv_path: str, mp4_path: str,
             options: Optional[DemoOptions] = None) -> DemoReport:
    options = options or DemoOptions()
    started = time.perf_counter()
    for path, what in ((nrrd_path, "NRRD"), (csv_path, "CSV")):
        if not os.path.isfile(path):
            raise DemoError(f"{what} file not found: {path}")
    out_dir = os.path.dirname(os.path.abspath(mp4_path))
    if not os.path.isdir(out_dir):
        raise DemoError(f"output directory does not exist: {out_dir}")

    try:
        volume = Volume.from_nrrd(nrrd_path)
    except (ValueError, OSError) as exc:
        raise DemoError(f"cannot read volume '{nrrd_path}': {exc}") from exc

    to_index = None
    if options.coordinates == "world":
        if not volume.has_world_transform:
            raise DemoError("--coordinates world needs 'space directions' in the NRRD header")
        to_index = volume.world_to_index_affine()
    elif options.coordinates != "index":
        raise DemoError(f"unknown coordinate system '{options.coordinates}'")

    try:
        records = parse_planes_csv(csv_path, eps=options.eps)
    except (OSError, UnicodeDecodeError) as exc:
        raise DemoError(f"cannot read CSV '{csv_path}': {exc}") from exc
    if not records:
        raise DemoError(f"no planes found in '{csv_path}'")

    window = IntensityWindow(*(options.window or volume.intensity_range()))
    renderer = FrameRenderer([r.plane.output_shape for r in records if r.is_valid],
                             scale=options.scale, captions=options.captions)
    projector = PlaneProjector(volume)
    report = DemoReport(output_path=mp4_path, frame_size=renderer.frame_size)
    if options.frames_dir:
        os.makedirs(options.frames_dir, exist_ok=True)

    with Mp4Writer(mp4_path, renderer.frame_size, options.fps) as writer:
        report.codec = writer.codec
        for index, record in enumerate(records):
            frame = _render(record, index, len(records), projector, to_index, window, renderer, report)
            writer.write(frame)
            if options.frames_dir:
                cv2.imwrite(os.path.join(options.frames_dir, f"frame_{index:04d}.png"), frame)
        report.frames = writer.frames_written
    report.total_seconds = time.perf_counter() - started
    return report


def _render(record: PlaneRecord, index: int, count: int, projector: PlaneProjector, to_index,
            window: IntensityWindow, renderer: FrameRenderer, report: DemoReport):
    label = f"frame {index + 1}/{count} | csv line {record.line_number}"
    if not record.is_valid:
        report.errors.append((record.line_number, record.error))
        return renderer.render_error(f"INVALID PLANE - {label}", record.error)
    start = time.perf_counter()
    image = projector.project(record.plane, to_index)
    report.projection_seconds += time.perf_counter() - start
    report.valid_frames += 1
    shape = "triangle" if record.plane.is_triangle else "trapezoid"
    return renderer.render_image(window.to_uint8(image),
                                 f"{label} | {shape} {image.shape[1]}x{image.shape[0]}")
