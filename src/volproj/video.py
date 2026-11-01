"""Turning projected images into video frames and writing an mp4 file."""

from __future__ import annotations

import math
import textwrap
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

MAX_CANVAS_SIDE = 2048
CAPTION_HEIGHT = 28
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_RED = (40, 40, 230)  # BGR


class IntensityWindow:
    """Linear mapping of raw voxel values ``[low, high]`` to 8-bit grey levels.

    One window is used for the whole video so that brightness is comparable between frames.
    """

    def __init__(self, low: float, high: float):
        if not high > low:
            raise ValueError(f"invalid intensity window [{low}, {high}]")
        self.low = float(low)
        self.high = float(high)

    def to_uint8(self, image: np.ndarray) -> np.ndarray:
        scaled = (image.astype(np.float32) - self.low) * (255.0 / (self.high - self.low))
        return np.clip(scaled, 0, 255, out=scaled).astype(np.uint8)


class FrameRenderer:
    """Places images on a fixed-size BGR canvas (videos need a constant frame size)."""

    def __init__(self, image_shapes: Sequence[Tuple[int, int]], scale: Optional[int] = None,
                 captions: bool = True, min_size: Tuple[int, int] = (320, 240)):
        max_rows = max((s[0] for s in image_shapes), default=min_size[1])
        max_cols = max((s[1] for s in image_shapes), default=min_size[0])
        if scale is None:  # auto: enlarge small volumes so the video is watchable
            scale = max(1, min(4, math.ceil(480 / max(max_rows, max_cols, 1))))
        self.scale = scale
        self.captions = captions
        width = min(MAX_CANVAS_SIDE, max(min_size[0], max_cols * scale))
        height = min(MAX_CANVAS_SIDE, max(min_size[1], max_rows * scale))
        self.image_area = (height + height % 2, width + width % 2)  # H.264 needs even sizes
        extra = CAPTION_HEIGHT if captions else 0
        self.frame_size = (self.image_area[1], self.image_area[0] + extra)  # (width, height)

    def _blank(self) -> np.ndarray:
        return np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)

    def render_image(self, grey: np.ndarray, caption: str = "") -> np.ndarray:
        frame = self._blank()
        area_h, area_w = self.image_area
        rows, cols = grey.shape
        factor = min(self.scale, area_h / rows, area_w / cols)
        new_w, new_h = max(1, int(round(cols * factor))), max(1, int(round(rows * factor)))
        if (new_w, new_h) != (cols, rows):
            interpolation = cv2.INTER_NEAREST if factor >= 1 else cv2.INTER_AREA
            grey = cv2.resize(grey, (new_w, new_h), interpolation=interpolation)
        top, left = (area_h - new_h) // 2, (area_w - new_w) // 2
        frame[top:top + new_h, left:left + new_w] = grey[:, :, None]
        self._caption(frame, caption, (230, 230, 230))
        return frame

    def render_error(self, title: str, message: str) -> np.ndarray:
        """A black frame with a red X, a red border and the error text."""
        frame = self._blank()
        area_h, area_w = self.image_area
        cv2.rectangle(frame, (2, 2), (area_w - 3, area_h - 3), _RED, 3)
        cv2.line(frame, (0, 0), (area_w - 1, area_h - 1), (20, 20, 110), 2)
        cv2.line(frame, (area_w - 1, 0), (0, area_h - 1), (20, 20, 110), 2)
        font_scale = max(0.4, min(1.0, area_w / 700))
        line_height = int(30 * font_scale) + 6
        chars_per_line = max(16, int(area_w / (19 * font_scale)))
        lines = [title] + textwrap.wrap(message, chars_per_line)
        y = max(line_height, (area_h - line_height * len(lines)) // 2)
        for index, text in enumerate(lines):
            thickness = 2 if index == 0 else 1
            (text_w, _), _ = cv2.getTextSize(text, _FONT, font_scale, thickness)
            x = max(6, (area_w - text_w) // 2)
            cv2.putText(frame, text, (x, y), _FONT, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), _FONT, font_scale, _RED if index == 0 else (255, 255, 255),
                        thickness, cv2.LINE_AA)
            y += line_height
        self._caption(frame, title, _RED)
        return frame

    def _caption(self, frame: np.ndarray, caption: str, color) -> None:
        if not self.captions or not caption:
            return
        y = self.frame_size[1] - 9
        cv2.putText(frame, caption, (8, y), _FONT, 0.5, color, 1, cv2.LINE_AA)


class Mp4Writer:
    """Context manager around ``cv2.VideoWriter`` (H.264 when available, else MPEG-4 part 2)."""

    CODECS = ("avc1", "mp4v")

    def __init__(self, path: str, frame_size: Tuple[int, int], fps: float):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.path = path
        self.frame_size = frame_size
        self.frames_written = 0
        self.codec = None
        self._writer = None
        for codec in self.CODECS:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, frame_size)
            if writer.isOpened():
                self._writer, self.codec = writer, codec
                break
            writer.release()
        if self._writer is None:
            raise IOError(f"cannot open an mp4 video writer for '{path}'")

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            raise ValueError(f"frame size {frame.shape[1]}x{frame.shape[0]} != {self.frame_size}")
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
