"""Reading the list of planes from a CSV file.

Format (one plane per line, see docs/SRS.md, "Input formats")::

    # comment lines and blank lines are ignored; an all-text first line is a header
    ax,ay,az,bx,by,bz,cx,cy,cz                 <- 3 points: isosceles triangle (a, b, c)
    a1x,a1y,a1z,a2x,a2y,a2z,bx,by,bz,cx,cy,cz  <- 4 points: isosceles trapezoid (a1, a2, b, c)

Brackets / parentheses around coordinates, e.g. ``(1,2,3),(4,5,6),(7,8,9)``, are tolerated.
Invalid lines do not abort reading: each yields a record carrying the error message so
the caller can decide what to do (the demo renders an "error frame").
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Optional

from .plane import DEFAULT_EPS, InvalidPlaneError, Plane


@dataclass(frozen=True)
class PlaneRecord:
    line_number: int
    text: str
    plane: Optional[Plane] = None
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.plane is not None


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _tokens(row: List[str]) -> List[str]:
    tokens = []
    for cell in row:
        cell = cell.translate(str.maketrans("", "", "()[]{}")).strip()
        tokens.extend(part for part in cell.split() if part)
    return tokens


def parse_planes_csv(path: str, eps: float = DEFAULT_EPS) -> List[PlaneRecord]:
    """Parse ``path`` into one :class:`PlaneRecord` per non-empty, non-comment line."""
    records: List[PlaneRecord] = []
    seen_data = False
    with open(path, newline="", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            tokens = _tokens(next(csv.reader([text])))
            if not seen_data and tokens and not any(_is_number(t) for t in tokens):
                seen_data = True  # header line
                continue
            seen_data = True
            records.append(_parse_line(line_number, text, tokens, eps))
    return records


def _parse_line(line_number: int, text: str, tokens: List[str], eps: float) -> PlaneRecord:
    bad = [t for t in tokens if not _is_number(t)]
    if bad:
        return PlaneRecord(line_number, text, error=f"non-numeric value(s): {', '.join(bad[:3])}")
    if len(tokens) not in (9, 12):
        return PlaneRecord(line_number, text,
                           error=f"expected 9 (3 points) or 12 (4 points) numbers, got {len(tokens)}")
    values = [float(t) for t in tokens]
    points = [values[i:i + 3] for i in range(0, len(values), 3)]
    try:
        return PlaneRecord(line_number, text, plane=Plane.from_points(points, eps=eps))
    except InvalidPlaneError as exc:
        return PlaneRecord(line_number, text, error=str(exc))
