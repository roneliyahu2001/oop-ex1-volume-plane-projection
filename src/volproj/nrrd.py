"""Minimal, dependency-free NRRD reader / writer.

Only the subset of the NRRD0001-NRRD0005 specification that is relevant for 3D
scalar volumes is implemented:

* attached data (header followed by a blank line) and detached data (``data file:``)
* encodings: raw, gzip, bzip2, ascii (text) and hex
* all scalar types (int8 ... uint64, float, double), little / big endian
* ``space``, ``space directions`` and ``space origin`` (used for world coordinates)

Specification: http://teem.sourceforge.net/nrrd/format.html
"""

from __future__ import annotations

import bz2
import gzip
import os
import re
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, List, Optional, Tuple

import numpy as np


class NrrdError(ValueError):
    """Raised when an NRRD file is malformed or not supported."""


_TYPE_ALIASES: Dict[str, str] = {}


def _register(numpy_type: str, *names: str) -> None:
    for name in names:
        _TYPE_ALIASES[name] = numpy_type


_register("i1", "signed char", "int8", "int8_t")
_register("u1", "uchar", "unsigned char", "uint8", "uint8_t")
_register("i2", "short", "short int", "signed short", "signed short int", "int16", "int16_t")
_register("u2", "ushort", "unsigned short", "unsigned short int", "uint16", "uint16_t")
_register("i4", "int", "signed int", "int32", "int32_t")
_register("u4", "uint", "unsigned int", "uint32", "uint32_t")
_register("i8", "longlong", "long long", "long long int", "signed long long",
          "signed long long int", "int64", "int64_t")
_register("u8", "ulonglong", "unsigned long long", "unsigned long long int", "uint64", "uint64_t")
_register("f4", "float")
_register("f8", "double")

_WRITE_TYPE_NAMES = {"i1": "int8", "u1": "uint8", "i2": "int16", "u2": "uint16", "i4": "int32",
                     "u4": "uint32", "i8": "int64", "u8": "uint64", "f4": "float", "f8": "double"}

_VECTOR_RE = re.compile(r"\(([^)]*)\)|none", re.IGNORECASE)


@dataclass
class NrrdHeader:
    """Parsed NRRD header (only the fields this project needs are typed)."""

    version: int
    fields: Dict[str, str] = field(default_factory=dict)
    key_values: Dict[str, str] = field(default_factory=dict)

    # --- typed accessors -------------------------------------------------
    def _required(self, name: str) -> str:
        try:
            return self.fields[name]
        except KeyError:
            raise NrrdError(f"missing required NRRD field '{name}'") from None

    @property
    def dtype(self) -> np.dtype:
        type_name = " ".join(self._required("type").lower().split())
        try:
            code = _TYPE_ALIASES[type_name]
        except KeyError:
            raise NrrdError(f"unsupported NRRD type '{type_name}'") from None
        dtype = np.dtype(code)
        if dtype.itemsize > 1:
            endian = self.fields.get("endian", "little").strip().lower()
            if endian not in ("little", "big"):
                raise NrrdError(f"invalid endian '{endian}'")
            dtype = dtype.newbyteorder("<" if endian == "little" else ">")
        return dtype

    @property
    def dimension(self) -> int:
        try:
            return int(self._required("dimension"))
        except ValueError:
            raise NrrdError("'dimension' must be an integer") from None

    @property
    def sizes(self) -> Tuple[int, ...]:
        try:
            sizes = tuple(int(s) for s in self._required("sizes").split())
        except ValueError:
            raise NrrdError("'sizes' must be integers") from None
        if len(sizes) != self.dimension:
            raise NrrdError(f"'sizes' has {len(sizes)} entries but dimension is {self.dimension}")
        if any(s < 1 for s in sizes):
            raise NrrdError(f"invalid sizes {sizes}")
        return sizes

    @property
    def encoding(self) -> str:
        enc = self._required("encoding").strip().lower()
        aliases = {"gz": "gzip", "bz2": "bzip2", "txt": "ascii", "text": "ascii"}
        enc = aliases.get(enc, enc)
        if enc not in ("raw", "gzip", "bzip2", "ascii", "hex"):
            raise NrrdError(f"unsupported encoding '{enc}'")
        return enc

    @property
    def space_directions(self) -> Optional[np.ndarray]:
        """3x3 matrix whose *columns* are the index-axis direction vectors (or None)."""
        raw = self.fields.get("space directions")
        if raw is None:
            return None
        vectors = []
        for match in _VECTOR_RE.finditer(raw):
            if match.group(1) is None:  # "none" (non-spatial axis)
                continue
            vectors.append([float(v) for v in match.group(1).split(",")])
        if len(vectors) != 3 or any(len(v) != 3 for v in vectors):
            return None
        return np.array(vectors, dtype=np.float64).T

    @property
    def space_origin(self) -> Optional[np.ndarray]:
        raw = self.fields.get("space origin")
        if raw is None:
            return None
        match = _VECTOR_RE.search(raw)
        if not match or match.group(1) is None:
            return None
        values = [float(v) for v in match.group(1).split(",")]
        return np.array(values, dtype=np.float64) if len(values) == 3 else None


@dataclass
class NrrdImage:
    header: NrrdHeader
    data: np.ndarray  # shape == sizes, indexed data[x, y, z] (first axis = fastest in file)


def _parse_header(stream: BinaryIO) -> NrrdHeader:
    magic = stream.readline().decode("ascii", errors="replace").strip()
    match = re.fullmatch(r"NRRD000([1-5])", magic)
    if not match:
        raise NrrdError(f"not an NRRD file (magic line was {magic[:20]!r})")
    header = NrrdHeader(version=int(match.group(1)))
    while True:
        raw = stream.readline()
        if not raw:  # EOF: header-only (detached) file without trailing blank line
            break
        line = raw.decode("latin-1").rstrip("\r\n")
        if line == "":
            break
        if line.startswith("#"):
            continue
        if ":=" in line:
            key, value = line.split(":=", 1)
            header.key_values[key] = value
        elif ": " in line:
            key, value = line.split(": ", 1)
            header.fields[key.strip().lower()] = value.strip()
        else:
            raise NrrdError(f"malformed header line: {line!r}")
    return header


def _decode(payload: bytes, header: NrrdHeader, count: int) -> np.ndarray:
    dtype = header.dtype
    encoding = header.encoding
    if encoding == "ascii":
        values = np.array(payload.split(), dtype=np.float64)
        if values.size < count:
            raise NrrdError(f"expected {count} ascii values, found {values.size}")
        return values[:count].astype(dtype.newbyteorder("="))
    if encoding == "gzip":
        try:
            payload = gzip.decompress(payload)
        except (OSError, EOFError) as exc:
            raise NrrdError(f"corrupt gzip data: {exc}") from None
    elif encoding == "bzip2":
        try:
            payload = bz2.decompress(payload)
        except (OSError, ValueError) as exc:
            raise NrrdError(f"corrupt bzip2 data: {exc}") from None
    elif encoding == "hex":
        payload = bytes.fromhex(payload.decode("ascii"))
    expected = count * dtype.itemsize
    if len(payload) < expected:
        raise NrrdError(f"data too short: expected {expected} bytes, found {len(payload)}")
    if encoding == "raw" and int(header.fields.get("byte skip", "0")) == -1:
        payload = payload[len(payload) - expected:]
    return np.frombuffer(payload, dtype=dtype, count=count)


def read_nrrd(path: str) -> NrrdImage:
    """Read an NRRD file. ``data`` is returned in native byte order, indexed [x, y, z]."""
    with open(path, "rb") as stream:
        header = _parse_header(stream)
        sizes = header.sizes
        count = int(np.prod(sizes, dtype=np.int64))
        data_file = header.fields.get("data file") or header.fields.get("datafile")
        if data_file:
            if data_file.startswith("LIST") or "%" in data_file:
                raise NrrdError("multi-file detached data is not supported")
            detached = data_file if os.path.isabs(data_file) else os.path.join(
                os.path.dirname(os.path.abspath(path)), data_file)
            with open(detached, "rb") as data_stream:
                payload = data_stream.read()
        else:
            payload = stream.read()

    for _ in range(int(header.fields.get("line skip", "0"))):
        payload = payload.split(b"\n", 1)[1] if b"\n" in payload else b""
    byte_skip = int(header.fields.get("byte skip", "0"))
    if byte_skip > 0:
        payload = payload[byte_skip:]

    flat = _decode(payload, header, count)
    flat = flat.astype(flat.dtype.newbyteorder("="), copy=False)
    # NRRD stores the first axis fastest -> Fortran order.
    data = flat.reshape(sizes, order="F")
    return NrrdImage(header=header, data=data)


def write_nrrd(path: str, data: np.ndarray, encoding: str = "gzip", endian: str = "little",
               space_directions: Optional[np.ndarray] = None,
               space_origin: Optional[np.ndarray] = None,
               extra_fields: Optional[Dict[str, str]] = None) -> None:
    """Write ``data`` (indexed [x, y, z, ...]) as an attached NRRD file (used for tests/datasets)."""
    code = data.dtype.newbyteorder("=").str[1:]
    if code not in _WRITE_TYPE_NAMES:
        raise NrrdError(f"cannot write dtype {data.dtype}")
    lines: List[str] = ["NRRD0004", "# written by volproj",
                        f"type: {_WRITE_TYPE_NAMES[code]}",
                        f"dimension: {data.ndim}",
                        "sizes: " + " ".join(str(s) for s in data.shape),
                        f"endian: {endian}",
                        f"encoding: {encoding}"]
    if space_directions is not None:
        cols = np.asarray(space_directions, dtype=np.float64).T
        lines.append("space: left-posterior-superior")
        lines.append("space directions: " + " ".join(
            "(" + ",".join(repr(float(v)) for v in col) + ")" for col in cols))
    if space_origin is not None:
        lines.append("space origin: (" + ",".join(repr(float(v)) for v in space_origin) + ")")
    for key, value in (extra_fields or {}).items():
        lines.append(f"{key}: {value}")

    target = data.astype(data.dtype.newbyteorder("<" if endian == "little" else ">"), copy=False)
    flat = target.ravel(order="F")
    if encoding == "ascii":
        payload = (" ".join(str(v) for v in flat.tolist()) + "\n").encode("ascii")
    else:
        payload = flat.tobytes()
        if encoding == "gzip":
            payload = gzip.compress(payload, compresslevel=6)
        elif encoding == "bzip2":
            payload = bz2.compress(payload)
        elif encoding != "raw":
            raise NrrdError(f"cannot write encoding '{encoding}'")
    with open(path, "wb") as stream:
        stream.write(("\n".join(lines) + "\n\n").encode("ascii"))
        stream.write(payload)
