#!/usr/bin/env python3
"""Download extra NRRD volumes from the 3D Slicer sample-data store (SHA-256 verified).

    python tools/download_slicer_data.py [NAME ...]      # default: all

Files are saved in data/slicer/ (git-ignored, they are large). The URLs and hashes are
taken from Slicer's SampleData module (Modules/Scripted/SampleData/SampleData.py).
"""

import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "data", "slicer")
BASE_URL = "https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/"

DATASETS = {  # name: (file name, sha256)
    "MRBrainTumor1": ("RegLib_C01_1.nrrd", "998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95"),
    "MRBrainTumor2": ("RegLib_C01_2.nrrd", "1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97"),
    "CTChest": ("CT-chest.nrrd", "4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e"),
    "CTACardio": ("CTA-cardio.nrrd", "3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2"),
    "CTMRBrain-CT": ("CT-brain.nrrd", "6a5b6caccb76576a863beb095e3bfb910c50ca78f4c9bf043aa42f976cfa53d1"),
    "MRHead-Slicer": ("MR-head.nrrd", "cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name):
    file_name, expected = DATASETS[name]
    path = os.path.join(TARGET, file_name)
    if os.path.exists(path) and sha256(path) == expected:
        print(f"{name}: already present ({path})")
        return
    print(f"{name}: downloading {file_name} ...", flush=True)
    urllib.request.urlretrieve(BASE_URL + expected, path + ".part")
    if sha256(path + ".part") != expected:
        os.remove(path + ".part")
        raise SystemExit(f"{name}: checksum mismatch, file discarded")
    os.replace(path + ".part", path)
    print(f"{name}: ok -> {os.path.relpath(path, ROOT)}")


def main(names):
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)}")
    os.makedirs(TARGET, exist_ok=True)
    for name in names or DATASETS:
        download(name)


if __name__ == "__main__":
    main(sys.argv[1:])
