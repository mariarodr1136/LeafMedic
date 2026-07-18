#!/usr/bin/env python3

"""
PlantVillage Dataset Downloader
===============================
Fetches the PlantVillage colour images into the image-folder layout that
training/train.py expects:

    training/data/plantvillage/
        Apple___Apple_scab/
        Apple___Black_rot/
        ...

The dataset is CC0 (public domain) and hosted in the spMohanty/PlantVillage-Dataset
GitHub repository. The full colour set is ~54,000 images across 38 classes and
roughly 2 GB, so the default is a capped subset — enough to validate the whole
pipeline end to end before committing to a full run.

Usage:
    python3 training/download_dataset.py                   # 200 images/class
    python3 training/download_dataset.py --per-class 0     # everything
    python3 training/download_dataset.py --classes Tomato___healthy Corn_(maize)___healthy
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "training" / "data" / "plantvillage"

API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
RAW = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"
HEADERS = {"User-Agent": "leafmedic-training"}


def http_json(url: str):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def list_classes() -> list[str]:
    return sorted(entry["name"] for entry in http_json(API) if entry["type"] == "dir")


def list_images(class_name: str) -> list[str]:
    url = f"{API}/{urllib.parse.quote(class_name)}"
    return [
        entry["name"] for entry in http_json(url)
        if entry["name"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def download_one(class_name: str, filename: str, *, dest_dir: Path) -> bool:
    target = dest_dir / filename
    if target.exists() and target.stat().st_size > 0:
        return True
    url = f"{RAW}/{urllib.parse.quote(class_name)}/{urllib.parse.quote(filename)}"
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        return True
    except (urllib.error.URLError, TimeoutError) as err:
        print(f"  ! {class_name}/{filename}: {err}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PlantVillage into training/data")
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--per-class", type=int, default=200,
                        help="images per class; 0 downloads every image")
    parser.add_argument("--classes", nargs="*", help="only these classes (default: all 38)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--list", action="store_true", help="list class names and exit")
    args = parser.parse_args()

    if args.list:
        for name in list_classes():
            print(name)
        return 0

    classes = args.classes or list_classes()
    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(classes)} classes to {args.dest}")
    print(f"Per class: {'all' if args.per_class == 0 else args.per_class} images\n")

    total = 0
    for index, class_name in enumerate(classes, 1):
        try:
            filenames = list_images(class_name)
        except urllib.error.HTTPError as err:
            print(f"[{index}/{len(classes)}] {class_name}: listing failed ({err})", file=sys.stderr)
            continue

        if args.per_class:
            filenames = filenames[: args.per_class]
        class_dir = args.dest / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(
                functools.partial(download_one, class_name, dest_dir=class_dir), filenames
            ))
        got = sum(results)
        total += got
        print(f"[{index}/{len(classes)}] {class_name}: {got}/{len(filenames)}")

    print(f"\n✓ {total} images in {args.dest}")
    print("Next: python3 training/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
