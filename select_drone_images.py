"""Select non-overlapping drone images using GPS + geodesic distance.

Hardcoded min distance (skeleton value): 15.9 meters.

Install (bird_env):
    pip install exif geopy

Usage:
    python select_drone_images.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
from exif import Image
from geopy.distance import geodesic

# --- CONFIGURATION ---
SOURCE_DIRS = [
    Path("HI_052426/Astor_Bird_30_Optical_A"),
    Path("HI_052426/Astor_Bird_30_Optical_B"),
]
OUTPUT_DIR = Path("selected_images_30m")

# Minimum distance (meters) between kept photos
MIN_DISTANCE_METERS = 15.9


def dms_to_dd(coords, ref: str) -> float:
    """Convert Degrees/Minutes/Seconds to Decimal Degrees."""
    degrees, minutes, seconds = coords
    dd = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ["S", "W"]:
        dd = -dd
    return dd


def get_image_location(image_path: Path) -> tuple[float, float] | None:
    """Extract latitude and longitude from image EXIF data."""
    with open(image_path, "rb") as img_file:
        img = Image(img_file)
        if not img.has_exif:
            return None
        try:
            lat = dms_to_dd(img.gps_latitude, img.gps_latitude_ref)
            lon = dms_to_dd(img.gps_longitude, img.gps_longitude_ref)
            return (lat, lon)
        except AttributeError:
            return None


def main() -> None:
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    images: list[Path] = []
    for folder in SOURCE_DIRS:
        if not folder.exists():
            print(f"WARNING: missing folder {folder.resolve()}")
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(folder / name)

    if not images:
        raise FileNotFoundError(
            "No images found. Put Optical_A / Optical_B under birdproject "
            "or edit SOURCE_DIRS at the top of this script."
        )

    print(f"Min distance between kept photos: {MIN_DISTANCE_METERS} m")
    print(f"Scanning {len(images)} images...\n")

    last_coords = None
    copied_count = 0
    all_rows = []
    selected_rows = []

    for img_path in images:
        current_coords = get_image_location(img_path)

        if current_coords is None:
            print(f"Skipping {img_path.name}: No GPS data found.")
            continue

        lat, lon = current_coords
        row = {
            "path": str(img_path),
            "filename": img_path.name,
            "folder": img_path.parent.name,
            "latitude": lat,
            "longitude": lon,
        }
        all_rows.append(row)

        # Keep first image, or any image far enough from the last kept one
        if last_coords is None or geodesic(last_coords, current_coords).meters >= MIN_DISTANCE_METERS:
            dst_name = f"{img_path.parent.name}__{img_path.name}"
            shutil.copy2(img_path, OUTPUT_DIR / dst_name)
            last_coords = current_coords
            copied_count += 1
            selected_rows.append(row)
            print(f"Selected: {img_path.parent.name}/{img_path.name}")

    pd.DataFrame(all_rows).to_csv("all_drone_gps.csv", index=False)
    pd.DataFrame(selected_rows).to_csv("selected_drone_images.csv", index=False)

    print(f"\nDone! Selected {copied_count} out of {len(images)} images.")
    print(f"Copied to: {OUTPUT_DIR.resolve()}")
    print("CSVs: all_drone_gps.csv, selected_drone_images.csv")


if __name__ == "__main__":
    main()
