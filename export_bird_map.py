"""Export detected birds to a geographic CSV / GeoJSON map layer.

Each YOLO detection in a tile is converted to a full-image pixel position,
then to latitude/longitude using GPS EXIF from the source aerial image.

Prerequisites:
    - Source image with GPS EXIF (260524_191405_086.jpg)
    - tiles_640/ from your tiling step
    - Trained weights (e.g. runs/detect/birds_v2/weights/best.pt)

Usage:
    python export_bird_map.py
    python export_bird_map.py --weights runs/detect/birds_v2/weights/best.pt --source tiles_640
    python export_bird_map.py --conf 0.35 --output bird_map.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "260524_191405_086.jpg"
DEFAULT_TILES = ROOT / "tiles_640"
DEFAULT_MANIFEST = DEFAULT_TILES / "tile_manifest_640.csv"
DEFAULT_WEIGHTS = ROOT / "runs/detect/birds_v2/weights/best.pt"
CLASS_NAMES = {0: "brownbird", 1: "whitebird"}

# Sony ILX-LR1 full-frame sensor width (mm) — used for ground sample distance
SENSOR_WIDTH_MM = 35.9


def dms_to_decimal(dms: tuple, ref: str) -> float:
    degrees, minutes, seconds = (float(x) for x in dms)
    dec = degrees + minutes / 60 + seconds / 3600
    if ref in ("S", "W"):
        dec = -dec
    return dec


def read_image_gps(image_path: Path) -> dict:
    """Read camera GPS + altitude + focal length from source image EXIF."""
    with Image.open(image_path) as im:
        width, height = im.size
        exif = im.getexif()
        if 34853 not in exif:
            raise ValueError(f"No GPS EXIF in {image_path}")

        gps = exif.get_ifd(34853)
        lat_ref = gps[1].decode() if isinstance(gps[1], bytes) else gps[1]
        lon_ref = gps[3].decode() if isinstance(gps[3], bytes) else gps[3]
        lat = dms_to_decimal(gps[2], lat_ref)
        lon = dms_to_decimal(gps[4], lon_ref)
        alt = float(gps[6])

        focal_mm = None
        focal_tag = exif.get(37386)  # FocalLength
        if focal_tag is not None:
            focal_mm = float(Fraction(focal_tag)) if not isinstance(focal_tag, (int, float)) else float(focal_tag)

    if focal_mm is None:
        focal_mm = 24.0

    return {
        "latitude": lat,
        "longitude": lon,
        "altitude_m": alt,
        "focal_length_mm": focal_mm,
        "image_width": width,
        "image_height": height,
    }


def ground_sample_distance(altitude_m: float, focal_mm: float, image_width_px: int) -> float:
    """Meters per pixel (approximate nadir aerial photo)."""
    return (altitude_m * (SENSOR_WIDTH_MM / 1000.0)) / ((focal_mm / 1000.0) * image_width_px)


def pixel_to_latlon(
    pixel_x: float,
    pixel_y: float,
    meta: dict,
    gsd_m_per_px: float,
) -> tuple[float, float]:
    """Convert full-image pixel (x right, y down) to WGS84 lat/lon."""
    cx = meta["image_width"] / 2.0
    cy = meta["image_height"] / 2.0
    east_m = (pixel_x - cx) * gsd_m_per_px
    north_m = -(pixel_y - cy) * gsd_m_per_px  # y grows downward in image

    lat_rad = math.radians(meta["latitude"])
    lat = meta["latitude"] + north_m / 111_320.0
    lon = meta["longitude"] + east_m / (111_320.0 * math.cos(lat_rad))
    return lat, lon


def parse_tile_offsets(filename: str) -> tuple[int, int]:
    """Parse col_off, row_off from tile_0023_x1440_y480.jpg."""
    match = re.search(r"_x(\d+)_y(\d+)\.jpg$", filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse tile offsets from filename: {filename}")
    return int(match.group(1)), int(match.group(2))


def load_tile_offsets(manifest_path: Path) -> dict[str, tuple[int, int]]:
    if not manifest_path.exists():
        return {}
    df = pd.read_csv(manifest_path)
    return {
        str(row["filename"]): (int(row["col_off"]), int(row["row_off"]))
        for _, row in df.iterrows()
    }


def export_geojson(rows: list[dict], path: Path) -> None:
    features = []
    for row in rows:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["longitude"], row["latitude"]],
                },
                "properties": {k: v for k, v in row.items() if k not in ("latitude", "longitude")},
            }
        )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export bird detections with lat/lon")
    parser.add_argument("--source-image", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--tiles", type=Path, default=DEFAULT_TILES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--output", type=Path, default=ROOT / "bird_detections.csv")
    parser.add_argument("--geojson", type=Path, default=ROOT / "bird_detections.geojson")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.source_image.exists():
        raise FileNotFoundError(f"Source image not found: {args.source_image}")

    meta = read_image_gps(args.source_image)
    gsd = ground_sample_distance(
        meta["altitude_m"], meta["focal_length_mm"], meta["image_width"]
    )
    tile_offsets = load_tile_offsets(args.manifest)

    print(f"Camera GPS: {meta['latitude']:.6f}, {meta['longitude']:.6f}")
    print(f"Altitude: {meta['altitude_m']:.1f} m  |  GSD: {gsd*100:.2f} cm/pixel")

    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(args.tiles),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=False,
        verbose=False,
    )

    rows: list[dict] = []
    bird_id = 0
    for result in results:
        tile_name = Path(result.path).name
        if tile_name in tile_offsets:
            col_off, row_off = tile_offsets[tile_name]
        else:
            col_off, row_off = parse_tile_offsets(tile_name)

        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx_tile = (x1 + x2) / 2.0
            cy_tile = (y1 + y2) / 2.0
            pixel_x = col_off + cx_tile
            pixel_y = row_off + cy_tile
            lat, lon = pixel_to_latlon(pixel_x, pixel_y, meta, gsd)

            cls_id = int(box.cls[0])
            rows.append(
                {
                    "bird_id": bird_id,
                    "species": CLASS_NAMES.get(cls_id, str(cls_id)),
                    "class_id": cls_id,
                    "confidence": round(float(box.conf[0]), 4),
                    "tile": tile_name,
                    "tile_col_off": col_off,
                    "tile_row_off": row_off,
                    "tile_x1": round(x1, 1),
                    "tile_y1": round(y1, 1),
                    "tile_x2": round(x2, 1),
                    "tile_y2": round(y2, 1),
                    "pixel_x_full": round(pixel_x, 1),
                    "pixel_y_full": round(pixel_y, 1),
                    "latitude": round(lat, 8),
                    "longitude": round(lon, 8),
                    "camera_altitude_m": meta["altitude_m"],
                }
            )
            bird_id += 1

    if not rows:
        print("No birds detected. Try lowering --conf (e.g. 0.25).")
        return

    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    export_geojson(rows, args.geojson)

    print(f"\nDetected {len(rows)} birds")
    print(f"CSV:     {args.output.resolve()}")
    print(f"GeoJSON: {args.geojson.resolve()}")
    print("\nOpen the CSV in Excel or import the GeoJSON into QGIS / Google Earth.")


if __name__ == "__main__":
    main()
