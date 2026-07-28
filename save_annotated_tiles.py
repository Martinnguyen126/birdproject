"""Save all tiles with YOLO-style boxes + lat/lon labels (no yellow circles).

Uses the same box colors as model.predict(save=True) validation output.

Usage:
    python save_annotated_tiles.py
    python save_annotated_tiles.py --weights runs/detect/birds_v2/weights/best.pt --output annotated_tiles_with_coords
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs/detect/birds_v2/weights/best.pt"
DEFAULT_TILES = ROOT / "tiles_640"
DEFAULT_CSV = ROOT / "bird_detections.csv"
DEFAULT_OUT = ROOT / "annotated_tiles_with_coords"


def add_geo_labels(bgr_img, rows: pd.DataFrame) -> None:
    """Add lat/lon under each box in white text (black outline)."""
    for _, row in rows.iterrows():
        if pd.notna(row.get("tile_x1")):
            x = int(row["tile_x1"])
            y = int(row["tile_y2"]) + 4
        else:
            continue
        text = f"{row['latitude']:.5f}, {row['longitude']:.5f}"
        cv2.putText(
            bgr_img,
            text,
            (x, min(y + 14, bgr_img.shape[0] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr_img,
            text,
            (x, min(y + 14, bgr_img.shape[0] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--tiles", type=Path, default=DEFAULT_TILES)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}. Run export_bird_map.py first.")

    df = pd.read_csv(args.csv)
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(args.tiles),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=False,
        verbose=False,
    )

    saved = 0
    for result in results:
        tile_name = Path(result.path).name
        tile_rows = df[df["tile"] == tile_name]
        if tile_rows.empty:
            continue

        # YOLO validation-style boxes and class colors
        img = result.plot(line_width=2, font_size=12)
        add_geo_labels(img, tile_rows)
        cv2.imwrite(str(args.output / tile_name), img)
        saved += 1

    print(f"Saved {saved} images to {args.output.resolve()}")


if __name__ == "__main__":
    main()
