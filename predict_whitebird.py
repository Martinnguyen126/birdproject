"""Run the trained whitebird detector on images or tiles.

Usage:
    python predict_whitebird.py --weights runs/detect/whitebird/weights/best.pt --source tiles_640
    python predict_whitebird.py --weights runs/detect/whitebird/weights/best.pt --source 260524_191405_086.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whitebirds with YOLO")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained weights, e.g. runs/detect/whitebird/weights/best.pt",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Image file, folder of images, or glob pattern",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="", help="cuda device id, cpu, or '' for auto")
    parser.add_argument("--project", default="runs/predict", help="Output parent folder")
    parser.add_argument("--name", default="whitebird", help="Run name")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO txt predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(args.source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device or None,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
    )

    save_dir = Path(results[0].save_dir) if results else args.project
    print(f"Predictions saved to: {save_dir}")


if __name__ == "__main__":
    main()
