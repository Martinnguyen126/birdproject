"""Train a YOLOv8 detector for whitebirds.

Prerequisites:
    pip install -r requirements.txt
    python prepare_labels.py

Usage:
    python train_whitebird.py
    python train_whitebird.py --model yolov8s.pt --epochs 150 --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train whitebird YOLO model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("whitebird.yaml"),
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base checkpoint (yolov8n.pt = fast, yolov8s.pt = more accurate)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="cuda device id, cpu, or '' for auto")
    parser.add_argument("--project", default="runs/detect", help="Output parent folder")
    parser.add_argument("--name", default="whitebird", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or "cpu",
        workers=0,
        project=args.project,
        name=args.name,
        patience=20,
        save=True,
        plots=False,
        amp=False,
        mosaic=0.0,
        cache=False,
        exist_ok=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights: {best_weights}")
    print(f"Metrics folder: {results.save_dir}")


if __name__ == "__main__":
    main()
