"""Train a two-class YOLOv8 detector (whitebird + brownbird).

Prerequisites:
    - Label tiles in tiles_640/ with whitebird (0) and brownbird (1)
    - Do NOT run prepare_labels.py (that forces everything to class 0)
    - python verify_labels.py
    - python split_dataset.py

Usage:
    python train_birds.py
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import torch

torch.set_num_threads(1)

from ultralytics import YOLO, settings

settings.update({"sync": False})


def main() -> None:
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="birds.yaml",
        epochs=30,
        imgsz=416,
        batch=1,
        workers=0,
        device="cpu",
        amp=False,
        mosaic=0.0,
        plots=False,
        cache=False,
        name="birds_v1",
        exist_ok=True,
        patience=10,
        verbose=True,
    )

    print("Best weights:", Path(results.save_dir) / "weights" / "best.pt")


if __name__ == "__main__":
    main()
