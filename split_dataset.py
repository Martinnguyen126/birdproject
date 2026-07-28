"""Copy labeled tiles from tiles_640 into dataset/ train/val split.

Uses a fixed random seed so the same tiles stay in val each run
(as long as the set of labeled files is the same).

Usage:
    python split_dataset.py
    python split_dataset.py --val-ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

TILES_DIR = Path("tiles_640")
DATASET_DIR = Path("dataset")
VAL_RATIO = 0.2
SEED = 42


def labeled_tiles() -> list[Path]:
    """Include tiles that have a .txt label file.

    - Non-empty .txt = birds present
    - Empty .txt = background / hard negative (no birds) — still used in training
    """
    images = []
    for img in sorted(TILES_DIR.glob("*.jpg")):
        lbl = img.with_suffix(".txt")
        if lbl.exists():  # empty files are OK (concrete-only negatives)
            images.append(img)
    return images


def copy_pair(img: Path, split: str) -> None:
    lbl = img.with_suffix(".txt")
    img_out = DATASET_DIR / "images" / split / img.name
    lbl_out = DATASET_DIR / "labels" / split / lbl.name
    img_out.parent.mkdir(parents=True, exist_ok=True)
    lbl_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, img_out)
    if lbl.exists():
        shutil.copy2(lbl, lbl_out)
    else:
        lbl_out.write_text("", encoding="utf-8")


def clear_split_dirs() -> None:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            folder = DATASET_DIR / kind / split
            if folder.exists():
                for f in folder.iterdir():
                    if f.is_file():
                        f.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tiles = labeled_tiles()
    if not tiles:
        raise FileNotFoundError(f"No labeled tiles found in {TILES_DIR.resolve()}")

    random.seed(args.seed)
    shuffled = tiles.copy()
    random.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * args.val_ratio))
    val_set = set(shuffled[:n_val])

    clear_split_dirs()

    train_n = val_n = 0
    for img in tiles:
        split = "val" if img in val_set else "train"
        copy_pair(img, split)
        if split == "val":
            val_n += 1
        else:
            train_n += 1

    print(f"Labeled tiles: {len(tiles)}")
    print(f"Train: {train_n}  Val: {val_n}  (seed={args.seed})")
    print(f"Output: {DATASET_DIR.resolve()}")
    print("Next: re-train with birds.yaml (two-class) or whitebird.yaml (single-class).")
    print("Do NOT run prepare_labels.py for two-class training.")


if __name__ == "__main__":
    main()
