"""Swap class 0 <-> class 1 in YOLO label files (fixes inverted white/brown IDs).

Your boxes stay the same — only the class number on each line changes.

Usage:
    python swap_class_ids.py              # tiles_640 only
    python swap_class_ids.py --dataset    # also dataset/labels/train and val
"""

from __future__ import annotations

import argparse
from pathlib import Path

TILES_DIR = Path("tiles_640")
DATASET_LABEL_DIRS = [Path("dataset/labels/train"), Path("dataset/labels/val")]


def swap_line(line: str) -> str:
    parts = line.split()
    if len(parts) < 5:
        return line
    if parts[0] == "0":
        parts[0] = "1"
    elif parts[0] == "1":
        parts[0] = "0"
    return " ".join(parts)


def swap_file(path: Path) -> tuple[int, int]:
    """Return (boxes_swapped, lines_kept)."""
    text = path.read_text(encoding="utf-8")
    out_lines = []
    swapped = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        new_line = swap_line(stripped)
        if new_line != stripped:
            swapped += 1
        out_lines.append(new_line)
    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return swapped, len(out_lines)


def process_dir(folder: Path) -> None:
    if not folder.exists():
        print(f"Skip (missing): {folder}")
        return
    files = 0
    boxes = 0
    for txt in sorted(folder.glob("*.txt")):
        if txt.name == "classes.txt":
            continue
        n_boxes, _ = swap_file(txt)
        if n_boxes:
            files += 1
            boxes += n_boxes
    print(f"{folder}: swapped class id on {boxes} boxes in {files} files")


def fix_classes_txt() -> None:
    correct = ["whitebird", "brownbird"]
    path = TILES_DIR / "classes.txt"
    path.write_text("\n".join(correct) + "\n", encoding="utf-8")
    print(f"Fixed {path} -> {correct}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Swap YOLO class 0 and 1 in label files")
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Also swap files under dataset/labels/ (or re-run split_dataset.py instead)",
    )
    args = parser.parse_args()

    process_dir(TILES_DIR)
    fix_classes_txt()
    if args.dataset:
        for d in DATASET_LABEL_DIRS:
            process_dir(d)
    else:
        print("\nNext: python split_dataset.py   (refreshes dataset/ from tiles_640)")
    print("Then: python verify_labels.py")
    print("Then: retrain with birds.yaml")


if __name__ == "__main__":
    main()
