"""Fix inverted white/brown class IDs — run this, then retrain.

Does ALL of these in order:
  1. Swap class 0 <-> 1 in every label file in tiles_640/
  2. Fix tiles_640/classes.txt (whitebird first, brownbird second)
  3. Copy fresh labels into dataset/ via split_dataset
  4. Delete YOLO label cache files (stale cache = wrong classes even after swap)

Usage:
    python fix_bird_labels.py

Then retrain with a NEW run name (e.g. birds_v2) from yolov8n.pt — not birds_v1.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TILES_DIR = ROOT / "tiles_640"
CLASSES_FILE = TILES_DIR / "classes.txt"
CORRECT_CLASSES = ["whitebird", "brownbird"]


def swap_line(line: str) -> str:
    parts = line.split()
    if len(parts) < 5:
        return line
    if parts[0] == "0":
        parts[0] = "1"
    elif parts[0] == "1":
        parts[0] = "0"
    return " ".join(parts)


def swap_tiles_labels() -> tuple[int, int]:
    files = boxes = 0
    for txt in sorted(TILES_DIR.glob("*.txt")):
        if txt.name == "classes.txt":
            continue
        lines_out = []
        changed = 0
        for line in txt.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                lines_out.append("")
                continue
            new_line = swap_line(stripped)
            if new_line != stripped:
                changed += 1
            lines_out.append(new_line)
        if changed:
            txt.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
            files += 1
            boxes += changed
    return files, boxes


def fix_classes_txt() -> None:
    CLASSES_FILE.write_text("\n".join(CORRECT_CLASSES) + "\n", encoding="utf-8")


def delete_label_caches() -> None:
    removed = 0
    for cache in (ROOT / "dataset" / "labels").rglob("*.cache"):
        cache.unlink()
        print(f"Deleted cache: {cache.relative_to(ROOT)}")
        removed += 1
    if removed == 0:
        print("No .cache files found (OK — YOLO will rebuild from labels).")


def count_boxes(folder: Path) -> tuple[int, int]:
    c0 = c1 = 0
    for txt in folder.glob("*.txt"):
        if txt.name == "classes.txt":
            continue
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                if parts[0] == "0":
                    c0 += 1
                elif parts[0] == "1":
                    c1 += 1
    return c0, c1


def main() -> None:
    print("=" * 60)
    print("STEP 1: Swap class 0 <-> 1 in tiles_640/*.txt")
    print("=" * 60)
    n_files, n_boxes = swap_tiles_labels()
    fix_classes_txt()
    print(f"Swapped {n_boxes} boxes in {n_files} files.")
    print(f"classes.txt -> {CORRECT_CLASSES}")

    c0, c1 = count_boxes(TILES_DIR)
    print(f"tiles_640 now: class 0 (whitebird) = {c0}, class 1 (brownbird) = {c1}")

    print()
    print("=" * 60)
    print("STEP 2: Refresh dataset/ from tiles_640 (split_dataset.py)")
    print("=" * 60)
    subprocess.run(
        [sys.executable, str(ROOT / "split_dataset.py")],
        cwd=ROOT,
        check=True,
    )

    tr0, tr1 = count_boxes(ROOT / "dataset" / "labels" / "train")
    va0, va1 = count_boxes(ROOT / "dataset" / "labels" / "val")
    print(f"dataset train: class 0 = {tr0}, class 1 = {tr1}")
    print(f"dataset val:   class 0 = {va0}, class 1 = {va1}")

    print()
    print("=" * 60)
    print("STEP 3: Delete YOLO label caches")
    print("=" * 60)
    delete_label_caches()

    print()
    print("=" * 60)
    print("DONE. Next steps:")
    print("=" * 60)
    print("  1. python verify_labels.py")
    print("  2. Retrain — use name='birds_v2' (NOT birds_v1)")
    print("  3. Test — runs/detect/birds_v2/weights/best.pt")
    print()
    print("If white/brown are STILL backwards after retraining, run this script")
    print("ONE more time, then train as birds_v3.")


if __name__ == "__main__":
    main()
