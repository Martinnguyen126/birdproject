"""Merge LabelImg class IDs into a single 'whitebird' class (id 0).

LabelImg created two labels due to a typo in classes.txt:
  0 = whtiebird
  1 = whitebird

Fixes labels in:
  - tiles_640/          (LabelImg source folder)
  - dataset/labels/     (train + val, used for YOLO training)

Run before training or after editing labels in LabelImg:
    python prepare_labels.py
"""

from pathlib import Path

LABEL_DIRS = [
    Path("tiles_640"),
    Path("dataset/labels/train"),
    Path("dataset/labels/val"),
]

CLASSES_FILE = Path("tiles_640/classes.txt")


def normalize_label_file(path: Path) -> int:
    """Rewrite every box to class 0. Returns number of boxes updated."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return 0

    fixed = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        parts[0] = "0"
        fixed.append(" ".join(parts))

    path.write_text("\n".join(fixed) + ("\n" if fixed else ""), encoding="utf-8")
    return len(fixed)


def main() -> None:
    total_files = 0
    total_boxes = 0
    class1_before = 0

    for label_dir in LABEL_DIRS:
        if not label_dir.exists():
            print(f"SKIP (missing): {label_dir}")
            continue
        for label_file in sorted(label_dir.glob("*.txt")):
            if label_file.name == "classes.txt":
                continue
            text = label_file.read_text(encoding="utf-8")
            class1_before += sum(1 for ln in text.splitlines() if ln.strip().startswith("1 "))
            total_boxes += normalize_label_file(label_file)
            total_files += 1

    CLASSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLASSES_FILE.write_text("whitebird\n", encoding="utf-8")

    print(f"Normalized {total_boxes} boxes in {total_files} label files.")
    print(f"Rewrote {class1_before} boxes that were class 1 -> class 0.")
    print(f"Updated {CLASSES_FILE} to a single class: whitebird")


if __name__ == "__main__":
    main()
