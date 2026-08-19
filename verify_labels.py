"""Check two-class labels before training.

Usage:
    python verify_labels.py
"""

from pathlib import Path

TILES_DIR = Path("tiles_640")
CLASSES = ["brownbird", "whitebird"]  # line order = class id in LabelImg


def main() -> None:
    classes_file = TILES_DIR / "classes.txt"
    lines = [ln.strip() for ln in classes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if lines != CLASSES:
        print("WARNING: classes.txt should be exactly:")
        for c in CLASSES:
            print(f"  {c}")
        print(f"Found: {lines}")
    else:
        print("classes.txt OK:", lines)

    tiles = 0
    backgrounds = 0
    unlabeled = 0
    c0 = c1 = 0
    both = 0
    for img in sorted(TILES_DIR.glob("*.jpg")):
        lbl = img.with_suffix(".txt")
        if not lbl.exists():
            unlabeled += 1
            continue
        if lbl.stat().st_size == 0:
            backgrounds += 1
            continue
        tiles += 1
        ids = set()
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                ids.add(parts[0])
                if parts[0] == "0":
                    c0 += 1
                elif parts[0] == "1":
                    c1 += 1
        if "0" in ids and "1" in ids:
            both += 1

    print(f"Labeled tiles (birds): {tiles}")
    print(f"Background tiles (empty .txt): {backgrounds}")
    print(f"Unlabeled jpgs (no .txt): {unlabeled}")
    print(f"Class 0 (brownbird) boxes: {c0}")
    print(f"Class 1 (whitebird) boxes: {c1}")
    print(f"Tiles with BOTH classes: {both}")
    if c1 == 0:
        print("\nPROBLEM: No class 1 boxes. White birds must be class 1 in LabelImg.")
    elif c1 < 5:
        print("\nWARNING: Very few whitebird boxes. Confirm white birds use class 'whitebird' (id 1).")


if __name__ == "__main__":
    main()
