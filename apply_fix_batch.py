"""Copy labels from fix_batch/ back to tiles_640/ after labeling.

Usage:
    python apply_fix_batch.py
"""

from pathlib import Path
import shutil

BATCH_DIR = Path("fix_batch")
TILES_DIR = Path("tiles_640")


def main() -> None:
    if not BATCH_DIR.exists():
        raise FileNotFoundError("No fix_batch/ folder found.")

    n_img = n_lbl = 0
    for img in BATCH_DIR.glob("*.jpg"):
        shutil.copy2(img, TILES_DIR / img.name)
        n_img += 1
        lbl = img.with_suffix(".txt")
        if lbl.exists():
            shutil.copy2(lbl, TILES_DIR / lbl.name)
            n_lbl += 1

    print(f"Applied {n_img} images and {n_lbl} labels to {TILES_DIR.resolve()}")
    print("Next: re-run your split script, then Task 3 steps 2-4.")


if __name__ == "__main__":
    main()
