"""Copy a small batch of tiles into fix_batch/ for stable labeling.

Usage (from birdproject folder):
    python make_fix_batch.py tile_0112_x5760_y2400.jpg tile_0108_x3840_y2400.jpg
    python make_fix_batch.py --from-list mistakes.txt
    python make_fix_batch.py --next 10

mistakes.txt = one tile filename per line (from error_hunt review).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TILES_DIR = Path("tiles_640")
BATCH_DIR = Path("fix_batch")


def copy_tiles(names: list[str]) -> None:
    BATCH_DIR.mkdir(exist_ok=True)
    copied = 0
    for name in names:
        name = name.strip()
        if not name:
            continue
        if not name.lower().endswith(".jpg"):
            name += ".jpg"
        src_img = TILES_DIR / name
        if not src_img.exists():
            print(f"SKIP (missing): {name}")
            continue
        shutil.copy2(src_img, BATCH_DIR / name)
        src_lbl = src_img.with_suffix(".txt")
        if src_lbl.exists():
            shutil.copy2(src_lbl, BATCH_DIR / src_lbl.name)
        else:
            (BATCH_DIR / src_lbl.name).touch()
        copied += 1
        print(f"OK: {name}")
    print(f"\nCopied {copied} tiles to {BATCH_DIR.resolve()}")
    print("Label in makesense.ai or LabelImg on fix_batch/ only.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tiles", nargs="*", help="tile filenames")
    parser.add_argument("--from-list", type=Path, help="text file, one filename per line")
    parser.add_argument("--next", type=int, help="copy first N tiles from list file mistakes.txt")
    args = parser.parse_args()

    names: list[str] = list(args.tiles)
    if args.from_list:
        names.extend(args.from_list.read_text(encoding="utf-8").splitlines())
    if args.next:
        list_path = Path("mistakes.txt")
        if not list_path.exists():
            raise FileNotFoundError("Create mistakes.txt with one filename per line.")
        lines = [ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        done = set()
        if BATCH_DIR.exists():
            done = {p.name for p in BATCH_DIR.glob("*.jpg")}
        remaining = [ln for ln in lines if (ln if ln.endswith(".jpg") else ln + ".jpg") not in done]
        names.extend(remaining[: args.next])

    if not names:
        parser.error("Provide tile names, --from-list, or --next N")
    copy_tiles(names)


if __name__ == "__main__":
    main()
