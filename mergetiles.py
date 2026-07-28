"""Merge labeled tiles from another folder into tiles_640 with a prefix."""
import shutil
from pathlib import Path

SRC = Path("tiles_640_155")      # folder with jpg + txt from other photo
DST = Path("tiles_640")
PREFIX = "p2_"                  # unique per source photo

DST.mkdir(exist_ok=True)
n = 0
for jpg in sorted(SRC.glob("*.jpg")):
    new_name = PREFIX + jpg.name
    shutil.copy2(jpg, DST / new_name)
    txt = jpg.with_suffix(".txt")
    if txt.exists():
        shutil.copy2(txt, DST / (PREFIX + txt.name))
    n += 1
print(f"Copied {n} tiles into {DST.resolve()}")