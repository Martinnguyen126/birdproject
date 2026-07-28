"""Fix class IDs on p2_ tiles only (photo 155 had whitebird first in classes.txt).

Your main tiles_640 labels are correct. Only the merged p2_* files need swapping.

Usage:
    python fix_p2_labels.py
    python split_dataset.py
    # then retrain as superbird_v2
"""

from pathlib import Path

TILES_DIR = Path("tiles_640")


def swap_line(line: str) -> str:
    parts = line.split()
    if len(parts) < 5:
        return line
    if parts[0] == "0":
        parts[0] = "1"
    elif parts[0] == "1":
        parts[0] = "0"
    return " ".join(parts)


def main() -> None:
    boxes = files = 0
    for txt in sorted(TILES_DIR.glob("p2_*.txt")):
        out = []
        changed = 0
        for line in txt.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                out.append("")
                continue
            new = swap_line(s)
            if new != s:
                changed += 1
            out.append(new)
        if changed:
            txt.write_text("\n".join(out) + "\n", encoding="utf-8")
            files += 1
            boxes += changed
    print(f"Swapped {boxes} boxes in {files} p2_* label files.")
    print("Original tiles (no p2_ prefix) were NOT changed.")
    print("\nNext:")
    print("  python verify_labels.py")
    print("  python split_dataset.py")
    print("  retrain with name='superbird_v2'")


if __name__ == "__main__":
    main()
