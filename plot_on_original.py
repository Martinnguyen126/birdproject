"""Draw all labeled birds onto the original source photo.

Labels in tiles_640/*.txt are in tile-normalized YOLO format (0-1).
Tile filenames (or tile_manifest_640.csv) give each tile's offset in the full image.

Usage:
    python plot_on_original.py                    # ground-truth labels
    python plot_on_original.py --from-csv         # model detections from CSV
    python plot_on_original.py --output my_map.jpg
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE_IMAGE = ROOT / "260524_191405_086.jpg"
TILES_DIR = ROOT / "tiles_640"
TILE_SIZE = 640
MANIFEST = TILES_DIR / "tile_manifest_640.csv"
DEFAULT_CSV = ROOT / "bird_detections.csv"

CLASS_COLORS = {
    0: (0, 128, 255),    # brownbird — orange (BGR)
    1: (255, 255, 0),    # whitebird — cyan (BGR)
}
CLASS_NAMES = {0: "brownbird", 1: "whitebird"}


def parse_tile_offset(filename: str) -> tuple[int, int]:
    match = re.search(r"_x(\d+)_y(\d+)\.jpg$", filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse tile offset: {filename}")
    return int(match.group(1)), int(match.group(2))


def load_manifest() -> dict[str, tuple[int, int]]:
    if not MANIFEST.exists():
        return {}
    df = pd.read_csv(MANIFEST)
    return {str(r["filename"]): (int(r["col_off"]), int(r["row_off"])) for _, r in df.iterrows()}


def yolo_line_to_full_box(
    line: str, col_off: int, row_off: int, tile_size: int = TILE_SIZE
) -> tuple[int, tuple[int, int, int, int]]:
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"Bad label line: {line}")
    cls_id = int(parts[0])
    cx, cy, w, h = (float(x) for x in parts[1:5])
    x1 = int(col_off + (cx - w / 2) * tile_size)
    y1 = int(row_off + (cy - h / 2) * tile_size)
    x2 = int(col_off + (cx + w / 2) * tile_size)
    y2 = int(row_off + (cy + h / 2) * tile_size)
    return cls_id, (x1, y1, x2, y2)


def draw_box(img, box: tuple[int, int, int, int], cls_id: int, label: str) -> None:
    color = CLASS_COLORS.get(cls_id, (0, 255, 0))
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def dedupe_boxes(
    items: list[tuple[int, tuple[int, int, int, int], str]],
    iou_thresh: float = 0.3,
) -> list[tuple[int, tuple[int, int, int, int], str]]:
    """Merge duplicate boxes from overlapping tiles (same bird in 2+ tiles)."""
    kept: list[tuple[int, tuple[int, int, int, int], str]] = []
    for cls_id, box, label in sorted(items, key=lambda x: x[0]):
        if any(cls_id == k[0] and box_iou(box, k[1]) > iou_thresh for k in kept):
            continue
        kept.append((cls_id, box, label))
    return kept


def collect_labels(offsets: dict[str, tuple[int, int]]) -> list[tuple[int, tuple[int, int, int, int], str]]:
    items: list[tuple[int, tuple[int, int, int, int], str]] = []
    for txt in sorted(TILES_DIR.glob("*.txt")):
        if txt.name == "classes.txt":
            continue
        lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue
        tile_name = txt.with_suffix(".jpg").name
        col_off, row_off = offsets.get(tile_name, parse_tile_offset(tile_name))
        for line in lines:
            cls_id, box = yolo_line_to_full_box(line, col_off, row_off)
            name = CLASS_NAMES.get(cls_id, str(cls_id))
            items.append((cls_id, box, name))
    return items


def collect_csv_boxes(csv_path: Path) -> list[tuple[int, tuple[int, int, int, int], str]]:
    df = pd.read_csv(csv_path)
    items: list[tuple[int, tuple[int, int, int, int], str]] = []
    for _, row in df.iterrows():
        if pd.notna(row.get("tile_x1")):
            box = (int(row["tile_x1"]), int(row["tile_y1"]), int(row["tile_x2"]), int(row["tile_y2"]))
            col_off, row_off = int(row["tile_col_off"]), int(row["tile_row_off"])
            box = (box[0] + col_off, box[1] + row_off, box[2] + col_off, box[3] + row_off)
        else:
            px, py = float(row["pixel_x_full"]), float(row["pixel_y_full"])
            box = (int(px - 15), int(py - 15), int(px + 15), int(py + 15))
        cls_id = int(row["class_id"])
        label = f"{row['species']} {row['confidence']:.2f}"
        items.append((cls_id, box, label))
    return items


def plot_items(img, items: list[tuple[int, tuple[int, int, int, int], str]], dedupe: bool) -> tuple[int, int]:
    raw_n = len(items)
    if dedupe:
        items = dedupe_boxes(items)
    for cls_id, box, label in items:
        draw_box(img, box, cls_id, label)
    return raw_n, len(items)


def plot_labels(img, offsets: dict[str, tuple[int, int]], dedupe: bool) -> tuple[int, int]:
    return plot_items(img, collect_labels(offsets), dedupe)


def plot_from_csv(img, csv_path: Path, dedupe: bool) -> tuple[int, int]:
    return plot_items(img, collect_csv_boxes(csv_path), dedupe)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot birds on original source photo")
    parser.add_argument("--source", type=Path, default=SOURCE_IMAGE)
    parser.add_argument("--from-csv", action="store_true", help="Use bird_detections.csv instead of label files")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=ROOT / "original_with_birds.jpg")
    parser.add_argument("--preview", type=Path, default=ROOT / "original_with_birds_preview.jpg")
    parser.add_argument("--preview-width", type=int, default=2400)
    parser.add_argument("--no-dedupe", action="store_true", help="Keep duplicate boxes from overlapping tiles")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source image not found: {args.source}")

    img = cv2.imread(str(args.source))
    if img is None:
        raise RuntimeError(f"Could not read image: {args.source}")

    dedupe = not args.no_dedupe
    if args.from_csv:
        if not args.csv.exists():
            raise FileNotFoundError(f"CSV not found: {args.csv}. Run export_bird_map.py first.")
        raw_n, n = plot_from_csv(img, args.csv, dedupe)
        mode = "detections (CSV)"
    else:
        raw_n, n = plot_labels(img, load_manifest(), dedupe)
        mode = "ground-truth labels"

    cv2.imwrite(str(args.output), img)

    scale = args.preview_width / img.shape[1]
    preview = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(args.preview), preview)

    print(f"Plotted {n} birds ({mode})" + (f" — removed {raw_n - n} duplicates from overlap" if dedupe and raw_n > n else ""))
    print(f"Full size: {args.output.resolve()}  ({img.shape[1]}x{img.shape[0]})")
    print(f"Preview:   {args.preview.resolve()}")


if __name__ == "__main__":
    main()
