"""Prepare a real YOLO dataset subset from downloaded TACO images.

This script uses only images that are physically present under data/raw/TACO/data,
converts COCO boxes to YOLO txt labels, and creates train/val/test splits.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TACO_ROOT = PROJECT_ROOT / "data" / "raw" / "TACO"
ANNOTATIONS_PATH = TACO_ROOT / "data" / "annotations.json"
IMAGES_ROOT = TACO_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "taco_yolo_real"

# Group fine-grained TACO classes into broader semantic buckets.
CLASS_GROUPS = {
    "plastic": {
        "plastic",
        "plastic bag",
        "plastic bottle",
        "plastic cup",
        "plastic container",
        "other plastic",
        "drink can pull tab",
        "plastic glooves",
        "plastic utensils",
        "pop tab",
        "single-use carrier bag",
        "polypropylene bag",
        "clear plastic bottle",
        "plastic film",
        "food can",
    },
    "metal": {
        "metal",
        "can",
        "drink can",
        "scrap metal",
        "aluminium foil",
        "metal bottle cap",
        "metal container",
    },
    "paper": {
        "paper",
        "carton",
        "paper bag",
        "paper cup",
        "paperboard",
        "normal paper",
    },
    "organic": {
        "wood",
        "food waste",
        "cigarette",
        "cigarette butt",
        "fruit",
        "leaf",
    },
    "glass": {
        "glass",
        "glass bottle",
        "broken glass",
    },
    "other": set(),
}

TARGET_CLASSES = ["plastic", "metal", "paper", "organic", "glass", "other"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


def map_category(name: str) -> str:
    lname = name.strip().lower()
    for group, values in CLASS_GROUPS.items():
        if lname in values:
            return group
    for group in ["plastic", "metal", "paper", "organic", "glass"]:
        if group in lname:
            return group
    return "other"


def coco_to_yolo_bbox(bbox, img_w: int, img_h: int):
    x_min, y_min, bw, bh = bbox
    x_center = (x_min + bw / 2.0) / img_w
    y_center = (y_min + bh / 2.0) / img_h
    w = bw / img_w
    h = bh / img_h
    return x_center, y_center, w, h


def main() -> None:
    random.seed(42)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with ANNOTATIONS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}

    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    available = []
    for img_id, img in images.items():
        img_path = IMAGES_ROOT / img["file_name"]
        if img_path.exists():
            available.append((img_id, img))

    random.shuffle(available)
    n = len(available)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)

    splits = {
        "train": available[:n_train],
        "val": available[n_train:n_train + n_val],
        "test": available[n_train + n_val:],
    }

    for split in splits:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {s: 0 for s in splits}
    class_counts = {c: 0 for c in TARGET_CLASSES}

    for split, items in splits.items():
        for img_id, img in items:
            src_img = IMAGES_ROOT / img["file_name"]
            # To avoid overwriting (since multiple batches might have 000000.jpg),
            # we prefix the batch directory name into the stem.
            stem = img["file_name"].replace('/', '_').replace('.jpg', '')
            dst_img = OUTPUT_ROOT / "images" / split / f"{stem}.jpg"
            dst_lbl = OUTPUT_ROOT / "labels" / split / f"{stem}.txt"

            shutil.copy2(src_img, dst_img)

            img_w = img["width"]
            img_h = img["height"]
            lines = []

            for ann in anns_by_img.get(img_id, []):
                cat_name = categories.get(ann["category_id"], "other")
                mapped = map_category(cat_name)
                class_counts[mapped] += 1
                cls_id = CLASS_TO_ID[mapped]
                x, y, w, h = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            dst_lbl.write_text("\n".join(lines), encoding="utf-8")
            stats[split] += 1

    names_yaml = "\n".join(
        [f"  {i}: {name}" for i, name in enumerate(TARGET_CLASSES)])
    yaml_text = (
        f"path: {OUTPUT_ROOT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(TARGET_CLASSES)}\n"
        "names:\n"
        f"{names_yaml}\n"
    )
    (OUTPUT_ROOT / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    report_lines = [
        f"available_images={n}",
        f"train_images={stats['train']}",
        f"val_images={stats['val']}",
        f"test_images={stats['test']}",
        "class_annotation_counts:",
    ]
    for cls_name in TARGET_CLASSES:
        report_lines.append(f"  {cls_name}: {class_counts[cls_name]}")

    (OUTPUT_ROOT / "prep_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nSaved YOLO dataset to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
