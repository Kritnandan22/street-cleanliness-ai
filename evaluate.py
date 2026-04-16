#!/usr/bin/env python3
"""
Evaluation & Ablation Study for Street Cleanliness Detection System

Runs four scoring modes on the test set and compares them:
  Mode A — Raw Count         : simply counts detections
  Mode B — Context-Aware     : normalises count by scene baseline
  Mode C — Weighted Semantic : weights by litter class importance
  Mode D — Full Pipeline     : context-aware + semantic weighting (proposed)

Also computes YOLOv8 mAP if trained weights are supplied.

Usage:
    python evaluate.py                              # ablation on test split
    python evaluate.py --weights models/best_yolov8_taco.pt --map
    python evaluate.py --split val --output output/eval_val
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "taco_yolo_real" / "images" / "test"
DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "taco_yolo_real" / "dataset.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "evaluation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_images(split_dir: Path, max_images: int = 0) -> List[Path]:
    """Return sorted list of image paths in a split directory."""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [p for p in sorted(split_dir.iterdir()) if p.suffix in exts]
    if max_images > 0:
        images = images[:max_images]
    return images


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Dict]:
    """Load YOLO .txt labels and convert to pixel bboxes."""
    detections = []
    if not label_path.exists():
        return detections
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        detections.append({
            "class_id": cls_id,
            "class_name": _class_name(cls_id),
            "confidence": 1.0,
            "bbox": (x1, y1, x2, y2),
            "area": w * h,  # normalised area
        })
    return detections


CLASS_NAMES = ["plastic", "metal", "paper", "organic", "glass", "other"]
SEMANTIC_WEIGHTS = {
    "plastic": 1.0, "glass": 0.95, "metal": 0.9,
    "paper": 0.7, "organic": 0.4, "other": 0.6,
}
SCENE_BASELINES = {"road": 15, "park": 8, "street": 20, "indoor": 3}


def _class_name(cls_id: int) -> str:
    return CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "other"


# ---------------------------------------------------------------------------
# Four scoring modes
# ---------------------------------------------------------------------------

def score_raw_count(detections: List[Dict], **_) -> float:
    """Mode A: Raw count mapped to 0-5 (higher count → lower score)."""
    count = len(detections)
    return max(0.0, 5.0 - (count / 4))  # 0 → 5.0; 20+ → ~0


def score_context_aware(
    detections: List[Dict],
    scene_class: str = "street",
    **_
) -> float:
    """Mode B: Normalise count by scene-specific baseline."""
    baseline = SCENE_BASELINES.get(scene_class, 12)
    ratio = min(len(detections) / baseline, 2.0)
    return round((1.0 - ratio / 2.0) * 5.0, 3)


def score_weighted_semantic(
    detections: List[Dict],
    img_w: int = 640,
    img_h: int = 480,
    **_
) -> float:
    """Mode C: Σ(class_weight × bbox_area × confidence), scaled to 0-5."""
    total = 0.0
    img_area = img_w * img_h or 1
    for det in detections:
        w = SEMANTIC_WEIGHTS.get(det["class_name"], 0.6)
        x1, y1, x2, y2 = det["bbox"]
        area = ((x2 - x1) * (y2 - y1)) / img_area
        total += w * area * det["confidence"]
    return round(max(0.0, 5.0 - total * 100), 3)


def score_full_pipeline(
    detections: List[Dict],
    scene_class: str = "street",
    img_w: int = 640,
    img_h: int = 480,
    **_
) -> float:
    """Mode D: Context-aware + weighted semantic (proposed system)."""
    ctx = score_context_aware(detections, scene_class=scene_class)
    wsem = score_weighted_semantic(detections, img_w=img_w, img_h=img_h)
    # Combine: context carries scene awareness, semantic carries class severity
    combined = 0.55 * ctx + 0.45 * wsem
    return round(np.clip(combined, 0.0, 5.0), 3)


# ---------------------------------------------------------------------------
# mAP evaluation via ultralytics
# ---------------------------------------------------------------------------

def run_map_evaluation(
    weights_path: Path,
    dataset_yaml: Path,
    split: str = "test",
    imgsz: int = 640,
    device: str = "cpu",
    output_dir: Path = DEFAULT_OUTPUT,
) -> Optional[Dict]:
    """Compute mAP50 and mAP50-95 using YOLOv8's built-in val()."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics not available — skipping mAP computation.")
        return None

    if not weights_path.exists():
        print(f"[!] Weights not found: {weights_path} — skipping mAP.")
        return None

    print(f"\n[*] Computing mAP on split='{split}' ...")
    model = YOLO(str(weights_path))
    results = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        project=str(output_dir / "yolo_val"),
        name="map_eval",
        save_json=True,
        plots=True,
        verbose=False,
    )

    metrics = {
        "mAP50": round(results.box.map50, 4),
        "mAP50_95": round(results.box.map, 4),
        "precision": round(results.box.mp, 4),
        "recall": round(results.box.mr, 4),
    }

    print(f"  mAP50     : {metrics['mAP50']}")
    print(f"  mAP50-95  : {metrics['mAP50_95']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")

    return metrics


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(
    split_dir: Path,
    output_dir: Path,
    scene_class: str = "street",
    max_images: int = 0,
    use_yolo: bool = False,
    yolo_weights: Optional[Path] = None,
    device: str = "cpu",
) -> List[Dict]:
    """Run all four scoring modes on the split images and return per-image rows."""
    images = load_test_images(split_dir, max_images)
    if not images:
        print(f"[!] No images found in {split_dir}")
        return []

    labels_base = split_dir.parent.parent / "labels" / split_dir.name

    print(f"\n[*] Running ablation on {len(images)} images from '{split_dir.name}' split ...")

    # Optional: use YOLO to generate live detections
    yolo_model = None
    if use_yolo and yolo_weights and yolo_weights.exists():
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(yolo_weights))
            print(f"[*] Using YOLO detections from: {yolo_weights}")
        except Exception as e:
            print(f"[!] Could not load YOLO model: {e}")

    rows = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        if yolo_model is not None:
            # Use live YOLO detections securely bound to the correct device
            result = yolo_model(img, conf=0.35, device=device, verbose=False)[0]
            detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                detections.append({
                    "class_id": cls_id,
                    "class_name": _class_name(cls_id),
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "area": ((x2 - x1) * (y2 - y1)) / (w * h),
                })
        else:
            # Fall back to ground-truth labels
            label_path = labels_base / (img_path.stem + ".txt")
            detections = load_yolo_labels(label_path, w, h)

        # Compute all four scores
        sa = score_raw_count(detections)
        sb = score_context_aware(detections, scene_class=scene_class)
        sc = score_weighted_semantic(detections, img_w=w, img_h=h)
        sd = score_full_pipeline(detections, scene_class=scene_class, img_w=w, img_h=h)

        rows.append({
            "image": img_path.name,
            "num_detections": len(detections),
            "scene_class": scene_class,
            "mode_A_raw_count": sa,
            "mode_B_context_aware": sb,
            "mode_C_weighted_semantic": sc,
            "mode_D_full_pipeline": sd,
            "delta_B_minus_A": round(sb - sa, 3),
            "delta_D_minus_A": round(sd - sa, 3),
        })

    print(f"[+] Ablation complete — {len(rows)} images processed.")
    return rows


def save_ablation_csv(rows: List[Dict], output_dir: Path):
    """Save ablation results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_results.csv"
    if not rows:
        return csv_path
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[+] Ablation CSV saved: {csv_path}")
    return csv_path


def print_ablation_summary(rows: List[Dict]):
    """Print aggregate comparison table to stdout."""
    if not rows:
        return
    keys = ["mode_A_raw_count", "mode_B_context_aware",
            "mode_C_weighted_semantic", "mode_D_full_pipeline"]
    labels = {
        "mode_A_raw_count": "A: Raw Count",
        "mode_B_context_aware": "B: Context-Aware",
        "mode_C_weighted_semantic": "C: Weighted Semantic",
        "mode_D_full_pipeline": "D: Full Pipeline (Ours)"
    }
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS (mean ± std over test images)")
    print("=" * 60)
    print(f"{'Mode':<28}  {'Mean':>6}  {'Std':>6}  {'Min':>6}  {'Max':>6}")
    print("-" * 60)
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"{labels[k]:<28}  {np.mean(vals):6.3f}  {np.std(vals):6.3f}  "
              f"{np.min(vals):6.3f}  {np.max(vals):6.3f}")
    print("=" * 60)
    print(f"Total images evaluated: {len(rows)}")


def save_ablation_chart(rows: List[Dict], output_dir: Path):
    """Save a bar chart comparing the four scoring modes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed — skipping chart.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    keys = ["mode_A_raw_count", "mode_B_context_aware",
            "mode_C_weighted_semantic", "mode_D_full_pipeline"]
    labels = ["A: Raw Count", "B: Context-Aware",
              "C: Weighted Semantic", "D: Full Pipeline\n(Ours)"]
    means = [np.mean([r[k] for r in rows]) for k in keys]
    stds = [np.std([r[k] for r in rows]) for k in keys]
    colors = ["#6c757d", "#17a2b8", "#ffc107", "#28a745"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation Study: Cleanliness Scoring Modes (0-5 scale)",
                 fontsize=14, fontweight="bold")

    # Bar chart — average scores
    ax = axes[0]
    bars = ax.bar(labels, means, yerr=stds, color=colors,
                  capsize=5, edgecolor="black", linewidth=0.8)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean Cleanliness Score (0-5)")
    ax.set_title("Average Score per Mode")
    ax.axhline(y=2.5, color="red", linestyle="--", alpha=0.5, label="Midpoint (2.5)")
    ax.legend(fontsize=9)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.05,
                f"{m:.2f}", ha="center", va="bottom", fontsize=10)

    # Per-image line plot for top 20 images
    ax2 = axes[1]
    n = min(30, len(rows))
    sample = rows[:n]
    x = range(n)
    for k, lbl, col in zip(keys, ["A", "B", "C", "D"], colors):
        ax2.plot(x, [r[k] for r in sample], marker="o", markersize=3,
                 label=lbl, color=col, linewidth=1.5)
    ax2.set_xlabel("Image Index")
    ax2.set_ylabel("Cleanliness Score (0-5)")
    ax2.set_title(f"Score Comparison (first {n} test images)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 5.2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / "ablation_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Ablation chart saved: {chart_path}")
    return chart_path


def save_summary_json(
    rows: List[Dict],
    map_metrics: Optional[Dict],
    output_dir: Path
):
    """Save full evaluation summary as JSON."""
    keys = ["mode_A_raw_count", "mode_B_context_aware",
            "mode_C_weighted_semantic", "mode_D_full_pipeline"]
    summary = {
        "num_images_evaluated": len(rows),
        "ablation": {
            k: {
                "mean": round(float(np.mean([r[k] for r in rows])), 4),
                "std": round(float(np.std([r[k] for r in rows])), 4),
                "min": round(float(np.min([r[k] for r in rows])), 4),
                "max": round(float(np.max([r[k] for r in rows])), 4),
            }
            for k in keys
        },
        "detection_stats": {
            "mean_detections": round(float(np.mean([r["num_detections"] for r in rows])), 2),
            "std_detections": round(float(np.std([r["num_detections"] for r in rows])), 2),
            "max_detections": int(max(r["num_detections"] for r in rows)),
        },
    }
    if map_metrics:
        summary["map_metrics"] = map_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Summary JSON saved: {json_path}")
    return json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate & ablate the Street Cleanliness Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Directory to save evaluation outputs")
    parser.add_argument("--weights", default=None,
                        help="Path to trained YOLOv8 weights (.pt) for live detections")
    parser.add_argument("--map", action="store_true",
                        help="Run mAP evaluation with provided --weights")
    parser.add_argument("--device", default="cpu",
                        help="Device for YOLOv8 inference")
    parser.add_argument("--scene", default="street",
                        choices=["road", "park", "street", "indoor"],
                        help="Scene class to use for context-aware scoring")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images to evaluate (0 = all)")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("STREET CLEANLINESS EVALUATION & ABLATION")
    print("=" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = (PROJECT_ROOT / "data" / "processed" / "taco_yolo_real"
                 / "images" / args.split)

    # --- mAP evaluation ---
    map_metrics = None
    if args.map and args.weights:
        map_metrics = run_map_evaluation(
            weights_path=Path(args.weights),
            dataset_yaml=DATASET_YAML,
            split=args.split,
            imgsz=args.imgsz,
            device=args.device,
            output_dir=output_dir,
        )

    # --- mAP-only evaluation if weights provided (but --map not set) ---
    weights_path = Path(args.weights) if args.weights else None

    # --- Ablation study ---
    rows = run_ablation(
        split_dir=split_dir,
        output_dir=output_dir,
        scene_class=args.scene,
        max_images=args.max_images,
        use_yolo=(weights_path is not None),
        yolo_weights=weights_path,
        device=args.device,
    )

    if rows:
        print_ablation_summary(rows)
        save_ablation_csv(rows, output_dir)
        save_ablation_chart(rows, output_dir)
        save_summary_json(rows, map_metrics, output_dir)

    print(f"\n[+] All evaluation outputs in: {output_dir}\n")


if __name__ == "__main__":
    main()
