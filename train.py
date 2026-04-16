#!/usr/bin/env python3
"""
YOLOv8 Fine-Tuning Script for Street Cleanliness Detection
Trains on the complete TACO dataset (1500 images, 6 semantic classes)

Usage:
    python train.py                              # default settings
    python train.py --epochs 50 --batch 16      # custom settings
    python train.py --device cuda --epochs 100  # GPU training
"""

import argparse
import shutil
from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "taco_yolo_real" / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"


def validate_dataset(dataset_yaml: Path) -> bool:
    """Validate that the processed dataset exists and has images."""
    if not dataset_yaml.exists():
        print(f"[!] Dataset YAML not found: {dataset_yaml}")
        print("    Run 'python data/prepare_taco_yolo_subset.py' first.")
        return False

    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", dataset_yaml.parent))
    train_path = base / cfg.get("train", "images/train")
    val_path = base / cfg.get("val", "images/val")

    if not train_path.exists():
        print(f"[!] Train images directory missing: {train_path}")
        return False

    train_imgs = list(train_path.glob("*.jpg")) + list(train_path.glob("*.JPG")) + list(train_path.glob("*.png"))
    val_imgs = list(val_path.glob("*.jpg")) + list(val_path.glob("*.JPG")) + list(val_path.glob("*.png"))

    print(f"[+] Dataset validated:")
    print(f"    Train images : {len(train_imgs)}")
    print(f"    Val   images : {len(val_imgs)}")
    print(f"    Classes      : {cfg.get('nc', 'unknown')}")
    print(f"    Class names  : {list(cfg.get('names', {}).values())}")
    return True


def train(
    epochs: int = 50,
    batch: int = 8,
    imgsz: int = 640,
    device: str = "cpu",
    name: str = "taco_cleanliness",
    patience: int = 15,
    base_weights: str = "yolov8n.pt",
    resume: bool = False,
    cache: bool = False,
):
    """Run YOLOv8 fine-tuning on the TACO litter dataset.

    Args:
        epochs    : Number of training epochs
        batch     : Batch size (reduce if OOM)
        imgsz     : Training image size (must be multiple of 32)
        device    : 'cpu' or 'cuda' or '0' for GPU index
        name      : Experiment name (saved under runs/detect/)
        patience  : Early stopping patience
        base_weights: Starting weights (yolov8n/s/m/l/x.pt)
        resume    : Resume from last checkpoint if True
        cache     : Cache images in RAM for faster training
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics is not installed.")
        print("    Install it: pip install ultralytics")
        return

    print("\n" + "=" * 60)
    print("STREET CLEANLINESS YOLOV8 TRAINING")
    print("=" * 60)

    if not validate_dataset(DATASET_YAML):
        return

    # Load model
    if resume:
        last_run = sorted(RUNS_DIR.glob(f"{name}*/weights/last.pt"))
        if last_run:
            weights = str(last_run[-1])
            print(f"[*] Resuming from: {weights}")
        else:
            print("[!] No checkpoint found to resume. Starting fresh.")
            weights = base_weights
    else:
        weights = base_weights
        # Check for local copy downloaded by main.py
        local_weights = PROJECT_ROOT / base_weights
        if local_weights.exists():
            weights = str(local_weights)

    print(f"\n[*] Base weights    : {weights}")
    print(f"[*] Dataset YAML    : {DATASET_YAML}")
    print(f"[*] Epochs          : {epochs}")
    print(f"[*] Batch size      : {batch}")
    print(f"[*] Image size      : {imgsz}")
    print(f"[*] Device          : {device}")
    print(f"[*] Experiment name : {name}")
    print(f"[*] Early stopping  : patience={patience}")
    print()

    model = YOLO(weights)

    # Train
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        name=name,
        project=str(RUNS_DIR),
        patience=patience,
        cache=cache,
        # Augmentation (on top of YOLOv8 defaults)
        fliplr=0.5,          # horizontal flip
        hsv_h=0.015,         # hue variation
        hsv_s=0.7,           # saturation variation
        hsv_v=0.4,           # brightness variation
        degrees=10.0,        # rotation ±10°
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Validation
        val=True,
        plots=True,
        save=True,
        save_period=10,      # save checkpoint every 10 epochs
        verbose=True,
    )

    # Locate best weights
    best_weights = RUNS_DIR / name / "weights" / "best.pt"
    if not best_weights.exists():
        # Try numbered variant (ultralytics appends index on re-run)
        candidates = sorted(RUNS_DIR.glob(f"{name}*/weights/best.pt"))
        best_weights = candidates[-1] if candidates else None

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    if best_weights and best_weights.exists():
        # Copy best weights to models/ for easy reference
        dest = PROJECT_ROOT / "models" / "best_yolov8_taco.pt"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(best_weights, dest)
        print(f"[+] Best weights saved : {best_weights}")
        print(f"[+] Also copied to     : {dest}")
    else:
        print("[!] Could not locate best.pt — check runs/detect/ manually.")

    # Print metrics
    try:
        metrics = results.results_dict
        print(f"\n--- Final Metrics ---")
        print(f"  mAP50     : {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP50-95  : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision : {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall    : {metrics.get('metrics/recall(B)', 0):.4f}")
    except Exception:
        pass

    print(f"\n[+] Run output at: {RUNS_DIR / name}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on the full TACO litter dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (reduce if OOM; 4 for <4GB RAM)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (must be multiple of 32)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu', 'cuda', or GPU index '0'")
    parser.add_argument("--name", type=str, default="taco_cleanliness",
                        help="Experiment name (subdirectory in runs/detect/)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early-stopping patience (0 = disabled)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt",
                        help="Base model weights (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--cache", action="store_true",
                        help="Cache images in RAM for faster training")
    parser.add_argument("--prepare-data", action="store_true",
                        help="Re-run dataset preparation before training")

    args = parser.parse_args()

    # Optionally re-run dataset preparation
    if args.prepare_data:
        print("[*] Re-running dataset preparation...")
        import subprocess, sys
        prep_script = PROJECT_ROOT / "data" / "prepare_taco_yolo_subset.py"
        subprocess.run([sys.executable, str(prep_script)], check=True)

    train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        patience=args.patience,
        base_weights=args.weights,
        resume=args.resume,
        cache=args.cache,
    )


if __name__ == "__main__":
    main()
