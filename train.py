#!/usr/bin/env python3

# todo: krit fix validate_dataset to check label count too
#todo: karan fix train epochs

import argparse
import shutil
from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "taco_yolo_real" / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"


def validate_dataset(dataset_yaml: Path) -> bool:
    if not dataset_yaml.exists():
        print(f"[!] dataset yaml not found: {dataset_yaml}")
        print("    run 'python data/prepare_taco_yolo_subset.py' first.")
        return False

    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", dataset_yaml.parent))
    train_path = base / cfg.get("train", "images/train")
    val_path = base / cfg.get("val", "images/val")

    if not train_path.exists():
        print(f"[!] train dir missing: {train_path}")
        return False

    train_imgs = list(train_path.glob("*.jpg")) + list(train_path.glob("*.JPG")) + list(train_path.glob("*.png"))
    val_imgs = list(val_path.glob("*.jpg")) + list(val_path.glob("*.JPG")) + list(val_path.glob("*.png"))

    print(f"[+] dataset ok:")
    print(f"    train images : {len(train_imgs)}")
    print(f"    val   images : {len(val_imgs)}")
    print(f"    classes      : {cfg.get('nc', 'unknown')}")
    print(f"    class names  : {list(cfg.get('names', {}).values())}")
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
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics not installed. pip install ultralytics")
        return

    print("\n" + "=" * 60)
    print("street cleanliness yolov8 training")
    print("=" * 60)

    if not validate_dataset(DATASET_YAML):
        return

    # pick weights: resume from last checkpoint or start fresh
    if resume:
        last_run = sorted(RUNS_DIR.glob(f"{name}*/weights/last.pt"))
        if last_run:
            weights = str(last_run[-1])
            print(f"[*] resuming: {weights}")
        else:
            print("[!] no checkpoint found, starting fresh.")
            weights = base_weights
    else:
        weights = base_weights
        local_weights = PROJECT_ROOT / base_weights
        if local_weights.exists():
            weights = str(local_weights)

    print(f"\n[*] weights     : {weights}")
    print(f"[*] dataset     : {DATASET_YAML}")
    print(f"[*] epochs      : {epochs}")
    print(f"[*] batch       : {batch}")
    print(f"[*] imgsz       : {imgsz}")
    print(f"[*] device      : {device}")
    print(f"[*] name        : {name}")
    print(f"[*] patience    : {patience}")
    print()

    model = YOLO(weights)

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
        # augmentation params
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,
        val=True,
        plots=True,
        save=True,
        save_period=10,
        verbose=True,
    )

    best_weights = RUNS_DIR / name / "weights" / "best.pt"
    if not best_weights.exists():
        candidates = sorted(RUNS_DIR.glob(f"{name}*/weights/best.pt"))
        best_weights = candidates[-1] if candidates else None

    print("\n" + "=" * 60)
    print("training complete")
    print("=" * 60)

    if best_weights and best_weights.exists():
        dest = PROJECT_ROOT / "models" / "best_yolov8_taco.pt"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(best_weights, dest)
        print(f"[+] best weights: {best_weights}")
        print(f"[+] copied to   : {dest}")
    else:
        print("[!] best.pt not found — check runs/detect/ manually.")

    try:
        metrics = results.results_dict
        print(f"\n--- metrics ---")
        print(f"  map50     : {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  map50-95  : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  precision : {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  recall    : {metrics.get('metrics/recall(B)', 0):.4f}")
    except Exception:
        pass

    print(f"\n[+] run output: {RUNS_DIR / name}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="train yolov8 on taco litter dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--name", type=str, default="taco_cleanliness")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--prepare-data", action="store_true")

    args = parser.parse_args()

    if args.prepare_data:
        print("[*] re-running dataset prep...")
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
