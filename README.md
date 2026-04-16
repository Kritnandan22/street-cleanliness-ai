# Context-Aware and Spatially-Intelligent Street Cleanliness Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://github.com/ultralytics/ultralytics)
[![Dataset](https://img.shields.io/badge/Dataset-TACO%201500%20images-green.svg)](http://tacodataset.org)

> **Final-Year EE-655 Computer Vision Project** — Transforms raw litter detection into intelligent environmental scoring through three novel contributions: context-aware scene normalisation, spatial heatmap analysis, and weighted semantic scoring.

---

## Novelty Summary

| Contribution | Description |
|---|---|
| **Context-Aware Scoring** | Score = f(detections, scene\_type). A park with 5 items scores worse than a street with 5 items. |
| **Spatial Heatmap** | 8×8 grid maps litter density; localises pollution hotspots for targeted cleanup. |
| **Weighted Semantic Scoring** | Plastic (weight=1.0) penalises more than organic (0.4); combines class, area, confidence. |

---

## Project Structure

```
street_cleanliness_system/
├── main.py                       ← CLI entry point (image / video / demo)
├── train.py                      ← YOLOv8 fine-tuning on full TACO dataset
├── evaluate.py                   ← mAP + 4-mode ablation study
├── config.py                     ← Global configuration
├── requirements.txt
│
├── data/
│   ├── prepare_taco_yolo_subset.py  ← COCO → YOLO conversion + train/val/test split
│   ├── raw/TACO/                    ← Downloaded TACO images (1500 images, 15 batches)
│   └── processed/taco_yolo_real/    ← YOLO-format processed dataset
│       ├── dataset.yaml
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
│
├── models/
│   └── scene_classifier.py       ← MobileNetV2-based scene classifier (road/park/street/indoor)
│
├── inference/
│   └── detection_pipeline.py     ← End-to-end system integrating all components
│
├── utils/
│   ├── context_aware_scorer.py   ← Novelty 1: scene-normalised cleanliness score
│   ├── spatial_heatmap.py        ← Novelty 2: 8×8 grid heatmap + hotspot detection
│   └── weighted_semantic_scorer.py ← Novelty 3: class-weighted litter severity
│
├── visualization/
│   └── visualizer.py             ← All drawing/overlay/chart utilities
│
└── docs/
    ├── PAPER.md                  ← CVPR-style academic write-up
    └── IMPLEMENTATION_GUIDE.md   ← Detailed implementation notes
```

---

## Prerequisites & Installation

```bash
# 1. Clone / navigate to project
cd /path/to/street_cleanliness_system

# 2. Create and activate virtual environment
python3 -m venv ../.venv
source ../.venv/bin/activate      # macOS/Linux
# ..\\.venv\\Scripts\\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt** pins:
```
ultralytics>=8.0.0   (YOLOv8)
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.65.0
pillow>=9.0.0
```

---

## Dataset Preparation

### Step 1 — Download TACO (1500 images)

The TACO annotations file is already included at `data/raw/TACO/data/annotations.json`.
Download all 15 batches from Flickr:

```bash
cd data/raw/TACO
python download.py --dataset_path ./data/annotations.json
cd ../../..
```

> ℹ️ TACO contains **1500 real-world litter images**, **60 fine-grained classes**, and **4784 annotations**. The download script is resumable — just re-run if interrupted.

### Step 2 — Convert to YOLO Format

```bash
python data/prepare_taco_yolo_subset.py
```

This:
- Maps 60 TACO classes → 6 semantic groups: `plastic`, `metal`, `paper`, `organic`, `glass`, `other`
- Converts COCO bboxes → YOLO normalised format
- Creates **70 / 20 / 10** train / val / test split
- Writes `data/processed/taco_yolo_real/dataset.yaml`

Expected output:
```
available_images=1500
train_images=1050
val_images=300
test_images=150
class_annotation_counts:
  plastic: ~2000
  metal:   ~500
  paper:   ~350
  organic: ~800
  glass:   ~250
  other:   ~900
```

---

## Training

Fine-tune YOLOv8n on the full TACO dataset:

```bash
# Default (50 epochs, batch 8, CPU)
python train.py

# GPU training (recommended)
python train.py --device cuda --epochs 100 --batch 16

# With dataset re-preparation
python train.py --prepare-data --epochs 100 --batch 16 --device cuda

# Resume from checkpoint
python train.py --resume --epochs 100
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Training epochs |
| `--batch` | 8 | Batch size |
| `--imgsz` | 640 | Image resolution |
| `--device` | cpu | `cpu`, `cuda`, or GPU index |
| `--name` | taco_cleanliness | Experiment name |
| `--patience` | 15 | Early-stop patience |
| `--weights` | yolov8n.pt | Base model |

Best weights are saved to `models/best_yolov8_taco.pt` and `runs/detect/taco_cleanliness/weights/best.pt`.

---

## Inference

### Image Analysis

```bash
# Basic analysis
python main.py image --source photo.jpg --output result.jpg

# Save JSON results
python main.py image --source photo.jpg --output result.jpg --save-json results.json

# Use fine-tuned weights
python main.py image --source photo.jpg \
    --yolo-weights models/best_yolov8_taco.pt --output result.jpg

# GPU inference
python main.py --device cuda image --source photo.jpg --output result.jpg
```

### Video / Webcam

```bash
# Process video
python main.py video --source street_video.mp4 --output annotated.mp4

# Webcam (source=0)
python main.py video --source 0 --output webcam_out.mp4

# Skip every 3 frames for speed
python main.py video --source video.mp4 --skip-frames 3
```

### Demo Mode

```bash
# Run on prepared test images
python main.py demo

# Run on custom image folder
python main.py demo --source-dir /path/to/images --output-dir output/my_demo
```

---

## Evaluation & Ablation Study

```bash
# Full ablation on test set (using ground-truth labels)
python evaluate.py

# mAP evaluation with trained weights
python evaluate.py --weights models/best_yolov8_taco.pt --map

# Ablation using live YOLO detections
python evaluate.py --weights models/best_yolov8_taco.pt

# Custom split / scene
python evaluate.py --split val --scene park --output output/eval_park
```

Output files in `output/evaluation/`:
- `ablation_results.csv` — per-image scores for all 4 modes
- `ablation_chart.png` — bar + line chart comparison
- `evaluation_summary.json` — aggregated statistics and mAP

### Ablation Modes

| Mode | Method | Novel? |
|---|---|---|
| A — Raw Count | count / 4 | Baseline |
| B — Context-Aware | count / scene\_baseline | ✅ Yes |
| C — Weighted Semantic | Σ(weight × area × conf) | ✅ Yes |
| D — Full Pipeline | 0.55×B + 0.45×C | ✅ Yes (proposed) |

---

## Output Description

The annotated output image contains:

1. **Colour-coded bounding boxes** — each litter class has a distinct colour
2. **Spatial heatmap** — JET-coloured overlay showing litter density regions  
3. **Hotspot markers** — top 5 dirtiest grid cells highlighted with severity labels
4. **HUD panel** — scene type, raw count, context score, weighted score, level, action

Score interpretation (0–5 scale):

| Score | Level | Action |
|---|---|---|
| 4.5–5.0 | Excellent | Maintain current practices |
| 3.5–4.5 | Good | Continue regular maintenance |
| 2.5–3.5 | Average | Schedule routine cleanup |
| 1.5–2.5 | Poor | Schedule immediate cleanup |
| 0.0–1.5 | Critical | Emergency cleanup required |

---

## Academic Write-Up

See [`docs/PAPER.md`](docs/PAPER.md) for the full CVPR-style paper covering:
- Abstract, Introduction, Related Work
- Methodology (with equations for all 3 novel components)
- Experiments & Results with ablation tables
- Conclusion & Future Work

---

## License

This project uses the [TACO dataset](http://tacodataset.org) which is licensed under Creative Commons Attribution 4.0. Code is for academic use.
