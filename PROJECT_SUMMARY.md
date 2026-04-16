# Project Summary

This workspace implements an end-to-end street cleanliness analysis pipeline:
- Litter detection via YOLOv8
- Scene classification (present but **not trained/calibrated** in this workspace; treat as heuristic)
- Context-aware cleanliness scoring
- Spatial heatmap + hotspot detection
- Weighted semantic scoring

## Project structure

```
street_cleanliness_system/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   ├── dataset_loader.py
│   └── ...
├── models/
│   └── scene_classifier.py
├── utils/
│   ├── context_aware_scorer.py
│   ├── spatial_heatmap.py
│   └── weighted_semantic_scorer.py
├── inference/
│   └── detection_pipeline.py
├── output/
│   └── ... (generated)
└── docs/
    ├── PAPER.md
    └── IMPLEMENTATION_GUIDE.md
```

## Practical run results (this workspace)

- **Dataset subset used**: 223 images (156 train / 44 val / 23 test), 6 broad classes
  (plastic/metal/paper/organic/glass/other)
- **Detector**: YOLOv8n fine-tuned for 8 epochs on CPU
- **Validation metrics (epoch 8)** (from `runs/detect/output/taco_real_train/results.csv`):
  - Precision: 0.620
  - Recall: 0.093
  - mAP@0.50: 0.111
  - mAP@0.50:0.95: 0.085
- **Example end-to-end outputs**:
  - `output/real_demo/000083_context_pipeline.jpg` + `.json`
  - `output/real_demo/000054_context_pipeline.jpg` + `.json`

No human evaluation or ablation studies were conducted as part of this practical run.

## How to reproduce the real demo outputs

```bash
cd street_cleanliness_system

python main.py \
  --yolo-weights runs/detect/output/taco_real_train/weights/best.pt \
  demo \
  --source-dir data/processed/taco_yolo_real/images/test \
  --output-dir output/real_demo \
  --max-images 5
```

## Key files

| File | Purpose |
|------|---------|
| `main.py` | CLI (`image`, `video`, `demo`) |
| `inference/detection_pipeline.py` | End-to-end pipeline integration |
| `utils/context_aware_scorer.py` | Context-aware cleanliness score |
| `utils/spatial_heatmap.py` | Heatmap + hotspot analysis |
| `utils/weighted_semantic_scorer.py` | Semantic weighting score |
| `models/scene_classifier.py` | Scene classifier (heuristic unless trained) |
| `data/dataset_loader.py` | Dataset utilities (COCO/YOLO handling) |
