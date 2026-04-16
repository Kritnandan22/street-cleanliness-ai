# Implementation Guide and Methodology Details

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run on image
python main.py image --source street_photo.jpg --output result.jpg

# 3. Run on video
python main.py video --source street_video.mp4 --output output.mp4

# 4. Run demo
python main.py demo
```

## Detailed Component Explanations

### 1. Context-Aware Scoring: Deep Dive

**Problem**: Different environments have different "normal" litter levels:
- Busy downtown street: some litter is normal
- Park: people expect cleanliness
- Indoor office: almost zero litter acceptable

**Our Solution**: Normalize against baseline expectations

#### Example Walk-Through

**Scenario**: 10 litter detections in a park

**Step 1: Identify Scene**
```
Scene Classifier → "Park"
```

**Step 2: Get Baseline**
```
Baselines["Park"] = 8
```

**Step 3: Compute Ratio**
```
Ratio = 10 / 8 = 1.25
```

**Step 4: Normalize**
```
Score = (1 - min(1.25, 2.0) / 2.0) × 5.0
      = (1 - 1.25/2.0) × 5.0
      = (1 - 0.625) × 5.0
      = 0.375 × 5.0
      = 1.875 ≈ 1.9/5.0
      → "Poor" (unexpected litter for park)
```

**Same 10 detections on a street:**
```
Baselines["Street"] = 20
Score = (1 - 10/20/2.0) × 5.0 = 3.75/5.0 → "Average"
```

**Key**: Context makes same detection count mean different things!

### 2. Spatial Heatmap Analysis: Deep Dive

**Purpose**: Show WHERE litter is concentrated, not just total count

#### Grid-Based Approach

**Divide image into 8×8 grid (64 cells)**

```
Original Image (640×480)
    ↓
     ┌─────────────────────────┐
     │ │ │ │ │ │ │ │ │        │
     ├─┼─┼─┼─┼─┼─┼─┼─┤        │
     │●│ │●│ │ │ │ │ │  ← Cells with litter
     ├─┼─┼─┼─┼─┼─┼─┼─┤        │
     │ │ │ │●│●│●│ │ │  ← Hotspot region
     ├─┼─┼─┼─┼─┼─┼─┼─┤        │
     │ │ │ │ │ │ │ │ │        │
     └─────────────────────────┘
```

**For each cell: Count detections overlapping that cell**

```
Cell(3,2): 3 detections → HIGH
Cell(1,0): 1 detection  → LOW
Cell(0,0): 0 detections → CLEAN
```

**Generate Heatmap**

1. Create 8×8 matrix with counts
2. Resize to original image size
3. Apply Gaussian blur (σ=31) for smooth gradient
4. Apply JET colormap (blue→green→red)

**Result**: Visual representation of pollution density

#### Hotspot Identification

**Find "problem zones"**: Top 25% dirtiest cells

```python
# Pseudo-code
heatmap_values = flatten_grid()  # Get all counts
threshold = percentile(heatmap_values, 75)  # 75th percentile

hotspots = []
for cell in grid:
    if cell.count >= threshold:
        hotspots.append(cell)
        
hotspots.sort(by=pollution_level)  # Rank by severity
```

**Output**: List of problem regions with pixel boundaries

### 3. Weighted Semantic Scoring: Deep Dive

**Core Idea**: Not all litter is equal

- 1 plastic bag = more important than 5 cigarette butts
- Even though cigarette count is higher!

#### Weight Assignment

```python
# Environmental impact hierarchy
weights = {
    "plastic": 1.0,      # ~200+ years to decompose
    "metal": 0.9,        # ~200 years
    "glass": 0.95,       # Dangerous, doesn't decompose
    "paper": 0.7,        # ~5-10 years
    "organic": 0.4       # <1 year
}
```

#### Score Computation

For each detection:
```
contribution = weight × area × confidence

Example:
- Large plastic bag (area=3%, conf=0.9, weight=1.0)
  contribution = 1.0 × 0.03 × 0.9 = 0.027

- Small cigarette (area=0.1%, conf=0.7, weight=0.3)  
  contribution = 0.3 × 0.001 × 0.7 = 0.00021

Plastic is ~128× more important!
```

**Total Score**: Sum all contributions, scale to 0-5

---

## Using the Library Directly

### Example 1: Process Single Image Programmatically

```python
from inference.detection_pipeline import StreetCleanlinessDetectionSystem

# Initialize
system = StreetCleanlinessDetectionSystem(device="cuda")

# Process
results = system.process_image("photo.jpg")

# Access results
print(f"Cleanliness: {results['context_aware_score']}")
print(f"Scene: {results['scene_class']}")
print(f"Litter: {results['litter_count']}")
print(f"Hotspots: {len(results['hotspots'])}")

# Visualize
vis = system.visualize_results("photo.jpg", results)
import cv2
cv2.imwrite("output.jpg", vis)
```

### Example 2: Using Components Independently

```python
from utils.context_aware_scorer import ContextAwareScorer
from utils.spatial_heatmap import SpatialHeatmapAnalyzer
from utils.weighted_semantic_scorer import WeightedSemanticScorer

# Just scoring
scorer = ContextAwareScorer()
score = scorer.compute_context_aware_score(detections, "park")

# Just spatial
analyzer = SpatialHeatmapAnalyzer(grid_size=8)
analyzer.create_grid(height, width)
analyzer.populate_grid(detections)
hotspots = analyzer.identify_hotspots()

# Just weighting
semantic = WeightedSemanticScorer()
report = semantic.generate_score_report(detections, width, height)
```

### Example 3: Custom Weights

```python
from utils.weighted_semantic_scorer import WeightedSemanticScorer

# Default weights
scorer = WeightedSemanticScorer()

# Custom weights (higher priority for hazardous items)
custom = {
    "plastic": 1.0,
    "glass": 1.0,  # Increase glass priority
    "metal": 0.95,
    "paper": 0.5,  # Decrease paper priority
    "cigarette": 0.1
}

scorer = WeightedSemanticScorer(custom)
report = scorer.generate_score_report(detections, width, height)
```

### Example 4: Custom Scene Baselines

```python
from utils.context_aware_scorer import ContextAwareScorer

# City-specific baselines
custom_baselines = {
    "park": 4,      # Very clean parks (Europe)
    "street": 30,   # Busy downtown (Hong Kong)
    "road": 25
}

scorer = ContextAwareScorer(custom_baselines)
score = scorer.compute_context_aware_score(detections, "park")
```

---

## Interpreting Results

### Output: Image Processing

```
# Example excerpt (from the practical run in this workspace):
# output/real_demo/000083_context_pipeline.json
{
  "image_path": "data/processed/taco_yolo_real/images/test/000083.jpg",
  "image_size": [2448, 3264],
  "scene_class": "indoor",
  "scene_confidence": 0.279,
  "litter_count": 1,
  "confident_litter_count": 1,
  "context_aware_score": 4.17,
  "cleanliness_level": "Good",
  "recommendation": "Continue regular maintenance",
  "weighted_semantic_score": 0.07,
  "grid_statistics": {
    "total_cells": 64,
    "cells_with_detections": 6,
    "detections_per_cell_avg": 0.09375,
    "spatial_concentration": 0.90625
  },
  "top_problematic_items": [
    {"class": "other", "count": 1, "weight": 0.6}
  ]
}
```

### Score Interpretation Table

| Score | Level | What It Means | Action |
|-------|-------|-------------|--------|
| 4.5-5.0 | Excellent | Practically no litter | Monitor periodically |
| 3.5-4.5 | Good | Minor litter present | Routine maintenance |
| 2.5-3.5 | Average | Normal for location | Schedule cleanup |
| 1.5-2.5 | Poor | Significant problem | Priority cleanup |
| 0.0-1.5 | Critical | Unacceptable litter | Emergency action |

---

## Advanced: Fine-Tuning Components

### Fine-tune Scene Classifier

```python
from models.scene_classifier import SceneClassifier, TrainingUtils
import torch

# Create dataset
train_loader = TrainingUtils.create_train_dataloader(
    image_dir="scene_images",
    annotations_file="scene_labels.csv",
    batch_size=32
)

val_loader = TrainingUtils.create_train_dataloader(
    image_dir="scene_images",
    annotations_file="scene_labels_val.csv",
    batch_size=32,
    shuffle=False
)

# Initialize model
model = SceneClassifier(num_classes=4, pretrained=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Train
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(50):
    loss = TrainingUtils.train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = TrainingUtils.evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Val Acc={val_acc:.4f}")

# Save
model.save("models/scene_classifier_custom.pth")
```

### Fine-tune YOLOv8

```python
from ultralytics import YOLO

# Create dataset.yaml with your data
# dataset.yaml should contain:
# path: /path/to/data
# train: images/train
# val: images/val
# nc: 60  # number of classes
# names: [class_names...]

# Train
model = YOLO('yolov8n.pt')
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    patience=20,  # early stopping
    save=True,
    device=0
)

# Evaluate
metrics = model.val()

# Export
model.export(format='onnx')  # for deployment
```

---

## Deployment Scenarios

### Scenario 1: Static Camera on Pole

```python
# Continuously monitor one location
import cv2
import threading
from pathlib import Path

system = StreetCleanlinessDetectionSystem(device="cuda")

def monitor_stream():
    while True:
        frame = camera.read()  # get frame from camera
        detections = system.detect_litter(frame)
        scene, _, _ = system.classify_scene(frame)
        score = system.context_scorer.compute_context_aware_score(detections, scene)
        
        # Log if severe
        if score < 2.0:
            Path("alerts.log").write_text(f"ALERT: Score {score} at {datetime.now()}")

thread = threading.Thread(target=monitor_stream, daemon=True)
thread.start()
```

### Scenario 2: Street View Crawler

```python
# Process thousands of street view images
import glob
from pathlib import Path
import json

system = StreetCleanlinessDetectionSystem(device="cuda")

results = []
for image_path in glob.glob("street_images/*.jpg"):
    result = system.process_image(image_path)
    results.append(result)
    
    # Save periodic checkpoints
    if len(results) % 100 == 0:
        with open("progress.json", "w") as f:
            json.dump(results, f)

# Analyze across city
avg_score = sum(r['context_aware_score'] for r in results) / len(results)
dirtiest = sorted(results, key=lambda x: x['context_aware_score'])[:10]
```

### Scenario 3: Autonomous Cleaning Vehicle

```python
# Real-time processing for cleanup route planning
from inference.detection_pipeline import StreetCleanlinessDetectionSystem

system = StreetCleanlinessDetectionSystem(device="cuda")

# Video from vehicle camera
video_stats = system.process_video(
    "vehicle_camera.mp4",
    output_path="annotated_video.mp4",
    skip_frames=2
)

# Use hotspots to guide cleaning path
# Dirtier areas → more frequent cleaning
```

---

## Performance Optimization

### Notes
- Runtime depends on hardware, image size, and model weights.
- For videos, use `--skip-frames` to reduce compute.
- If available, set `--device cuda` to use a GPU.

### Batch Processing
```python
import glob
from tqdm import tqdm

system = StreetCleanlinessDetectionSystem(device="cuda")

image_files = glob.glob("images/*.jpg")
results = []

for image_path in tqdm(image_files):
    result = system.process_image(image_path)
    results.append(result)
```

---

## Testing & Validation

### Unit Tests Example

```python
import unittest
from utils.context_aware_scorer import ContextAwareScorer, Detection

class TestContextScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = ContextAwareScorer()
    
    def test_clean_image(self):
        """Test perfect cleanliness score"""
        detections = []
        score = self.scorer.compute_context_aware_score(detections, "park")
        self.assertEqual(score, 5.0)
    
    def test_scene_normalization(self):
        """Test that context affects score"""
        # 8 detections
        detections = [Detection(...) for _ in range(8)]
        
        park_score = self.scorer.compute_context_aware_score(detections, "park")
        street_score = self.scorer.compute_context_aware_score(detections, "street")
        
        # Street should have higher (cleaner) score
        self.assertGreater(street_score, park_score)
```

---

## Troubleshooting Guide

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"

**Solution**:
```bash
pip install ultralytics>=8.0.0
```

### Issue: YOLO model very slow (>500ms per image)

**Solution**: Check if running on CPU
```python
import torch
print(torch.cuda.is_available())  # Should be True for GPU

# Force GPU
system = StreetCleanlinessDetectionSystem(device="cuda")
```

### Issue: Memory error with videos

**Solution**: Increase frame skip
```bash
python main.py video --source video.mp4 --skip-frames 10  # Process every 10th frame
```

### Issue: Low detection quality

**Solutions**:
1. Fine-tune YOLOv8 on your specific litter types
2. Adjust confidence threshold in config.py
3. Ensure good lighting in input images
4. Check image resolution (minimum 640×480)

---

## Citation & Attribution

If you use this system in research, please cite:

```bibtex
@software{street_cleanliness_2026,
    title = {Context-Aware and Spatially-Intelligent Street Cleanliness Detection System},
    author = {Kritnandan},
    year = {2026},
    institution = {IIT Kanpur},
    course = {StreetClean AI Architecture Blueprint}
}
```

---

**End of Implementation Guide**
