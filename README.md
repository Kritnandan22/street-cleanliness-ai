
# StreetClean AI: Context-Aware Urban Cleanliness System

![StreetClean AI](https://img.shields.io/badge/Status-Active-success)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)
![Flask](https://img.shields.io/badge/Framework-Flask-black)

### 🚀 **[Try the Live Web App Here!](https://huggingface.co/spaces/Nandan22/StreatClean_AI)**

## Overview
Traditional street cleanliness systems treat litter quantification purely as a simple object detection problem (counting bounding boxes). This naive approach fails to account for critical environmental nuances. **StreetClean AI** is an intelligent, open-source pipeline that shifts the paradigm from "how much litter is there?" to "how severe is this pollution contextually and ecologically?".

## Core Innovations
Our pipeline extends YOLOv8 by introducing three intelligent modules:

1. **Scene-Dependent Context Normalisation**
   Fusing ImageNet-derived `MobileNetV2` classification, the system modifies baseline cleanliness expectations based on the environment. Finding 3 bottles in an indoor setting triggers a heavy penalty, whereas 3 bottles in a commercial alleyway are mapped closer to the architectural baseline.

2. **Ecological-Weighted Semantic Scoring**
   The neural network categorizes detections into 6 severe groups: `plastic`, `metal`, `paper`, `organic`, `glass`, and `other`. Highly toxic, non-biodegradable items (like glass and plastic) degrade the overall score exponentially faster than paper wrappers.

## Primary Use Cases
- **Municipal Sanitation Orchestration:** Automatically route street sweepers and cleaning crews to explicitly identified spatial Hotspots rather than deploying blind grid sweeps.
- **Ecological Impact Tracking:** Analyze the ratio of highly-toxic non-biodegradables (plastics/glass) vs minor perishables (paper/organic) across different city sectors over time.
- **Automated Fleet Integration:** Deploy the headless inference engine onto dashcams of municipal vehicles (e.g., mail trucks, police cars) to passively map urban pollution severities constantly.

## Repository Directory Structure
This repository follows an organized modular paradigm:
```text
street-cleanliness-ai/
│
├── app.py                      ← Main Flask routing and Web Dashboard entrypoint
├── train.py                    ← YOLOv8 native model fine-tuning script
├── evaluate.py                 ← Generates ablation studies on metrics (mAP, counts, contexts)
│
├── docs/
│   └── SYSTEM_ARCHITECTURE.md  ← Deep-dive mathematical formulas for the cleanliness score
│
├── inference/
│   └── detection_pipeline.py   ← Core module loading YOLOv8 and mapping object bounds
│
├── models/
│   ├── best_yolov8_taco.pt     ← Native production detection weights
│   └── scene_classifier.py     ← MobileNetV2 scene-context (Grass vs Road) classifier
│
├── utils/
│   ├── context_aware_scorer.py ← Modifies baseline penalties based on detected scenery
│   ├── spatial_heatmap.py      ← Generates Gaussian thermal matrices to isolate Hotspots
│   └── weighted_semantic.py    ← Allocates ecological penalties (Plastic vs Paper)
│
├── visualization/
│   └── visualizer.py           ← Fuses YOLO bounds and Spatial heatmaps into a single image
│
└── templates/ & static/        ← HTML, CSS, and JS defining the frontend Quad-Split Interface
```

---

## The Dashboard Interface

The repository includes a highly-optimized, dynamic Flask Web Dashboard designed natively for production deployments. 
- Features a **Quad-Split Image Array** rendering the original image, bounding boxes, spatial heatmap, and the fully merged prediction.
- Real-time dynamic instructions: Generates customized recommendations (e.g. *Hazard Warning: Glass Detected*) automatically based on neural analysis.

## Setup & Deployment

### Local Installation
```bash
git clone https://github.com/YourUsername/street-cleanliness-ai.git
cd street-cleanliness-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Dashboard
```bash
python app.py
```
Navigate to `http://localhost:5001` to access the drag-and-drop inference engine.

### Cloud Deployment (Render.com)
This system is natively pre-configured for automated cloud deployment via **Render**. The `requirements.txt` forces PyTorch into CPU mode and strips OpenCV's GUI frameworks to fit within the constraints of cloud micro-instances securely.

## System Architecture Details
For deep mathematical breakdowns on exactly how the ecological weighting and context-normalisation formulas calculate the final cleanliness metric (0-5 scale), refer to the [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md) blueprint.

## License
MIT License. Open-source tracking module designed for urban monitoring and sanitation optimization.
