"""
Configuration file for Street Cleanliness Detection System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
DOCS_DIR = PROJECT_ROOT / "docs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# YOLOv8 Model Configuration
YOLO_MODEL_NAME = "yolov8n.pt"  # nano model for efficiency
YOLO_IMG_SIZE = 640
YOLO_CONFIDENCE_THRESHOLD = 0.4
YOLO_IOU_THRESHOLD = 0.5

# Scene Classification Configuration
SCENE_ENCODER = "mobilenet_v2"  # lightweight encoder
SCENE_CLASSES = {
    "road": 0,
    "park": 1,
    "street": 2,
    "indoor": 3
}

# Cleanliness Score Configuration
# Expected litter per scene type (baseline for normalization)
SCENE_LITTER_BASELINE = {
    "road": 15,        # medium tolerance
    "park": 8,         # low tolerance
    "street": 20,      # medium-high tolerance
    "indoor": 3        # very low tolerance
}

# Litter class semantic weights (importance-based)
SEMANTIC_WEIGHTS = {
    "plastic": 1.0,
    "metal": 0.9,
    "paper": 0.7,
    "organic": 0.4,
    "glass": 0.95,
    "foam": 0.85,
    "rubber": 0.8,
    "wood": 0.5,
    "textile": 0.6,
    "cigarette": 0.3,
    # Default weight for unknown classes
    "default": 0.6
}

# Spatial Grid Configuration
HEATMAP_GRID_SIZE = 8  # 8x8 grid for spatial analysis
HEATMAP_BLUR_KERNEL = (31, 31)
HEATMAP_COLORMAP = "JET"  # OpenCV colormap

# Visualization Configuration
VISUALIZATION_CONFIDENCE_THRESHOLD = 0.5
VISUALIZATION_BBOX_COLOR = (0, 255, 0)  # BGR format
VISUALIZATION_TEXT_COLOR = (255, 255, 255)
VISUALIZATION_BBOX_THICKNESS = 2
VISUALIZATION_FONT_SCALE = 0.6

# Scoring Configuration
CLEANLINESS_SCORE_MAX = 5.0  # 5-point scale
MIN_DETECTIONS_FOR_SCORING = 0  # Include 0 detections as clean

# Video Processing
VIDEO_SAVE_FPS = 30
VIDEO_CODEC = "mp4v"

# Dataset Configuration (for future use)
TACO_DATASET_URL = "https://github.com/pedropro/TACO"
COCO_FORMAT_ANNOTATION_FILE = "instances_train2017.json"

# Paths for models
YOLO_WEIGHTS_PATH = MODELS_DIR / YOLO_MODEL_NAME
SCENE_CLASSIFIER_PATH = MODELS_DIR / "scene_classifier.pth"

# Supported input formats
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
