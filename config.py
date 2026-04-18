import os
from pathlib import Path

# project root and dirs
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
DOCS_DIR = PROJECT_ROOT / "docs"

# create dirs if missing
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# yolo config
YOLO_MODEL_NAME = "yolov8n.pt"  # nano, fast on cpu
YOLO_IMG_SIZE = 640
YOLO_CONFIDENCE_THRESHOLD = 0.4
YOLO_IOU_THRESHOLD = 0.5

# todo: mudit fix SCENE_CLASSES to support more scene types
SCENE_ENCODER = "mobilenet_v2"
SCENE_CLASSES = {
    "road": 0,
    "park": 1,
    "street": 2,
    "indoor": 3
}

# litter baseline per scene
SCENE_LITTER_BASELINE = {
    "road": 15,
    "park": 8,       # strict
    "street": 20,    # lenient
    "indoor": 3      # very strict
}

# ecological impact weights per class
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
    "default": 0.6   # fallback for unknown class
}

# heatmap grid config
HEATMAP_GRID_SIZE = 8  # 8x8 cells
HEATMAP_BLUR_KERNEL = (31, 31)
HEATMAP_COLORMAP = "JET"

# viz config
VISUALIZATION_CONFIDENCE_THRESHOLD = 0.5
VISUALIZATION_BBOX_COLOR = (0, 255, 0)  # bgr
VISUALIZATION_TEXT_COLOR = (255, 255, 255)
VISUALIZATION_BBOX_THICKNESS = 2
VISUALIZATION_FONT_SCALE = 0.6

# score config
CLEANLINESS_SCORE_MAX = 5.0
MIN_DETECTIONS_FOR_SCORING = 0

# video config
VIDEO_SAVE_FPS = 30
VIDEO_CODEC = "mp4v"

# dataset urls
TACO_DATASET_URL = "https://github.com/pedropro/TACO"
COCO_FORMAT_ANNOTATION_FILE = "instances_train2017.json"

# model paths
YOLO_WEIGHTS_PATH = MODELS_DIR / YOLO_MODEL_NAME
SCENE_CLASSIFIER_PATH = MODELS_DIR / "scene_classifier.pth"

# supported formats
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
