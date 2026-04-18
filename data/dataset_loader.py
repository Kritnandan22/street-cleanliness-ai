import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yaml


@dataclass
class DatasetConfig:
    dataset_path: Path
    format: str  # 'coco', 'yolo', 'pascal_voc'
    image_extensions: List[str] = None

    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']


class COCODatasetLoader:
    # loads coco json annotations (taco uses this format)

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.annotations = None
        self.images = None
        self.categories = None

    def load_annotations(self, annotation_file: str = "instances_default.json"):
        annotation_path = self.dataset_path / annotation_file

        if not annotation_path.exists():
            raise FileNotFoundError(f"not found: {annotation_path}")

        with open(annotation_path, 'r') as f:
            data = json.load(f)

        self.annotations = data.get('annotations', [])
        self.images = {img['id']: img for img in data.get('images', [])}
        self.categories = {cat['id']: cat['name']
                           for cat in data.get('categories', [])}

        return self

    def get_image_annotations(self, image_id: int) -> List[Dict]:
        if self.annotations is None:
            raise RuntimeError("call load_annotations() first")
        return [ann for ann in self.annotations if ann['image_id'] == image_id]

    def get_category_name(self, category_id: int) -> str:
        return self.categories.get(category_id, "unknown")

    def get_all_category_names(self) -> List[str]:
        return list(set(self.categories.values()))


class YOLODatasetLoader:

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.classes = None

    def load_classes(self, classes_file: str = "classes.txt"):
        classes_path = self.dataset_path / classes_file
        if not classes_path.exists():
            raise FileNotFoundError(f"not found: {classes_path}")
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        return self

    # todo: karan fix load_annotation to handle missing label files
    def load_annotation(self, annotation_file: Path) -> List[Dict]:
        # yolo format: class_id x_center y_center width height (normalized)
        annotations = []
        if not annotation_file.exists():
            return annotations
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    annotations.append({
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
        return annotations


class DatasetConverter:

    @staticmethod
    def coco_to_yolo(
        coco_annotation: Dict,
        image_width: int,
        image_height: int
    ) -> Dict:
        # coco: [x_min, y_min, w, h] → yolo: [cx, cy, w, h] normalized
        x_min, y_min, width, height = coco_annotation['bbox']
        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        width_norm = width / image_width
        height_norm = height / image_height

        return {
            'class_id': coco_annotation['category_id'],
            'x_center': x_center,
            'y_center': y_center,
            'width': width_norm,
            'height': height_norm
        }

    @staticmethod
    def yolo_to_pixel_coordinates(
        yolo_annotation: Dict,
        image_width: int,
        image_height: int
    ) -> Tuple[int, int, int, int]:
        # yolo normalized → pixel x1y1x2y2
        x_center = yolo_annotation['x_center'] * image_width
        y_center = yolo_annotation['y_center'] * image_height
        width = yolo_annotation['width'] * image_width
        height = yolo_annotation['height'] * image_height

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        return (x_min, y_min, x_max, y_max)


class DatasetMerger:

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.class_mapping = {}
        self.next_class_id = 0

    def get_or_create_class_id(self, class_name: str) -> int:
        if class_name not in self.class_mapping:
            self.class_mapping[class_name] = self.next_class_id
            self.next_class_id += 1
        return self.class_mapping[class_name]

    def save_class_mapping(self, filename: str = "classes.txt"):
        sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])
        with open(self.output_path / filename, 'w') as f:
            for class_name, _ in sorted_classes:
                f.write(f"{class_name}\n")

    def merge_coco_dataset(
        self,
        coco_loader: COCODatasetLoader,
        images_dir: Path,
        dataset_name: str = "dataset1"
    ):
        images_dir = Path(images_dir)
        output_images = self.output_path / "images" / dataset_name
        output_labels = self.output_path / "labels" / dataset_name
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        for image_id, image_info in coco_loader.images.items():
            image_path = images_dir / image_info['file_name']
            if not image_path.exists():
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                continue

            height, width = img.shape[:2]
            output_image_path = output_images / image_info['file_name']
            cv2.imwrite(str(output_image_path), img)

            annotations = coco_loader.get_image_annotations(image_id)
            output_label_path = output_labels / \
                f"{Path(image_info['file_name']).stem}.txt"

            with open(output_label_path, 'w') as f:
                for ann in annotations:
                    yolo_ann = DatasetConverter.coco_to_yolo(ann, width, height)
                    category_name = coco_loader.get_category_name(yolo_ann['class_id'])
                    unified_class_id = self.get_or_create_class_id(category_name)
                    f.write(
                        f"{unified_class_id} "
                        f"{yolo_ann['x_center']:.6f} "
                        f"{yolo_ann['y_center']:.6f} "
                        f"{yolo_ann['width']:.6f} "
                        f"{yolo_ann['height']:.6f}\n"
                    )

    def create_yaml_config(self, dataset_name: str = "merged_dataset"):
        config = {
            'path': str(self.output_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_mapping),
            'names': {v: k for k, v in self.class_mapping.items()}
        }
        with open(self.output_path / 'dataset.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


class ImageAugmentation:
    # todo: mudit fix resize_image to return correct bbox coords too

    @staticmethod
    def resize_image(image: np.ndarray, target_size: int = 640) -> np.ndarray:
        # letterbox resize with grey padding
        height, width = image.shape[:2]
        scale = target_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height))

        # pad to square
        pad_width = target_size - new_width
        pad_height = target_size - new_height
        pad_top = pad_height // 2
        pad_left = pad_width // 2

        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_height - pad_top,
            pad_left, pad_width - pad_left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        return padded

    @staticmethod
    def horizontal_flip(image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 1)

    @staticmethod
    def brightness_contrast(
        image: np.ndarray,
        brightness: float = 0.1,
        contrast: float = 0.1
    ) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0
        img_float = img_float * (1 + contrast) + brightness
        img_float = np.clip(img_float, 0, 1)
        return (img_float * 255).astype(np.uint8)

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (width, height),
            borderMode=cv2.BORDER_REFLECT
        )
        return rotated
