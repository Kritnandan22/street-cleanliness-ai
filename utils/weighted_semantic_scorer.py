from typing import Dict, List, Tuple
import numpy as np


class WeightedSemanticScorer:

    # weights based on how bad each type of litter is for environment
    DEFAULT_WEIGHTS = {
        "plastic": 1.0,      # plastic is worst, takes 450 years to decompose
        "plastic bag": 1.0,
        "plastic bottle": 1.0,
        "plastic cup": 0.95,
        "styrofoam": 0.9,    # doesnt degrade at all
        "metal": 0.9,        # sharp edges, hazardous
        "metal can": 0.85,
        "metal container": 0.9,
        "glass": 0.95,       # sharp, dangerous for people and animals
        "glass bottle": 0.95,
        "foam": 0.85,        # not biodegradable
        "rubber": 0.8,       # persist long time but less toxic
        "tire": 0.8,
        "paper": 0.7,        # biodegradable but still mess
        "cardboard": 0.65,
        "newspaper": 0.6,
        "wood": 0.5,         # natural, degrades ok
        "organic": 0.4,      # decomposes fast so less bad
        "food": 0.35,
        "leaf": 0.2,         # its just a leaf, no problem
        "textile": 0.6,      # synthetic or natural mix
        "cloth": 0.6,
        "cigarette": 0.3,    # small but toxic chemicals inside
        "cigarette butt": 0.25,
        "straw": 0.3,        # small plastic item
        "bottle cap": 0.4,
        "unknown": 0.6       # default when we dont know class
    }

    WEIGHT_CATEGORIES = {
        "critical": (0.85, 1.0),      # plastic, glass, metal
        "high": (0.6, 0.85),          # foam, rubber, paper
        "medium": (0.3, 0.6),         # textiles, small items
        "low": (0.0, 0.3)             # natural stuff, small organic
    }

    def __init__(self, custom_weights: Dict[str, float] = None):
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if custom_weights:
            self.weights.update(custom_weights)

    def get_weight(self, class_name: str) -> float:
        class_key = class_name.lower().strip()

        # try exact match first
        if class_key in self.weights:
            return self.weights[class_key]

        # try partial match for variations like "plastic_bottle"
        for key, weight in self.weights.items():
            if key in class_key or class_key in key:
                return weight

        # fallback to unknown weight
        return self.weights.get("unknown", 0.6)

    def compute_weighted_score(
        self,
        detections: List,
        image_width: int,
        image_height: int,
        confidence_threshold: float = 0.15
    ) -> float:
        image_area = image_width * image_height
        total_weighted_score = 0.0

        for detection in detections:
            # skip low confidence detections
            if detection.confidence < confidence_threshold:
                continue

            class_weight = self.get_weight(detection.class_name)

            # compute what fraction of image the bbox covers
            x_min, y_min, x_max, y_max = detection.bbox
            bbox_area = (x_max - x_min) * (y_max - y_min)
            normalized_area = bbox_area / image_area if image_area > 0 else 0

            # multiply weight by area and confidence
            contribution = class_weight * normalized_area * detection.confidence
            total_weighted_score += contribution

        return total_weighted_score

    def compute_per_class_score(
        self,
        detections: List,
        image_width: int,
        image_height: int,
        confidence_threshold: float = 0.15
    ) -> Dict[str, Dict]:
        image_area = image_width * image_height
        class_scores = {}

        for detection in detections:
            if detection.confidence < confidence_threshold:
                continue

            class_name = detection.class_name
            if class_name not in class_scores:
                class_scores[class_name] = {
                    "count": 0,
                    "weight": self.get_weight(class_name),
                    "total_area": 0.0,
                    "total_confidence": 0.0,
                    "total_weighted_score": 0.0
                }

            class_stats = class_scores[class_name]
            class_stats["count"] += 1

            x_min, y_min, x_max, y_max = detection.bbox
            bbox_area = (x_max - x_min) * (y_max - y_min)
            normalized_area = bbox_area / image_area if image_area > 0 else 0

            class_stats["total_area"] += normalized_area
            class_stats["total_confidence"] += detection.confidence

            contribution = class_stats["weight"] * \
                normalized_area * detection.confidence
            class_stats["total_weighted_score"] += contribution

        # compute averages for each class
        for class_name, stats in class_scores.items():
            if stats["count"] > 0:
                stats["avg_area"] = stats["total_area"] / stats["count"]
                stats["avg_confidence"] = stats["total_confidence"] / \
                    stats["count"]
            else:
                stats["avg_area"] = 0.0
                stats["avg_confidence"] = 0.0

        return class_scores

    def compute_importance_score(
        self,
        detections: List,
        image_width: int,
        image_height: int,
        confidence_threshold: float = 0.15
    ) -> float:
        # call the main weighted score first
        weighted_score = self.compute_weighted_score(
            detections, image_width, image_height, confidence_threshold
        )

        # old formula was weighted_score * 5 but that gives 0 for small items
        # new way: per item score with area amplifier so small litter still register
        image_area = image_width * image_height
        per_item_score = 0.0
        for d in detections:
            if d.confidence < confidence_threshold:
                continue
            w = self.get_weight(d.class_name)
            x_min, y_min, x_max, y_max = d.bbox
            bbox_area = (x_max - x_min) * (y_max - y_min)
            norm_area = bbox_area / image_area if image_area > 0 else 0
            # area amplifier: bigger item get more penalty, small item atleast 1x
            area_amp = 1.0 + 10.0 * norm_area
            per_item_score += w * d.confidence * area_amp

        scaled_score = min(per_item_score, 5.0)
        return scaled_score

    def get_weight_category(self, class_name: str) -> str:
        weight = self.get_weight(class_name)

        for category, (min_w, max_w) in self.WEIGHT_CATEGORIES.items():
            if min_w <= weight <= max_w:
                return category

        return "unknown"

    def generate_score_report(
        self,
        detections: List,
        image_width: int,
        image_height: int,
        scene_class: str = "street",
        confidence_threshold: float = 0.15
    ) -> Dict:
        total_weighted = self.compute_weighted_score(
            detections, image_width, image_height, confidence_threshold
        )
        importance_score = self.compute_importance_score(
            detections, image_width, image_height, confidence_threshold
        )
        per_class = self.compute_per_class_score(
            detections, image_width, image_height, confidence_threshold
        )

        # sort classes by score, worst first
        ranked_classes = sorted(
            per_class.items(),
            key=lambda x: x[1]["total_weighted_score"],
            reverse=True
        )

        # top 5 worst classes
        top_classes = [
            {
                "class": name,
                "count": stats["count"],
                "weight": round(stats["weight"], 2),
                "category": self.get_weight_category(name),
                "total_score": round(stats["total_weighted_score"], 4),
                "avg_area": round(stats["avg_area"], 4),
                "avg_confidence": round(stats["avg_confidence"], 2)
            }
            for name, stats in ranked_classes[:5]
        ]

        return {
            "total_detections": len(detections),
            "confident_detections": sum(
                1 for d in detections if d.confidence >= confidence_threshold
            ),
            "total_weighted_score": round(total_weighted, 4),
            "importance_score_0_5": round(importance_score, 2),
            "scene_class": scene_class,
            "top_problematic_items": top_classes,
            "per_class_breakdown": {
                name: {
                    "count": stats["count"],
                    "weight": round(stats["weight"], 2),
                    "category": self.get_weight_category(name),
                    "total_score": round(stats["total_weighted_score"], 4)
                }
                for name, stats in per_class.items()
            }
        }

    def update_weights(self, new_weights: Dict[str, float]):
        self.weights.update(new_weights)

    def get_all_weights(self) -> Dict[str, float]:
        return self.weights.copy()
