"""
Novelty Component 3: Weighted Semantic Scoring
Assigns importance-based weights to different litter classes
"""

from typing import Dict, List, Tuple
import numpy as np


class WeightedSemanticScorer:
    """Compute importance-weighted litter scores

    Key Innovation:
    Different litter types have different environmental impact:
    - Plastic: High persistence, toxic (weight=1.0)
    - Metal: Sharp, hazardous (weight=0.9)
    - Paper: Quickly degrades (weight=0.7)
    - Organic: Natural decomposition (weight=0.4)

    Weighted score = Σ(class_weight × bbox_area × confidence)
    """

    # Default semantic importance weights
    # Based on environmental impact, persistence, and hazard level
    DEFAULT_WEIGHTS = {
        "plastic": 1.0,      # Most persistent, toxic to environment
        "plastic bag": 1.0,
        "plastic bottle": 1.0,
        "plastic cup": 0.95,
        "styrofoam": 0.9,    # Not biodegradable
        "metal": 0.9,        # Sharp, hazardous
        "metal can": 0.85,
        "metal container": 0.9,
        "glass": 0.95,       # Sharp, hazardous
        "glass bottle": 0.95,
        "foam": 0.85,        # Not biodegradable
        "rubber": 0.8,       # Persistent but less toxic
        "tire": 0.8,
        "paper": 0.7,        # Biodegradable but messy
        "cardboard": 0.65,
        "newspaper": 0.6,
        "wood": 0.5,         # Natural material, biodegradable
        "organic": 0.4,      # Decomposes naturally
        "food": 0.35,
        "leaf": 0.2,         # Natural litter
        "textile": 0.6,      # Synthetic or natural
        "cloth": 0.6,
        "cigarette": 0.3,    # Small but numerous
        "cigarette butt": 0.25,
        "straw": 0.3,        # Small plastic item
        "bottle cap": 0.4,
        "unknown": 0.6       # Default for unclassified
    }

    WEIGHT_CATEGORIES = {
        "critical": (0.85, 1.0),      # Plastic, glass, metal
        "high": (0.6, 0.85),          # Foam, rubber, paper
        "medium": (0.3, 0.6),         # Textiles, small items
        "low": (0.0, 0.3)             # Natural, small organic items
    }

    def __init__(self, custom_weights: Dict[str, float] = None):
        """Initialize scorer with optional custom weights

        Args:
            custom_weights: Dictionary mapping class names to weights [0, 1]
        """
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if custom_weights:
            self.weights.update(custom_weights)

    def get_weight(self, class_name: str) -> float:
        """Get weight for a class (case-insensitive)

        Args:
            class_name: Name of litter class

        Returns:
            Weight value [0, 1]
        """
        class_key = class_name.lower().strip()

        # Try exact match first
        if class_key in self.weights:
            return self.weights[class_key]

        # Try substring matching for common variations
        for key, weight in self.weights.items():
            if key in class_key or class_key in key:
                return weight

        # Default weight
        return self.weights.get("unknown", 0.6)

    def compute_weighted_score(
        self,
        detections: List,
        image_width: int,
        image_height: int,
        confidence_threshold: float = 0.15
    ) -> float:
        """Compute overall weighted litter score

        Algorithm:
        weighted_score = Σ(weight(class) × normalized_area × confidence)

        Where:
        - weight(class): Importance of litter type [0, 1]
        - normalized_area: Bbox area / image area [0, 1]
        - confidence: Detection confidence [0, 1]

        Args:
            detections: List of Detection objects
            image_width: Image width in pixels
            image_height: Image height in pixels
            confidence_threshold: Minimum confidence to include

        Returns:
            Weighted score [0, max_possible_score]
        """
        image_area = image_width * image_height
        total_weighted_score = 0.0

        for detection in detections:
            # Skip low-confidence detections
            if detection.confidence < confidence_threshold:
                continue

            # Get weight for class
            class_weight = self.get_weight(detection.class_name)

            # Compute normalized bounding box area
            x_min, y_min, x_max, y_max = detection.bbox
            bbox_area = (x_max - x_min) * (y_max - y_min)
            normalized_area = bbox_area / image_area if image_area > 0 else 0

            # Compute weighted contribution
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
        """Compute weighted score breakdown by class

        Returns:
            Dict: {class_name: {count, weight, total_score, avg_area, avg_confidence}}
        """
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

            # Update stats
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

        # Compute averages
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
        """Compute importance-normalized score (0-5 scale)

        Scales weighted score to human-readable 0-5 range

        Args:
            detections: List of Detection objects
            image_width: Image width
            image_height: Image height
            confidence_threshold: Minimum confidence threshold

        Returns:
            Score on 0-5 scale where higher score = more problematic
        """
        weighted_score = self.compute_weighted_score(
            detections, image_width, image_height, confidence_threshold
        )

        # Map normalized weighted score to 0-5 scale.
        # The old formula (weighted_score * 5) assumed litter fills ~100% of
        # the image, which makes real-world small items round to 0.0.
        #
        # New approach: each detection contributes up to (class_weight * confidence)
        # points on a per-item basis, area is used as a mild amplifier rather than
        # a primary factor. Reference: 1 high-impact item confidently detected →
        # score ≈ 1.5–2.0; 5 items → approaches 5.0.
        #
        # Formula: Σ(class_weight × confidence × (1 + 10 × normalized_area))
        # This gives ~0.6–1.0 per small item and caps naturally at 5.
        image_area = image_width * image_height
        per_item_score = 0.0
        for d in detections:
            if d.confidence < confidence_threshold:
                continue
            w = self.get_weight(d.class_name)
            x_min, y_min, x_max, y_max = d.bbox
            bbox_area = (x_max - x_min) * (y_max - y_min)
            norm_area = bbox_area / image_area if image_area > 0 else 0
            # area amplifier: small item → ×1.0 baseline, large item → up to ×11
            area_amp = 1.0 + 10.0 * norm_area
            per_item_score += w * d.confidence * area_amp

        scaled_score = min(per_item_score, 5.0)
        return scaled_score

    def get_weight_category(self, class_name: str) -> str:
        """Get importance category for a class"""
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
        """Generate comprehensive weighted scoring report

        Returns:
            Dictionary with detailed scoring breakdown
        """
        total_weighted = self.compute_weighted_score(
            detections, image_width, image_height, confidence_threshold
        )
        importance_score = self.compute_importance_score(
            detections, image_width, image_height, confidence_threshold
        )
        per_class = self.compute_per_class_score(
            detections, image_width, image_height, confidence_threshold
        )

        # Rank classes by weighted score
        ranked_classes = sorted(
            per_class.items(),
            key=lambda x: x[1]["total_weighted_score"],
            reverse=True
        )

        # Top problematic classes
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
        """Update semantic weights (e.g., for fine-tuning)

        Args:
            new_weights: Dictionary of class_name -> weight pairs
        """
        self.weights.update(new_weights)

    def get_all_weights(self) -> Dict[str, float]:
        """Get all current weights"""
        return self.weights.copy()
