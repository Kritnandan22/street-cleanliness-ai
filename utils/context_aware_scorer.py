"""
Novelty Component 1: Context-Aware Cleanliness Scoring
Computes scene-dependent cleanliness scores with normalization
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    """Bounding box detection container"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    area: float = 0.0

    def compute_area(self, image_width: int, image_height: int):
        """Compute bounding box area as percentage of image"""
        x_min, y_min, x_max, y_max = self.bbox
        bbox_area = (x_max - x_min) * (y_max - y_min)
        image_area = image_width * image_height
        self.area = bbox_area / image_area if image_area > 0 else 0


class ContextAwareScorer:
    """Compute context-aware cleanliness scores

    Key Innovation:
    Normalizes raw detection count by expected baseline for each scene type.
    Score accounts for what is "normal" litter in different environments.
    """

    # Scene-specific litter baselines (expected litter detection count)
    SCENE_BASELINES = {
        "road": 15,        # roads accumulate more litter
        "park": 8,         # parks are maintained, low tolerance
        "street": 20,      # downtown streets have high litter
        "indoor": 3        # indoors should be clean
    }

    # Normalization bounds
    MIN_SCORE = 0.0
    MAX_SCORE = 5.0  # 5-point scale

    def __init__(self, custom_baselines: Dict[str, int] = None):
        """Initialize scorer with optional custom baselines"""
        if custom_baselines:
            self.SCENE_BASELINES.update(custom_baselines)

    def compute_raw_score(self, detections: List[Detection]) -> int:
        """Get raw litter detection count"""
        return len(detections)

    def compute_confidence_weighted_score(
        self,
        detections: List[Detection],
        confidence_threshold: float = 0.4
    ) -> float:
        """Weight detections by confidence scores

        Returns:
            Sum of confidence scores for detections above threshold
        """
        return sum(
            d.confidence for d in detections
            if d.confidence >= confidence_threshold
        )

    def compute_context_aware_score(
        self,
        detections: List[Detection],
        scene_class: str = "street",
        confidence_threshold: float = 0.4
    ) -> float:
        """Compute context-aware cleanliness score

        Algorithm:
        1. Count confident detections
        2. Get baseline for scene type
        3. Normalize: score = 1 - (count / baseline)
        4. Scale to 0-5 cleanliness rating

        Interpretation:
        - Score 5.0: Perfectly clean (0 detections)
        - Score 3.0: Average for scene type (count == baseline)
        - Score 0.0: Extremely dirty (count >= 2x baseline)

        Args:
            detections: List of Detection objects
            scene_class: Scene type (road, park, street, indoor)
            confidence_threshold: Minimum confidence to count detection

        Returns:
            Cleanliness score on 0-5 scale
        """
        # Count detections above confidence threshold
        confident_detections = [
            d for d in detections
            if d.confidence >= confidence_threshold
        ]
        detection_count = len(confident_detections)

        # Get baseline for scene
        baseline = self.SCENE_BASELINES.get(scene_class, 10)

        # Normalize: ratio of actual to expected
        normalization_ratio = detection_count / baseline if baseline > 0 else 0

        # Clamp to [0, 2] range (beyond 2x baseline is equally bad)
        normalized_ratio = min(normalization_ratio, 2.0)

        # Inverse: 1 - ratio (more litter = lower score)
        inverse_score = 1.0 - (normalized_ratio / 2.0)

        # Scale to 0-5 cleanliness rating
        cleanliness_score = inverse_score * self.MAX_SCORE

        # Ensure in valid range
        cleanliness_score = np.clip(
            cleanliness_score,
            self.MIN_SCORE,
            self.MAX_SCORE
        )

        return cleanliness_score

    def compute_area_weighted_score(
        self,
        detections: List[Detection],
        image_width: int,
        image_height: int,
        scene_class: str = "street"
    ) -> float:
        """Compute score weighted by bounding box areas

        Rationale: Large litter areas are more noticeable and problematic
        than small detections in image

        Args:
            detections: List of Detection objects
            image_width: Image width in pixels
            image_height: Image height in pixels
            scene_class: Scene type

        Returns:
            Area-weighted cleanliness score
        """
        # Compute areas for all detections
        for detection in detections:
            detection.compute_area(image_width, image_height)

        # Sum up areas
        total_area = sum(d.area for d in detections)

        # Use area as "effective detection count"
        baseline = self.SCENE_BASELINES.get(scene_class, 10)

        # Normalize by baseline (typical area threshold)
        baseline_area_percent = baseline / (image_width * image_height / 10000)
        normalized_area = total_area / \
            baseline_area_percent if baseline_area_percent > 0 else 0

        # Same scoring as detection count
        normalized_ratio = min(normalized_area, 2.0)
        inverse_score = 1.0 - (normalized_ratio / 2.0)
        area_weighted_score = inverse_score * self.MAX_SCORE

        return np.clip(area_weighted_score, self.MIN_SCORE, self.MAX_SCORE)

    def get_scene_analysis(self, scene_class: str) -> Dict:
        """Get analysis info for a scene type"""
        return {
            "scene_class": scene_class,
            "baseline_litter_count": self.SCENE_BASELINES.get(scene_class, 10),
            "description": self._get_scene_description(scene_class)
        }

    @staticmethod
    def _get_scene_description(scene_class: str) -> str:
        """Get human-readable description of scene"""
        descriptions = {
            "road": "Roads accumulate more litter due to traffic and outdoor exposure",
            "park": "Parks are maintained; low litter tolerance expected",
            "street": "Downtown streets see high foot traffic; higher litter baseline",
            "indoor": "Indoor spaces should be clean; very low litter tolerance"
        }
        return descriptions.get(scene_class, "Unknown scene type")

    def compute_improvement_score(
        self,
        before_detections: List[Detection],
        after_detections: List[Detection],
        scene_class: str = "street"
    ) -> Dict:
        """Compute improvement between two states (useful for temporal analysis)

        Returns:
            Dict with before score, after score, and improvement percentage
        """
        before_score = self.compute_context_aware_score(
            before_detections, scene_class
        )
        after_score = self.compute_context_aware_score(
            after_detections, scene_class
        )

        improvement = (after_score - before_score) / self.MAX_SCORE * 100

        return {
            "before_score": round(before_score, 2),
            "after_score": round(after_score, 2),
            "improvement_percent": round(improvement, 2),
            "status": "improved" if improvement > 0 else ("worsened" if improvement < 0 else "unchanged")
        }


class ScoreInterpretation:
    """Interpret cleanliness scores for human understanding"""

    SCORE_LEVELS = {
        (4.5, 5.0): ("Excellent", "Very clean, minimal litter detected"),
        (3.5, 4.5): ("Good", "Clean with minor litter"),
        (2.5, 3.5): ("Average", "Normal for this location, some cleanup needed"),
        (1.5, 2.5): ("Poor", "Significant litter accumulation"),
        (0.0, 1.5): ("Critical", "Severe cleanliness issues, immediate cleanup needed")
    }

    @classmethod
    def interpret_score(cls, score: float) -> Tuple[str, str]:
        """Interpret a cleanliness score

        Returns:
            (level_name, description)
        """
        for (min_score, max_score), (level, desc) in cls.SCORE_LEVELS.items():
            if min_score <= score <= max_score:
                return level, desc

        return "Unknown", "Score out of range"

    @classmethod
    def get_recommendation(cls, score: float, scene_class: str = "street") -> str:
        """Get action recommendation based on score"""
        level, _ = cls.interpret_score(score)

        recommendations = {
            "Excellent": "Maintain current practices",
            "Good": "Continue regular maintenance",
            "Average": "Schedule routine cleanup",
            "Poor": "Schedule immediate cleanup",
            "Critical": "Emergency cleanup required"
        }

        return recommendations.get(level, "No recommendation available")
