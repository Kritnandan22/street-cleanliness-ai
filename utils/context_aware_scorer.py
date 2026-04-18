from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    area: float = 0.0

    def compute_area(self, image_width: int, image_height: int):
        x_min, y_min, x_max, y_max = self.bbox
        bbox_area = (x_max - x_min) * (y_max - y_min)
        image_area = image_width * image_height
        self.area = bbox_area / image_area if image_area > 0 else 0


class ContextAwareScorer:

    # how much litter is normal for each scene type
    SCENE_BASELINES = {
        "road": 15,        # road have more litter usually
        "park": 8,         # park is maintained so less litter ok
        "street": 20,      # streets downtown have lot of litter
        "indoor": 3        # inside should be clean
    }

    # min and max for score range
    MIN_SCORE = 0.0
    MAX_SCORE = 5.0  # we use 5 point scale

    def __init__(self, custom_baselines: Dict[str, int] = None):
        if custom_baselines:
            self.SCENE_BASELINES.update(custom_baselines)

    def compute_raw_score(self, detections: List[Detection]) -> int:
        # just count how many things detected
        return len(detections)

    def compute_confidence_weighted_score(
        self,
        detections: List[Detection],
        confidence_threshold: float = 0.4
    ) -> float:
        # sum confidence of all detection above threshold
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
        # only count detection we are confident about
        confident_detections = [
            d for d in detections
            if d.confidence >= confidence_threshold
        ]
        detection_count = len(confident_detections)

        # get how much litter is normal for this scene
        baseline = self.SCENE_BASELINES.get(scene_class, 10)

        # how much more or less litter than normal
        normalization_ratio = detection_count / baseline if baseline > 0 else 0

        # cap at 2x baseline, after that its just very dirty
        normalized_ratio = min(normalization_ratio, 2.0)

        # flip it so more litter = lower score
        inverse_score = 1.0 - (normalized_ratio / 2.0)

        # scale to 0-5
        cleanliness_score = inverse_score * self.MAX_SCORE

        # make sure its in valid range
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
        # compute bbox area for all detections
        for detection in detections:
            detection.compute_area(image_width, image_height)

        # sum all area
        total_area = sum(d.area for d in detections)

        baseline = self.SCENE_BASELINES.get(scene_class, 10)

        # normalize by baseline area threshold
        baseline_area_percent = baseline / (image_width * image_height / 10000)
        normalized_area = total_area / \
            baseline_area_percent if baseline_area_percent > 0 else 0

        normalized_ratio = min(normalized_area, 2.0)
        inverse_score = 1.0 - (normalized_ratio / 2.0)
        area_weighted_score = inverse_score * self.MAX_SCORE

        return np.clip(area_weighted_score, self.MIN_SCORE, self.MAX_SCORE)

    def get_scene_analysis(self, scene_class: str) -> Dict:
        return {
            "scene_class": scene_class,
            "baseline_litter_count": self.SCENE_BASELINES.get(scene_class, 10),
            "description": self._get_scene_description(scene_class)
        }

    @staticmethod
    def _get_scene_description(scene_class: str) -> str:
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

    SCORE_LEVELS = {
        (4.5, 5.0): ("Excellent", "Very clean, minimal litter detected"),
        (3.5, 4.5): ("Good", "Clean with minor litter"),
        (2.5, 3.5): ("Average", "Normal for this location, some cleanup needed"),
        (1.5, 2.5): ("Poor", "Significant litter accumulation"),
        (0.0, 1.5): ("Critical", "Severe cleanliness issues, immediate cleanup needed")
    }

    @classmethod
    def interpret_score(cls, score: float) -> Tuple[str, str]:
        for (min_score, max_score), (level, desc) in cls.SCORE_LEVELS.items():
            if min_score <= score <= max_score:
                return level, desc

        return "Unknown", "Score out of range"

    @classmethod
    def get_recommendation(cls, score: float, scene_class: str = "street") -> str:
        level, _ = cls.interpret_score(score)

        recommendations = {
            "Excellent": "Maintain current practices",
            "Good": "Continue regular maintenance",
            "Average": "Schedule routine cleanup",
            "Poor": "Schedule immediate cleanup",
            "Critical": "Emergency cleanup required"
        }

        return recommendations.get(level, "No recommendation available")
