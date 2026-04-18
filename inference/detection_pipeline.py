import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch

from models.scene_classifier import SceneClassificationPipeline
from utils.context_aware_scorer import Detection, ContextAwareScorer, ScoreInterpretation
from utils.spatial_heatmap import SpatialHeatmapAnalyzer
from utils.weighted_semantic_scorer import WeightedSemanticScorer


class StreetCleanlinessDetectionSystem:

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        scene_classifier_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.device = device

        # load yolo model, download if not found
        try:
            from ultralytics import YOLO
            if yolo_model_path:
                self.yolo_model = YOLO(yolo_model_path)
            else:
                # will auto download if not present
                self.yolo_model = YOLO('yolov8n.pt')
        except ImportError:
            print("WARNING: ultralytics not installed. YOLO detection will not work.")
            self.yolo_model = None

        # init scene classifier
        self.scene_classifier = SceneClassificationPipeline(
            device=device,
            model_path=scene_classifier_path
        )

        # init all scoring modules
        self.context_scorer = ContextAwareScorer()
        self.spatial_analyzer = SpatialHeatmapAnalyzer(grid_size=8)
        self.semantic_scorer = WeightedSemanticScorer()

    def detect_litter(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.15
    ) -> List[Detection]:
        if self.yolo_model is None:
            print("ERROR: YOLO model not initialized")
            return []

        # run yolo inference on image
        results = self.yolo_model(
            image, conf=confidence_threshold, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                # get box coords and class info
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names.get(class_id, f"class_{class_id}")

                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(int(x_min), int(y_min), int(x_max), int(y_max))
                )

                # compute normalized area of detection
                detection.compute_area(image.shape[1], image.shape[0])

                detections.append(detection)

        return detections

    def classify_scene(
        self,
        image: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        # returns scene name, confidence, and all class probabilities
        return self.scene_classifier.predict(image)

    def process_image(
        self,
        image_path: str,
        return_detections: bool = True,
        return_analysis: bool = True
    ) -> Dict:
        # load image from disk
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        height, width = image.shape[:2]

        # step 1: detect litter with yolo
        detections = self.detect_litter(image)

        # step 2: classify what kind of scene it is
        scene_class, scene_confidence, scene_probs = self.classify_scene(image)

        # step 3: context aware score based on scene baseline
        context_score = self.context_scorer.compute_context_aware_score(
            detections, scene_class
        )

        # step 4: spatial grid heatmap analysis
        self.spatial_analyzer.create_grid(height, width)
        self.spatial_analyzer.populate_grid(detections)
        hotspots = self.spatial_analyzer.identify_hotspots(
            threshold_percentile=75.0)
        grid_stats = self.spatial_analyzer.get_grid_statistics()

        # step 5: weighted semantic score based on ecological impact
        weighted_report = self.semantic_scorer.generate_score_report(
            detections, width, height, scene_class
        )

        # put everything in result dict
        result = {
            "image_path": str(image_path),
            "image_size": (width, height),
            "scene_class": scene_class,
            "scene_confidence": round(scene_confidence, 3),
            "scene_probabilities": {k: round(v, 3) for k, v in scene_probs.items()},
            "litter_count": len(detections),
            "confident_litter_count": sum(1 for d in detections if d.confidence >= 0.4),
            "context_aware_score": round(context_score, 2),
            "cleanliness_level": ScoreInterpretation.interpret_score(context_score)[0],
            "recommendation": ScoreInterpretation.get_recommendation(context_score, scene_class),
            "weighted_semantic_score": round(weighted_report["importance_score_0_5"], 2),
            "grid_statistics": grid_stats,
            "hotspots": hotspots,
            "top_problematic_items": weighted_report["top_problematic_items"]
        }

        if return_detections:
            result["detections"] = detections

        if return_analysis:
            result["full_semantic_report"] = weighted_report

        return result

    def visualize_results(
        self,
        image_path: str,
        results: Dict,
        show_heatmap: bool = True,
        show_hotspots: bool = True,
        show_bboxes: bool = True
    ) -> np.ndarray:
        image = cv2.imread(image_path)
        vis = image.copy()

        detections = results.get("detections", [])

        # draw bounding boxes for confident detections
        if show_bboxes:
            for detection in detections:
                if detection.confidence >= 0.4:
                    x_min, y_min, x_max, y_max = detection.bbox

                    # draw the box
                    cv2.rectangle(vis, (x_min, y_min),
                                  (x_max, y_max), (0, 255, 0), 2)

                    # add label above box
                    label = f"{detection.class_name}: {detection.confidence:.2f}"
                    cv2.putText(
                        vis, label,
                        (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

        # overlay heatmap
        if show_heatmap:
            heatmap_overlay = self.spatial_analyzer.generate_heatmap_overlay(
                vis, alpha=0.5
            )
            vis = heatmap_overlay

        # draw hotspot rectangles
        if show_hotspots and results.get("hotspots"):
            vis = self.spatial_analyzer.visualize_hotspots(
                vis, results["hotspots"],
                color=(0, 0, 255), thickness=2
            )

        # show score info as text overlay
        text_lines = [
            f"Scene: {results['scene_class']} ({results['scene_confidence']:.2f})",
            f"Litter Count: {results['litter_count']}",
            f"Context Score: {results['context_aware_score']}/5.0",
            f"Weighted Score: {results['weighted_semantic_score']}/5.0",
            f"Level: {results['cleanliness_level']}"
        ]

        y_pos = 30
        for line in text_lines:
            cv2.putText(
                vis, line,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2, cv2.LINE_AA
            )
            y_pos += 35

        return vis

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        skip_frames: int = 1
    ) -> Dict:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # read video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # setup output writer if needed
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

        frame_idx = 0
        all_scores = []
        all_scenes = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # skip frames we dont want to process
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            # run detection and scoring per frame
            detections = self.detect_litter(frame)
            scene_class, _, _ = self.classify_scene(frame)
            score = self.context_scorer.compute_context_aware_score(
                detections, scene_class
            )

            all_scores.append(score)
            all_scenes.append(scene_class)

            # update spatial grid for this frame
            self.spatial_analyzer.create_grid(height, width)
            self.spatial_analyzer.populate_grid(detections)
            hotspots = self.spatial_analyzer.identify_hotspots()

            vis = frame.copy()

            # draw detections on frame
            for detection in detections:
                if detection.confidence >= 0.4:
                    x_min, y_min, x_max, y_max = detection.bbox
                    cv2.rectangle(vis, (x_min, y_min),
                                  (x_max, y_max), (0, 255, 0), 2)

            # apply heatmap overlay
            heatmap_vis = self.spatial_analyzer.generate_heatmap_overlay(
                vis, alpha=0.4
            )

            # add score text
            cv2.putText(
                heatmap_vis,
                f"Score: {score:.1f}/5.0 | Scene: {scene_class}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # write frame to output video
            if writer:
                writer.write(heatmap_vis)

            frame_idx += 1

        # release everything
        cap.release()
        if writer:
            writer.release()

        # aggregate stats for whole video
        return {
            "total_frames": total_frames,
            "processed_frames": len(all_scores),
            "fps": fps,
            "average_score": round(np.mean(all_scores), 2) if all_scores else 0,
            "max_score": round(np.max(all_scores), 2) if all_scores else 0,
            "min_score": round(np.min(all_scores), 2) if all_scores else 0,
            "scene_distribution": {
                scene: all_scenes.count(scene)
                for scene in set(all_scenes)
            },
            "output_path": output_path
        }
