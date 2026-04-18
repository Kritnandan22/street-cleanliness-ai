#!/usr/bin/env python3

import os
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from inference.detection_pipeline import StreetCleanlinessDetectionSystem
from visualization.visualizer import create_full_visualization
from utils.context_aware_scorer import ScoreInterpretation


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16mb max upload

PROJECT_ROOT = Path(__file__).resolve().parent

print("[*] loading detection system...")
weights_path = PROJECT_ROOT / "models" / "best_yolov8_taco.pt"
if not weights_path.exists():
    weights_path = "yolov8n.pt"  # fallback to base yolo

system = StreetCleanlinessDetectionSystem(
    yolo_model_path=str(weights_path),
    scene_classifier_path=None,
    device="cpu"
)

def image_to_base64(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

# todo: chayan fix analyze to handle multi image batch
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # read image directly to numpy, no disk save yet
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image file"}), 400

        try:
            # need temp file because process_image takes a path
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                temp_path = temp_img.name
                cv2.imwrite(temp_path, img)

            try:
                # run full detection pipeline
                results = system.process_image(image_path=temp_path)

                # generate all 4 view layers
                vis_full = create_full_visualization(
                    temp_path, results, system.spatial_analyzer,
                    show_heatmap=True, show_bboxes=True, show_hotspots=True, show_panel=False)

                vis_labels = create_full_visualization(
                    temp_path, results, system.spatial_analyzer,
                    show_heatmap=False, show_bboxes=True, show_hotspots=False, show_panel=False)

                vis_heatmap = create_full_visualization(
                    temp_path, results, system.spatial_analyzer,
                    show_heatmap=True, show_bboxes=False, show_hotspots=False, show_panel=False)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            def encode_img(image_array):
                _, buffer = cv2.imencode('.jpg', image_array)
                return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

            img_b64 = encode_img(vis_full)
            labels_b64 = encode_img(vis_labels)
            heatmap_b64 = encode_img(vis_heatmap)
            original_b64 = encode_img(img)

            # compute final score: clean image uses ctx directly, dirty uses blend
            _litter_count = int(results["litter_count"])
            _ctx = float(results["context_aware_score"])
            _sem = float(results["weighted_semantic_score"])
            _final_score = _ctx if _litter_count == 0 else min(
                (0.55 * _ctx) + (0.45 * (5.0 - _sem)), 5.0
            )
            # badge from final score not context score
            _level = ScoreInterpretation.interpret_score(_final_score)[0]

            # recommendation from final score
            dynamic_rec = ScoreInterpretation.get_recommendation(
                _final_score, results["scene_class"]
            )

            # append class-specific alerts
            detected_classes = set(d.class_name for d in results.get("detections", []))
            if "glass" in detected_classes:
                dynamic_rec += " Hazard warning: glass detected; require specialized cleanup gear."
            elif "plastic" in detected_classes or "metal" in detected_classes:
                dynamic_rec += " Ecological alert: High-impact non-biodegradable materials present."

            # append hotspot info if any
            valid_hotspots = [h for h in results.get("hotspots", []) if h.get("severity") in ("Critical", "High", "Medium")]
            if valid_hotspots:
                worst_severity = str(valid_hotspots[0].get("severity", "Medium"))
                dynamic_rec += f" Dispatch targeted sweep team straight to {worst_severity}-severity spatial clusters."

            results_safe = {
                "litter_count": _litter_count,
                "scene_class": str(results["scene_class"]),
                "context_aware_score": round(_ctx, 2),
                "weighted_semantic_score": round(_sem, 2),
                "final_cleanliness_score": round(_final_score, 2),
                "cleanliness_level": _level,
                "recommendation": dynamic_rec,
                "hotspots": [
                    {
                        "severity": str(h.get("severity", "Unknown")),
                        "litter_count": int(h.get("detection_count", 0)),
                        "bounds": [int(x) for x in h.get("pixel_bounds", [0,0,0,0])]
                    }
                    for h in results.get("hotspots", [])[:3]
                ],
                "detections": [
                    {
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox": [int(x) for x in d.bbox]
                    }
                    for d in results.get("detections", [])
                ]
            }

            return jsonify({
                "status": "success",
                "results": results_safe,
                "image_data": img_b64,
                "labels_data": labels_b64,
                "heatmap_data": heatmap_b64,
                "original_data": original_b64
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"[*] starting server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
