"""
Visualization Module for Street Cleanliness Detection System
Provides all drawing, overlay, and chart generation utilities.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── Color Palettes ──────────────────────────────────────────────────────────

CLASS_COLORS = {
    "plastic":  (0, 80, 255),    # vivid blue
    "metal":    (0, 165, 255),   # orange
    "paper":    (0, 255, 180),   # cyan-green
    "organic":  (50, 220, 50),   # green
    "glass":    (200, 0, 200),   # magenta
    "other":    (128, 128, 128), # gray
}
DEFAULT_COLOR = (0, 255, 0)

SEVERITY_COLORS = {
    "Critical": (0, 0, 220),
    "High":     (0, 100, 255),
    "Medium":   (0, 200, 255),
    "Low":      (60, 220, 120),
    "None":     (100, 100, 100),
}

SCORE_COLORS = {
    "Excellent": (0, 200, 80),
    "Good":      (80, 220, 0),
    "Average":   (0, 180, 220),
    "Poor":      (0, 100, 255),
    "Critical":  (0, 0, 220),
}


# ─── Bounding Box Drawing ─────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    detections: list,
    confidence_threshold: float = 0.35,
    show_confidence: bool = True,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """Draw bounding boxes with class-colored labels.

    Args:
        image: BGR image (numpy array)
        detections: List of Detection objects or dicts with 'class_name',
                    'confidence', 'bbox' (x1,y1,x2,y2)
        confidence_threshold: Minimum confidence to draw
    Returns:
        Annotated copy of image
    """
    vis = image.copy()

    for det in detections:
        # Support both Detection objects and plain dicts
        if hasattr(det, "confidence"):
            conf = det.confidence
            cname = det.class_name
            bbox = det.bbox
        else:
            conf = det.get("confidence", 1.0)
            cname = det.get("class_name", "other")
            bbox = det.get("bbox", (0, 0, 10, 10))

        if conf < confidence_threshold:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = CLASS_COLORS.get(cname.lower(), DEFAULT_COLOR)

        # Draw box with slight shadow for depth
        cv2.rectangle(vis, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1),
                      (0, 0, 0), thickness)   # shadow
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Label background
        label = f"{cname}"
        if show_confidence:
            label += f" {conf:.2f}"
        (lw, lh), bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y1 = max(y1 - lh - bl - 4, 0)
        cv2.rectangle(vis,
                      (x1, label_y1),
                      (x1 + lw + 4, label_y1 + lh + bl + 4),
                      color, -1)
        cv2.putText(vis, label, (x1 + 2, label_y1 + lh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ─── Score Panel / HUD ────────────────────────────────────────────────────────

def draw_score_panel(
    image: np.ndarray,
    results: Dict,
    panel_alpha: float = 0.75,
    panel_width: int = 300,
) -> np.ndarray:
    """Draw a semi-transparent HUD panel showing scene + scores.

    Args:
        image: BGR image
        results: Results dict from StreetCleanlinessDetectionSystem.process_image()
        panel_alpha: Opacity of background panel
    Returns:
        Image with HUD overlay
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    # Panel rectangle
    pw = min(panel_width, w - 10)
    ph = 185
    panel = vis[5: 5 + ph, 5: 5 + pw].copy()
    overlay = np.zeros_like(panel)
    overlay[:] = (20, 20, 20)
    panel_blended = cv2.addWeighted(panel, 1 - panel_alpha, overlay, panel_alpha, 0)
    vis[5: 5 + ph, 5: 5 + pw] = panel_blended

    scene_class = results.get("scene_class", "unknown")
    ctx_score = results.get("context_aware_score", 0.0)
    wsem_score = results.get("weighted_semantic_score", 0.0)
    litter = results.get("litter_count", 0)
    level = results.get("cleanliness_level", "Unknown")
    rec = results.get("recommendation", "")

    level_color = SCORE_COLORS.get(level, (200, 200, 200))
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = [
        (f"Scene      : {scene_class}", (200, 200, 200)),
        (f"Litter Det : {litter}", (200, 200, 200)),
        (f"Context Sc : {ctx_score:.1f}/5.0", level_color),
        (f"Semantic Sc: {wsem_score:.1f}/5.0", level_color),
        (f"Level      : {level}", level_color),
        (f"Action     : {rec[:28]}", (180, 220, 180)),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(vis, text, (12, 28 + i * 27),
                    font, 0.52, color, 1, cv2.LINE_AA)

    return vis


# ─── Heatmap Overlay ─────────────────────────────────────────────────────────

def draw_heatmap_overlay(
    image: np.ndarray,
    spatial_analyzer,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a spatial litter heatmap over the image.

    Args:
        image: BGR image
        spatial_analyzer: Populated SpatialHeatmapAnalyzer instance
        alpha: Heatmap visibility  [0=invisible, 1=fully opaque]
    """
    return spatial_analyzer.generate_heatmap_overlay(
        image, colormap=colormap, alpha=alpha
    )


# ─── Hotspot Markers ──────────────────────────────────────────────────────────

def draw_hotspot_markers(
    image: np.ndarray,
    hotspots: List[Dict],
    max_show: int = 5,
    thickness: int = 2,
) -> np.ndarray:
    """Draw severity-colored hotspot rectangles.

    Args:
        image: BGR image
        hotspots: Output of SpatialHeatmapAnalyzer.identify_hotspots()
        max_show: Maximum number of hotspots to show
    """
    vis = image.copy()
    for i, hs in enumerate(hotspots[:max_show]):
        x1, y1, x2, y2 = hs["pixel_bounds"]
        sev = hs.get("severity", "None")
        color = SEVERITY_COLORS.get(sev, (128, 128, 128))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = f"H{i + 1} {sev}"
        cv2.putText(vis, label,
                    (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)

    return vis


# ─── Full Composite Visualisation ─────────────────────────────────────────────

def create_full_visualization(
    image_path: str,
    results: Dict,
    spatial_analyzer,
    show_heatmap: bool = True,
    show_bboxes: bool = True,
    show_hotspots: bool = True,
    show_panel: bool = True,
    heatmap_alpha: float = 0.40,
    bbox_confidence: float = 0.35,
) -> np.ndarray:
    """Create the complete annotated output image.

    Args:
        image_path: Path to original image
        results: Output of StreetCleanlinessDetectionSystem.process_image()
        spatial_analyzer: Populated SpatialHeatmapAnalyzer instance
    Returns:
        Fully annotated BGR image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    vis = img.copy()

    # 1. Heatmap
    if show_heatmap:
        vis = draw_heatmap_overlay(vis, spatial_analyzer, alpha=heatmap_alpha)

    # 2. Bounding boxes
    if show_bboxes:
        dets = results.get("detections", [])
        vis = draw_detections(vis, dets,
                              confidence_threshold=bbox_confidence)

    # 3. Hotspot markers
    if show_hotspots and results.get("hotspots"):
        vis = draw_hotspot_markers(vis, results["hotspots"])

    # 4. Score HUD panel
    if show_panel:
        vis = draw_score_panel(vis, results)

    return vis


# ─── Ablation Comparison Chart ────────────────────────────────────────────────

def create_ablation_chart(
    rows: List[Dict],
    output_path: Optional[str] = None,
    title: str = "Ablation Study: Cleanliness Scoring Modes",
) -> Optional[np.ndarray]:
    """Generate a matplotlib figure comparing the four scoring modes.

    Args:
        rows: List of per-image ablation result dicts
        output_path: If given, save to this path
    Returns:
        BGR image of the chart (as numpy array), or None if matplotlib unavailable
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return None

    if not rows:
        return None

    keys = ["mode_A_raw_count", "mode_B_context_aware",
            "mode_C_weighted_semantic", "mode_D_full_pipeline"]
    labels = ["A: Raw Count", "B: Context-Aware",
              "C: Weighted Semantic", "D: Full Pipeline\n(Ours)"]
    colors = ["#6c757d", "#17a2b8", "#ffc107", "#28a745"]
    means = [float(np.mean([r[k] for r in rows if k in r])) for k in keys]
    stds = [float(np.std([r[k] for r in rows if k in r])) for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Bar chart
    bars = ax1.bar(labels, means, yerr=stds, color=colors,
                   capsize=6, edgecolor="black", linewidth=0.7)
    ax1.set_ylim(0, 6.0)
    ax1.set_ylabel("Mean Cleanliness Score (0-5)")
    ax1.set_title("Average Score per Mode (Higher = Cleaner)")
    ax1.axhline(y=2.5, color="red", ls="--", alpha=0.4, label="Midline (2.5)")
    ax1.legend(fontsize=8)
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 m + s + 0.08, f"{m:.2f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Line plot (first 30 images)
    n = min(30, len(rows))
    sample = rows[:n]
    xi = list(range(n))
    for k, lbl, col, mk in zip(keys, ["A", "B", "C", "D (Ours)"], colors, ["s", "D", "^", "o"]):
        vals = [r.get(k, 0) for r in sample]
        ax2.plot(xi, vals, marker=mk, ms=4, label=lbl, color=col, lw=1.5)
    ax2.set_xlabel("Image Index")
    ax2.set_ylabel("Score (0-5)")
    ax2.set_title(f"Per-Image Scores (first {n} test images)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 5.5)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    # Convert to BGR numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


# ─── Score Bar (horizontal banner) ────────────────────────────────────────────

def draw_score_banner(
    image: np.ndarray,
    score: float,
    label: str = "Cleanliness Score",
    position: str = "bottom",
) -> np.ndarray:
    """Draw a horizontal colored score bar at top or bottom of image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # Map score 0-5 to color  (red ← → green)
    t = max(0.0, min(1.0, score / 5.0))
    r = int(220 * (1 - t))
    g = int(200 * t)
    bar_color = (0, g, r)

    bar_h = 30
    y = h - bar_h if position == "bottom" else 0

    cv2.rectangle(vis, (0, y), (w, y + bar_h), (20, 20, 20), -1)
    filled_w = int(w * t)
    cv2.rectangle(vis, (0, y), (filled_w, y + bar_h), bar_color, -1)

    text = f"{label}: {score:.1f}/5.0"
    cv2.putText(vis, text, (10, y + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis
