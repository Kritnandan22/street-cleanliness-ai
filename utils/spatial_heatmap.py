"""
Novelty Component 2: Spatial Heatmap Analysis
Generates region-wise pollution heatmaps and identifies hotspots
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class GridCell:
    """Represents one cell in the spatial grid"""
    x_idx: int
    y_idx: int
    detection_count: int = 0
    total_area: float = 0.0
    avg_confidence: float = 0.0
    detections: List = None

    def __post_init__(self):
        if self.detections is None:
            self.detections = []


class SpatialHeatmapAnalyzer:
    """Analyzes spatial distribution of litter using grid-based heatmapping

    Key Innovation:
    Divides image into spatial grid and computes per-region pollution density.
    Identifies "hotspots" of high litter concentration.
    Enables location-specific cleanup prioritization.
    """

    def __init__(self, grid_size: int = 8):
        """Initialize analyzer

        Args:
            grid_size: Divide image into grid_size x grid_size cells
        """
        self.grid_size = grid_size
        self.grid: List[List[GridCell]] = []

    def create_grid(self, image_height: int, image_width: int):
        """Initialize empty grid for image dimensions"""
        cell_height = image_height // self.grid_size
        cell_width = image_width // self.grid_size

        self.grid = [
            [
                GridCell(x_idx=j, y_idx=i)
                for j in range(self.grid_size)
            ]
            for i in range(self.grid_size)
        ]

        self.cell_height = cell_height
        self.cell_width = cell_width
        self.image_height = image_height
        self.image_width = image_width

    def assign_detection_to_grid(
        self,
        detection,
        detection_idx: int
    ):
        """Assign a detection to appropriate grid cell(s)

        A detection that spans multiple cells is assigned to all cells it covers
        """
        x_min, y_min, x_max, y_max = detection.bbox

        # Determine which grid cells the detection overlaps
        grid_x_min = max(0, x_min // self.cell_width)
        grid_x_max = min(self.grid_size - 1, x_max // self.cell_width)
        grid_y_min = max(0, y_min // self.cell_height)
        grid_y_max = min(self.grid_size - 1, y_max // self.cell_height)

        # Assign to all overlapping cells
        for y_idx in range(grid_y_min, grid_y_max + 1):
            for x_idx in range(grid_x_min, grid_x_max + 1):
                cell = self.grid[y_idx][x_idx]
                cell.detection_count += 1
                cell.total_area += detection.area
                cell.detections.append(detection)

        # Update average confidence
        for y_idx in range(grid_y_min, grid_y_max + 1):
            for x_idx in range(grid_x_min, grid_x_max + 1):
                cell = self.grid[y_idx][x_idx]
                if cell.detection_count > 0:
                    cell.avg_confidence = np.mean(
                        [d.confidence for d in cell.detections]
                    )

    def populate_grid(self, detections: List):
        """Fill grid with detection data

        Args:
            detections: List of Detection objects
        """
        for idx, detection in enumerate(detections):
            self.assign_detection_to_grid(detection, idx)

    def get_heatmap_matrix(self, use_area: bool = False) -> np.ndarray:
        """Generate 2D heatmap matrix

        Args:
            use_area: If True, use area instead of count

        Returns:
            2D numpy array with values [0, max_value]
        """
        heatmap = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32
        )

        for y_idx in range(self.grid_size):
            for x_idx in range(self.grid_size):
                cell = self.grid[y_idx][x_idx]
                if use_area:
                    heatmap[y_idx, x_idx] = cell.total_area
                else:
                    heatmap[y_idx, x_idx] = cell.detection_count

        return heatmap

    def generate_heatmap_overlay(
        self,
        image: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.6,
        use_area: bool = False
    ) -> np.ndarray:
        """Generate visual heatmap overlay on image.

        Uses per-pixel density-weighted alpha blending so that zero-density
        cells remain as the original image and high-density cells glow with
        a warm heat colour (red/orange/yellow).  Grid lines are also drawn
        to show the spatial decomposition.

        Args:
            image: Input image (BGR)
            colormap: OpenCV colormap (cv2.COLORMAP_*). Ignored in favour of
                      COLORMAP_HOT when the default COLORMAP_JET is passed so
                      that cold/empty areas are not coloured blue.
            alpha: Maximum heatmap opacity at the hottest cell [0, 1]
            use_area: If True, use total detection area instead of count

        Returns:
            Image with heatmap + grid overlay
        """
        vis = image.copy()
        h, w = vis.shape[:2]

        # ── 1.  Build density matrix ─────────────────────────────────────────
        heatmap_data = self.get_heatmap_matrix(use_area=use_area)

        # Normalise to [0, 1]
        max_val = heatmap_data.max()
        if max_val > 0:
            heatmap_norm_f = heatmap_data / max_val          # float [0,1]
        else:
            heatmap_norm_f = np.zeros_like(heatmap_data, dtype=np.float32)

        # Scale to uint8 for colormap
        heatmap_u8 = (heatmap_norm_f * 255).astype(np.uint8)

        # ── 2.  Resize → blur → colourmap ──────────────────────────────────
        heatmap_resized = cv2.resize(
            heatmap_u8,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (51, 51), 0)

        # Use COLORMAP_JET (blue→cyan→green→yellow→red rainbow) for vivid
        # heatmap like the reference image.
        heatmap_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)

        # ── 3.  Per-pixel alpha mask (density-proportional) ─────────────────
        # alpha_mask is 0 where no detections, up to `alpha` at the hottest spot
        alpha_mask = (heatmap_blurred.astype(np.float32) / 255.0) * alpha
        alpha_3ch = np.stack([alpha_mask] * 3, axis=-1)   # (H, W, 3)

        # Blend: output = original * (1 - a) + heatmap_colour * a
        vis = (
            vis.astype(np.float32) * (1.0 - alpha_3ch)
            + heatmap_colored.astype(np.float32) * alpha_3ch
        ).clip(0, 255).astype(np.uint8)

        # ── 4.  Draw spatial grid lines (thick & white for visibility) ──────
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        grid_color = (255, 255, 255)  # white — visible over any heatmap color
        grid_thickness = 3

        for i in range(1, self.grid_size):
            # Horizontal lines
            cv2.line(vis, (0, i * cell_h), (w, i * cell_h),
                     grid_color, grid_thickness, cv2.LINE_AA)
            # Vertical lines
            cv2.line(vis, (i * cell_w, 0), (i * cell_w, h),
                     grid_color, grid_thickness, cv2.LINE_AA)

        # Outer border
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), grid_color, grid_thickness)

        return vis

    def identify_hotspots(
        self,
        threshold_percentile: float = 75.0,
        use_area: bool = False
    ) -> List[Dict]:
        """Identify high-pollution regions (hotspots)

        Args:
            threshold_percentile: Percentile threshold for hotspot detection
                                  (e.g., 75 = top 25% dirtiest cells)
            use_area: Use area instead of count

        Returns:
            List of hotspot dictionaries with location and stats
        """
        # Get heatmap data
        heatmap_data = self.get_heatmap_matrix(use_area=use_area)

        # Calculate threshold
        threshold = np.percentile(heatmap_data, threshold_percentile)

        # Find cells above threshold
        hotspots = []
        for y_idx in range(self.grid_size):
            for x_idx in range(self.grid_size):
                value = heatmap_data[y_idx, x_idx]
                if value >= threshold and value > 0:
                    cell = self.grid[y_idx][x_idx]
                    x_min = x_idx * self.cell_width
                    x_max = (x_idx + 1) * self.cell_width
                    y_min = y_idx * self.cell_height
                    y_max = (y_idx + 1) * self.cell_height

                    hotspots.append({
                        "grid_x": int(x_idx),
                        "grid_y": int(y_idx),
                        "pixel_bounds": (int(x_min), int(y_min), int(x_max), int(y_max)),
                        "detection_count": int(cell.detection_count),
                        "total_area": float(cell.total_area),
                        "avg_confidence": float(cell.avg_confidence),
                        "pollution_level": float(value),
                        "severity": self._get_severity_level(float(value), float(heatmap_data.max()))
                    })

        # Sort by pollution level (descending)
        hotspots.sort(key=lambda x: x["pollution_level"], reverse=True)

        return hotspots

    @staticmethod
    def _get_severity_level(value: float, max_value: float) -> str:
        """Get severity label for pollution value"""
        if max_value == 0:
            return "None"

        normalized = value / max_value
        if normalized >= 0.75:
            return "Critical"
        elif normalized >= 0.5:
            return "High"
        elif normalized >= 0.25:
            return "Medium"
        else:
            return "Low"

    def visualize_hotspots(
        self,
        image: np.ndarray,
        hotspots: List[Dict],
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw hotspot rectangles on image

        Args:
            image: Input image
            hotspots: List of hotspot dicts from identify_hotspots()
            color: Rectangle color (BGR)
            thickness: Line thickness

        Returns:
            Image with hotspot rectangles
        """
        output = image.copy()

        for i, hotspot in enumerate(hotspots):
            x_min, y_min, x_max, y_max = hotspot["pixel_bounds"]

            # Draw rectangle
            cv2.rectangle(output, (x_min, y_min),
                          (x_max, y_max), color, thickness)

            # Add label
            label = f"Hotspot {i+1}\n{hotspot['severity']}"
            cv2.putText(
                output,
                label.split('\n')[0],
                (x_min + 5, y_min + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        return output

    def get_grid_statistics(self) -> Dict:
        """Get overall statistics for the grid

        Returns:
            Dictionary with statistics
        """
        all_counts = [
            cell.detection_count for row in self.grid for cell in row]
        all_areas = [cell.total_area for row in self.grid for cell in row]
        all_confidences = [
            cell.avg_confidence for row in self.grid for cell in row if cell.avg_confidence > 0]

        return {
            "total_cells": int(self.grid_size * self.grid_size),
            "cells_with_detections": int(sum(1 for count in all_counts if count > 0)),
            "detections_per_cell_avg": float(np.mean(all_counts)) if all_counts else 0.0,
            "detections_per_cell_max": int(np.max(all_counts)) if all_counts else 0,
            "detections_per_cell_min": int(np.min(all_counts)) if all_counts else 0,
            "area_per_cell_avg": float(np.mean(all_areas)) if all_areas else 0.0,
            "area_per_cell_max": float(np.max(all_areas)) if all_areas else 0.0,
            "avg_confidence": float(np.mean(all_confidences)) if all_confidences else 0.0,
            "spatial_concentration": float(self._compute_concentration_metric(all_counts))
        }

    @staticmethod
    def _compute_concentration_metric(counts: List[int]) -> float:
        """Compute how concentrated detections are (0=distributed, 1=concentrated)

        Uses Gini coefficient concept
        """
        if not counts or sum(counts) == 0:
            return 0.0

        counts = np.array(counts)
        counts_sorted = np.sort(counts)
        n = len(counts)

        cumsum = np.cumsum(counts_sorted)
        concentration = (2 * np.sum(np.arange(1, n + 1) *
                         counts_sorted)) / (n * cumsum[-1]) - (n + 1) / n

        return float(np.clip(concentration, 0, 1))
