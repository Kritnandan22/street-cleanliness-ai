#!/usr/bin/env python3
"""
Street Cleanliness Detection System - Main CLI Interface
Complete end-to-end system for analyzing street cleanliness using AI
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from inference.detection_pipeline import StreetCleanlinessDetectionSystem


def setup_system(
    yolo_path: Optional[str] = None,
    scene_classifier_path: Optional[str] = None,
    device: str = "cpu"
) -> StreetCleanlinessDetectionSystem:
    """Initialize the detection system"""
    print("[*] Initializing Street Cleanliness Detection System...")
    system = StreetCleanlinessDetectionSystem(
        yolo_model_path=yolo_path,
        scene_classifier_path=scene_classifier_path,
        device=device
    )
    print("[+] System initialized successfully!")
    return system


def process_image_command(args):
    """Handle image processing command"""
    print(f"\n[*] Processing image: {args.source}")

    # Setup system
    system = setup_system(
        yolo_path=args.yolo_weights,
        scene_classifier_path=args.scene_classifier,
        device=args.device
    )

    # Validate input
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[!] Error: Image file not found: {args.source}")
        return

    try:
        # Process image
        print("\n[*] Analyzing image...")
        results = system.process_image(str(source_path))

        # Print results
        print("\n" + "="*60)
        print("STREET CLEANLINESS ANALYSIS RESULTS")
        print("="*60)

        print(f"\nImage: {source_path.name}")
        print(f"Size: {results['image_size'][0]}x{results['image_size'][1]}")

        print("\n--- Scene Classification ---")
        print(f"Scene Type: {results['scene_class']}")
        print(f"Confidence: {results['scene_confidence']:.1%}")
        print("Probabilities:")
        for scene, prob in results['scene_probabilities'].items():
            print(f"  {scene}: {prob:.1%}")

        print("\n--- Litter Detection ---")
        print(f"Total Litter Objects: {results['litter_count']}")
        print(
            f"High-Confidence Detections: {results['confident_litter_count']}"
        )

        print("\n--- Cleanliness Scores ---")
        print(
            "Context-Aware Score: "
            f"{results['context_aware_score']}/5.0 "
            f"({results['cleanliness_level']})"
        )
        print(
            "Weighted Semantic Score: "
            f"{results['weighted_semantic_score']}/5.0"
        )
        print(f"Recommendation: {results['recommendation']}")

        print("\n--- Spatial Analysis ---")
        grid_stats = results['grid_statistics']
        print(
            "Cells with Detections: "
            f"{grid_stats['cells_with_detections']}/"
            f"{grid_stats['total_cells']}"
        )
        print(
            "Avg Detections per Cell: "
            f"{grid_stats['detections_per_cell_avg']:.2f}"
        )
        print(
            "Spatial Concentration: "
            f"{grid_stats['spatial_concentration']:.3f}"
        )

        if results.get('hotspots'):
            print("\n--- Top Hotspots ---")
            for i, hotspot in enumerate(results['hotspots'][:3], 1):
                print(
                    f"Hotspot {i}: {hotspot['severity']} | "
                    f"Detections: {hotspot['detection_count']} | "
                    "Position: Grid "
                    f"({hotspot['grid_x']}, {hotspot['grid_y']})"
                )

        if results.get('top_problematic_items'):
            print("\n--- Most Problematic Litter Types ---")
            for i, item in enumerate(results['top_problematic_items'], 1):
                print(
                    f"{i}. {item['class']}: {item['count']} objects "
                    f"(Weight: {item['weight']})"
                )

        # Save visualization
        if args.output:
            print("\n[*] Generating visualization...")
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            vis = system.visualize_results(
                str(source_path),
                results,
                show_heatmap=True,
                show_hotspots=True,
                show_bboxes=True
            )

            cv2.imwrite(str(output_path), vis)
            print(f"[+] Visualization saved to: {output_path}")

        # Save JSON results
        if args.save_json:
            json_path = Path(args.save_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove non-serializable objects
            results_copy = results.copy()
            results_copy.pop("detections", None)
            results_copy.pop("full_semantic_report", None)

            with open(json_path, 'w') as f:
                json.dump(results_copy, f, indent=2)
            print(f"[+] Results saved to: {json_path}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"[!] Error processing image: {e}")
        import traceback
        traceback.print_exc()


def process_video_command(args):
    """Handle video processing command"""
    print(f"\n[*] Processing video: {args.source}")

    # Setup system
    system = setup_system(
        yolo_path=args.yolo_weights,
        scene_classifier_path=args.scene_classifier,
        device=args.device
    )

    # Validate input
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[!] Error: Video file not found: {args.source}")
        return

    try:
        # Determine output path
        output_video = args.output or "output/output_video.mp4"

        print(
            "\n[*] Processing video (skipping every "
            f"{args.skip_frames} frames)..."
        )
        stats = system.process_video(
            str(source_path),
            output_path=output_video,
            skip_frames=args.skip_frames
        )

        print("\n" + "="*60)
        print("VIDEO PROCESSING RESULTS")
        print("="*60)

        print(f"\nVideo: {source_path.name}")
        print(f"FPS: {stats['fps']:.1f}")
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Processed Frames: {stats['processed_frames']}")

        print("\n--- Cleanliness Statistics ---")
        print(f"Average Score: {stats['average_score']}/5.0")
        print(f"Min Score: {stats['min_score']}/5.0")
        print(f"Max Score: {stats['max_score']}/5.0")

        print("\n--- Scene Distribution ---")
        for scene, count in stats['scene_distribution'].items():
            percentage = (count / stats['processed_frames']
                          * 100) if stats['processed_frames'] > 0 else 0
            print(f"{scene}: {count} frames ({percentage:.1f}%)")

        if output_video:
            print(f"\n[+] Output video saved to: {output_video}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"[!] Error processing video: {e}")
        import traceback
        traceback.print_exc()


def demo_command(args):
    """Run a demo.

    If a prepared real TACO subset exists, this runs the full pipeline on real
    images and writes visualizations and JSON reports.
    Otherwise it falls back to a synthetic sanity-check image.
    """
    print("\n[*] Running demonstration...")

    source_dir = Path(getattr(args, "source_dir",
                      "data/processed/taco_yolo_real/images/test"))
    output_dir = Path(getattr(args, "output_dir", "output/real_demo"))
    max_images = int(getattr(args, "max_images", 5))

    default_weights = Path(
        "runs/detect/output/taco_real_train/weights/best.pt")
    yolo_weights = args.yolo_weights or (
        str(default_weights) if default_weights.exists() else None)

    # Preferred: real demo on the prepared TACO subset
    if source_dir.exists():
        images = sorted(source_dir.glob("*.jpg"))
        if not images:
            print(f"[!] No images found in: {source_dir}")
            return

        print(f"[*] Using real images from: {source_dir}")
        if yolo_weights:
            print(f"[*] Using YOLO weights: {yolo_weights}")
        else:
            print("[*] Using default YOLO weights (yolov8n.pt)")

        output_dir.mkdir(parents=True, exist_ok=True)

        system = setup_system(
            yolo_path=yolo_weights,
            scene_classifier_path=args.scene_classifier,
            device=args.device,
        )

        if system.yolo_model is None:
            print("[!] Demo cannot run (ultralytics/YOLO unavailable)")
            return

        selected = []
        for img_path in images:
            results = system.process_image(str(img_path))
            if results.get("confident_litter_count", 0) > 0:
                selected.append((img_path, results))
            if len(selected) >= max_images:
                break

        if not selected:
            # Still generate at least one output.
            img_path = images[0]
            selected = [(img_path, system.process_image(str(img_path)))]

        for img_path, results in selected:
            vis = system.visualize_results(
                str(img_path),
                results,
                show_heatmap=True,
                show_hotspots=True,
                show_bboxes=True,
            )

            out_img = output_dir / f"{img_path.stem}_context_pipeline.jpg"
            cv2.imwrite(str(out_img), vis)

            out_json = output_dir / f"{img_path.stem}_context_pipeline.json"
            results_copy = results.copy()
            results_copy.pop("detections", None)
            results_copy.pop("full_semantic_report", None)
            with open(out_json, "w") as f:
                json.dump(results_copy, f, indent=2)

            print(f"[+] {img_path.name} -> {out_img} (+ {out_json})")

        print(f"[+] Real demo complete. Outputs in: {output_dir}")
        return

    # Fallback: synthetic sanity check image
    print("[*] Real demo dataset not found; using synthetic demo image")

    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = (100, 150, 120)

    cv2.rectangle(test_image, (100, 200), (150, 250), (0, 0, 255), -1)
    cv2.rectangle(test_image, (400, 300), (450, 350), (200, 200, 0), -1)

    test_image_path = output_dir / "test_image.jpg"
    cv2.imwrite(str(test_image_path), test_image)

    print(f"[+] Test image created: {test_image_path}")

    system = setup_system(device=args.device)

    try:
        results = system.process_image(str(test_image_path))
        print("[+] Synthetic demo completed")
        print(f"    Scene: {results.get('scene_class', 'unknown')}")
        print(f"    Litter Count: {results.get('litter_count', 0)}")
        print(
            "    Context Score: "
            f"{results.get('context_aware_score', 0):.1f}/5.0"
        )
    except Exception as e:
        print("[!] Synthetic demo failed")
        print(f"    Error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Street Cleanliness Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Process single image
  python main.py image --source street_photo.jpg --output result.jpg

  # Process video with heatmap
  python main.py video --source street_video.mp4 --output output_video.mp4

  # Run demo
  python main.py demo

  # Use CUDA GPU
  python main.py image --source photo.jpg --device cuda

  # Save results as JSON
  python main.py image --source photo.jpg --save-json results.json
        """
    )

    # Global arguments
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--yolo-weights",
        default=None,
        help="Path to YOLOv8 weights (auto-download if not specified)"
    )
    parser.add_argument(
        "--scene-classifier",
        default=None,
        help="Path to scene classifier weights"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Image command
    image_parser = subparsers.add_parser("image", help="Process single image")
    image_parser.add_argument(
        "--source", "-s",
        required=True,
        help="Path to input image"
    )
    image_parser.add_argument(
        "--output", "-o",
        help="Path to save output visualization"
    )
    image_parser.add_argument(
        "--save-json",
        help="Path to save results as JSON"
    )
    image_parser.set_defaults(func=process_image_command)

    # Video command
    video_parser = subparsers.add_parser("video", help="Process video")
    video_parser.add_argument(
        "--source", "-s",
        required=True,
        help="Path to input video"
    )
    video_parser.add_argument(
        "--output", "-o",
        default="output/output_video.mp4",
        help="Path to save output video"
    )
    video_parser.add_argument(
        "--skip-frames",
        type=int,
        default=5,
        help="Process every nth frame (default: 5)"
    )
    video_parser.set_defaults(func=process_video_command)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration")
    demo_parser.add_argument(
        "--source-dir",
        default="data/processed/taco_yolo_real/images/test",
        help="Folder of demo images (default: prepared TACO test split)",
    )
    demo_parser.add_argument(
        "--output-dir",
        default="output/real_demo",
        help="Folder to write demo outputs",
    )
    demo_parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Maximum number of demo images to process",
    )
    demo_parser.set_defaults(func=demo_command)

    # Parse arguments
    args = parser.parse_args()

    # Print banner
    print("\n" + "="*60)
    print("STREET CLEANLINESS DETECTION SYSTEM")
    print("Context-Aware • Spatial Analysis • Weighted Scoring")
    print("="*60)

    # Execute command
    if args.command:
        args.func(args)
    else:
        parser.print_help()

    print("\n")


if __name__ == "__main__":
    main()
