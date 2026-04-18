#!/usr/bin/env python3

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
    print("[*] loading system...")
    system = StreetCleanlinessDetectionSystem(
        yolo_model_path=yolo_path,
        scene_classifier_path=scene_classifier_path,
        device=device
    )
    print("[+] system ready.")
    return system


def process_image_command(args):
    print(f"\n[*] processing: {args.source}")

    system = setup_system(
        yolo_path=args.yolo_weights,
        scene_classifier_path=args.scene_classifier,
        device=args.device
    )

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[!] not found: {args.source}")
        return

    try:
        print("\n[*] analyzing...")
        results = system.process_image(str(source_path))

        print("\n" + "="*60)
        print("results")
        print("="*60)

        print(f"\nimage: {source_path.name}")
        print(f"size: {results['image_size'][0]}x{results['image_size'][1]}")

        print("\n--- scene ---")
        print(f"type: {results['scene_class']}")
        print(f"conf: {results['scene_confidence']:.1%}")
        for scene, prob in results['scene_probabilities'].items():
            print(f"  {scene}: {prob:.1%}")

        print("\n--- detections ---")
        print(f"total: {results['litter_count']}")
        print(f"confident: {results['confident_litter_count']}")

        print("\n--- scores ---")
        print(f"context: {results['context_aware_score']}/5.0 ({results['cleanliness_level']})")
        print(f"semantic: {results['weighted_semantic_score']}/5.0")
        print(f"action: {results['recommendation']}")

        print("\n--- spatial ---")
        grid_stats = results['grid_statistics']
        print(f"cells: {grid_stats['cells_with_detections']}/{grid_stats['total_cells']}")
        print(f"avg per cell: {grid_stats['detections_per_cell_avg']:.2f}")
        print(f"concentration: {grid_stats['spatial_concentration']:.3f}")

        # todo: mudit fix hotspot display to show pixel coords too
        if results.get('hotspots'):
            print("\n--- hotspots ---")
            for i, hs in enumerate(results['hotspots'][:3], 1):
                print(f"  {i}: {hs['severity']} | count={hs['detection_count']} | grid=({hs['grid_x']},{hs['grid_y']})")

        if results.get('top_problematic_items'):
            print("\n--- worst litter types ---")
            for i, item in enumerate(results['top_problematic_items'], 1):
                print(f"  {i}. {item['class']}: {item['count']} (w={item['weight']})")

        if args.output:
            print("\n[*] saving viz...")
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            vis = system.visualize_results(
                str(source_path), results,
                show_heatmap=True, show_hotspots=True, show_bboxes=True
            )
            cv2.imwrite(str(output_path), vis)
            print(f"[+] saved: {output_path}")

        if args.save_json:
            json_path = Path(args.save_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            results_copy = results.copy()
            results_copy.pop("detections", None)
            results_copy.pop("full_semantic_report", None)
            with open(json_path, 'w') as f:
                json.dump(results_copy, f, indent=2)
            print(f"[+] json: {json_path}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"[!] error: {e}")
        import traceback
        traceback.print_exc()


def process_video_command(args):
    print(f"\n[*] processing video: {args.source}")

    system = setup_system(
        yolo_path=args.yolo_weights,
        scene_classifier_path=args.scene_classifier,
        device=args.device
    )

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[!] not found: {args.source}")
        return

    try:
        output_video = args.output or "output/output_video.mp4"

        print(f"\n[*] processing (skip={args.skip_frames})...")
        stats = system.process_video(
            str(source_path),
            output_path=output_video,
            skip_frames=args.skip_frames
        )

        print("\n" + "="*60)
        print("video results")
        print("="*60)

        print(f"\nvideo: {source_path.name}")
        print(f"fps: {stats['fps']:.1f}")
        print(f"frames: {stats['total_frames']} total / {stats['processed_frames']} processed")

        print("\n--- scores ---")
        print(f"avg: {stats['average_score']}/5.0")
        print(f"min: {stats['min_score']}/5.0")
        print(f"max: {stats['max_score']}/5.0")

        print("\n--- scenes ---")
        for scene, count in stats['scene_distribution'].items():
            pct = (count / stats['processed_frames'] * 100) if stats['processed_frames'] > 0 else 0
            print(f"  {scene}: {count} ({pct:.1f}%)")

        if output_video:
            print(f"\n[+] video: {output_video}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"[!] error: {e}")
        import traceback
        traceback.print_exc()


def demo_command(args):
    print("\n[*] running demo...")

    source_dir = Path(getattr(args, "source_dir",
                      "data/processed/taco_yolo_real/images/test"))
    output_dir = Path(getattr(args, "output_dir", "output/real_demo"))
    max_images = int(getattr(args, "max_images", 5))

    default_weights = Path("runs/detect/output/taco_real_train/weights/best.pt")
    yolo_weights = args.yolo_weights or (
        str(default_weights) if default_weights.exists() else None)

    # try real taco images first
    if source_dir.exists():
        images = sorted(source_dir.glob("*.jpg"))
        if not images:
            print(f"[!] no images in: {source_dir}")
            return

        print(f"[*] using: {source_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        system = setup_system(
            yolo_path=yolo_weights,
            scene_classifier_path=args.scene_classifier,
            device=args.device,
        )

        if system.yolo_model is None:
            print("[!] yolo unavailable")
            return

        selected = []
        for img_path in images:
            results = system.process_image(str(img_path))
            if results.get("confident_litter_count", 0) > 0:
                selected.append((img_path, results))
            if len(selected) >= max_images:
                break

        if not selected:
            img_path = images[0]
            selected = [(img_path, system.process_image(str(img_path)))]

        for img_path, results in selected:
            vis = system.visualize_results(
                str(img_path), results,
                show_heatmap=True, show_hotspots=True, show_bboxes=True,
            )
            out_img = output_dir / f"{img_path.stem}_context_pipeline.jpg"
            cv2.imwrite(str(out_img), vis)

            out_json = output_dir / f"{img_path.stem}_context_pipeline.json"
            results_copy = results.copy()
            results_copy.pop("detections", None)
            results_copy.pop("full_semantic_report", None)
            with open(out_json, "w") as f:
                json.dump(results_copy, f, indent=2)

            print(f"[+] {img_path.name} -> {out_img}")

        print(f"[+] done. outputs in: {output_dir}")
        return

    # fallback to synthetic image
    print("[*] no real data found, using synthetic image")

    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = (100, 150, 120)
    cv2.rectangle(test_image, (100, 200), (150, 250), (0, 0, 255), -1)
    cv2.rectangle(test_image, (400, 300), (450, 350), (200, 200, 0), -1)

    test_image_path = output_dir / "test_image.jpg"
    cv2.imwrite(str(test_image_path), test_image)
    print(f"[+] test image: {test_image_path}")

    system = setup_system(device=args.device)

    try:
        results = system.process_image(str(test_image_path))
        print("[+] synthetic demo done")
        print(f"    scene: {results.get('scene_class', 'unknown')}")
        print(f"    count: {results.get('litter_count', 0)}")
        print(f"    score: {results.get('context_aware_score', 0):.1f}/5.0")
    except Exception as e:
        print(f"[!] failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="street cleanliness detection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--scene-classifier", default=None)

    subparsers = parser.add_subparsers(dest="command")

    # image subcommand
    image_parser = subparsers.add_parser("image")
    image_parser.add_argument("--source", "-s", required=True)
    image_parser.add_argument("--output", "-o")
    image_parser.add_argument("--save-json")
    image_parser.set_defaults(func=process_image_command)

    # video subcommand
    video_parser = subparsers.add_parser("video")
    video_parser.add_argument("--source", "-s", required=True)
    video_parser.add_argument("--output", "-o", default="output/output_video.mp4")
    video_parser.add_argument("--skip-frames", type=int, default=5)
    video_parser.set_defaults(func=process_video_command)

    # demo subcommand
    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--source-dir",
                             default="data/processed/taco_yolo_real/images/test")
    demo_parser.add_argument("--output-dir", default="output/real_demo")
    demo_parser.add_argument("--max-images", type=int, default=5)
    demo_parser.set_defaults(func=demo_command)

    args = parser.parse_args()

    print("\n" + "="*60)
    print("street cleanliness detection system")
    print("context-aware | spatial analysis | weighted scoring")
    print("="*60)

    if args.command:
        args.func(args)
    else:
        parser.print_help()

    print("\n")


if __name__ == "__main__":
    main()
