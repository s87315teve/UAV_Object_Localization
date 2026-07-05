#!/usr/bin/env python3
"""Run vehicle localization on every augmented test image and summarize results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scripts/localize_vehicles.py on a directory of augmented images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("augmented_test_data_run_20260704/images"),
        help="Directory containing augmented .jpg images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("augmented_test_data_run_20260704_localization_outputs"),
        help="Directory where per-image outputs and summary files are written.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device passed to localize_vehicles.py, e.g. cpu, mps, cuda:0.",
    )
    parser.add_argument("--yolo-model", default="yolo26x.pt")
    parser.add_argument("--detector", default="auto", choices=("auto", "yolo", "white-heuristic"))
    parser.add_argument("--vehicle-classes", default="car")
    parser.add_argument("--yolo-batch-size", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--tile-upscales", default="1,2")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--max-detections", type=int, default=None)
    parser.add_argument("--target-verifier", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-verifier-min-score", type=float, default=0.18)
    parser.add_argument("--target-verifier-min-white-ratio", type=float, default=0.15)
    parser.add_argument("--target-verifier-min-red-pixels", type=int, default=35)
    parser.add_argument("--orientations", default="all")
    parser.add_argument("--match-workers", type=int, default=4)
    parser.add_argument("--feature-max-dim", type=int, default=1200)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run images even if their vehicle_localization.json already exists.",
    )
    return parser.parse_args()


def localize_command(args: argparse.Namespace, image_path: Path, image_output: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/localize_vehicles.py",
        "--frame",
        str(image_path),
        "--output-dir",
        str(image_output),
        "--detector",
        args.detector,
        "--yolo-model",
        args.yolo_model,
        "--vehicle-classes",
        args.vehicle_classes,
        "--yolo-batch-size",
        str(args.yolo_batch_size),
        "--imgsz",
        str(args.imgsz),
        "--tile-upscales",
        args.tile_upscales,
        "--orientations",
        args.orientations,
        "--match-workers",
        str(args.match_workers),
        "--feature-max-dim",
        str(args.feature_max_dim),
        "--no-save-crops",
    ]
    if args.tile_size is not None:
        command.extend(["--tile-size", str(args.tile_size)])
    if args.tile_overlap is not None:
        command.extend(["--tile-overlap", str(args.tile_overlap)])
    if args.conf is not None:
        command.extend(["--conf", str(args.conf)])
    if args.max_detections is not None:
        command.extend(["--max-detections", str(args.max_detections)])
    if args.target_verifier:
        command.extend(
            [
                "--target-verifier",
                "--target-verifier-min-score",
                str(args.target_verifier_min_score),
                "--target-verifier-min-white-ratio",
                str(args.target_verifier_min_white_ratio),
                "--target-verifier-min-red-pixels",
                str(args.target_verifier_min_red_pixels),
            ]
        )
    if args.device:
        command.extend(["--device", args.device])
    return command


def report_row(
    image_path: Path,
    image_output: Path,
    status: str,
    returncode: int,
    elapsed_seconds: float,
    args: argparse.Namespace,
    error: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "filename": image_path.name,
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "output_dir": str(image_output),
        "device": args.device or "",
        "yolo_model": args.yolo_model,
        "yolo_batch_size": args.yolo_batch_size,
        "detections_count": "",
        "target_verifier_kept": "",
        "target_verifier_rejected": "",
        "map_method": "",
        "map_score": "",
        "map_orientation": "",
        "timing_total": "",
        "error": error,
    }

    report_path = image_output / "vehicle_localization.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            row["status"] = "report_parse_failed"
            row["error"] = str(exc)
            return row
        row.update(
            {
                "detections_count": report.get("detections_count", ""),
                "target_verifier_kept": report.get("target_verifier", {}).get("kept_count", ""),
                "target_verifier_rejected": report.get("target_verifier", {}).get("rejected_count", ""),
                "map_method": report.get("map_match", {}).get("method", ""),
                "map_score": report.get("map_match", {}).get("score", ""),
                "map_orientation": report.get("map_match", {}).get("orientation", ""),
                "timing_total": report.get("timings_seconds", {}).get("total", ""),
            }
        )
    return row


def write_summary(output_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    summary_csv = output_root / "summary.csv"
    summary_json = output_root / "summary.json"
    with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    images = sorted(args.input_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg images found in {args.input_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    print(
        f"Running {len(images)} images with device={args.device or 'auto'}, "
        f"yolo_batch_size={args.yolo_batch_size}"
    )

    for index, image_path in enumerate(images, start=1):
        image_output = args.output_root / image_path.stem
        image_output.mkdir(parents=True, exist_ok=True)
        report_path = image_output / "vehicle_localization.json"

        if report_path.is_file() and not args.overwrite:
            print(f"[{index:02d}/{len(images):02d}] {image_path.name} skipped")
            rows.append(report_row(image_path, image_output, "skipped", 0, 0.0, args))
            write_summary(args.output_root, rows)
            continue

        print(f"[{index:02d}/{len(images):02d}] {image_path.name}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(
            localize_command(args, image_path, image_output),
            text=True,
            capture_output=True,
        )
        elapsed_seconds = time.perf_counter() - started
        (image_output / "run.stdout.log").write_text(result.stdout, encoding="utf-8")
        (image_output / "run.stderr.log").write_text(result.stderr, encoding="utf-8")

        error = ""
        if result.returncode != 0:
            stderr_lines = result.stderr.strip().splitlines()
            error = stderr_lines[-1] if stderr_lines else "command failed"
        rows.append(
            report_row(
                image_path,
                image_output,
                "ok" if result.returncode == 0 else "failed",
                result.returncode,
                elapsed_seconds,
                args,
                error,
            )
        )
        write_summary(args.output_root, rows)

    status_counts = {
        status: sum(1 for row in rows if row["status"] == status)
        for status in sorted({str(row["status"]) for row in rows})
    }
    print(f"Wrote summary: {args.output_root / 'summary.csv'}")
    print(f"Wrote summary: {args.output_root / 'summary.json'}")
    print(f"Status counts: {status_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
