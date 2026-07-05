#!/usr/bin/env python3
"""Create a low-quality video03 test clip with a target car in every frame."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from generate_target_verifier_test_data import build_car_alpha, draw_target_marker


DEFAULT_SOURCE_VIDEO = Path("raw_videos/video03.MP4")
DEFAULT_OUTPUT_VIDEO = Path("augmented_test_data/videos/video03_target_low_quality.mp4")
DEFAULT_TARGET_CAR = Path("vehicle_localization_outputs/frame_000051/crops/veh_001.jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Downsample video03, degrade its visual quality, and paste a white target "
            "car with a red marker and white X into every output frame."
        )
    )
    parser.add_argument("--input-video", type=Path, default=DEFAULT_SOURCE_VIDEO)
    parser.add_argument("--output-video", type=Path, default=DEFAULT_OUTPUT_VIDEO)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--target-car", type=Path, default=DEFAULT_TARGET_CAR)
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--bitrate", default="900k")
    parser.add_argument("--max-size-mb", type=float, default=30.0)
    parser.add_argument("--target-long-side", type=int, default=118)
    parser.add_argument("--jpeg-quality", type=int, default=34)
    parser.add_argument("--noise-sigma", type=float, default=5.0)
    parser.add_argument("--blur-radius", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_can_write(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")


def make_real_crop_target_car(car_path: Path, long_side: int, rng: random.Random) -> tuple[Image.Image, str]:
    if not car_path.is_file():
        raise FileNotFoundError(f"Target car crop not found: {car_path}")

    car = Image.open(car_path).convert("RGB")
    alpha = build_car_alpha(car)
    rgba = car.convert("RGBA")
    rgba.putalpha(alpha)

    scale = long_side / float(max(rgba.size))
    target_size = (
        max(24, int(round(rgba.width * scale))),
        max(24, int(round(rgba.height * scale))),
    )
    rgba = rgba.resize(target_size, Image.Resampling.LANCZOS)
    rgba, marker_style = draw_target_marker(rgba, rng, (0.34, 0.46), "target_x")
    rgba = rgba.rotate(-9.0, expand=True, resample=Image.Resampling.BICUBIC)
    return rgba, marker_style


def target_position(frame_index: int, frame_count: int, frame_size: tuple[int, int], target: Image.Image) -> tuple[int, int]:
    width, height = frame_size
    progress = 0.0 if frame_count <= 1 else frame_index / float(frame_count - 1)
    margin_x = max(12, int(width * 0.08))
    x_min = margin_x
    x_max = max(x_min, width - margin_x - target.width)
    y_base = int(height * 0.58)
    y_wave = int(math.sin(progress * math.tau * 1.7) * height * 0.055)
    x = int(round(x_min + (x_max - x_min) * progress))
    y = y_base + y_wave
    y = min(max(int(height * 0.18), y), max(int(height * 0.18), height - int(height * 0.10) - target.height))
    return x, y


def alpha_bbox(image: Image.Image, origin: tuple[int, int]) -> list[int]:
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > 20)
    if len(xs) == 0:
        x, y = origin
        return [x, y, x + image.width, y + image.height]
    x, y = origin
    return [x + int(xs.min()), y + int(ys.min()), x + int(xs.max() + 1), y + int(ys.max() + 1)]


def paste_target(frame_rgb: np.ndarray, target: Image.Image, x: int, y: int) -> tuple[np.ndarray, list[int]]:
    frame = Image.fromarray(frame_rgb, "RGB").convert("RGBA")
    shadow = Image.new("RGBA", target.size, (0, 0, 0, 0))
    shadow_alpha = target.getchannel("A").filter(ImageFilter.GaussianBlur(3.0)).point(lambda value: int(value * 0.22))
    shadow.putalpha(shadow_alpha)
    frame.alpha_composite(shadow, (x + 3, y + 3))
    frame.alpha_composite(target, (x, y))
    return np.asarray(frame.convert("RGB")), alpha_bbox(target, (x, y))


def degrade_frame(
    frame_bgr: np.ndarray,
    output_size: tuple[int, int],
    target: Image.Image,
    output_index: int,
    total_outputs: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[np.ndarray, list[int]]:
    width, height = output_size
    resized = cv2.resize(frame_bgr, output_size, interpolation=cv2.INTER_AREA)
    low = cv2.resize(resized, (max(32, int(width * 0.64)), max(32, int(height * 0.64))), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(low, output_size, interpolation=cv2.INTER_LINEAR)
    resized = cv2.convertScaleAbs(resized, alpha=0.88, beta=-4)

    frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x, y = target_position(output_index, total_outputs, output_size, target)
    frame_rgb, bbox = paste_target(frame_rgb, target, x, y)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if args.blur_radius > 0:
        kernel_size = max(3, int(round(args.blur_radius * 4)) | 1)
        frame_bgr = cv2.GaussianBlur(frame_bgr, (kernel_size, kernel_size), args.blur_radius)
    if args.noise_sigma > 0:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, args.noise_sigma, frame_bgr.shape)
        frame_bgr = np.clip(frame_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    ok, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    if not ok:
        raise RuntimeError("Failed to recompress frame as JPEG")
    frame_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return frame_bgr, bbox


def video_info(video_path: Path) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open input video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Could not read video metadata from: {video_path}")
    return width, height, fps, frame_count


def make_ffmpeg_command(output_path: Path, output_size: tuple[int, int], fps: float, bitrate: str, overwrite: bool) -> list[str]:
    width, height = output_size
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-b:v",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        "1800k",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0")
    if args.output_width <= 0:
        raise ValueError("--output-width must be greater than 0")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    metadata_path = args.metadata or args.output_video.with_suffix(".metadata.json")
    preview_path = args.preview or args.output_video.with_suffix(".preview.jpg")
    ensure_can_write(args.output_video, args.overwrite)
    ensure_can_write(metadata_path, args.overwrite)
    ensure_can_write(preview_path, args.overwrite)

    source_width, source_height, source_fps, source_frame_count = video_info(args.input_video)
    output_height = int(round(args.output_width * source_height / source_width))
    if output_height % 2:
        output_height += 1
    output_size = (args.output_width, output_height)
    duration = source_frame_count / source_fps
    total_outputs = max(1, int(math.floor(duration * args.fps)))

    rng = random.Random(args.seed)
    target, marker_style = make_real_crop_target_car(args.target_car, args.target_long_side, rng)

    command = make_ffmpeg_command(args.output_video, output_size, args.fps, args.bitrate, args.overwrite)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")

    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open input video: {args.input_video}")

    rows: list[dict[str, object]] = []
    output_index = 0
    next_output_time = 0.0
    frame_index = 0
    preview_written = False
    try:
        while output_index < total_outputs:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            frame_index += 1
            if timestamp + (0.5 / source_fps) < next_output_time:
                continue

            degraded, bbox = degrade_frame(frame, output_size, target, output_index, total_outputs, args, rng)
            process.stdin.write(degraded.tobytes())
            if not preview_written:
                if not cv2.imwrite(str(preview_path), degraded, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                    raise RuntimeError(f"Failed to write preview: {preview_path}")
                preview_written = True
            rows.append(
                {
                    "output_frame_index": output_index,
                    "source_frame_index": frame_index - 1,
                    "source_timestamp_seconds": round(timestamp, 4),
                    "target_bbox_xyxy": bbox,
                }
            )
            output_index += 1
            next_output_time = output_index / args.fps
    finally:
        capture.release()
        process.stdin.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr.strip()}")

    size_bytes = args.output_video.stat().st_size
    max_bytes = int(args.max_size_mb * 1024 * 1024)
    if size_bytes > max_bytes:
        raise RuntimeError(
            f"Output video is {size_bytes / 1024 / 1024:.2f} MB, above {args.max_size_mb:.2f} MB: {args.output_video}"
        )

    metadata = {
        "output_video": str(args.output_video),
        "preview": str(preview_path),
        "source_video": str(args.input_video),
        "source_width": source_width,
        "source_height": source_height,
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "output_width": output_size[0],
        "output_height": output_size[1],
        "output_fps": args.fps,
        "output_frame_count": output_index,
        "output_size_bytes": size_bytes,
        "output_size_mb": round(size_bytes / 1024 / 1024, 3),
        "max_size_mb": args.max_size_mb,
        "target_car_crop": str(args.target_car),
        "target_marker_style": marker_style,
        "degradation": {
            "downscale_then_upscale": 0.64,
            "jpeg_quality": args.jpeg_quality,
            "noise_sigma": args.noise_sigma,
            "blur_radius": args.blur_radius,
            "bitrate": args.bitrate,
        },
        "seed": args.seed,
        "frames": rows,
    }
    write_metadata(metadata_path, metadata)
    print(f"Wrote {args.output_video} ({metadata['output_size_mb']} MB, {output_index} frames)")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
