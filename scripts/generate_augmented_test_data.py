#!/usr/bin/env python3
"""Generate a small degraded augmented UAV test image set."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


DEFAULT_TEST_IMAGES = [
    Path("test_image/frame_000051.jpg"),
    Path("test_image/frame_000161.jpg"),
]
DEFAULT_CAR_CROPS = [
    Path("vehicle_localization_outputs/frame_000051/crops/veh_001.jpg"),
    Path("vehicle_localization_outputs/frame_000051/crops/veh_002.jpg"),
    Path("vehicle_localization_outputs/frame_000161/crops/veh_001.jpg"),
]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".MP4", ".MOV", ".M4V", ".AVI"}


@dataclass
class SourceFrame:
    image: Image.Image
    source_type: str
    source_path: Path
    timestamp_seconds: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create augmented test images from existing test frames and random raw-video frames."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("augmented_test_data"),
        help="Directory for images and metadata. Default: augmented_test_data",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Total augmented output images. Default: 30",
    )
    parser.add_argument(
        "--video-frame-count",
        type=int,
        default=7,
        help="Number of random frames to sample from raw videos. Default: 7",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260703,
        help="Random seed for reproducible sampling and augmentation.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("raw_videos"),
        help="Directory containing raw videos. Default: raw_videos",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path, overwrite: bool) -> Path:
    images_dir = output_dir / "images"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def find_videos(video_dir: Path) -> list[Path]:
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    return sorted(path for path in video_dir.iterdir() if path.is_file() and path.suffix in VIDEO_EXTENSIONS)


def video_duration(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_video_frame(video_path: Path, timestamp_seconds: float, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(destination),
    ]
    subprocess.run(command, check=True)


def load_test_sources() -> list[SourceFrame]:
    sources = []
    for path in DEFAULT_TEST_IMAGES:
        if not path.is_file():
            raise FileNotFoundError(f"Test image not found: {path}")
        sources.append(SourceFrame(Image.open(path).convert("RGB"), "test_image", path))
    return sources


def sample_video_sources(video_dir: Path, count: int, rng: random.Random) -> list[SourceFrame]:
    if count <= 0:
        return []
    videos = find_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {video_dir}")

    durations = {video: video_duration(video) for video in videos}
    sources: list[SourceFrame] = []
    with tempfile.TemporaryDirectory(prefix="uav_aug_frames_") as temp_root:
        temp_dir = Path(temp_root)
        for index in range(count):
            video = rng.choice(videos)
            duration = durations[video]
            timestamp = rng.uniform(3.0, max(3.1, duration - 3.0))
            frame_path = temp_dir / f"video_frame_{index:03d}.jpg"
            extract_video_frame(video, timestamp, frame_path)
            sources.append(
                SourceFrame(
                    Image.open(frame_path).convert("RGB"),
                    "raw_video_frame",
                    video,
                    timestamp,
                )
            )
    return sources


def image_corner_fill(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    samples = [
        image.getpixel((0, 0)),
        image.getpixel((width - 1, 0)),
        image.getpixel((0, height - 1)),
        image.getpixel((width - 1, height - 1)),
    ]
    return tuple(int(sum(pixel[channel] for pixel in samples) / len(samples)) for channel in range(3))


def zoom_crop(image: Image.Image, zoom: float, rng: random.Random) -> Image.Image:
    if zoom <= 1.001:
        return image
    width, height = image.size
    crop_width = int(width / zoom)
    crop_height = int(height / zoom)
    left = rng.randint(0, width - crop_width)
    top = rng.randint(0, height - crop_height)
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)


def add_noise(image: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    if sigma <= 0:
        return image
    array = np.asarray(image).astype(np.float32)
    noise_rng = np.random.default_rng(rng.randrange(2**32))
    array += noise_rng.normal(0, sigma, array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def perspective_coefficients(
    source_points: list[tuple[float, float]],
    target_points: list[tuple[float, float]],
) -> list[float]:
    matrix = []
    vector = []
    for (source_x, source_y), (target_x, target_y) in zip(source_points, target_points):
        matrix.append([source_x, source_y, 1, 0, 0, 0, -target_x * source_x, -target_x * source_y])
        matrix.append([0, 0, 0, source_x, source_y, 1, -target_y * source_x, -target_y * source_y])
        vector.extend([target_x, target_y])
    return np.linalg.solve(np.array(matrix, dtype=np.float64), np.array(vector, dtype=np.float64)).tolist()


def mild_perspective_warp(image: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    width, height = image.size
    max_shift_x = width * rng.uniform(0.010, 0.035)
    max_shift_y = height * rng.uniform(0.010, 0.035)
    source_points = [(0, 0), (width, 0), (width, height), (0, height)]
    target_points = [
        (rng.uniform(0, max_shift_x), rng.uniform(0, max_shift_y)),
        (width - rng.uniform(0, max_shift_x), rng.uniform(0, max_shift_y)),
        (width - rng.uniform(0, max_shift_x), height - rng.uniform(0, max_shift_y)),
        (rng.uniform(0, max_shift_x), height - rng.uniform(0, max_shift_y)),
    ]
    coefficients = perspective_coefficients(target_points, source_points)
    warped = image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=image_corner_fill(image),
    )
    return warped, f"perspective_shift={max_shift_x:.1f}x{max_shift_y:.1f}"


def degrade_resolution(image: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    width, height = image.size
    scale = rng.uniform(0.42, 0.78)
    low_width = max(32, int(width * scale))
    low_height = max(32, int(height * scale))
    low_res = image.resize((low_width, low_height), Image.Resampling.BILINEAR)
    restored = low_res.resize((width, height), Image.Resampling.BICUBIC)
    return restored, f"downup_scale={scale:.3f}"


def apply_color_cast(image: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    array = np.asarray(image).astype(np.float32)
    cast = np.array(
        [
            rng.uniform(0.86, 1.16),
            rng.uniform(0.88, 1.12),
            rng.uniform(0.84, 1.18),
        ],
        dtype=np.float32,
    )
    array *= cast.reshape(1, 1, 3)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB"), (
        f"color_cast={cast[0]:.3f},{cast[1]:.3f},{cast[2]:.3f}"
    )


def recompress_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def augment_base_image(image: Image.Image, rng: random.Random) -> tuple[Image.Image, list[str]]:
    operations: list[str] = []
    result = image.copy()

    if rng.random() < 0.45:
        result = ImageOps.mirror(result)
        operations.append("mirror")

    zoom = rng.uniform(1.0, 1.16)
    if zoom > 1.015:
        result = zoom_crop(result, zoom, rng)
        operations.append(f"zoom={zoom:.3f}")

    angle = rng.uniform(-8.0, 8.0)
    if abs(angle) > 0.4:
        result = result.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            fillcolor=image_corner_fill(result),
        )
        operations.append(f"rotate={angle:.2f}")

    if rng.random() < 0.45:
        result, detail = mild_perspective_warp(result, rng)
        operations.append(detail)

    result, detail = degrade_resolution(result, rng)
    operations.append(detail)

    brightness = rng.uniform(0.62, 1.28)
    contrast = rng.uniform(0.64, 1.34)
    color = rng.uniform(0.62, 1.38)
    result = ImageEnhance.Brightness(result).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    result = ImageEnhance.Color(result).enhance(color)
    result, color_cast = apply_color_cast(result, rng)
    operations.extend(
        [
            f"brightness={brightness:.3f}",
            f"contrast={contrast:.3f}",
            f"color={color:.3f}",
            color_cast,
        ]
    )

    if rng.random() < 0.85:
        radius = rng.uniform(0.65, 2.20)
        result = result.filter(ImageFilter.GaussianBlur(radius))
        operations.append(f"blur={radius:.2f}")

    sigma = rng.uniform(3.0, 12.0)
    result = add_noise(result, sigma, rng)
    operations.append(f"noise_sigma={sigma:.2f}")

    jpeg_quality = rng.randint(38, 74)
    result = recompress_jpeg(result, jpeg_quality)
    operations.append(f"jpeg_quality={jpeg_quality}")

    return result, operations


def build_car_alpha(car: Image.Image) -> Image.Image:
    rgb = np.asarray(car.convert("RGB")).astype(np.int16)
    border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background, axis=2)
    threshold = max(18.0, float(np.percentile(distance, 63)))
    alpha = np.where(distance > threshold, 255, 0).astype(np.uint8)
    coverage = float(alpha.mean() / 255.0)

    if coverage < 0.12 or coverage > 0.75:
        width, height = car.size
        alpha_image = Image.new("L", car.size, 0)
        margin_x = max(2, int(width * 0.08))
        margin_y = max(2, int(height * 0.08))
        alpha_image.paste(255, (margin_x, margin_y, width - margin_x, height - margin_y))
    else:
        alpha_image = Image.fromarray(alpha, "L")

    alpha_image = alpha_image.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2.0))
    return alpha_image


def load_car_overlays() -> list[tuple[Path, Image.Image]]:
    overlays = []
    for path in DEFAULT_CAR_CROPS:
        if not path.is_file():
            raise FileNotFoundError(f"Car crop not found: {path}")
        car = Image.open(path).convert("RGB")
        alpha = build_car_alpha(car)
        rgba = car.convert("RGBA")
        rgba.putalpha(alpha)
        overlays.append((path, rgba))
    return overlays


def placement_score(frame: Image.Image, x: int, y: int, width: int, height: int) -> float:
    patch = frame.crop((x, y, x + width, y + height)).resize((24, 24), Image.Resampling.BILINEAR)
    array = np.asarray(patch).astype(np.float32) / 255.0
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    saturation = array.max(axis=2) - array.min(axis=2)
    brightness = array.mean(axis=2)
    green_excess = np.maximum(0.0, green - red) + np.maximum(0.0, green - blue)
    bright_penalty = max(0.0, float(brightness.mean()) - 0.78) * 2.5
    dark_penalty = max(0.0, 0.18 - float(brightness.mean())) * 2.5

    return (
        (1.0 - float(saturation.mean())) * 2.0
        - float(green_excess.mean()) * 3.0
        - abs(float(brightness.mean()) - 0.48) * 0.9
        - bright_penalty
        - dark_penalty
    )


def choose_car_position(frame: Image.Image, car: Image.Image, rng: random.Random) -> tuple[int, int, float]:
    width, height = frame.size
    margin_x = max(1, int(width * 0.08))
    margin_y = max(1, int(height * 0.08))
    max_x = max(margin_x, width - margin_x - car.width)
    max_y = max(margin_y, height - margin_y - car.height)

    best: tuple[int, int, float] | None = None
    for _ in range(140):
        x = rng.randint(margin_x, max_x)
        y = rng.randint(margin_y, max_y)
        score = placement_score(frame, x, y, car.width, car.height)
        if best is None or score > best[2]:
            best = (x, y, score)

    if best is None:
        return margin_x, margin_y, 0.0
    return best


def add_car(image: Image.Image, overlays: list[tuple[Path, Image.Image]], rng: random.Random) -> tuple[Image.Image, str, str]:
    car_path, car = rng.choice(overlays)
    scale = rng.uniform(0.70, 1.30)
    target_width = max(50, int(car.width * scale))
    target_height = max(50, int(car.height * scale))
    car = car.resize((target_width, target_height), Image.Resampling.LANCZOS)

    angle = rng.uniform(0, 360)
    car = car.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    x, y, score = choose_car_position(image, car, rng)

    result = image.convert("RGBA")
    shadow = Image.new("RGBA", car.size, (0, 0, 0, 0))
    shadow_alpha = car.getchannel("A").filter(ImageFilter.GaussianBlur(3.0)).point(lambda value: int(value * 0.22))
    shadow.putalpha(shadow_alpha)
    result.alpha_composite(shadow, (x + 3, y + 3))
    result.alpha_composite(car, (x, y))
    detail = (
        f"x={x};y={y};scale={scale:.3f};angle={angle:.2f};"
        f"size={car.width}x{car.height};placement_score={score:.3f}"
    )
    return result.convert("RGB"), str(car_path), detail


def write_metadata(output_dir: Path, rows: list[dict[str, str]]) -> None:
    csv_path = output_dir / "metadata.csv"
    json_path = output_dir / "metadata.json"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be greater than 0")
    if args.video_frame_count < 0:
        raise ValueError("--video-frame-count cannot be negative")

    rng = random.Random(args.seed)
    images_dir = ensure_output_dir(args.output_dir, args.overwrite)
    test_sources = load_test_sources()
    video_sources = sample_video_sources(args.video_dir, args.video_frame_count, rng)
    car_overlays = load_car_overlays()

    rows: list[dict[str, str]] = []
    video_outputs = min(args.count // 2, args.video_frame_count * 2)
    test_outputs = args.count - video_outputs

    output_index = 1
    for index in range(test_outputs):
        source = test_sources[index % len(test_sources)]
        augmented, operations = augment_base_image(source.image, rng)
        output_name = f"aug_{output_index:06d}_test.jpg"
        augmented.save(images_dir / output_name, quality=86, optimize=True)
        rows.append(
            {
                "filename": output_name,
                "source_type": source.source_type,
                "source_path": str(source.source_path),
                "timestamp_seconds": "",
                "car_crop": "",
                "car_overlay": "",
                "augmentations": "|".join(operations),
                "seed": str(args.seed),
            }
        )
        output_index += 1

    for index in range(video_outputs):
        source = video_sources[index % len(video_sources)]
        with_car, car_path, car_detail = add_car(source.image, car_overlays, rng)
        augmented, operations = augment_base_image(with_car, rng)
        output_name = f"aug_{output_index:06d}_video_car.jpg"
        augmented.save(images_dir / output_name, quality=86, optimize=True)
        rows.append(
            {
                "filename": output_name,
                "source_type": source.source_type,
                "source_path": str(source.source_path),
                "timestamp_seconds": f"{source.timestamp_seconds:.3f}" if source.timestamp_seconds is not None else "",
                "car_crop": car_path,
                "car_overlay": car_detail,
                "augmentations": "|".join(operations),
                "seed": str(args.seed),
            }
        )
        output_index += 1

    write_metadata(args.output_dir, rows)
    print(f"Wrote {len(rows)} augmented images to {images_dir}")
    print(f"Wrote metadata to {args.output_dir / 'metadata.csv'} and {args.output_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
