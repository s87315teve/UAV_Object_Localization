#!/usr/bin/env python3
"""Generate verifier test cases with ordinary, target, and confusing vehicles."""

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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


DEFAULT_BACKGROUNDS = [
    Path("test_image/frame_000051.jpg"),
    Path("test_image/frame_000161.jpg"),
    Path("stream_outputs/marker_review/video01_003s.jpg"),
]
DEFAULT_CAR_CROPS = [
    Path("vehicle_localization_outputs/frame_000051/crops/veh_001.jpg"),
    Path("vehicle_localization_outputs/frame_000051/crops/veh_002.jpg"),
    Path("vehicle_localization_outputs/frame_000051/crops/veh_003.jpg"),
    Path("vehicle_localization_outputs/frame_000051/crops/veh_004.jpg"),
    Path("vehicle_localization_outputs/frame_000161/crops/veh_001.jpg"),
]
DEFAULT_TARGET_CAR_CROPS = [
    Path("vehicle_localization_outputs/frame_000051/crops/veh_001.jpg"),
    Path("vehicle_localization_outputs/frame_000051/crops/veh_002.jpg"),
]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".MP4", ".MOV", ".M4V", ".AVI"}


@dataclass
class SourceFrame:
    image: Image.Image
    source_type: str
    source_path: Path
    timestamp_seconds: float | None = None


@dataclass
class VehicleAsset:
    source_path: Path
    image: Image.Image


@dataclass
class PlacedObject:
    role: str
    bbox_xyxy: tuple[int, int, int, int]
    marker_bbox_xyxy: tuple[int, int, int, int] | None
    source_crop: str
    style: str
    marker_style: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create augmented verifier test images. Each image contains ordinary vehicles, "
            "target vehicles with red X markers, and confusing vehicles."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("target_verifier_test_data"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--video-dir", type=Path, default=Path("raw_videos"))
    parser.add_argument("--video-frame-count", type=int, default=6)
    parser.add_argument("--ordinary-cars", type=int, default=2)
    parser.add_argument("--target-cars", type=int, default=1)
    parser.add_argument("--confuser-cars", type=int, default=2)
    parser.add_argument("--marker-size-min", type=float, default=0.16)
    parser.add_argument("--marker-size-max", type=float, default=0.30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_dir(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def find_videos(video_dir: Path) -> list[Path]:
    if not video_dir.is_dir():
        return []
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


def load_background_sources(args: argparse.Namespace, rng: random.Random) -> list[SourceFrame]:
    sources: list[SourceFrame] = []
    for path in DEFAULT_BACKGROUNDS:
        if path.is_file():
            sources.append(SourceFrame(Image.open(path).convert("RGB"), "background_image", path))

    videos = find_videos(args.video_dir)
    if videos and args.video_frame_count > 0:
        durations = {video: video_duration(video) for video in videos}
        with tempfile.TemporaryDirectory(prefix="uav_target_bg_") as temp_root:
            temp_dir = Path(temp_root)
            for index in range(args.video_frame_count):
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

    if not sources:
        raise FileNotFoundError("No background sources found.")
    return sources


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
    return alpha_image.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2.0))


def load_vehicle_assets() -> list[VehicleAsset]:
    assets = []
    for path in DEFAULT_CAR_CROPS:
        if not path.is_file():
            continue
        car = Image.open(path).convert("RGB")
        alpha = build_car_alpha(car)
        rgba = car.convert("RGBA")
        rgba.putalpha(alpha)
        assets.append(VehicleAsset(path, rgba))
    if not assets:
        raise FileNotFoundError("No vehicle crops found under vehicle_localization_outputs.")
    return assets


def select_target_vehicle_assets(assets: list[VehicleAsset]) -> list[VehicleAsset]:
    target_paths = {path.resolve() for path in DEFAULT_TARGET_CAR_CROPS if path.is_file()}
    target_assets = [asset for asset in assets if asset.source_path.resolve() in target_paths]
    return target_assets or assets


def colorize_vehicle(vehicle: Image.Image, role: str, rng: random.Random) -> tuple[Image.Image, str]:
    rgba = vehicle.convert("RGBA")
    rgb = np.asarray(rgba.convert("RGB")).astype(np.float32)
    alpha = np.asarray(rgba.getchannel("A"))

    if role == "target":
        result = rgba.copy()
        result.putalpha(Image.fromarray(alpha, "L"))
        return result, "real_white_vehicle_crop_with_marker"
    elif role == "ordinary":
        palette = [
            np.array([210, 210, 205], dtype=np.float32),
            np.array([48, 78, 132], dtype=np.float32),
            np.array([58, 58, 58], dtype=np.float32),
            np.array([204, 178, 70], dtype=np.float32),
        ]
        target = rng.choice(palette)
        rgb = rgb * 0.45 + target.reshape(1, 1, 3) * 0.55
        style = "ordinary_any_color"
    else:
        style = rng.choice(["white_no_marker", "red_vehicle_no_x", "red_patch_on_white", "red_x_on_colored"])
        if style in {"white_no_marker", "red_patch_on_white"}:
            target = np.array([235, 235, 225], dtype=np.float32)
            rgb = rgb * 0.22 + target.reshape(1, 1, 3) * 0.78
        else:
            target = np.array([160, 35, 45], dtype=np.float32)
            rgb = rgb * 0.35 + target.reshape(1, 1, 3) * 0.65

    result = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    result.putalpha(Image.fromarray(alpha, "L"))
    return result, style


def draw_target_marker(
    vehicle: Image.Image,
    rng: random.Random,
    marker_size_range: tuple[float, float],
    marker_kind: str,
) -> tuple[Image.Image, str]:
    result = vehicle.copy()
    base_alpha = result.getchannel("A")
    alpha_array = np.asarray(base_alpha)
    ys, xs = np.where(alpha_array > 40)
    width, height = result.size
    if len(xs) == 0 or len(ys) == 0:
        return result, "none"

    car_x1, car_y1, car_x2, car_y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    car_width = max(1, car_x2 - car_x1)
    car_height = max(1, car_y2 - car_y1)
    size_fraction = rng.uniform(*marker_size_range)
    side = max(10, int(min(car_width, car_height) * size_fraction))
    side = min(side, car_width, car_height)
    jitter_x = max(1, int(car_width * 0.12))
    jitter_y = max(1, int(car_height * 0.12))
    cx = (car_x1 + car_x2) // 2 + rng.randint(-jitter_x, jitter_x)
    cy = (car_y1 + car_y2) // 2 + rng.randint(-jitter_y, jitter_y)
    x1 = min(max(car_x1, cx - side // 2), max(car_x1, car_x2 - side))
    y1 = min(max(car_y1, cy - side // 2), max(car_y1, car_y2 - side))
    x2 = min(car_x2 - 1, x1 + side)
    y2 = min(car_y2 - 1, y1 + side)
    red = rng.choice([(180, 24, 42, 235), (205, 45, 75, 220), (150, 30, 64, 235)])
    white = rng.choice([(245, 245, 235, 235), (235, 238, 230, 225), (255, 245, 245, 230)])
    if marker_kind in {"target_x", "confuser_x"}:
        thickness = max(3, int(side * rng.uniform(0.28, 0.40)))
    else:
        thickness = max(2, int(side * rng.uniform(0.12, 0.22)))

    marker_layer = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(marker_layer)
    if marker_kind in {"target_x", "confuser_x"}:
        draw.rectangle((x1, y1, x2, y2), fill=red)
        inset = max(1, int(side * rng.uniform(0.10, 0.18)))
        draw.line((x1 + inset, y1 + inset, x2 - inset, y2 - inset), fill=white, width=thickness)
        draw.line((x1 + inset, y2 - inset, x2 - inset, y1 + inset), fill=white, width=thickness)
    elif marker_kind == "red_patch":
        draw.rectangle((x1, y1, x2, y2), fill=red)
    else:
        return result, "none"

    marker_alpha = np.asarray(marker_layer.getchannel("A"))
    clipped_alpha = np.minimum(marker_alpha, alpha_array).astype(np.uint8)
    marker_layer.putalpha(Image.fromarray(clipped_alpha, "L"))
    result.alpha_composite(marker_layer)
    result.putalpha(base_alpha)

    return result, f"{marker_kind};red_background={red[:3]};fat_white_x={white[:3]};side={side};thickness={thickness}"


def red_marker_bbox(vehicle: Image.Image) -> tuple[int, int, int, int] | None:
    rgba = np.asarray(vehicle.convert("RGBA"))
    red = rgba[:, :, 0].astype(np.int16)
    green = rgba[:, :, 1].astype(np.int16)
    blue = rgba[:, :, 2].astype(np.int16)
    alpha = rgba[:, :, 3]
    mask = (alpha > 40) & (red > green + 10) & (red > blue + 3) & (red > 120)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def transform_vehicle(asset: VehicleAsset, role: str, args: argparse.Namespace, rng: random.Random) -> tuple[Image.Image, str, str]:
    vehicle, style = colorize_vehicle(asset.image, role, rng)
    marker_style = "none"
    if role == "target":
        vehicle, marker_style = draw_target_marker(
            vehicle,
            rng,
            (args.marker_size_min, args.marker_size_max),
            "target_x",
        )
    elif role == "confuser" and style == "red_patch_on_white":
        vehicle, marker_style = draw_target_marker(vehicle, rng, (0.10, 0.22), "red_patch")
    elif role == "confuser" and style == "red_x_on_colored":
        vehicle, marker_style = draw_target_marker(vehicle, rng, (0.12, 0.24), "confuser_x")

    scale = rng.uniform(0.95, 1.25) if role == "target" else rng.uniform(0.72, 1.26)
    target_size = (
        max(32, int(vehicle.width * scale)),
        max(32, int(vehicle.height * scale)),
    )
    vehicle = vehicle.resize(target_size, Image.Resampling.LANCZOS)
    angle = rng.uniform(0, 360)
    vehicle = vehicle.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    marker_style = f"{marker_style};scale={scale:.3f};angle={angle:.2f}"
    return vehicle, style, marker_style


def bbox_from_alpha(vehicle: Image.Image, x: int, y: int) -> tuple[int, int, int, int]:
    alpha = np.asarray(vehicle.getchannel("A"))
    ys, xs = np.where(alpha > 20)
    if len(xs) == 0:
        return x, y, x + vehicle.width, y + vehicle.height
    return x + int(xs.min()), y + int(ys.min()), x + int(xs.max() + 1), y + int(ys.max() + 1)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def placement_score(frame: Image.Image, x: int, y: int, width: int, height: int) -> float:
    patch = frame.crop((x, y, x + width, y + height)).resize((24, 24), Image.Resampling.BILINEAR)
    array = np.asarray(patch).astype(np.float32) / 255.0
    saturation = array.max(axis=2) - array.min(axis=2)
    brightness = array.mean(axis=2)
    green = array[:, :, 1]
    red = array[:, :, 0]
    blue = array[:, :, 2]
    green_excess = np.maximum(0.0, green - red) + np.maximum(0.0, green - blue)
    return (
        (1.0 - float(saturation.mean())) * 1.8
        - float(green_excess.mean()) * 3.0
        - abs(float(brightness.mean()) - 0.52) * 0.8
    )


def choose_position(
    frame: Image.Image,
    vehicle: Image.Image,
    occupied: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int]:
    width, height = frame.size
    margin_x = max(1, int(width * 0.05))
    margin_y = max(1, int(height * 0.05))
    max_x = max(margin_x, width - margin_x - vehicle.width)
    max_y = max(margin_y, height - margin_y - vehicle.height)
    best: tuple[int, int, float] | None = None
    for _ in range(160):
        x = rng.randint(margin_x, max_x)
        y = rng.randint(margin_y, max_y)
        bbox = bbox_from_alpha(vehicle, x, y)
        if any(bbox_iou(bbox, other) > 0.03 for other in occupied):
            continue
        score = placement_score(frame, x, y, vehicle.width, vehicle.height)
        if best is None or score > best[2]:
            best = (x, y, score)
    if best is not None:
        return best[0], best[1]
    return rng.randint(margin_x, max_x), rng.randint(margin_y, max_y)


def paste_vehicle(
    frame: Image.Image,
    vehicle: Image.Image,
    role: str,
    asset_path: Path,
    style: str,
    marker_style: str,
    occupied: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[Image.Image, PlacedObject]:
    x, y = choose_position(frame, vehicle, occupied, rng)
    result = frame.convert("RGBA")
    shadow = Image.new("RGBA", vehicle.size, (0, 0, 0, 0))
    shadow_alpha = vehicle.getchannel("A").filter(ImageFilter.GaussianBlur(3.0)).point(lambda value: int(value * 0.20))
    shadow.putalpha(shadow_alpha)
    result.alpha_composite(shadow, (x + 3, y + 3))
    result.alpha_composite(vehicle, (x, y))
    bbox = bbox_from_alpha(vehicle, x, y)
    occupied.append(bbox)
    local_marker = red_marker_bbox(vehicle) if not marker_style.startswith("none") else None
    marker_bbox = None
    if local_marker is not None:
        mx1, my1, mx2, my2 = local_marker
        marker_bbox = (x + mx1, y + my1, x + mx2, y + my2)
    return result.convert("RGB"), PlacedObject(role, bbox, marker_bbox, str(asset_path), style, marker_style)


def apply_scene_degradation(image: Image.Image, rng: random.Random) -> tuple[Image.Image, list[str]]:
    operations: list[str] = []
    result = image.copy()
    brightness = rng.uniform(0.76, 1.18)
    contrast = rng.uniform(0.78, 1.22)
    color = rng.uniform(0.78, 1.18)
    result = ImageEnhance.Brightness(result).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    result = ImageEnhance.Color(result).enhance(color)
    operations.extend([f"brightness={brightness:.3f}", f"contrast={contrast:.3f}", f"color={color:.3f}"])
    if rng.random() < 0.65:
        radius = rng.uniform(0.35, 1.45)
        result = result.filter(ImageFilter.GaussianBlur(radius))
        operations.append(f"blur={radius:.2f}")
    if rng.random() < 0.80:
        array = np.asarray(result).astype(np.float32)
        sigma = rng.uniform(1.5, 8.0)
        noise_rng = np.random.default_rng(rng.randrange(2**32))
        array += noise_rng.normal(0, sigma, array.shape)
        result = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
        operations.append(f"noise_sigma={sigma:.2f}")
    quality = rng.randint(58, 88)
    buffer = io.BytesIO()
    result.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    result = Image.open(buffer).convert("RGB")
    operations.append(f"jpeg_quality={quality}")
    return result, operations


def object_to_row(image_name: str, obj: PlacedObject) -> dict[str, str]:
    return {
        "filename": image_name,
        "role": obj.role,
        "bbox_xyxy": json.dumps(list(obj.bbox_xyxy)),
        "marker_bbox_xyxy": json.dumps(list(obj.marker_bbox_xyxy) if obj.marker_bbox_xyxy else []),
        "source_crop": obj.source_crop,
        "style": obj.style,
        "marker_style": obj.marker_style,
    }


def write_metadata(output_dir: Path, image_rows: list[dict[str, str]], object_rows: list[dict[str, str]]) -> None:
    metadata_csv = output_dir / "metadata.csv"
    objects_csv = output_dir / "objects.csv"
    metadata_json = output_dir / "metadata.json"
    if image_rows:
        with metadata_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(image_rows)
    if object_rows:
        with objects_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(object_rows[0].keys()))
            writer.writeheader()
            writer.writerows(object_rows)
    metadata_json.write_text(
        json.dumps({"images": image_rows, "objects": object_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be greater than 0")
    if args.target_cars <= 0:
        raise ValueError("--target-cars must be greater than 0")
    if args.ordinary_cars < 0 or args.confuser_cars < 0:
        raise ValueError("Vehicle counts cannot be negative")
    if not 0.05 <= args.marker_size_min <= args.marker_size_max <= 0.55:
        raise ValueError("Marker size range must satisfy 0.05 <= min <= max <= 0.55")

    rng = random.Random(args.seed)
    images_dir, labels_dir = ensure_output_dir(args.output_dir, args.overwrite)
    sources = load_background_sources(args, rng)
    assets = load_vehicle_assets()
    target_assets = select_target_vehicle_assets(assets)

    image_rows: list[dict[str, str]] = []
    object_rows: list[dict[str, str]] = []
    roles = ["ordinary"] * args.ordinary_cars + ["target"] * args.target_cars + ["confuser"] * args.confuser_cars

    for index in range(1, args.count + 1):
        source = sources[(index - 1) % len(sources)]
        frame = source.image.copy()
        occupied: list[tuple[int, int, int, int]] = []
        placed: list[PlacedObject] = []
        shuffled_roles = roles[:]
        rng.shuffle(shuffled_roles)
        for role in shuffled_roles:
            asset = rng.choice(target_assets if role == "target" else assets)
            vehicle, style, marker_style = transform_vehicle(asset, role, args, rng)
            frame, obj = paste_vehicle(frame, vehicle, role, asset.source_path, style, marker_style, occupied, rng)
            placed.append(obj)

        frame, operations = apply_scene_degradation(frame, rng)
        image_name = f"target_case_{index:06d}.jpg"
        label_name = f"target_case_{index:06d}.json"
        frame.save(images_dir / image_name, quality=90, optimize=True)

        label_payload = {
            "filename": image_name,
            "source_type": source.source_type,
            "source_path": str(source.source_path),
            "timestamp_seconds": source.timestamp_seconds,
            "augmentations": operations,
            "objects": [
                {
                    "role": obj.role,
                    "bbox_xyxy": list(obj.bbox_xyxy),
                    "marker_bbox_xyxy": list(obj.marker_bbox_xyxy) if obj.marker_bbox_xyxy else [],
                    "source_crop": obj.source_crop,
                    "style": obj.style,
                    "marker_style": obj.marker_style,
                }
                for obj in placed
            ],
        }
        (labels_dir / label_name).write_text(json.dumps(label_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        image_rows.append(
            {
                "filename": image_name,
                "label": str(Path("labels") / label_name),
                "source_type": source.source_type,
                "source_path": str(source.source_path),
                "timestamp_seconds": f"{source.timestamp_seconds:.3f}" if source.timestamp_seconds is not None else "",
                "ordinary_count": str(args.ordinary_cars),
                "target_count": str(args.target_cars),
                "confuser_count": str(args.confuser_cars),
                "augmentations": "|".join(operations),
                "seed": str(args.seed),
            }
        )
        for obj in placed:
            object_rows.append(object_to_row(image_name, obj))

    write_metadata(args.output_dir, image_rows, object_rows)
    print(f"Wrote {len(image_rows)} images to {images_dir}")
    print(f"Wrote labels to {labels_dir}")
    print(f"Wrote metadata to {args.output_dir / 'metadata.csv'} and {args.output_dir / 'objects.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
