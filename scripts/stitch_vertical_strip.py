#!/usr/bin/env python3
"""Build a UAV strip mosaic with constrained vertical translation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


@dataclass
class FrameInfo:
    path: str
    source: str
    time_seconds: float | None
    width: int
    height: int


@dataclass
class PairEstimate:
    previous: str
    current: str
    phase_dx: float
    phase_dy: float
    placement_dx: float
    placement_dy: float
    response: float
    low_response: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stitch a fixed-height UAV flight strip using constrained translation "
            "instead of free homography warping."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("extracted_frames"),
        help=(
            "Directory containing extracted frames, or raw videos if no images are "
            "present. Default: extracted_frames"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stitched_outputs/vertical_strip_mosaic.png"),
        help="Output mosaic path. Default: stitched_outputs/vertical_strip_mosaic.png",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path. Default: <output stem>_report.json",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=None,
        help=(
            "Directory for frames sampled from videos. "
            "Default: <output stem>_sampled_frames"
        ),
    )
    parser.add_argument(
        "--frame-interval-seconds",
        type=float,
        default=2.0,
        help="Sampling interval when --input-dir contains videos. Default: 2",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Base start time applied to every sampled video. Default: 0",
    )
    parser.add_argument(
        "--skip-first-start-seconds",
        type=float,
        default=0.0,
        help=(
            "Additional seconds to skip only at the start of the first video. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--skip-last-end-seconds",
        type=float,
        default=0.0,
        help=(
            "Seconds to discard only at the end of the last video. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Video sampling duration. Use 0 to continue to video end. Default: 0",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=80,
        help=(
            "Safety cap for direct video sampling. Use 0 for no cap. "
            "Default: 80"
        ),
    )
    parser.add_argument(
        "--axis",
        choices=("y", "both"),
        default="y",
        help="Allowed placement motion. 'y' forces dx=0. Default: y",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.18,
        help=(
            "Fraction cropped from each image edge before estimating motion. "
            "Default: 0.18"
        ),
    )
    parser.add_argument(
        "--work-scale",
        type=float,
        default=0.5,
        help="Scale used for motion estimation only. Default: 0.5",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=1.0,
        help=(
            "Scale used only when rendering the final mosaic. Motion is still "
            "estimated in original pixels. Default: 1.0"
        ),
    )
    parser.add_argument(
        "--smoothing",
        choices=("none", "median-step", "linear-path"),
        default="median-step",
        help="Smooth noisy per-frame motion estimates. Default: median-step",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Odd window size for median-step smoothing. Default: 5",
    )
    parser.add_argument(
        "--min-response",
        type=float,
        default=0.05,
        help="Warn when phase correlation response is below this value. Default: 0.05",
    )
    parser.add_argument(
        "--feather-radius",
        type=float,
        default=80.0,
        help="Blend width near image borders in pixels. Use 0 to disable. Default: 80",
    )
    parser.add_argument(
        "--max-canvas-pixels",
        type=int,
        default=600_000_000,
        help="Safety limit for output canvas pixels. Default: 600000000",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output/report files.",
    )
    return parser.parse_args()


def find_files(input_dir: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def image_info(path: Path, source: str, time_seconds: float | None) -> FrameInfo:
    image = read_image(path)
    height, width = image.shape[:2]
    return FrameInfo(
        path=str(path),
        source=source,
        time_seconds=time_seconds,
        width=width,
        height=height,
    )


def sample_video_frames(
    videos: list[Path],
    temp_dir: Path,
    interval_seconds: float,
    start_seconds: float,
    skip_first_start_seconds: float,
    skip_last_end_seconds: float,
    duration_seconds: float,
    max_frames_per_video: int,
) -> list[FrameInfo]:
    if interval_seconds <= 0:
        raise ValueError("--frame-interval-seconds must be greater than 0")
    if start_seconds < 0:
        raise ValueError("--start-seconds must be non-negative")
    if skip_first_start_seconds < 0:
        raise ValueError("--skip-first-start-seconds must be non-negative")
    if skip_last_end_seconds < 0:
        raise ValueError("--skip-last-end-seconds must be non-negative")
    if duration_seconds < 0:
        raise ValueError("--duration-seconds must be non-negative")
    if max_frames_per_video < 0:
        raise ValueError("--max-frames-per-video must be non-negative")

    frames: list[FrameInfo] = []
    video_count = len(videos)
    for video_index, video in enumerate(videos, start=1):
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0 or frame_count <= 0:
            capture.release()
            raise RuntimeError(f"Cannot read FPS/frame count for {video}")

        video_duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        video_start_seconds = start_seconds
        if video_index == 1:
            video_start_seconds += skip_first_start_seconds

        video_stop_seconds = video_duration
        if video_index == video_count:
            video_stop_seconds -= skip_last_end_seconds

        stop_seconds = (
            video_start_seconds + duration_seconds
            if duration_seconds > 0
            else video_stop_seconds
        )
        stop_seconds = min(stop_seconds, video_stop_seconds)
        if stop_seconds <= video_start_seconds:
            capture.release()
            raise RuntimeError(
                f"Cannot determine sampling duration for {video}. "
                f"start={video_start_seconds:.2f}s, stop={stop_seconds:.2f}s"
            )

        start_frame = max(0, int(np.ceil(video_start_seconds * fps)))
        stop_frame = min(int(frame_count) - 1, int(np.floor(stop_seconds * fps)))
        frame_step = max(1, int(round(interval_seconds * fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        sample_index = 0
        frame_index = start_frame
        while frame_index <= stop_frame:
            if max_frames_per_video and sample_index >= max_frames_per_video:
                break
            ok, frame = capture.read()
            if not ok:
                break

            time_seconds = frame_index / fps
            output_path = temp_dir / f"{video_index:02d}_{video.stem}_{sample_index:06d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                capture.release()
                raise RuntimeError(f"Failed to write sampled frame: {output_path}")
            height, width = frame.shape[:2]
            frames.append(
                FrameInfo(
                    path=str(output_path),
                    source=str(video),
                    time_seconds=float(time_seconds),
                    width=width,
                    height=height,
                )
            )
            sample_index += 1
            next_frame_index = frame_index + frame_step
            frames_to_skip = max(0, next_frame_index - frame_index - 1)
            skipped_all = True
            for _ in range(frames_to_skip):
                if not capture.grab():
                    skipped_all = False
                    break
            if not skipped_all:
                break
            frame_index = next_frame_index

        capture.release()
        print(
            f"Sampled {sample_index} frames from {video.name} "
            f"({video_start_seconds:.2f}s to {stop_seconds:.2f}s)",
            flush=True,
        )

    return frames


def load_frame_list(input_dir: Path, sample_dir: Path, args: argparse.Namespace) -> list[FrameInfo]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = find_files(input_dir, IMAGE_EXTENSIONS)
    if image_paths:
        return [image_info(path, str(path), None) for path in image_paths]

    video_paths = find_files(input_dir, VIDEO_EXTENSIONS)
    if not video_paths:
        raise FileNotFoundError(f"No images or videos found in: {input_dir}")

    if sample_dir.exists() and any(sample_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Sample directory is not empty: {sample_dir}. "
            "Use --overwrite or choose --sample-dir."
        )
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_video_frames(
        video_paths,
        sample_dir,
        args.frame_interval_seconds,
        args.start_seconds,
        args.skip_first_start_seconds,
        args.skip_last_end_seconds,
        args.duration_seconds,
        args.max_frames_per_video,
    )


def validate_frame_sizes(frames: list[FrameInfo]) -> tuple[int, int]:
    if len(frames) < 2:
        raise RuntimeError("Need at least 2 frames to build a mosaic")

    first_width = frames[0].width
    first_height = frames[0].height
    for frame in frames[1:]:
        if frame.width != first_width or frame.height != first_height:
            raise RuntimeError(
                "All frames must have the same size for this constrained stitcher: "
                f"first={first_width}x{first_height}, "
                f"{Path(frame.path).name}={frame.width}x{frame.height}"
            )
    return first_width, first_height


def preprocess_for_motion(
    image: np.ndarray,
    crop_margin: float,
    work_scale: float,
) -> np.ndarray:
    if not 0 <= crop_margin < 0.45:
        raise ValueError("--crop-margin must be in [0, 0.45)")
    if work_scale <= 0:
        raise ValueError("--work-scale must be greater than 0")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    margin_x = int(round(width * crop_margin))
    margin_y = int(round(height * crop_margin))
    cropped = gray[margin_y : height - margin_y, margin_x : width - margin_x]
    if cropped.size == 0:
        raise RuntimeError("Crop margin removed the whole image")

    if work_scale != 1.0:
        resized_width = max(8, int(round(cropped.shape[1] * work_scale)))
        resized_height = max(8, int(round(cropped.shape[0] * work_scale)))
        cropped = cv2.resize(
            cropped,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if work_scale < 1 else cv2.INTER_LINEAR,
        )

    signal = cropped.astype(np.float32)
    signal = cv2.GaussianBlur(signal, (0, 0), 1.2)
    signal -= float(signal.mean())
    std = float(signal.std())
    if std > 1e-6:
        signal /= std
    return signal


def estimate_pair_shift(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    previous = preprocess_for_motion(previous_image, args.crop_margin, args.work_scale)
    current = preprocess_for_motion(current_image, args.crop_margin, args.work_scale)
    if previous.shape != current.shape:
        raise RuntimeError("Motion-estimation crops have different sizes")

    window = cv2.createHanningWindow((previous.shape[1], previous.shape[0]), cv2.CV_32F)
    (dx_scaled, dy_scaled), response = cv2.phaseCorrelate(previous, current, window)
    return (
        float(dx_scaled / args.work_scale),
        float(dy_scaled / args.work_scale),
        float(response),
    )


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 2:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("--smooth-window must be odd")

    radius = window // 2
    smoothed = np.zeros_like(values)
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        smoothed[index] = np.median(values[start:stop], axis=0)
    return smoothed


def smooth_steps(steps: np.ndarray, smoothing: str, window: int) -> np.ndarray:
    if smoothing == "none":
        return steps.copy()
    if smoothing == "median-step":
        return rolling_median(steps, window)
    if smoothing == "linear-path":
        raw_positions = np.vstack(
            [np.zeros((1, 2), dtype=np.float64), np.cumsum(steps, axis=0)]
        )
        indices = np.arange(len(raw_positions), dtype=np.float64)
        fitted_positions = np.zeros_like(raw_positions)
        for axis_index in range(2):
            slope, intercept = np.polyfit(indices, raw_positions[:, axis_index], 1)
            fitted_positions[:, axis_index] = slope * indices + intercept
        fitted_positions -= fitted_positions[0]
        return np.diff(fitted_positions, axis=0)
    raise ValueError(f"Unsupported smoothing mode: {smoothing}")


def estimate_positions(
    frames: list[FrameInfo],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[PairEstimate]]:
    raw_steps: list[tuple[float, float]] = []
    estimates: list[PairEstimate] = []
    previous_image = read_image(Path(frames[0].path))

    for previous_info, current_info in zip(frames, frames[1:]):
        current_image = read_image(Path(current_info.path))
        phase_dx, phase_dy, response = estimate_pair_shift(
            previous_image,
            current_image,
            args,
        )

        placement_dx = -phase_dx
        placement_dy = -phase_dy
        if args.axis == "y":
            placement_dx = 0.0

        low_response = response < args.min_response
        if low_response:
            print(
                "Warning: low phase-correlation response "
                f"{response:.4f} for {Path(previous_info.path).name} -> "
                f"{Path(current_info.path).name}",
                file=sys.stderr,
                flush=True,
            )

        raw_steps.append((placement_dx, placement_dy))
        estimates.append(
            PairEstimate(
                previous=previous_info.path,
                current=current_info.path,
                phase_dx=phase_dx,
                phase_dy=phase_dy,
                placement_dx=placement_dx,
                placement_dy=placement_dy,
                response=response,
                low_response=low_response,
            )
        )
        previous_image = current_image
        print(
            f"Estimated {Path(current_info.path).name}: "
            f"phase=({phase_dx:.2f}, {phase_dy:.2f}), "
            f"place=({placement_dx:.2f}, {placement_dy:.2f}), "
            f"response={response:.4f}",
            flush=True,
        )

    steps = np.array(raw_steps, dtype=np.float64)
    smoothed_steps = smooth_steps(steps, args.smoothing, args.smooth_window)
    positions = np.vstack(
        [np.zeros((1, 2), dtype=np.float64), np.cumsum(smoothed_steps, axis=0)]
    )
    if args.axis == "y":
        positions[:, 0] = 0.0
    return positions, estimates


def compute_canvas(
    positions: np.ndarray,
    frame_width: int,
    frame_height: int,
    max_canvas_pixels: int,
    render_scale: float,
) -> tuple[np.ndarray, int, int]:
    if render_scale <= 0:
        raise ValueError("--render-scale must be greater than 0")

    min_x = float(np.floor(positions[:, 0].min()))
    min_y = float(np.floor(positions[:, 1].min()))
    max_x = float(np.ceil((positions[:, 0] + frame_width).max()))
    max_y = float(np.ceil((positions[:, 1] + frame_height).max()))
    canvas_width = int(max_x - min_x)
    canvas_height = int(max_y - min_y)
    render_width = max(1, int(round(canvas_width * render_scale)))
    render_height = max(1, int(round(canvas_height * render_scale)))
    canvas_pixels = render_width * render_height
    if canvas_width <= 0 or canvas_height <= 0:
        raise RuntimeError("Computed an invalid output canvas")
    if canvas_pixels > max_canvas_pixels:
        raise RuntimeError(
            f"Output canvas would be {render_width}x{render_height} "
            f"({canvas_pixels} pixels), above --max-canvas-pixels={max_canvas_pixels}"
        )

    offset = np.array([-min_x, -min_y], dtype=np.float64)
    return offset, canvas_width, canvas_height


def make_weight_mask(height: int, width: int, feather_radius: float) -> np.ndarray:
    valid = np.ones((height, width), dtype=np.uint8)
    if feather_radius <= 0:
        return valid.astype(np.float32)

    valid[0, :] = 0
    valid[-1, :] = 0
    valid[:, 0] = 0
    valid[:, -1] = 0
    distance = cv2.distanceTransform(valid, cv2.DIST_L2, 3)
    return np.clip(distance / feather_radius, 0.0, 1.0).astype(np.float32)


def render_mosaic(
    frames: list[FrameInfo],
    positions: np.ndarray,
    offset: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    feather_radius: float,
    render_scale: float,
) -> np.ndarray:
    if render_scale <= 0:
        raise ValueError("--render-scale must be greater than 0")

    render_width = max(1, int(round(canvas_width * render_scale)))
    render_height = max(1, int(round(canvas_height * render_scale)))
    scaled_positions = positions * render_scale
    scaled_offset = offset * render_scale
    scaled_feather_radius = feather_radius * render_scale

    accumulator = np.zeros((render_height, render_width, 3), dtype=np.float32)
    weights = np.zeros((render_height, render_width), dtype=np.float32)

    source_weight: np.ndarray | None = None
    source_shape: tuple[int, int] | None = None

    for index, (frame_info, position) in enumerate(zip(frames, scaled_positions), start=1):
        image = read_image(Path(frame_info.path))
        if render_scale != 1.0:
            image = cv2.resize(
                image,
                (
                    max(1, int(round(image.shape[1] * render_scale))),
                    max(1, int(round(image.shape[0] * render_scale))),
                ),
                interpolation=cv2.INTER_AREA if render_scale < 1 else cv2.INTER_LINEAR,
            )
        height, width = image.shape[:2]
        if source_shape != (height, width):
            source_weight = make_weight_mask(height, width, scaled_feather_radius)
            source_shape = (height, width)
        assert source_weight is not None

        tx = float(position[0] + scaled_offset[0])
        ty = float(position[1] + scaled_offset[1])
        matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
        warped_image = cv2.warpAffine(
            image,
            matrix,
            (render_width, render_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_weight = cv2.warpAffine(
            source_weight,
            matrix,
            (render_width, render_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        accumulator += warped_image.astype(np.float32) * warped_weight[:, :, None]
        weights += warped_weight
        print(f"Rendered {index}/{len(frames)}: {Path(frame_info.path).name}", flush=True)

    mosaic = np.zeros((render_height, render_width, 3), dtype=np.uint8)
    valid = weights > 1e-6
    mosaic[valid] = np.clip(accumulator[valid] / weights[valid, None], 0, 255).astype(np.uint8)
    return mosaic


def write_report(
    report_path: Path,
    frames: list[FrameInfo],
    estimates: list[PairEstimate],
    positions: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    render_width: int,
    render_height: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "input_dir": str(args.input_dir),
        "output": str(args.output),
        "canvas": {
            "width": render_width,
            "height": render_height,
            "scale": args.render_scale,
        },
        "source_canvas": {"width": canvas_width, "height": canvas_height},
        "settings": {
            "axis": args.axis,
            "crop_margin": args.crop_margin,
            "work_scale": args.work_scale,
            "render_scale": args.render_scale,
            "start_seconds": args.start_seconds,
            "skip_first_start_seconds": args.skip_first_start_seconds,
            "skip_last_end_seconds": args.skip_last_end_seconds,
            "frame_interval_seconds": args.frame_interval_seconds,
            "duration_seconds": args.duration_seconds,
            "max_frames_per_video": args.max_frames_per_video,
            "smoothing": args.smoothing,
            "smooth_window": args.smooth_window,
            "min_response": args.min_response,
            "feather_radius": args.feather_radius,
        },
        "frames": [
            {
                **asdict(frame),
                "position_x": float(position[0]),
                "position_y": float(position[1]),
            }
            for frame, position in zip(frames, positions)
        ],
        "pair_estimates": [asdict(estimate) for estimate in estimates],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        report_path = args.report
        if report_path is None:
            report_path = args.output.with_name(args.output.stem + "_report.json")
        sample_dir = args.sample_dir
        if sample_dir is None:
            sample_dir = args.output.with_name(args.output.stem + "_sampled_frames")

        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output}")
        if report_path.exists() and not args.overwrite:
            raise FileExistsError(f"Report already exists: {report_path}")

        frames = load_frame_list(args.input_dir, sample_dir, args)
        frame_width, frame_height = validate_frame_sizes(frames)
        print(f"Loaded {len(frames)} frames at {frame_width}x{frame_height}", flush=True)

        positions, estimates = estimate_positions(frames, args)
        offset, canvas_width, canvas_height = compute_canvas(
            positions,
            frame_width,
            frame_height,
            args.max_canvas_pixels,
            args.render_scale,
        )
        render_width = max(1, int(round(canvas_width * args.render_scale)))
        render_height = max(1, int(round(canvas_height * args.render_scale)))
        print(
            f"Source canvas: {canvas_width}x{canvas_height}; "
            f"render canvas: {render_width}x{render_height}",
            flush=True,
        )

        mosaic = render_mosaic(
            frames,
            positions,
            offset,
            canvas_width,
            canvas_height,
            args.feather_radius,
            args.render_scale,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), mosaic):
            raise RuntimeError(f"Failed to write output image: {args.output}")
        render_height, render_width = mosaic.shape[:2]
        write_report(
            report_path,
            frames,
            estimates,
            positions,
            canvas_width,
            canvas_height,
            render_width,
            render_height,
            args,
        )

        print(f"Wrote mosaic: {args.output}", flush=True)
        print(f"Wrote report: {report_path}", flush=True)
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
