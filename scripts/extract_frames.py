#!/usr/bin/env python3
"""Extract frames from mission videos at a fixed interval."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one image every N seconds from videos in filename order, "
            "saving all frames into one sequentially numbered folder."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("raw_videos"),
        help="Directory containing mission videos. Default: raw_videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extracted_frames"),
        help="Directory where extracted images will be saved. Default: extracted_frames",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Seconds between extracted frames. Default: 3",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Output image filename prefix. Default: frame",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def find_videos(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def ensure_ready(args: argparse.Namespace) -> None:
    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found in PATH")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. "
            "Use --overwrite or choose another --output-dir."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)


def extract_video(video_path: Path, temp_dir: Path, interval: float) -> list[Path]:
    video_temp_dir = temp_dir / video_path.stem
    video_temp_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = video_temp_dir / "%06d.jpg"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)
    return sorted(video_temp_dir.glob("*.jpg"))


def main() -> int:
    args = parse_args()

    try:
        ensure_ready(args)
        videos = find_videos(args.input_dir)
        if not videos:
            raise FileNotFoundError(f"No videos found in: {args.input_dir}")

        frame_index = 1
        with tempfile.TemporaryDirectory(prefix="frame_extract_") as temp_root:
            temp_dir = Path(temp_root)
            for video in videos:
                extracted_frames = extract_video(video, temp_dir, args.interval)
                for frame_path in extracted_frames:
                    destination = args.output_dir / f"{args.prefix}_{frame_index:06d}.jpg"
                    shutil.move(str(frame_path), destination)
                    frame_index += 1

        print(
            f"Extracted {frame_index - 1} frames from {len(videos)} videos "
            f"into {args.output_dir}"
        )
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"Error: ffmpeg failed with exit code {error.returncode}", file=sys.stderr)
        return error.returncode


if __name__ == "__main__":
    raise SystemExit(main())
