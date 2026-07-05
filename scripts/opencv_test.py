#!/usr/bin/env python3
"""Display a video source, record it, save periodic frames, and trigger localization."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "protocol_whitelist;file,udp,rtp|"
    "fflags;nobuffer|"
    "flags;low_delay"
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "stream_outputs"
DEFAULT_SOURCE = "stream.sdp"
cv2 = None


def load_cv2():
    global cv2
    if cv2 is None:
        import cv2 as cv2_module

        cv2 = cv2_module
    return cv2


@dataclass
class ResolvedSource:
    value: str | int
    label: str
    uses_ffmpeg: bool


@dataclass
class Button:
    action: str
    label: str
    rect: tuple[int, int, int, int]
    enabled: bool = True


class SegmentRecorder:
    def __init__(self, output_dir: Path, fps: float, segment_seconds: float) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self.segment_seconds = segment_seconds
        self.writer: cv2.VideoWriter | None = None
        self.segment_started = 0.0
        self.segment_index = 0
        self.frame_size: tuple[int, int] | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame) -> None:
        now = time.monotonic()
        height, width = frame.shape[:2]
        frame_size = (width, height)
        should_rotate = (
            self.writer is None
            or self.frame_size != frame_size
            or (
                self.segment_seconds > 0
                and now - self.segment_started >= self.segment_seconds
            )
        )
        if should_rotate:
            self._open_next_segment(frame_size, now)
        if self.writer is not None:
            self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def _open_next_segment(self, frame_size: tuple[int, int], started: float) -> None:
        self.release()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"recording_{timestamp}_part{self.segment_index:03d}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, frame_size)
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {path}")
        self.writer = writer
        self.frame_size = frame_size
        self.segment_started = started
        self.segment_index += 1
        print(f"Recording segment: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open stream.sdp, a webcam, or a video file; keep recording, save periodic "
            "frames, and use on-screen buttons to run scripts/localize_vehicles.py on the current frame."
        )
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=(
            "Video source. Use stream.sdp, a video path, 0, webcam:0, or a URL. "
            "Default: stream.sdp"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "ffmpeg", "default"),
        default="auto",
        help="OpenCV backend. Auto uses FFmpeg for SDP/files and default backend for webcams.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for recordings, periodic frames, manual frames, and localization results.",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=2.0,
        help="Seconds between automatic saved frames. Use 0 to disable. Default: 2.0",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=30.0,
        help="FPS used for output recordings when the source FPS is unavailable. Default: 30",
    )
    parser.add_argument(
        "--record-segment-seconds",
        type=float,
        default=300.0,
        help=(
            "Rotate recording files every N seconds to reduce data loss if the process stops "
            "unexpectedly. Use 0 for one continuous file. Default: 300"
        ),
    )
    parser.add_argument(
        "--window-name",
        default="UAV stream",
        help="OpenCV display window name.",
    )
    parser.add_argument(
        "--localize-device",
        default=None,
        help="Optional device passed to localize_vehicles.py, e.g. mps, cpu, cuda:0.",
    )
    parser.add_argument(
        "--localize-model",
        default="yolo26x.pt",
        help="YOLO model passed to localize_vehicles.py. Default: yolo26x.pt",
    )
    parser.add_argument(
        "--localize-vehicle-classes",
        default="car",
        help="Vehicle classes passed to localize_vehicles.py. Default: car",
    )
    parser.add_argument(
        "--localize-imgsz",
        type=int,
        default=1280,
        help="YOLO image size passed to localize_vehicles.py. Default: 1280",
    )
    parser.add_argument(
        "--localize-tile-upscales",
        default="1,2",
        help="Tile upscales passed to localize_vehicles.py. Default: 1,2",
    )
    parser.add_argument(
        "--localize-yolo-batch-size",
        type=int,
        default=4,
        help="YOLO batch size passed to localize_vehicles.py. Default: 4",
    )
    parser.add_argument(
        "--localize-conf",
        type=float,
        default=0.12,
        help="YOLO confidence threshold passed to localize_vehicles.py. Default: 0.12",
    )
    parser.add_argument(
        "--localize-output-root",
        type=Path,
        default=None,
        help="Output root for shortcut localization. Default: <output-root>/detections",
    )
    parser.add_argument(
        "--target-verifier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass --target-verifier to scripts/localize_vehicles.py when Detect is clicked.",
    )
    parser.add_argument(
        "--target-verifier-min-score",
        type=float,
        default=0.18,
        help="Target verifier score threshold passed to scripts/localize_vehicles.py.",
    )
    parser.add_argument(
        "--target-verifier-min-white-ratio",
        type=float,
        default=0.15,
        help="Target verifier white vehicle threshold passed to scripts/localize_vehicles.py.",
    )
    parser.add_argument(
        "--target-verifier-min-red-pixels",
        type=int,
        default=35,
        help="Target verifier red marker pixel threshold passed to scripts/localize_vehicles.py.",
    )
    return parser.parse_args()


def resolve_source(raw_source: str) -> ResolvedSource:
    source = raw_source.strip()
    lower_source = source.lower()
    if lower_source.startswith("webcam:"):
        return ResolvedSource(int(source.split(":", 1)[1]), f"webcam:{source.split(':', 1)[1]}", False)
    if source.isdecimal():
        return ResolvedSource(int(source), f"webcam:{source}", False)
    if "://" in source:
        return ResolvedSource(source, source, True)

    path = Path(source).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            return ResolvedSource(str(resolved), str(resolved), True)

    if path.suffix or "/" in source:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Cannot find video source '{source}'. Searched: {searched}")

    return ResolvedSource(source, source, True)


def open_capture(source: ResolvedSource, backend: str) -> cv2.VideoCapture:
    if backend == "ffmpeg" or (backend == "auto" and source.uses_ffmpeg):
        cap = cv2.VideoCapture(source.value, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source.value)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source.label}")
    return cap


def source_fps(cap: cv2.VideoCapture, fallback_fps: float) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 1:
        return float(fps)
    return fallback_fps


def timestamped_frame_path(output_dir: Path, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return output_dir / f"{prefix}_{timestamp}.jpg"


def save_frame(frame, output_dir: Path, prefix: str) -> Path:
    path = timestamped_frame_path(output_dir, prefix)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Cannot write frame: {path}")
    return path


def start_localization(frame_path: Path, args: argparse.Namespace) -> subprocess.Popen:
    output_root = args.localize_output_root or (args.output_root / "detections")
    output_dir = output_root / frame_path.stem
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "localize_vehicles.py"),
        "--frame",
        str(frame_path),
        "--output-dir",
        str(output_dir),
        "--yolo-model",
        args.localize_model,
        "--vehicle-classes",
        args.localize_vehicle_classes,
        "--imgsz",
        str(args.localize_imgsz),
        "--tile-upscales",
        args.localize_tile_upscales,
        "--yolo-batch-size",
        str(args.localize_yolo_batch_size),
        "--conf",
        str(args.localize_conf),
    ]
    if args.localize_device:
        command.extend(["--device", args.localize_device])
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
    print(f"Running localization: {frame_path}")
    print(f"Localization output: {output_dir}")
    return subprocess.Popen(command, cwd=str(REPO_ROOT))


def reap_finished_job(job: subprocess.Popen | None) -> subprocess.Popen | None:
    if job is None:
        return None
    returncode = job.poll()
    if returncode is None:
        return job
    print(f"Localization job finished with return code {returncode}")
    return None


def handle_button_click(x: int, y: int, buttons: list[Button]) -> str | None:
    for button in buttons:
        x1, y1, x2, y2 = button.rect
        if button.enabled and x1 <= x <= x2 and y1 <= y <= y2:
            return button.action
    return None


def make_buttons(width: int, detection_running: bool) -> list[Button]:
    button_width = max(150, min(220, width // 5))
    button_height = 46
    gap = 12
    x = 16
    y = 14
    buttons = [
        Button("save", "Save Frame", (x, y, x + button_width, y + button_height)),
        Button(
            "detect",
            "Detect Running" if detection_running else "Detect",
            (x + button_width + gap, y, x + button_width * 2 + gap, y + button_height),
            enabled=not detection_running,
        ),
        Button(
            "quit",
            "Quit",
            (x + (button_width + gap) * 2, y, x + button_width * 3 + gap * 2, y + button_height),
        ),
    ]
    return buttons


def draw_button(output, button: Button) -> None:
    if not button.enabled:
        fill = (95, 95, 95)
        border = (130, 130, 130)
        text = (215, 215, 215)
    elif button.action == "detect":
        fill = (30, 110, 210)
        border = (210, 230, 255)
        text = (255, 255, 255)
    elif button.action == "quit":
        fill = (40, 40, 40)
        border = (210, 210, 210)
        text = (255, 255, 255)
    else:
        fill = (245, 245, 245)
        border = (80, 80, 80)
        text = (20, 20, 20)

    x1, y1, x2, y2 = button.rect
    cv2.rectangle(output, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
    cv2.rectangle(output, (x1, y1), (x2, y2), border, 2, cv2.LINE_AA)
    text_size, _baseline = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, 0.66, 2)
    text_x = x1 + max(8, (x2 - x1 - text_size[0]) // 2)
    text_y = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(output, button.label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, text, 2, cv2.LINE_AA)


def draw_status(frame, source_label: str, detection_running: bool) -> tuple[object, list[Button]]:
    output = frame.copy()
    buttons = make_buttons(output.shape[1], detection_running)
    panel_height = 78
    cv2.rectangle(output, (0, 0), (output.shape[1], panel_height), (25, 25, 25), -1)
    for button in buttons:
        draw_button(output, button)
    status = f"source: {source_label}"
    if detection_running:
        status += " | localization running"
    cv2.putText(output, status, (16, panel_height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(output, status, (16, panel_height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return output, buttons


def main() -> int:
    args = parse_args()
    load_cv2()
    args.output_root = args.output_root.expanduser().resolve()
    if args.localize_output_root is not None:
        args.localize_output_root = args.localize_output_root.expanduser().resolve()

    source = resolve_source(args.source)
    cap = open_capture(source, args.backend)
    fps = source_fps(cap, args.record_fps)
    recorder = SegmentRecorder(
        args.output_root / "recordings",
        fps=fps,
        segment_seconds=args.record_segment_seconds,
    )
    periodic_dir = args.output_root / "frames"
    manual_dir = args.output_root / "manual_frames"
    localization_job: subprocess.Popen | None = None
    running = True
    last_periodic_save = 0.0
    current_buttons: list[Button] = []
    requested_action: str | None = None

    def stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal requested_action
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        requested_action = handle_button_click(x, y, current_buttons)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(args.window_name, on_mouse)

    print(f"Opened source: {source.label}")
    print(f"Recording FPS: {fps:.2f}")
    print(f"Outputs: {args.output_root}")
    print("Buttons: Save Frame, Detect, Quit")

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                print("No frame received")
                time.sleep(0.05)
                localization_job = reap_finished_job(localization_job)
                continue

            recorder.write(frame)

            now = time.monotonic()
            if args.frame_interval > 0 and now - last_periodic_save >= args.frame_interval:
                path = save_frame(frame, periodic_dir, "frame")
                print(f"Saved periodic frame: {path}")
                last_periodic_save = now

            localization_job = reap_finished_job(localization_job)
            detection_running = localization_job is not None
            display_frame, current_buttons = draw_status(frame, source.label, detection_running)
            cv2.imshow(args.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            action = requested_action
            requested_action = None
            if action == "quit":
                break
            if action == "save":
                path = save_frame(frame, manual_dir, "manual")
                print(f"Saved manual frame: {path}")
            if action == "detect" and localization_job is None:
                path = save_frame(frame, manual_dir, "detect")
                localization_job = start_localization(path, args)
    finally:
        cap.release()
        recorder.release()
        cv2.destroyAllWindows()
        if localization_job is not None and localization_job.poll() is None:
            print(f"Localization job still running: pid={localization_job.pid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
