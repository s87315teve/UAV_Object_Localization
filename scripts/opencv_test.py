#!/usr/bin/env python3
"""Display a video source, record it, save periodic frames, and trigger localization."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
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
LOOPABLE_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}
TELEMETRY_DISPLAY_INTERVAL_SECONDS = 0.5
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


@dataclass
class LocalizationJob:
    process: subprocess.Popen
    frame_path: Path
    output_dir: Path


@dataclass
class TelemetrySnapshot:
    altitude_m: float | None = None
    relative_altitude_m: float | None = None
    battery_voltage_v: float | None = None
    timestamp: str | None = None
    sequence: int | None = None


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
        "--loop-source",
        action="store_true",
        help=(
            "When the source is a local video file, restart it from the beginning "
            "after EOF. Use Quit, Esc, or Ctrl-C to stop."
        ),
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record the displayed source under <output-root>/recordings. Default: disabled.",
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
        "--window-width",
        type=int,
        default=1600,
        help="Initial display window width in pixels. Default: 1600",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=900,
        help="Initial display window height in pixels. Default: 900",
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
        "--show-detection-result",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show 03_process_overview.jpg in a separate OpenCV window after a Detect "
            "localization job finishes. Default: enabled."
        ),
    )
    parser.add_argument(
        "--result-window-name",
        default="UAV localization result",
        help="OpenCV window name for the latest Detect result.",
    )
    parser.add_argument(
        "--telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Listen for UAV telemetry UDP JSON packets and show them below the image. Default: enabled.",
    )
    parser.add_argument(
        "--telemetry-host",
        default="0.0.0.0",
        help="Local IP address to bind for telemetry UDP packets. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--telemetry-port",
        type=int,
        default=6001,
        help="Local UDP port for telemetry JSON packets. Default: 6001",
    )
    parser.add_argument(
        "--require-telemetry",
        action="store_true",
        help="Exit immediately if telemetry UDP binding fails.",
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


def is_loopable_video_file(source: ResolvedSource) -> bool:
    if not isinstance(source.value, str):
        return False
    path = Path(source.value)
    return path.is_file() and path.suffix.lower() in LOOPABLE_VIDEO_EXTENSIONS


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


def start_localization(frame_path: Path, args: argparse.Namespace) -> LocalizationJob:
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
    process = subprocess.Popen(command, cwd=str(REPO_ROOT))
    return LocalizationJob(process=process, frame_path=frame_path, output_dir=output_dir)


def localization_result_image(output_dir: Path) -> Path | None:
    for name in ("03_process_overview.jpg", "01_frame_vehicle_detections.jpg", "02_map_vehicle_coordinates.jpg"):
        path = output_dir / name
        if path.is_file():
            return path
    return None


def show_localization_result(job: LocalizationJob, args: argparse.Namespace) -> None:
    if not args.show_detection_result:
        return
    result_path = localization_result_image(job.output_dir)
    if result_path is None:
        print(f"No localization result image found under: {job.output_dir}")
        return
    image = cv2.imread(str(result_path))
    if image is None:
        print(f"Cannot read localization result image: {result_path}")
        return
    cv2.namedWindow(args.result_window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(args.result_window_name, image)
    cv2.waitKey(1)
    print(f"Showing localization result: {result_path}")


def reap_finished_job(job: LocalizationJob | None, args: argparse.Namespace) -> LocalizationJob | None:
    if job is None:
        return None
    returncode = job.process.poll()
    if returncode is None:
        return job
    print(f"Localization job finished with return code {returncode}")
    if returncode == 0:
        show_localization_result(job, args)
    return None


def handle_button_click(x: int, y: int, buttons: list[Button]) -> str | None:
    for button in buttons:
        x1, y1, x2, y2 = button.rect
        if button.enabled and x1 <= x <= x2 and y1 <= y <= y2:
            return button.action
    return None


class TelemetryUdpReceiver:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.snapshot = TelemetrySnapshot()
        self.sock: socket.socket | None = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.packet_count = 0

    def start(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.host, self.port))
            sock.settimeout(0.2)
        except OSError as exc:
            print(f"Telemetry disabled: cannot bind UDP {self.host}:{self.port}: {exc}")
            self.close()
            return
        self.sock = sock
        print(f"Listening for telemetry UDP on {self.host}:{self.port}")
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, name="telemetry-udp", daemon=True)
        self.thread.start()

    def is_active(self) -> bool:
        return self.sock is not None and self.running

    def receive_latest(self) -> TelemetrySnapshot:
        with self.lock:
            return self.snapshot

    def _receive_loop(self) -> None:
        if self.sock is None:
            return

        while self.running:
            try:
                data, address = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError as exc:
                if self.running:
                    print(f"Telemetry disabled: UDP receive failed: {exc}")
                break

            try:
                packet = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                print(f"Invalid telemetry packet from {address[0]}:{address[1]}: {exc}")
                continue

            snapshot = TelemetrySnapshot(
                altitude_m=packet.get("altitude_m"),
                relative_altitude_m=packet.get("relative_altitude_m"),
                battery_voltage_v=packet.get("battery_voltage_v"),
                timestamp=packet.get("timestamp"),
                sequence=packet.get("sequence"),
            )
            with self.lock:
                self.snapshot = snapshot
                self.packet_count += 1
                packet_count = self.packet_count

            if packet_count <= 3:
                print(
                    f"Telemetry received from {address[0]}:{address[1]}: "
                    f"Altitude={format_telemetry_value(snapshot.altitude_m, 'm')}, "
                    f"Relative Altitude={format_telemetry_value(snapshot.relative_altitude_m, 'm')}, "
                    f"Battery Voltage={format_telemetry_value(snapshot.battery_voltage_v, 'V')}"
                )

    def close(self) -> None:
        self.running = False
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.thread = None


def make_buttons(width: int, y: int, detection_running: bool) -> list[Button]:
    button_width = max(150, min(220, width // 5))
    button_height = 46
    gap = 12
    x = 16
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


def format_telemetry_value(value: float | int | None, unit: str) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f} {unit}"
    return "None"


def draw_status(frame, detection_running: bool, telemetry: TelemetrySnapshot) -> tuple[object, list[Button]]:
    panel_height = 116
    output = cv2.copyMakeBorder(
        frame,
        0,
        panel_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )
    frame_height = frame.shape[0]
    buttons = make_buttons(output.shape[1], frame_height + 14, detection_running)
    for button in buttons:
        draw_button(output, button)

    telemetry_text = (
        f"Telemetry | Altitude: {format_telemetry_value(telemetry.altitude_m, 'm')} | "
        f"Relative Altitude: {format_telemetry_value(telemetry.relative_altitude_m, 'm')} | "
        f"Battery Voltage: {format_telemetry_value(telemetry.battery_voltage_v, 'V')}"
    )
    text_x = min(output.shape[1] - 16, buttons[-1].rect[2] + 24)
    text_y = frame_height + 34
    font_scale = 0.66
    text_size, _baseline = cv2.getTextSize(telemetry_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    if text_x + text_size[0] > output.shape[1] - 16:
        text_x = 16
        text_y = frame_height + 100
        available_width = output.shape[1] - 32
        if text_size[0] > available_width:
            font_scale = max(0.45, font_scale * available_width / text_size[0])
    cv2.putText(output, telemetry_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return output, buttons


def main() -> int:
    args = parse_args()
    load_cv2()
    if args.window_width <= 0 or args.window_height <= 0:
        raise ValueError("--window-width and --window-height must be greater than 0")
    args.output_root = args.output_root.expanduser().resolve()
    if args.localize_output_root is not None:
        args.localize_output_root = args.localize_output_root.expanduser().resolve()

    source = resolve_source(args.source)
    cap = open_capture(source, args.backend)
    loop_enabled = args.loop_source and is_loopable_video_file(source)
    if args.loop_source and not loop_enabled:
        print("--loop-source is only supported for local video files; continuing without looping.")
    fps = source_fps(cap, args.record_fps)
    recorder = (
        SegmentRecorder(
            args.output_root / "recordings",
            fps=fps,
            segment_seconds=args.record_segment_seconds,
        )
        if args.record
        else None
    )
    periodic_dir = args.output_root / "frames"
    manual_dir = args.output_root / "manual_frames"
    localization_job: LocalizationJob | None = None
    telemetry_receiver = TelemetryUdpReceiver(args.telemetry_host, args.telemetry_port) if args.telemetry else None
    if telemetry_receiver is not None:
        telemetry_receiver.start()
        if args.require_telemetry and not telemetry_receiver.is_active():
            raise RuntimeError(
                f"Telemetry is required, but UDP {args.telemetry_host}:{args.telemetry_port} is not available."
            )
    displayed_telemetry = TelemetrySnapshot()
    last_telemetry_display_update = 0.0
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
    cv2.resizeWindow(args.window_name, args.window_width, args.window_height)
    cv2.setMouseCallback(args.window_name, on_mouse)

    print(f"Opened source: {source.label}")
    if args.record:
        print(f"Recording FPS: {fps:.2f}")
    else:
        print("Recording disabled")
    print(f"Outputs: {args.output_root}")
    print("Buttons: Save Frame, Detect, Quit")

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                if loop_enabled:
                    print("Reached end of video; restarting source.")
                    cap.release()
                    cap = open_capture(source, args.backend)
                    localization_job = reap_finished_job(localization_job, args)
                    continue
                print("No frame received")
                time.sleep(0.05)
                localization_job = reap_finished_job(localization_job, args)
                continue

            if recorder is not None:
                recorder.write(frame)

            now = time.monotonic()
            if args.frame_interval > 0 and now - last_periodic_save >= args.frame_interval:
                path = save_frame(frame, periodic_dir, "frame")
                print(f"Saved periodic frame: {path}")
                last_periodic_save = now

            localization_job = reap_finished_job(localization_job, args)
            detection_running = localization_job is not None
            latest_telemetry = telemetry_receiver.receive_latest() if telemetry_receiver is not None else TelemetrySnapshot()
            if now - last_telemetry_display_update >= TELEMETRY_DISPLAY_INTERVAL_SECONDS:
                displayed_telemetry = latest_telemetry
                last_telemetry_display_update = now
            display_frame, current_buttons = draw_status(frame, detection_running, displayed_telemetry)
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
        if recorder is not None:
            recorder.release()
        if telemetry_receiver is not None:
            telemetry_receiver.close()
        cv2.destroyAllWindows()
        if localization_job is not None and localization_job.process.poll() is None:
            print(f"Localization job still running: pid={localization_job.process.pid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
