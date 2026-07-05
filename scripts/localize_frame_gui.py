#!/usr/bin/env python3
"""GUI for dropping a frame and visualizing vehicle localization output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, DISABLED, END, LEFT, NORMAL, RIGHT, TOP, X, Button, Frame, Label, StringVar, Text, filedialog, messagebox, ttk

from PIL import Image, ImageTk


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "frame_gui_outputs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = None
    TkinterDnD = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a GUI, drop/select one frame, run localize_vehicles.py, and show the result."
    )
    parser.add_argument("--frame", type=Path, default=None, help="Optional frame to load at startup.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=950)
    parser.add_argument("--detector", default="yolo", choices=("auto", "yolo", "white-heuristic"))
    parser.add_argument("--localize-device", default="cuda:0")
    parser.add_argument("--localize-model", default="yolo26x.pt")
    parser.add_argument("--localize-vehicle-classes", default="car")
    parser.add_argument("--localize-imgsz", type=int, default=1600)
    parser.add_argument("--localize-tile-upscales", default="1,4")
    parser.add_argument("--localize-yolo-batch-size", type=int, default=16)
    parser.add_argument("--localize-conf", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=960)
    parser.add_argument("--tile-overlap", type=int, default=240)
    parser.add_argument("--max-detections", type=int, default=40)
    parser.add_argument("--orientations", default="all", choices=("none", "rotations", "flips", "all"))
    parser.add_argument("--match-workers", type=int, default=4)
    parser.add_argument("--feature-max-dim", type=int, default=1200)
    parser.add_argument("--target-verifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-verifier-min-score", type=float, default=0.18)
    parser.add_argument("--target-verifier-min-white-ratio", type=float, default=0.15)
    parser.add_argument("--target-verifier-min-red-pixels", type=int, default=35)
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_root():
    if TkinterDnD is not None:
        return TkinterDnD.Tk()
    import tkinter as tk

    return tk.Tk()


def safe_output_name(frame_path: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{frame_path.stem}_{timestamp}"


class FrameLocalizationGui:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.args.output_root = self.args.output_root.expanduser().resolve()
        self.root = make_root()
        self.root.title("UAV Frame Localization")
        self.root.geometry(f"{args.window_width}x{args.window_height}")
        self.root.minsize(1100, 700)

        self.frame_path: Path | None = None
        self.current_image_ref: ImageTk.PhotoImage | None = None
        self.running = False

        self.status_text = StringVar(value="Drop a frame here, or click Open Frame.")
        self.frame_text = StringVar(value="No frame selected")
        self.output_text = StringVar(value=f"Output root: {self.args.output_root}")

        self._build_ui()
        if args.frame is not None:
            self.load_frame(args.frame)

    def _build_ui(self) -> None:
        toolbar = Frame(self.root)
        toolbar.pack(side=TOP, fill=X, padx=10, pady=8)

        Button(toolbar, text="Open Frame", command=self.open_frame).pack(side=LEFT, padx=(0, 8))
        self.run_button = Button(toolbar, text="Run Detect", command=self.run_detection)
        self.run_button.pack(side=LEFT, padx=(0, 8))
        Button(toolbar, text="Open Output Folder", command=self.open_output_folder).pack(side=LEFT)

        Label(toolbar, textvariable=self.status_text, anchor="w").pack(side=LEFT, fill=X, expand=True, padx=12)

        info = Frame(self.root)
        info.pack(side=TOP, fill=X, padx=10)
        Label(info, textvariable=self.frame_text, anchor="w").pack(side=TOP, fill=X)
        Label(info, textvariable=self.output_text, anchor="w").pack(side=TOP, fill=X)

        body = Frame(self.root)
        body.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=8)

        left = Frame(body)
        left.pack(side=LEFT, fill=BOTH, expand=True)
        right = Frame(body, width=420)
        right.pack(side=RIGHT, fill=BOTH, padx=(10, 0))

        self.drop_label = Label(
            left,
            text="Drop frame here\nor use Open Frame",
            relief="groove",
            background="#202020",
            foreground="#f5f5f5",
            anchor="center",
            justify="center",
        )
        self.drop_label.pack(fill=BOTH, expand=True)

        if TkinterDnD is not None and DND_FILES is not None:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self.handle_drop)
        else:
            self.status_text.set("Drag-and-drop needs tkinterdnd2; Open Frame button is available.")

        Label(right, text="Run Log", anchor="w").pack(side=TOP, fill=X)
        self.log = Text(right, width=54, height=24, wrap="word")
        self.log.pack(side=TOP, fill=BOTH, expand=True)

        params = ttk.LabelFrame(right, text="Active Parameters")
        params.pack(side=TOP, fill=X, pady=(8, 0))
        Label(params, text=self.parameter_summary(), justify="left", anchor="w").pack(fill=X, padx=8, pady=6)

    def parameter_summary(self) -> str:
        return "\n".join(
            [
                f"device: {self.args.localize_device}",
                f"model: {self.args.localize_model}",
                f"classes: {self.args.localize_vehicle_classes}",
                f"imgsz: {self.args.localize_imgsz}",
                f"tile-upscales: {self.args.localize_tile_upscales}",
                f"batch-size: {self.args.localize_yolo_batch_size}",
                f"conf: {self.args.localize_conf}",
                f"target-verifier: {self.args.target_verifier}",
            ]
        )

    def handle_drop(self, event) -> None:
        paths = self.root.tk.splitlist(event.data)
        if paths:
            self.load_frame(Path(paths[0]))

    def open_frame(self) -> None:
        filename = filedialog.askopenfilename(
            title="Choose frame",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.load_frame(Path(filename))

    def open_output_folder(self) -> None:
        self.args.output_root.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["xdg-open", str(self.args.output_root)])

    def load_frame(self, frame_path: Path) -> None:
        frame_path = frame_path.expanduser().resolve()
        if not frame_path.is_file():
            messagebox.showerror("Invalid frame", f"File not found:\n{frame_path}")
            return
        if frame_path.suffix.lower() not in IMAGE_EXTENSIONS:
            messagebox.showerror("Invalid frame", f"Unsupported image type:\n{frame_path}")
            return
        self.frame_path = frame_path
        self.frame_text.set(f"Frame: {frame_path}")
        self.status_text.set("Frame loaded. Click Run Detect.")
        self.show_image(frame_path)

    def show_image(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        self.drop_label.update_idletasks()
        max_width = max(500, self.drop_label.winfo_width() - 24)
        max_height = max(400, self.drop_label.winfo_height() - 24)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.current_image_ref = ImageTk.PhotoImage(image)
        self.drop_label.configure(image=self.current_image_ref, text="")

    def append_log(self, text: str) -> None:
        self.log.insert(END, text)
        self.log.see(END)

    def set_running(self, running: bool) -> None:
        self.running = running
        self.run_button.configure(state=DISABLED if running else NORMAL)

    def run_detection(self) -> None:
        if self.running:
            return
        if self.frame_path is None:
            messagebox.showinfo("No frame", "Drop or open a frame first.")
            return
        output_dir = self.args.output_root / safe_output_name(self.frame_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_text.set(f"Output: {output_dir}")
        self.status_text.set("Running localization...")
        self.set_running(True)
        self.append_log(f"\nRunning frame: {self.frame_path}\nOutput: {output_dir}\n")
        thread = threading.Thread(target=self._run_detection_worker, args=(self.frame_path, output_dir), daemon=True)
        thread.start()

    def _run_detection_worker(self, frame_path: Path, output_dir: Path) -> None:
        command = self.localize_command(frame_path, output_dir)
        result = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
        (output_dir / "run.stdout.log").write_text(result.stdout, encoding="utf-8")
        (output_dir / "run.stderr.log").write_text(result.stderr, encoding="utf-8")
        self.root.after(0, self._detection_finished, result.returncode, output_dir, result.stderr)

    def localize_command(self, frame_path: Path, output_dir: Path) -> list[str]:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "localize_vehicles.py"),
            "--frame",
            str(frame_path),
            "--output-dir",
            str(output_dir),
            "--detector",
            self.args.detector,
            "--yolo-model",
            self.args.localize_model,
            "--vehicle-classes",
            self.args.localize_vehicle_classes,
            "--imgsz",
            str(self.args.localize_imgsz),
            "--tile-size",
            str(self.args.tile_size),
            "--tile-overlap",
            str(self.args.tile_overlap),
            "--tile-upscales",
            self.args.localize_tile_upscales,
            "--yolo-batch-size",
            str(self.args.localize_yolo_batch_size),
            "--conf",
            str(self.args.localize_conf),
            "--max-detections",
            str(self.args.max_detections),
            "--orientations",
            self.args.orientations,
            "--match-workers",
            str(self.args.match_workers),
            "--feature-max-dim",
            str(self.args.feature_max_dim),
        ]
        if self.args.localize_device:
            command.extend(["--device", self.args.localize_device])
        if self.args.target_verifier:
            command.extend(
                [
                    "--target-verifier",
                    "--target-verifier-min-score",
                    str(self.args.target_verifier_min_score),
                    "--target-verifier-min-white-ratio",
                    str(self.args.target_verifier_min_white_ratio),
                    "--target-verifier-min-red-pixels",
                    str(self.args.target_verifier_min_red_pixels),
                ]
            )
        if not self.args.save_crops:
            command.append("--no-save-crops")
        return command

    def _detection_finished(self, returncode: int, output_dir: Path, stderr: str) -> None:
        self.set_running(False)
        if returncode != 0:
            self.status_text.set("Localization failed.")
            self.append_log(stderr.strip() + "\n")
            messagebox.showerror("Localization failed", stderr.strip() or f"Return code: {returncode}")
            return
        self.status_text.set("Localization complete.")
        self.append_log(self.summary_text(output_dir) + "\n")
        result_image = output_dir / "03_process_overview.jpg"
        if result_image.is_file():
            self.show_image(result_image)
        else:
            self.append_log(f"Missing result image: {result_image}\n")

    def summary_text(self, output_dir: Path) -> str:
        report_path = output_dir / "vehicle_localization.json"
        if not report_path.is_file():
            return f"Done. Output: {output_dir}"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return f"Done. Could not parse report: {exc}"
        verifier = report.get("target_verifier", {})
        timing = report.get("timings_seconds", {})
        return (
            f"Done. detections={report.get('detections_count', '')}, "
            f"kept={verifier.get('kept_count', '')}, "
            f"rejected={verifier.get('rejected_count', '')}, "
            f"total={timing.get('total', '')}s\nOutput: {output_dir}"
        )

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    args = parse_args()
    app = FrameLocalizationGui(args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
