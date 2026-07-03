#!/usr/bin/env python3
"""Interactively adjust strip-mosaic leg offsets and save a corrected mosaic."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stitch_vertical_strip import (
    compute_canvas,
    make_weight_mask,
    read_image,
    render_mosaic,
)


LEFT_KEYS = {ord("h"), 2424832, 65361, 81}
RIGHT_KEYS = {ord("l"), 2555904, 65363, 83}
UP_KEYS = {ord("k"), 2490368, 65362, 82}
DOWN_KEYS = {ord("j"), 2621440, 65364, 84}


@dataclass
class Segment:
    start: int
    end: int
    label: str


@dataclass
class GuiState:
    selected: int
    step: float
    zoom: float
    pan_x: float
    pan_y: float
    dirty: bool
    last_mouse_full: tuple[float, float] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open an OpenCV GUI to manually nudge straight strip-mosaic legs "
            "up/down/left/right, then save a corrected mosaic."
        )
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("stitched_outputs/video01_02_03_vertical_strip_report.json"),
        help="Report JSON from stitch_vertical_strip.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stitched_outputs/video01_02_03_vertical_strip_adjusted.png"),
        help="Corrected mosaic output path.",
    )
    parser.add_argument(
        "--adjustments",
        type=Path,
        default=None,
        help="Adjustment JSON path. Default: <output stem>_adjustments.json",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.12,
        help="Scale for the interactive preview. Default: 0.12",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=None,
        help="Final output scale. Default: use scale from report canvas.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1800,
        help="GUI window width. Default: 1800",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1100,
        help="GUI window height. Default: 1100",
    )
    parser.add_argument(
        "--gui-backend",
        choices=("buttons", "opencv"),
        default="buttons",
        help="GUI backend. 'buttons' uses Tk buttons, 'opencv' uses key controls.",
    )
    parser.add_argument(
        "--initial-step",
        type=float,
        default=10.0,
        help="Initial nudge step in source-image pixels. Default: 10",
    )
    parser.add_argument(
        "--turn-dx-threshold",
        type=float,
        default=80.0,
        help="Break a segment when horizontal motion exceeds this many pixels.",
    )
    parser.add_argument(
        "--turn-ratio",
        type=float,
        default=0.45,
        help="Break a segment when abs(dx) is this fraction of abs(dy).",
    )
    parser.add_argument(
        "--min-vertical-step",
        type=float,
        default=120.0,
        help="Minimum abs(dy) to classify a step as vertical. Default: 120",
    )
    parser.add_argument(
        "--max-canvas-pixels",
        type=int,
        default=800_000_000,
        help="Safety limit for the final rendered canvas pixels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output and adjustment JSON files.",
    )
    parser.add_argument(
        "--print-segments",
        action="store_true",
        help="Print auto-detected segments and exit without opening the GUI.",
    )
    parser.add_argument(
        "--keep-label-prefix",
        action="append",
        default=[],
        help=(
            "Keep only segments whose label starts with this prefix. "
            "Can be repeated. Example: --keep-label-prefix vertical"
        ),
    )
    parser.add_argument(
        "--drop-label",
        action="append",
        default=[],
        help="Drop segments with this exact label. Can be repeated.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save the current filtered mosaic and exit without opening the GUI.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_positions(report: dict) -> np.ndarray:
    return np.array(
        [[frame["position_x"], frame["position_y"]] for frame in report["frames"]],
        dtype=np.float64,
    )


def classify_step(
    dx: float,
    dy: float,
    turn_dx_threshold: float,
    turn_ratio: float,
    min_vertical_step: float,
) -> str:
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    is_turn = abs_dx >= turn_dx_threshold and abs_dx >= turn_ratio * max(abs_dy, 1.0)
    if is_turn or abs_dy < min_vertical_step:
        return "turn"
    return "vertical_down" if dy > 0 else "vertical_up"


def detect_segments(positions: np.ndarray, args: argparse.Namespace) -> list[Segment]:
    if len(positions) < 2:
        return [Segment(0, len(positions) - 1, "single")]

    steps = np.diff(positions, axis=0)
    labels = [
        classify_step(
            float(dx),
            float(dy),
            args.turn_dx_threshold,
            args.turn_ratio,
            args.min_vertical_step,
        )
        for dx, dy in steps
    ]

    segments: list[Segment] = []
    start = 0
    current_label = labels[0]
    for step_index, label in enumerate(labels[1:], start=1):
        if label != current_label:
            segments.append(Segment(start, step_index, current_label))
            start = step_index
            current_label = label
    segments.append(Segment(start, len(positions) - 1, current_label))
    return merge_tiny_segments(segments, min_frames=2)


def merge_tiny_segments(segments: list[Segment], min_frames: int) -> list[Segment]:
    if len(segments) <= 1:
        return segments

    merged: list[Segment] = []
    for segment in segments:
        frame_count = segment.end - segment.start + 1
        if frame_count < min_frames and merged:
            previous = merged[-1]
            merged[-1] = Segment(previous.start, segment.end, previous.label)
        else:
            merged.append(segment)
    return merged


def filter_segments(segments: list[Segment], args: argparse.Namespace) -> list[Segment]:
    filtered = segments
    if args.keep_label_prefix:
        prefixes = tuple(args.keep_label_prefix)
        filtered = [segment for segment in filtered if segment.label.startswith(prefixes)]
    if args.drop_label:
        drop_labels = set(args.drop_label)
        filtered = [segment for segment in filtered if segment.label not in drop_labels]
    if not filtered:
        raise RuntimeError("No segments remain after filtering")
    return filtered


def frame_paths(report: dict) -> list[Path]:
    return [Path(frame["path"]) for frame in report["frames"]]


def frame_size(report: dict) -> tuple[int, int]:
    first = report["frames"][0]
    return int(first["width"]), int(first["height"])


def segment_offsets_for_frames(
    positions: np.ndarray,
    segments: list[Segment],
    offsets: np.ndarray,
) -> np.ndarray:
    adjusted = positions.copy()
    for segment_index, segment in enumerate(segments):
        adjusted[segment.start : segment.end + 1] += offsets[segment_index]
    return adjusted


def segment_frame_indices(segments: list[Segment]) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for segment in segments:
        for frame_index in range(segment.start, segment.end + 1):
            if frame_index not in seen:
                indices.append(frame_index)
                seen.add(frame_index)
    return indices


def segment_positions(positions: np.ndarray, segments: list[Segment]) -> np.ndarray:
    return positions[segment_frame_indices(segments)]


def render_segment_layer(
    paths: list[Path],
    positions: np.ndarray,
    segment: Segment,
    offset: np.ndarray,
    preview_scale: float,
    width: int,
    height: int,
    feather_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    source_weight: np.ndarray | None = None
    source_shape: tuple[int, int] | None = None
    scaled_feather_radius = feather_radius * preview_scale

    for frame_index in range(segment.start, segment.end + 1):
        image = read_image(paths[frame_index])
        if preview_scale != 1.0:
            image = cv2.resize(
                image,
                (
                    max(1, int(round(image.shape[1] * preview_scale))),
                    max(1, int(round(image.shape[0] * preview_scale))),
                ),
                interpolation=cv2.INTER_AREA if preview_scale < 1 else cv2.INTER_LINEAR,
            )

        image_height, image_width = image.shape[:2]
        if source_shape != (image_height, image_width):
            source_weight = make_weight_mask(
                image_height,
                image_width,
                scaled_feather_radius,
            )
            source_shape = (image_height, image_width)
        assert source_weight is not None

        tx = float((positions[frame_index, 0] + offset[0]) * preview_scale)
        ty = float((positions[frame_index, 1] + offset[1]) * preview_scale)
        matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
        warped_image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_weight = cv2.warpAffine(
            source_weight,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        accumulator += warped_image.astype(np.float32) * warped_weight[:, :, None]
        weights += warped_weight

    image = np.zeros((height, width, 3), dtype=np.uint8)
    valid = weights > 1e-6
    image[valid] = np.clip(accumulator[valid] / weights[valid, None], 0, 255).astype(np.uint8)
    return image, weights


def build_preview_layers(
    paths: list[Path],
    positions: np.ndarray,
    segments: list[Segment],
    frame_width: int,
    frame_height: int,
    preview_scale: float,
    feather_radius: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, int, int]:
    offset, source_width, source_height = compute_canvas(
        segment_positions(positions, segments),
        frame_width,
        frame_height,
        max_canvas_pixels=2_000_000_000,
        render_scale=preview_scale,
    )
    preview_width = max(1, int(round(source_width * preview_scale)))
    preview_height = max(1, int(round(source_height * preview_scale)))
    shifted_positions = positions + offset

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for index, segment in enumerate(segments):
        print(
            f"Preparing preview layer {index + 1}/{len(segments)}: "
            f"frames {segment.start}-{segment.end} ({segment.label})",
            flush=True,
        )
        layers.append(
            render_segment_layer(
                paths,
                shifted_positions,
                segment,
                np.zeros(2, dtype=np.float64),
                preview_scale,
                preview_width,
                preview_height,
                feather_radius,
            )
        )
    return layers, offset, preview_width, preview_height


def composite_preview(
    layers: list[tuple[np.ndarray, np.ndarray]],
    offsets: np.ndarray,
    preview_scale: float,
    selected: int,
) -> np.ndarray:
    height, width = layers[0][0].shape[:2]
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    for index, (image, weight) in enumerate(layers):
        dx = float(offsets[index, 0] * preview_scale)
        dy = float(offsets[index, 1] * preview_scale)
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
        shifted_image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        shifted_weight = cv2.warpAffine(
            weight,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if index == selected:
            tint = np.zeros_like(shifted_image)
            tint[:, :, 1] = 70
            shifted_image = cv2.addWeighted(shifted_image, 1.0, tint, 0.35, 0.0)
        accumulator += shifted_image.astype(np.float32) * shifted_weight[:, :, None]
        weights += shifted_weight

    preview = np.zeros((height, width, 3), dtype=np.uint8)
    valid = weights > 1e-6
    preview[valid] = np.clip(accumulator[valid] / weights[valid, None], 0, 255).astype(np.uint8)
    return preview


def segment_boxes(
    positions: np.ndarray,
    segments: list[Segment],
    offsets: np.ndarray,
    frame_width: int,
    frame_height: int,
    canvas_offset: np.ndarray,
    preview_scale: float,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for index, segment in enumerate(segments):
        segment_positions = positions[segment.start : segment.end + 1] + offsets[index]
        x0 = float(segment_positions[:, 0].min() + canvas_offset[0])
        y0 = float(segment_positions[:, 1].min() + canvas_offset[1])
        x1 = float((segment_positions[:, 0] + frame_width).max() + canvas_offset[0])
        y1 = float((segment_positions[:, 1] + frame_height).max() + canvas_offset[1])
        boxes.append(
            (
                int(round(x0 * preview_scale)),
                int(round(y0 * preview_scale)),
                int(round(x1 * preview_scale)),
                int(round(y1 * preview_scale)),
            )
        )
    return boxes


def crop_view(full_preview: np.ndarray, state: GuiState, width: int, height: int) -> np.ndarray:
    preview_height, preview_width = full_preview.shape[:2]
    view_width = max(1, int(round(width / state.zoom)))
    view_height = max(1, int(round(height / state.zoom)))
    max_x = max(0, preview_width - view_width)
    max_y = max(0, preview_height - view_height)
    state.pan_x = float(np.clip(state.pan_x, 0, max_x))
    state.pan_y = float(np.clip(state.pan_y, 0, max_y))
    x0 = int(round(state.pan_x))
    y0 = int(round(state.pan_y))
    crop = full_preview[y0 : y0 + view_height, x0 : x0 + view_width]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def draw_overlay(
    view: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    segments: list[Segment],
    offsets: np.ndarray,
    state: GuiState,
    preview_scale: float,
) -> np.ndarray:
    output = view.copy()
    height, width = output.shape[:2]
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        sx0 = int(round((x0 - state.pan_x) * state.zoom))
        sy0 = int(round((y0 - state.pan_y) * state.zoom))
        sx1 = int(round((x1 - state.pan_x) * state.zoom))
        sy1 = int(round((y1 - state.pan_y) * state.zoom))
        if sx1 < 0 or sy1 < 0 or sx0 >= width or sy0 >= height:
            continue
        color = (0, 255, 255) if index == state.selected else (255, 180, 0)
        cv2.rectangle(output, (sx0, sy0), (sx1, sy1), color, 2)
        cv2.putText(
            output,
            str(index + 1),
            (max(5, sx0 + 8), max(20, sy0 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    selected_segment = segments[state.selected]
    selected_offset = offsets[state.selected]
    lines = [
        f"segment {state.selected + 1}/{len(segments)} "
        f"{selected_segment.label} frames {selected_segment.start}-{selected_segment.end}",
        f"offset dx={selected_offset[0]:.1f}px dy={selected_offset[1]:.1f}px "
        f"step={state.step:.1f}px zoom={state.zoom:.2f}",
        "arrows/hjkl move | n/p select | +/- step | z/x zoom | w/a/e/d pan | c center | r reset | 0 reset all | s save | q quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            output,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 28

    if state.last_mouse_full is not None:
        mx, my = state.last_mouse_full
        text = f"mouse preview=({mx / preview_scale:.0f}, {my / preview_scale:.0f}) source px"
        cv2.putText(
            output,
            text,
            (14, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def center_on_segment(
    state: GuiState,
    boxes: list[tuple[int, int, int, int]],
    window_width: int,
    window_height: int,
) -> None:
    x0, y0, x1, y1 = boxes[state.selected]
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    state.pan_x = center_x - window_width / (2.0 * state.zoom)
    state.pan_y = center_y - window_height / (2.0 * state.zoom)


def save_outputs(
    args: argparse.Namespace,
    report: dict,
    positions: np.ndarray,
    segments: list[Segment],
    offsets: np.ndarray,
) -> None:
    adjustment_path = args.adjustments
    if adjustment_path is None:
        adjustment_path = args.output.with_name(args.output.stem + "_adjustments.json")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    if adjustment_path.exists() and not args.overwrite:
        raise FileExistsError(f"Adjustment file already exists: {adjustment_path}")

    frame_width, frame_height = frame_size(report)
    render_scale = args.render_scale
    if render_scale is None:
        render_scale = float(report.get("canvas", {}).get("scale", 1.0))

    adjusted_positions = segment_offsets_for_frames(positions, segments, offsets)
    kept_indices = segment_frame_indices(segments)
    kept_positions = adjusted_positions[kept_indices]
    canvas_offset, canvas_width, canvas_height = compute_canvas(
        kept_positions,
        frame_width,
        frame_height,
        args.max_canvas_pixels,
        render_scale,
    )
    print(
        f"Saving adjusted mosaic: source canvas {canvas_width}x{canvas_height}, "
        f"render scale {render_scale}",
        flush=True,
    )
    mosaic = render_mosaic(
        [
            type("FrameInfoLike", (), {"path": str(path)})()
            for index, path in enumerate(frame_paths(report))
            if index in set(kept_indices)
        ],
        kept_positions,
        canvas_offset,
        canvas_width,
        canvas_height,
        float(report.get("settings", {}).get("feather_radius", 80.0)),
        render_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), mosaic):
        raise RuntimeError(f"Failed to write output image: {args.output}")

    render_height, render_width = mosaic.shape[:2]
    payload = {
        "source_report": str(args.report),
        "output": str(args.output),
        "canvas": {
            "width": render_width,
            "height": render_height,
            "scale": render_scale,
        },
        "source_canvas": {"width": canvas_width, "height": canvas_height},
        "segments": [
            {
                "index": index,
                "start": segment.start,
                "end": segment.end,
                "label": segment.label,
                "offset_x": float(offsets[index, 0]),
                "offset_y": float(offsets[index, 1]),
            }
            for index, segment in enumerate(segments)
        ],
        "frames": [
            {
                **frame,
                "adjusted_position_x": float(adjusted_positions[index, 0]),
                "adjusted_position_y": float(adjusted_positions[index, 1]),
            }
            for index, frame in enumerate(report["frames"])
            if index in set(kept_indices)
        ],
    }
    adjustment_path.parent.mkdir(parents=True, exist_ok=True)
    adjustment_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote adjusted mosaic: {args.output}", flush=True)
    print(f"Wrote adjustments: {adjustment_path}", flush=True)


def image_to_tk_photo(image: np.ndarray):
    import tkinter as tk

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".png", rgb)
    if not ok:
        raise RuntimeError("Failed to encode preview image")
    payload = base64.b64encode(encoded).decode("ascii")
    return tk.PhotoImage(data=payload)


def run_button_gui(
    args: argparse.Namespace,
    report: dict,
    positions: np.ndarray,
    segments: list[Segment],
    paths: list[Path],
    frame_width: int,
    frame_height: int,
    layers: list[tuple[np.ndarray, np.ndarray]],
    canvas_offset: np.ndarray,
    preview_width: int,
    preview_height: int,
) -> None:
    import tkinter as tk
    from tkinter import messagebox

    control_width = 310
    view_width = max(640, args.window_width - control_width - 30)
    view_height = max(480, args.window_height - 40)

    offsets = np.zeros((len(segments), 2), dtype=np.float64)
    state = GuiState(
        selected=0,
        step=args.initial_step,
        zoom=1.0,
        pan_x=0.0,
        pan_y=0.0,
        dirty=True,
        last_mouse_full=None,
    )
    full_preview = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
    boxes = segment_boxes(
        positions,
        segments,
        offsets,
        frame_width,
        frame_height,
        canvas_offset,
        args.preview_scale,
    )
    center_on_segment(state, boxes, view_width, view_height)

    root = tk.Tk()
    root.title("Adjust UAV Strip Mosaic")
    root.geometry(f"{args.window_width}x{args.window_height}")

    selected_var = tk.StringVar()
    offset_var = tk.StringVar()
    step_var = tk.StringVar()
    status_var = tk.StringVar(value="Ready")
    save_var = tk.StringVar(value=f"Output: {args.output}")

    outer = tk.Frame(root)
    outer.pack(fill=tk.BOTH, expand=True)

    controls = tk.Frame(outer, width=control_width, padx=10, pady=10)
    controls.pack(side=tk.LEFT, fill=tk.Y)
    controls.pack_propagate(False)

    image_label = tk.Label(outer, bg="black")
    image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def update_text() -> None:
        segment = segments[state.selected]
        selected_var.set(
            f"Segment {state.selected + 1}/{len(segments)}\n"
            f"{segment.label}\nframes {segment.start}-{segment.end}"
        )
        offset = offsets[state.selected]
        offset_var.set(f"dx {offset[0]:.1f} px\n dy {offset[1]:.1f} px")
        step_var.set(f"Step: {state.step:.2f} px\nZoom: {state.zoom:.2f}x")

    def redraw(force_composite: bool = False) -> None:
        nonlocal full_preview, boxes
        if state.dirty or force_composite:
            status_var.set("Updating preview...")
            root.update_idletasks()
            full_preview = composite_preview(
                layers,
                offsets,
                args.preview_scale,
                state.selected,
            )
            boxes = segment_boxes(
                positions,
                segments,
                offsets,
                frame_width,
                frame_height,
                canvas_offset,
                args.preview_scale,
            )
            state.dirty = False

        view = crop_view(full_preview, state, view_width, view_height)
        view = draw_overlay(view, boxes, segments, offsets, state, args.preview_scale)
        photo = image_to_tk_photo(view)
        image_label.configure(image=photo)
        image_label.image = photo
        update_text()
        status_var.set("Ready")

    def move_selected(dx: float, dy: float) -> None:
        offsets[state.selected, 0] += dx * state.step
        offsets[state.selected, 1] += dy * state.step
        state.dirty = True
        redraw()

    def select_segment(index: int) -> None:
        state.selected = int(np.clip(index, 0, len(segments) - 1))
        center_on_segment(state, boxes, view_width, view_height)
        state.dirty = True
        redraw()

    def select_delta(delta: int) -> None:
        select_segment((state.selected + delta) % len(segments))

    def change_step(multiplier: float) -> None:
        state.step = max(0.25, state.step * multiplier)
        update_text()

    def change_zoom(multiplier: float) -> None:
        center_before_x = state.pan_x + view_width / (2.0 * state.zoom)
        center_before_y = state.pan_y + view_height / (2.0 * state.zoom)
        state.zoom = float(np.clip(state.zoom * multiplier, 0.25, 8.0))
        state.pan_x = center_before_x - view_width / (2.0 * state.zoom)
        state.pan_y = center_before_y - view_height / (2.0 * state.zoom)
        redraw()

    def pan(dx: float, dy: float) -> None:
        state.pan_x += dx * view_width / (4.0 * state.zoom)
        state.pan_y += dy * view_height / (4.0 * state.zoom)
        redraw()

    def reset_selected() -> None:
        offsets[state.selected] = 0.0
        state.dirty = True
        redraw()

    def reset_all() -> None:
        offsets[:] = 0.0
        state.dirty = True
        redraw()

    def save_now() -> None:
        status_var.set("Saving final mosaic...")
        root.update_idletasks()
        try:
            save_outputs(args, report, positions, segments, offsets)
        except Exception as error:  # noqa: BLE001 - show GUI error to user.
            messagebox.showerror("Save failed", str(error))
            status_var.set("Save failed")
            return
        status_var.set("Saved")
        messagebox.showinfo("Saved", f"Wrote:\n{args.output}")

    def center_selected() -> None:
        center_on_segment(state, boxes, view_width, view_height)
        redraw()

    tk.Label(controls, text="UAV Strip Adjust", font=("Helvetica", 18, "bold")).pack(
        anchor="w",
        pady=(0, 8),
    )
    action_frame = tk.LabelFrame(controls, text="Actions", padx=8, pady=8)
    action_frame.pack(fill=tk.X, pady=(0, 10))
    tk.Button(
        action_frame,
        text="Save Adjusted Mosaic",
        command=save_now,
        bg="#2f7d32",
        fg="white",
        height=2,
    ).pack(fill=tk.X, pady=(0, 6))
    tk.Button(action_frame, text="Quit", command=root.destroy).pack(fill=tk.X)
    tk.Label(
        controls,
        textvariable=status_var,
        justify=tk.LEFT,
        fg="#1b5e20",
    ).pack(anchor="w", fill=tk.X, pady=(0, 8))

    tk.Label(controls, textvariable=selected_var, justify=tk.LEFT, font=("Helvetica", 13)).pack(
        anchor="w",
        fill=tk.X,
        pady=(0, 6),
    )
    tk.Label(controls, textvariable=offset_var, justify=tk.LEFT, font=("Helvetica", 12)).pack(
        anchor="w",
        fill=tk.X,
        pady=(0, 6),
    )
    tk.Label(controls, textvariable=step_var, justify=tk.LEFT).pack(anchor="w", pady=(0, 6))

    segment_frame = tk.LabelFrame(controls, text="Segments", padx=8, pady=8)
    segment_frame.pack(fill=tk.X, pady=(0, 10))
    for index, segment in enumerate(segments):
        button = tk.Button(
            segment_frame,
            text=f"{index + 1}. {segment.label} {segment.start}-{segment.end}",
            command=lambda value=index: select_segment(value),
            anchor="w",
        )
        button.pack(fill=tk.X, pady=2)

    move_frame = tk.LabelFrame(controls, text="Move Current Segment", padx=8, pady=8)
    move_frame.pack(fill=tk.X, pady=(0, 10))
    tk.Button(move_frame, text="↑", height=2, command=lambda: move_selected(0, -1)).grid(
        row=0,
        column=1,
        sticky="nsew",
        padx=3,
        pady=3,
    )
    tk.Button(move_frame, text="←", height=2, command=lambda: move_selected(-1, 0)).grid(
        row=1,
        column=0,
        sticky="nsew",
        padx=3,
        pady=3,
    )
    tk.Button(move_frame, text="→", height=2, command=lambda: move_selected(1, 0)).grid(
        row=1,
        column=2,
        sticky="nsew",
        padx=3,
        pady=3,
    )
    tk.Button(move_frame, text="↓", height=2, command=lambda: move_selected(0, 1)).grid(
        row=2,
        column=1,
        sticky="nsew",
        padx=3,
        pady=3,
    )
    for column in range(3):
        move_frame.columnconfigure(column, weight=1)

    select_frame = tk.Frame(controls)
    select_frame.pack(fill=tk.X, pady=(0, 10))
    tk.Button(select_frame, text="Previous", command=lambda: select_delta(-1)).pack(
        side=tk.LEFT,
        expand=True,
        fill=tk.X,
        padx=(0, 4),
    )
    tk.Button(select_frame, text="Next", command=lambda: select_delta(1)).pack(
        side=tk.LEFT,
        expand=True,
        fill=tk.X,
        padx=(4, 0),
    )

    step_frame = tk.LabelFrame(controls, text="Step / Zoom / View", padx=8, pady=8)
    step_frame.pack(fill=tk.X, pady=(0, 10))
    tk.Button(step_frame, text="Step -", command=lambda: change_step(0.5)).grid(
        row=0,
        column=0,
        sticky="ew",
        padx=3,
        pady=3,
    )
    tk.Button(step_frame, text="Step +", command=lambda: change_step(2.0)).grid(
        row=0,
        column=1,
        sticky="ew",
        padx=3,
        pady=3,
    )
    tk.Button(step_frame, text="Zoom -", command=lambda: change_zoom(0.8)).grid(
        row=1,
        column=0,
        sticky="ew",
        padx=3,
        pady=3,
    )
    tk.Button(step_frame, text="Zoom +", command=lambda: change_zoom(1.25)).grid(
        row=1,
        column=1,
        sticky="ew",
        padx=3,
        pady=3,
    )
    tk.Button(step_frame, text="Center", command=center_selected).grid(
        row=2,
        column=0,
        sticky="ew",
        padx=3,
        pady=3,
    )
    tk.Button(step_frame, text="Reset Segment", command=reset_selected).grid(
        row=2,
        column=1,
        sticky="ew",
        padx=3,
        pady=3,
    )
    for column in range(2):
        step_frame.columnconfigure(column, weight=1)

    pan_frame = tk.LabelFrame(controls, text="Pan View", padx=8, pady=8)
    pan_frame.pack(fill=tk.X, pady=(0, 10))
    tk.Button(pan_frame, text="View ↑", command=lambda: pan(0, -1)).grid(
        row=0,
        column=1,
        sticky="ew",
        padx=3,
        pady=3,
    )
    tk.Button(pan_frame, text="View ←", command=lambda: pan(-1, 0)).grid(
        row=1,
        column=0,
        sticky="ew",
        padx=3,
        pady=3,
    )
    tk.Button(pan_frame, text="View →", command=lambda: pan(1, 0)).grid(
        row=1,
        column=2,
        sticky="ew",
        padx=3,
        pady=3,
    )
    tk.Button(pan_frame, text="View ↓", command=lambda: pan(0, 1)).grid(
        row=2,
        column=1,
        sticky="ew",
        padx=3,
        pady=3,
    )
    for column in range(3):
        pan_frame.columnconfigure(column, weight=1)

    tk.Button(controls, text="Reset All", command=reset_all).pack(fill=tk.X, pady=(0, 6))
    tk.Label(controls, textvariable=save_var, justify=tk.LEFT, wraplength=control_width - 20).pack(
        anchor="w",
        fill=tk.X,
        pady=(0, 8),
    )

    def on_image_click(event) -> None:
        full_x = state.pan_x + event.x / state.zoom
        full_y = state.pan_y + event.y / state.zoom
        state.last_mouse_full = (full_x, full_y)
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            if x0 <= full_x <= x1 and y0 <= full_y <= y1:
                select_segment(index)
                break

    image_label.bind("<Button-1>", on_image_click)
    root.bind("<Left>", lambda _event: move_selected(-1, 0))
    root.bind("<Right>", lambda _event: move_selected(1, 0))
    root.bind("<Up>", lambda _event: move_selected(0, -1))
    root.bind("<Down>", lambda _event: move_selected(0, 1))
    root.bind("h", lambda _event: move_selected(-1, 0))
    root.bind("l", lambda _event: move_selected(1, 0))
    root.bind("k", lambda _event: move_selected(0, -1))
    root.bind("j", lambda _event: move_selected(0, 1))
    root.bind("n", lambda _event: select_delta(1))
    root.bind("p", lambda _event: select_delta(-1))
    root.bind("+", lambda _event: change_step(2.0))
    root.bind("=", lambda _event: change_step(2.0))
    root.bind("-", lambda _event: change_step(0.5))
    root.bind("z", lambda _event: change_zoom(0.8))
    root.bind("x", lambda _event: change_zoom(1.25))
    root.bind("c", lambda _event: center_selected())
    root.bind("r", lambda _event: reset_selected())
    root.bind("0", lambda _event: reset_all())
    root.bind("s", lambda _event: save_now())
    root.bind("q", lambda _event: root.destroy())
    root.bind("<Escape>", lambda _event: root.destroy())

    redraw(force_composite=True)
    root.mainloop()


def run_opencv_gui(
    args: argparse.Namespace,
    report: dict,
    positions: np.ndarray,
    segments: list[Segment],
    frame_width: int,
    frame_height: int,
    layers: list[tuple[np.ndarray, np.ndarray]],
    canvas_offset: np.ndarray,
    preview_width: int,
    preview_height: int,
) -> None:
    offsets = np.zeros((len(segments), 2), dtype=np.float64)
    state = GuiState(
        selected=0,
        step=args.initial_step,
        zoom=1.0,
        pan_x=0.0,
        pan_y=0.0,
        dirty=True,
        last_mouse_full=None,
    )
    full_preview = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
    boxes = segment_boxes(
        positions,
        segments,
        offsets,
        frame_width,
        frame_height,
        canvas_offset,
        args.preview_scale,
    )
    center_on_segment(state, boxes, args.window_width, args.window_height)

    window_name = "Adjust UAV Strip Mosaic"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        full_x = state.pan_x + x / state.zoom
        full_y = state.pan_y + y / state.zoom
        state.last_mouse_full = (full_x, full_y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for index, (x0, y0, x1, y1) in enumerate(boxes):
                if x0 <= full_x <= x1 and y0 <= full_y <= y1:
                    state.selected = index
                    center_on_segment(state, boxes, args.window_width, args.window_height)
                    state.dirty = True
                    break

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if state.dirty:
            full_preview = composite_preview(
                layers,
                offsets,
                args.preview_scale,
                state.selected,
            )
            boxes = segment_boxes(
                positions,
                segments,
                offsets,
                frame_width,
                frame_height,
                canvas_offset,
                args.preview_scale,
            )
            state.dirty = False

        view = crop_view(full_preview, state, args.window_width, args.window_height)
        view = draw_overlay(view, boxes, segments, offsets, state, args.preview_scale)
        cv2.imshow(window_name, view)
        key = cv2.waitKeyEx(30)
        if key < 0:
            continue

        if key in (ord("q"), 27):
            break
        if key in RIGHT_KEYS:
            offsets[state.selected, 0] += state.step
            state.dirty = True
        elif key in LEFT_KEYS:
            offsets[state.selected, 0] -= state.step
            state.dirty = True
        elif key in UP_KEYS:
            offsets[state.selected, 1] -= state.step
            state.dirty = True
        elif key in DOWN_KEYS:
            offsets[state.selected, 1] += state.step
            state.dirty = True
        elif key == ord("n"):
            state.selected = (state.selected + 1) % len(segments)
            center_on_segment(state, boxes, args.window_width, args.window_height)
            state.dirty = True
        elif key == ord("p"):
            state.selected = (state.selected - 1) % len(segments)
            center_on_segment(state, boxes, args.window_width, args.window_height)
            state.dirty = True
        elif key in (ord("+"), ord("=")):
            state.step *= 2.0
        elif key in (ord("-"), ord("_")):
            state.step = max(0.25, state.step / 2.0)
        elif key == ord("z"):
            state.zoom = max(0.25, state.zoom / 1.25)
        elif key == ord("x"):
            state.zoom = min(8.0, state.zoom * 1.25)
        elif key == ord("w"):
            state.pan_y -= args.window_height / (4.0 * state.zoom)
        elif key == ord("a"):
            state.pan_x -= args.window_width / (4.0 * state.zoom)
        elif key == ord("d"):
            state.pan_x += args.window_width / (4.0 * state.zoom)
        elif key == ord("e"):
            state.pan_y += args.window_height / (4.0 * state.zoom)
        elif key == ord("s"):
            view[:, :] = 0
            cv2.putText(
                view,
                "Saving adjusted mosaic...",
                (40, args.window_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, view)
            cv2.waitKey(1)
            save_outputs(args, report, positions, segments, offsets)
        elif key == ord("c"):
            center_on_segment(state, boxes, args.window_width, args.window_height)
        elif key == ord("r"):
            offsets[state.selected] = 0.0
            state.dirty = True
        elif key == ord("0"):
            offsets[:] = 0.0
            state.dirty = True

    cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    try:
        if args.preview_scale <= 0:
            raise ValueError("--preview-scale must be greater than 0")

        report = load_report(args.report)
        positions = load_positions(report)
        paths = frame_paths(report)
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Sampled frame not found: {path}")

        all_segments = detect_segments(positions, args)
        segments = filter_segments(all_segments, args)
        if args.print_segments:
            for index, segment in enumerate(all_segments, start=1):
                marker = "*" if segment in segments else " "
                print(
                    f"{marker} {index:02d}: frames {segment.start:04d}-{segment.end:04d} "
                    f"{segment.label}"
                )
            return 0

        if args.save_only:
            offsets = np.zeros((len(segments), 2), dtype=np.float64)
            save_outputs(args, report, positions, segments, offsets)
            return 0

        frame_width, frame_height = frame_size(report)
        layers, canvas_offset, preview_width, preview_height = build_preview_layers(
            paths,
            positions,
            segments,
            frame_width,
            frame_height,
            args.preview_scale,
            float(report.get("settings", {}).get("feather_radius", 80.0)),
        )
        if args.gui_backend == "buttons":
            run_button_gui(
                args,
                report,
                positions,
                segments,
                paths,
                frame_width,
                frame_height,
                layers,
                canvas_offset,
                preview_width,
                preview_height,
            )
        else:
            run_opencv_gui(
                args,
                report,
                positions,
                segments,
                frame_width,
                frame_height,
                layers,
                canvas_offset,
                preview_width,
                preview_height,
            )
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
