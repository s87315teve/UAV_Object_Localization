#!/usr/bin/env python3
"""Select a rectangular image region and save it as a new file."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CropBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass
class ViewState:
    scale: float
    pan_x: float
    pan_y: float
    drawing: bool
    drag_start: tuple[int, int] | None
    current_mouse: tuple[int, int] | None
    crop_box: CropBox | None
    message: str


@dataclass
class ViewGeometry:
    source_x: int
    source_y: int
    source_width: int
    source_height: int
    display_x: int
    display_y: int
    display_width: int
    display_height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively select a rectangle from a large image and save the "
            "original-resolution crop."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("stitched_outputs/video01_02_03_vertical_only_adjusted.png"),
        help="Input image path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stitched_outputs/selected_roi.png"),
        help="Output cropped image path.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional JSON metadata path. Default: <output stem>_metadata.json",
    )
    parser.add_argument("--x", type=int, default=None, help="Crop left x in source pixels.")
    parser.add_argument("--y", type=int, default=None, help="Crop top y in source pixels.")
    parser.add_argument("--width", type=int, default=None, help="Crop width in source pixels.")
    parser.add_argument("--height", type=int, default=None, help="Crop height in source pixels.")
    parser.add_argument(
        "--window-width",
        type=int,
        default=1600,
        help="Interactive window width. Default: 1600",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1000,
        help="Interactive window height. Default: 1000",
    )
    parser.add_argument(
        "--initial-scale",
        type=float,
        default=0.0,
        help="Initial display scale. Use 0 to fit the whole image. Default: 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output/metadata files.",
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def fit_scale(image_width: int, image_height: int, window_width: int, window_height: int) -> float:
    return max(0.02, min(window_width / image_width, window_height / image_height))


def validate_crop_box(box: CropBox, image_width: int, image_height: int) -> CropBox:
    x1 = int(np.clip(box.x, 0, image_width - 1))
    y1 = int(np.clip(box.y, 0, image_height - 1))
    x2 = int(np.clip(box.x2, x1 + 1, image_width))
    y2 = int(np.clip(box.y2, y1 + 1, image_height))
    return CropBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def write_crop(
    image: np.ndarray,
    box: CropBox,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Metadata already exists: {metadata_path}")

    image_height, image_width = image.shape[:2]
    box = validate_crop_box(box, image_width, image_height)
    crop = image[box.y : box.y2, box.x : box.x2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), crop):
        raise RuntimeError(f"Failed to write crop: {output_path}")

    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "image_width": image_width,
        "image_height": image_height,
        "crop": asdict(box),
        "crop_x2": box.x2,
        "crop_y2": box.y2,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote crop: {output_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Crop box: x={box.x}, y={box.y}, width={box.width}, height={box.height}")


def compute_geometry(
    image_width: int,
    image_height: int,
    state: ViewState,
    window_width: int,
    window_height: int,
) -> ViewGeometry:
    view_width = min(image_width, max(1, int(np.floor(window_width / state.scale))))
    view_height = min(image_height, max(1, int(np.floor(window_height / state.scale))))
    state.pan_x = float(np.clip(state.pan_x, 0, max(0, image_width - view_width)))
    state.pan_y = float(np.clip(state.pan_y, 0, max(0, image_height - view_height)))

    source_x = int(round(state.pan_x))
    source_y = int(round(state.pan_y))
    source_width = min(view_width, image_width - source_x)
    source_height = min(view_height, image_height - source_y)
    display_width = min(window_width, max(1, int(round(source_width * state.scale))))
    display_height = min(window_height, max(1, int(round(source_height * state.scale))))
    display_x = (window_width - display_width) // 2
    display_y = (window_height - display_height) // 2
    return ViewGeometry(
        source_x=source_x,
        source_y=source_y,
        source_width=source_width,
        source_height=source_height,
        display_x=display_x,
        display_y=display_y,
        display_width=display_width,
        display_height=display_height,
    )


def source_to_screen(
    x: int,
    y: int,
    state: ViewState,
    geometry: ViewGeometry,
) -> tuple[int, int]:
    return (
        geometry.display_x + int(round((x - geometry.source_x) * state.scale)),
        geometry.display_y + int(round((y - geometry.source_y) * state.scale)),
    )


def screen_to_source(
    x: int,
    y: int,
    state: ViewState,
    geometry: ViewGeometry,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    local_x = np.clip(x - geometry.display_x, 0, max(0, geometry.display_width - 1))
    local_y = np.clip(y - geometry.display_y, 0, max(0, geometry.display_height - 1))
    source_x = int(round(local_x / state.scale + geometry.source_x))
    source_y = int(round(local_y / state.scale + geometry.source_y))
    return (
        int(np.clip(source_x, 0, image_width - 1)),
        int(np.clip(source_y, 0, image_height - 1)),
    )


def box_from_points(
    p1: tuple[int, int],
    p2: tuple[int, int],
) -> CropBox:
    x1, y1 = p1
    x2, y2 = p2
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    return CropBox(left, top, max(1, right - left), max(1, bottom - top))


def render_view(
    image: np.ndarray,
    state: ViewState,
    window_width: int,
    window_height: int,
) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    geometry = compute_geometry(image_width, image_height, state, window_width, window_height)
    x1 = geometry.source_x
    y1 = geometry.source_y
    x2 = x1 + geometry.source_width
    y2 = y1 + geometry.source_height
    crop = image[y1:y2, x1:x2]
    scaled = cv2.resize(
        crop,
        (geometry.display_width, geometry.display_height),
        interpolation=cv2.INTER_AREA if state.scale < 1 else cv2.INTER_LINEAR,
    )
    view = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    view[
        geometry.display_y : geometry.display_y + geometry.display_height,
        geometry.display_x : geometry.display_x + geometry.display_width,
    ] = scaled

    active_box = state.crop_box
    if state.drawing and state.drag_start is not None and state.current_mouse is not None:
        active_box = box_from_points(state.drag_start, state.current_mouse)

    if active_box is not None:
        active_box = validate_crop_box(active_box, image_width, image_height)
        sx1, sy1 = source_to_screen(active_box.x, active_box.y, state, geometry)
        sx2, sy2 = source_to_screen(active_box.x2, active_box.y2, state, geometry)
        overlay = view.copy()
        cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (0, 255, 255), -1)
        view = cv2.addWeighted(overlay, 0.18, view, 0.82, 0.0)
        cv2.rectangle(view, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
        label = (
            f"x={active_box.x} y={active_box.y} "
            f"w={active_box.width} h={active_box.height}"
        )
        cv2.putText(
            view,
            label,
            (max(8, sx1 + 8), max(28, sy1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            label,
            (max(8, sx1 + 8), max(28, sy1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    instructions = [
        "Drag left mouse to select rectangle | s save | r reset | f fit whole image | +/- zoom | w/a/x/d pan | q/esc quit",
        state.message,
    ]
    y = 28
    for line in instructions:
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        y += 28
    return view


def interactive_select(
    image: np.ndarray,
    args: argparse.Namespace,
) -> CropBox | None:
    image_height, image_width = image.shape[:2]
    initial_scale = (
        fit_scale(image_width, image_height, args.window_width, args.window_height)
        if args.initial_scale <= 0
        else args.initial_scale
    )
    state = ViewState(
        scale=initial_scale,
        pan_x=0.0,
        pan_y=0.0,
        drawing=False,
        drag_start=None,
        current_mouse=None,
        crop_box=None,
        message=f"Image: {image_width}x{image_height}",
    )

    window_name = "Select Crop Rectangle"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        geometry = compute_geometry(
            image_width,
            image_height,
            state,
            args.window_width,
            args.window_height,
        )
        source_point = screen_to_source(x, y, state, geometry, image_width, image_height)
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.drag_start = source_point
            state.current_mouse = source_point
            state.message = "Dragging..."
        elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
            state.current_mouse = source_point
        elif event == cv2.EVENT_LBUTTONUP and state.drawing:
            state.drawing = False
            state.current_mouse = source_point
            assert state.drag_start is not None
            state.crop_box = validate_crop_box(
                box_from_points(state.drag_start, source_point),
                image_width,
                image_height,
            )
            state.message = "Selection ready. Press s to save."

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        view = render_view(image, state, args.window_width, args.window_height)
        cv2.imshow(window_name, view)
        key = cv2.waitKeyEx(30)
        if key < 0:
            continue
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            return None
        if key == ord("s"):
            if state.crop_box is None:
                state.message = "No rectangle selected yet."
                continue
            cv2.destroyWindow(window_name)
            return state.crop_box
        if key == ord("r"):
            state.crop_box = None
            state.message = "Selection reset."
        elif key == ord("f"):
            state.scale = fit_scale(image_width, image_height, args.window_width, args.window_height)
            state.pan_x = 0.0
            state.pan_y = 0.0
            state.message = "Fit whole image."
        elif key in (ord("+"), ord("=")):
            center_x = state.pan_x + args.window_width / (2.0 * state.scale)
            center_y = state.pan_y + args.window_height / (2.0 * state.scale)
            state.scale = min(8.0, state.scale * 1.25)
            state.pan_x = center_x - args.window_width / (2.0 * state.scale)
            state.pan_y = center_y - args.window_height / (2.0 * state.scale)
        elif key in (ord("-"), ord("_")):
            center_x = state.pan_x + args.window_width / (2.0 * state.scale)
            center_y = state.pan_y + args.window_height / (2.0 * state.scale)
            state.scale = max(0.02, state.scale / 1.25)
            state.pan_x = center_x - args.window_width / (2.0 * state.scale)
            state.pan_y = center_y - args.window_height / (2.0 * state.scale)
        elif key == ord("w"):
            state.pan_y -= args.window_height / (4.0 * state.scale)
        elif key == ord("a"):
            state.pan_x -= args.window_width / (4.0 * state.scale)
        elif key == ord("d"):
            state.pan_x += args.window_width / (4.0 * state.scale)
        elif key == ord("x"):
            state.pan_y += args.window_height / (4.0 * state.scale)


def crop_box_from_args(args: argparse.Namespace) -> CropBox | None:
    values = [args.x, args.y, args.width, args.height]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("--x, --y, --width, and --height must be provided together")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be greater than 0")
    return CropBox(args.x, args.y, args.width, args.height)


def main() -> int:
    args = parse_args()
    try:
        image = read_image(args.input)
        metadata_path = args.metadata
        if metadata_path is None:
            metadata_path = args.output.with_name(args.output.stem + "_metadata.json")

        crop_box = crop_box_from_args(args)
        if crop_box is None:
            crop_box = interactive_select(image, args)
            if crop_box is None:
                print("No crop saved.")
                return 0

        write_crop(
            image,
            crop_box,
            args.input,
            args.output,
            metadata_path,
            args.overwrite,
        )
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
