#!/usr/bin/env python3
"""Create a reusable georeferenced basemap from four clicked GPS control points."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np


DEFAULT_OUTPUT_ROOT = Path("georeferenced_maps")
POINT_NAMES = ("P1", "P2", "P3", "P4")


@dataclass
class ControlPoint:
    name: str
    x: float
    y: float
    latitude: float
    longitude: float


@dataclass
class ViewState:
    scale: float
    pan_x: float
    pan_y: float
    points: list[tuple[int, int]]
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
            "Select an image, click four map control points, enter their WGS84 "
            "GPS coordinates, and save a reusable basemap georeference JSON."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input basemap image. If omitted, a file picker is opened.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: georeferenced_maps/<image stem>.",
    )
    parser.add_argument(
        "--points",
        default=None,
        help=(
            "Non-interactive points as "
            "'x,y,latitude,longitude;x,y,latitude,longitude;...' with exactly 4 points."
        ),
    )
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=1000)
    parser.add_argument(
        "--initial-scale",
        type=float,
        default=0.0,
        help="Initial display scale. Use 0 to fit the whole image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output files in the selected output directory.",
    )
    return parser.parse_args()


def choose_image_file() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("tkinter is unavailable; pass --image instead.") from exc

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose basemap image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        raise RuntimeError("No image selected.")
    return Path(path)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def fit_scale(image_width: int, image_height: int, window_width: int, window_height: int) -> float:
    return max(0.02, min(window_width / image_width, window_height / image_height))


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
    point: tuple[int, int],
    state: ViewState,
    geometry: ViewGeometry,
) -> tuple[int, int]:
    x, y = point
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
    view = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    scaled = cv2.resize(
        crop,
        (geometry.display_width, geometry.display_height),
        interpolation=cv2.INTER_AREA if state.scale < 1 else cv2.INTER_LINEAR,
    )
    view[
        geometry.display_y : geometry.display_y + geometry.display_height,
        geometry.display_x : geometry.display_x + geometry.display_width,
    ] = scaled

    visible_points = []
    for index, point in enumerate(state.points, start=1):
        sx, sy = source_to_screen(point, state, geometry)
        if not (0 <= sx < window_width and 0 <= sy < window_height):
            continue
        visible_points.append((sx, sy))
        cv2.drawMarker(view, (sx, sy), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 28, 2)
        draw_text(view, POINT_NAMES[index - 1], (sx + 10, sy - 10), 0.72)
    if len(visible_points) >= 2:
        cv2.polylines(view, [np.array(visible_points, dtype=np.int32)], False, (0, 255, 255), 2)

    instructions = [
        "Click 4 GPS control points | enter/space finish | u undo | f fit | +/- zoom | w/a/s/d pan | q/esc quit",
        state.message,
    ]
    y = 28
    for line in instructions:
        draw_text(view, line, (12, y), 0.62)
        y += 28
    return view


def draw_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (x, y), font, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def collect_points_interactively(image: np.ndarray, args: argparse.Namespace) -> list[tuple[int, int]]:
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
        points=[],
        message=f"Image: {image_width}x{image_height}. Click {POINT_NAMES[0]}.",
    )
    window_name = "Create Georeferenced Basemap"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(state.points) >= 4:
            return
        geometry = compute_geometry(
            image_width,
            image_height,
            state,
            args.window_width,
            args.window_height,
        )
        point = screen_to_source(x, y, state, geometry, image_width, image_height)
        state.points.append(point)
        if len(state.points) < 4:
            state.message = f"Set {POINT_NAMES[len(state.points) - 1]}. Click {POINT_NAMES[len(state.points)]}."
        else:
            state.message = "Four points selected. Press enter or space to continue."

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, render_view(image, state, args.window_width, args.window_height))
        key = cv2.waitKeyEx(30)
        if key < 0:
            continue
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Point selection cancelled.")
        if key in (13, 10, 32):
            if len(state.points) == 4:
                cv2.destroyWindow(window_name)
                return state.points
            state.message = f"Need 4 points; currently selected {len(state.points)}."
        elif key in (ord("u"), 8, 127):
            if state.points:
                state.points.pop()
            next_index = min(len(state.points), 3)
            state.message = f"Undo. Click {POINT_NAMES[next_index]}."
        elif key == ord("f"):
            state.scale = fit_scale(image_width, image_height, args.window_width, args.window_height)
            state.pan_x = 0.0
            state.pan_y = 0.0
            state.message = "Fit whole image."
        elif key in (ord("+"), ord("=")):
            zoom_around_window_center(state, args.window_width, args.window_height, 1.25)
        elif key in (ord("-"), ord("_")):
            zoom_around_window_center(state, args.window_width, args.window_height, 1 / 1.25)
        elif key == ord("w"):
            state.pan_y -= args.window_height / (4.0 * state.scale)
        elif key == ord("a"):
            state.pan_x -= args.window_width / (4.0 * state.scale)
        elif key == ord("s"):
            state.pan_y += args.window_height / (4.0 * state.scale)
        elif key == ord("d"):
            state.pan_x += args.window_width / (4.0 * state.scale)


def zoom_around_window_center(
    state: ViewState,
    window_width: int,
    window_height: int,
    factor: float,
) -> None:
    center_x = state.pan_x + window_width / (2.0 * state.scale)
    center_y = state.pan_y + window_height / (2.0 * state.scale)
    state.scale = float(np.clip(state.scale * factor, 0.02, 8.0))
    state.pan_x = center_x - window_width / (2.0 * state.scale)
    state.pan_y = center_y - window_height / (2.0 * state.scale)


def prompt_gps_for_points(points: list[tuple[int, int]]) -> list[ControlPoint]:
    print("\nEnter GPS coordinates for each clicked point as latitude,longitude.")
    control_points = []
    for name, (x, y) in zip(POINT_NAMES, points):
        while True:
            try:
                raw = input(f"{name} pixel=({x},{y}) GPS lat,lon: ").strip()
            except EOFError as exc:
                raise RuntimeError(
                    "Could not read GPS input from the terminal. If you used "
                    "`conda run`, rerun with `conda run --no-capture-output ...`, "
                    "or activate the environment first and then run `python ...`."
                ) from exc
            try:
                latitude, longitude = parse_lat_lon(raw)
            except ValueError as exc:
                print(f"Invalid GPS coordinate: {exc}")
                continue
            control_points.append(ControlPoint(name, float(x), float(y), latitude, longitude))
            break
    return control_points


def parse_lat_lon(raw: str) -> tuple[float, float]:
    cleaned = raw.replace(" ", ",")
    parts = [part for part in cleaned.split(",") if part]
    if len(parts) != 2:
        raise ValueError("use two numbers, for example 23.456789,120.123456")
    latitude = float(parts[0])
    longitude = float(parts[1])
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude must be between -90 and 90")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("longitude must be between -180 and 180")
    return latitude, longitude


def parse_points_arg(raw: str) -> list[ControlPoint]:
    points = []
    for index, point_raw in enumerate(raw.split(";"), start=1):
        parts = [part.strip() for part in point_raw.split(",") if part.strip()]
        if len(parts) != 4:
            raise ValueError("--points entries must be x,y,latitude,longitude")
        x, y, latitude, longitude = map(float, parts)
        if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
            raise ValueError("--points contains an invalid latitude or longitude")
        points.append(ControlPoint(POINT_NAMES[index - 1], x, y, latitude, longitude))
    if len(points) != 4:
        raise ValueError("--points must contain exactly 4 semicolon-separated points")
    return points


def compute_affines(control_points: list[ControlPoint]) -> tuple[np.ndarray, np.ndarray]:
    pixel_points = np.float64([[point.x, point.y] for point in control_points])
    gps_points = np.float64([[point.longitude, point.latitude] for point in control_points])
    pixel_to_gps = fit_affine(pixel_points, gps_points, "pixel control points")
    gps_to_pixel = fit_affine(gps_points, pixel_points, "GPS control points")
    return pixel_to_gps, gps_to_pixel


def fit_affine(source_points: np.ndarray, target_points: np.ndarray, label: str) -> np.ndarray:
    design = np.column_stack([source_points, np.ones(len(source_points), dtype=np.float64)])
    rank = np.linalg.matrix_rank(design)
    if rank < 3:
        raise ValueError(f"{label} must not be collinear; cannot fit a whole-map linear transform")
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        design,
        target_points,
        rcond=None,
    )
    return coefficients.T


def transform_pixel_to_gps(pixel_to_gps: np.ndarray, x: float, y: float) -> tuple[float, float]:
    mapped = pixel_to_gps @ np.array([x, y, 1.0], dtype=np.float64)
    longitude = float(mapped[0])
    latitude = float(mapped[1])
    return latitude, longitude


def full_image_bounds(
    width: int,
    height: int,
    pixel_to_gps_affine: np.ndarray,
) -> dict[str, float]:
    corner_pixels = [
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (float(width - 1), float(height - 1)),
        (0.0, float(height - 1)),
    ]
    corners = [
        transform_pixel_to_gps(pixel_to_gps_affine, x, y)
        for x, y in corner_pixels
    ]
    latitudes = [latitude for latitude, _longitude in corners]
    longitudes = [longitude for _latitude, longitude in corners]
    return {
        "north": max(latitudes),
        "south": min(latitudes),
        "west": min(longitudes),
        "east": max(longitudes),
    }


def build_payload(
    input_path: Path,
    copied_image_path: Path,
    image: np.ndarray,
    control_points: list[ControlPoint],
    pixel_to_gps_affine: np.ndarray,
    gps_to_pixel_affine: np.ndarray,
    preview_path: Path,
    csv_path: Path,
) -> dict[str, object]:
    height, width = image.shape[:2]
    latitudes = [point.latitude for point in control_points]
    longitudes = [point.longitude for point in control_points]
    image_bounds = full_image_bounds(width, height, pixel_to_gps_affine)
    return {
        "schema": "uav_object_localization.georeferenced_basemap.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_image": str(input_path),
        "map_image": copied_image_path.name,
        "image": {"width": width, "height": height},
        "crs": "EPSG:4326",
        "georeference": {
            "model": "pixel_to_wgs84_affine",
            "pixel_axis": {"x": "image column", "y": "image row"},
            "gps_axis": {"x": "longitude", "y": "latitude"},
            "fit": "least_squares_from_4_control_points",
            "pixel_to_gps_affine": matrix_to_list(pixel_to_gps_affine),
            "gps_to_pixel_affine": matrix_to_list(gps_to_pixel_affine),
            "bounds": image_bounds,
            "control_point_bounds": {
                "north": max(latitudes),
                "south": min(latitudes),
                "west": min(longitudes),
                "east": max(longitudes),
            },
        },
        "control_points": [
            {
                "name": point.name,
                "pixel": {"x": round(point.x, 3), "y": round(point.y, 3)},
                "gps": {
                    "latitude": round(point.latitude, 8),
                    "longitude": round(point.longitude, 8),
                },
            }
            for point in control_points
        ],
        "outputs": {"preview": preview_path.name, "control_points_csv": csv_path.name},
    }


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def write_outputs(
    input_path: Path,
    image: np.ndarray,
    control_points: list[ControlPoint],
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"basemap{input_path.suffix.lower()}"
    json_path = output_dir / "basemap_georef.json"
    csv_path = output_dir / "control_points.csv"
    preview_path = output_dir / "basemap_georef_preview.jpg"

    for path in (image_path, json_path, csv_path, preview_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}")

    pixel_to_gps_affine, gps_to_pixel_affine = compute_affines(control_points)
    shutil.copy2(input_path, image_path)

    preview = draw_preview(image, control_points, pixel_to_gps_affine)
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Failed to write preview image: {preview_path}")

    write_control_points_csv(csv_path, control_points)
    payload = build_payload(
        input_path,
        image_path,
        image,
        control_points,
        pixel_to_gps_affine,
        gps_to_pixel_affine,
        preview_path,
        csv_path,
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "image": image_path,
        "json": json_path,
        "csv": csv_path,
        "preview": preview_path,
    }


def draw_preview(
    image: np.ndarray,
    control_points: list[ControlPoint],
    pixel_to_gps: np.ndarray,
) -> np.ndarray:
    output = image.copy()
    pixel_polygon = np.array([[round(point.x), round(point.y)] for point in control_points], dtype=np.int32)
    cv2.polylines(output, [pixel_polygon], True, (0, 255, 255), 3, cv2.LINE_AA)
    for point in control_points:
        x = int(round(point.x))
        y = int(round(point.y))
        cv2.drawMarker(output, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 32, 3)
        label = f"{point.name} {point.latitude:.7f},{point.longitude:.7f}"
        draw_preview_label(output, label, (x + 12, y - 12))

    center_x = image.shape[1] / 2.0
    center_y = image.shape[0] / 2.0
    center_latitude, center_longitude = transform_pixel_to_gps(pixel_to_gps, center_x, center_y)
    draw_preview_label(
        output,
        f"center {center_latitude:.7f},{center_longitude:.7f}",
        (int(center_x) + 12, int(center_y) - 12),
    )
    cv2.drawMarker(
        output,
        (int(round(center_x)), int(round(center_y))),
        (255, 0, 255),
        cv2.MARKER_CROSS,
        30,
        2,
    )
    return output


def draw_preview_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, image.shape[1] - width - 8))
    y = max(height + 8, min(y, image.shape[0] - baseline - 4))
    cv2.rectangle(image, (x - 4, y - height - 6), (x + width + 4, y + baseline + 4), (255, 255, 255), -1)
    cv2.putText(image, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def write_control_points_csv(path: Path, control_points: list[ControlPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["name", "pixel_x", "pixel_y", "latitude", "longitude"])
        writer.writeheader()
        for point in control_points:
            writer.writerow(
                {
                    "name": point.name,
                    "pixel_x": round(point.x, 3),
                    "pixel_y": round(point.y, 3),
                    "latitude": round(point.latitude, 8),
                    "longitude": round(point.longitude, 8),
                }
            )


def main() -> int:
    args = parse_args()
    try:
        input_path = args.image if args.image is not None else choose_image_file()
        input_path = input_path.expanduser().resolve()
        image = read_image(input_path)
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_ROOT / input_path.stem

        if args.points:
            control_points = parse_points_arg(args.points)
        else:
            clicked_points = collect_points_interactively(image, args)
            control_points = prompt_gps_for_points(clicked_points)

        outputs = write_outputs(input_path, image, control_points, output_dir, args.overwrite)
        print("Wrote georeferenced basemap:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
        print("\nUse with:")
        print(f"  python3 scripts/georeference_map.py pixel --georef-json {outputs['json']} --x 100 --y 100")
        return 0
    except (FileExistsError, RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
