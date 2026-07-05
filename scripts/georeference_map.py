#!/usr/bin/env python3
"""Convert pixels on the reference aerial map to GPS and localize query images."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_GEOREF_JSON = Path(
    "georeferenced_maps/localize_ready_selected_roi/uav_selected_roi_compressed_georef.json"
)
DEFAULT_MAP_IMAGE = Path("衛星影像/aerial_gps_range_clean.png")
DEFAULT_DRAW_OUTPUT = Path("stitched_outputs/georef/aerial_reference_grid.png")
DEFAULT_MATCH_OUTPUT = Path("stitched_outputs/georef/match_result.png")


@dataclass(frozen=True)
class GeoReference:
    width: int = 1600
    height: int = 1000
    north: float = 23.4583306
    south: float = 23.4478994
    west: float = 120.2700692
    east: float = 120.2932708
    pixel_to_gps_affine: tuple[tuple[float, ...], ...] | None = None
    gps_to_pixel_affine: tuple[tuple[float, ...], ...] | None = None
    pixel_to_gps_homography: tuple[tuple[float, ...], ...] | None = None
    gps_to_pixel_homography: tuple[tuple[float, ...], ...] | None = None
    control_points: tuple[dict[str, Any], ...] = ()
    source: str = "default_bounds"

    def contains_pixel(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height

    def pixel_to_gps(self, x: float, y: float) -> tuple[float, float]:
        if not self.contains_pixel(x, y):
            raise ValueError(
                f"Pixel ({x:.2f}, {y:.2f}) is outside the map range "
                f"0..{self.width}, 0..{self.height}"
            )
        if self.pixel_to_gps_affine is not None:
            matrix = np.array(self.pixel_to_gps_affine, dtype=np.float64)
            mapped = matrix @ np.array([x, y, 1.0], dtype=np.float64)
            longitude = float(mapped[0])
            latitude = float(mapped[1])
            return latitude, longitude
        if self.pixel_to_gps_homography is not None:
            matrix = np.array(self.pixel_to_gps_homography, dtype=np.float64)
            mapped = cv2.perspectiveTransform(np.float32([[[x, y]]]), matrix).reshape(2)
            longitude = float(mapped[0])
            latitude = float(mapped[1])
            return latitude, longitude
        longitude = self.west + (x / self.width) * (self.east - self.west)
        latitude = self.north - (y / self.height) * (self.north - self.south)
        return latitude, longitude

    def gps_to_pixel(self, latitude: float, longitude: float) -> tuple[float, float]:
        if self.gps_to_pixel_affine is not None:
            matrix = np.array(self.gps_to_pixel_affine, dtype=np.float64)
            mapped = matrix @ np.array([longitude, latitude, 1.0], dtype=np.float64)
            x = float(mapped[0])
            y = float(mapped[1])
            if not self.contains_pixel(x, y):
                raise ValueError(
                    f"GPS ({latitude:.7f}, {longitude:.7f}) maps outside the image "
                    f"to pixel ({x:.2f}, {y:.2f})"
                )
            return x, y
        if self.gps_to_pixel_homography is not None:
            matrix = np.array(self.gps_to_pixel_homography, dtype=np.float64)
            mapped = cv2.perspectiveTransform(np.float32([[[longitude, latitude]]]), matrix).reshape(2)
            x = float(mapped[0])
            y = float(mapped[1])
            if not self.contains_pixel(x, y):
                raise ValueError(
                    f"GPS ({latitude:.7f}, {longitude:.7f}) maps outside the image "
                    f"to pixel ({x:.2f}, {y:.2f})"
                )
            return x, y
        if not (self.south <= latitude <= self.north and self.west <= longitude <= self.east):
            raise ValueError(
                f"GPS ({latitude:.7f}, {longitude:.7f}) is outside the configured map bounds"
            )
        x = (longitude - self.west) / (self.east - self.west) * self.width
        y = (self.north - latitude) / (self.north - self.south) * self.height
        return x, y


KNOWN_POINTS = {
    "P1": ((270, 132), (23.45695, 120.27399)),
    "P2": ((212, 632), (23.45174, 120.27314)),
    "P3": ((1388, 392), (23.45424, 120.29020)),
    "P4": ((1253, 868), (23.44928, 120.28824)),
    "P5": ((801, 258), (23.45564, 120.28169)),
    "P6": ((728, 756), (23.45044, 120.28062)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert pixels and GPS coordinates using the default compressed UAV "
            "basemap, or another georeferenced map passed with --georef-json."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    draw_parser = subparsers.add_parser("draw-map", help="Save a visual reference map")
    add_map_args(draw_parser)
    draw_parser.add_argument("--output", type=Path, default=DEFAULT_DRAW_OUTPUT)
    draw_parser.add_argument("--grid-cols", type=int, default=8)
    draw_parser.add_argument("--grid-rows", type=int, default=5)
    draw_parser.add_argument(
        "--show",
        action="store_true",
        help="Open an OpenCV window to display the generated map.",
    )

    pixel_parser = subparsers.add_parser("pixel", help="Convert one map pixel to GPS")
    add_georef_json_arg(pixel_parser)
    pixel_parser.add_argument("--x", type=float, required=True)
    pixel_parser.add_argument("--y", type=float, required=True)

    gps_parser = subparsers.add_parser("gps", help="Convert one GPS coordinate to map pixel")
    add_georef_json_arg(gps_parser)
    gps_parser.add_argument("--lat", type=float, required=True)
    gps_parser.add_argument("--lon", type=float, required=True)

    click_parser = subparsers.add_parser("click", help="Open the map and print GPS on clicks")
    add_map_args(click_parser)
    click_parser.add_argument("--output", type=Path, default=DEFAULT_DRAW_OUTPUT)
    click_parser.add_argument(
        "--window-scale",
        type=float,
        default=0.8,
        help="Initial OpenCV window scale. Default: 0.8",
    )

    match_parser = subparsers.add_parser(
        "match", help="Find the best matching location of a query image on the map"
    )
    add_map_args(match_parser)
    match_parser.add_argument("--query", type=Path, required=True, help="Input image to localize")
    match_parser.add_argument(
        "--query-roi",
        help="Optional crop inside the query image as x,y,width,height before matching.",
    )
    match_parser.add_argument(
        "--query-point",
        help="Optional point in the original query image as x,y to convert after matching.",
    )
    match_parser.add_argument(
        "--orientations",
        choices=("none", "rotations", "flips", "all"),
        default="none",
        help=(
            "Try orientation candidates before matching. Use all when the query "
            "image may be rotated or flipped. Default: none"
        ),
    )
    match_parser.add_argument("--output", type=Path, default=DEFAULT_MATCH_OUTPUT)
    match_parser.add_argument(
        "--show",
        action="store_true",
        help="Open an OpenCV window to display the match result.",
    )
    match_parser.add_argument(
        "--method",
        choices=("auto", "feature", "template"),
        default="auto",
        help="Matching method. Default: auto",
    )
    match_parser.add_argument("--max-features", type=int, default=8000)
    match_parser.add_argument(
        "--feature-max-dim",
        type=int,
        default=1800,
        help=(
            "Downscale map/query images so their longest side is at most this "
            "many pixels before feature matching. Use 0 to disable. Default: 1800"
        ),
    )
    match_parser.add_argument("--ratio", type=float, default=0.75)
    match_parser.add_argument("--min-matches", type=int, default=20)
    match_parser.add_argument("--min-inliers", type=int, default=10)
    match_parser.add_argument(
        "--match-workers",
        type=int,
        default=1,
        help=(
            "Parallel workers for independent orientation candidates. "
            "Use 1 to keep sequential behavior. Default: 1"
        ),
    )
    match_parser.add_argument(
        "--min-template-score",
        type=float,
        default=0.65,
        help="Template matches below this score are reported with a warning. Default: 0.65",
    )
    match_parser.add_argument(
        "--template-scales",
        default="auto",
        help=(
            "Comma-separated query scales for template matching fallback, or auto. "
            "Default: auto"
        ),
    )
    return parser.parse_args()


def add_map_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--map-image",
        type=Path,
        default=DEFAULT_MAP_IMAGE,
        help=f"Reference map image. Default: {DEFAULT_MAP_IMAGE}",
    )
    add_georef_json_arg(parser)


def add_georef_json_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--georef-json",
        type=Path,
        default=DEFAULT_GEOREF_JSON,
        help=(
            "Optional georeferenced basemap JSON generated by "
            f"scripts/create_georeferenced_basemap.py. Default: {DEFAULT_GEOREF_JSON}"
        ),
    )


def load_georef(path: Path | None = None) -> GeoReference:
    if path is None:
        return GeoReference()

    payload = json.loads(path.read_text(encoding="utf-8"))
    image_info = payload.get("image", {})
    georef_info = payload.get("georeference", {})
    bounds = georef_info.get("bounds", {})
    width = int(image_info.get("width", payload.get("width")))
    height = int(image_info.get("height", payload.get("height")))
    pixel_to_gps_affine = georef_info.get("pixel_to_gps_affine")
    gps_to_pixel_affine = georef_info.get("gps_to_pixel_affine")
    pixel_to_gps_homography = georef_info.get("pixel_to_gps_homography")
    gps_to_pixel_homography = georef_info.get("gps_to_pixel_homography")
    if (
        (pixel_to_gps_affine is None or gps_to_pixel_affine is None)
        and (pixel_to_gps_homography is None or gps_to_pixel_homography is None)
    ):
        raise ValueError(
            f"{path} does not contain affine or homography georeference matrices"
        )

    return GeoReference(
        width=width,
        height=height,
        north=float(bounds.get("north", 90.0)),
        south=float(bounds.get("south", -90.0)),
        west=float(bounds.get("west", -180.0)),
        east=float(bounds.get("east", 180.0)),
        pixel_to_gps_affine=matrix_tuple(pixel_to_gps_affine),
        gps_to_pixel_affine=matrix_tuple(gps_to_pixel_affine),
        pixel_to_gps_homography=matrix_tuple(pixel_to_gps_homography),
        gps_to_pixel_homography=matrix_tuple(gps_to_pixel_homography),
        control_points=tuple(payload.get("control_points", ())),
        source=str(path),
    )


def matrix_tuple(raw_matrix: object | None) -> tuple[tuple[float, ...], ...] | None:
    if raw_matrix is None:
        return None
    return tuple(tuple(float(value) for value in row) for row in raw_matrix)


def map_image_from_georef(path: Path) -> Path | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_map_image = payload.get("map_image")
    if not raw_map_image:
        return None
    map_image = Path(raw_map_image)
    if map_image.exists() or map_image.is_absolute():
        return map_image
    candidate = path.parent / map_image
    if candidate.exists():
        return candidate
    return map_image


def resolve_map_image_path(args: argparse.Namespace) -> Path:
    if args.georef_json is not None and args.map_image == DEFAULT_MAP_IMAGE:
        map_image = map_image_from_georef(args.georef_json)
        if map_image is not None:
            return map_image
    return args.map_image


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def ensure_map_size(image: np.ndarray, georef: GeoReference) -> None:
    height, width = image.shape[:2]
    if (width, height) != (georef.width, georef.height):
        raise ValueError(
            f"Map size is {width}x{height}, but the georeference expects "
            f"{georef.width}x{georef.height}"
        )


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    if not cv2.imwrite(str(path), image, params):
        raise RuntimeError(f"Failed to write image: {path}")


def show_image(window_name: str, image: np.ndarray, max_width: int = 1400, max_height: int = 900) -> None:
    display = resize_to_fit(image, max_width=max_width, max_height=max_height)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display)
    print("Press q or Esc in the image window to close.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyWindow(window_name)


def draw_reference_map(
    image: np.ndarray,
    georef: GeoReference,
    grid_cols: int,
    grid_rows: int,
) -> np.ndarray:
    output = image.copy()
    overlay = output.copy()

    for col in range(1, grid_cols):
        x = round(col * georef.width / grid_cols)
        cv2.line(overlay, (x, 0), (x, georef.height), (255, 255, 255), 1, cv2.LINE_AA)
    for row in range(1, grid_rows):
        y = round(row * georef.height / grid_rows)
        cv2.line(overlay, (0, y), (georef.width, y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)

    cv2.rectangle(output, (0, 0), (georef.width - 1, georef.height - 1), (0, 255, 255), 2)
    label = (
        f"N {georef.north:.7f}  S {georef.south:.7f}  "
        f"W {georef.west:.7f}  E {georef.east:.7f}"
    )
    draw_label(output, label, (16, 30), (0, 0, 0), (255, 255, 255))
    if georef.source != "default_bounds":
        draw_label(output, f"georef: {georef.source}", (16, 58), (0, 0, 0), (255, 255, 255))

    if georef.control_points:
        points_to_draw = []
        for point in georef.control_points:
            pixel = point["pixel"]
            gps = point["gps"]
            points_to_draw.append(
                (
                    str(point.get("name", "")),
                    (float(pixel["x"]), float(pixel["y"])),
                    (float(gps["latitude"]), float(gps["longitude"])),
                )
            )
    else:
        points_to_draw = [
            (name, (float(x), float(y)), (float(lat), float(lon)))
            for name, ((x, y), (lat, lon)) in KNOWN_POINTS.items()
        ]

    for name, (x, y), (lat, lon) in points_to_draw:
        point = (int(round(x)), int(round(y)))
        cv2.drawMarker(output, point, (0, 0, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        draw_label(
            output,
            f"{name} ({point[0]},{point[1]}) {lat:.5f},{lon:.5f}",
            (point[0] + 10, point[1] - 10),
            (0, 0, 0),
            (255, 255, 255),
        )
    return output


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, image.shape[1] - width - 8))
    y = max(height + 8, min(y, image.shape[0] - baseline - 4))
    cv2.rectangle(
        image,
        (x - 4, y - height - 6),
        (x + width + 4, y + baseline + 4),
        bg_color,
        -1,
    )
    cv2.putText(image, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)


def command_draw_map(args: argparse.Namespace) -> None:
    georef = load_georef(args.georef_json)
    map_image = read_bgr(resolve_map_image_path(args))
    ensure_map_size(map_image, georef)
    output = draw_reference_map(map_image, georef, args.grid_cols, args.grid_rows)
    write_image(args.output, output)
    print(f"Wrote visual reference map: {args.output}")
    if args.show:
        show_image("aerial reference map", output)


def command_pixel(args: argparse.Namespace) -> None:
    georef = load_georef(args.georef_json)
    latitude, longitude = georef.pixel_to_gps(args.x, args.y)
    print(json.dumps(pixel_result(args.x, args.y, latitude, longitude), ensure_ascii=False, indent=2))


def command_gps(args: argparse.Namespace) -> None:
    georef = load_georef(args.georef_json)
    x, y = georef.gps_to_pixel(args.lat, args.lon)
    print(
        json.dumps(
            {"latitude": args.lat, "longitude": args.lon, "x": x, "y": y},
            ensure_ascii=False,
            indent=2,
        )
    )


def command_click(args: argparse.Namespace) -> None:
    georef = load_georef(args.georef_json)
    map_image = read_bgr(resolve_map_image_path(args))
    ensure_map_size(map_image, georef)
    display = draw_reference_map(map_image, georef, 8, 5)
    write_image(args.output, display)
    print(f"Wrote clickable reference preview: {args.output}")
    print("Left-click inside the map to print GPS. Press q or Esc to quit.")

    window_name = "aerial map pixel to GPS"
    state = {"image": display}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        try:
            latitude, longitude = georef.pixel_to_gps(x, y)
        except ValueError as exc:
            print(exc)
            return
        print(json.dumps(pixel_result(x, y, latitude, longitude), ensure_ascii=False))
        cv2.drawMarker(state["image"], (x, y), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 28, 2)
        draw_label(
            state["image"],
            f"{x},{y} -> {latitude:.7f}, {longitude:.7f}",
            (x + 12, y - 12),
            (0, 0, 0),
            (255, 255, 255),
        )
        cv2.imshow(window_name, state["image"])

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        max(320, int(georef.width * args.window_scale)),
        max(240, int(georef.height * args.window_scale)),
    )
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, state["image"])
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()


def pixel_result(x: float, y: float, latitude: float, longitude: float) -> dict[str, float]:
    return {
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "latitude": round(latitude, 8),
        "longitude": round(longitude, 8),
    }


def command_match(args: argparse.Namespace) -> None:
    georef = load_georef(args.georef_json)
    map_image = read_bgr(resolve_map_image_path(args))
    ensure_map_size(map_image, georef)
    original_query_image = read_bgr(args.query)
    query_roi = parse_query_roi(args.query_roi, original_query_image.shape)
    query_image = crop_to_roi(original_query_image, query_roi)
    query_point = parse_query_point(args.query_point) if args.query_point else None

    match_result, warnings = find_best_match(map_image, query_image, args)

    center_x, center_y = match_result["center"]
    latitude, longitude = georef.pixel_to_gps(center_x, center_y)
    output = visualize_match(
        map_image,
        match_result["oriented_query_image"],
        match_result,
        latitude,
        longitude,
    )
    write_image(args.output, output)

    report = {
        "method": match_result["method"],
        "score": match_result["score"],
        "orientation": match_result["orientation"],
        "center_pixel": {"x": round(center_x, 3), "y": round(center_y, 3)},
        "center_gps": {"latitude": round(latitude, 8), "longitude": round(longitude, 8)},
        "corners_pixel": [
            {"x": round(float(x), 3), "y": round(float(y), 3)}
            for x, y in match_result["corners"]
        ],
        "output": str(args.output),
    }
    if args.query_roi:
        report["query_roi"] = {
            "x": query_roi[0],
            "y": query_roi[1],
            "width": query_roi[2],
            "height": query_roi[3],
        }
    if query_point is not None:
        point_map_x, point_map_y = map_query_point(match_result, query_point, query_roi)
        point_latitude, point_longitude = georef.pixel_to_gps(point_map_x, point_map_y)
        report["query_point"] = {"x": query_point[0], "y": query_point[1]}
        report["query_point_map_pixel"] = {
            "x": round(point_map_x, 3),
            "y": round(point_map_y, 3),
        }
        report["query_point_gps"] = {
            "latitude": round(point_latitude, 8),
            "longitude": round(point_longitude, 8),
        }
    if "inliers" in match_result:
        report["inliers"] = match_result["inliers"]
        report["matches"] = match_result["matches"]
    if "scale" in match_result:
        report["scale"] = match_result["scale"]
    if "feature_scale" in match_result:
        report["feature_scale"] = match_result["feature_scale"]
    if "orientation_scores" in match_result:
        report["orientation_scores"] = match_result["orientation_scores"]
    if warnings:
        report["warnings"] = warnings
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.show:
        show_image("aerial match result", output)


def find_best_match(
    map_image: np.ndarray,
    query_image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[str]]:
    candidates = orientation_candidates(args.orientations)
    warnings = []
    scored_results = []
    failures = []
    match_workers = max(1, int(getattr(args, "match_workers", 1)))

    def evaluate_orientation(orientation: str) -> tuple[dict[str, object] | None, list[str], str | None]:
        oriented_query = apply_orientation(query_image, orientation)
        try:
            match_result, candidate_warnings = match_single_orientation(
                map_image,
                oriented_query,
                args,
            )
        except RuntimeError as exc:
            return None, [], f"{orientation}: {exc}"

        match_result["orientation"] = orientation
        match_result["orientation_matrix"] = orientation_matrix(
            orientation,
            query_image.shape[1],
            query_image.shape[0],
        )
        match_result["oriented_query_image"] = oriented_query
        orientation_warnings = [f"{orientation}: {warning}" for warning in candidate_warnings]
        return match_result, orientation_warnings, None

    if match_workers > 1 and len(candidates) > 1:
        with ThreadPoolExecutor(max_workers=min(match_workers, len(candidates))) as executor:
            orientation_results = list(executor.map(evaluate_orientation, candidates))
    else:
        orientation_results = [evaluate_orientation(orientation) for orientation in candidates]

    for match_result, candidate_warnings, failure in orientation_results:
        if failure is not None:
            failures.append(failure)
            continue
        if match_result is not None:
            scored_results.append(match_result)
            warnings.extend(candidate_warnings)

    if not scored_results:
        joined_failures = "; ".join(failures)
        raise RuntimeError(f"No orientation candidate matched successfully. {joined_failures}")

    best_result = max(scored_results, key=match_quality)
    identity_result = next(
        (result for result in scored_results if result["orientation"] == "identity"),
        None,
    )
    orientation_switch_margin = float(getattr(args, "orientation_switch_margin", 0.0))
    if (
        identity_result is not None
        and best_result["orientation"] != "identity"
        and match_quality(best_result) - match_quality(identity_result) < orientation_switch_margin
    ):
        warnings.append(
            "Best rotated orientation score did not exceed identity by "
            f"{orientation_switch_margin:.3f}; kept identity to avoid a weak texture false match."
        )
        best_result = identity_result
    best_result["orientation_scores"] = [
        {
            "orientation": result["orientation"],
            "method": result["method"],
            "score": result["score"],
        }
        for result in sorted(scored_results, key=match_quality, reverse=True)
    ]

    if len(scored_results) > 1:
        print(
            "Best orientation: "
            f"{best_result['orientation']} ({best_result['method']} score={best_result['score']})"
        )
    return best_result, warnings


def match_single_orientation(
    map_image: np.ndarray,
    query_image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[str]]:
    warnings = []
    if args.method in ("auto", "feature"):
        try:
            return (
                feature_match(
                    map_image,
                    query_image,
                    args.max_features,
                    getattr(args, "feature_max_dim", 1800),
                    args.ratio,
                    args.min_matches,
                    args.min_inliers,
                ),
                warnings,
            )
        except RuntimeError as exc:
            if args.method == "feature":
                raise
            warnings.append(str(exc))

    if args.method in ("auto", "template"):
        scales = parse_template_scales(args.template_scales, map_image, query_image)
        match_result = template_match(map_image, query_image, scales)
        if match_result["score"] < args.min_template_score:
            warnings.append(
                f"Low template score {match_result['score']} < {args.min_template_score}. "
                "Treat this as a rough guess; use --query-roi around stable landmarks "
                "or provide a rough candidate area before trusting the GPS."
            )
        return match_result, warnings

    raise RuntimeError("No matching method was available")


def match_quality(match_result: dict[str, object]) -> float:
    score = float(match_result["score"])
    if str(match_result["method"]).startswith("feature:"):
        return 1.0 + score
    return score


def orientation_candidates(mode: str) -> list[str]:
    if mode == "none":
        return ["identity"]
    if mode == "rotations":
        return ["identity", "rotate90_cw", "rotate180", "rotate90_ccw"]
    if mode == "flips":
        return ["identity", "flip_horizontal", "flip_vertical", "rotate180"]
    if mode == "all":
        return [
            "identity",
            "rotate90_cw",
            "rotate180",
            "rotate90_ccw",
            "flip_horizontal",
            "flip_vertical",
            "transpose",
            "anti_transpose",
        ]
    raise ValueError(f"Unknown orientation mode: {mode}")


def apply_orientation(image: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "identity":
        return image.copy()
    if orientation == "rotate90_cw":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "rotate180":
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == "rotate90_ccw":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == "flip_horizontal":
        return cv2.flip(image, 1)
    if orientation == "flip_vertical":
        return cv2.flip(image, 0)
    if orientation == "transpose":
        return cv2.transpose(image)
    if orientation == "anti_transpose":
        return cv2.flip(cv2.transpose(image), -1)
    raise ValueError(f"Unknown orientation: {orientation}")


def orientation_matrix(orientation: str, width: int, height: int) -> np.ndarray:
    max_x = width - 1
    max_y = height - 1
    if orientation == "identity":
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    if orientation == "rotate90_cw":
        return np.array([[0, -1, max_y], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    if orientation == "rotate180":
        return np.array([[-1, 0, max_x], [0, -1, max_y], [0, 0, 1]], dtype=np.float64)
    if orientation == "rotate90_ccw":
        return np.array([[0, 1, 0], [-1, 0, max_x], [0, 0, 1]], dtype=np.float64)
    if orientation == "flip_horizontal":
        return np.array([[-1, 0, max_x], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    if orientation == "flip_vertical":
        return np.array([[1, 0, 0], [0, -1, max_y], [0, 0, 1]], dtype=np.float64)
    if orientation == "transpose":
        return np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    if orientation == "anti_transpose":
        return np.array([[0, -1, max_y], [-1, 0, max_x], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"Unknown orientation: {orientation}")


def parse_query_roi(raw: str | None, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    if raw is None:
        return 0, 0, width, height
    values = parse_int_tuple(raw, 4, "--query-roi")
    x, y, roi_width, roi_height = values
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("--query-roi width and height must be positive")
    if x < 0 or y < 0 or x + roi_width > width or y + roi_height > height:
        raise ValueError(
            f"--query-roi {raw} is outside the query image size {width}x{height}"
        )
    return x, y, roi_width, roi_height


def crop_to_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    return image[y : y + height, x : x + width].copy()


def parse_query_point(raw: str) -> tuple[float, float]:
    x, y = parse_float_tuple(raw, 2, "--query-point")
    return x, y


def parse_int_tuple(raw: str, expected_count: int, argument_name: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != expected_count:
        raise ValueError(f"{argument_name} must have {expected_count} comma-separated values")
    return tuple(int(part) for part in parts)


def parse_float_tuple(raw: str, expected_count: int, argument_name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != expected_count:
        raise ValueError(f"{argument_name} must have {expected_count} comma-separated values")
    return tuple(float(part) for part in parts)


def map_query_point(
    match_result: dict[str, object],
    query_point: tuple[float, float],
    query_roi: tuple[int, int, int, int],
) -> tuple[float, float]:
    roi_x, roi_y, roi_width, roi_height = query_roi
    local_x = query_point[0] - roi_x
    local_y = query_point[1] - roi_y
    if not (0 <= local_x <= roi_width and 0 <= local_y <= roi_height):
        raise ValueError("--query-point must be inside --query-roi")

    oriented_point = match_result["orientation_matrix"] @ np.array(
        [local_x, local_y, 1.0],
        dtype=np.float64,
    )
    oriented_x = float(oriented_point[0] / oriented_point[2])
    oriented_y = float(oriented_point[1] / oriented_point[2])

    if "homography" in match_result:
        point = np.float32([[[oriented_x, oriented_y]]])
        mapped = cv2.perspectiveTransform(point, match_result["homography"]).reshape(2)
        return float(mapped[0]), float(mapped[1])

    top_left_x, top_left_y = match_result["top_left"]
    scale = float(match_result["scale"])
    return float(top_left_x + oriented_x * scale), float(top_left_y + oriented_y * scale)


def parse_template_scales(raw: str, map_image: np.ndarray, query_image: np.ndarray) -> list[float]:
    map_height, map_width = map_image.shape[:2]
    query_height, query_width = query_image.shape[:2]
    max_fit_scale = min(map_width / query_width, map_height / query_height)

    if raw.strip().lower() == "auto":
        base_scales = [
            max_fit_scale * factor
            for factor in (0.98, 0.85, 0.70, 0.55, 0.40, 0.30, 0.22, 0.16, 0.12, 0.09, 0.07)
        ]
        base_scales.extend([0.40, 0.30, 0.20, 0.15, 0.10, 0.075, 0.05])
    else:
        base_scales = [float(item.strip()) for item in raw.split(",") if item.strip()]

    scales = []
    seen = set()
    for scale in base_scales:
        if scale <= 0:
            raise ValueError("Template scales must be positive")
        scaled_width = int(round(query_width * scale))
        scaled_height = int(round(query_height * scale))
        if scaled_width < 16 or scaled_height < 16:
            continue
        if scaled_width > map_width or scaled_height > map_height:
            continue
        key = round(scale, 6)
        if key in seen:
            continue
        seen.add(key)
        scales.append(scale)

    if any(scale <= 0 for scale in scales):
        raise ValueError("Template scales must be positive")
    if not scales:
        raise ValueError(
            "No usable template scale fits inside the map. Use a smaller crop, "
            "or pass smaller --template-scales values such as 0.1,0.075,0.05."
        )
    return scales


def feature_match(
    map_image: np.ndarray,
    query_image: np.ndarray,
    max_features: int,
    feature_max_dim: int,
    ratio: float,
    min_matches: int,
    min_inliers: int,
) -> dict[str, object]:
    detector_name, detector, norm_type = create_feature_detector(max_features)
    scaled_map, map_scale = resize_for_feature_match(map_image, feature_max_dim)
    scaled_query, query_scale = resize_for_feature_match(query_image, feature_max_dim)
    map_gray = cv2.cvtColor(scaled_map, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(scaled_query, cv2.COLOR_BGR2GRAY)
    map_keypoints, map_descriptors = detector.detectAndCompute(map_gray, None)
    query_keypoints, query_descriptors = detector.detectAndCompute(query_gray, None)
    if map_descriptors is None or query_descriptors is None:
        raise RuntimeError(f"{detector_name} did not find enough descriptors")

    matcher = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(query_descriptors, map_descriptors, k=2)
    good_matches = []
    for candidates in raw_matches:
        if len(candidates) != 2:
            continue
        best, second_best = candidates
        if best.distance < ratio * second_best.distance:
            good_matches.append(best)
    if len(good_matches) < min_matches:
        raise RuntimeError(f"Only {len(good_matches)} feature matches; need {min_matches}")

    query_points = np.float32([query_keypoints[match.queryIdx].pt for match in good_matches])
    map_points = np.float32([map_keypoints[match.trainIdx].pt for match in good_matches])
    ransac_threshold = max(3.0, 5.0 * min(map_scale, query_scale))
    homography_scaled, inlier_mask = cv2.findHomography(
        query_points,
        map_points,
        cv2.RANSAC,
        ransac_threshold,
    )
    if homography_scaled is None or inlier_mask is None:
        raise RuntimeError("Could not estimate query-to-map homography")
    inliers = int(inlier_mask.sum())
    if inliers < min_inliers:
        raise RuntimeError(f"Only {inliers} RANSAC inliers; need {min_inliers}")

    homography = rescale_homography(homography_scaled, query_scale, map_scale)

    query_height, query_width = query_image.shape[:2]
    query_corners = np.float32(
        [[0, 0], [query_width - 1, 0], [query_width - 1, query_height - 1], [0, query_height - 1]]
    ).reshape(-1, 1, 2)
    map_corners = cv2.perspectiveTransform(query_corners, homography).reshape(-1, 2)
    center = cv2.perspectiveTransform(
        np.float32([[[query_width / 2.0, query_height / 2.0]]]), homography
    ).reshape(2)

    if not point_inside_image(center, map_image):
        raise RuntimeError("Feature match center landed outside the map")
    validate_match_geometry(map_corners, query_width, query_height, map_image)

    return {
        "method": f"feature:{detector_name}",
        "score": round(inliers / len(good_matches), 4),
        "center": (float(center[0]), float(center[1])),
        "corners": [(float(x), float(y)) for x, y in map_corners],
        "matches": len(good_matches),
        "inliers": inliers,
        "homography": homography,
        "feature_scale": {"map": round(map_scale, 5), "query": round(query_scale, 5)},
    }


def resize_for_feature_match(image: np.ndarray, feature_max_dim: int) -> tuple[np.ndarray, float]:
    if feature_max_dim <= 0:
        return image, 1.0
    height, width = image.shape[:2]
    longest_side = max(width, height)
    if longest_side <= feature_max_dim:
        return image, 1.0
    scale = feature_max_dim / float(longest_side)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def rescale_homography(homography_scaled: np.ndarray, query_scale: float, map_scale: float) -> np.ndarray:
    query_to_scaled = np.array([[query_scale, 0, 0], [0, query_scale, 0], [0, 0, 1]], dtype=np.float64)
    scaled_to_map = np.array([[1 / map_scale, 0, 0], [0, 1 / map_scale, 0], [0, 0, 1]], dtype=np.float64)
    return scaled_to_map @ homography_scaled @ query_to_scaled


def validate_match_geometry(
    map_corners: np.ndarray,
    query_width: int,
    query_height: int,
    map_image: np.ndarray,
) -> None:
    corner_points = map_corners.astype(np.float32)
    area = abs(float(cv2.contourArea(corner_points)))
    query_area = max(1.0, float(query_width * query_height))
    map_area = max(1.0, float(map_image.shape[0] * map_image.shape[1]))
    if area < query_area * 0.0005:
        raise RuntimeError("Feature match projected area is implausibly small")
    if area > map_area * 0.85:
        raise RuntimeError("Feature match projected area is implausibly large")

    x, y, width, height = cv2.boundingRect(corner_points.astype(np.int32))
    if width <= 0 or height <= 0:
        raise RuntimeError("Feature match produced a degenerate bounding box")
    aspect = width / float(height)
    query_aspect = query_width / float(query_height)
    if aspect < query_aspect / 6.0 or aspect > query_aspect * 6.0:
        raise RuntimeError("Feature match projected aspect ratio is implausible")


def create_feature_detector(max_features: int) -> tuple[str, object, int]:
    if hasattr(cv2, "SIFT_create"):
        return "sift", cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    return "orb", cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING


def template_match(
    map_image: np.ndarray,
    query_image: np.ndarray,
    scales: list[float],
) -> dict[str, object]:
    map_gray = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    map_height, map_width = map_gray.shape[:2]

    best = None
    for scale in scales:
        scaled_width = max(8, int(round(query_gray.shape[1] * scale)))
        scaled_height = max(8, int(round(query_gray.shape[0] * scale)))
        if scaled_width > map_width or scaled_height > map_height:
            continue
        scaled_query = cv2.resize(
            query_gray,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        response = cv2.matchTemplate(map_gray, scaled_query, cv2.TM_CCOEFF_NORMED)
        _min_value, max_value, _min_loc, max_loc = cv2.minMaxLoc(response)
        if best is None or max_value > best["score"]:
            best = {
                "score": float(max_value),
                "top_left": max_loc,
                "width": scaled_width,
                "height": scaled_height,
                "scale": scale,
            }

    if best is None:
        raise RuntimeError(
            "No template scale fits inside the map. Use a smaller query crop or "
            "smaller --template-scales values."
        )

    x, y = best["top_left"]
    width = best["width"]
    height = best["height"]
    corners = [(x, y), (x + width - 1, y), (x + width - 1, y + height - 1), (x, y + height - 1)]
    return {
        "method": "template",
        "score": round(float(best["score"]), 4),
        "center": (float(x + width / 2.0), float(y + height / 2.0)),
        "corners": [(float(px), float(py)) for px, py in corners],
        "scale": best["scale"],
        "top_left": (x, y),
    }


def point_inside_image(point: np.ndarray, image: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    height, width = image.shape[:2]
    return 0 <= x < width and 0 <= y < height


def visualize_match(
    map_image: np.ndarray,
    query_image: np.ndarray,
    match_result: dict[str, object],
    latitude: float,
    longitude: float,
) -> np.ndarray:
    output = map_image.copy()
    corners = np.array(match_result["corners"], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(output, [corners], True, (0, 255, 255), 3, cv2.LINE_AA)
    center_x, center_y = match_result["center"]
    center = (int(round(center_x)), int(round(center_y)))
    cv2.drawMarker(output, center, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 34, 3, cv2.LINE_AA)
    draw_label(
        output,
        f"{match_result['method']} score={match_result['score']} "
        f"GPS={latitude:.7f},{longitude:.7f}",
        (center[0] + 12, center[1] - 12),
        (0, 0, 0),
        (255, 255, 255),
    )

    inset = resize_to_fit(query_image, max_width=360, max_height=240)
    inset_height, inset_width = inset.shape[:2]
    margin = 12
    x0 = margin
    y0 = output.shape[0] - inset_height - margin
    output[y0 : y0 + inset_height, x0 : x0 + inset_width] = inset
    cv2.rectangle(output, (x0, y0), (x0 + inset_width, y0 + inset_height), (255, 255, 255), 2)
    draw_label(output, "query image", (x0 + 8, y0 + 24), (0, 0, 0), (255, 255, 255))
    return output


def resize_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if math.isclose(scale, 1.0):
        return image.copy()
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def main() -> None:
    args = parse_args()
    if args.command == "draw-map":
        command_draw_map(args)
    elif args.command == "pixel":
        command_pixel(args)
    elif args.command == "gps":
        command_gps(args)
    elif args.command == "click":
        command_click(args)
    elif args.command == "match":
        command_match(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
