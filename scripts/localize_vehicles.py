#!/usr/bin/env python3
"""Detect vehicles in a UAV frame and project detections onto the reference map."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

try:
    from georeference_map import (
        DEFAULT_MAP_IMAGE,
        GeoReference,
        draw_label,
        find_best_match,
        map_query_point,
        read_bgr,
        resize_to_fit,
        show_image,
        write_image,
    )
except ModuleNotFoundError:
    from scripts.georeference_map import (
        DEFAULT_MAP_IMAGE,
        GeoReference,
        draw_label,
        find_best_match,
        map_query_point,
        read_bgr,
        resize_to_fit,
        show_image,
        write_image,
    )


DEFAULT_FRAME = Path("extracted_frames/frame_000051.jpg")
DEFAULT_OUTPUT_ROOT = Path("vehicle_localization_outputs")
DEFAULT_YOLO_MODEL = "yolo26m.pt"
DEFAULT_VEHICLE_CLASSES = "car,bus,truck"


@dataclass
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str
    detector: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect vehicles in a UAV frame, localize their centers on "
            "衛星影像/aerial_gps_range_clean.png, and save demo visualizations."
        )
    )
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--map-image", type=Path, default=DEFAULT_MAP_IMAGE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: vehicle_localization_outputs/<frame filename stem>."
        ),
    )

    parser.add_argument(
        "--detector",
        choices=("auto", "yolo", "white-heuristic"),
        default="auto",
        help="auto tries YOLO first and falls back to a white-vehicle heuristic.",
    )
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--vehicle-classes", default=DEFAULT_VEHICLE_CLASSES)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--model-iou", type=float, default=0.55)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=960)
    parser.add_argument("--tile-overlap", type=int, default=240)
    parser.add_argument(
        "--tile-upscales",
        default="1,2",
        help="Comma-separated tile upscales. Use larger values when cars are tiny.",
    )
    parser.add_argument("--max-detections", type=int, default=20)

    parser.add_argument("--heuristic-min-area", type=float, default=450.0)
    parser.add_argument("--heuristic-max-area", type=float, default=16000.0)

    parser.add_argument(
        "--map-method",
        choices=("auto", "feature", "template"),
        default="template",
        help=(
            "Reference-map matching method. Template is the safer default for "
            "frame_000051 because full-frame SIFT can lock onto repeated fields."
        ),
    )
    parser.add_argument(
        "--orientations",
        choices=("none", "rotations", "flips", "all"),
        default="rotations",
    )
    parser.add_argument("--template-scales", default="0.12,0.10,0.08,0.07,0.06,0.05,0.04")
    parser.add_argument("--min-template-score", type=float, default=0.45)
    parser.add_argument(
        "--orientation-switch-margin",
        type=float,
        default=0.08,
        help=(
            "When testing rotations, keep identity unless the best rotated match exceeds "
            "identity by this score margin. Default: 0.08"
        ),
    )
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=10)

    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an OpenCV window showing the process overview image after saving outputs.",
    )
    return parser.parse_args()


def parse_float_list(raw: str, argument_name: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{argument_name} must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError(f"{argument_name} values must be positive")
    return values


def parse_class_filter(raw: str) -> set[str] | None:
    if raw.strip().lower() in {"", "all", "*"}:
        return None
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def generate_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("--tile-overlap must be non-negative and smaller than --tile-size")

    step = tile_size - overlap
    x_starts = list(range(0, max(width - tile_size, 0) + 1, step))
    y_starts = list(range(0, max(height - tile_size, 0) + 1, step))
    if not x_starts or x_starts[-1] != max(width - tile_size, 0):
        x_starts.append(max(width - tile_size, 0))
    if not y_starts or y_starts[-1] != max(height - tile_size, 0):
        y_starts.append(max(height - tile_size, 0))

    tiles = []
    for y in y_starts:
        for x in x_starts:
            tiles.append((x, y, min(tile_size, width - x), min(tile_size, height - y)))
    return tiles


def detect_with_yolo(
    image: np.ndarray,
    args: argparse.Namespace,
    upscales: list[float],
    class_filter: set[str] | None,
) -> list[Detection]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install requirements or use "
            "--detector white-heuristic for an offline demo."
        ) from exc

    model = YOLO(args.yolo_model)
    names = getattr(model, "names", {})
    tiles = generate_tiles(image.shape[1], image.shape[0], args.tile_size, args.tile_overlap)
    detections: list[Detection] = []

    for tile_x, tile_y, tile_width, tile_height in tiles:
        tile = image[tile_y : tile_y + tile_height, tile_x : tile_x + tile_width]
        for upscale in upscales:
            if math.isclose(upscale, 1.0):
                model_input = tile
            else:
                model_input = cv2.resize(
                    tile,
                    (max(1, round(tile_width * upscale)), max(1, round(tile_height * upscale))),
                    interpolation=cv2.INTER_CUBIC,
                )

            results = model.predict(
                source=model_input,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.model_iou,
                verbose=False,
            )
            if not results:
                continue
            boxes = results[0].boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for bbox, confidence, class_id in zip(xyxy, confs, classes):
                class_name = str(names.get(int(class_id), class_id)).lower()
                if class_filter is not None and class_name not in class_filter:
                    continue
                x1, y1, x2, y2 = [float(value) / upscale for value in bbox]
                detections.append(
                    Detection(
                        bbox_xyxy=(
                            clamp(x1 + tile_x, 0, image.shape[1] - 1),
                            clamp(y1 + tile_y, 0, image.shape[0] - 1),
                            clamp(x2 + tile_x, 0, image.shape[1] - 1),
                            clamp(y2 + tile_y, 0, image.shape[0] - 1),
                        ),
                        confidence=float(confidence),
                        class_name=class_name,
                        detector=f"yolo:{args.yolo_model}:tile_x{upscale:g}",
                    )
                )

    return apply_nms(detections, args.nms_iou, args.max_detections)


def detect_white_vehicle_candidates(image: np.ndarray, args: argparse.Namespace) -> list[Detection]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 165), (180, 80, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        bbox_area = float(width * height)
        if bbox_area < args.heuristic_min_area or bbox_area > args.heuristic_max_area:
            continue
        if width < 12 or height < 12:
            continue
        aspect = width / float(height)
        if not 0.30 <= aspect <= 4.50:
            continue
        contour_area = cv2.contourArea(contour)
        extent = contour_area / bbox_area if bbox_area else 0.0
        if extent < 0.28:
            continue
        confidence = min(0.99, 0.30 + extent * 0.45 + min(bbox_area / 8000.0, 1.0) * 0.20)
        detections.append(
            Detection(
                bbox_xyxy=(float(x), float(y), float(x + width), float(y + height)),
                confidence=float(confidence),
                class_name="white_vehicle_candidate",
                detector="white-heuristic",
            )
        )
    return apply_nms(detections, args.nms_iou, args.max_detections)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def apply_nms(detections: list[Detection], iou_threshold: float, max_count: int) -> list[Detection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda det: det.confidence, reverse=True)
    kept: list[Detection] = []
    for detection in detections:
        if all(bbox_iou(detection.bbox_xyxy, kept_detection.bbox_xyxy) <= iou_threshold for kept_detection in kept):
            kept.append(detection)
        if len(kept) >= max_count:
            break
    return kept


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union_area = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - inter_area
    )
    return inter_area / union_area if union_area else 0.0


def localize_frame_on_map(
    frame: np.ndarray,
    map_image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    match_args = SimpleNamespace(
        method=args.map_method,
        orientations=args.orientations,
        max_features=args.max_features,
        ratio=args.ratio,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        min_template_score=args.min_template_score,
        template_scales=args.template_scales,
        orientation_switch_margin=args.orientation_switch_margin,
    )
    return find_best_match(map_image, frame, match_args)


def draw_detections_on_frame(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = image.copy()
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        color = (0, 0, 255) if detection.detector.startswith("yolo:") else (0, 180, 255)
        callout_box = expand_box_for_display(
            detection.bbox_xyxy,
            image_width=image.shape[1],
            image_height=image.shape[0],
            min_width=160,
            min_height=160,
        )
        cx1, cy1, cx2, cy2 = [int(round(value)) for value in callout_box]
        cv2.rectangle(output, (cx1, cy1), (cx2, cy2), color, 6, cv2.LINE_AA)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 3, cv2.LINE_AA)
        center_x, center_y = detection.center
        cv2.drawMarker(
            output,
            (int(round(center_x)), int(round(center_y))),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            28,
            2,
            cv2.LINE_AA,
        )
        draw_label(
            output,
            f"veh_{index:03d} {detection.class_name} {detection.confidence:.2f}",
            (x1, max(24, y1 - 8)),
            (0, 0, 0),
            (255, 255, 255),
        )
    draw_detection_insets(output, detections)
    return output


def expand_box_for_display(
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    min_width: int,
    min_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = max(x2 - x1, float(min_width))
    height = max(y2 - y1, float(min_height))
    return (
        clamp(center_x - width / 2.0, 0, image_width - 1),
        clamp(center_y - height / 2.0, 0, image_height - 1),
        clamp(center_x + width / 2.0, 0, image_width - 1),
        clamp(center_y + height / 2.0, 0, image_height - 1),
    )


def draw_detection_insets(image: np.ndarray, detections: list[Detection], max_insets: int = 3) -> None:
    if not detections:
        return

    inset_width = min(720, max(360, image.shape[1] // 5))
    inset_height = min(430, max(240, image.shape[0] // 5))
    margin = 28
    gap = 18

    for index, detection in enumerate(detections[:max_insets], start=1):
        x1, y1, x2, y2 = detection.bbox_xyxy
        center_x, center_y = detection.center
        crop_half_side = max(120.0, (max(x2 - x1, y2 - y1) * 4.0))
        crop_x1 = int(clamp(center_x - crop_half_side, 0, image.shape[1] - 1))
        crop_y1 = int(clamp(center_y - crop_half_side, 0, image.shape[0] - 1))
        crop_x2 = int(clamp(center_x + crop_half_side, 0, image.shape[1]))
        crop_y2 = int(clamp(center_y + crop_half_side, 0, image.shape[0]))
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if crop.size == 0:
            continue

        resized = cv2.resize(crop, (inset_width, inset_height), interpolation=cv2.INTER_CUBIC)
        scale_x = inset_width / max(1, crop_x2 - crop_x1)
        scale_y = inset_height / max(1, crop_y2 - crop_y1)
        local_box = (
            int(round((x1 - crop_x1) * scale_x)),
            int(round((y1 - crop_y1) * scale_y)),
            int(round((x2 - crop_x1) * scale_x)),
            int(round((y2 - crop_y1) * scale_y)),
        )
        cv2.rectangle(resized, local_box[:2], local_box[2:], (0, 255, 255), 3, cv2.LINE_AA)
        cv2.drawMarker(
            resized,
            (int(round((center_x - crop_x1) * scale_x)), int(round((center_y - crop_y1) * scale_y))),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            28,
            2,
            cv2.LINE_AA,
        )
        draw_label(
            resized,
            f"zoom veh_{index:03d} {detection.class_name} {detection.confidence:.2f}",
            (12, 28),
            (0, 0, 0),
            (255, 255, 255),
        )

        inset_x2 = image.shape[1] - margin
        inset_x1 = inset_x2 - inset_width
        inset_y1 = margin + (index - 1) * (inset_height + gap)
        inset_y2 = inset_y1 + inset_height
        if inset_y2 > image.shape[0] - margin:
            break
        image[inset_y1:inset_y2, inset_x1:inset_x2] = resized
        cv2.rectangle(image, (inset_x1, inset_y1), (inset_x2, inset_y2), (255, 255, 255), 5)
        cv2.rectangle(image, (inset_x1, inset_y1), (inset_x2, inset_y2), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(
            image,
            (int(round(center_x)), int(round(center_y))),
            (inset_x1, inset_y1 + inset_height // 2),
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )


def draw_map_overlay(
    map_image: np.ndarray,
    match_result: dict[str, Any],
    localized_detections: list[dict[str, Any]],
) -> np.ndarray:
    output = map_image.copy()
    corners = np.array(match_result["corners"], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(output, [corners], True, (0, 255, 255), 3, cv2.LINE_AA)

    for item in localized_detections:
        if item.get("map_pixel") is None:
            continue
        pixel = item["map_pixel"]
        x = int(round(pixel["x"]))
        y = int(round(pixel["y"]))
        cv2.drawMarker(output, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 34, 3, cv2.LINE_AA)
        gps = item["gps"]
        draw_label(
            output,
            f"{item['vehicle_id']} {gps['latitude']:.7f},{gps['longitude']:.7f}",
            (x + 10, y - 10),
            (0, 0, 0),
            (255, 255, 255),
        )

    center_x, center_y = match_result["center"]
    draw_label(
        output,
        f"frame match: {match_result['method']} score={float(match_result['score']):.4f}",
        (int(round(center_x)) + 12, int(round(center_y)) - 12),
        (0, 0, 0),
        (255, 255, 255),
    )
    return output


def make_process_board(
    frame_overlay: np.ndarray,
    map_overlay: np.ndarray,
    localized_detections: list[dict[str, Any]],
    match_result: dict[str, Any],
) -> np.ndarray:
    left = resize_to_fit(frame_overlay, max_width=900, max_height=520)
    right = resize_to_fit(map_overlay, max_width=900, max_height=520)
    image_row_height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + right.shape[1] + 18
    panel_height = coordinate_panel_height(localized_detections)
    board = np.full((image_row_height + 46 + panel_height, width, 3), 245, dtype=np.uint8)
    draw_label(board, "1 UAV frame detections", (12, 28), (0, 0, 0), (255, 255, 255))
    draw_label(board, "2 projected map coordinates", (left.shape[1] + 30, 28), (0, 0, 0), (255, 255, 255))
    board[46 : 46 + left.shape[0], 0 : left.shape[1]] = left
    x_offset = left.shape[1] + 18
    board[46 : 46 + right.shape[0], x_offset : x_offset + right.shape[1]] = right
    draw_coordinate_panel(
        board,
        top=46 + image_row_height,
        width=width,
        localized_detections=localized_detections,
        match_result=match_result,
    )
    return board


def coordinate_panel_height(localized_detections: list[dict[str, Any]]) -> int:
    visible_rows = max(1, min(len(localized_detections), 6))
    overflow_note_height = 34 if len(localized_detections) > visible_rows else 0
    return 76 + visible_rows * 34 + overflow_note_height


def draw_coordinate_panel(
    board: np.ndarray,
    top: int,
    width: int,
    localized_detections: list[dict[str, Any]],
    match_result: dict[str, Any],
) -> None:
    panel = board[top:, :]
    panel[:] = (255, 255, 255)
    cv2.line(board, (0, top), (width, top), (180, 180, 180), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 16
    table_left = margin
    table_right = width - margin
    columns = [
        ("id", table_left),
        ("class / conf", table_left + 112),
        ("image center", table_left + 300),
        ("WGS84 lat, lon", table_left + 500),
        ("TWD97 x, y", table_left + 850),
    ]

    cv2.putText(
        board,
        f"Vehicle coordinates  |  map match: {match_result['method']} score={float(match_result['score']):.4f}",
        (margin, top + 26),
        font,
        0.62,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    header_top = top + 42
    header_bottom = top + 70
    cv2.rectangle(board, (table_left, header_top), (table_right, header_bottom), (238, 242, 246), -1)
    cv2.rectangle(board, (table_left, header_top), (table_right, header_bottom), (190, 190, 190), 1)

    for label, x in columns:
        cv2.putText(board, label, (x, top + 61), font, 0.48, (65, 65, 65), 1, cv2.LINE_AA)
    for _label, x in columns[1:]:
        cv2.line(board, (x - 12, header_top), (x - 12, board.shape[0] - 12), (220, 220, 220), 1, cv2.LINE_AA)

    if not localized_detections:
        cv2.putText(
            board,
            "No vehicle detections.",
            (margin, top + 100),
            font,
            0.55,
            (0, 0, 180),
            1,
            cv2.LINE_AA,
        )
        return

    visible_items = localized_detections[:6]
    for row_index, item in enumerate(visible_items):
        row_top = header_bottom + row_index * 34
        row_bottom = row_top + 34
        row_y = row_top + 23
        bg_color = (255, 255, 255) if row_index % 2 == 0 else (248, 250, 252)
        cv2.rectangle(board, (table_left, row_top), (table_right, row_bottom), bg_color, -1)
        cv2.line(board, (table_left, row_bottom), (table_right, row_bottom), (226, 226, 226), 1, cv2.LINE_AA)

        center = item["center_image_pixel"]
        gps = item.get("gps")
        twd97 = item.get("twd97")
        if gps and twd97:
            gps_text = f"{gps['latitude']:.8f}, {gps['longitude']:.8f}"
            twd97_text = f"{twd97['x']:.3f}, {twd97['y']:.3f}"
        else:
            gps_text = "localization failed"
            twd97_text = "-"
        values = [
            item["vehicle_id"],
            f"{item['class_name'][:14]} / {item['confidence']:.2f}",
            f"({center['x']:.1f}, {center['y']:.1f})",
            gps_text,
            twd97_text,
        ]
        for (_label, x), value in zip(columns, values):
            cv2.putText(board, value, (x, row_y), font, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

    remaining = len(localized_detections) - len(visible_items)
    if remaining > 0:
        cv2.putText(
            board,
            f"... {remaining} more detections are in vehicle_localization.json / .csv",
            (margin, header_bottom + len(visible_items) * 34 + 25),
            font,
            0.52,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )


def save_detection_crops(
    image: np.ndarray,
    detections: list[Detection],
    output_dir: Path,
) -> list[str]:
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_paths: list[str] = []
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        pad = 24
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)
        crop = image[y1:y2, x1:x2]
        crop_path = crop_dir / f"veh_{index:03d}.jpg"
        write_image(crop_path, crop)
        crop_paths.append(str(crop_path))
    return crop_paths


def wgs84_to_twd97(latitude: float, longitude: float) -> tuple[float, float]:
    """Convert WGS84 lat/lon to TWD97 TM2 zone 121 easting/northing."""
    axis_a = 6378137.0
    axis_b = 6356752.314245
    lon0 = math.radians(121.0)
    k0 = 0.9999
    dx = 250000.0

    lat = math.radians(latitude)
    lon = math.radians(longitude)
    eccentricity_sq = 1.0 - (axis_b * axis_b) / (axis_a * axis_a)
    second_eccentricity_sq = eccentricity_sq / (1.0 - eccentricity_sq)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)

    n_radius = axis_a / math.sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = second_eccentricity_sq * cos_lat * cos_lat
    a = cos_lat * (lon - lon0)
    meridian_arc = axis_a * (
        (1 - eccentricity_sq / 4 - 3 * eccentricity_sq**2 / 64 - 5 * eccentricity_sq**3 / 256)
        * lat
        - (3 * eccentricity_sq / 8 + 3 * eccentricity_sq**2 / 32 + 45 * eccentricity_sq**3 / 1024)
        * math.sin(2 * lat)
        + (15 * eccentricity_sq**2 / 256 + 45 * eccentricity_sq**3 / 1024) * math.sin(4 * lat)
        - (35 * eccentricity_sq**3 / 3072) * math.sin(6 * lat)
    )

    x = dx + k0 * n_radius * (
        a
        + (1 - t + c) * a**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * second_eccentricity_sq) * a**5 / 120
    )
    y = k0 * (
        meridian_arc
        + n_radius
        * tan_lat
        * (
            a**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * second_eccentricity_sq) * a**6 / 720
        )
    )
    return x, y


def build_report(
    frame_path: Path,
    args: argparse.Namespace,
    detections: list[Detection],
    localized_items: list[dict[str, Any]],
    match_result: dict[str, Any],
    match_warnings: list[str],
    detector_warning: str | None,
    outputs: dict[str, str],
) -> dict[str, Any]:
    warnings = list(match_warnings)
    if detector_warning:
        warnings.append(detector_warning)
    if float(match_result["score"]) < args.min_template_score and match_result["method"] == "template":
        warnings.append(
            "Map template score is below threshold; treat map coordinates as rough demo output."
        )

    return {
        "frame": str(frame_path),
        "map_image": str(args.map_image),
        "detector": args.detector,
        "yolo_model": args.yolo_model,
        "detections_count": len(detections),
        "map_match": {
            "method": match_result["method"],
            "score": match_result["score"],
            "orientation": match_result["orientation"],
            "center_pixel": {
                "x": round(float(match_result["center"][0]), 3),
                "y": round(float(match_result["center"][1]), 3),
            },
            "corners_pixel": [
                {"x": round(float(x), 3), "y": round(float(y), 3)}
                for x, y in match_result["corners"]
            ],
            "orientation_scores": match_result.get("orientation_scores", []),
        },
        "vehicles": localized_items,
        "outputs": outputs,
        "warnings": warnings,
        "deferred_work": [
            "Train or fine-tune a competition-specific detector for the roof marker X.",
            "Use drone GPS/IMU or manually selected stable-landmark ROI to constrain map matching.",
            "Add multi-frame tracking so the same vehicle is assigned a stable ID over time.",
        ],
    }


def localize_detections(
    detections: list[Detection],
    match_result: dict[str, Any],
    georef: GeoReference,
    frame_shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    frame_height, frame_width = frame_shape[:2]
    query_roi = (0, 0, frame_width, frame_height)
    for index, detection in enumerate(detections, start=1):
        center_x, center_y = detection.center
        item: dict[str, Any] = {
            "vehicle_id": f"veh_{index:03d}",
            "class_name": detection.class_name,
            "confidence": round(detection.confidence, 4),
            "detector": detection.detector,
            "bbox_xyxy": [round(float(value), 3) for value in detection.bbox_xyxy],
            "center_image_pixel": {"x": round(center_x, 3), "y": round(center_y, 3)},
        }
        try:
            map_x, map_y = map_query_point(match_result, (center_x, center_y), query_roi)
            if not georef.contains_pixel(map_x, map_y):
                raise ValueError(f"mapped point ({map_x:.2f}, {map_y:.2f}) outside reference map")
            latitude, longitude = georef.pixel_to_gps(map_x, map_y)
            twd97_x, twd97_y = wgs84_to_twd97(latitude, longitude)
            item["map_pixel"] = {"x": round(map_x, 3), "y": round(map_y, 3)}
            item["gps"] = {"latitude": round(latitude, 8), "longitude": round(longitude, 8)}
            item["twd97"] = {"x": round(twd97_x, 3), "y": round(twd97_y, 3)}
        except Exception as exc:
            item["map_pixel"] = None
            item["gps"] = None
            item["twd97"] = None
            item["localization_error"] = str(exc)
        items.append(item)
    return items


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "vehicle_id",
                "class_name",
                "confidence",
                "bbox_xyxy",
                "center_image_x",
                "center_image_y",
                "map_x",
                "map_y",
                "latitude",
                "longitude",
                "twd97_x",
                "twd97_y",
            ],
        )
        writer.writeheader()
        for item in items:
            map_pixel = item.get("map_pixel") or {}
            gps = item.get("gps") or {}
            twd97 = item.get("twd97") or {}
            center = item["center_image_pixel"]
            writer.writerow(
                {
                    "vehicle_id": item["vehicle_id"],
                    "class_name": item["class_name"],
                    "confidence": item["confidence"],
                    "bbox_xyxy": json.dumps(item["bbox_xyxy"]),
                    "center_image_x": center["x"],
                    "center_image_y": center["y"],
                    "map_x": map_pixel.get("x"),
                    "map_y": map_pixel.get("y"),
                    "latitude": gps.get("latitude"),
                    "longitude": gps.get("longitude"),
                    "twd97_x": twd97.get("x"),
                    "twd97_y": twd97.get("y"),
                }
            )


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / args.frame.stem
    upscales = parse_float_list(args.tile_upscales, "--tile-upscales")
    class_filter = parse_class_filter(args.vehicle_classes)

    frame = read_bgr(args.frame)
    map_image = read_bgr(args.map_image)
    georef = GeoReference()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector_warning = None
    if args.detector in {"auto", "yolo"}:
        try:
            detections = detect_with_yolo(frame, args, upscales, class_filter)
        except RuntimeError as exc:
            if args.detector == "yolo":
                raise
            detector_warning = str(exc)
            detections = detect_white_vehicle_candidates(frame, args)
    else:
        detections = detect_white_vehicle_candidates(frame, args)

    match_result, match_warnings = localize_frame_on_map(frame, map_image, args)
    localized_items = localize_detections(detections, match_result, georef, frame.shape)

    frame_overlay = draw_detections_on_frame(frame, detections)
    map_overlay = draw_map_overlay(map_image, match_result, localized_items)
    process_board = make_process_board(frame_overlay, map_overlay, localized_items, match_result)

    frame_overlay_path = args.output_dir / "01_frame_vehicle_detections.jpg"
    map_overlay_path = args.output_dir / "02_map_vehicle_coordinates.jpg"
    board_path = args.output_dir / "03_process_overview.jpg"
    json_path = args.output_dir / "vehicle_localization.json"
    csv_path = args.output_dir / "vehicle_localization.csv"

    write_image(frame_overlay_path, frame_overlay)
    write_image(map_overlay_path, map_overlay)
    write_image(board_path, process_board)

    crop_paths = save_detection_crops(frame, detections, args.output_dir) if args.save_crops else []
    for item, crop_path in zip(localized_items, crop_paths):
        item["crop"] = crop_path

    outputs = {
        "frame_overlay": str(frame_overlay_path),
        "map_overlay": str(map_overlay_path),
        "process_overview": str(board_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }
    report = build_report(
        args.frame,
        args,
        detections,
        localized_items,
        match_result,
        match_warnings,
        detector_warning,
        outputs,
    )
    write_json(json_path, report)
    write_csv(csv_path, localized_items)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.show:
        show_image("vehicle localization process overview", process_board)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
