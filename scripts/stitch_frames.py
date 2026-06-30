#!/usr/bin/env python3
"""Stitch extracted UAV frames into one black-background mosaic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stitch sorted frame images into one mosaic without stretching the "
            "final result to a rectangular map. Empty canvas areas stay black."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("extracted_frames"),
        help="Directory containing extracted frames. Default: extracted_frames",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stitched_outputs/mosaic.png"),
        help="Output mosaic path. Default: stitched_outputs/mosaic.png",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use OpenCV CUDA ORB features when available.",
    )
    parser.add_argument(
        "--transform",
        choices=("homography", "affine"),
        default="homography",
        help="Transform model between adjacent frames. Default: homography",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=8000,
        help="Maximum local features per frame. Default: 8000",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio test threshold for feature matches. Default: 0.75",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=30,
        help="Minimum good matches required between adjacent frames. Default: 30",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=4.0,
        help="RANSAC reprojection threshold in pixels. Default: 4.0",
    )
    parser.add_argument(
        "--feather-radius",
        type=float,
        default=60.0,
        help="Blend width near image borders in pixels. Use 0 to disable. Default: 60",
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
        help="Allow overwriting an existing output image.",
    )
    return parser.parse_args()


def find_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def create_feature_backend(use_cuda: bool, max_features: int) -> tuple[str, object, int]:
    if use_cuda:
        cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_device_count < 1:
            raise RuntimeError("CUDA was requested, but OpenCV found no CUDA devices")
        if not hasattr(cv2.cuda, "ORB_create"):
            raise RuntimeError("CUDA ORB is not available in this OpenCV build")
        return "cuda_orb", cv2.cuda.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING

    if hasattr(cv2, "SIFT_create"):
        return "sift", cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2

    return "orb", cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING


def detect_features(
    image: np.ndarray,
    backend_name: str,
    detector: object,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if backend_name == "cuda_orb":
        gpu_gray = cv2.cuda_GpuMat()
        gpu_gray.upload(gray)
        keypoints_gpu, descriptors_gpu = detector.detectAndComputeAsync(gpu_gray, None)
        keypoints = detector.convert(keypoints_gpu)
        descriptors = descriptors_gpu.download() if descriptors_gpu is not None else None
        return keypoints, descriptors

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(
    previous_features: tuple[list[cv2.KeyPoint], np.ndarray | None],
    current_features: tuple[list[cv2.KeyPoint], np.ndarray | None],
    norm_type: int,
    ratio: float,
) -> list[cv2.DMatch]:
    previous_keypoints, previous_descriptors = previous_features
    current_keypoints, current_descriptors = current_features
    if previous_descriptors is None or current_descriptors is None:
        return []
    if len(previous_keypoints) < 2 or len(current_keypoints) < 2:
        return []

    matcher = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(previous_descriptors, current_descriptors, k=2)
    good_matches = []
    for candidates in raw_matches:
        if len(candidates) != 2:
            continue
        best, second_best = candidates
        if best.distance < ratio * second_best.distance:
            good_matches.append(best)
    return good_matches


def estimate_current_to_previous(
    previous_keypoints: list[cv2.KeyPoint],
    current_keypoints: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    transform: str,
    ransac_threshold: float,
) -> tuple[np.ndarray | None, int]:
    if not matches:
        return None, 0

    previous_points = np.float32([previous_keypoints[m.queryIdx].pt for m in matches])
    current_points = np.float32([current_keypoints[m.trainIdx].pt for m in matches])

    if transform == "affine":
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            current_points,
            previous_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
        )
        if affine is None:
            return None, 0
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :] = affine
    else:
        matrix, inlier_mask = cv2.findHomography(
            current_points,
            previous_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
        )
        if matrix is None:
            return None, 0

    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return matrix.astype(np.float64), inlier_count


def build_global_transforms(args: argparse.Namespace, image_paths: list[Path]) -> tuple[list[Path], list[tuple[int, int]], list[np.ndarray]]:
    backend_name, detector, norm_type = create_feature_backend(args.cuda, args.max_features)
    print(f"Feature backend: {backend_name}")

    first_image = read_image(image_paths[0])
    previous_features = detect_features(first_image, backend_name, detector)
    used_paths = [image_paths[0]]
    image_sizes = [(first_image.shape[1], first_image.shape[0])]
    transforms = [np.eye(3, dtype=np.float64)]

    for image_path in image_paths[1:]:
        current_image = read_image(image_path)
        current_features = detect_features(current_image, backend_name, detector)
        matches = match_features(previous_features, current_features, norm_type, args.ratio)

        if len(matches) < args.min_matches:
            raise RuntimeError(
                f"Only {len(matches)} good matches between {used_paths[-1].name} "
                f"and {image_path.name}; need at least {args.min_matches}"
            )

        local_transform, inliers = estimate_current_to_previous(
            previous_features[0],
            current_features[0],
            matches,
            args.transform,
            args.ransac_threshold,
        )
        if local_transform is None:
            raise RuntimeError(f"Could not estimate transform for {image_path.name}")

        global_transform = transforms[-1] @ local_transform
        transforms.append(global_transform)
        used_paths.append(image_path)
        image_sizes.append((current_image.shape[1], current_image.shape[0]))
        previous_features = current_features
        print(
            f"Aligned {image_path.name}: {len(matches)} matches, "
            f"{inliers} RANSAC inliers"
        )

    return used_paths, image_sizes, transforms


def compute_canvas(
    image_sizes: list[tuple[int, int]],
    transforms: list[np.ndarray],
    max_canvas_pixels: int,
) -> tuple[np.ndarray, int, int]:
    all_corners = []
    for (width, height), transform in zip(image_sizes, transforms):
        corners = np.float32(
            [[0, 0], [width, 0], [width, height], [0, height]]
        ).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, transform).reshape(-1, 2)
        all_corners.append(warped_corners)

    all_corners_array = np.vstack(all_corners)
    min_x, min_y = np.floor(all_corners_array.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners_array.max(axis=0)).astype(int)
    canvas_width = int(max_x - min_x)
    canvas_height = int(max_y - min_y)
    canvas_pixels = canvas_width * canvas_height

    if canvas_width <= 0 or canvas_height <= 0:
        raise RuntimeError("Computed an invalid output canvas")
    if canvas_pixels > max_canvas_pixels:
        raise RuntimeError(
            f"Output canvas would be {canvas_width}x{canvas_height} "
            f"({canvas_pixels} pixels), above --max-canvas-pixels={max_canvas_pixels}"
        )

    translation = np.array(
        [[1.0, 0.0, -float(min_x)], [0.0, 1.0, -float(min_y)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return translation, canvas_width, canvas_height


def make_weight_mask(mask: np.ndarray, feather_radius: float) -> np.ndarray:
    valid = (mask > 0).astype(np.uint8)
    if feather_radius <= 0:
        return valid.astype(np.float32)

    distance = cv2.distanceTransform(valid, cv2.DIST_L2, 3)
    return np.clip(distance / feather_radius, 0.0, 1.0).astype(np.float32)


def render_mosaic(
    image_paths: list[Path],
    transforms: list[np.ndarray],
    translation: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    feather_radius: float,
) -> np.ndarray:
    accumulator = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weights = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    for index, (image_path, transform) in enumerate(zip(image_paths, transforms), start=1):
        image = read_image(image_path)
        full_transform = translation @ transform
        warped_image = cv2.warpPerspective(
            image,
            full_transform,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(
            source_mask,
            full_transform,
            (canvas_width, canvas_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        weight_mask = make_weight_mask(warped_mask, feather_radius)
        accumulator += warped_image.astype(np.float32) * weight_mask[:, :, None]
        weights += weight_mask
        print(f"Rendered {index}/{len(image_paths)}: {image_path.name}")

    mosaic = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    valid = weights > 0
    mosaic[valid] = np.clip(accumulator[valid] / weights[valid, None], 0, 255).astype(np.uint8)
    return mosaic


def main() -> int:
    args = parse_args()
    try:
        if not args.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output}")

        image_paths = find_images(args.input_dir)
        if len(image_paths) < 2:
            raise RuntimeError(f"Need at least 2 images in {args.input_dir}")

        used_paths, image_sizes, transforms = build_global_transforms(args, image_paths)
        translation, canvas_width, canvas_height = compute_canvas(
            image_sizes,
            transforms,
            args.max_canvas_pixels,
        )
        print(f"Output canvas: {canvas_width}x{canvas_height}")

        mosaic = render_mosaic(
            used_paths,
            transforms,
            translation,
            canvas_width,
            canvas_height,
            args.feather_radius,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), mosaic):
            raise RuntimeError(f"Failed to write output image: {args.output}")
        print(f"Wrote mosaic: {args.output}")
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
