#!/usr/bin/env python3
"""Progressively stitch extracted frames through the UDIS++ reference code."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the external UDIS2/UDIS++ implementation to progressively "
            "stitch sorted frame images into one mosaic."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("extracted_frames"),
        help="Directory containing extracted frames. Default: extracted_frames",
    )
    parser.add_argument(
        "--udis2-root",
        type=Path,
        default=Path("third_party/UDIS2"),
        help="Path to the cloned nie-lang/UDIS2 repository. Default: third_party/UDIS2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stitched_outputs/udis2"),
        help="Directory for UDIS2 pair outputs and final mosaic. Default: stitched_outputs/udis2",
    )
    parser.add_argument(
        "--output-name",
        default="udis2_mosaic.jpg",
        help="Final mosaic filename inside --output-dir. Default: udis2_mosaic.jpg",
    )
    parser.add_argument(
        "--udis2-python",
        default=sys.executable,
        help="Python executable from the UDIS2 environment. Default: current Python",
    )
    parser.add_argument(
        "--gpu",
        default="-1",
        help="GPU id passed to UDIS2 test scripts. Use -1 for CPU. Default: -1",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Warp adaption iterations for each image pair. Default: 50",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    parser.add_argument(
        "--clean-work",
        action="store_true",
        help="Remove per-pair intermediate folders after the final mosaic is written.",
    )
    parser.add_argument(
        "--max-input-side",
        type=int,
        default=0,
        help=(
            "Downscale each image before UDIS2 so its longest side is at most this many "
            "pixels. Use 0 to keep original resolution. Default: 0"
        ),
    )
    return parser.parse_args()


def find_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def require_file(path: Path, message: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{message}: {path}")


def require_dir(path: Path, message: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{message}: {path}")


def validate_udis2(udis2_root: Path) -> None:
    require_dir(udis2_root, "UDIS2 repository not found")
    require_file(udis2_root / "Warp/Codes/test_other.py", "UDIS2 Warp test entrypoint not found")
    require_file(
        udis2_root / "Composition/Codes/test_other.py",
        "UDIS2 Composition test entrypoint not found",
    )

    warp_checkpoints = sorted((udis2_root / "Warp/model").glob("*.pth"))
    composition_checkpoints = sorted((udis2_root / "Composition/model").glob("*.pth"))
    if not warp_checkpoints:
        raise FileNotFoundError(
            "No Warp checkpoint found. Put the official UDIS2 Warp .pth file "
            f"under {udis2_root / 'Warp/model'}"
        )
    if not composition_checkpoints:
        raise FileNotFoundError(
            "No Composition checkpoint found. Put the official UDIS2 Composition .pth file "
            f"under {udis2_root / 'Composition/model'}"
        )


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"Running in {cwd}: {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def copy_or_resize_image(source: Path, destination: Path, max_side: int) -> None:
    if max_side <= 0:
        shutil.copy2(source, destination)
        return

    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError(f"Unable to read image: {source}")

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        shutil.copy2(source, destination)
        return

    scale = max_side / longest_side
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    if not cv2.imwrite(str(destination), resized):
        raise RuntimeError(f"Unable to write resized image: {destination}")


def prepare_pair(pair_dir: Path, image1: Path, image2: Path, max_input_side: int) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    copy_or_resize_image(image1, pair_dir / "input1.jpg", max_input_side)
    copy_or_resize_image(image2, pair_dir / "input2.jpg", max_input_side)


def run_udis2_pair(
    args: argparse.Namespace,
    pair_dir: Path,
    env: dict[str, str],
) -> Path:
    pair_path = str(pair_dir.resolve()) + "/"
    warp_codes = args.udis2_root.resolve() / "Warp/Codes"
    composition_codes = args.udis2_root.resolve() / "Composition/Codes"

    run_command(
        [
            args.udis2_python,
            "test_other.py",
            "--gpu",
            args.gpu,
            "--max_iter",
            str(args.max_iter),
            "--path",
            pair_path,
            "--img1_name",
            "input1.jpg",
            "--img2_name",
            "input2.jpg",
        ],
        cwd=warp_codes,
        env=env,
    )
    run_command(
        [
            args.udis2_python,
            "test_other.py",
            "--gpu",
            args.gpu,
            "--path",
            pair_path,
        ],
        cwd=composition_codes,
        env=env,
    )

    composition = pair_dir / "composition.jpg"
    require_file(composition, "UDIS2 composition output not found")
    return composition


def main() -> int:
    args = parse_args()

    try:
        if not args.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        validate_udis2(args.udis2_root)

        images = find_images(args.input_dir)
        if len(images) < 2:
            raise RuntimeError(f"Need at least 2 images in {args.input_dir}")

        if args.output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {args.output_dir}. "
                    "Use --overwrite or choose another --output-dir."
                )
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True)
        torch_home = args.udis2_root / ".torch_cache"
        torch_home.mkdir(parents=True, exist_ok=True)
        udis2_env = os.environ.copy()
        udis2_env["TORCH_HOME"] = str(torch_home.resolve())
        compat_path = Path(__file__).resolve().parent / "udis2_compat"
        existing_pythonpath = udis2_env.get("PYTHONPATH")
        udis2_env["PYTHONPATH"] = (
            str(compat_path)
            if not existing_pythonpath
            else str(compat_path) + os.pathsep + existing_pythonpath
        )
        udis2_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if args.gpu == "-1":
            udis2_env["UDIS2_FORCE_CPU_LOAD"] = "1"

        current = args.output_dir / "seed_input.jpg"
        shutil.copy2(images[0], current)

        for pair_index, next_image in enumerate(images[1:], start=1):
            pair_dir = args.output_dir / f"pair_{pair_index:04d}"
            print(
                f"Pair {pair_index}/{len(images) - 1}: "
                f"{current.name} + {next_image.name}"
            )
            prepare_pair(pair_dir, current, next_image, args.max_input_side)
            composition = run_udis2_pair(args, pair_dir, udis2_env)
            current = args.output_dir / f"progress_{pair_index:04d}.jpg"
            shutil.copy2(composition, current)

        final_output = args.output_dir / args.output_name
        shutil.copy2(current, final_output)
        print(f"Wrote final UDIS++ mosaic: {final_output}")

        if args.clean_work:
            for path in args.output_dir.glob("pair_*"):
                if path.is_dir():
                    shutil.rmtree(path)
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
