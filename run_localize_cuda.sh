#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda. Please initialize conda or update this script." >&2
    exit 1
fi

conda activate uav_contest_env

python scripts/localize_vehicles.py \
    --frame test_image/frame_000161.jpg \
    --device cuda:0 \
    --yolo-batch-size 8 \
    --match-workers 4 \
    "$@"
