#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TELEMETRY_PORT=6001

if ! python3 - <<PY
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.bind(("0.0.0.0", $TELEMETRY_PORT))
except OSError as exc:
    print(f"UDP port $TELEMETRY_PORT is not available: {exc}", file=sys.stderr)
    sys.exit(1)
finally:
    sock.close()
PY
then
  echo
  echo "Close the process using UDP $TELEMETRY_PORT before starting opencv_test."
  echo "Check with:"
  echo "  sudo lsof -nP -iUDP:$TELEMETRY_PORT"
  echo
  echo "Common fixes:"
  echo "  pkill -f uav_telemetry_udp_receiver.py"
  echo "  pkill -f opencv_test.py"
  exit 1
fi

python3 scripts/opencv_test.py \
  --source stream.sdp \
  --backend ffmpeg \
  --window-width 1600 \
  --window-height 900 \
  --output-root stream_outputs/contest_stream_test \
  --frame-interval 2 \
  --record \
  --record-segment-seconds 120 \
  --show-detection-result \
  --localize-device cuda:0 \
  --localize-model yolo26x.pt \
  --localize-vehicle-classes car \
  --localize-imgsz 1280 \
  --localize-tile-upscales 1,4 \
  --localize-yolo-batch-size 16 \
  --localize-conf 0.12 \
  --target-verifier \
  --target-verifier-min-score 0.18 \
  --target-verifier-min-white-ratio 0.15 \
  --target-verifier-min-red-pixels 35 \
  --telemetry \
  --telemetry-host 0.0.0.0 \
  --telemetry-port "$TELEMETRY_PORT" \
  --require-telemetry
