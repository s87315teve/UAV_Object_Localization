#!/usr/bin/env python3
"""Read UAV telemetry from Pixhawk and send it as JSON over UDP."""

import argparse
import json
import socket
import time
from datetime import datetime, timezone

from pymavlink import mavutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read altitude and battery telemetry from Pixhawk, then send JSON packets over UDP."
    )
    parser.add_argument(
        "--serial",
        default="/dev/ttyACM0",
        help="Pixhawk serial device. Default: /dev/ttyACM0",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Pixhawk serial baud rate. Default: 115200",
    )
    parser.add_argument(
        "--host",
        default="192.168.1.150",
        help="UDP receiver IP address. Default: 192.168.1.150",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6001,
        help="UDP receiver port. Default: 6001",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=5.0,
        help="Maximum UDP send rate after all fields are available. Default: 5",
    )
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def main() -> None:
    args = parse_args()
    min_interval = 0.0 if args.rate_hz <= 0 else 1.0 / args.rate_hz

    master = mavutil.mavlink_connection(args.serial, baud=args.baud)

    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"Connected! Sending telemetry to {args.host}:{args.port}")

    altitude = None
    relative_altitude = None
    battery_voltage = None
    sequence = 0
    last_sent = 0.0

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        while True:
            msg = master.recv_match(blocking=True)
            if msg is None:
                continue

            msg_type = msg.get_type()
            updated = False

            if msg_type == "GLOBAL_POSITION_INT":
                altitude = msg.alt / 1000.0
                relative_altitude = msg.relative_alt / 1000.0
                updated = True
            elif msg_type == "SYS_STATUS":
                battery_voltage = msg.voltage_battery / 1000.0
                updated = True

            if not updated:
                continue

            if altitude is None or relative_altitude is None or battery_voltage is None:
                continue

            now = time.monotonic()
            if min_interval > 0 and now - last_sent < min_interval:
                continue

            packet = {
                "timestamp": utc_timestamp(),
                "sequence": sequence,
                "altitude_m": altitude,
                "relative_altitude_m": relative_altitude,
                "battery_voltage_v": battery_voltage,
            }
            payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
            sock.sendto(payload, (args.host, args.port))

            print(
                f"Sent #{sequence}: "
                f"Altitude: {altitude:.2f} m | "
                f"Relative Altitude: {relative_altitude:.2f} m | "
                f"Battery Voltage: {battery_voltage:.2f} V"
            )

            sequence += 1
            last_sent = now


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped telemetry sender.")
