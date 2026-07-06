#!/usr/bin/env python3
"""Receive UAV telemetry JSON packets over UDP and print them."""

import argparse
import json
import socket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen for UAV telemetry JSON packets over UDP and print altitude/battery values."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Local IP address to bind. Use 0.0.0.0 to listen on all interfaces. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6001,
        help="Local UDP port to listen on. Default: 6001",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also print the decoded JSON object.",
    )
    return parser.parse_args()


def format_number(value: object, suffix: str) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f} {suffix}"
    return f"N/A {suffix}"


def main() -> None:
    args = parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.host, args.port))
        print(f"Listening for UAV telemetry on {args.host}:{args.port}")

        while True:
            data, address = sock.recvfrom(4096)
            try:
                packet = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                print(f"Invalid packet from {address[0]}:{address[1]}: {exc}")
                continue

            sequence = packet.get("sequence", "N/A")
            timestamp = packet.get("timestamp", "N/A")
            altitude = format_number(packet.get("altitude_m"), "m")
            relative_altitude = format_number(packet.get("relative_altitude_m"), "m")
            battery_voltage = format_number(packet.get("battery_voltage_v"), "V")

            print(
                f"From {address[0]}:{address[1]} | "
                f"#{sequence} | {timestamp} | "
                f"Altitude: {altitude} | "
                f"Relative Altitude: {relative_altitude} | "
                f"Battery Voltage: {battery_voltage}"
            )
            if args.show_raw:
                print(packet)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped telemetry receiver.")
