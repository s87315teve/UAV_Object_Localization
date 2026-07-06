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
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for Pixhawk heartbeat before reconnecting. Default: 10",
    )
    parser.add_argument(
        "--message-timeout",
        type=float,
        default=5.0,
        help="Seconds without MAVLink messages before reconnecting. Default: 5",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=5.0,
        help="Seconds to wait before retrying after a connection/read failure. Default: 5",
    )
    parser.add_argument(
        "--request-rate-hz",
        type=float,
        default=5.0,
        help="MAVLink message rate to request from Pixhawk for required telemetry. Default: 5",
    )
    parser.add_argument(
        "--telemetry-timeout",
        type=float,
        default=15.0,
        help="Seconds without a complete UDP telemetry packet before reconnecting. Default: 15",
    )
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def close_mavlink(master) -> None:
    if master is None:
        return
    try:
        master.close()
    except Exception as exc:
        print(f"Warning: failed to close MAVLink connection cleanly: {exc}")


def missing_fields(altitude, relative_altitude, battery_voltage) -> str:
    missing = []
    if altitude is None:
        missing.append("altitude_m")
    if relative_altitude is None:
        missing.append("relative_altitude_m")
    if battery_voltage is None:
        missing.append("battery_voltage_v")
    return ", ".join(missing) if missing else "none"


def request_message_interval(master, message_id: int, rate_hz: float) -> None:
    if rate_hz <= 0:
        return
    interval_us = int(1_000_000 / rate_hz)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        interval_us,
        0,
        0,
        0,
        0,
        0,
    )


def request_telemetry_streams(master, rate_hz: float) -> None:
    if rate_hz <= 0:
        return

    try:
        request_message_interval(master, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, rate_hz)
        request_message_interval(master, mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, rate_hz)
    except Exception as exc:
        print(f"Warning: MAV_CMD_SET_MESSAGE_INTERVAL request failed: {exc}")

    try:
        request_rate = max(1, int(rate_hz))
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            request_rate,
            1,
        )
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
            request_rate,
            1,
        )
    except Exception as exc:
        print(f"Warning: legacy MAVLink stream request failed: {exc}")

    print(f"Requested Pixhawk telemetry messages at {rate_hz:.1f} Hz")


def run_telemetry_loop(args: argparse.Namespace, sock: socket.socket, state: dict) -> None:
    min_interval = 0.0 if args.rate_hz <= 0 else 1.0 / args.rate_hz
    altitude = None
    relative_altitude = None
    battery_voltage = None
    last_sent = 0.0
    last_message = time.monotonic()
    last_packet_sent = time.monotonic()
    master = None

    try:
        print(f"Opening Pixhawk serial port {args.serial} at {args.baud} baud...")
        master = mavutil.mavlink_connection(args.serial, baud=args.baud)

        print("Waiting for heartbeat...")
        heartbeat = master.wait_heartbeat(timeout=args.heartbeat_timeout)
        if heartbeat is None:
            raise TimeoutError(f"No heartbeat for {args.heartbeat_timeout:.1f} seconds")
        print(f"Connected! Sending telemetry to {args.host}:{args.port}")
        request_telemetry_streams(master, args.request_rate_hz)
        last_message = time.monotonic()
        last_packet_sent = last_message

        while True:
            msg = master.recv_match(blocking=True, timeout=1.0)
            now = time.monotonic()

            if msg is None:
                if now - last_message >= args.message_timeout:
                    raise TimeoutError(f"No MAVLink messages for {args.message_timeout:.1f} seconds")
                if now - last_packet_sent >= args.telemetry_timeout:
                    raise TimeoutError(
                        f"No complete telemetry packet for {args.telemetry_timeout:.1f} seconds; "
                        f"missing: {missing_fields(altitude, relative_altitude, battery_voltage)}"
                    )
                continue
            last_message = now

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
                if now - last_packet_sent >= args.telemetry_timeout:
                    raise TimeoutError(
                        f"No complete telemetry packet for {args.telemetry_timeout:.1f} seconds; "
                        f"missing: {missing_fields(altitude, relative_altitude, battery_voltage)}"
                    )
                continue

            if altitude is None or relative_altitude is None or battery_voltage is None:
                if now - last_packet_sent >= args.telemetry_timeout:
                    raise TimeoutError(
                        f"No complete telemetry packet for {args.telemetry_timeout:.1f} seconds; "
                        f"missing: {missing_fields(altitude, relative_altitude, battery_voltage)}"
                    )
                continue

            if min_interval > 0 and now - last_sent < min_interval:
                continue

            packet = {
                "timestamp": utc_timestamp(),
                "sequence": state["sequence"],
                "altitude_m": altitude,
                "relative_altitude_m": relative_altitude,
                "battery_voltage_v": battery_voltage,
            }
            payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
            sock.sendto(payload, (args.host, args.port))

            print(
                f"Sent #{state['sequence']}: "
                f"Altitude: {altitude:.2f} m | "
                f"Relative Altitude: {relative_altitude:.2f} m | "
                f"Battery Voltage: {battery_voltage:.2f} V"
            )

            state["sequence"] += 1
            last_sent = now
            last_packet_sent = now
    finally:
        close_mavlink(master)


def main() -> None:
    args = parse_args()
    state = {"sequence": 0}

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        while True:
            try:
                run_telemetry_loop(args, sock, state)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"Telemetry connection failed: {exc}")
                print(f"Reconnecting in {args.reconnect_delay:.1f} seconds...")
                time.sleep(max(0.0, args.reconnect_delay))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped telemetry sender.")
