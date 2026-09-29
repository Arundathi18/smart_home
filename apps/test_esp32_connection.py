"""Command-line tool for testing the ESP32 hardware connection.

Use this once the board is flashed with
esp32_firmware/gesture_smart_home_firmware/ and plugged into your Mac over
USB, to verify the wiring and firmware work *before* connecting it to the
rest of the gesture pipeline. No camera or gesture logic involved here.

    python -m apps.test_esp32_connection list
    python -m apps.test_esp32_connection ping <port>
    python -m apps.test_esp32_connection set <port> led on
    python -m apps.test_esp32_connection set <port> fan off
    python -m apps.test_esp32_connection state <port>
"""

import argparse
import sys

from gesture_smart_home.esp32_bridge.serial_bridge import (
    ESP32Bridge,
    ESP32ConnectionError,
    list_serial_ports,
)


def cmd_list(args):
    ports = list_serial_ports()
    if not ports:
        print("No serial ports found. Is the ESP32 plugged in over USB?")
        return
    print("Available serial ports:")
    for device, description in ports:
        print(f"  {device}  ({description})")
    print(
        "\nYour ESP32 usually shows up as something like "
        "/dev/cu.usbserial-XXXX or /dev/cu.SLAB_USBtoUART, depending on its "
        "USB-to-serial chip."
    )


def cmd_ping(args):
    with ESP32Bridge(port=args.port) as bridge:
        bridge.ping()
        print(f"ESP32 on {args.port} responded to PING. Connection OK.")


def cmd_set(args):
    is_on = args.state.lower() == "on"
    with ESP32Bridge(port=args.port) as bridge:
        if args.device == "led":
            bridge.set_led(is_on)
        else:
            bridge.set_fan(is_on)
        print(f"Set {args.device.upper()} to {'ON' if is_on else 'OFF'}.")


def cmd_state(args):
    with ESP32Bridge(port=args.port) as bridge:
        led_is_on, fan_is_on = bridge.get_state()
        print(f"  LED: {'ON' if led_is_on else 'OFF'}")
        print(f"  Fan: {'ON' if fan_is_on else 'OFF'}")


def main():
    parser = argparse.ArgumentParser(description="Test the ESP32 hardware connection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available serial ports")

    ping_parser = subparsers.add_parser("ping", help="Check the ESP32 responds")
    ping_parser.add_argument("port")

    set_parser = subparsers.add_parser("set", help="Set the LED or fan state")
    set_parser.add_argument("port")
    set_parser.add_argument("device", choices=["led", "fan"])
    set_parser.add_argument("state", choices=["on", "off"])

    state_parser = subparsers.add_parser("state", help="Read back the LED and fan state")
    state_parser.add_argument("port")

    args = parser.parse_args()

    try:
        if args.command == "list":
            cmd_list(args)
        elif args.command == "ping":
            cmd_ping(args)
        elif args.command == "set":
            cmd_set(args)
        elif args.command == "state":
            cmd_state(args)
    except ESP32ConnectionError as e:
        print(f"Connection error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
