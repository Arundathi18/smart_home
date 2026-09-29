"""USB serial connection to the ESP32 running the companion firmware in
esp32_firmware/gesture_smart_home_firmware/.

Deliberately synchronous and simple: the gesture-control loop already runs
as a tight webcam loop, so sending one short command and waiting for one
short reply per gesture event is cheap and keeps timing predictable - no
need for background threads or async I/O here.
"""

import serial
import serial.tools.list_ports

from gesture_smart_home.esp32_bridge import protocol

DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT = 1.0  # seconds to wait for a reply before giving up


class ESP32ConnectionError(RuntimeError):
    """Raised when the ESP32 can't be reached or doesn't respond as expected."""


def list_serial_ports():
    """Return [(device_path, description), ...] for every serial port macOS
    currently sees - use this to find which one is your ESP32."""
    return [(p.device, p.description) for p in serial.tools.list_ports.comports()]


class ESP32Bridge:
    def __init__(self, port=None, baudrate=DEFAULT_BAUDRATE, timeout=DEFAULT_TIMEOUT, serial_connection=None):
        """Connect to the ESP32.

        Pass `port` (e.g. "/dev/cu.usbserial-1420") for normal use. The
        `serial_connection` parameter exists so tests can inject a fake
        serial-like object instead of opening a real port.
        """
        if serial_connection is not None:
            self._serial = serial_connection
        else:
            if port is None:
                raise ValueError("Provide either `port` or `serial_connection`.")
            try:
                self._serial = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            except serial.SerialException as e:
                raise ESP32ConnectionError(
                    f"Could not open serial port {port!r}: {e}. Run "
                    "'python -m apps.test_esp32_connection list' to see available ports."
                ) from e

    def _send_line(self, line):
        self._serial.write(line.encode("utf-8"))

    def _read_line(self):
        raw = self._serial.readline()  # blocks up to `timeout` seconds
        if not raw:
            raise ESP32ConnectionError(
                "No response from the ESP32 within the timeout - check the board "
                "is powered, flashed with the firmware, and connected at 115200 baud."
            )
        return raw.decode("utf-8", errors="replace").strip()

    def ping(self):
        self._send_line(protocol.encode_ping())
        reply = self._read_line()
        if reply != "READY":
            raise ESP32ConnectionError(f"Unexpected reply to PING: {reply!r}")
        return True

    def _send_command(self, command):
        self._send_line(protocol.encode_command(command))
        reply = self._read_line()
        if reply.startswith("ERR"):
            raise ESP32ConnectionError(f"ESP32 rejected command: {reply}")
        return reply

    def set_led(self, is_on):
        return self._send_command("LED_ON" if is_on else "LED_OFF")

    def set_fan(self, is_on):
        return self._send_command("FAN_ON" if is_on else "FAN_OFF")

    def get_state(self):
        """Returns (led_is_on, fan_is_on)."""
        self._send_line(protocol.encode_state_query())
        reply = self._read_line()
        return protocol.parse_state_response(reply)

    def sync_devices(self, smart_home):
        """Send the current on/off state of the LED and fan in `smart_home`
        to the ESP32 - e.g. call this right after a gesture changes the
        simulation, so the real hardware always matches it."""
        led_device, fan_device = smart_home.devices
        self.set_led(led_device.is_on)
        self.set_fan(fan_device.is_on)

    def close(self):
        self._serial.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
