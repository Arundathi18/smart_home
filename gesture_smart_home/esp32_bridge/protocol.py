"""Wire protocol shared between the Python bridge and the ESP32 firmware.

Two physical devices: one LED (the light) and one fan, driven through one
relay channel. Plain text, newline-terminated commands - easy to read in
the Arduino Serial Monitor while debugging, and easy to parse with
Arduino's Serial.readStringUntil('\\n'). See
esp32_firmware/gesture_smart_home_firmware/ for the matching firmware.

Commands (Python -> ESP32):
    "LED_ON\\n"    turn the LED on
    "LED_OFF\\n"   turn the LED off
    "FAN_ON\\n"    turn the fan on
    "FAN_OFF\\n"   turn the fan off
    "PING\\n"      health check
    "STATE\\n"     ask for the current on/off state of both devices

Responses (ESP32 -> Python):
    "READY\\n"            reply to PING
    "OK:<command>\\n"     command applied successfully, e.g. "OK:LED_ON"
    "STATE:<led>,<fan>\\n" reply to STATE, each 0 or 1
    "ERR:<reason>\\n"     command rejected (unrecognized command, etc.)
"""

COMMANDS = {"LED_ON", "LED_OFF", "FAN_ON", "FAN_OFF"}


def encode_command(command):
    if command not in COMMANDS:
        raise ValueError(f"Unknown command: {command!r}, expected one of {sorted(COMMANDS)}")
    return f"{command}\n"


def encode_ping():
    return "PING\n"


def encode_state_query():
    return "STATE\n"


def parse_state_response(line):
    """Parse a "STATE:0,1" reply into (led_is_on, fan_is_on) booleans."""
    line = line.strip()
    if not line.startswith("STATE:"):
        raise ValueError(f"Not a STATE response: {line!r}")
    values = line[len("STATE:"):].split(",")
    if len(values) != 2:
        raise ValueError(f"Expected 2 values (led,fan), got {len(values)}: {line!r}")
    return values[0] == "1", values[1] == "1"
