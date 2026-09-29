"""Simulated smart home interface - no webcam or gestures yet.

Shows the on/off status of the LED and the fan in one window. Press the
number keys 1-2 to toggle each device, 'q' to quit. This lets you test and
demo the device model on its own before it's wired up to the gesture
classifier and the ESP32.

Run from the project root (with the virtual environment activated):
    python -m apps.smart_home_demo
"""

import cv2

from gesture_smart_home import config
from gesture_smart_home.smart_home.devices import SmartHome
from gesture_smart_home.ui.panel import render_panel


def main():
    home = SmartHome(config.LIGHT_NAMES, config.FAN_NAME)
    max_key = ord(str(len(home.devices)))

    print("Simulated smart home demo running.")
    print(f"Press 1-{len(home.devices)} to toggle a device, 'q' to quit.")

    while True:
        panel = render_panel(home.devices)
        cv2.imshow(config.PANEL_WINDOW_NAME, panel)

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        if ord("1") <= key <= max_key:
            home.toggle_by_index(key - ord("1"))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
