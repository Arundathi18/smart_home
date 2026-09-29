"""Combined foundation demo: webcam hand tracking + device panel, side by side.

This is the integration point that the gesture classifier and ESP32 code
will eventually plug into: hand landmarks feed a classifier, the
classifier's predicted gesture toggles a device, and the device state gets
sent to the ESP32. None of that exists yet - for now devices are toggled
with the number keys 1-2 so you can see the whole picture working end to
end with placeholder logic.

Run from the project root (with the virtual environment activated):
    python -m apps.combined_demo

Press 1-2 to toggle a device, 'q' to quit.
"""

import cv2
import numpy as np

from gesture_smart_home import config
from gesture_smart_home.vision.hand_tracker import HandTracker
from gesture_smart_home.smart_home.devices import SmartHome
from gesture_smart_home.ui.panel import render_panel


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the webcam. On macOS, check System Settings > "
            "Privacy & Security > Camera and make sure your terminal app "
            "is allowed to use the camera."
        )

    tracker = HandTracker(
        max_hands=config.MAX_NUM_HANDS,
        detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    home = SmartHome(config.LIGHT_NAMES, config.FAN_NAME)
    max_key = ord(str(len(home.devices)))

    print("Combined foundation demo running.")
    print(f"Press 1-{len(home.devices)} to toggle a device, 'q' to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)
            tracker.process(frame)
            tracker.draw_landmarks(frame)

            detected = tracker.hands_detected()
            status_text = "Hand Detected" if detected else "No Hand Detected"
            status_color = config.COLOR_DETECTED if detected else config.COLOR_NOT_DETECTED
            cv2.putText(
                frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA,
            )

            panel = render_panel(home.devices, height=frame.shape[0])
            combined = np.hstack((frame, panel))

            cv2.imshow(config.COMBINED_WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if ord("1") <= key <= max_key:
                home.toggle_by_index(key - ord("1"))
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
