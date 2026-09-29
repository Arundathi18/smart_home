"""Unified live dashboard - everything in one window for a demo.

Combines every earlier piece: webcam feed with hand landmarks, a clear
hand-detection status, the raw per-frame gesture prediction with its
confidence, the "stable" gesture once it's held long enough, both
simulated device states (LED + fan), and (optionally) the real ESP32
hardware connection status.

Run from the project root (with the virtual environment activated):
    python -m apps.dashboard                                  # simulation only
    python -m apps.dashboard --port /dev/cu.usbserial-XXXX     # + real hardware

If --port is omitted, or the ESP32 can't be reached, the dashboard keeps
working in simulation-only mode - it never blocks a demo just because the
hardware isn't plugged in (or doesn't exist yet).

Same gesture mapping as apps/gesture_control_demo.py:
    OPEN_PALM    LED ON
    FIST         LED OFF
    THUMBS_UP    fan ON
    TWO_FINGERS  fan OFF

Number keys 1-2 still directly toggle a device. Requires a trained model:
python -m apps.train_model. Press 'q' to quit.
"""

import argparse

import cv2
import numpy as np

from gesture_smart_home import config
from gesture_smart_home.vision.hand_tracker import HandTracker
from gesture_smart_home.gestures.features import extract_feature_vector
from gesture_smart_home.ml.model_io import load_model
from gesture_smart_home.ml.stability import StablePredictor
from gesture_smart_home.smart_home.devices import SmartHome
from gesture_smart_home.smart_home.gesture_control import apply_gesture
from gesture_smart_home.ui.panel import render_panel
from gesture_smart_home.esp32_bridge.serial_bridge import ESP32Bridge, ESP32ConnectionError

NOT_CONNECTED_STATUS = "Not connected (simulation only)"


def connect_to_esp32(port):
    """Try to connect to the ESP32. Returns (bridge_or_None, status_text).
    Never raises - a missing/unreachable board just means simulation-only."""
    if not port:
        return None, NOT_CONNECTED_STATUS
    try:
        bridge = ESP32Bridge(port=port)
        bridge.ping()
        return bridge, f"Connected on {port}"
    except ESP32ConnectionError as e:
        print(f"Could not connect to ESP32 on {port}: {e}")
        print("Continuing in simulation-only mode.")
        return None, NOT_CONNECTED_STATUS


def main():
    parser = argparse.ArgumentParser(description="Live gesture-controlled smart home dashboard.")
    parser.add_argument(
        "--port", default=None,
        help="ESP32 serial port, e.g. /dev/cu.usbserial-XXXX (optional - omit to run simulation-only)",
    )
    args = parser.parse_args()

    try:
        pipeline, metadata = load_model()
    except FileNotFoundError as e:
        print(e)
        return

    bridge, hardware_status = connect_to_esp32(args.port)

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
    stabilizer = StablePredictor(
        required_consecutive_frames=config.PREDICTION_STABLE_FRAMES,
        confidence_threshold=config.PREDICTION_CONFIDENCE_THRESHOLD,
    )
    home = SmartHome(config.LIGHT_NAMES, config.FAN_NAME)
    max_key = ord(str(len(home.devices)))
    previous_stable_gesture = None

    print("Gesture smart home dashboard running.")
    print(f"ESP32 hardware: {hardware_status}")
    print("OPEN_PALM=LED on  FIST=LED off  THUMBS_UP=fan on  TWO_FINGERS=fan off")
    print(f"Keys 1-{len(home.devices)} still toggle a device directly. Press 'q' to quit.")

    def sync_hardware_if_connected():
        nonlocal bridge, hardware_status
        if bridge is None:
            return
        try:
            bridge.sync_devices(home)
        except ESP32ConnectionError as e:
            print(f"Lost connection to the ESP32: {e}")
            bridge = None
            hardware_status = "Disconnected (lost connection)"

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)
            tracker.process(frame)
            tracker.draw_landmarks(frame)
            hand_landmarks = tracker.first_hand_landmarks()

            stable_gesture = None
            if hand_landmarks is not None:
                features = extract_feature_vector(hand_landmarks).reshape(1, -1)
                probabilities = pipeline.predict_proba(features)[0]
                best_index = probabilities.argmax()
                predicted_gesture = pipeline.classes_[best_index]
                confidence = probabilities[best_index]
                stable_gesture = stabilizer.update(predicted_gesture, confidence)

                cv2.putText(
                    frame, "Hand Detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_DETECTED, 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, f"Raw: {predicted_gesture} ({confidence:.0%})", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, config.COLOR_TEXT, 2, cv2.LINE_AA,
                )
            else:
                stabilizer.reset()
                cv2.putText(
                    frame, "No Hand Detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_NOT_DETECTED, 2, cv2.LINE_AA,
                )

            # Fire only on the frame the gesture *becomes* stable, not on
            # every frame it stays held.
            if stable_gesture is not None and stable_gesture != previous_stable_gesture:
                apply_gesture(home, stable_gesture)
                sync_hardware_if_connected()
            previous_stable_gesture = stable_gesture

            if stable_gesture:
                cv2.putText(
                    frame, f"Stable gesture: {stable_gesture}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_DETECTED, 2, cv2.LINE_AA,
                )

            hw_color = config.COLOR_DETECTED if bridge is not None else (0, 165, 255)
            cv2.putText(
                frame, f"ESP32: {hardware_status}", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, hw_color, 1, cv2.LINE_AA,
            )

            panel = render_panel(home.devices, height=frame.shape[0])
            combined = np.hstack((frame, panel))
            cv2.imshow(config.DASHBOARD_WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if ord("1") <= key <= max_key:
                home.toggle_by_index(key - ord("1"))
                sync_hardware_if_connected()
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        if bridge is not None:
            bridge.close()


if __name__ == "__main__":
    main()
