"""Gesture-controlled simulated smart home. Still NOT connected to any real
hardware - this replaces the keyboard-only control in combined_demo.py with
the trained gesture classifier. Each gesture maps to exactly one fixed
action (see gesture_smart_home/smart_home/gesture_control.py):

    OPEN_PALM    turn the LED ON
    FIST         turn the LED OFF
    THUMBS_UP    turn the fan ON
    TWO_FINGERS  turn the fan OFF

A gesture fires once per hold - the instant it becomes "stable" (see
gesture_smart_home/ml/stability.py) - not repeatedly every frame you hold
it, otherwise holding OPEN_PALM would spam turn_on() every frame.

Number keys 1-2 still directly toggle a device, as a manual fallback if a
gesture doesn't land the way you want.

Run from the project root (with the virtual environment activated):
    python -m apps.gesture_control_demo

Requires a trained model: python -m apps.train_model. Press 'q' to quit.
"""

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


def main():
    try:
        pipeline, metadata = load_model()
    except FileNotFoundError as e:
        print(e)
        return

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

    print("Gesture-controlled smart home demo running (no hardware connected).")
    print("OPEN_PALM=LED on  FIST=LED off  THUMBS_UP=fan on  TWO_FINGERS=fan off")
    print(f"Keys 1-{len(home.devices)} still toggle a device directly. Press 'q' to quit.")

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
                    frame, f"Raw: {predicted_gesture} ({confidence:.0%})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2, cv2.LINE_AA,
                )
            else:
                stabilizer.reset()
                cv2.putText(
                    frame, "No Hand Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.COLOR_NOT_DETECTED, 2, cv2.LINE_AA,
                )

            # Fire only on the frame the gesture *becomes* stable, not on
            # every frame it stays held.
            if stable_gesture is not None and stable_gesture != previous_stable_gesture:
                apply_gesture(home, stable_gesture)
            previous_stable_gesture = stable_gesture

            if stable_gesture:
                cv2.putText(
                    frame, f"Stable gesture: {stable_gesture}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_DETECTED, 2, cv2.LINE_AA,
                )

            panel = render_panel(home.devices, height=frame.shape[0])
            combined = np.hstack((frame, panel))
            cv2.imshow(config.GESTURE_CONTROL_WINDOW_NAME, combined)

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
