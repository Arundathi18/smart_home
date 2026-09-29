"""Gesture dataset collection app.

Records normalized MediaPipe hand-landmark feature vectors, labeled with one
of four gesture classes, into a local dataset (data/landmarks.csv and
data/sessions.json). apps/train_model.py trains a classifier on this data
later.

Run from the project root (with the virtual environment activated):
    python -m apps.collect_dataset

Controls:
    o / f / t / 2   select the current gesture
                    (OPEN_PALM / FIST / THUMBS_UP / TWO_FINGERS)
    SPACE           start/stop recording samples for the selected gesture
    d               delete all samples for the selected gesture (confirm y/n)
    r               reset - delete the ENTIRE dataset (confirm y/n)
    q               quit

Tip: collect data across at least 3 separate runs of this app (e.g. on
different days, or with your hand at a different distance/angle from the
camera each time) - apps/train_model.py needs at least 3 recording sessions
so it can hold out a genuinely unseen session for testing.
"""

import cv2
import numpy as np

from gesture_smart_home import config
from gesture_smart_home.vision.hand_tracker import HandTracker
from gesture_smart_home.gestures.constants import GESTURES, GESTURE_KEYS
from gesture_smart_home.gestures.features import extract_feature_vector
from gesture_smart_home.gestures.dataset import GestureDataset
from gesture_smart_home.ui.dataset_panel import render_dataset_panel


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
    dataset = GestureDataset()
    session_id = dataset.new_session()
    session_sample_count = 0
    counts = dataset.counts_by_gesture()

    selected_gesture = GESTURES[0]
    recording = False
    pending_confirm = None  # None, "reset", or "delete"

    print(f"Gesture dataset collection running. Session ID: {session_id}")
    print("Keys: o/f/t/2 select gesture, SPACE record on/off, d delete gesture, r reset all, q quit.")

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

            if recording and hand_landmarks is not None and pending_confirm is None:
                features = extract_feature_vector(hand_landmarks)
                dataset.append_sample(session_id, selected_gesture, features)
                counts[selected_gesture] += 1
                session_sample_count += 1

            detected = hand_landmarks is not None
            status_text = "Hand Detected" if detected else "No Hand Detected"
            status_color = config.COLOR_DETECTED if detected else config.COLOR_NOT_DETECTED
            cv2.putText(
                frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"Selected gesture: {selected_gesture}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2, cv2.LINE_AA,
            )

            if pending_confirm == "reset":
                confirm_text = "Reset ENTIRE dataset? This cannot be undone. (y/n)"
                cv2.putText(frame, confirm_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            elif pending_confirm == "delete":
                confirm_text = f"Delete all '{selected_gesture}' samples? (y/n)"
                cv2.putText(frame, confirm_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

            panel = render_dataset_panel(GESTURES, selected_gesture, counts, recording, height=frame.shape[0])
            combined = np.hstack((frame, panel))
            cv2.imshow(config.DATASET_WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF

            if pending_confirm is not None:
                if key == ord("y"):
                    if pending_confirm == "reset":
                        dataset.reset()
                        counts = {g: 0 for g in GESTURES}
                        session_sample_count = 0
                        print("Dataset reset: all samples deleted.")
                    elif pending_confirm == "delete":
                        dataset.delete_gesture(selected_gesture)
                        counts[selected_gesture] = 0
                        print(f"Deleted all samples for {selected_gesture}.")
                    recording = False
                    pending_confirm = None
                elif key == ord("n"):
                    pending_confirm = None
                continue  # ignore every other key while a confirmation is pending

            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording
            elif key == ord("r"):
                pending_confirm = "reset"
                recording = False
            elif key == ord("d"):
                pending_confirm = "delete"
                recording = False
            elif key != 255 and chr(key) in GESTURE_KEYS:
                selected_gesture = GESTURE_KEYS[chr(key)]
                recording = False  # avoid accidentally mislabeling a frame on switch
    finally:
        dataset.end_session(session_id, sample_count=session_sample_count)
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Session {session_id} ended. Recorded {session_sample_count} samples this run.")
    print("Totals by gesture:", dataset.counts_by_gesture())


if __name__ == "__main__":
    main()
