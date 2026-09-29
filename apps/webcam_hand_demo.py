"""Webcam hand-tracking demo.

Opens your MacBook's webcam, runs MediaPipe Hands on each frame, draws the
detected hand landmarks, and shows a clear "Hand Detected" / "No Hand
Detected" status on screen. No gesture classification or smart home logic
yet - this just proves the vision pipeline works.

Run from the project root (with the virtual environment activated):
    python -m apps.webcam_hand_demo

Press 'q' in the window to quit.
"""

import cv2

from gesture_smart_home import config
from gesture_smart_home.vision.hand_tracker import HandTracker


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

    print("Webcam hand-tracking demo running. Press 'q' in the window to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror image feels more natural
            tracker.process(frame)
            tracker.draw_landmarks(frame)

            detected = tracker.hands_detected()
            status_text = "Hand Detected" if detected else "No Hand Detected"
            status_color = config.COLOR_DETECTED if detected else config.COLOR_NOT_DETECTED

            cv2.putText(
                frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA,
            )

            cv2.imshow(config.WEBCAM_WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
