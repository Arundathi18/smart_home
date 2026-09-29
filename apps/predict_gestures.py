"""Real-time gesture prediction demo. NOT connected to any hardware yet.

Loads the trained model from models/, runs it on live webcam hand
landmarks, and shows the predicted gesture with its confidence. A gesture
only becomes the shown "Stable gesture" once it has been predicted
consistently, above a confidence threshold, for several frames in a row -
this avoids the display flickering between classes on individual noisy
frames. See gesture_smart_home/ml/stability.py.

Run from the project root (with the virtual environment activated):
    python -m apps.predict_gestures

Requires a model trained with 'python -m apps.train_model'. Press 'q' to quit.
"""

import cv2

from gesture_smart_home import config
from gesture_smart_home.vision.hand_tracker import HandTracker
from gesture_smart_home.gestures.features import extract_feature_vector
from gesture_smart_home.ml.model_io import load_model
from gesture_smart_home.ml.stability import StablePredictor


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

    print("Real-time gesture prediction running (no hardware connected). Press 'q' to quit.")

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

            if hand_landmarks is not None:
                features = extract_feature_vector(hand_landmarks).reshape(1, -1)
                probabilities = pipeline.predict_proba(features)[0]
                best_index = probabilities.argmax()
                predicted_gesture = pipeline.classes_[best_index]
                confidence = probabilities[best_index]
                stable_gesture = stabilizer.update(predicted_gesture, confidence)

                cv2.putText(
                    frame, f"Raw: {predicted_gesture} ({confidence:.0%})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_TEXT, 2, cv2.LINE_AA,
                )
                if stable_gesture:
                    cv2.putText(
                        frame, f"Stable gesture: {stable_gesture}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, config.COLOR_DETECTED, 2, cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        frame, "Stabilizing...", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA,
                    )
            else:
                stabilizer.reset()
                cv2.putText(
                    frame, "No Hand Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.COLOR_NOT_DETECTED, 2, cv2.LINE_AA,
                )

            cv2.imshow(config.PREDICTION_WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
