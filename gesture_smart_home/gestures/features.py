"""Turns MediaPipe hand landmarks into a feature vector for the classifier.

The key idea: MediaPipe gives landmark coordinates relative to the whole
camera frame, so the same gesture produces very different numbers depending
on where your hand is in the frame and how far it is from the camera. We
fix that with two steps before the numbers ever get used:

1. Translate - subtract the wrist position from every landmark, so the
   wrist becomes (0, 0, 0). This removes any dependence on *where* in the
   frame the hand is.
2. Scale - divide every coordinate by the wrist-to-middle-finger-knuckle
   distance. This removes any dependence on hand size / distance from the
   camera (a hand close to the camera and a hand far away, making the same
   gesture, end up with the same normalized numbers).

This exact function is used both when collecting training data
(apps/collect_dataset.py) and when predicting live (apps/predict_gestures.py),
so the model always sees data preprocessed the same way it was trained on.
"""

import numpy as np

WRIST_INDEX = 0
MIDDLE_FINGER_MCP_INDEX = 9  # base knuckle of the middle finger, used for scale


def landmarks_to_array(hand_landmarks):
    """Convert one MediaPipe hand_landmarks object into a (21, 3) array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float64,
    )


def normalize_landmarks(points):
    """Make a (21, 3) landmark array invariant to hand position and scale."""
    points = points.copy()
    points -= points[WRIST_INDEX]

    scale = np.linalg.norm(points[MIDDLE_FINGER_MCP_INDEX])
    if scale < 1e-6:
        scale = 1e-6  # guard against divide-by-zero on a degenerate frame

    points /= scale
    return points


def extract_feature_vector(hand_landmarks):
    """Full pipeline: MediaPipe landmarks -> flat normalized feature vector.

    Returns a length-63 numpy array (21 landmarks x 3 coordinates), matching
    the column order of gesture_smart_home.gestures.constants.FEATURE_COLUMNS.
    """
    points = landmarks_to_array(hand_landmarks)
    normalized = normalize_landmarks(points)
    return normalized.flatten()
