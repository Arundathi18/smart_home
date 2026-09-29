"""Gesture class definitions and landmark feature layout.

These constants define the "shape" of the gesture data: how many classes
there are, what they're called, which keyboard key selects each one during
data collection, and how a MediaPipe hand's 21 landmarks are laid out as a
flat feature vector. Both the data-collection app and the prediction app
import from here so they always agree.
"""

GESTURES = ["OPEN_PALM", "FIST", "THUMBS_UP", "TWO_FINGERS"]

# Keyboard key (pressed in apps/collect_dataset.py) -> gesture name.
GESTURE_KEYS = {
    "o": "OPEN_PALM",
    "f": "FIST",
    "t": "THUMBS_UP",
    "2": "TWO_FINGERS",
}

# MediaPipe Hands always returns 21 landmarks, each with (x, y, z).
NUM_LANDMARKS = 21
NUM_COORDS_PER_LANDMARK = 3
FEATURE_COLUMNS = [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]
