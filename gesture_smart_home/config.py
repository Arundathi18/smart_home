"""Shared configuration for the whole project.

Keeping these values in one place means the demos, and later the gesture
classifier + ESP32 code, all agree on device names, camera settings, etc.
"""

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

# --- MediaPipe Hands ---
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# --- Simulated smart home devices ---
# 1 LED + 1 fan = 2 devices, matching the actual hardware (one LED, one fan
# on the relay). Mapped to keyboard keys 1-2 as a manual fallback.
LIGHT_NAMES = ["LED"]
FAN_NAME = "Fan"

# --- Window titles ---
WEBCAM_WINDOW_NAME = "Gesture Smart Home - Hand Tracking"
PANEL_WINDOW_NAME = "Gesture Smart Home - Devices"
COMBINED_WINDOW_NAME = "Gesture Smart Home - Foundation Demo"
DATASET_WINDOW_NAME = "Gesture Smart Home - Dataset Collection"
PREDICTION_WINDOW_NAME = "Gesture Smart Home - Gesture Prediction"
GESTURE_CONTROL_WINDOW_NAME = "Gesture Smart Home - Gesture Control (Simulated)"
DASHBOARD_WINDOW_NAME = "Gesture Smart Home - Dashboard"

# --- Colors (OpenCV uses BGR, not RGB) ---
COLOR_DETECTED = (0, 200, 0)      # green
COLOR_NOT_DETECTED = (0, 0, 220)  # red
COLOR_TEXT = (255, 255, 255)      # white

# --- Gesture model training (see gesture_smart_home/ml/training.py) ---
# Minimums exist so we refuse to train (and silently produce a misleading
# accuracy number) on a dataset too small or too repetitive to trust.
MIN_SESSIONS_REQUIRED = 3
MIN_SAMPLES_PER_GESTURE = 30
TEST_SPLIT_SIZE = 0.2
VAL_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# --- Real-time gesture prediction (apps/predict_gestures.py) ---
PREDICTION_STABLE_FRAMES = 8
PREDICTION_CONFIDENCE_THRESHOLD = 0.75
