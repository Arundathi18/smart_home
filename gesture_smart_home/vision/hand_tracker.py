"""Thin wrapper around MediaPipe Hands.

Keeps all the MediaPipe-specific setup (BGR/RGB conversion, drawing
utilities, model options) in one place so the rest of the app only has to
call three methods: process(), draw_landmarks(), hands_detected().
"""

import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self.results = None

    def process(self, frame_bgr):
        """Run hand detection on a BGR frame (as read by OpenCV)."""
        # MediaPipe expects RGB; OpenCV gives us BGR.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.results = self._hands.process(frame_rgb)
        return self.results

    def draw_landmarks(self, frame_bgr):
        """Draw the detected hand skeleton onto frame_bgr, in place."""
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )
        return frame_bgr

    def hands_detected(self):
        return bool(self.results and self.results.multi_hand_landmarks)

    def first_hand_landmarks(self):
        """Return the first detected hand's landmarks, or None if no hand."""
        if not self.results or not self.results.multi_hand_landmarks:
            return None
        return self.results.multi_hand_landmarks[0]

    def hand_count(self):
        if not self.results or not self.results.multi_hand_landmarks:
            return 0
        return len(self.results.multi_hand_landmarks)

    def close(self):
        self._hands.close()
