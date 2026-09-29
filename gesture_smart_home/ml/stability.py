"""Stability gating for real-time gesture predictions.

A single frame's prediction is noisy - MediaPipe landmarks jitter a little
frame to frame, so the raw prediction can flicker between classes even when
your hand isn't moving. StablePredictor only reports a gesture as "stable"
once it has been the top prediction, with high enough confidence, for
several consecutive frames in a row.
"""


class StablePredictor:
    def __init__(self, required_consecutive_frames=8, confidence_threshold=0.75):
        self.required_consecutive_frames = required_consecutive_frames
        self.confidence_threshold = confidence_threshold
        self._candidate = None
        self._consecutive_count = 0
        self.stable_gesture = None

    def update(self, predicted_gesture, confidence):
        """Feed in one frame's raw prediction.

        Returns the current stable gesture (or None if not yet stable).
        """
        if confidence < self.confidence_threshold:
            self._candidate = None
            self._consecutive_count = 0
            self.stable_gesture = None
            return self.stable_gesture

        if predicted_gesture == self._candidate:
            self._consecutive_count += 1
        else:
            self._candidate = predicted_gesture
            self._consecutive_count = 1

        self.stable_gesture = (
            self._candidate if self._consecutive_count >= self.required_consecutive_frames else None
        )
        return self.stable_gesture

    def reset(self):
        self._candidate = None
        self._consecutive_count = 0
        self.stable_gesture = None
