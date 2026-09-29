"""Renders the dataset-collection status panel.

Shows which gesture is currently selected, whether recording is on, and
how many samples have been collected for each gesture so far.
"""

import cv2
import numpy as np

PANEL_WIDTH = 340
ROW_HEIGHT = 42
TOP_MARGIN = 100
BOTTOM_MARGIN = 20

BG_COLOR = (30, 30, 30)
SELECTED_COLOR = (0, 140, 255)
UNSELECTED_COLOR = (70, 70, 70)
TEXT_COLOR = (255, 255, 255)
TITLE_COLOR = (200, 200, 200)
RECORDING_COLOR = (0, 0, 220)
IDLE_COLOR = (90, 90, 90)


def panel_height(num_gestures):
    return TOP_MARGIN + num_gestures * ROW_HEIGHT + BOTTOM_MARGIN


def render_dataset_panel(gestures, selected_gesture, counts, recording, height=None, width=PANEL_WIDTH):
    content_height = panel_height(len(gestures))
    canvas_height = max(height or 0, content_height)
    panel = np.full((canvas_height, width, 3), BG_COLOR, dtype=np.uint8)

    cv2.putText(
        panel, "Gesture Dataset Collection", (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TITLE_COLOR, 1, cv2.LINE_AA,
    )

    rec_text = "RECORDING" if recording else "Paused (SPACE to record)"
    rec_color = RECORDING_COLOR if recording else IDLE_COLOR
    cv2.circle(panel, (24, 55), 8, rec_color, -1)
    cv2.putText(
        panel, rec_text, (42, 61),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA,
    )

    total = sum(counts.values())
    cv2.putText(
        panel, f"Total samples: {total}", (16, 84),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TITLE_COLOR, 1, cv2.LINE_AA,
    )

    row_box_height = ROW_HEIGHT - 10
    for index, gesture in enumerate(gestures):
        y = TOP_MARGIN + index * ROW_HEIGHT
        is_selected = gesture == selected_gesture
        color = SELECTED_COLOR if is_selected else UNSELECTED_COLOR
        cv2.rectangle(panel, (16, y), (width - 16, y + row_box_height), color, -1)

        label = f"{gesture}: {counts.get(gesture, 0)}"
        cv2.putText(
            panel, label, (26, y + row_box_height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA,
        )

    return panel
