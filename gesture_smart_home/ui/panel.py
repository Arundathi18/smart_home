"""Renders the simulated device states as an OpenCV image.

Draws one row per device: a colored block (green = on, gray = off) plus
a text label, so the panel can be shown in its own window or stitched
next to the webcam feed.
"""

import cv2
import numpy as np

PANEL_WIDTH = 300
ROW_HEIGHT = 60
TOP_MARGIN = 50
BOTTOM_MARGIN = 20

BG_COLOR = (30, 30, 30)
ON_COLOR = (0, 200, 0)
OFF_COLOR = (70, 70, 70)
TEXT_COLOR = (255, 255, 255)
TITLE_COLOR = (200, 200, 200)


def panel_height(num_devices):
    return TOP_MARGIN + num_devices * ROW_HEIGHT + BOTTOM_MARGIN


def render_panel(devices, height=None, width=PANEL_WIDTH):
    """Build a BGR image showing each device's on/off state.

    If `height` is larger than the content needs (e.g. to match a webcam
    frame's height for side-by-side display), the panel is padded rather
    than stretched.
    """
    content_height = panel_height(len(devices))
    canvas_height = max(height or 0, content_height)

    panel = np.full((canvas_height, width, 3), BG_COLOR, dtype=np.uint8)

    cv2.putText(
        panel, f"Devices (keys 1-{len(devices)})", (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TITLE_COLOR, 1, cv2.LINE_AA,
    )

    row_box_height = ROW_HEIGHT - 12
    for index, device in enumerate(devices):
        y = TOP_MARGIN + index * ROW_HEIGHT
        color = ON_COLOR if device.is_on else OFF_COLOR
        cv2.rectangle(panel, (16, y), (width - 16, y + row_box_height), color, -1)

        state_text = "ON" if device.is_on else "OFF"
        label = f"{index + 1}. {device.name} - {state_text}"
        text_y = y + row_box_height // 2 + 5
        cv2.putText(
            panel, label, (26, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA,
        )

    return panel
