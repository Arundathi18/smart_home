"""Maps stable gesture predictions directly to actions on a SmartHome
instance. Each gesture is a fixed, single action - no "selected device" or
cursor concept to track:

    OPEN_PALM    turn the LED ON
    FIST         turn the LED OFF
    THUMBS_UP    turn the fan ON
    TWO_FINGERS  turn the fan OFF

Kept separate from any OpenCV/webcam code so it can be tested without a
camera - see gesture_smart_home/ml/stability.py for the "only fire once per
gesture hold" logic that should sit in front of apply_gesture().
"""


def _led_on(smart_home):
    smart_home.lights[0].turn_on()


def _led_off(smart_home):
    smart_home.lights[0].turn_off()


def _fan_on(smart_home):
    smart_home.fan.turn_on()


def _fan_off(smart_home):
    smart_home.fan.turn_off()


GESTURE_ACTIONS = {
    "OPEN_PALM": _led_on,
    "FIST": _led_off,
    "THUMBS_UP": _fan_on,
    "TWO_FINGERS": _fan_off,
}


def apply_gesture(smart_home, gesture):
    """Apply one gesture event to a SmartHome instance.

    Call this once per gesture *becoming* stable, not on every frame it
    stays held - otherwise holding OPEN_PALM would call turn_on() on every
    single frame.
    """
    action = GESTURE_ACTIONS.get(gesture)
    if action:
        action(smart_home)
