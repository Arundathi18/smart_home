"""In-memory simulation of the physical devices (1 LED + 1 fan).

This has no hardware or network code yet - it exists so the rest of the
app, and later the gesture classifier, has something concrete to control
before the ESP32 wiring and relay/LED circuit are ready.
"""

from dataclasses import dataclass


@dataclass
class Device:
    name: str
    is_on: bool = False

    def toggle(self):
        self.is_on = not self.is_on

    def turn_on(self):
        self.is_on = True

    def turn_off(self):
        self.is_on = False


class SmartHome:
    def __init__(self, light_names, fan_name="Fan"):
        self.lights = [Device(name) for name in light_names]
        self.fan = Device(fan_name)

    @property
    def devices(self):
        """All devices in a fixed order: lights first, then the fan."""
        return [*self.lights, self.fan]

    def toggle_by_index(self, index):
        devices = self.devices
        if 0 <= index < len(devices):
            devices[index].toggle()

    def all_off(self):
        for device in self.devices:
            device.turn_off()

    def status_lines(self):
        return [f"{d.name}: {'ON' if d.is_on else 'OFF'}" for d in self.devices]
