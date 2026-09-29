# Gesture-Controlled Smart Home

College CBP project: control smart home devices with hand gestures, using a
MacBook webcam, OpenCV, MediaPipe, a trained gesture classifier, and an
ESP32 + relay/LEDs.

This project is being built in phases over 20 days.

## What's here right now

**Phase 1 - software foundation:**
- A modular Python package (`gesture_smart_home/`) with clean seams for the
  parts that don't exist yet.
- A webcam app that detects your hand with MediaPipe and draws its
  landmarks in real time, with a clear "Hand Detected" / "No Hand Detected"
  status.
- A simulated smart home model for 1 LED + 1 fan (matching the actual
  hardware), with a small on-screen panel showing each device's on/off state.
- A combined demo that shows both side by side (devices toggled by keyboard
  for now, as a stand-in for gestures).

**Phase 2 - gesture dataset + ML model (new):**
- A data collection app that records normalized hand-landmark samples for
  four gestures: `OPEN_PALM`, `FIST`, `THUMBS_UP`, `TWO_FINGERS`.
- A training script that fits a scikit-learn Random Forest classifier on
  that data, with a train/validation/test split that never lets frames
  from the same recording session leak across the split.
- Saved accuracy/precision/recall/F1 metrics and a confusion matrix, ready
  to paste into your CBP report.
- A real-time prediction app that loads the trained model and shows the
  live predicted gesture with a confidence score. It is **not** wired up to
  the simulated devices or to any hardware yet - that's a later phase.

**Phase 3 - gesture-controlled simulated smart home:**
- Stable gesture predictions now directly control the simulated devices:
  OPEN_PALM = LED on, FIST = LED off, THUMBS_UP = fan on, TWO_FINGERS = fan
  off. Still simulated only - no real hardware.

**Phase 4 - ESP32 firmware + hardware bridge (new, software side only):**
- ESP32 firmware (`esp32_firmware/`) that drives one LED + one relay
  channel (the fan) over a simple named-command text protocol -
  **compiled successfully with arduino-cli against a real ESP32 target**,
  though not yet flashed/tested on physical hardware since the board
  hasn't been bought yet.
- A Python serial bridge (`gesture_smart_home/esp32_bridge/`) that speaks
  the same protocol - its command encoding, response parsing, and error
  handling are unit-tested against a fake serial port (no hardware needed
  to verify the logic).
- A command-line tool (`apps/test_esp32_connection.py`) for bringing up the
  hardware once you have it: list ports, ping, set a device, read back state.

**Phase 5 - unified live dashboard (new):**
- One window for demos: webcam + hand landmarks, hand-detection status, raw
  gesture + confidence, the stable gesture, both device states, and the
  ESP32 connection status, all at once.
- Optionally connects to the real ESP32 (`--port`) and mirrors every gesture
  action to the hardware - but if the board isn't plugged in (or doesn't
  exist yet), it automatically falls back to simulation-only without
  crashing or blocking the demo.

## Project structure

```
GestureSmartHome/
├── gesture_smart_home/         # the actual package
│   ├── config.py               # shared constants (camera, devices, colors, ML settings)
│   ├── vision/
│   │   └── hand_tracker.py     # MediaPipe Hands wrapper
│   ├── smart_home/
│   │   ├── devices.py          # Device / SmartHome simulation model
│   │   └── gesture_control.py  # maps a stable gesture -> one fixed action
│   ├── ui/
│   │   ├── panel.py            # draws the device status panel
│   │   └── dataset_panel.py    # draws the dataset-collection status panel
│   ├── gestures/
│   │   ├── constants.py        # gesture names, keyboard keys, feature layout
│   │   ├── features.py         # landmarks -> position/scale-invariant feature vector
│   │   └── dataset.py          # reads/writes the local CSV dataset + sessions
│   ├── ml/
│   │   ├── training.py         # session-aware split, trains + evaluates the model
│   │   ├── model_io.py         # saves/loads the model, metadata, evaluation, plots
│   │   └── stability.py        # requires stable consecutive-frame predictions
│   └── esp32_bridge/
│       ├── protocol.py         # wire protocol shared with the ESP32 firmware
│       └── serial_bridge.py    # USB serial connection to the ESP32
├── esp32_firmware/
│   └── gesture_smart_home_firmware/
│       └── gesture_smart_home_firmware.ino  # Arduino sketch for the ESP32
├── apps/                        # runnable entry points
│   ├── webcam_hand_demo.py     # webcam + hand landmarks + status
│   ├── smart_home_demo.py      # simulated devices only, keyboard controlled
│   ├── combined_demo.py        # both together
│   ├── collect_dataset.py      # record labeled gesture samples
│   ├── manage_dataset.py       # list/delete recording sessions (no camera)
│   ├── train_model.py          # train + evaluate the classifier
│   ├── predict_gestures.py     # live webcam gesture prediction (no hardware)
│   ├── gesture_control_demo.py # gestures control the simulated devices (no hardware)
│   ├── test_esp32_connection.py # CLI for ESP32 hardware bring-up/testing
│   └── dashboard.py             # everything in one window (+ optional real hardware)
├── data/                        # your recorded dataset (gitignored, created on first run)
├── models/                      # your trained model + evaluation (gitignored, created by training)
├── requirements.txt
└── README.md
```

## Setup (macOS)

Your Mac's default `python3` may be too new for MediaPipe (as of writing,
MediaPipe does not yet support Python 3.13+). This project uses
**Python 3.11**, installed via Homebrew, in a dedicated virtual
environment - it will not affect your system Python.

Run these from the project root (`GestureSmartHome/`):

```bash
# 1. Install Python 3.11 if you don't already have it
brew install python@3.11

# 2. Create a virtual environment using Python 3.11
/opt/homebrew/bin/python3.11 -m venv .venv

# 3. Activate it (do this every time you open a new terminal for this project)
source .venv/bin/activate

# 4. Upgrade pip, then install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

You'll know the virtual environment is active because your terminal prompt
will start with `(.venv)`. To leave it later, run `deactivate`.

### Camera permission

The first time you run a demo, macOS will ask for permission for your
terminal app (Terminal/iTerm/VS Code) to access the camera. Click **Allow**.
If you missed the prompt or accidentally denied it, go to
**System Settings > Privacy & Security > Camera** and enable it manually,
then restart your terminal.

## Running the demos

Always run these as modules, from the project root, with the virtual
environment activated:

```bash
# Just the webcam + hand tracking
python -m apps.webcam_hand_demo

# Just the simulated devices (no camera needed)
python -m apps.smart_home_demo

# Both together (keyboard-controlled placeholder for gestures)
python -m apps.combined_demo
```

Controls:
- **1-2**: toggle a device (LED, then the fan) in `smart_home_demo`
  and `combined_demo`.
- **q**: quit any of the demos.

## Phase 2: collecting gesture data and training a model

This is a three-step pipeline: collect data, train, then try live
prediction. Run all commands from the project root with the virtual
environment activated.

### Step 1 - collect a dataset

```bash
python -m apps.collect_dataset
```

Controls:
- **o / f / t / 2** - select the gesture you're about to record:
  `o` = OPEN_PALM, `f` = FIST, `t` = THUMBS_UP, `2` = TWO_FINGERS
- **SPACE** - start/stop recording samples for the selected gesture
- **d** - delete all samples for the selected gesture (asks `y`/`n` to confirm)
- **r** - reset the entire dataset (asks `y`/`n` to confirm)
- **q** - quit

The on-screen panel shows how many samples you've collected per gesture so
far. Each time you launch `collect_dataset`, that's counted as one
**recording session** - the training step later needs data from **at least
3 separate sessions**, so run this app at least 3 different times (it's
fine to do them back to back, but try to vary your hand's distance/angle
from the camera and the lighting a bit between runs - this makes the model
more robust and, more importantly, gives the held-out test set real
variation to prove the model generalizes on). Aim for at least 30-50
samples per gesture per session.

Your data is saved to `data/landmarks.csv` (the labeled feature vectors)
and `data/sessions.json` (when each session happened).

> **Keyboard focus gotcha:** the gesture-select keys (`o`/`f`/`t`/`2`) only
> work if the video window is the focused/active window - click its title
> bar first. If you forget, you'll silently keep recording under whatever
> gesture was last selected. Check the highlighted row in the side panel
> to confirm your key press actually changed the selected gesture.

If you end up with a bad session (e.g. the focus gotcha above bit you and
a whole session is mislabeled), inspect and prune it without re-recording
everything:

```bash
# List every session with its per-gesture sample counts
python -m apps.manage_dataset list

# Delete one specific session by id (asks you to confirm)
python -m apps.manage_dataset delete <session_id>
```

### Step 2 - train the model

```bash
python -m apps.train_model
```

This prints accuracy, precision, recall, and F1 for both a validation set
and a genuinely held-out test set (no session appears in both train and
test), and saves:

- `models/gesture_classifier.joblib` - the trained model
- `models/metadata.json` - gesture list, feature layout, split sizes
- `models/evaluation.json` - full metrics + confusion matrix (machine-readable)
- `models/evaluation_summary.txt` - the same metrics, readable/paste-able into your report
- `models/confusion_matrix.png` - a confusion matrix plot, for your report

If you haven't collected enough data yet (fewer than 3 sessions, or fewer
than 30 samples for some gesture), it will tell you exactly what's missing
instead of training on data too thin to trust.

### Step 3 - try live prediction

```bash
python -m apps.predict_gestures
```

Shows the raw per-frame prediction and confidence, plus a "Stable gesture"
that only appears once the same prediction has held for several consecutive
frames above a confidence threshold (see `PREDICTION_STABLE_FRAMES` and
`PREDICTION_CONFIDENCE_THRESHOLD` in `gesture_smart_home/config.py` if you
want to tune this). This app only displays predictions - it does not
control the simulated devices or any hardware yet.

## Phase 3: controlling the simulated smart home with gestures

```bash
python -m apps.gesture_control_demo
```

Replaces `combined_demo`'s keyboard-only control with the trained gesture
classifier. Each gesture is one fixed action:

| Gesture | Action |
|---|---|
| OPEN_PALM | turn the **LED ON** |
| FIST | turn the **LED OFF** |
| THUMBS_UP | turn the **fan ON** |
| TWO_FINGERS | turn the **fan OFF** |

A gesture fires once per hold (the instant it becomes "stable"), not
repeatedly every frame - so holding OPEN_PALM won't spam-toggle. Number keys
**1-2** still directly toggle a device, as a manual fallback. Press `q` to
quit.

Still no real hardware - this only controls the simulated devices, same as
Phase 1's `combined_demo`.

## Phase 4: ESP32 firmware + hardware bridge

The board hasn't been bought yet, so this phase is **software-only for
now**: firmware written and compile-verified, a Python bridge with its
protocol logic unit-tested, and a CLI ready for the moment the hardware
arrives. Nothing here is wired into the gesture pipeline yet.

### Wiring (once you have the ESP32)

Two physical devices - one LED, one fan (through one relay channel):

| Device | ESP32 pin | Notes |
|---|---|---|
| LED | GPIO 18 | LED anode -> resistor (~220-330 ohm) -> GPIO; cathode -> GND |
| Fan | GPIO 25 (relay `IN` pin) | Relay VCC -> 5V, GND -> ESP32 GND (common ground required) |

The relay module's second channel is unused by the current firmware -
wire the fan to whichever channel is convenient and match its `IN` pin to
`RELAY_FAN_PIN` in the sketch. Most cheap relay modules are **active-low**
(relay switches ON when the input pin reads LOW) - the firmware already
accounts for this (`RELAY_ACTIVE_LOW = true`). If your fan behaves
backwards, flip that one line. Avoid GPIOs 0, 2, 12, 15 (strapping pins)
and 34-39 (input-only) if you need to change the pin assignment.

### Wire protocol

Plain text, newline-terminated commands (see
`gesture_smart_home/esp32_bridge/protocol.py`):

| Gesture | Python sends | ESP32 action |
|---|---|---|
| Open palm | `LED_ON` | Turn the LED on |
| Fist | `LED_OFF` | Turn the LED off |
| Thumbs up | `FAN_ON` | Activate the fan relay |
| Two fingers | `FAN_OFF` | Deactivate the fan relay |

Plus `PING` (replies `READY`) for a health check and `STATE` (replies
`STATE:<led>,<fan>`, each `0`/`1`) to read back the current state of both
devices.

### Flashing the firmware

1. Install the Arduino IDE, then add ESP32 board support: **Preferences >
   Additional Board Manager URLs** ->
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`,
   then **Tools > Board > Boards Manager**, search "esp32", install it.
2. Open `esp32_firmware/gesture_smart_home_firmware/gesture_smart_home_firmware.ino`.
3. **Tools > Board** -> select your ESP32 board variant. **Tools > Port**
   -> select the ESP32's USB port.
4. **Sketch > Upload.**
5. **Tools > Serial Monitor** at 115200 baud - you should see `READY`
   printed once, and you can type commands like `LED_ON` directly to test
   without any Python involved.

(This firmware has already been compiled successfully against the generic
`esp32:esp32:esp32` target using `arduino-cli` - it's the actual
upload/hardware step that's still pending the board arriving.)

### Testing the connection from Python

```bash
python -m apps.test_esp32_connection list                 # find the ESP32's port
python -m apps.test_esp32_connection ping <port>           # check it responds
python -m apps.test_esp32_connection set <port> led on     # turn the LED on
python -m apps.test_esp32_connection set <port> fan off    # turn the fan off
python -m apps.test_esp32_connection state <port>          # read back both states
```

Once this confirms the board responds correctly, use it with the dashboard
below via `--port` to drive the real hardware.

## Phase 5: the unified dashboard

```bash
python -m apps.dashboard                                    # simulation only
python -m apps.dashboard --port /dev/cu.usbserial-XXXX       # + real hardware
```

One window with everything: webcam + hand landmarks, hand-detection status,
the raw per-frame gesture prediction with confidence, the stable gesture
once it locks in, both device states, and the ESP32 connection status
along the bottom of the camera feed. Same gesture mapping and keys 1-2
fallback as `gesture_control_demo`.

If you don't pass `--port`, or the ESP32 can't be reached, it prints why
and keeps running in simulation-only mode - it never crashes or blocks a
demo just because the hardware isn't plugged in. Once connected, every
gesture action (and every manual 1-2 keypress) is mirrored to the real
devices; if the connection drops mid-demo, it falls back to
simulation-only automatically rather than stalling the whole dashboard.

## Troubleshooting

- **`pip install mediapipe` fails / no matching distribution** - you're
  probably not inside the Python 3.11 virtual environment. Run
  `python --version` while `.venv` is activated; it should print `3.11.x`.
- **`numpy.dtype size changed` or similar import error** - a NumPy 2.x
  install slipped in. Run `pip install "numpy<2"` inside the venv.
- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`** -
  you have a newer mediapipe than this project expects. Run
  `pip install "mediapipe==0.10.14"` inside the venv (newer releases removed
  the legacy `solutions` API this code uses).
- **Camera window is black or won't open** - check the camera permission
  steps above, and make sure no other app (Zoom, Photo Booth, etc.) is
  already using the webcam.
- **`ModuleNotFoundError: No module named 'gesture_smart_home'`** - you're
  running a file directly (e.g. `python apps/webcam_hand_demo.py`) instead
  of as a module. Use the `python -m apps.<name>` form shown above from the
  project root.
- **`train_model` says "Not enough data to train reliably"** - it's telling
  you exactly what's missing (more sessions, or more samples for a specific
  gesture). Run `collect_dataset` again to add what's needed; it won't
  overwrite what you already have.
- **`FileNotFoundError: No trained model found`** when running
  `predict_gestures` - run `python -m apps.train_model` first; it hasn't
  been trained yet (or `models/` was deleted).
- **Accuracy looks unrealistically high (e.g. 100%) after training on very
  little data** - this usually means most/all of your sessions are nearly
  identical (recorded back-to-back, same hand position, same lighting).
  Collect a couple of sessions with real variation (different distance
  from the camera, different lighting, slightly different hand angle) so
  the held-out test set actually tests generalization. If it's still 100%
  after that, it may be genuinely separable data - check the confusion
  matrix and the per-class support numbers in `evaluation_summary.txt`
  before trusting it (see the next point).
- **`train_model` says only N session(s) contain samples for all 4
  gestures** - validation and test sets are only ever built from sessions
  that recorded every gesture, so a "targeted" session recording just 1-2
  gestures (e.g. to fix a specific confusion) can't accidentally become an
  incomplete, misleading test set. It still contributes to training. Record
  one more session covering all 4 gestures, or check what you have with
  `python -m apps.manage_dataset list`.
- **A whole session got mislabeled** (e.g. the keyboard-focus gotcha in
  `collect_dataset` bit you) - run `python -m apps.manage_dataset list` to
  see each session's per-gesture counts, then
  `python -m apps.manage_dataset delete <session_id>` to remove just that
  one session without losing everything else.
- **`Connection error: Could not open serial port`** from
  `test_esp32_connection` - run the `list` subcommand to see what's
  actually available, make sure the ESP32 is plugged in over USB (not just
  power), and double-check no other program (Arduino Serial Monitor, etc.)
  already has that port open - only one program can use a serial port at a time.
- **`No response from the ESP32 within the timeout`** - the firmware isn't
  running or isn't at 115200 baud. Re-flash it, and check with the Arduino
  Serial Monitor first (see Phase 4) before trying the Python side.

## Roadmap (remaining days)

1. **Done:** software foundation - webcam + MediaPipe + simulated devices.
2. **Done:** gesture dataset collection + Random Forest classifier training
   and evaluation + real-time prediction display.
3. **Done:** gesture-controlled simulated smart home (`gesture_control_demo`)
   - each stable gesture triggers one fixed device action.
4. **Done (software side):** ESP32 firmware (compile-verified) + Python
   serial bridge (unit-tested) + `test_esp32_connection` CLI.
5. **Done:** unified live dashboard (`apps/dashboard.py`) - everything in one
   window, with optional real-hardware control that gracefully falls back
   to simulation-only when no board is connected.
6. **Next:** buy the ESP32, wire up the LED + relay per the Phase 4 wiring
   table, flash the firmware, and confirm it with `test_esp32_connection`.
7. Run `apps.dashboard --port <port>` against the real board, test
   reliability end to end, polish the UI, and prepare the demo/report.
