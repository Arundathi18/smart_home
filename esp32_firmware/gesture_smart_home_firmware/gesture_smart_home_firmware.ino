/*
  Gesture Smart Home - ESP32 firmware

  Listens on the USB serial connection for simple text commands from the
  Python bridge (gesture_smart_home/esp32_bridge/) and drives one LED
  (the light) plus one relay channel (the fan).

  Wire protocol (see gesture_smart_home/esp32_bridge/protocol.py):
    "LED_ON\n" / "LED_OFF\n"   -> set the LED, replies "OK:LED_ON\n" / "OK:LED_OFF\n"
    "FAN_ON\n" / "FAN_OFF\n"   -> set the fan, replies "OK:FAN_ON\n" / "OK:FAN_OFF\n"
    "PING\n"                   -> replies "READY\n"
    "STATE\n"                  -> replies "STATE:<led>,<fan>\n" (each 0 or 1)
    anything else               -> replies "ERR:<reason>\n"

  Wiring:
    - LED: ESP32 GPIO -> current-limiting resistor (~220-330 ohm) -> LED
      anode; LED cathode -> GND.
    - Relay module (fan channel): ESP32 GPIO -> relay IN pin; relay VCC ->
      5V (check your module's voltage rating); relay GND -> ESP32 GND (a
      common ground between the ESP32 and the relay module is required).
    - Most cheap relay modules are ACTIVE-LOW (the relay switches ON when
      the input pin reads LOW). If the fan behaves backwards from what you
      expect, flip RELAY_ACTIVE_LOW below.
    - The 2-channel relay's second channel is unused by this firmware -
      wire the fan to whichever channel is convenient and match its IN pin
      to RELAY_FAN_PIN below.
    - Pin numbers below are a reasonable default for a typical ESP32
      DevKitC-style board. If your board is a different variant, avoid the
      strapping pins (0, 2, 12, 15) and the input-only pins (34-39).

  To upload: install ESP32 board support in the Arduino IDE (File >
  Preferences > Additional Board URLs, then Tools > Board > Boards
  Manager > search "esp32"), select your board and port under Tools, then
  Sketch > Upload. Afterwards, Tools > Serial Monitor at 115200 baud lets
  you watch it print replies - handy for debugging without the Python side
  at all (type e.g. "LED_ON" and press Enter).
*/

const int LED_PIN = 18;
const int RELAY_FAN_PIN = 25;

const bool RELAY_ACTIVE_LOW = true;  // most common for cheap relay modules

bool ledIsOn = false;
bool fanIsOn = false;

void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  pinMode(RELAY_FAN_PIN, OUTPUT);
  setRelay(RELAY_FAN_PIN, false);

  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      handleCommand(line);
    }
  }
}

void setRelay(int pin, bool isOn) {
  bool electricalHigh = RELAY_ACTIVE_LOW ? !isOn : isOn;
  digitalWrite(pin, electricalHigh ? HIGH : LOW);
}

void handleCommand(String line) {
  if (line == "PING") {
    Serial.println("READY");
    return;
  }

  if (line == "STATE") {
    String reply = "STATE:";
    reply += ledIsOn ? "1" : "0";
    reply += ",";
    reply += fanIsOn ? "1" : "0";
    Serial.println(reply);
    return;
  }

  if (line == "LED_ON") {
    ledIsOn = true;
    digitalWrite(LED_PIN, HIGH);
    Serial.println("OK:LED_ON");
    return;
  }

  if (line == "LED_OFF") {
    ledIsOn = false;
    digitalWrite(LED_PIN, LOW);
    Serial.println("OK:LED_OFF");
    return;
  }

  if (line == "FAN_ON") {
    fanIsOn = true;
    setRelay(RELAY_FAN_PIN, true);
    Serial.println("OK:FAN_ON");
    return;
  }

  if (line == "FAN_OFF") {
    fanIsOn = false;
    setRelay(RELAY_FAN_PIN, false);
    Serial.println("OK:FAN_OFF");
    return;
  }

  Serial.println("ERR:unknown command, expected LED_ON/LED_OFF/FAN_ON/FAN_OFF/PING/STATE");
}
