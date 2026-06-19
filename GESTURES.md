# Gesture & Message Contract

Reference for implementing the **robot-side consumer** (the raspi). This repo is
the *publisher*: webcam → MediaPipe → hand gestures → a control message on MQTT.
The robot subscribes and acts. The two sides never talk directly — they meet on
the MQTT broker.

```
laptop (this repo)  ──publish robot/cmd──►  broker (mosquitto)  ──►  raspi (you write this)
```

---

## 1. The control message

Published as JSON. Topic: **`robot/cmd`**.

```json
{
  "estop": false,
  "mode": "offroad",
  "lane": "center",
  "linear": 0.10,
  "angular": -0.50,
  "enable": true,
  "ts_ms": 1700000000000
}
```

> **JSON key order is meaningless** — read by key name, never by position. The
> order above is just for human readability.

| Field     | Type    | Range / values                  | Meaning |
|-----------|---------|---------------------------------|---------|
| `estop`   | bool    | `true` / `false`                | Emergency stop latched. **When `true`, `linear` and `angular` are already forced to 0** by the publisher. Stays `true` until the operator re-arms. |
| `mode`    | string  | `"onroad"` / `"offroad"`        | `onroad` = automatic-with-assistance (ADAS). `offroad` = manual. Default `offroad`. |
| `lane`    | string  | `"left"` / `"center"` / `"right"` | Target lane. Only changes while `mode == "onroad"`. Default `center`. |
| `linear`  | float   | m/s, approx `-0.22 .. 0.22`     | Forward velocity. **Positive = forward.** |
| `angular` | float   | rad/s, approx `-2.84 .. 2.84`   | Turn rate. **Positive = turn left (counter-clockwise).** |
| `enable`  | bool    | `true` / `false`                | Drive armed. When `false`, treat as stop regardless of `linear`/`angular`. |
| `ts_ms`   | int     | monotonic milliseconds          | Publisher timestamp. Use it for the deadman (see §4), **not** for wall-clock. |

> **⚠ Motion-field representation is not final.** This documents *robot-side
> mixing*: the wire carries `linear`/`angular` (twist) and **the robot computes
> wheel PWM** from its own wheelbase/PWM config. If instead client-side mixing is
> chosen, these two fields are replaced by `left` / `right` integer PWM
> (`-255..255`) and the robot just writes them to the pins. Confirm before coding
> the motor stage.

Optional fields (`mode`, `lane`, `estop`) are omitted from the JSON when unset,
but in normal operation the publisher always populates them.

---

## 2. Gestures (how the fields get set)

| Command        | How it's triggered                          | Effect | Notes |
|----------------|---------------------------------------------|--------|-------|
| **Steer / speed** | Both hands open, handlebar pose          | Sets `linear` + `angular` continuously | The core driving metaphor |
| **Arm / disarm**  | Both open palms, held ~1.5 s             | Toggles `enable`; also **clears `estop`** | Re-arming is the way to release an e-stop |
| **E-STOP**        | **Keyboard `SPACE`** (not a gesture)     | Toggles `estop`; forces motion to 0 | Keyboard on purpose — a vision e-stop fails when vision fails |
| **Mode toggle**   | Both thumbs up, held ~0.3 s              | `onroad` ⇄ `offroad` | Single toggle, not two gestures |
| **Lane left**     | Left hand points left, held ~0.3 s       | Steps `lane` one toward `left` | **Only in `onroad`**, clamped at ends |
| **Lane right**    | Right hand points right, held ~0.3 s     | Steps `lane` one toward `right` | **Only in `onroad`**, clamped at ends |

Discrete gestures are **debounced + one-shot**: hold the pose briefly, it fires
*once*, and won't fire again until you drop the pose and remake it.

---

## 3. State rules the robot can rely on

- `mode` defaults to `offroad`, `lane` to `center`, `estop` to `false` on startup.
- `lane` only steps while `onroad`; it does not wrap (left-of-left stays left).
- `estop` latches. The only ways it clears: keyboard `SPACE` again, or a re-arm
  (both-palms gesture / keyboard `e` enabling drive).
- While `estop` is `true`, the publisher guarantees `linear == 0` and
  `angular == 0`. Even so — see §4 — **do not trust this as your only stop.**

---

## 4. Safety — read this before the wheels turn

This message stream is **not** a safety system. The robot owns its own safety:

1. **Deadman timer (mandatory).** If no message arrives for ~300–500 ms, stop the
   motors. The network can drop; the publisher can crash. A robot that holds its
   last throttle when messages stop is how you put a hole in a wall.
   Use message arrival time, not `ts_ms`, for the deadman trigger.
2. **Honor `estop` and `enable`** independently of the motion fields: either one
   "off" means zero output.
3. The publisher's real emergency stop is **"hands out of frame"** — that zeroes
   the stream, which your deadman then enforces. The `estop` flag is a
   convenience layer on top, not the backstop.

---

## 5. Minimal consumer skeleton (Python / paho-mqtt)

```python
import json, time, threading
import paho.mqtt.client as mqtt

DEADMAN_S = 0.5
_last_msg = 0.0
_lock = threading.Lock()

def apply_command(linear, angular, mode, lane, estop, enable):
    if estop or not enable:
        linear = angular = 0.0
    # TODO: mix (linear, angular) -> left/right PWM using YOUR wheelbase, write pins.
    # TODO: use `mode` / `lane` for ADAS behavior when you build it.

def on_message(client, userdata, msg):
    global _last_msg
    try:
        d = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        return  # ignore garbage, deadman will stop us if it persists
    with _lock:
        _last_msg = time.monotonic()
    apply_command(
        float(d.get("linear", 0.0)),
        float(d.get("angular", 0.0)),
        d.get("mode", "offroad"),
        d.get("lane", "center"),
        bool(d.get("estop", False)),
        bool(d.get("enable", False)),
    )

def deadman_loop():
    while True:
        time.sleep(0.05)
        with _lock:
            stale = (time.monotonic() - _last_msg) > DEADMAN_S
        if stale:
            apply_command(0.0, 0.0, "offroad", "center", True, False)  # fail safe

c = mqtt.Client()
c.on_message = on_message
c.connect("localhost", 1883)   # broker on the robot, or the LAN IP of wherever it runs
c.subscribe("robot/cmd")
threading.Thread(target=deadman_loop, daemon=True).start()
c.loop_forever()
```

The `apply_command` stub is the only part that's genuinely yours: turn intent
into motor output, and decide what `mode`/`lane` mean for your robot.
