# tracker.py — Update Notes

## Why PD and not PID

The original `EvidenceAndCommands.ipynb` used a PD controller. We kept PD and did not
upgrade to PID, because the servos are **continuous rotation** — and integral control is
wrong for that hardware.

A PID's integral term accumulates position error over time. On a positional servo this
eliminates steady-state drift. On a continuous rotation servo there is no position feedback
— the motor just spins. The integral keeps growing during occlusions or when the drone
leaves the frame, and when the target reappears the servo lunges at full speed with nothing
to stop it. **PD is the correct controller here:**

- **P** — spin at a speed proportional to how far off-center the drone is
- **D** — slow down as the drone approaches center, dampening overshoot

---

## How the servo commands work

The 0–180 values sent over UART are **not rotation angles**. They are PWM speed+direction
commands. The MG996R continuous rotation variant interprets them as:

```
Value sent →    0           90          180
PWM pulse  →  1.0 ms      1.5 ms      2.0 ms
Servo does →  full speed    STOP     full speed
              (one way)               (other way)
```

The servo physically rotates without limit — it can spin indefinitely in either direction.
`90` = stop. The 0–180 range is purely the PWM command vocabulary; it has nothing to do
with the physical rotation of the shaft.

---

## Servo Hardware — TowerPro MG996R

The MG996R is sold in two variants depending on the batch or supplier:

| Variant | PWM signal means | Behavior |
|---|---|---|
| 180° positional (most common) | Target angle | Moves to position and holds, hard stop at each end |
| 360° continuous rotation | Speed + direction | Spins indefinitely, no position hold |

**To check which one you have:** send a 90° / 1.5 ms pulse from the STM32.
If it holds still → positional. If it starts spinning → continuous.

The code is written for the **continuous rotation** variant.

---

## Files Created

### `tracker.py`
Production-ready Python extraction of the full detection + tracking pipeline,
previously only available as `EvidenceAndCommands.ipynb`. Additions on top of the
original pipeline:

- **`PD` class** — manages its own `dt` internally via `time.time()`. Separate instances
  `pd_x` and `pd_y` for pan and tilt. `reset()` clears derivative state on target lock/loss.

- **`pd_to_servo_speed(pd_output)`** — maps PD output `[-1, 1]` to `[0, 180]` speed command.
  `90` = stop. `SERVO_SCALE` controls how aggressively the servo responds; start small (30.0)
  and tune up.

- **`open_serial()` / `send_servo_speeds()`** — opens `SERIAL_PORT` at `BAUD_RATE` with
  `try/except`; returns `None` on failure (no crash). Sends `"pan_spd,tilt_spd\n"` to the
  STM32F401CCU6 only when a drone is actively tracked. Sends `90,90` (stop) otherwise.
  Gracefully disabled if `pyserial` is not installed.

- **All tunable constants at the top of the file:**
  `KP_X/Y`, `KD_X/Y`, `SERVO_SCALE`, `SERIAL_PORT`, `BAUD_RATE`

- **Display** shows live speed commands on screen: `servo spd pan=NN tilt=MM`

---

## Files Modified

None — all pre-existing `.py` files (`YoLO_ncnn_model/model_ncnn.py`) were left untouched.

---

## Notebooks Intentionally Left Untouched

| File                        | Reason                                                            |
|-----------------------------|-------------------------------------------------------------------|
| `EvidenceAndCommands.ipynb` | Source pipeline — extracted to `tracker.py` without modification |
| `Galatic.ipynb`             | Dataset visualization + training — not production                 |
| `tuning.ipynb`              | Label file utility — not production                               |
| `exporting.ipynb`           | NCNN/OpenVINO export — not production                             |

---

## Quadrant-Based Correction

The quadrant system (`TOP-LEFT`, `TOP-RIGHT`, `BOTTOM-LEFT`, `BOTTOM-RIGHT`, `CENTER`) is
already fully driving the servo corrections — it does not need to be sent over UART separately.

Here is what happens end-to-end when the drone is, for example, in the top-left:

```
Drone is TOP-LEFT
  → ex < 0  (drone is left of frame center)
  → ey < 0  (drone is above frame center)
  → PD outputs negative cmd_x and cmd_y
  → pan_spd  < 90  → pan servo spins to chase drone left
  → tilt_spd < 90  → tilt servo spins to chase drone up
  → errors shrink as drone approaches center
  → commands approach 90 (servos slow down)
  → drone enters CENTER deadband → servos stop (90, 90)
```

The quadrant label is shown on screen and printed to the terminal for human monitoring.
The STM32 already receives everything it needs (`pan_spd, tilt_spd`) — direction is implicit
in whether the value is above or below 90. Sending the quadrant string over UART is not
done yet and can be added later if needed (e.g. for LED indicators on the STM32 side),
which would change the format to `"pan_spd,tilt_spd,quadrant\n"`.

---

## To Run

1. Adjust `VIDEO_SOURCE`, `MODEL_PATH`, `SERIAL_PORT`, and PD gains at the top of `tracker.py`.
2. Run: `python tracker.py`
