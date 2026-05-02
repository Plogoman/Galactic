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

---

# Merge — Mark's Additions

The following files and changes were written by **Mark** and merged into the project.

---

## New Files

### `kalman.py`

A 2D constant-velocity Kalman filter for the tracked target's centroid.

**Problem it solves:** The raw centroid coming out of the OpenCV tracker jitters
frame-to-frame. Without smoothing, the PD controller sees high-frequency noise as
large derivatives and over-drives the servos — the turret looks twitchy even when
the target is barely moving.

**How it works:**
- State vector `[x, y, vx, vy]` — position and velocity
- Measurement `[x, y]` — only position is observed
- Each frame: **predict** (project position forward using velocity model) then
  **update** (fuse prediction with the noisy measurement)
- Returns `(smoothed_x, smoothed_y)` — the PD controller uses this instead of raw centroid

**Key tuning knobs (inside `kalman.py`):**

| Parameter | Effect |
|---|---|
| `R *= 5.0` | Measurement noise — higher = smoother but laggier |
| `Q *= 0.1` | Process noise — higher = reacts faster to acceleration |
| `dt = 1/30` | Expected frame interval — adjust for your camera FPS |

**Dependencies:** `pip install filterpy`
If `filterpy` is not installed the import fails silently and `USE_KALMAN` is
automatically treated as `False` — no crash.

**Lead-aim bonus:** `kalman.predict_next()` returns where the target will be one
step ahead. Not wired into the PD loop yet, but available for future latency
compensation.

---

### `detection_smoother.py`

An anti-flicker layer that sits between the YOLO detector and the rest of the pipeline.

**Problem it solves:** YOLO detects a drone in frame N, misses it in N+1 (occlusion,
motion blur, confidence dip), finds it again in N+2. Without smoothing, that one-frame
gap causes the tracker to drop lock and snap to whatever else is in frame, or the drawn
boxes flash in and out.

**How it works:**
- Keeps each detection "alive" for `SMOOTHER_TTL` frames after its last sighting
- Each new frame, fresh detections are matched to live entries by class + centroid proximity
  (within `SMOOTHER_MATCH_DIST` px)
- Matched entries get their TTL reset and a hit counter incremented
- Unmatched entries age by 1 TTL; entries that reach 0 are dropped
- Returns the flicker-free list, ordered most-stable first

**`stable_only()`** — returns only detections seen 2+ consecutive frames. Used by
auto-lock so the system won't immediately lock onto a single-frame noise detection.

**Config knobs (in `tracker.py`):**

| Constant | Default | Effect |
|---|---|---|
| `SMOOTHER_TTL` | 3 | How many frames a detection stays alive after a miss |
| `SMOOTHER_MATCH_DIST` | 60 px | Max centroid movement to count as the same detection |

---

## Changes to `tracker.py`

### `LockState` State Machine

Replaced the implicit `tracker is None` / `misses` / `lost_hold` integer logic with
an explicit three-state machine:

| State | Meaning | What runs |
|---|---|---|
| `IDLE` | No target, searching | Detector every frame, pick closest-to-center |
| `LOCKED` | Tracker active, PD running | Tracker every frame, detector every `DETECT_EVERY_N` |
| `LOST` | Tracker dropped, recovery window | Detector every frame, try to re-acquire trusted target |

**Transitions:**

```
IDLE ──(detection found)──► LOCKED
LOCKED ──(misses ≥ MAX)───► LOST
LOST ──(re-acquired)──────► LOCKED
LOST ──(hold expires)─────► IDLE
```

All transitions are printed to the terminal: `[IDLE→LOCKED]`, `[LOCKED→LOST]`, etc.

**Why this is better than the old approach:**
- Recovery logic is no longer tangled into the acquisition block via a countdown variable
- Each state has one clear responsibility
- Adding a new behaviour (e.g. manual click-to-lock) means adding a branch, not
  patching a chain of `if tracker is None` conditions

### Kalman integrated into PD section

In the LOCKED PD block, `tracker_bbox` centroid is fed through `kalman.update()` before
computing errors. The Kalman-smoothed `(sx, sy)` drives `compute_errors()` and the
on-screen aim crosshair. The raw tracker bounding box is still drawn unmodified.

Kalman is reset (`kalman.reset()`) on every tracker re-initialisation — new lock,
detector snap, and recovery re-acquire — so the velocity model always starts from zero
on a fresh target.

### Detection smoother integrated into detect loop

Every call to `detect(model, frame)` is now wrapped:
```python
raw_dets   = detect(model, frame)
detections = smoother.update(raw_dets)
```
On frames where the detector is skipped (throttled LOCKED frames), `smoother.update([])`
is called so the TTL counters still tick and stale ghosts don't linger indefinitely.

When the LOST window expires and the system resets to IDLE, `smoother.reset()` is also
called to clear any lingering entries from the previous track.
