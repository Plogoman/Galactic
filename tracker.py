import time
from enum import Enum, auto
import cv2
from ultralytics import YOLO

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from detection_smoother import DetectionSmoother

try:
    from kalman import CentroidKalman
    _KALMAN_AVAILABLE = True
except ImportError:
    _KALMAN_AVAILABLE = False
    print("[kalman] filterpy not installed — Kalman disabled. pip install filterpy")

# ===========================================================================
# CONFIG — all tunable constants here, not buried in logic
# ===========================================================================

VIDEO_SOURCE = r"C:\Users\ahmed\Downloads\antidrone\Anti-UAV-RGBT\test\20190925_111757_1_9\visible.mp4"
LOOP_VIDEO = False

MODEL_PATH = r"C:\Users\ahmed\Desktop\me\me\Coding\HARDCORE\Galatic_Defender\galactic_int8_openvino_model"
CONF = 0.45
IMGSZ = 640

FRAME_W = 640
FRAME_H = 480

TRACKER_TYPE = "CSRT"          # CSRT, KCF, MOSSE
TRACKER_MAX_MISSES = 8
DETECT_EVERY_N = 2
LOST_HOLD_FRAMES = 30

CENTER_BOX_W = 80
CENTER_BOX_H = 60

# --- PD tuning: X axis (pan) ---
KP_X = 0.50
KD_X = 0.10

# --- PD tuning: Y axis (tilt) ---
KP_Y = 0.50
KD_Y = 0.10

# --- Servo output ---
# PD output is in [-1, 1]; 90 = stop, <90 = one direction, >90 = other.
SERVO_SCALE = 30.0
MAX_CMD_X = 1.0
MAX_CMD_Y = 1.0

# --- UART ---
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

# --- Re-acquisition ---
REACQUIRE_MAX_DIST = 150       # px
REACQUIRE_MIN_IOU = 0.02
STRONG_MATCH_IOU = 0.10
STRONG_MATCH_DIST = 90         # px
PRINT_EVERY_N = 5

TARGET_CLASS = None            # filter by class name; None = accept all
USE_CARTESIAN_Y = False
SHOW_ALL_DETECTIONS = True

# --- Detection Smoother ---
SMOOTHER_TTL = 3               # frames a detection stays alive after last sighting
SMOOTHER_MATCH_DIST = 60       # px — max centroid movement to still be same detection

# --- Kalman filter ---
USE_KALMAN = True              # smooth PD input; set False if filterpy not available


# ===========================================================================
# STATE MACHINE
# ===========================================================================

class LockState(Enum):
    IDLE   = auto()   # searching, no active tracker
    LOCKED = auto()   # tracker running, PD active
    LOST   = auto()   # tracker dropped, recovery window active


# ===========================================================================
# HELPERS
# ===========================================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def dist2(p1, p2):
    dx, dy = p1[0] - p2[0], p1[1] - p2[1]
    return dx * dx + dy * dy


def center_of_bbox(b):
    x, y, w, h = b
    return int(x + w / 2), int(y + h / 2)


def in_deadband(cx, cy, fx, fy, bw, bh):
    return abs(cx - fx) <= bw // 2 and abs(cy - fy) <= bh // 2


def quadrant(cx, cy, fx, fy, bw, bh):
    if in_deadband(cx, cy, fx, fy, bw, bh):
        return "CENTER"
    if cx < fx and cy < fy:
        return "TOP-LEFT"
    if cx >= fx and cy < fy:
        return "TOP-RIGHT"
    if cx < fx and cy >= fy:
        return "BOTTOM-LEFT"
    return "BOTTOM-RIGHT"


def compute_errors(cx, cy, fx, fy):
    ex = (cx - fx) / (FRAME_W / 2.0)
    ey = (fy - cy) / (FRAME_H / 2.0) if USE_CARTESIAN_Y else (cy - fy) / (FRAME_H / 2.0)
    return ex, ey


def pd_to_servo_speed(pd_output):
    """Map PD output [-1, 1] to a continuous servo speed command [0, 180]. 90 = stop."""
    return int(clamp(90 + pd_output * SERVO_SCALE, 0, 180))


# ===========================================================================
# PD CONTROLLER
# Continuous rotation servos need speed commands, not position — integral would
# cause runaway spinning with no position feedback, so PD is the right choice.
# ===========================================================================

class PD:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self._prev_error = None
        self._prev_t = None

    def reset(self):
        self._prev_error = None
        self._prev_t = None

    def update(self, error):
        now = time.time()
        dt = 0.0 if self._prev_t is None else max(now - self._prev_t, 1e-3)
        self._prev_t = now

        derivative = 0.0
        if self._prev_error is not None and dt > 0:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        return self.kp * error + self.kd * derivative


# ===========================================================================
# TRACKER FACTORY
# ===========================================================================

def make_tracker(kind="CSRT"):
    kind = kind.upper()
    legacy = getattr(cv2, "legacy", None)

    if kind == "CSRT":
        ctor = getattr(cv2, "TrackerCSRT_create", None) or getattr(legacy, "TrackerCSRT_create", None)
    elif kind == "KCF":
        ctor = getattr(cv2, "TrackerKCF_create", None) or getattr(legacy, "TrackerKCF_create", None)
    elif kind == "MOSSE":
        ctor = getattr(legacy, "TrackerMOSSE_create", None)
    else:
        raise ValueError(f"Unsupported tracker type: {kind}")

    if ctor is None:
        raise RuntimeError(
            f"Tracker '{kind}' not available. Install opencv-contrib-python."
        )
    return ctor()


# ===========================================================================
# DETECTION
# ===========================================================================

def detect(model, frame):
    result = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    out = []

    if result.boxes is None or len(result.boxes) == 0:
        return out

    xyxy  = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss  = result.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, clss):
        label = model.names[int(cid)]
        if TARGET_CLASS and label != TARGET_CLASS:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        out.append({
            "bbox": (x1, y1, w, h),
            "cx": cx,
            "cy": cy,
            "conf": float(conf),
            "label": label,
            "class_id": int(cid),
        })

    return out


def pick_center_target(dets, frame_cx, frame_cy):
    if not dets:
        return None
    return min(dets, key=lambda d: (d["cx"] - frame_cx) ** 2 + (d["cy"] - frame_cy) ** 2)


def pick_best_detection_for_reference(detections, ref_bbox, ref_class_id=None):
    if not detections or ref_bbox is None:
        return None

    rcx, rcy = center_of_bbox(ref_bbox)
    same_class = [d for d in detections if d["class_id"] == ref_class_id] if ref_class_id is not None else []
    candidates = same_class if same_class else detections

    best, best_score = None, -1e9
    for d in candidates:
        this_iou = iou_xywh(d["bbox"], ref_bbox)
        this_d2  = dist2((d["cx"], d["cy"]), (rcx, rcy))
        max_d2   = REACQUIRE_MAX_DIST * REACQUIRE_MAX_DIST

        if this_iou < REACQUIRE_MIN_IOU and this_d2 > max_d2:
            continue

        dist_score = 1.0 / (1.0 + this_d2 / float(max_d2))
        score = 2.8 * this_iou + 1.0 * dist_score + 0.25 * d["conf"]

        if score > best_score:
            best_score = score
            best = d

    return best


def init_tracker_on_detection(frame, det):
    tracker = make_tracker(TRACKER_TYPE)
    bbox = tuple(map(int, det["bbox"]))
    try:
        ok_init = tracker.init(frame, bbox)
    except cv2.error as e:
        print(f"[tracker] init failed: {e}")
        return None, None
    if ok_init is False:
        return None, None
    return tracker, bbox


# ===========================================================================
# DRAWING
# ===========================================================================

_STATE_COLORS = {
    LockState.IDLE:   (0, 165, 255),   # orange
    LockState.LOCKED: (0, 255, 0),     # green
    LockState.LOST:   (0, 80, 255),    # red-orange
}


def draw(frame, state, tracker_bbox, target_label, all_dets,
         fx, fy, cmd_x, cmd_y, q, deadband_locked,
         pan_spd, tilt_spd, trusted_bbox=None, smooth_target=None):
    h, w = frame.shape[:2]
    state_color = _STATE_COLORS[state]

    if SHOW_ALL_DETECTIONS:
        for d in all_dets:
            x, y, bw, bh = d["bbox"]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (120, 120, 120), 1)

    cv2.line(frame, (fx, 0), (fx, h), (70, 70, 70), 1)
    cv2.line(frame, (0, fy), (w, fy), (70, 70, 70), 1)

    dbx1 = fx - CENTER_BOX_W // 2
    dby1 = fy - CENTER_BOX_H // 2
    dbx2 = fx + CENTER_BOX_W // 2
    dby2 = fy + CENTER_BOX_H // 2
    cv2.rectangle(frame, (dbx1, dby1), (dbx2, dby2), (0, 255, 255), 1)
    cv2.drawMarker(frame, (fx, fy), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)

    if trusted_bbox is not None:
        x, y, bw, bh = trusted_bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 0), 1)

    if tracker_bbox is not None:
        x, y, bw, bh = tracker_bbox
        raw_cx, raw_cy = center_of_bbox(tracker_bbox)

        # Draw the raw tracker box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), state_color, 2)
        cv2.drawMarker(frame, (raw_cx, raw_cy), state_color, cv2.MARKER_CROSS, 16, 2)

        # Line and crosshair follow the Kalman-smoothed point if available
        aim_cx = smooth_target[0] if smooth_target else raw_cx
        aim_cy = smooth_target[1] if smooth_target else raw_cy
        cv2.line(frame, (fx, fy), (aim_cx, aim_cy), (255, 0, 255), 2)
        if smooth_target:
            cv2.drawMarker(frame, (aim_cx, aim_cy), (255, 0, 255), cv2.MARKER_CROSS, 12, 1)

        cv2.putText(frame, f"{target_label} | {q}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)
        cv2.putText(frame, f"cmd=({cmd_x:+.3f}, {cmd_y:+.3f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(frame, f"servo spd pan={pan_spd} tilt={tilt_spd}", (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 255), 2)
    else:
        cv2.putText(frame, f"State: {state.name}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)

    # State badge bottom-right
    cv2.putText(frame, state.name, (w - 110, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)


# ===========================================================================
# UART
# ===========================================================================

def open_serial():
    if not SERIAL_AVAILABLE:
        print("[uart] pyserial not installed — UART disabled")
        return None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[uart] opened {SERIAL_PORT} @ {BAUD_RATE} baud")
        return ser
    except serial.SerialException as e:
        print(f"[uart] could not open {SERIAL_PORT}: {e} — UART disabled")
        return None


def send_servo_speeds(ser, pan, tilt):
    if ser is None:
        return
    try:
        ser.write(f"{pan},{tilt}\n".encode())
    except serial.SerialException as e:
        print(f"[uart] write error: {e}")


# ===========================================================================
# MAIN LOOP
# ===========================================================================

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    model    = YOLO(MODEL_PATH)
    ser      = open_serial()
    smoother = DetectionSmoother(ttl=SMOOTHER_TTL, match_dist=SMOOTHER_MATCH_DIST)
    kalman   = CentroidKalman() if (USE_KALMAN and _KALMAN_AVAILABLE) else None

    if kalman:
        print("[kalman] Kalman filter active")

    # --- tracking state ---
    state            = LockState.IDLE
    tracker          = None
    tracker_bbox     = None
    tracker_label    = ""
    tracker_class_id = None
    trusted_bbox     = None
    trusted_label    = ""
    trusted_class_id = None
    misses           = 0
    lost_hold        = 0
    frame_idx        = 0

    pd_x = PD(KP_X, KD_X)
    pd_y = PD(KP_Y, KD_Y)

    while True:
        ok, frame = cap.read()
        if not ok:
            if LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame_idx += 1
        fx, fy = FRAME_W // 2, FRAME_H // 2

        # ---------------------------------------------------------------
        # DETECTION — run every frame when IDLE/LOST, throttled when LOCKED
        # ---------------------------------------------------------------
        run_detect = (
            state in (LockState.IDLE, LockState.LOST) or
            (state == LockState.LOCKED and
             (frame_idx % DETECT_EVERY_N == 0 or misses > 0))
        )
        if run_detect:
            raw_dets = detect(model, frame)
            detections = smoother.update(raw_dets)
        else:
            detections = smoother.update([])

        # ---------------------------------------------------------------
        # IDLE — search for first target
        # ---------------------------------------------------------------
        if state == LockState.IDLE:
            target = pick_center_target(detections, fx, fy)
            if target is not None:
                new_tracker, new_bbox = init_tracker_on_detection(frame, target)
                if new_tracker is not None:
                    tracker          = new_tracker
                    tracker_bbox     = new_bbox
                    tracker_label    = target["label"]
                    tracker_class_id = target["class_id"]
                    trusted_bbox     = new_bbox
                    trusted_label    = tracker_label
                    trusted_class_id = tracker_class_id
                    misses           = 0
                    state            = LockState.LOCKED
                    if kalman:
                        kalman.reset()
                    print(f"[IDLE→LOCKED] acquired: {tracker_label} bbox={tracker_bbox}")

        # ---------------------------------------------------------------
        # LOCKED — advance tracker, snap to detector when they disagree
        # ---------------------------------------------------------------
        elif state == LockState.LOCKED:
            ok_track, bbox = tracker.update(frame)

            if ok_track:
                tracker_bbox = tuple(map(int, bbox))
                misses = 0
            else:
                misses += 1

            if detections:
                ref_bbox     = trusted_bbox     if trusted_bbox     is not None else tracker_bbox
                ref_class_id = trusted_class_id if trusted_class_id is not None else tracker_class_id
                best = pick_best_detection_for_reference(detections, ref_bbox, ref_class_id)

                if best is not None:
                    best_bbox    = tuple(map(int, best["bbox"]))
                    agree_iou    = iou_xywh(best_bbox, tracker_bbox) if tracker_bbox else 0.0
                    agree_d2     = dist2(center_of_bbox(best_bbox), center_of_bbox(tracker_bbox)) if tracker_bbox else 10**9

                    trusted_bbox     = best_bbox
                    trusted_label    = best["label"]
                    trusted_class_id = best["class_id"]

                    if (not ok_track) or (agree_iou < STRONG_MATCH_IOU) or (agree_d2 > STRONG_MATCH_DIST ** 2):
                        new_tracker, new_bbox = init_tracker_on_detection(frame, best)
                        if new_tracker is not None:
                            tracker          = new_tracker
                            tracker_bbox     = new_bbox
                            tracker_label    = best["label"]
                            tracker_class_id = best["class_id"]
                            misses           = 0
                            if kalman:
                                kalman.reset()
                            print(f"[snap] tracker -> detector {tracker_label} bbox={tracker_bbox}")
                    else:
                        tracker_label    = best["label"]
                        tracker_class_id = best["class_id"]

            if misses >= TRACKER_MAX_MISSES:
                print("[LOCKED→LOST] tracker dropped, entering recovery hold")
                tracker      = None
                tracker_bbox = None
                lost_hold    = LOST_HOLD_FRAMES
                misses       = 0
                state        = LockState.LOST
                if kalman:
                    kalman.reset()
                pd_x.reset()
                pd_y.reset()

        # ---------------------------------------------------------------
        # LOST — recovery window: try to re-acquire the trusted target
        # ---------------------------------------------------------------
        elif state == LockState.LOST:
            target = None
            if trusted_bbox is not None and detections:
                target = pick_best_detection_for_reference(detections, trusted_bbox, trusted_class_id)

            if target is not None:
                new_tracker, new_bbox = init_tracker_on_detection(frame, target)
                if new_tracker is not None:
                    tracker          = new_tracker
                    tracker_bbox     = new_bbox
                    tracker_label    = target["label"]
                    tracker_class_id = target["class_id"]
                    trusted_bbox     = new_bbox
                    trusted_label    = tracker_label
                    trusted_class_id = tracker_class_id
                    misses           = 0
                    state            = LockState.LOCKED
                    if kalman:
                        kalman.reset()
                    print(f"[LOST→LOCKED] re-acquired: {tracker_label} bbox={tracker_bbox}")

            if state == LockState.LOST:  # still lost after re-acquire attempt
                lost_hold -= 1
                if lost_hold <= 0:
                    print("[LOST→IDLE] recovery window expired, full reset")
                    trusted_bbox     = None
                    trusted_label    = ""
                    trusted_class_id = None
                    smoother.reset()
                    state            = LockState.IDLE

        # ---------------------------------------------------------------
        # PD → Kalman-smoothed errors → servo speed commands → UART
        # Only active while LOCKED.
        # ---------------------------------------------------------------
        cmd_x, cmd_y    = 0.0, 0.0
        pan_spd         = 90
        tilt_spd        = 90
        q               = "NONE"
        deadband_locked = False
        smooth_target   = None

        if state == LockState.LOCKED and tracker_bbox is not None:
            raw_cx, raw_cy = center_of_bbox(tracker_bbox)

            if kalman:
                sx, sy = kalman.update(raw_cx, raw_cy)
                cx, cy = int(sx), int(sy)
                smooth_target = (cx, cy)
            else:
                cx, cy = raw_cx, raw_cy

            ex, ey = compute_errors(cx, cy, fx, fy)
            deadband_locked = in_deadband(cx, cy, fx, fy, CENTER_BOX_W, CENTER_BOX_H)
            q               = quadrant(cx, cy, fx, fy, CENTER_BOX_W, CENTER_BOX_H)

            if deadband_locked:
                pd_x.reset()
                pd_y.reset()
            else:
                cmd_x = clamp(pd_x.update(ex), -MAX_CMD_X, MAX_CMD_X)
                cmd_y = clamp(pd_y.update(ey), -MAX_CMD_Y, MAX_CMD_Y)

            pan_spd  = pd_to_servo_speed(cmd_x)
            tilt_spd = pd_to_servo_speed(cmd_y)

            send_servo_speeds(ser, pan_spd, tilt_spd)

            if frame_idx % PRINT_EVERY_N == 0:
                print(
                    f"[{state.name}] "
                    f"cmd_x={cmd_x:+.3f} cmd_y={cmd_y:+.3f} "
                    f"err_x={ex:+.3f} err_y={ey:+.3f} "
                    f"pan_spd={pan_spd} tilt_spd={tilt_spd} "
                    f"quadrant={q} deadband={deadband_locked}"
                )

        draw(
            frame, state, tracker_bbox, tracker_label, detections,
            fx, fy, cmd_x, cmd_y, q, deadband_locked,
            pan_spd, tilt_spd,
            trusted_bbox=trusted_bbox,
            smooth_target=smooth_target,
        )

        cv2.imshow("Drone Tracker", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    if ser is not None:
        ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
