import cv2
import numpy as np
import pickle
import os
import serial
import time
import av
import threading
from collections import deque

# ─────────────────────────────────────────────
#  TUNING CONSTANTS  (edit these freely)
# ─────────────────────────────────────────────
RTSP_URL        = "rtsp://192.168.100.1:8080/?action=stream"
ARDUINO_PORT    = "/dev/ttyUSB0"
CALIB_PATH      = "/home/buuf/Desktop/projs/projs/vr_cam/easy_pikachu/additional/sportcam.pckl"
WIDTH, HEIGHT   = 640, 360
MARKER_LENGTH   = 0.05          # metres

# Alignment tolerance: how close the bore-sight ray must pass
# to the marker's Y-axis before we call it "aligned"
THRESHOLD_DIST  = 0.10          # metres  ← was 0.03, increased 3× for less spinning

# How many consecutive frames must agree before a state change is accepted
# (prevents jitter / single-bad-frame flips)
HYSTERESIS_FRAMES = 3

# After sending a command, ignore the same repeated command for this many frames
CMD_COOLDOWN_FRAMES = 3

# ─────────────────────────────────────────────
#  RTSP STREAM (threaded, reconnects on error)
# ─────────────────────────────────────────────
class RTSPStream:
    def __init__(self, url):
        self.url     = url
        self.frame   = None
        self.running = True
        self._lock   = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            try:
                print(f"[INFO] Connecting to RTSP stream …")
                container = av.open(
                    self.url, mode="r",
                    options={
                        "rtsp_transport": "udp",
                        "fflags":         "nobuffer+discardcorrupt",
                        "flags":          "low_delay",
                    }
                )
                for raw in container.decode(video=0):
                    if not self.running:
                        break
                    img = raw.to_ndarray(format="bgr24")
                    if img.shape[1] != WIDTH or img.shape[0] != HEIGHT:
                        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
                    with self._lock:
                        self.frame = img
            except Exception as e:
                print(f"[ERROR] Stream error: {e}  — retrying in 2 s")
                time.sleep(2)

    def read(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────
#  ARUCO DETECTOR  (sensitivity-tuned params)
# ─────────────────────────────────────────────
def build_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()

    # ── Adaptive-threshold window: try more scales ──────────────────────
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 53   # was default 23 → much wider sweep
    params.adaptiveThreshWinSizeStep = 4    # finer steps between min/max
    params.adaptiveThreshConstant    = 7    # slightly lower → more sensitive edges

    # ── Corner refinement → sub-pixel accuracy for better PnP ──────────
    params.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize   = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy  = 0.1

    # ── Contour / candidate filtering: relax so small / rotated markers pass ─
    params.minMarkerPerimeterRate    = 0.02  # was ~0.03 → catch smaller markers
    params.maxMarkerPerimeterRate    = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate     = 0.02
    params.minDistanceToBorder       = 2    # allow markers near the edge

    # ── Bit extraction ──────────────────────────────────────────────────
    params.perspectiveRemovePixelPerCell = 8
    params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    params.maxErroneousBitsInBorderRate  = 0.35   # more forgiving of border noise
    params.minOtsuStdDev                 = 3.0    # lower → works in low-contrast

    # ── Error correction: allow 1 bit flip ─────────────────────────────
    params.errorCorrectionRate = 0.8   # up from default 0.6

    return cv2.aruco.ArucoDetector(aruco_dict, params)


# ─────────────────────────────────────────────
#  IMAGE PRE-PROCESSING  for better detection
# ─────────────────────────────────────────────
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE: boost local contrast without blowing out highlights
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # Mild sharpening kernel
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]], dtype=np.float32)
    gray = cv2.filter2D(gray, -1, kernel)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


# ─────────────────────────────────────────────
#  MARKER SIZE FILTER
# ─────────────────────────────────────────────

# Minimum pixel area a marker must have to be considered real.
# Raise this if you still see small false positives.
MIN_MARKER_AREA_PX = 400   # pixels²  (≈ 20×20 px square)

def marker_area(corner):
    """
    Shoelace formula → signed area of the 4-corner quad (always positive).
    `corner` is shape (1, 4, 2).
    """
    pts = corner[0]   # (4, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(
        (x[0]*y[1] - x[1]*y[0]) +
        (x[1]*y[2] - x[2]*y[1]) +
        (x[2]*y[3] - x[3]*y[2]) +
        (x[3]*y[0] - x[0]*y[3])
    )

def keep_largest_marker(corners, ids):
    """
    From all detected markers:
      1. Discard any whose pixel area < MIN_MARKER_AREA_PX  (noise filter)
      2. Of the survivors, keep only the single largest one  (false-positive filter)
    Returns (corners_1, ids_1, area_px) for the winner, or (None, None, 0).
    """
    if ids is None or len(ids) == 0:
        return None, None, 0.0

    best_corner = None
    best_id     = None
    best_area   = 0.0

    for i, corner in enumerate(corners):
        area = marker_area(corner)
        if area < MIN_MARKER_AREA_PX:
            continue                   # too small → almost certainly noise
        if area > best_area:
            best_area   = area
            best_corner = corner
            best_id     = ids[i]

    if best_corner is None:
        return None, None, 0.0

    # Wrap back into the list/array shape the rest of the code expects
    return [best_corner], np.array([[best_id]]), best_area


# ─────────────────────────────────────────────
#  MULTI-SCALE DETECTION  (catches blurry / small markers)
# ─────────────────────────────────────────────
def detect_multiscale(detector, gray, scales=(1.0, 1.5, 2.0)):
    """
    Run the detector at several zoom levels and merge results.
    Returns (corners, ids) in original-image coordinates,
    already filtered to the single largest marker.
    """
    best_corners, best_ids = None, None

    for scale in scales:
        if scale == 1.0:
            g = gray
        else:
            new_w = int(gray.shape[1] * scale)
            new_h = int(gray.shape[0] * scale)
            g = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        corners, ids, _ = detector.detectMarkers(g)

        if ids is not None:
            if scale != 1.0:
                corners = [c / scale for c in corners]   # back to original coords
            # Keep whichever scale found the most markers (pre-filter)
            if best_ids is None or len(ids) > len(best_ids):
                best_corners, best_ids = corners, ids

    # ── NOW apply the size filter ─────────────────────────────────────────
    best_corners, best_ids, area = keep_largest_marker(best_corners, best_ids)
    return best_corners, best_ids, area


# ─────────────────────────────────────────────
#  HYSTERESIS STATE MACHINE
# ─────────────────────────────────────────────
class HysteresisState:
    """
    Only switches state after HYSTERESIS_FRAMES consecutive frames agree.
    Stops the drone from thrashing between ALIGN / ADVANCE on a single noisy frame.
    """
    def __init__(self, initial, window=HYSTERESIS_FRAMES):
        self.state   = initial
        self.window  = window
        self._recent = deque(maxlen=window)

    def update(self, proposed):
        self._recent.append(proposed)
        if len(self._recent) == self.window and all(s == proposed for s in self._recent):
            self.state = proposed
        return self.state


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def detect_and_align_aruco():
    stream   = RTSPStream(RTSP_URL)
    detector = build_detector()

    # ── Arduino ──────────────────────────────────────────────────────────
    ser = None
    try:
        ser = serial.Serial(ARDUINO_PORT, 9600, timeout=0.1)
        time.sleep(2)
        print(f"[INFO] Arduino connected at {ARDUINO_PORT}")
    except Exception as e:
        print(f"[WARN] Arduino not connected: {e}")

    last_cmd       = ('S', 'S')
    cmd_cooldown   = 0

    def send_command(x_cmd, y_cmd):
        nonlocal last_cmd, cmd_cooldown
        cmd = (x_cmd, y_cmd)
        if cmd == last_cmd and cmd_cooldown > 0:
            cmd_cooldown -= 1
            return
        if ser:
            ser.write(f"{x_cmd}{y_cmd}".encode())
        last_cmd     = cmd
        cmd_cooldown = CMD_COOLDOWN_FRAMES

    # ── Calibration ───────────────────────────────────────────────────────
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH, 'rb') as f:
            data = pickle.load(f)
        camera_matrix = data['camera_matrix'] if isinstance(data, dict) else data[0]
        dist_coeffs   = data['dist_coeffs']   if isinstance(data, dict) else data[1]
        print(f"[INFO] Calibration loaded from {CALIB_PATH}")
    else:
        print("[WARN] No calibration file – using placeholder values.")
        camera_matrix = np.array([[800,0,320],[0,800,180],[0,0,1]], dtype=np.float32)
        dist_coeffs   = np.zeros((4,1))

    # ── Marker geometry ───────────────────────────────────────────────────
    h = MARKER_LENGTH / 2
    obj_points = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], dtype=np.float32)

    # ── State machines ────────────────────────────────────────────────────
    align_state   = HysteresisState("SEARCHING")
    rotate_state  = HysteresisState('S')        # 'L', 'R', or 'S'

    # Exponential-smoothing for tvec (reduces pose jitter)
    tvec_smooth   = None
    ALPHA         = 0.4   # 0 = heavy smoothing, 1 = no smoothing

    print("[INFO] Starting ArUco drone aligner …  (press Q to quit)")

    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        gray                    = preprocess(frame)
        corners, ids, area_px   = detect_multiscale(detector, gray)

        active_x_cmd   = 'S'
        active_y_cmd   = 'S'

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # ── Show detected marker area (helps you tune MIN_MARKER_AREA_PX) ──
            cv2.putText(frame, f"Marker area: {int(area_px)} px²", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 2)

            _, rvec, tvec = cv2.solvePnP(
                obj_points, corners[0],
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE   # more accurate for flat square markers
            )

            # ── Smooth tvec ──────────────────────────────────────────────
            t = tvec.flatten()
            if tvec_smooth is None:
                tvec_smooth = t.copy()
            else:
                tvec_smooth = ALPHA * t + (1 - ALPHA) * tvec_smooth

            R, _         = cv2.Rodrigues(rvec)
            forward_vec  = R[:, 2]
            P0           = tvec_smooth
            V            = forward_vec

            # ── Closest-approach test ─────────────────────────────────────
            denom          = V[0]**2 + V[2]**2
            proposed_align = "ALIGNING"
            proposed_rot   = 'S'

            if denom > 1e-6:
                t_min = -(P0[0]*V[0] + P0[2]*V[2]) / denom
                if t_min > 0:
                    cx   = P0[0] + t_min * V[0]
                    cz   = P0[2] + t_min * V[2]
                    dist = np.sqrt(cx**2 + cz**2)

                    if dist < THRESHOLD_DIST:
                        proposed_align = "ADVANCING"
                    else:
                        proposed_rot = 'R' if cx > 0 else 'L'

            # Apply hysteresis before acting
            state   = align_state.update(proposed_align)
            rot_cmd = rotate_state.update(proposed_rot)

            if state == "ADVANCING":
                active_x_cmd = 'S'
                active_y_cmd = 'D'
                label_col    = (0, 255, 0)
                label_txt    = "ADVANCING"
            else:
                active_x_cmd = rot_cmd
                active_y_cmd = 'S'
                label_col    = (0, 80, 255)
                label_txt    = f"ALIGNING → {rot_cmd}"

            cv2.putText(frame, label_txt, (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_col, 2)

            # ── Draw bore-sight ray ───────────────────────────────────────
            try:
                ray_end = (P0 + V * 0.25).reshape(1,3).astype(np.float32)
                p1_img, _ = cv2.projectPoints(P0.reshape(1,3).astype(np.float32),
                                              np.zeros((3,1)), np.zeros((3,1)),
                                              camera_matrix, dist_coeffs)
                p2_img, _ = cv2.projectPoints(ray_end,
                                              np.zeros((3,1)), np.zeros((3,1)),
                                              camera_matrix, dist_coeffs)
                p1 = tuple(map(int, np.round(p1_img[0,0])))
                p2 = tuple(map(int, np.round(p2_img[0,0])))
                cv2.line(frame, p1, p2, label_col, 2)
            except Exception:
                pass

        else:
            # No marker visible → search by rotating
            align_state.update("SEARCHING")
            rotate_state.update('S')
            tvec_smooth  = None     # reset smoothing when marker lost
            active_x_cmd = 'R'
            active_y_cmd = 'S'
            cv2.putText(frame, "SEARCHING (CW)", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2)

        send_command(active_x_cmd, active_y_cmd)

        cv2.imshow("Drone ArUco Aligner", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            send_command('S', 'S')
            stream.stop()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_and_align_aruco()