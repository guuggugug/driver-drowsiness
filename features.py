"""Trich xuat dac trung khuon mat + cua so thoi gian (dung chung cho collect / train / app).

Ket hop 2 nguon:
  1) Dac trung hinh hoc tu landmark (EAR, MAR, pitch, roll) - nhu truoc.
  2) Model CO SAN cua MediaPipe (da train san boi Google, khong can du lieu cua ban):
       - blendshape eyeBlinkLeft/Right  -> diem nham mat 0..1 (it bi anh huong boi
         goc quay dau va hinh dang mat tung nguoi hon EAR)
       - blendshape jawOpen             -> diem ha mieng 0..1 (phat hien ngap)
       - ma tran bien doi khuon mat     -> goc quay ngang (yaw) that, tinh bang do
"""
import math
import time
import numpy as np

WIN = 45  # so frame moi cua so (~1.5-2 giay o 25-30 FPS)
LABELS = {0: "alert", 1: "drowsy", 2: "yawn"}

# Vector frame (sau normalize): [ear_n, mar, dpitch, roll, yaw, blink, jaw]
FRAME_FEATURES = ["ear_n", "mar", "dpitch", "roll", "yaw", "blink", "jaw"]
FEATURE_NAMES = ["ear_mean", "ear_min", "ear_std", "closed_frac", "mar_mean",
                 "mar_max", "yawn_frac", "pitch_mean", "pitch_min", "roll_abs",
                 "yaw_abs_mean", "yaw_abs_max", "away_frac",
                 "blink_mean", "blink_max", "blink_closed_frac", "jaw_max", "jaw_frac"]
FI = {n: i for i, n in enumerate(FEATURE_NAMES)}   # tra chi so theo ten

# Chi so landmark MediaPipe Face Mesh
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 308, 13, 14, 81, 178, 311, 402]
CHEEK_R, CHEEK_L, NOSE = 234, 454, 1

# yaw luu duoi dang (goc_do / 100): 0.25 ~ quay 25 do
YAW_AWAY = 0.25      # |yaw| > nguong nay = dang quay mat sang ben
YAW_ONE_EYE = 0.12   # |yaw| > nguong nay = EAR chi tin con mat gan camera
BLINK_CLOSED = 0.5   # blink_n > nguong nay = mat nham (theo model MediaPipe)
JAW_OPEN = 0.4       # jaw > nguong nay = mieng ha to


def _d(a, b):
    return float(np.linalg.norm(a - b))


def _ear(p):
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2 * _d(p[0], p[3]) + 1e-9)


class FaceTracker:
    """Tra ve landmark; sau moi lan goi landmarks(), self.extras chua
    {'blink', 'jaw', 'yaw_deg'} tu model MediaPipe (None neu khong co)."""

    def __init__(self):
        import cv2
        import mediapipe as mp
        self.cv2 = cv2
        self._mode = None
        self.extras = {}

        # mediapipe 1.0+ Tasks API (co blendshape + ma tran bien doi)
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            import urllib.request, os, tempfile
            model_url = ("https://storage.googleapis.com/mediapipe-models/"
                         "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
            model_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
            if not os.path.exists(model_path):
                print("Dang tai face_landmarker.task (~50 MB)...")
                urllib.request.urlretrieve(model_url, model_path)
            base_opts = mp_python.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1)
            self._detector = mp_vision.FaceLandmarker.create_from_options(opts)
            self._mode = "tasks"
        except Exception:
            pass

        # Du phong: legacy solutions API (mediapipe < 1.0, khong co blendshape)
        if self._mode is None:
            try:
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
                self._mode = "solutions"
            except Exception as e:
                raise RuntimeError(f"Khong the khoi tao FaceTracker: {e}")
        print("FaceTracker:", "MediaPipe Tasks (co blendshape)" if self._mode == "tasks"
              else "MediaPipe legacy (KHONG co blendshape, dung EAR)")

    def landmarks(self, frame):
        h, w = frame.shape[:2]
        self.extras = {}
        if self._mode == "tasks":
            import mediapipe as mp
            rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_img)
            if not result.face_landmarks:
                return None
            if result.face_blendshapes:
                bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
                self.extras["blink_l"] = bs.get("eyeBlinkLeft")
                self.extras["blink_r"] = bs.get("eyeBlinkRight")
                self.extras["jaw"] = bs.get("jawOpen")
            if result.facial_transformation_matrixes:
                R = np.array(result.facial_transformation_matrixes[0])[:3, :3]
                self.extras["yaw_deg"] = math.degrees(math.atan2(R[0, 2], R[2, 2]))
            lm = result.face_landmarks[0]
            # Lay ca toa do z (cung ty le voi x) -> tinh EAR 3D, it bi meo khi quay dau
            return np.array([(p.x * w, p.y * h, p.z * w) for p in lm])
        else:
            res = self.mesh.process(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                return None
            return np.array([(p.x * w, p.y * h, p.z * w)
                             for p in res.multi_face_landmarks[0].landmark])


def _yaw_from_landmarks(pts):
    """Du phong khi khong co ma tran: uoc luong yaw tu vi tri mui giua 2 ma (gioi han +-0.9)."""
    xr, xl, xn = pts[CHEEK_R][0], pts[CHEEK_L][0], pts[NOSE][0]
    den = xl - xr
    if abs(den) < 1e-6:
        return 0.9
    return float(np.clip(((xn - xr) / den - 0.5) * 1.6, -0.9, 0.9))


def raw_features(pts, extras=None):
    """Tra ve [EAR, MAR, pitch, roll, yaw, blink, jaw]. blink/jaw = NaN neu khong co model."""
    ex = extras or {}
    yaw = ex["yaw_deg"] / 100.0 if ex.get("yaw_deg") is not None else _yaw_from_landmarks(pts)

    ear_l, ear_r = _ear(pts[L_EYE]), _ear(pts[R_EYE])
    w_l = abs(pts[L_EYE[0]][0] - pts[L_EYE[3]][0])
    w_r = abs(pts[R_EYE[0]][0] - pts[R_EYE[3]][0])
    near_left = w_l >= w_r            # mat co do rong 2D lon hon = mat gan camera
    if abs(yaw) > YAW_ONE_EYE:
        ear = ear_l if near_left else ear_r
    else:
        ear = (ear_l + ear_r) / 2

    bl, br = ex.get("blink_l"), ex.get("blink_r")
    if bl is not None and br is not None:
        # Quay dau: con mat phia xa bi che -> diem nham mat cua no khong tin duoc.
        # Lay mat "mo hon" (min): khi buon ngu that thi CA HAI mat deu nham nen van bat duoc.
        blink = min(bl, br) if abs(yaw) > YAW_ONE_EYE else (bl + br) / 2
    else:
        blink = float("nan")
    jaw = ex.get("jaw")
    jaw = float("nan") if jaw is None else float(jaw)

    m = pts[MOUTH]
    mar = (_d(m[2], m[3]) + _d(m[4], m[5]) + _d(m[6], m[7])) / (3 * _d(m[0], m[1]) + 1e-9)
    p2 = pts[:, :2]
    eye_mid = (p2[33] + p2[263]) / 2
    face_h = _d(eye_mid, p2[152]) + 1e-9
    pitch = (p2[1][1] - eye_mid[1]) / face_h          # mui so voi mat: tuong doi gat dau
    roll = math.degrees(math.atan2(p2[263][1] - p2[33][1], p2[263][0] - p2[33][0]))
    return np.array([ear, mar, pitch, roll, yaw, blink, jaw], dtype=float)


def calibrate_from(raws):
    arr = np.array(raws, dtype=float)
    front = arr[np.abs(arr[:, 4]) < YAW_ONE_EYE]
    if len(front) >= 10:
        arr = front
    blink = arr[:, 5][~np.isnan(arr[:, 5])]
    return {"ear": float(np.percentile(arr[:, 0], 80)),
            "pitch": float(np.median(arr[:, 2])),
            # muc "nham mat" nen khi dang mo mat binh thuong (mat hip / nhin xuong co the > 0)
            "blink": float(np.clip(np.median(blink), 0, 0.6)) if len(blink) else 0.0}


def normalize(raw, base):
    """Ca nhan hoa theo baseline cua nguoi dung.
    Tra ve [ear_n, mar, dpitch, roll, yaw, blink_n, jaw]."""
    ear_n = raw[0] / base["ear"]
    b0 = base.get("blink") or 0.0
    if np.isnan(raw[5]):
        # Khong co blendshape -> uoc luong diem nham mat tu EAR
        blink_n = float(np.clip((0.95 - ear_n) / 0.55, 0, 1))
    else:
        blink_n = float(np.clip((raw[5] - b0) / (1 - b0 + 1e-9), 0, 1))
    jaw = float(np.clip(raw[1], 0, 1)) if np.isnan(raw[6]) else raw[6]
    return np.array([ear_n, raw[1], raw[2] - base["pitch"], raw[3], raw[4], blink_n, jaw])


def looking_away(n, yaw_thresh=YAW_AWAY):
    """True neu dang quay mat sang ben (n = vector da normalize)."""
    return abs(n[4]) > yaw_thresh


def window_features(w):
    """w: mang (n,7) [ear_n, mar, dpitch, roll, yaw, blink, jaw] -> vector dac trung."""
    e, m, p, r, y, b, j = np.asarray(w, dtype=float).T
    ay = np.abs(y)
    return [e.mean(), e.min(), e.std(), float((e < 0.75).mean()),
            m.mean(), m.max(), float((m > 0.6).mean()),
            p.mean(), p.min(), float(np.abs(r).mean()),
            ay.mean(), ay.max(), float((ay > YAW_AWAY).mean()),
            b.mean(), b.max(), float((b > BLINK_CLOSED).mean()),
            j.max(), float((j > JAW_OPEN).mean())]


def run_calibration(cap, tracker, seconds=8):
    import cv2
    raws, t0 = [], time.time()
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        pts = tracker.landmarks(frame)
        if pts is not None:
            raws.append(raw_features(pts, tracker.extras))
        left = seconds - (time.time() - t0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"CALIBRATION ({left:.0f}s): Nhin thang vao camera, mat mo to",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frames OK: {len(raws)}  |  Giu dau thang, mat nhin thang",
                    (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
        cv2.imshow("Drowsiness Monitor", frame)
        cv2.waitKey(1)
    if len(raws) < 10:
        raise RuntimeError(
            f"Khong thay khuon mat khi hieu chinh ({len(raws)} frame). "
            "Kiem tra camera va anh sang, dam bao khuon mat trong khung hinh.")
    return calibrate_from(raws)
