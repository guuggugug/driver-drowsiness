"""Trich xuat dac trung khuon mat + cua so thoi gian (dung chung cho collect / train / app)."""
import math
import time
import numpy as np

WIN = 45  # so frame moi cua so (~1.5-2 giay o 25-30 FPS)
LABELS = {0: "alert", 1: "drowsy", 2: "yawn"}
FEATURE_NAMES = ["ear_mean", "ear_min", "ear_std", "closed_frac", "mar_mean",
                 "mar_max", "yawn_frac", "pitch_mean", "pitch_min", "roll_abs"]

# Chi so landmark MediaPipe Face Mesh
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 308, 13, 14, 81, 178, 311, 402]


def _d(a, b):
    return float(np.linalg.norm(a - b))


def _ear(p):
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2 * _d(p[0], p[3]) + 1e-9)


class FaceTracker:
    def __init__(self):
        import cv2
        import mediapipe as mp
        self.cv2 = cv2
        self._mode = None

        # Try mediapipe 1.0+ Tasks API first
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
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1)
            self._detector = mp_vision.FaceLandmarker.create_from_options(opts)
            self._mode = "tasks"
        except Exception as e:
            pass

        # Fallback: legacy solutions API (mediapipe < 1.0)
        if self._mode is None:
            try:
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
                self._mode = "solutions"
            except Exception as e:
                raise RuntimeError(f"Khong the khoi tao FaceTracker: {e}")

    def landmarks(self, frame):
        h, w = frame.shape[:2]
        if self._mode == "tasks":
            import mediapipe as mp
            rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_img)
            if not result.face_landmarks:
                return None
            lm = result.face_landmarks[0]
            return np.array([(p.x * w, p.y * h) for p in lm])
        else:
            res = self.mesh.process(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                return None
            return np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark])


def raw_features(pts):
    """Tra ve [EAR, MAR, pitch, roll] tu 468 landmark."""
    ear = (_ear(pts[L_EYE]) + _ear(pts[R_EYE])) / 2
    m = pts[MOUTH]
    mar = (_d(m[2], m[3]) + _d(m[4], m[5]) + _d(m[6], m[7])) / (3 * _d(m[0], m[1]) + 1e-9)
    eye_mid = (pts[33] + pts[263]) / 2
    face_h = _d(eye_mid, pts[152]) + 1e-9
    pitch = (pts[1][1] - eye_mid[1]) / face_h          # mui so voi mat: tuong doi gat dau
    roll = math.degrees(math.atan2(pts[263][1] - pts[33][1], pts[263][0] - pts[33][0]))
    return np.array([ear, mar, pitch, roll])


def calibrate_from(raws):
    arr = np.array(raws)
    return {"ear": float(np.percentile(arr[:, 0], 80)), "pitch": float(np.median(arr[:, 2]))}


def normalize(raw, base):
    """EAR chia baseline cua tai xe, pitch tru baseline -> model dung duoc cho nhieu nguoi."""
    return np.array([raw[0] / base["ear"], raw[1], raw[2] - base["pitch"], raw[3]])


def window_features(w):
    """w: mang (n,4) [ear_n, mar, dpitch, roll] -> vector dac trung."""
    e, m, p, r = w.T
    return [e.mean(), e.min(), e.std(), float((e < 0.75).mean()),
            m.mean(), m.max(), float((m > 0.6).mean()),
            p.mean(), p.min(), float(np.abs(r).mean())]


def run_calibration(cap, tracker, seconds=8):
    import cv2
    raws, t0 = [], time.time()
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        pts = tracker.landmarks(frame)
        if pts is not None:
            raws.append(raw_features(pts))
        left = seconds - (time.time() - t0)
        # Overlay calibration instructions
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        color = (0, 255, 255)
        cv2.putText(frame, f"CALIBRATION ({left:.0f}s): Nhin thang vao camera, mat mo to",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Frames OK: {len(raws)}  |  Giu dau thang, mat nhin thang",
                    (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
        cv2.imshow("Driver Monitor", frame)
        cv2.waitKey(1)
    if len(raws) < 10:
        raise RuntimeError(
            f"Khong thay khuon mat khi hieu chinh ({len(raws)} frame). "
            "Kiem tra camera va anh sang, dam bao khuon mat trong khung hinh.")
    return calibrate_from(raws)
