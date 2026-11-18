"""Nhan dien TAI XE la ai bang model SFace cua OpenCV (da train san, khong can du lieu cua ban).

- Dung luon landmark MediaPipe de can chinh khuon mat (khong can them model do tim mat).
- Moi tai xe dang ky ~20 "dau van mat" (vector 128 so) luu trong monitor.db (bang faces).
- So khop bang do tuong dong cosine; >= 0.363 la cung mot nguoi (nguong khuyen nghi cua OpenCV).
"""
import os
import urllib.request

import cv2
import numpy as np

MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "face_recognition_sface_2021dec.onnx")
MATCH_THRESH = 0.363

# Landmark MediaPipe cho 5 diem can chinh: 2 mat, mui, 2 khoe mieng
_EYE_A = [33, 133]      # mat ben trai anh
_EYE_B = [362, 263]     # mat ben phai anh
_NOSE = 1
_MOUTH_A, _MOUTH_B = 61, 291


def face_box(pts, frame_shape=None, pad=0.08):
    """Khung bao khuon mat (x, y, w, h) tu landmark."""
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0, x1, y1 = x0 - px, y0 - py, x1 + px, y1 + py
    if frame_shape is not None:
        h, w = frame_shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def _yunet_row(pts, frame_shape):
    """Dung vector khuon mat dang YuNet (15 so) tu landmark MediaPipe cho alignCrop."""
    ea, eb = pts[_EYE_A, :2].mean(0), pts[_EYE_B, :2].mean(0)
    ma, mb = pts[_MOUTH_A, :2], pts[_MOUTH_B, :2]
    if ea[0] > eb[0]:
        ea, eb = eb, ea
    if ma[0] > mb[0]:
        ma, mb = mb, ma
    x, y, w, h = face_box(pts, frame_shape)
    return np.array([[x, y, w, h, *ea, *eb, *pts[_NOSE, :2], *ma, *mb, 1.0]], dtype=np.float32)


class FaceID:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("Dang tai model nhan dien khuon mat SFace (~37 MB)...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
        self.rec = cv2.FaceRecognizerSF.create(model_path, "")

    def embed(self, frame, pts):
        """Vector 128 so (da chuan hoa) dai dien cho khuon mat."""
        crop = self.rec.alignCrop(frame, _yunet_row(pts, frame.shape))
        feat = self.rec.feature(crop).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9)

    @staticmethod
    def identify(emb, gallery):
        """gallery: {ten: mang (k,128)} -> (ten | None, diem). Diem = cosine trung binh top-3."""
        best, best_s = None, -1.0
        for name, embs in gallery.items():
            sims = np.sort(embs @ emb)[::-1]
            s = float(sims[:3].mean())
            if s > best_s:
                best, best_s = name, s
        return (best if best_s >= MATCH_THRESH else None), best_s


def try_load():
    """Tra ve FaceID hoac None neu khong tai duoc model (app van chay, chi khong nhan dien)."""
    try:
        return FaceID()
    except Exception as e:
        print("Khong dung duoc nhan dien tai xe:", e)
        return None


# ─── Ve khung mat + vien mat ─────────────────────────────────────────────────
R_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
L_EYE_CONTOUR = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]


def draw_face(frame, pts, name, closed, color_face=(255, 200, 0)):
    """Khung mat + ten tai xe, vien 2 mat (xanh = mo, do = nham), khung quanh moi mat."""
    x, y, w, h = [int(v) for v in face_box(pts, frame.shape)]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_face, 2)
    label = name or "Nguoi la"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color_face, -1)
    cv2.putText(frame, label, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    eye_col = (0, 0, 255) if closed else (0, 255, 0)
    for contour in (R_EYE_CONTOUR, L_EYE_CONTOUR):
        poly = pts[contour, :2].astype(np.int32)
        cv2.polylines(frame, [poly], True, eye_col, 1, cv2.LINE_AA)
        ex0, ey0 = poly.min(0)
        ex1, ey1 = poly.max(0)
        m = max(4, (ex1 - ex0) // 5)
        cv2.rectangle(frame, (ex0 - m, ey0 - m), (ex1 + m, ey1 + m), eye_col, 1)
    # Mong mat (iris) neu co du 478 diem
    if len(pts) >= 478:
        for c in (468, 473):
            cv2.circle(frame, tuple(pts[c, :2].astype(int)), 2, eye_col, -1)

