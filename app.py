"""Giam sat tai xe real-time – canh bao bang TIENG VIET.
Thu tu uu tien: model_window.joblib > model_frame.joblib > luat EAR/MAR co ban.
Phim: q=thoat  c=hieu chinh lai
"""
import argparse
import collections
import os
import time
import threading

import cv2
import joblib
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont

import db
import features as F

# ─── Text-to-Speech tieng Viet ───────────────────────────────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 155)
    for v in _tts_engine.getProperty("voices"):
        if "vi" in v.id.lower() or "viet" in v.name.lower():
            _tts_engine.setProperty("voice", v.id)
            break
    _TTS_OK = True
except Exception:
    _TTS_OK = False

_tts_lock  = threading.Lock()
_tts_busy  = False


def speak(text: str):
    """Phat am trong luong nen, khong chan vong lap chinh."""
    global _tts_busy
    if not _TTS_OK or _tts_busy:
        return

    def _run():
        global _tts_busy
        with _tts_lock:
            _tts_busy = True
            try:
                _tts_engine.say(text)
                _tts_engine.runAndWait()
            finally:
                _tts_busy = False

    threading.Thread(target=_run, daemon=True).start()


# ─── Font Unicode de ve chu Viet qua Pillow ──────────────────────────────────
_FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/times.ttf",
]
_font_lg = _font_md = _font_sm = None
for _fp in _FONT_PATHS:
    if os.path.exists(_fp):
        _font_lg = ImageFont.truetype(_fp, 40)
        _font_md = ImageFont.truetype(_fp, 28)
        _font_sm = ImageFont.truetype(_fp, 22)
        break
if _font_lg is None:
    _font_lg = _font_md = _font_sm = ImageFont.load_default()


def put_vn(frame, text, pos, font, color=(255, 255, 255), bg=None):
    """Ve chu Unicode len frame BGR cua OpenCV."""
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d   = ImageDraw.Draw(pil)
    x, y = pos
    if bg is not None:
        bb = d.textbbox((x, y), text, font=font)
        p  = 5
        d.rectangle([bb[0]-p, bb[1]-p, bb[2]+p, bb[3]+p], fill=bg)
    d.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ─── Beep ────────────────────────────────────────────────────────────────────
def make_beep(freq, dur):
    t      = np.linspace(0, dur, int(44100 * dur), False)
    mono   = (np.sin(2 * np.pi * freq * t) * 0.6 * 32767).astype(np.int16)
    stereo = np.column_stack([mono, mono])
    return pygame.sndarray.make_sound(stereo)


# ─── Luat du phong ───────────────────────────────────────────────────────────
def rule_predict(f, ear_thresh):
    # f[3] = closed_frac (ty le frame mat nham trong cua so 1.5s)
    if f[3] > ear_thresh or f[7] < -0.12:
        return 1   # buon ngu
    if f[6] > 0.5:
        return 2   # ngap
    return 0


# ─── CLI ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--driver",       default="driver1")
ap.add_argument("--camera",       type=int,   default=0)
ap.add_argument("--model-window", default="model_window.joblib")
ap.add_argument("--model-frame",  default="model_frame.joblib")
ap.add_argument("--warn-after",   type=float, default=1.0,
                help="giay mat nham -> canh bao nhe")
ap.add_argument("--alarm-after",  type=float, default=2.0,
                help="giay mat nham -> bao dong manh")
ap.add_argument("--ear-thresh",   type=float, default=0.60,
                help="EAR_n < nguong nay = mat nham (mac dinh 0.60)")
ap.add_argument("--recalibrate",  action="store_true")
args = ap.parse_args()

pygame.mixer.init(frequency=44100, size=-16, channels=1)
beeps = {1: make_beep(700, 0.3), 2: make_beep(1400, 0.7)}

con = db.connect()
drv = db.get_driver(con, args.driver)

mode, model = "rule", None
if os.path.exists(args.model_window):
    model = joblib.load(args.model_window)["model"]
    mode  = "window"
elif os.path.exists(args.model_frame):
    model = joblib.load(args.model_frame)["model"]
    mode  = "frame"

print("Che do:", {"window": "MODEL WINDOW (chinh xac nhat)",
                  "frame":  "MODEL FRAME (du phong)",
                  "rule":   "LUAT CO BAN (chua co model)"}[mode])

cap     = cv2.VideoCapture(args.camera)
tracker = F.FaceTracker()
base    = None if args.recalibrate else db.get_baseline(drv)
if base is None:
    base = F.run_calibration(cap, tracker)
    db.save_baseline(con, drv["id"], base)

sid        = db.new_session(con, drv["id"], "drive")
win        = collections.deque(maxlen=F.WIN)
vote       = collections.deque(maxlen=7)
bad_since  = no_face_since = None
level      = 0
last_beep  = last_yawn = last_speak = 0.0
pred, conf = 0, 0.0
TTS_CD     = 4.0   # cooldown giua cac lan phat am (giay)

STATE_VN = {0: "Tỉnh táo ✓", 1: "BUỒN NGỦ ⚠", 2: "ĐANG NGÁP"}

# ─── Vong lap chinh ──────────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        break
    now              = time.time()
    pts              = tracker.landmarks(frame)
    new_level, etype = 0, "drowsy"

    # --- Khong co khuon mat ---
    if pts is None:
        win.clear(); vote.clear(); bad_since = None
        no_face_since = no_face_since or now
        if now - no_face_since > 2:
            new_level, etype = 1, "no_face"
        status_id = -1
    else:
        no_face_since = None
        n = F.normalize(F.raw_features(pts), base)

        # --- Du doan ---
        if mode == "frame":
            proba     = model.predict_proba([n])[0]
            vote.append(int(model.classes_[proba.argmax()]))
            pred      = collections.Counter(vote).most_common(1)[0][0]
            conf      = float(np.mean([v == pred for v in vote]))
        else:
            win.append(n)
            if len(win) == F.WIN:
                f = F.window_features(np.array(win))
                if mode == "window":
                    proba     = model.predict_proba([f])[0]
                    pred, conf = int(model.classes_[proba.argmax()]), float(proba.max())
                else:
                    pred, conf = rule_predict(f, 1 - args.ear_thresh), 1.0

        # --- Ep buon ngu neu mat nham tit ---
        # n[0] = EAR_normalized; < ear_thresh nghia la mat dang nham
        if n[0] < args.ear_thresh and pred == 0:
            pred = 1
            conf = 0.92

        # --- Muc canh bao ---
        if pred == 1:
            bad_since = bad_since or now
            dur       = now - bad_since
            new_level = (2 if dur >= args.alarm_after
                         else 1 if dur >= args.warn_after
                         else 0)
        else:
            bad_since = None

        # --- Phat hien ngap ---
        if pred == 2 and now - last_yawn > 5:
            last_yawn = now
            db.add_event(con, sid, "yawn", 1, conf)
            beeps[1].play()
            if now - last_speak > TTS_CD:
                speak("Bạn đang ngáp. Hãy nghỉ ngơi!")
                last_speak = now

        status_id = pred

        # Ve diem mat
        for i in F.L_EYE + F.R_EYE:
            cv2.circle(frame, tuple(pts[i].astype(int)), 1, (0, 255, 0), -1)
        cv2.putText(frame,
                    f"EAR_n={n[0]:.2f}  MAR={n[1]:.2f}  pitch={n[2]:+.2f}",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # --- Cap nhat muc + beep ---
    if new_level > level:
        db.add_event(con, sid, etype, new_level, conf)
    level = new_level

    if level and now - last_beep > (0.7 if level == 2 else 2.0):
        beeps[level].play()
        last_beep = now

    # --- TTS canh bao buon ngu ---
    if level == 1 and now - last_speak > TTS_CD:
        speak("Cảnh báo! Bạn đang buồn ngủ. Hãy tỉnh táo!")
        last_speak = now
    elif level == 2 and now - last_speak > TTS_CD:
        speak("Nguy hiểm! Mắt nhắm quá lâu. Dừng xe ngay!")
        last_speak = now

    # --- Overlay canh bao ---
    h, w = frame.shape[:2]
    if level:
        color = (0, 0, 255) if level == 2 else (0, 140, 255)
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 14)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-95), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        warn_text = ("🚨 NGUY HIỂM! DỪNG XE NGAY!"
                     if level == 2
                     else "⚠  CẢNH BÁO: BẠN ĐANG BUỒN NGỦ!")
        put_vn(frame, warn_text, (18, h - 82), _font_lg,
               color=(255, 255, 255),
               bg=(160, 0, 0) if level == 2 else (0, 70, 180))

    # --- Thanh trang thai ---
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, 0), (w, 48), (0, 0, 0), -1)
    cv2.addWeighted(ov2, 0.5, frame, 0.5, 0, frame)

    if status_id == -1:
        state_str = "Không thấy mặt"
        sc = (100, 100, 255)
    else:
        state_str = STATE_VN.get(status_id, "?")
        sc = (80, 255, 80) if status_id == 0 else (80, 80, 255)

    put_vn(frame, f"[{mode}]  {state_str}  ({conf:.0%})",
           (12, 8), _font_sm, color=sc)

    cv2.imshow("Driver Monitor – Giam sat tai xe", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("c"):
        base = F.run_calibration(cap, tracker)
        db.save_baseline(con, drv["id"], base)
        win.clear(); vote.clear()

db.end_session(con, sid)
cap.release()
cv2.destroyAllWindows()
