"""Giam sat buon ngu real-time – canh bao bang TIENG VIET.
Thu tu uu tien: model_window.joblib > model_frame.joblib > luat EAR/MAR co ban.
Phim: q=thoat  c=hieu chinh lai
"""
import argparse
import collections
import os
import time
import threading

import cv2
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont

import db
import detector as D
import face_id as FID
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


# ─── CLI ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--user",       default=None,
                help="ten tai xe; bo trong = tu nhan dien bang khuon mat")
ap.add_argument("--no-faceid",    action="store_true", help="tat nhan dien tai xe")
ap.add_argument("--camera",       type=int,   default=0)
ap.add_argument("--recalibrate",  action="store_true")
D.add_detector_args(ap)
args = ap.parse_args()

pygame.mixer.init(frequency=44100, size=-16, channels=1)
beeps = {1: make_beep(700, 0.3), 2: make_beep(1400, 0.7)}

con = db.connect()
mode, model = D.load_model(args.model_window, args.model_frame)
print("Che do:", {"window": "MODEL WINDOW cua ban + MediaPipe",
                  "frame":  "MODEL FRAME cua ban + MediaPipe",
                  "rule":   "CHI MediaPipe (chua train model rieng)"}[mode])
print(f"Lam muot: {args.smooth_frames} frame, doi trang thai khi >= {args.switch_ratio:.0%}")

cap     = cv2.VideoCapture(args.camera)
tracker = F.FaceTracker()

# ─── Nhan dien tai xe ────────────────────────────────────────────────────────
fid     = None if args.no_faceid else FID.try_load()
gallery = db.load_gallery(con) if fid else {}
if fid and not gallery:
    print("Chua co tai xe nao dang ky khuon mat -> chay: python enroll.py --user ten_tai_xe")


def identify_driver(seconds=3.0):
    """Nhin camera vai giay, bo phieu -> ten tai xe da dang ky hoac None."""
    votes, t0 = collections.Counter(), time.time()
    while time.time() - t0 < seconds:
        ok, fr = cap.read()
        if not ok:
            break
        p = tracker.landmarks(fr)
        if p is not None:
            name, s = FID.FaceID.identify(fid.embed(fr, p), gallery)
            votes[name] += 1
            FID.draw_face(fr, p, f"{name or 'Nguoi la'} ({s:.2f})", closed=False)
        cv2.rectangle(fr, (0, 0), (fr.shape[1], 45), (0, 0, 0), -1)
        cv2.putText(fr, "DANG NHAN DIEN TAI XE... nhin vao camera", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Drowsiness Monitor", fr)
        cv2.waitKey(1)
    if not votes:
        return None
    name, cnt = votes.most_common(1)[0]
    return name if cnt >= 0.6 * sum(votes.values()) else None


user_name = args.user
if user_name is None:
    found = identify_driver() if gallery else None
    user_name = found or "khach"
    print("Tai xe:", found or "KHONG nhan ra -> dung ho so 'khach' "
          "(dang ky bang: python enroll.py --user ten_ban)")
usr = db.get_user(con, user_name)

base = None if args.recalibrate else db.get_baseline(usr)
if base is None:
    base = F.run_calibration(cap, tracker)
    db.save_baseline(con, usr["id"], base)

det        = D.Detector(mode, model, args)
sid        = db.new_session(con, usr["id"], "monitor")
seen_name, seen_score = (user_name if user_name in gallery else None), 0.0
name_votes = collections.deque(maxlen=5)     # lam muot ten hien thi (5 lan nhan dien ~5s)
switch_votes = collections.deque(maxlen=3)   # 3 lan lien tiep thay nguoi khac -> doi ho so
last_id    = 0.0
ID_EVERY   = 1.0   # giay giua 2 lan nhan dien lai
level, cur_etype = 0, "drowsy"
last_beep  = last_yawn = last_speak = 0.0
TTS_CD     = 4.0   # cooldown giua cac lan phat am (giay)

STATE_VN = {D.ALERT: "Tỉnh táo ✓", D.DROWSY: "BUỒN NGỦ ⚠", D.YAWN: "ĐANG NGÁP",
            D.AWAY: "Đang nhìn chỗ khác", D.NO_FACE: "Không thấy mặt"}

# ─── Vong lap chinh ──────────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        break
    now = time.time()
    pts = tracker.landmarks(frame)

    if pts is None:
        r = det.no_face(now)
    else:
        raw = F.raw_features(pts, tracker.extras)

        # --- Nhan dien lai tai xe moi ~1s (chi khi mat gan nhin thang) ---
        if fid and gallery and now - last_id > ID_EVERY and abs(raw[4]) < 0.2:
            last_id = now
            nm, seen_score = FID.FaceID.identify(fid.embed(frame, pts), gallery)
            name_votes.append(nm)
            switch_votes.append(nm)
            seen_name = collections.Counter(name_votes).most_common(1)[0][0]
            # Doi tai xe: 3 lan lien tiep la nguoi khac (da dang ky + da hieu chinh)
            if (args.user is None and nm and nm != usr["name"]
                    and len(switch_votes) == 3 and len(set(switch_votes)) == 1):
                new_usr  = db.get_user(con, nm)
                new_base = db.get_baseline(new_usr)
                if new_base is not None:
                    print(f"Doi tai xe: {usr['name']} -> {nm}")
                    db.end_session(con, sid)
                    usr, base = new_usr, new_base
                    sid = db.new_session(con, usr["id"], "monitor")
                    det.reset()
                    speak(f"Xin chào {nm}")

        n = F.normalize(raw, base)
        r = det.update(n, now)

        # --- Phat hien ngap (co cooldown 5s) ---
        if r["yawn"] and now - last_yawn > 5:
            last_yawn = now
            db.add_event(con, sid, "yawn", 1, r["conf"])
            beeps[1].play()
            if now - last_speak > TTS_CD:
                speak("Bạn đang ngáp. Hãy nghỉ ngơi!")
                last_speak = now

        # Ve khung mat + ten tai xe + vien mat (xanh = mo, do = nham)
        if fid and gallery:
            label = f"{seen_name or 'Nguoi la'} ({seen_score:.2f})"
            fcol  = (255, 200, 0) if seen_name else (0, 165, 255)
        else:
            label, fcol = usr["name"], (255, 200, 0)
        FID.draw_face(frame, pts, label, closed=r["closed"], color_face=fcol)
        dn = det.disp_n
        cv2.putText(frame,
                    f"Nham mat={dn[5]:.2f}  Ha mieng={dn[6]:.2f}  pitch={dn[2]:+.2f}  yaw={dn[4]:+.2f}",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    status_id = r["status"]

    # --- Cap nhat muc + beep ---
    if r["level"] > level or (r["level"] and r["etype"] != cur_etype):
        db.add_event(con, sid, r["etype"], r["level"], r["conf"])
    level, cur_etype = r["level"], r["etype"]

    if level and now - last_beep > (0.7 if level == 2 else 2.0):
        beeps[level].play()
        last_beep = now

    # --- TTS canh bao ---
    if level and now - last_speak > TTS_CD:
        if cur_etype == "distracted":
            speak("Hãy nhìn về phía trước!")
        elif cur_etype == "no_face":
            speak("Không thấy khuôn mặt.")
        elif level == 1:
            speak("Cảnh báo! Bạn đang buồn ngủ. Hãy tỉnh táo!")
        else:
            speak("Nguy hiểm! Mắt nhắm quá lâu. Hãy nghỉ ngơi ngay!")
        last_speak = now

    # --- Overlay canh bao ---
    h, w = frame.shape[:2]
    if level:
        color = (0, 0, 255) if level == 2 else (0, 140, 255)
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 14)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-95), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        if cur_etype == "distracted":
            warn_text = "👀 HÃY NHÌN VỀ PHÍA TRƯỚC!"
        elif cur_etype == "no_face":
            warn_text = "⚠  KHÔNG THẤY KHUÔN MẶT"
        elif level == 2:
            warn_text = "🚨 NGUY HIỂM! HÃY NGHỈ NGƠI NGAY!"
        else:
            warn_text = "⚠  CẢNH BÁO: BẠN ĐANG BUỒN NGỦ!"
        put_vn(frame, warn_text, (18, h - 82), _font_lg,
               color=(255, 255, 255),
               bg=(160, 0, 0) if level == 2 else (0, 70, 180))

    # --- Thanh trang thai ---
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, 0), (w, 48), (0, 0, 0), -1)
    cv2.addWeighted(ov2, 0.5, frame, 0.5, 0, frame)
    state_str = STATE_VN.get(status_id, "?")
    sc = ((80, 255, 80) if status_id == D.ALERT
          else (0, 220, 255) if status_id == D.AWAY
          else (100, 100, 255) if status_id == D.NO_FACE
          else (80, 80, 255))
    put_vn(frame, f"[{mode}+MediaPipe]  {state_str}  ({r['conf']:.0%})",
           (12, 8), _font_sm, color=sc)

    cv2.imshow("Drowsiness Monitor", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("c"):
        base = F.run_calibration(cap, tracker)
        db.save_baseline(con, usr["id"], base)
        det.reset()

db.end_session(con, sid)
cap.release()
cv2.destroyAllWindows()
