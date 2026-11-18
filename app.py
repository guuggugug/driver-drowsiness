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
import joblib
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont

import db
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


# ─── Luat du phong (dua tren model co san cua MediaPipe) ─────────────────────
FI = F.FI


def rule_predict(f):
    if f[FI["away_frac"]] > 0.3:
        return 0   # dang nhin cho khac -> khong ket luan buon ngu
    if f[FI["blink_closed_frac"]] > 0.6:
        return 1   # mat nham phan lon thoi gian trong cua so ~1.5s
    if f[FI["pitch_mean"]] < -0.15 and f[FI["blink_mean"]] > 0.35:
        return 1   # gat dau + mat lim dim
    if f[FI["jaw_frac"]] > 0.3 or f[FI["yawn_frac"]] > 0.5:
        return 2   # ngap
    return 0


# ─── CLI ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--user",       default=None,
                help="ten tai xe; bo trong = tu nhan dien bang khuon mat")
ap.add_argument("--no-faceid",    action="store_true", help="tat nhan dien tai xe")
ap.add_argument("--camera",       type=int,   default=0)
ap.add_argument("--model-window", default="model_window.joblib")
ap.add_argument("--model-frame",  default="model_frame.joblib")
ap.add_argument("--warn-after",   type=float, default=1.0,
                help="giay mat nham -> canh bao nhe")
ap.add_argument("--alarm-after",  type=float, default=2.0,
                help="giay mat nham -> bao dong manh")
ap.add_argument("--blink-thresh", type=float, default=F.BLINK_CLOSED,
                help="diem nham mat (MediaPipe) > nguong nay = mat nham (mac dinh 0.5)")
ap.add_argument("--open-veto",    type=float, default=0.25,
                help="diem nham mat TB < nguong nay = mat dang MO ro -> bo canh bao buon ngu cua model")
ap.add_argument("--recalibrate",  action="store_true")
ap.add_argument("--yaw-thresh",   type=float, default=F.YAW_AWAY,
                help="|yaw| > nguong nay = dang quay mat (khong tinh la buon ngu)")
ap.add_argument("--distract-after", type=float, default=3.0,
                help="giay quay mat lien tuc -> nhac nhin ve phia truoc (0 = tat)")
args = ap.parse_args()

pygame.mixer.init(frequency=44100, size=-16, channels=1)
beeps = {1: make_beep(700, 0.3), 2: make_beep(1400, 0.7)}

con = db.connect()

mode, model = "rule", None
for _m, _path, _nfeat in (("window", args.model_window, len(F.FEATURE_NAMES)),
                          ("frame",  args.model_frame,  len(F.FRAME_FEATURES))):
    if os.path.exists(_path):
        _mdl = joblib.load(_path)["model"]
        if getattr(_mdl, "n_features_in_", _nfeat) != _nfeat:
            print(f"Bo qua {_path}: model cu (khac bo dac trung) -> hay chay lai train.py")
            continue
        mode, model = _m, _mdl
        break

print("Che do:", {"window": "MODEL WINDOW cua ban + MediaPipe",
                  "frame":  "MODEL FRAME cua ban + MediaPipe",
                  "rule":   "CHI MediaPipe (chua train model rieng)"}[mode])

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

base    = None if args.recalibrate else db.get_baseline(usr)
if base is None:
    base = F.run_calibration(cap, tracker)
    db.save_baseline(con, usr["id"], base)

sid        = db.new_session(con, usr["id"], "monitor")
seen_name, seen_score = (user_name if user_name in gallery else None), 0.0
switch_votes = collections.deque(maxlen=3)   # 3 lan lien tiep thay nguoi khac -> doi ho so
last_id    = 0.0
ID_EVERY   = 1.0   # giay giua 2 lan nhan dien lai
win        = collections.deque(maxlen=F.WIN)
vote       = collections.deque(maxlen=7)
blink_hist = collections.deque(maxlen=15)   # ~0.5s diem nham mat gan nhat
bad_since  = no_face_since = away_since = None
level      = 0
cur_etype  = "drowsy"
last_beep  = last_yawn = last_speak = 0.0
pred, conf = 0, 0.0
TTS_CD     = 4.0   # cooldown giua cac lan phat am (giay)

STATE_VN = {0: "Tỉnh táo ✓", 1: "BUỒN NGỦ ⚠", 2: "ĐANG NGÁP", 3: "Đang nhìn chỗ khác"}

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
        win.clear(); vote.clear(); blink_hist.clear(); bad_since = None; away_since = None
        no_face_since = no_face_since or now
        if now - no_face_since > 2:
            new_level, etype = 1, "no_face"
        status_id = -1
    else:
        no_face_since = None
        raw  = F.raw_features(pts, tracker.extras)

        # --- Nhan dien lai tai xe moi ~1s (chi khi mat gan nhin thang) ---
        if fid and gallery and now - last_id > ID_EVERY and abs(raw[4]) < 0.2:
            last_id = now
            seen_name, seen_score = FID.FaceID.identify(fid.embed(frame, pts), gallery)
            switch_votes.append(seen_name)
            # Doi tai xe: 3 lan lien tiep la nguoi khac (da dang ky + da hieu chinh) -> doi ho so
            if (args.user is None and seen_name and seen_name != usr["name"]
                    and len(switch_votes) == 3 and len(set(switch_votes)) == 1):
                new_usr  = db.get_user(con, seen_name)
                new_base = db.get_baseline(new_usr)
                if new_base is not None:
                    print(f"Doi tai xe: {usr['name']} -> {seen_name}")
                    db.end_session(con, sid)
                    usr, base = new_usr, new_base
                    sid = db.new_session(con, usr["id"], "monitor")
                    win.clear(); vote.clear(); blink_hist.clear(); bad_since = None
                    speak(f"Xin chào {seen_name}")

        n    = F.normalize(raw, base)
        away = F.looking_away(n, args.yaw_thresh)
        blink_hist.append(n[5])
        blink_avg = float(np.mean(blink_hist))

        # --- 1) Du doan bang model cua ban (hoac luat MediaPipe neu chua co) ---
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
                    pred, conf = rule_predict(f), 1.0

        # --- 2) Ket hop voi model co san MediaPipe ---
        # Phu quyet: model bao buon ngu nhung MediaPipe thay mat dang MO ro
        # (va khong gat dau) -> khong bao.
        if pred == 1 and blink_avg < args.open_veto and n[2] > -0.15:
            pred = 0

        if away:
            # Quay mat sang ben -> KHONG coi la buon ngu
            away_since = away_since or now
            if pred == 1:
                pred = 0
            if args.distract_after > 0 and now - away_since >= args.distract_after:
                new_level, etype = 1, "distracted"
        else:
            away_since = None
            # Ep buon ngu neu MediaPipe thay mat nham (khi nhin thang, khong dang ngap).
            # Chop mat binh thuong ~0.3s se bi loc boi --warn-after (1s).
            if n[5] > args.blink_thresh and n[6] < F.JAW_OPEN and pred == 0:
                pred = 1
                conf = float(n[5])

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

        status_id = 3 if (away and pred != 2) else pred

        # Ve khung mat + ten tai xe + vien mat (xanh = mo, do = nham)
        if fid and gallery:
            label = f"{seen_name or 'Nguoi la'} ({seen_score:.2f})"
            fcol  = (255, 200, 0) if seen_name else (0, 165, 255)
        else:
            label, fcol = usr["name"], (255, 200, 0)
        FID.draw_face(frame, pts, label, closed=n[5] > args.blink_thresh, color_face=fcol)
        cv2.putText(frame,
                    f"Nham mat={n[5]:.2f}  Ha mieng={n[6]:.2f}  pitch={n[2]:+.2f}  yaw={n[4]:+.2f}",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # --- Cap nhat muc + beep ---
    if new_level > level or (new_level and etype != cur_etype):
        db.add_event(con, sid, etype, new_level, conf)
    level, cur_etype = new_level, etype

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

    if status_id == -1:
        state_str = "Không thấy mặt"
        sc = (100, 100, 255)
    else:
        state_str = STATE_VN.get(status_id, "?")
        sc = ((80, 255, 80) if status_id == 0
              else (0, 220, 255) if status_id == 3
              else (80, 80, 255))

    put_vn(frame, f"[{mode}+MediaPipe]  {state_str}  ({conf:.0%})",
           (12, 8), _font_sm, color=sc)

    cv2.imshow("Drowsiness Monitor – Giam sat buon ngu", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("c"):
        base = F.run_calibration(cap, tracker)
        db.save_baseline(con, usr["id"], base)
        win.clear(); vote.clear()

db.end_session(con, sid)
cap.release()
cv2.destroyAllWindows()
