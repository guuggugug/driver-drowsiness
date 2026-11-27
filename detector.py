"""Bo quyet dinh trang thai (dung chung cho app.py va evaluate.py).

Dau vao moi frame: vector n da normalize [ear_n, mar, dpitch, roll, yaw, blink, jaw].
Dau ra: trang thai hien thi da LAM MUOT + muc canh bao.

Cac co che chong "nhay chu":
  - smooth_frames : so frame bo phieu cho trang thai hien thi (mac dinh 30 ~ 1 giay)
  - switch_ratio  : trang thai moi phai chiem >= ty le nay trong cua so moi duoc doi
  - hysteresis yaw: vao "nhin cho khac" o yaw_thresh, thoat o 0.8 * yaw_thresh
  - release_after : mat mo lai < release_after giay thi chua huy dem thoi gian nham mat
  - hold_alert    : canh bao da bat thi giu toi thieu hold_alert giay
"""
import collections
import os

import joblib
import numpy as np

import features as F

FI = F.FI
NO_FACE, ALERT, DROWSY, YAWN, AWAY = -1, 0, 1, 2, 3
STATE_NAMES = {NO_FACE: "khong_thay_mat", ALERT: "tinh_tao", DROWSY: "buon_ngu",
               YAWN: "ngap", AWAY: "nhin_cho_khac"}


def rule_predict(f, blink_thresh=F.BLINK_CLOSED):
    """Luat dua tren model co san cua MediaPipe (khi chua train model rieng)."""
    if f[FI["away_frac"]] > 0.3:
        return ALERT
    if f[FI["blink_closed_frac"]] > 0.6:
        return DROWSY      # mat nham phan lon thoi gian trong cua so ~1.5s
    if f[FI["pitch_mean"]] < -0.15 and (
            f[FI["blink_closed_frac"]] > 0.35 or f[FI["blink_mean"]] > blink_thresh * 0.9):
        return DROWSY      # gat dau + mat nham / lim dim ro
    if f[FI["jaw_frac"]] > 0.5 or f[FI["yawn_frac"]] > 0.6:
        return YAWN        # mieng ha to KEO DAI (>~0.75s) -> ngap, khong phai noi chuyen
    return ALERT


def load_model(model_window="model_window.joblib", model_frame="model_frame.joblib"):
    """-> (mode, model). mode in {'window', 'frame', 'rule'}."""
    for m, path, nfeat in (("window", model_window, len(F.FEATURE_NAMES)),
                           ("frame", model_frame, len(F.FRAME_FEATURES))):
        if path and os.path.exists(path):
            mdl = joblib.load(path)["model"]
            if getattr(mdl, "n_features_in_", nfeat) != nfeat:
                print(f"Bo qua {path}: model cu (khac bo dac trung) -> hay chay lai train.py")
                continue
            return m, mdl
    return "rule", None


def add_detector_args(ap):
    """Them cac tham so cua bo quyet dinh vao argparse (dung chung app/evaluate)."""
    ap.add_argument("--model-window", default="model_window.joblib")
    ap.add_argument("--model-frame", default="model_frame.joblib")
    ap.add_argument("--warn-after", type=float, default=1.0, help="giay mat nham -> canh bao nhe")
    ap.add_argument("--alarm-after", type=float, default=2.0, help="giay mat nham -> bao dong manh")
    ap.add_argument("--blink-thresh", type=float, default=F.BLINK_CLOSED,
                    help="diem nham mat > nguong nay = mat nham (mac dinh 0.5)")
    ap.add_argument("--open-veto", type=float, default=0.25,
                    help="diem nham mat TB < nguong nay = mat MO ro -> bo canh bao buon ngu cua model")
    ap.add_argument("--yaw-thresh", type=float, default=F.YAW_AWAY,
                    help="|yaw| > nguong nay = dang quay mat")
    ap.add_argument("--distract-after", type=float, default=3.0,
                    help="giay quay mat lien tuc -> nhac nhin ve phia truoc (0 = tat)")
    ap.add_argument("--smooth-frames", type=int, default=30,
                    help="so frame lam muot trang thai hien thi (cang lon cang it nhay, mac dinh 30)")
    ap.add_argument("--switch-ratio", type=float, default=0.6,
                    help="trang thai moi phai chiem ty le nay trong cua so moi duoc doi")
    ap.add_argument("--release-after", type=float, default=0.4,
                    help="mat mo lai ngan hon so giay nay thi van tinh la dang nham")
    ap.add_argument("--hold-alert", type=float, default=1.5,
                    help="canh bao da bat thi giu toi thieu so giay nay")
    return ap


class Detector:
    def __init__(self, mode, model, cfg):
        self.mode, self.model, self.c = mode, model, cfg
        self.reset()

    def reset(self):
        self.win = collections.deque(maxlen=F.WIN)
        self.vote = collections.deque(maxlen=7)
        self.blink_hist = collections.deque(maxlen=15)
        self.status_hist = collections.deque(maxlen=max(1, self.c.smooth_frames))
        self.status = ALERT
        self.away_state = False
        self.bad_since = self.last_bad = self.away_since = self.no_face_since = None
        self.level, self.etype, self.level_until = 0, "drowsy", 0.0
        self.pred, self.conf, self.conf_disp = ALERT, 0.0, 0.0
        self.disp_n = None

    # ── Lam muot trang thai hien thi: chi doi khi trang thai moi ap dao ──
    def _smooth_status(self, raw_status):
        self.status_hist.append(raw_status)
        cand, cnt = collections.Counter(self.status_hist).most_common(1)[0]
        if cand != self.status and cnt >= self.c.switch_ratio * len(self.status_hist):
            self.status = cand
        return self.status

    def _set_level(self, new_level, etype, now):
        """Canh bao da bat thi giu toi thieu hold_alert giay (khong chop tat)."""
        if new_level > 0:
            if new_level >= self.level or now >= self.level_until:
                self.level, self.etype = new_level, etype
                self.level_until = now + self.c.hold_alert
        elif now >= self.level_until:
            self.level = 0

    def no_face(self, now):
        self.win.clear(); self.vote.clear(); self.blink_hist.clear()
        self.away_state = False
        self.bad_since = self.last_bad = self.away_since = None
        self.disp_n = None
        self.no_face_since = self.no_face_since or now
        lvl = 1 if now - self.no_face_since > 2 else 0
        self._set_level(lvl, "no_face", now)
        status = self._smooth_status(NO_FACE)
        return {"status": status, "raw_status": NO_FACE, "level": self.level,
                "etype": self.etype, "conf": self.conf_disp, "yawn": False}

    def update(self, n, now):
        c = self.c
        self.no_face_since = None
        self.disp_n = n.copy() if self.disp_n is None else 0.8 * self.disp_n + 0.2 * n

        # Hysteresis goc quay dau
        thr = c.yaw_thresh * (0.8 if self.away_state else 1.0)
        self.away_state = abs(n[4]) > thr
        away = self.away_state
        self.blink_hist.append(n[5])
        blink_avg = float(np.mean(self.blink_hist))

        # 1) Model rieng / luat MediaPipe
        pred, conf = self.pred, self.conf
        if self.mode == "frame":
            proba = self.model.predict_proba([n])[0]
            self.vote.append(int(self.model.classes_[proba.argmax()]))
            pred = collections.Counter(self.vote).most_common(1)[0][0]
            conf = float(np.mean([v == pred for v in self.vote]))
        else:
            self.win.append(n)
            if len(self.win) == F.WIN:
                f = F.window_features(np.array(self.win))
                if self.mode == "window":
                    proba = self.model.predict_proba([f])[0]
                    pred, conf = int(self.model.classes_[proba.argmax()]), float(proba.max())
                else:
                    pred, conf = rule_predict(f, c.blink_thresh), 1.0
        self.pred, self.conf = pred, conf

        # 2) Ket hop MediaPipe: phu quyet khi mat MO ro, ep buon ngu khi mat nham
        if pred == DROWSY and blink_avg < c.open_veto and n[2] > -0.15:
            pred = ALERT
        new_level, etype = 0, "drowsy"
        if away:
            self.away_since = self.away_since or now
            if pred == DROWSY:
                pred = ALERT
            if c.distract_after > 0 and now - self.away_since >= c.distract_after:
                new_level, etype = 1, "distracted"
        else:
            self.away_since = None
            # Dung diem nham mat TRUNG BINH ~0.5s (khong dung tung frame) de nhieu
            # nhat thoi (cui dau, anh sang) khong bi tinh la nham mat.
            if blink_avg > c.blink_thresh and n[6] < F.JAW_OPEN and pred == ALERT:
                pred, conf = DROWSY, blink_avg

        # 3) Dem thoi gian nham mat (co thoi gian an han release_after)
        if pred == DROWSY:
            self.bad_since = self.bad_since or now
            self.last_bad = now
        elif self.last_bad is not None and now - self.last_bad > c.release_after:
            self.bad_since = self.last_bad = None
        if self.bad_since is not None:
            dur = now - self.bad_since
            lvl = 2 if dur >= c.alarm_after else 1 if dur >= c.warn_after else 0
            if lvl:
                new_level, etype = lvl, "drowsy"
        self._set_level(new_level, etype, now)

        raw_status = AWAY if (away and pred != YAWN) else pred
        status = self._smooth_status(raw_status)
        self.conf_disp = 0.85 * self.conf_disp + 0.15 * conf
        return {"status": status, "raw_status": raw_status, "level": self.level,
                "etype": self.etype, "conf": self.conf_disp,
                "yawn": status == YAWN, "closed": blink_avg > c.blink_thresh}
