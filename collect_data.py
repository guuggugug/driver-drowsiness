"""Thu dataset tu webcam va luu vao SQLite.
Phim: 0=tinh tao  1=buon ngu (nham mat lau/lim dim/gat dau)  2=ngap
      SPACE=bat/tat ghi   c=hieu chinh lai   q=thoat"""
import argparse
import time
import cv2
import db
import features as F

ap = argparse.ArgumentParser()
ap.add_argument("--user", default="user1")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--recalibrate", action="store_true")
args = ap.parse_args()

con = db.connect()
usr = db.get_user(con, args.user)
cap = cv2.VideoCapture(args.camera)
tracker = F.FaceTracker()

base = None if args.recalibrate else db.get_baseline(usr)
if base is None:
    base = F.run_calibration(cap, tracker)
    db.save_baseline(con, usr["id"], base)

label, recording, sid, buf = 0, False, None, []
counts = {0: 0, 1: 0, 2: 0}

while True:
    ok, frame = cap.read()
    if not ok:
        break
    pts = tracker.landmarks(frame)
    n = None
    if pts is not None:
        n = F.normalize(F.raw_features(pts, tracker.extras), base)
    if n is not None and recording:
        buf.append(db.sample_row(sid, time.time(), n, label))
        counts[label] += 1
        if len(buf) >= 100:
            db.add_samples(con, buf)
            buf = []

    color = (0, 0, 255) if recording else (200, 200, 200)
    cv2.putText(frame, f"{'REC' if recording else 'PAUSED'} | label={label}:{F.LABELS[label]}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"alert={counts[0]} drowsy={counts[1]} yawn={counts[2]}" +
                (f"  blink={n[5]:.2f} yaw={n[4]:+.2f}" if n is not None else "  NO FACE"),
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if label == 0:
        cv2.putText(frame, "Goi y: mat MO, quay dau trai/phai, nhin guong, nhin xuong, nhin len",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.imshow("Drowsiness Monitor", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k in (ord("0"), ord("1"), ord("2")):
        label = int(chr(k))
    elif k == ord(" "):
        recording = not recording
        if recording:
            sid = db.new_session(con, usr["id"], "collect")   # moi lan bat ghi = 1 phien lien tuc
        else:
            if buf:
                db.add_samples(con, buf)
                buf = []
            db.end_session(con, sid)
    elif k == ord("c"):
        base = F.run_calibration(cap, tracker)
        db.save_baseline(con, usr["id"], base)

if buf:
    db.add_samples(con, buf)
if recording and sid:
    db.end_session(con, sid)
cap.release()
cv2.destroyAllWindows()
print("Da luu:", counts, "->", db.DB_PATH)
