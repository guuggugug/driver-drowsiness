"""Thu dataset tu webcam va luu vao SQLite.
Phim: 0=tinh tao  1=buon ngu (nham mat lau/lim dim/gat dau)  2=ngap
      SPACE=bat/tat ghi   c=hieu chinh lai   q=thoat"""
import argparse
import time
import cv2
import db
import features as F

ap = argparse.ArgumentParser()
ap.add_argument("--driver", default="driver1")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--recalibrate", action="store_true")
args = ap.parse_args()

con = db.connect()
drv = db.get_driver(con, args.driver)
cap = cv2.VideoCapture(args.camera)
tracker = F.FaceTracker()

base = None if args.recalibrate else db.get_baseline(drv)
if base is None:
    base = F.run_calibration(cap, tracker)
    db.save_baseline(con, drv["id"], base)

label, recording, sid, buf = 0, False, None, []
counts = {0: 0, 1: 0, 2: 0}

while True:
    ok, frame = cap.read()
    if not ok:
        break
    pts = tracker.landmarks(frame)
    if pts is not None and recording:
        n = F.normalize(F.raw_features(pts), base)
        buf.append((sid, time.time(), float(n[0]), float(n[1]), float(n[2]), float(n[3]), label))
        counts[label] += 1
        if len(buf) >= 100:
            db.add_samples(con, buf)
            buf = []

    color = (0, 0, 255) if recording else (200, 200, 200)
    cv2.putText(frame, f"{'REC' if recording else 'PAUSED'} | label={label}:{F.LABELS[label]}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"alert={counts[0]} drowsy={counts[1]} yawn={counts[2]}" +
                ("" if pts is not None else "  NO FACE"),
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Driver Monitor", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k in (ord("0"), ord("1"), ord("2")):
        label = int(chr(k))
    elif k == ord(" "):
        recording = not recording
        if recording:
            sid = db.new_session(con, drv["id"], "collect")   # moi lan bat ghi = 1 phien lien tuc
        else:
            if buf:
                db.add_samples(con, buf)
                buf = []
            db.end_session(con, sid)
    elif k == ord("c"):
        base = F.run_calibration(cap, tracker)
        db.save_baseline(con, drv["id"], base)

if buf:
    db.add_samples(con, buf)
if recording and sid:
    db.end_session(con, sid)
cap.release()
cv2.destroyAllWindows()
print("Da luu:", counts, "->", db.DB_PATH)
