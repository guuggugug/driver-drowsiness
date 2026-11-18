"""Dang ky khuon mat tai xe de app tu nhan ra ai dang lai.

    python enroll.py --user ten_tai_xe

Nhin vao camera, xoay dau nhe trai/phai/len/xuong theo huong dan (~10 giay).
Sau do app hieu chinh luon mat cua tai xe (8 giay, nhin thang, mat mo binh thuong).
Phim: q = huy.
"""
import argparse
import time

import cv2
import numpy as np

import db
import face_id as FID
import features as F

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True, help="ten tai xe")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--add", action="store_true", help="them anh vao ho so cu thay vi ghi de")
args = ap.parse_args()

fid = FID.FaceID()
con = db.connect()
usr = db.get_user(con, args.user)
gallery = {k: v for k, v in db.load_gallery(con).items() if k != args.user}

cap = cv2.VideoCapture(args.camera)
tracker = F.FaceTracker()
STEPS = ["Nhin THANG vao camera", "Xoay nhe sang TRAI", "Xoay nhe sang PHAI",
         "Ngang dau LEN mot chut", "Cui XUONG mot chut"]
embs, last, t0 = [], 0.0, time.time()
while time.time() - t0 < args.seconds:
    ok, frame = cap.read()
    if not ok:
        break
    now = time.time()
    pts = tracker.landmarks(frame)
    step = STEPS[min(int((now - t0) / args.seconds * len(STEPS)), len(STEPS) - 1)]
    if pts is not None:
        if now - last > 0.3:
            embs.append(fid.embed(frame, pts))
            last = now
        FID.draw_face(frame, pts, args.user, closed=False)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(frame, f"DANG KY '{args.user}' ({args.seconds - (now - t0):.0f}s): {step}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Da lay {len(embs)} mau" + ("" if pts is not None else "  -  KHONG THAY MAT"),
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
    cv2.imshow("Drowsiness Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        raise SystemExit("Da huy.")

if len(embs) < 8:
    raise SystemExit(f"Chi lay duoc {len(embs)} mau - can it nhat 8. Kiem tra anh sang/camera roi thu lai.")

# Canh bao neu khuon mat nay giong mot tai xe khac da dang ky
if gallery:
    mean_emb = np.mean(embs, 0); mean_emb /= np.linalg.norm(mean_emb)
    other, s = FID.FaceID.identify(mean_emb, gallery)
    if other:
        print(f"!! Chu y: khuon mat nay giong tai xe '{other}' da dang ky (diem {s:.2f}).")

db.add_faces(con, usr["id"], embs, replace=not args.add)
print(f"Da luu {len(embs)} mau khuon mat cho '{args.user}'.")

base = F.run_calibration(cap, tracker)
db.save_baseline(con, usr["id"], base)
print("Da hieu chinh xong. Chay: python app.py  (app se tu nhan ra tai xe)")
cap.release()
cv2.destroyAllWindows()

