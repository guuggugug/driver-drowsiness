"""Xuat dataset ra CSV va train model tu SQLite.

Tao ra toi da 2 file model:
  model_frame.joblib  - hoc tren TUNG ANH rieng le (dung duoc voi du lieu NTHU dang anh,
                         khong can thu tu thoi gian). Dung lam phuong an du phong trong app.py.
  model_window.joblib - hoc tren CUA SO ~1.5s lien tuc (webcam collect_data.py, hoac video
                         NTHU goc). Chinh xac hon vi bat duoc toc do chop mat/PERCLOS, va
                         la model app.py se uu tien dung.
"""
import argparse
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GroupShuffleSplit

import db
import features as F

ap = argparse.ArgumentParser()
ap.add_argument("--out-frame", default="model_frame.joblib")
ap.add_argument("--out-window", default="model_window.joblib")
ap.add_argument("--stride", type=int, default=5, help="buoc truot cua so khi tao mau window")
args = ap.parse_args()

con = sqlite3.connect(db.DB_PATH)
df = pd.read_sql_query(
    "SELECT s.session_id, se.kind, d.name AS driver, s.t, s.ear_n, s.mar, s.dpitch, s.roll, s.label "
    "FROM samples s JOIN sessions se ON se.id=s.session_id JOIN drivers d ON d.id=se.driver_id "
    "ORDER BY s.session_id, s.id", con)
if df.empty:
    raise SystemExit("Chua co du lieu. Chay collect_data.py hoac ingest_nthu.py truoc.")
df.to_csv("dataset_frames.csv", index=False)
print("Tong so mau (frame):", len(df), df["label"].value_counts().to_dict())
print("Nguon du lieu theo loai phien:",
      df.assign(src=df["kind"].str.split(":").str[0]).groupby("src")["label"].count().to_dict())

# ---------- 1) MODEL FRAME: moi mau la 1 anh/frame doc lap ----------
Xf = df[["ear_n", "mar", "dpitch", "roll"]].to_numpy()
yf = df["label"].to_numpy()
gf = df["session_id"].to_numpy()
if len(set(yf)) >= 2:
    clf_f = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    sw_f = compute_sample_weight("balanced", yf)
    sw_f = (sw_f / sw_f.sum()).astype(np.float64)  # normalize tranh bug sklearn+numpy2
    if len(set(gf)) >= 3:
        tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(Xf, yf, gf))
        clf_f.fit(Xf[tr], yf[tr], sample_weight=sw_f[tr])
        print("\n=== Danh gia MODEL FRAME ===")
        print(classification_report(yf[te], clf_f.predict(Xf[te]), zero_division=0))
    clf_f.fit(Xf, yf, sample_weight=sw_f)
    joblib.dump({"mode": "frame", "model": clf_f, "features": ["ear_n", "mar", "dpitch", "roll"]}, args.out_frame)
    print("Da luu model frame ->", args.out_frame)
else:
    print("Bo qua model frame: can it nhat 2 nhan khac nhau.")

# ---------- 2) MODEL WINDOW: chi dung session co du khung lien tuc ----------
# Du lieu anh roi rac tu NTHU (kind bat dau bang 'nthu_img') khong co thu tu thoi gian
# that nen KHONG dung de tao cua so truot; chi dung session tu webcam hoac video goc.
win_df = df[~df["kind"].str.startswith("nthu_img")]
Xw, yw, gw = [], [], []
for sid, part in win_df.groupby("session_id"):
    arr = part[["ear_n", "mar", "dpitch", "roll"]].to_numpy()
    lab = part["label"].to_numpy()
    for i in range(0, len(arr) - F.WIN + 1, args.stride):
        seg = lab[i:i + F.WIN]
        top = np.bincount(seg).argmax()
        if (seg == top).mean() >= 0.8:
            Xw.append(F.window_features(arr[i:i + F.WIN]))
            yw.append(int(top))
            gw.append(sid)
Xw, yw, gw = np.array(Xw), np.array(yw), np.array(gw)

if len(Xw) == 0:
    print("\nBo qua model window: chua co session lien tuc du dai (>= "
          f"{F.WIN} frame lien tiep). Chay collect_data.py de thu them tu webcam, "
          "hoac nap video NTHU goc bang ingest_nthu.py.")
elif len(set(yw)) < 2:
    print("\nBo qua model window: can it nhat 2 nhan khac nhau trong du lieu lien tuc.")
else:
    pd.DataFrame(Xw, columns=F.FEATURE_NAMES).assign(label=yw, session_id=gw).to_csv(
        "dataset_windows.csv", index=False)
    print("\nSo cua so (window):", len(Xw), np.bincount(yw))
    clf_w = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    sw_w = compute_sample_weight("balanced", yw)
    sw_w = (sw_w / sw_w.sum()).astype(np.float64)  # normalize tranh bug sklearn+numpy2
    if len(set(gw)) >= 3:
        tr, te = next(GroupShuffleSplit(test_size=0.25, random_state=42).split(Xw, yw, gw))
        clf_w.fit(Xw[tr], yw[tr], sample_weight=sw_w[tr])
        print("\n=== Danh gia MODEL WINDOW ===")
        print(classification_report(yw[te], clf_w.predict(Xw[te]), zero_division=0))
    else:
        print("It phien ghi -> bo qua danh gia tach rieng (hay ghi >= 3 phien).")
    clf_w.fit(Xw, yw, sample_weight=sw_w)
    for n, imp in sorted(zip(F.FEATURE_NAMES, clf_w.feature_importances_), key=lambda z: -z[1])[:5]:
        print(f"  {n}: {imp:.3f}")
    joblib.dump({"mode": "window", "model": clf_w, "win": F.WIN}, args.out_window)
    print("Da luu model window ->", args.out_window)
