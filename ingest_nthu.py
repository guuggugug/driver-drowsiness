"""Nap dataset NTHU-DDD (hoac dataset tuong tu) tu Kaggle vao driver.db.

Ho tro 2 dang pho bien tren Kaggle:
  A) ANH da tach san, chia theo thu muc lop, vd:
     root/Drowsy/xxx.jpg, root/NonDrowsy/yyy.jpg, root/Yawning/zzz.jpg
  B) VIDEO goc + file nhan theo frame (.txt/.csv cung ten hoac trong cung thu muc)

LUON chay --dry-run truoc de xem script doan nhan/nhom nhu the nao, roi moi nap that.

Vi du:
    python ingest_nthu.py --root "D:/data/nthu-ddd" --dry-run
    python ingest_nthu.py --root "D:/data/nthu-ddd" --max-per-group 3000
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

import db
import features as F

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}

# Tu khoa doan nhan tu ten thu muc / ten file (khong phan biet hoa thuong).
# Neu dataset cua ban dat ten khac, sua truc tiep o day hoac dung --map.
# Thu tu kiem tra QUAN TRONG: cac tu phu dinh ("non_drowsy", "khong ngap"...) phai
# duoc kiem tra TRUOC tu goc ("drowsy", "yawn"), vi khong thi "nondrowsy" se bi nhan
# nham la "drowsy" do chua chuoi con "drowsy".
KEYWORDS = [
    (0, ["nondrowsy", "non_drowsy", "notdrowsy", "not_drowsy", "non-drowsy",
         "noyawn", "no_yawn", "no-yawn",
         "alert", "awake", "normal", "focused", "active", "nonsleepy", "non_sleepy"]),
    (2, ["yawn"]),
    (1, ["drowsy", "sleepy", "fatigue", "microsleep", "nod", "drows", "sleep"]),
]


def load_map(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k.lower(): int(v) for k, v in raw.items()}


def guess_label(custom_map, *names):
    text = " ".join(n for n in names if n).lower()
    if custom_map:
        for key, lab in custom_map.items():
            if key in text:
                return lab
    for lab, kws in KEYWORDS:
        if any(kw in text for kw in kws):
            return lab
    return None


def find_files(root, exts):
    for dirpath, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in exts:
                yield os.path.join(dirpath, fn)


def group_key(root, path, custom_map):
    """Doan ma nhom (subject/scenario) = cac thu muc con giua root va file, bo ten trung voi nhan."""
    rel = os.path.relpath(os.path.dirname(path), root)
    parts = [p for p in rel.split(os.sep) if p not in (".", "")]
    parts = [p for p in parts if guess_label(custom_map, p) is None]
    return "/".join(parts) if parts else "root"


def dry_run(root, custom_map):
    imgs = list(find_files(root, IMG_EXT))
    vids = list(find_files(root, VID_EXT))
    print(f"Tim thay {len(imgs)} anh, {len(vids)} video trong {root}\n")

    if imgs:
        counts = Counter()
        unknown = Counter()
        for p in imgs:
            folder = os.path.basename(os.path.dirname(p))
            lab = guess_label(custom_map, folder, os.path.basename(p))
            counts[(folder, F.LABELS.get(lab, "???"))] += 1
            if lab is None:
                unknown[folder] += 1
        print("-- ANH: so luong theo (thu_muc -> nhan doan duoc) --")
        for (folder, lab), n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {folder:35s} -> {lab:10s} : {n}")
        if unknown:
            print("\n!! KHONG doan duoc nhan cho cac thu muc sau (se BI BO QUA khi nap that):")
            for folder, n in unknown.items():
                print(f"  {folder}: {n} anh")
            print("   -> Sua KEYWORDS trong ingest_nthu.py, hoac tao file map.json vd:")
            print('      {"ten_thu_muc_1": 0, "ten_thu_muc_2": 1} roi chay voi --map map.json')
        groups = Counter(group_key(root, p, custom_map) for p in imgs)
        print(f"\nSo nhom (subject/scenario) doan duoc tu duong dan: {len(groups)}")
        for g, n in list(groups.items())[:12]:
            print(f"  {g}: {n} anh")

    if vids:
        print("\n-- VIDEO tim thay (toi da 10 dong dau) --")
        for v in vids[:10]:
            lp = find_label_file(v)
            print(f"  {v}  [nhan: {lp or 'KHONG THAY'}]")


def find_label_file(video_path):
    base, _ = os.path.splitext(video_path)
    for ext in (".txt", ".csv", ".lab"):
        if os.path.exists(base + ext):
            return base + ext
    d = os.path.dirname(video_path)
    for fn in os.listdir(d):
        if fn.lower().endswith((".txt", ".csv")) and "label" in fn.lower():
            return os.path.join(d, fn)
    return None


def read_frame_labels(path):
    vals = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().replace(",", " ")
            if not line:
                continue
            tok = line.split()[0]
            try:
                vals.append(int(float(tok)))
            except ValueError:
                continue
    return vals


def ingest_images(root, custom_map, max_per_group, stride):
    con = db.connect()
    tracker = F.FaceTracker()
    driver = db.get_driver(con, "nthu_dataset")

    # Group by (group_key, label) to ensure balanced sampling per class
    by_group_label = defaultdict(list)
    for p in find_files(root, IMG_EXT):
        folder = os.path.basename(os.path.dirname(p))
        lab = guess_label(custom_map, folder, os.path.basename(p))
        if lab is not None:
            gkey = group_key(root, p, custom_map)
            by_group_label[(gkey, lab)].append(p)

    # If everything collapsed to one subject group, split by label so
    # max_per_group applies equally to each class (avoids alphabetical bias).
    group_keys = set(k for k, _ in by_group_label)
    if len(group_keys) == 1:
        by_group = defaultdict(list)
        for (gkey, lab), paths in by_group_label.items():
            by_group[f"label_{lab}"].extend((p, lab) for p in paths)
    else:
        by_group = defaultdict(list)
        for (gkey, lab), paths in by_group_label.items():
            by_group[gkey].extend((p, lab) for p in paths)

    total_count = sum(len(v) for v in by_group.values())
    print(f"Se nap {total_count} anh trong {len(by_group)} nhom.")
    total_ok = total_bad = 0
    for gi, (gkey, items) in enumerate(by_group.items(), 1):
        if stride > 1:
            items = items[::stride]
        if max_per_group:
            items = items[:max_per_group]
        sid = db.new_session(con, driver["id"], f"nthu_img:{gkey}")
        raws, labs = [], []
        for p, lab in items:
            img = cv2.imread(p)
            if img is None:
                total_bad += 1
                continue
            pts = tracker.landmarks(img)
            if pts is None:
                total_bad += 1
                continue
            raws.append(F.raw_features(pts))
            labs.append(lab)
        if not raws:
            db.end_session(con, sid)
            print(f"[{gi}/{len(by_group)}] {gkey}: KHONG co anh hop le, bo qua")
            continue
        raws, labs = np.array(raws), np.array(labs)
        alert_mask = labs == 0
        base = F.calibrate_from(raws[alert_mask] if alert_mask.sum() >= 10 else raws)
        rows = []
        for k, (r, lab) in enumerate(zip(raws, labs)):
            n = F.normalize(r, base)
            rows.append((sid, time.time() + k * 1e-3, float(n[0]), float(n[1]), float(n[2]), float(n[3]), int(lab)))
        db.add_samples(con, rows)
        db.end_session(con, sid)
        total_ok += len(rows)
        c = Counter(labs.tolist())
        print(f"[{gi}/{len(by_group)}] {gkey}: {len(rows)} mau OK "
              f"(alert={c.get(0,0)} drowsy={c.get(1,0)} yawn={c.get(2,0)})")
    print(f"\nTONG: {total_ok} mau da luu, {total_bad} anh bo qua (khong thay mat / loi doc file).")


def ingest_videos(root, stride):
    con = db.connect()
    tracker = F.FaceTracker()
    driver = db.get_driver(con, "nthu_dataset")
    vids = list(find_files(root, VID_EXT))
    print(f"Tim thay {len(vids)} video.")
    total_ok = 0
    for vi, vp in enumerate(vids, 1):
        lp = find_label_file(vp)
        if not lp:
            print(f"[{vi}/{len(vids)}] {os.path.basename(vp)}: KHONG thay file nhan, bo qua")
            continue
        labels = read_frame_labels(lp)
        cap = cv2.VideoCapture(vp)
        raws, labs, i = [], [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0 and i < len(labels):
                pts = tracker.landmarks(frame)
                if pts is not None:
                    # Nhan goc NTHU thuong la nhi phan 0/1 (khong tach rieng ngap);
                    # neu file nhan cua ban co dang khac, cho minh biet de chinh lai.
                    raws.append(F.raw_features(pts))
                    labs.append(1 if labels[i] >= 1 else 0)
            i += 1
        cap.release()
        if not raws:
            print(f"[{vi}/{len(vids)}] {os.path.basename(vp)}: khong doc duoc frame hop le, bo qua")
            continue
        raws, labs = np.array(raws), np.array(labs)
        alert_mask = labs == 0
        base = F.calibrate_from(raws[alert_mask] if alert_mask.sum() >= 10 else raws)
        sid = db.new_session(con, driver["id"], f"nthu_vid:{os.path.relpath(vp, root)}")
        rows = []
        for k, (r, lab) in enumerate(zip(raws, labs)):
            n = F.normalize(r, base)
            rows.append((sid, time.time() + k * 1e-3, float(n[0]), float(n[1]), float(n[2]), float(n[3]), int(lab)))
        db.add_samples(con, rows)
        db.end_session(con, sid)
        total_ok += len(rows)
        print(f"[{vi}/{len(vids)}] {os.path.basename(vp)}: {len(rows)} frame, nhan={Counter(labs.tolist())}")
    print(f"\nTONG: {total_ok} frame da luu tu video.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Thu muc goc cua dataset da giai nen")
    ap.add_argument("--dry-run", action="store_true", help="Chi xem cau truc & nhan doan duoc, khong nap")
    ap.add_argument("--map", default=None, help="File JSON tu-khoa-thu-muc -> nhan (0/1/2), ghi de KEYWORDS")
    ap.add_argument("--max-per-group", type=int, default=4000, help="Gioi han so anh moi nhom (tranh qua lau)")
    ap.add_argument("--stride", type=int, default=1, help="Chi lay 1/N anh hoac frame (giam thoi gian xu ly)")
    args = ap.parse_args()

    custom_map = load_map(args.map)
    if args.dry_run:
        dry_run(args.root, custom_map)
    else:
        has_img = any(True for _ in find_files(args.root, IMG_EXT))
        has_vid = any(True for _ in find_files(args.root, VID_EXT))
        if has_img:
            ingest_images(args.root, custom_map, args.max_per_group, args.stride)
        if has_vid:
            ingest_videos(args.root, args.stride)
        if not has_img and not has_vid:
            print("Khong tim thay anh hoac video nao trong --root. Kiem tra lai duong dan.")
