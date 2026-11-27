"""Kiem thu & tinh TY LE CHINH XAC cua app theo kich ban (dung chung logic voi app.py).

Cach dung:
    python evaluate.py --user ten_ban                     # chay tat ca kich ban
    python evaluate.py --user ten_ban --only TC03,TC05    # chi chay vai kich ban
    python evaluate.py --user ten_ban --condition deo_kinh  # gan nhan dieu kien (kinh, toi...)
    python evaluate.py --summary                          # tong hop tat ca lan test da chay

Moi kich ban: 3 giay chuan bi (doc huong dan) -> ghi N giay. Lam theo chu tren man hinh.
Phim: s = bo qua kich ban,  r = lam lai kich ban,  q = dung va xuat bao cao.
Ket qua: thu muc eval_results/<thoi_gian>_<dieu_kien>/ gom frames.csv + report.md
"""
import argparse
import collections
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import db
import detector as D
import features as F

# ─── Kich ban test ───────────────────────────────────────────────────────────
# expect: trang thai dung; alarm: loai canh bao PHAI xuat hien (None = KHONG duoc co canh bao
# buon ngu); grace: so giay dau KHONG cham diem (thoi gian chuyen trang thai hop le).
SCENARIOS = [
    dict(id="TC01", name="Nhìn thẳng, mắt mở bình thường", sec=20, expect=D.ALERT, alarm=None, grace=1.0),
    dict(id="TC02", name="Chớp mắt bình thường, liên tục", sec=15, expect=D.ALERT, alarm=None, grace=1.0),
    dict(id="TC03", name="NHẮM MẮT hoàn toàn và giữ nguyên", sec=6, expect=D.DROWSY, alarm="drowsy", grace=1.5),
    dict(id="TC04", name="Lim dim (mắt hé rất nhỏ)", sec=8, expect=D.DROWSY, alarm="drowsy", grace=2.0),
    dict(id="TC05", name="Quay đầu sang TRÁI, mắt mở", sec=8, expect=D.AWAY, alarm=None, grace=1.5),
    dict(id="TC06", name="Quay đầu sang PHẢI, mắt mở", sec=8, expect=D.AWAY, alarm=None, grace=1.5),
    dict(id="TC07", name="Cúi nhìn xuống (như xem điện thoại), mắt mở", sec=8, expect=D.ALERT, alarm=None, grace=1.5),
    dict(id="TC08", name="NGÁP (há miệng to, giữ 3-4 giây)", sec=6, expect=D.YAWN, alarm=None, grace=1.5),
    dict(id="TC09", name="Nói chuyện / cười tự nhiên", sec=10, expect=D.ALERT, alarm=None, grace=1.0),
    dict(id="TC10", name="Gật gù + nhắm mắt (như ngủ gật)", sec=6, expect=D.DROWSY, alarm="drowsy", grace=2.0),
    dict(id="TC11", name="Rời khỏi khung hình (che camera/đứng dậy)", sec=6, expect=D.NO_FACE, alarm="no_face", grace=2.5),
]
NAMES = D.STATE_NAMES
PREP_SEC = 3.0
OUT_ROOT = "eval_results"

# ─── Font tieng Viet ─────────────────────────────────────────────────────────
_font = None
for _fp in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/tahoma.ttf"]:
    if os.path.exists(_fp):
        _font = (ImageFont.truetype(_fp, 26), ImageFont.truetype(_fp, 20))
        break
if _font is None:
    _font = (ImageFont.load_default(), ImageFont.load_default())


def put_lines(frame, lines):
    """lines: [(text, (B,G,R), big?)] ve tren nen den o goc tren."""
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    h = sum(36 if big else 28 for _, _, big in lines) + 12
    d.rectangle([0, 0, pil.width, h], fill=(0, 0, 0))
    y = 6
    for text, col, big in lines:
        d.text((12, y), text, font=_font[0 if big else 1], fill=(col[2], col[1], col[0]))
        y += 36 if big else 28
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ─── Bao cao ─────────────────────────────────────────────────────────────────
def _class_table(df):
    """Precision / recall / F1 theo tung trang thai tren cac frame duoc cham diem."""
    rows = []
    labels = sorted(set(df["expect"]) | set(df["pred"]))
    for lab in labels:
        tp = int(((df.expect == lab) & (df.pred == lab)).sum())
        fp = int(((df.expect != lab) & (df.pred == lab)).sum())
        fn = int(((df.expect == lab) & (df.pred != lab)).sum())
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * p * r / (p + r) if p + r and not np.isnan(p + r) else float("nan")
        rows.append(dict(trang_thai=NAMES.get(lab, lab), so_frame=int((df.expect == lab).sum()),
                         precision=p, recall=r, f1=f1))
    return pd.DataFrame(rows)


def _confusion(df):
    labels = sorted(set(df["expect"]) | set(df["pred"]))
    cm = pd.crosstab(df["expect"].map(NAMES), df["pred"].map(NAMES),
                     rownames=["thuc_te"], colnames=["du_doan"], dropna=False)
    order = [NAMES[l] for l in labels]
    return cm.reindex(index=[o for o in order if o in cm.index],
                      columns=[o for o in order if o in cm.columns], fill_value=0)


def scenario_table(df):
    rows = []
    for (run, sc_id), g in df.groupby(["run", "scenario"], sort=False):
        meta = next(s for s in SCENARIOS if s["id"] == sc_id)
        scored = g[g.scored == 1]
        acc = float((scored.expect == scored.pred).mean()) if len(scored) else float("nan")
        # Canh bao: suon len cua level >= 1
        lv = g["level"].to_numpy()
        rises = np.flatnonzero((lv[1:] >= 1) & (lv[:-1] == 0)) + 1
        if len(lv) and lv[0] >= 1:
            rises = np.r_[0, rises]
        et = g["etype"].to_numpy()
        drowsy_alarms = int(sum(et[i] == "drowsy" for i in rises))
        if meta["alarm"]:
            hit = g[(g.level >= 1) & (g.etype == meta["alarm"])]
            detected = len(hit) > 0
            latency = float(hit["t"].iloc[0]) if detected else float("nan")
            alarm_ok = detected
        else:
            detected, latency = drowsy_alarms > 0, float("nan")
            alarm_ok = drowsy_alarms == 0          # khong duoc bao buon ngu nham
        rows.append(dict(lan=run, ma=sc_id, kich_ban=meta["name"], ky_vong=NAMES[meta["expect"]],
                         do_chinh_xac_frame=acc,
                         canh_bao=("phai co: " + meta["alarm"]) if meta["alarm"] else "khong duoc co",
                         bao_nham_buon_ngu=0 if meta["alarm"] == "drowsy" else drowsy_alarms,
                         canh_bao_dung=bool(alarm_ok), do_tre_canh_bao_s=latency,
                         dat=bool(alarm_ok and (acc >= 0.8 if not np.isnan(acc) else False))))
    return pd.DataFrame(rows)


def write_report(df, out_dir, title, extra=""):
    scored = df[df.scored == 1]
    sc = scenario_table(df)
    overall = float((scored.expect == scored.pred).mean()) if len(scored) else float("nan")
    must = sc[sc.canh_bao.str.startswith("phai")]
    must_not = sc[~sc.canh_bao.str.startswith("phai")]
    det_rate = must.canh_bao_dung.mean() if len(must) else float("nan")
    fa_total = int(must_not.bao_nham_buon_ngu.sum())
    minutes = (df[df.scenario.isin(must_not.ma)].groupby(["run", "scenario"])["t"].max().sum()) / 60
    fa_rate = fa_total / minutes if minutes else float("nan")
    lat = must[must.ma.isin(["TC03", "TC04", "TC10"])].do_tre_canh_bao_s.mean()

    fmt = lambda v, pct=True: "-" if pd.isna(v) else (f"{v:.1%}" if pct else f"{v:.2f}")
    lines = [f"# {title}", "", extra, "",
             "## Tong ket", "",
             "| Chi so | Gia tri |", "|---|---|",
             f"| Do chinh xac theo frame (sau thoi gian chuyen) | **{fmt(overall)}** |",
             f"| Ty le phat hien canh bao dung (kich ban phai canh bao) | **{fmt(det_rate)}** |",
             f"| So lan bao NHAM buon ngu (kich ban khong buon ngu) | **{fa_total}** |",
             f"| Bao nham / phut | {fmt(fa_rate, False)} |",
             f"| Do tre canh bao buon ngu trung binh (giay) | {fmt(lat, False)} |",
             f"| So kich ban dat / tong | {int(sc.dat.sum())} / {len(sc)} |",
             f"| Tong so frame cham diem | {len(scored)} |", "",
             "## Theo tung kich ban", "",
             "| Lan | Ma | Kich ban | Ky vong | Chinh xac frame | Canh bao | Bao nham | Tre (s) | Dat |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in sc.iterrows():
        lines.append(f"| {r.lan} | {r.ma} | {r.kich_ban} | {r.ky_vong} | {fmt(r.do_chinh_xac_frame)} | "
                     f"{r.canh_bao} {'✅' if r.canh_bao_dung else '❌'} | {r.bao_nham_buon_ngu} | "
                     f"{fmt(r.do_tre_canh_bao_s, False)} | {'✅' if r.dat else '❌'} |")
    ct = _class_table(scored)
    lines += ["", "## Precision / Recall / F1 theo trang thai", "",
              "| Trang thai | So frame | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for _, r in ct.iterrows():
        lines.append(f"| {r.trang_thai} | {r.so_frame} | {fmt(r.precision)} | {fmt(r.recall)} | {fmt(r.f1)} |")
    cm = _confusion(scored)
    lines += ["", "## Ma tran nham lan (hang = thuc te, cot = app du doan)", "",
              "| thuc_te \\ du_doan | " + " | ".join(cm.columns) + " |",
              "|---" * (len(cm.columns) + 1) + "|"]
    for idx, row in cm.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(int(v)) for v in row.values) + " |")
    if "id_expect" in df and df["id_ok"].notna().any():
        ids = df[df["id_ok"].notna()]
        lines += ["", "## Nhan dien tai xe", "",
                  f"- Ky vong: **{ids['id_expect'].iloc[0]}**",
                  f"- So lan nhan dien: {len(ids)}, dung: {int(ids['id_ok'].sum())} "
                  f"(**{ids['id_ok'].mean():.1%}**)"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[4:16]))
    print("\nDa luu bao cao ->", path)
    return path


def summary():
    files = sorted(glob.glob(os.path.join(OUT_ROOT, "*", "frames.csv")))
    if not files:
        raise SystemExit("Chua co lan test nao trong eval_results/. Chay evaluate.py truoc.")
    dfs = []
    for i, fp in enumerate(files, 1):
        d = pd.read_csv(fp)
        d["run"] = f"{i}:{os.path.basename(os.path.dirname(fp))}"
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    conds = ", ".join(sorted(df["condition"].astype(str).unique()))
    write_report(df, os.path.join(OUT_ROOT, "_TONG_HOP"), "BAO CAO TONG HOP KIEM THU",
                 f"So lan test: {len(files)} | Dieu kien: {conds}")


# ─── Chay kiem thu ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="khach", help="ho so (baseline) dung khi test")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--only", default="", help="vd: TC03,TC05")
    ap.add_argument("--condition", default="binh_thuong",
                    help="nhan dieu kien test: binh_thuong, deo_kinh, thieu_sang, nguoc_sang...")
    ap.add_argument("--expect-name", default=None,
                    help="ten tai xe ky vong khi kiem tra nhan dien (mac dinh = --user; 'none' = nguoi la)")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--summary", action="store_true", help="chi tong hop cac lan test da chay")
    D.add_detector_args(ap)
    args = ap.parse_args()
    if args.summary:
        return summary()

    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    scenarios = [s for s in SCENARIOS if not only or s["id"] in only]

    con = db.connect()
    usr = db.get_user(con, args.user)
    mode, model = D.load_model(args.model_window, args.model_frame)
    cap = cv2.VideoCapture(args.camera)
    tracker = F.FaceTracker()
    base = None if args.recalibrate else db.get_baseline(usr)
    if base is None:
        base = F.run_calibration(cap, tracker)
        db.save_baseline(con, usr["id"], base)
    det = D.Detector(mode, model, args)

    # Nhan dien tai xe (neu da dang ky khuon mat)
    fid, gallery = None, {}
    try:
        import face_id as FID
        gallery = db.load_gallery(con)
        fid = FID.try_load() if gallery else None
    except Exception:
        pass
    id_expect = args.expect_name or (args.user if args.user in gallery else None)
    if args.expect_name and args.expect_name.lower() == "none":
        id_expect = "none"

    rows, quit_all = [], False
    i = 0
    while i < len(scenarios) and not quit_all:
        sc = scenarios[i]
        det.reset()
        rec, t0, redo, skip, last_id = [], time.time(), False, False, 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                quit_all = True
                break
            now = time.time()
            el = now - t0
            pts = tracker.landmarks(frame)
            id_ok = None
            if pts is None:
                r = det.no_face(now)
            else:
                raw = F.raw_features(pts, tracker.extras)
                r = det.update(F.normalize(raw, base), now)
                if fid and id_expect and el >= PREP_SEC and now - last_id > 1.0 and abs(raw[4]) < 0.2:
                    last_id = now
                    nm, _ = FID.FaceID.identify(fid.embed(frame, pts), gallery)
                    id_ok = int((nm or "none") == id_expect)
                for idx in F.L_EYE + F.R_EYE:
                    cv2.circle(frame, tuple(pts[idx][:2].astype(int)), 1,
                               (0, 0, 255) if r["closed"] else (0, 255, 0), -1)

            recording = el >= PREP_SEC
            t_rec = el - PREP_SEC
            if recording:
                rec.append(dict(run=1, condition=args.condition, scenario=sc["id"], t=round(t_rec, 3),
                                expect=sc["expect"], pred=r["status"], raw_pred=r["raw_status"],
                                level=r["level"], etype=r["etype"] if r["level"] else "",
                                scored=int(t_rec >= sc["grace"]), id_expect=id_expect, id_ok=id_ok))
                if t_rec >= sc["sec"]:
                    break

            ok_now = r["status"] == sc["expect"]
            head = (f"{sc['id']} ({i + 1}/{len(scenarios)}): {sc['name']}", (0, 255, 255), True)
            if not recording:
                info = (f"CHUẨN BỊ... bắt đầu sau {PREP_SEC - el:.0f}s", (200, 200, 200), False)
            else:
                info = (f"● ĐANG GHI {t_rec:4.1f}/{sc['sec']}s   (s=bỏ qua  r=làm lại  q=dừng)",
                        (0, 0, 255), False)
            stat = (f"App thấy: {NAMES[r['status']]}   |   Kỳ vọng: {NAMES[sc['expect']]}   "
                    f"{'✓' if ok_now else '✗'}   cảnh báo={r['level']}",
                    (80, 255, 80) if ok_now else (80, 80, 255), False)
            put_lines(frame, [head, info, stat])
            cv2.imshow("Drowsiness Monitor - KIEM THU", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                quit_all = True
                break
            if k == ord("s"):
                skip = True
                break
            if k == ord("r"):
                redo = True
                break
        if redo:
            continue
        if not skip and rec and rec[-1]["t"] >= sc["sec"] - 0.1:
            rows += rec
            acc = np.mean([x["expect"] == x["pred"] for x in rec if x["scored"]] or [np.nan])
            print(f"{sc['id']} xong: chinh xac frame = {acc:.1%}")
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    if not rows:
        raise SystemExit("Khong co du lieu test nao duoc ghi.")
    df = pd.DataFrame(rows)
    out_dir = os.path.join(OUT_ROOT, time.strftime("%Y%m%d_%H%M%S") + "_" + args.condition)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "frames.csv"), index=False)
    cfg = (f"Ho so: {args.user} | Che do: {mode}+MediaPipe | Dieu kien: {args.condition} | "
           f"smooth={args.smooth_frames} switch={args.switch_ratio} blink={args.blink_thresh} "
           f"yaw={args.yaw_thresh} warn={args.warn_after}s alarm={args.alarm_after}s")
    write_report(df, out_dir, "BAO CAO KIEM THU", cfg)


if __name__ == "__main__":
    main()
