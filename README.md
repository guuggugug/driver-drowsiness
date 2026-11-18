# Drowsiness Detection

## Cai dat
    pip install -r requirements.txt

## Nhan dien tai xe & ve khung mat/mat
    python enroll.py --user ten_tai_xe   # dang ky khuon mat (~10s) + hieu chinh mat (8s)
    python app.py                        # tu nhan dien tai xe (hoac --user ten_ban de chi dinh)

- App ve khung khuon mat kem ten tai xe (hoac `Nguoi la`), va vien + khung quanh 2 mat
  (xanh = mo, do = nham).
- Neu co nhieu tai xe da dang ky, app tu dong doi ho so khi thay nguoi khac ngoi lai.
- Them `--no-faceid` neu muon tat nhan dien khuon mat.

## Cach 1: Chi dung webcam cua ban (nhanh, it du lieu)
    python collect_data.py --user ten_ban   # 1) thu du lieu (phim 0/1/2, SPACE bat/tat ghi, q thoat)
    python train.py                           # 2) train -> model_frame.joblib (+ model_window.joblib neu du du lieu)
    python app.py --user ten_ban            # 3) chay real-time

### Cach app quyet dinh (ket hop 2 model)
1. Model CO SAN cua MediaPipe (khong can train): diem nham mat `eyeBlink` 0..1,
   diem ha mieng `jawOpen`, goc quay dau `yaw` that (do).
2. Model RIENG cua ban (train.py) neu co; chua co thi dung luat dua tren (1).
- Model rieng bao buon ngu nhung MediaPipe thay mat MO ro -> KHONG bao (`--open-veto 0.25`).
- MediaPipe thay mat nham > 1s (nhin thang, khong ngap) -> bao buon ngu (`--blink-thresh 0.5`).
- Quay mat (`|yaw|` > 0.25 ~ 25 do) -> khong bao buon ngu, chi nhac nhin ve phia truoc sau 3s.
- train.py tu bo qua du lieu cu (thu truoc khi co diem nham mat) va canh bao neu nhan sai.

### Meo thu du lieu de model KHONG bao nham khi quay mat
- Ghi nhieu phien (moi lan SPACE bat/tat = 1 phien, nen >= 3 phien moi nhan).
- Nhan 0 (tinh tao): mat MO, nhung hay quay dau trai/phai, nhin guong, nhin xuong
  bang dong ho, nhin len -> model hoc duoc "quay mat di != buon ngu".
- Nhan 1 (buon ngu): NHAM MAT that su / lim dim / gat dau (dung ghi khi mat dang mo!).
- Nhan 2 (ngap): ngap that hoac gia ngap.
- app.py: `--yaw-thresh 0.25` (nguong coi la quay mat, xem so `yaw=` tren man hinh),
  `--distract-after 3` (giay quay mat lien tuc -> nhac nhin ve phia truoc, 0 = tat).

## Cach 2: Dung them dataset NTHU-DDD tu Kaggle (chinh xac hon, nhieu nguoi hon)
    1) Tai va giai nen dataset Kaggle ve may ban.
    2) Xem thu cau truc va kiem tra nhan doan duoc (KHONG nap gi ca):
         python ingest_nthu.py --root "duong/dan/toi/dataset" --dry-run
       - Neu co thu muc bi "KHONG doan duoc nhan", tao file map.json, vi du:
           {"ten_thu_muc_cua_ban_1": 0, "ten_thu_muc_cua_ban_2": 1}
         (0=tinh tao, 1=buon ngu, 2=ngap) roi chay lai dry-run voi --map map.json de kiem tra.
    3) Nap that vao database:
         python ingest_nthu.py --root "duong/dan/toi/dataset" --map map.json
       (them --stride 3 neu dataset qua lon, --max-per-group de gioi han so anh/nhom)
    4) Train lai (gop ca du lieu NTHU + webcam neu co):
         python train.py
    5) Chay real-time (uu tien tu dong model_window, roi den model_frame):
         python app.py --user ten_ban

Ghi chu:
- ingest_nthu.py tu nhan dien 2 dang dataset pho bien: ANH da tach theo thu muc lop,
  hoac VIDEO goc kem file nhan .txt/.csv theo frame. Neu dataset cua ban co dang khac
  (vd file nhan CSV nhieu cot), gui minh vai dong dau cua file nhan de minh chinh lai parser.
- model_window.joblib chinh xac hon vi dung chuoi frame lien tuc (bat duoc toc do chop
  mat/PERCLOS); model_frame.joblib du doan tung anh doc lap, dung lam du phong khi chua
  co du lieu video/webcam lien tuc.
- Xem monitor.db bang DB Browser for SQLite. Bang: users, sessions, samples, events.
