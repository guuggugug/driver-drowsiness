# Driver Drowsiness Detection

## Cai dat
    pip install -r requirements.txt

## Cach 1: Chi dung webcam cua ban (nhanh, it du lieu)
    python collect_data.py --driver ten_ban   # 1) thu du lieu (phim 0/1/2, SPACE bat/tat ghi, q thoat)
    python train.py                           # 2) train -> model_frame.joblib (+ model_window.joblib neu du du lieu)
    python app.py --driver ten_ban            # 3) chay real-time

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
         python app.py --driver ten_ban

Ghi chu:
- ingest_nthu.py tu nhan dien 2 dang dataset pho bien: ANH da tach theo thu muc lop,
  hoac VIDEO goc kem file nhan .txt/.csv theo frame. Neu dataset cua ban co dang khac
  (vd file nhan CSV nhieu cot), gui minh vai dong dau cua file nhan de minh chinh lai parser.
- model_window.joblib chinh xac hon vi dung chuoi frame lien tuc (bat duoc toc do chop
  mat/PERCLOS); model_frame.joblib du doan tung anh doc lap, dung lam du phong khi chua
  co du lieu video/webcam lien tuc.
- Xem driver.db bang DB Browser for SQLite. Bang: drivers, sessions, samples, events.
