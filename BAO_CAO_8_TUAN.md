# BÁO CÁO CHI TIẾT TIẾN ĐỘ THỰC HIỆN ĐỒ ÁN (8 TUẦN)

**Trường:** Đại học CMC — Khoa Công nghệ Thông tin (Khóa K2)  
**Tên đề tài:** Hệ thống Giám sát Buồn ngủ và Nhận diện Người dùng Thời gian thực qua Camera  
**Thời gian thực hiện:** 8 tuần (28/09/2026 – 22/11/2026, hoàn thiện & kiểm thử đến 24/11/2026)  

## 1. Thông tin nhóm sinh viên thực hiện

| STT | Họ và tên SV | MSV | Ngày sinh | Lớp | Email |
|---|---|---|---|---|---|
| 1 | Vũ Đức Duy Hoàng | BIT230166 | 26/10/2005 | 23ITGM | BIT230166@st.cmcu.edu.vn |
| 2 | Lê Anh Tú | BIT230423 | 05/03/2005 | 23IT4 | BIT230423@st.cmcu.edu.vn |
| 3 | Đỗ Lan Anh | BIT230020 | 26/08/2005 | 23IT4 | BIT230020@st.cmcu.edu.vn |
| 4 | Nguyễn Ngọc Ánh | BIT230044 | 26/01/2005 | 23IT4 | BIT230044@st.cmcu.edu.vn |

---

## 2. Bảng tổng hợp tiến độ theo Mẫu 07

| Tuần | Công việc thực hiện | Ngày kiểm tra | Kết quả đạt được | Nhận xét của GVHD |
|---|---|---|---|---|
| **1** *(28/09 – 04/10/2026)* | Nộp đề cương ĐA/KLTN; khảo sát bài toán phát hiện buồn ngủ và lựa chọn công nghệ | | Hoàn thành đề cương chi tiết, chốt kiến trúc hệ thống và bộ thư viện (`requirements.txt`) | |
| **2** *(05/10 – 11/10/2026)* | Viết nội dung từng phần của ĐA/KLTN – SV gặp trao đổi với GVHD; thiết kế CSDL & luồng đặc trưng | | Hoàn thành các phần lý thuyết đầu, thiết kế lược đồ `monitor.db` và công thức EAR/MAR | |
| **3** *(12/10 – 18/10/2026)* | SV gửi và gặp GVHD chỉnh sửa chi tiết nội dung cập nhật lần 1; triển khai module trích xuất đặc trưng & thu dữ liệu | | Hoàn thành chỉnh sửa báo cáo lần 1; xây dựng `db.py`, `features.py`, `collect_data.py`, `ingest_nthu.py` | |
| **4** *(19/10 – 25/10/2026)* | SV gửi và gặp GVHD chỉnh sửa chi tiết nội dung cập nhật lần 2; hoàn thiện codebase v1 và demo đầu tiên | **27/10/2026** | Demo đầu có camera và codebase (`Initial commit` 22/10/2026 với `train.py`, `app.py`) | Bỏ các phần liên quan đến ô tô, sửa đổi biểu mẫu 2 và 4 |
| **5** *(26/10 – 01/11/2026)* | SV gửi và gặp GVHD chỉnh sửa chi tiết nội dung cập nhật lần 3; sửa đổi biểu mẫu 2, 4 theo góp ý | | Hoàn thiện chỉnh sửa biểu mẫu 2 và 4; chốt phương án loại bỏ nội dung ô tô trong báo cáo & mã nguồn | |
| **6** *(02/11 – 08/11/2026)* | SV gửi bản hoàn thiện cho GVHD; loại bỏ hoàn toàn các phần liên quan đến ô tô trong code & CSDL | | Đổi `drivers` → `users` (`_migrate()` trong `db.py`), tổng quát hóa toàn bộ mã nguồn (`21947ab`, 07/11/2026) | |
| **7** *(09/11 – 15/11/2026)* | Nâng cấp thuật toán phát hiện buồn ngủ khi quay đầu & kết hợp MediaPipe Blendshapes | | Bổ sung `yaw` 3D, `eyeBlink`/`jawOpen`, `ExtraTreesClassifier`, `run_app.bat` (`e963e43`, 13/11/2026) | |
| **8** *(16/11 – 22/11/2026)* | Tích hợp nhận diện khuôn mặt (OpenCV SFace), tối ưu ổn định giao diện, viết hướng dẫn & kiểm thử | | Hoàn thành `face_id.py`, `enroll.py`, bảng `faces`, `HUONG_DAN.md`, `status_vote`, hysteresis `yaw` | |

---

## 3. Chi tiết công việc thực hiện từng tuần

### Tuần 1 (28/09/2026 – 04/10/2026): Lập đề cương & Khảo sát công nghệ
- **Mục tiêu:** Xác định rõ phạm vi đề tài, bài toán cần giải quyết và bộ công nghệ triển khai.
- **Công việc chi tiết:**
  - Nộp đề cương ĐA/KLTN cho Giảng viên hướng dẫn (GVHD).
  - Nghiên cứu các phương pháp phát hiện buồn ngủ dựa trên thị giác máy tính: tỷ lệ khung mắt **EAR** (Eye Aspect Ratio), tỷ lệ mở miệng **MAR** (Mouth Aspect Ratio), tỷ lệ nhắm mắt theo thời gian (**PERCLOS**) và góc tư thế đầu (**pitch**, **roll**).
  - Khảo sát và lựa chọn ngăn xếp công nghệ gọn nhẹ, chạy thời gian thực trên CPU bằng webcam thông thường: **Python**, **OpenCV**, **MediaPipe** (Face Mesh / Face Landmarker), **scikit-learn**, **SQLite**, **Pygame** (âm thanh bíp), **Pillow** (vẽ chữ tiếng Việt Unicode), **pyttsx3** (đọc cảnh báo giọng nói tiếng Việt).
- **Kết quả đạt được:** Chốt đề cương ĐA/KLTN, cấu trúc thư mục dự án và danh sách thư viện trong `requirements.txt`.

---

### Tuần 2 (05/10/2026 – 11/10/2026): Thiết kế hệ thống & Viết nội dung các phần lý thuyết
- **Mục tiêu:** Thiết kế kiến trúc phần mềm mô-đun hóa và hoàn thiện cơ sở lý thuyết trong báo cáo.
- **Công việc chi tiết:**
  - Viết nội dung từng phần của ĐA/KLTN; gặp gỡ, trao đổi trực tiếp với GVHD để được giải đáp thắc mắc và thống nhất bố cục báo cáo.
  - Thiết kế cơ sở dữ liệu SQLite (`monitor.db` trong `db.py`) gồm 4 bảng chính:
    - Bảng hồ sơ người dùng (lưu ngưỡng hiệu chỉnh `base_ear`, `base_pitch` riêng từng người).
    - `sessions`: quản lý từng phiên thu dữ liệu hoặc phiên giám sát (`kind`, `started`, `ended`).
    - `samples`: lưu chuỗi đặc trưng đã chuẩn hóa theo từng frame (`ear_n`, `mar`, `dpitch`, `roll`, `label`).
    - `events`: ghi nhận lịch sử sự kiện cảnh báo (`type`, `level`, `score`, `t`).
  - Thiết kế luồng xử lý tách bạch 5 khối: trích xuất đặc trưng (`features.py`), thu dữ liệu (`collect_data.py`), nạp dữ liệu ngoài (`ingest_nthu.py`), huấn luyện (`train.py`) và chạy giám sát (`app.py`).
- **Kết quả đạt được:** Hoàn thành bản thảo các phần đầu của báo cáo và bản thiết kế kiến trúc hệ thống + CSDL.

---

### Tuần 3 (12/10/2026 – 18/10/2026): Chỉnh sửa nội dung lần 1 & Xây dựng module đặc trưng, thu dữ liệu
- **Mục tiêu:** Cập nhật báo cáo theo góp ý lần 1 của GVHD và hiện thực hóa khối trích xuất đặc trưng + thu thập dữ liệu.
- **Công việc chi tiết:**
  - Gửi báo cáo và gặp GVHD chỉnh sửa chi tiết nội dung cập nhật lần 1.
  - Xây dựng `db.py` khởi tạo lược đồ SQLite (`monitor.db`) và các hàm quản lý phiên, mẫu dữ liệu, sự kiện cảnh báo.
  - Xây dựng `features.py`: trích xuất 468 landmark khuôn mặt bằng MediaPipe (`FaceTracker`), tính `EAR`, `MAR`, `pitch`, `roll`; viết hàm hiệu chỉnh cá nhân hóa 8 giây (`run_calibration`) để chuẩn hóa đặc trưng theo kích thước mắt và tư thế ngồi của từng người.
  - Xây dựng `collect_data.py` cho phép ghi dữ liệu webcam theo phiên với 3 nhãn (`0`: tỉnh táo, `1`: buồn ngủ, `2`: ngáp) và `ingest_nthu.py` hỗ trợ đọc thêm bộ dữ liệu ảnh/video NTHU-DDD.
- **Kết quả đạt được:** Hoàn thành chỉnh sửa báo cáo lần 1; hoàn thiện các module nền tảng `db.py`, `features.py`, `collect_data.py`, `ingest_nthu.py`.

---

### Tuần 4 (19/10/2026 – 25/10/2026, kiểm tra ngày 27/10/2026): Hoàn thiện Codebase v1 & Demo đầu tiên với Camera
- **Mục tiêu:** Hoàn thiện pipeline huấn luyện/giám sát và trình bày bản chạy thử trực tiếp cho GVHD kiểm tra tiến độ.
- **Công việc chi tiết:**
  - Gửi báo cáo và gặp GVHD chỉnh sửa chi tiết nội dung cập nhật lần 2.
  - Xây dựng `train.py` huấn luyện 2 mức mô hình: `model_frame.joblib` (dự đoán từng frame) và `model_window.joblib` (trượt cửa sổ `WIN = 45` frame ~1.5 giây để bắt nhịp chớp mắt/PERCLOS).
  - Xây dựng `app.py` chạy giám sát thời gian thực với 2 mức cảnh báo (nhắc nhở nhẹ sau `1.0s`, báo động mạnh sau `2.0s`), phát âm thanh bíp và giọng nói tiếng Việt; hoàn thiện phiên bản codebase đầu tiên (`Initial commit` `58c3097` ngày 22/10/2026).
  - Thực hiện buổi demo trực tiếp trên camera vào ngày **27/10/2026**: chạy hiệu chỉnh mắt, kiểm tra phát hiện nhắm mắt kéo dài, phát hiện ngáp, cảnh báo âm thanh và lưu log vào `monitor.db`.
- **Kết quả đạt được:** Demo đầu chạy thông suốt có camera và codebase.
- **Nhận xét của GVHD (27/10/2026):** *"Bỏ các phần liên quan đến ô tô, sửa đổi biểu mẫu 2 và 4"*.

---

### Tuần 5 (26/10/2026 – 01/11/2026): Chỉnh sửa biểu mẫu 2, 4 & Cập nhật nội dung lần 3
- **Mục tiêu:** Điều chỉnh tài liệu, biểu mẫu theo đúng góp ý của GVHD và lập kế hoạch chuẩn hóa phạm vi mã nguồn.
- **Công việc chi tiết:**
  - Gửi và gặp GVHD chỉnh sửa chi tiết nội dung cập nhật lần 3.
  - Thực hiện sửa đổi **biểu mẫu 2 và biểu mẫu 4** theo đúng yêu cầu của GVHD.
  - Rà soát toàn bộ nội dung báo cáo và mã nguồn để xác định các thuật ngữ, biến, bảng CSDL gắn với ngữ cảnh ô tô/lái xe cần loại bỏ hoặc tổng quát hóa.
- **Kết quả đạt được:** Hoàn thiện chỉnh sửa biểu mẫu 2, 4 và cập nhật báo cáo lần 3; chốt danh sách hạng mục cần refactor trong code và CSDL.

---

### Tuần 6 (02/11/2026 – 08/11/2026): Gửi bản hoàn thiện báo cáo & Tổng quát hóa Codebase
- **Mục tiêu:** Gửi bản hoàn thiện ĐA/KLTN cho GVHD và loại bỏ hoàn toàn các phần liên quan đến ô tô trong mã nguồn.
- **Công việc chi tiết:**
  - SV gửi bản hoàn thiện cho GVHD theo đúng tiến độ Mẫu 07.
  - Triển khai loại bỏ toàn bộ phần liên quan đến ô tô trong mã nguồn và CSDL (commit `21947ab` ngày 07/11/2026):
    - Đổi tên bảng `drivers` thành `users` và cột `driver_id` thành `user_id` trong `db.py` (viết hàm `_migrate()` tự động nâng cấp CSDL cũ không làm mất dữ liệu).
    - Cập nhật lại câu chữ, hướng dẫn và tên biến trong `app.py`, `collect_data.py`, `ingest_nthu.py`, `train.py`, `features.py`, `README.md`.
    - Loại bỏ file `dataset_frames.csv` dung lượng lớn khỏi Git để repo gọn nhẹ.
- **Kết quả đạt được:** Hoàn thành gửi bản hoàn thiện cho GVHD; toàn bộ codebase và CSDL đã được tổng quát hóa, không còn nội dung gắn với ô tô.

---

### Tuần 7 (09/11/2026 – 15/11/2026): Nâng cấp nhận diện khi quay đầu & Kết hợp 2 mô hình
- **Mục tiêu:** Khắc phục triệt để tình trạng báo nhầm buồn ngủ khi người dùng mở mắt hoặc quay đầu sang bên.
- **Công việc chi tiết:**
  - Phân tích nguyên nhân báo nhầm: khi quay đầu, một bên mắt bị khuất làm `EAR` giảm giả; đồng thời ngưỡng `EAR_n < 0.75` dễ trùng với người có mắt tự nhiên nhỏ.
  - Nâng cấp `features.py`:
    - Sử dụng tọa độ 3D (`x, y, z`) của MediaPipe để tính `EAR` ổn định hơn khi nghiêng đầu, và ưu tiên mắt gần camera khi quay ngang.
    - Bật `output_face_blendshapes=True` và `output_facial_transformation_matrixes=True` của MediaPipe Tasks để lấy trực tiếp điểm nhắm mắt **`eyeBlinkLeft/Right`** (`0..1`), điểm há miệng **`jawOpen`** (`0..1`) và góc quay ngang **`yaw`** (độ).
    - Mở rộng `window_features()` lên 18 đặc trưng (thêm `yaw_abs_mean`, `yaw_abs_max`, `away_frac`, `blink_mean`, `blink_max`, `blink_closed_frac`, `jaw_max`, `jaw_frac`).
  - Nâng cấp `app.py` theo cơ chế **kết hợp 2 mô hình**:
    - Nếu mô hình tự train dự đoán buồn ngủ nhưng MediaPipe thấy mắt đang mở rõ (`blink_avg < 0.25`) thì bác bỏ cảnh báo sai.
    - Nếu quay mặt quá `25°` (`|yaw| > 0.25`) thì không tính là buồn ngủ mà chuyển sang trạng thái *"Đang nhìn chỗ khác"* (nhắc sau 3 giây).
  - Nâng cấp `train.py`: chuyển sang `ExtraTreesClassifier` (khắc phục lỗi tương thích trọng số của `RandomForestClassifier` trên scikit-learn/NumPy mới), tăng trọng số gấp đôi cho mẫu khó *"tỉnh táo nhưng quay mặt"*, tự bỏ qua mẫu cũ thiếu `blink` và cảnh báo nếu dữ liệu gán nhãn sai.
  - Tạo `run_app.bat` hỗ trợ chạy nhanh bằng double-click trên Windows (commit `e963e43` ngày 13/11/2026).
- **Kết quả đạt được:** Giảm rõ rệt cảnh báo giả khi mở mắt và khi quay đầu; hệ thống chạy chính xác ngay cả khi chưa tự train mô hình riêng.

---

### Tuần 8 (16/11/2026 – 22/11/2026, hoàn thiện đến 24/11/2026): Nhận diện khuôn mặt, Ổn định giao diện & Hoàn thiện báo cáo
- **Mục tiêu:** Tích hợp nhận diện người dùng, xử lý triệt để hiện tượng nhấp nháy chữ trạng thái, kiểm thử toàn diện và chốt báo cáo 8 tuần.
- **Công việc chi tiết:**
  - Xây dựng module `face_id.py` dùng mô hình **OpenCV SFace** (`models/face_recognition_sface_2021dec.onnx`): căn chỉnh khuôn mặt từ 5 điểm landmark MediaPipe, trích xuất vector đặc trưng 128 chiều và so khớp danh tính theo độ tương đồng cosine (ngưỡng `0.363`).
  - Viết hàm `draw_face()` vẽ **khung bao khuôn mặt** kèm tên người dùng (hoặc `Nguoi la`), **viền đa giác + khung chữ nhật quanh 2 mắt** và tâm mống mắt (màu **xanh lá** khi mắt mở, màu **đỏ** khi mắt nhắm).
  - Mở rộng `db.py` thêm bảng `faces` (`add_faces()`, `load_gallery()`) và tạo `enroll.py` cho phép đăng ký khuôn mặt (~10 giây xoay các góc mặt + 8 giây hiệu chỉnh mắt); tích hợp tự nhận diện & chuyển hồ sơ vào `app.py` và viết `HUONG_DAN.md` (commit `1c2c448` ngày 18/11/2026).
  - Khắc phục lỗi khi mở mắt (viền mắt vẫn xanh) nhưng chữ trạng thái bị nhảy liên tục trong `app.py`:
    - Thêm **bộ lọc bỏ phiếu 8 frame (`status_vote`)** trước khi hiển thị `state_str`, loại bỏ dao động tức thời từng frame.
    - Thêm **vùng trễ (hysteresis)** cho trạng thái quay đầu (`away_state`: vào ở `yaw_thresh`, thoát ở `0.8 * yaw_thresh`).
    - Đồng bộ điều kiện gật đầu trong `rule_predict()` với ngưỡng nhắm mắt `blink_thresh` để tránh báo buồn ngủ khi mắt đang mở viền xanh.
    - Làm mượt chỉ số hiển thị (`disp_n`) bằng trung bình trượt và dùng thống nhất một tên cửa sổ `"Drowsiness Monitor"`.
  - Tổng hợp toàn bộ quá trình 8 tuần thành báo cáo chi tiết và bảng tiến độ theo Mẫu 07.
- **Kết quả đạt được:** Hệ thống nhận diện đúng người dùng, giao diện hiển thị khung mặt/mắt trực quan và chữ trạng thái ổn định; toàn bộ mã nguồn, tài liệu hướng dẫn và báo cáo 8 tuần đã hoàn chỉnh.
