# KỊCH BẢN KIỂM THỬ & CÁCH TÍNH TỈ LỆ CHÍNH XÁC

Tài liệu này mô tả các trường hợp test của hệ thống và cách dùng công cụ `evaluate.py` để tự động tính tỉ lệ chính xác.

---

## 1. Chạy kiểm thử tự động

Mở PowerShell tại thư mục dự án rồi chạy:

```powershell
.venv\Scripts\python.exe evaluate.py --user ten_ban
```

- Mỗi kịch bản có **3 giây chuẩn bị** (đọc hướng dẫn trên màn hình), sau đó app **ghi lại** trong N giây.
- Màn hình hiện: kịch bản đang test, trạng thái app nhận ra, trạng thái kỳ vọng, và dấu ✓ hoặc ✗.
- Phím: `s` bỏ qua kịch bản, `r` làm lại kịch bản, `q` dừng và xuất báo cáo.
- Kết quả lưu trong `eval_results/<thời_gian>_<điều_kiện>/`:
  - `frames.csv`: dữ liệu từng frame.
  - `report.md`: báo cáo tỉ lệ chính xác.

**Các lệnh hay dùng:**

| Mục đích | Lệnh |
|---|---|
| Chỉ chạy vài kịch bản | `.venv\Scripts\python.exe evaluate.py --user ten_ban --only TC03,TC05` |
| Ghi nhãn điều kiện test | `.venv\Scripts\python.exe evaluate.py --user ten_ban --condition deo_kinh` |
| Kiểm tra với người lạ (nhận diện) | `.venv\Scripts\python.exe evaluate.py --user khach --expect-name none` |
| Tổng hợp tất cả các lần test | `.venv\Scripts\python.exe evaluate.py --summary` (ra `eval_results/_TONG_HOP/report.md`) |
| Thử độ mượt khác | thêm `--smooth-frames 45` (mặc định 30) |

---

## 2. Danh sách kịch bản test (bộ quyết định buồn ngủ)

**Thời gian chuyển** là số giây đầu không chấm điểm, vì app cần một khoảng ngắn để đổi trạng thái (theo thiết kế: nhắm mắt hơn 1 giây mới cảnh báo).

| Mã | Kịch bản | Thời lượng | Kết quả mong đợi | Cảnh báo | Thời gian chuyển |
|---|---|---|---|---|---|
| TC01 | Nhìn thẳng, mắt mở bình thường | 20s | Tỉnh táo | **Không** được báo buồn ngủ | 1.0s |
| TC02 | Chớp mắt bình thường, liên tục | 15s | Tỉnh táo | **Không** được báo buồn ngủ | 1.0s |
| TC03 | Nhắm mắt hoàn toàn và giữ nguyên | 6s | Buồn ngủ | **Phải** báo buồn ngủ | 1.5s |
| TC04 | Lim dim (mắt hé rất nhỏ) | 8s | Buồn ngủ | **Phải** báo buồn ngủ | 2.0s |
| TC05 | Quay đầu sang trái, mắt mở | 8s | Nhìn chỗ khác | **Không** được báo buồn ngủ | 1.5s |
| TC06 | Quay đầu sang phải, mắt mở | 8s | Nhìn chỗ khác | **Không** được báo buồn ngủ | 1.5s |
| TC07 | Cúi nhìn xuống (như xem điện thoại), mắt mở | 8s | Tỉnh táo | **Không** được báo buồn ngủ | 1.5s |
| TC08 | Ngáp (há miệng to, giữ 3–4 giây) | 6s | Ngáp | **Không** được báo buồn ngủ | 1.5s |
| TC09 | Nói chuyện / cười tự nhiên | 10s | Tỉnh táo | **Không** được báo buồn ngủ | 1.0s |
| TC10 | Gật gù + nhắm mắt (như ngủ gật) | 6s | Buồn ngủ | **Phải** báo buồn ngủ | 2.0s |
| TC11 | Rời khỏi khung hình (che camera / đứng dậy) | 6s | Không thấy mặt | **Phải** báo không thấy mặt | 2.5s |

**Một kịch bản được tính là ĐẠT khi:**
- Độ chính xác theo frame ≥ 80%.
- Cảnh báo đúng yêu cầu: kịch bản "phải báo" thì có báo; kịch bản "không được báo" thì không có lần báo buồn ngủ nào.

---

## 3. Chạy lại bộ test trong nhiều điều kiện

Nên chạy cả bộ test ở mỗi điều kiện dưới đây, dùng `--condition` để ghi nhãn, rồi chạy `--summary` để tổng hợp:

| Mã | Điều kiện | Tham số |
|---|---|---|
| DK1 | Ánh sáng phòng bình thường | `--condition binh_thuong` |
| DK2 | Đeo kính cận / kính râm | `--condition deo_kinh` |
| DK3 | Thiếu sáng (tắt bớt đèn) | `--condition thieu_sang` |
| DK4 | Ngược sáng (cửa sổ phía sau) | `--condition nguoc_sang` |
| DK5 | Ngồi xa camera khoảng 1 m | `--condition xa_camera` |
| DK6 | Người khác (mỗi thành viên nhóm test 1 lần) | `--user ten_nguoi_do --condition nguoi_2` |

---

## 4. Kiểm thử nhận diện tài xế

`evaluate.py` tự đo nhận diện khoảng mỗi giây (khi mặt nhìn gần thẳng) nếu đã có người đăng ký khuôn mặt.

| Mã | Trường hợp | Cách chạy | Kết quả mong đợi |
|---|---|---|---|
| ID01 | Tài xế đã đăng ký ngồi trước camera | `--user ten_ban` | Nhận đúng tên |
| ID02 | Người lạ (chưa đăng ký) | `--user khach --expect-name none` | `Nguoi la` |
| ID03 | Tài xế đã đăng ký đeo kính / đội mũ | `--user ten_ban --condition deo_kinh` | Nhận đúng tên |
| ID04 | Hai tài xế đổi chỗ khi app đang chạy | Chạy `app.py`, đổi người ngồi | App tự đổi hồ sơ sau khoảng 3 giây |

Tỉ lệ nhận diện đúng in ở mục **"Nhan dien tai xe"** trong `report.md`. ID04 thì quan sát bằng mắt trên `app.py`.

---

## 5. Ý nghĩa các chỉ số trong báo cáo

| Chỉ số | Cách tính | Ý nghĩa |
|---|---|---|
| **Độ chính xác theo frame** | Số frame app đoán đúng trạng thái / tổng số frame được chấm (sau thời gian chuyển) | App hiển thị đúng bao nhiêu phần trăm thời gian |
| **Precision** (theo trạng thái) | Đúng / tổng số lần app nói là trạng thái đó | App báo "buồn ngủ" thì đáng tin đến mức nào |
| **Recall** (theo trạng thái) | Đúng / tổng số frame thực sự là trạng thái đó | Buồn ngủ thật thì app bắt được bao nhiêu |
| **F1** | 2 × P × R / (P + R) | Cân bằng giữa Precision và Recall |
| **Tỉ lệ phát hiện cảnh báo** | Số kịch bản "phải báo" có báo / tổng số kịch bản "phải báo" | Độ nhạy của cảnh báo |
| **Số lần báo nhầm** | Số lần bật cảnh báo buồn ngủ trong các kịch bản không buồn ngủ | Càng thấp càng tốt |
| **Báo nhầm / phút** | Số lần báo nhầm / tổng số phút test không buồn ngủ | So sánh giữa các cấu hình |
| **Độ trễ cảnh báo** | Thời điểm cảnh báo bật đầu tiên tính từ lúc bắt đầu ghi (TC03, TC04, TC10) | Mong đợi khoảng 1–1.5 giây |
| **Ma trận nhầm lẫn** | Hàng = thực tế, cột = app đoán | Cho biết app hay nhầm trạng thái nào với trạng thái nào |

---

## 6. Gợi ý chỉnh khi kết quả chưa tốt

| Hiện tượng trong báo cáo | Cách chỉnh |
|---|---|
| Chữ vẫn nhảy nhiều | Tăng `--smooth-frames 45` hoặc `--switch-ratio 0.7` |
| TC07 (cúi xuống) bị báo buồn ngủ | Tăng `--blink-thresh 0.6` |
| TC03/TC04 không báo hoặc báo chậm | Giảm `--blink-thresh 0.4` hoặc `--warn-after 0.8` |
| TC05/TC06 không ra "nhìn chỗ khác" | Giảm `--yaw-thresh 0.2` |
| TC09 (nói chuyện) bị nhận là ngáp | Thu thêm dữ liệu nhãn 0 khi nói chuyện rồi chạy lại `train.py` |

Sau mỗi lần chỉnh, chạy lại `evaluate.py` với cùng tham số đó, rồi so sánh các file `report.md`.
