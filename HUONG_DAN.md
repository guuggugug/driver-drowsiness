# Hướng dẫn sử dụng Hệ thống Giám sát Buồn ngủ & Nhận diện Tài xế

Tài liệu này hướng dẫn từng bước từ lúc mở môi trường, đăng ký khuôn mặt tài xế, chạy giám sát real-time, đến tự thu dữ liệu và train model riêng.

---

## Bước 1: Mở cửa sổ lệnh (CMD) và kích hoạt môi trường

1. Nhấn `Win + R`, gõ `cmd`, nhấn **Enter**.
2. Chạy 2 lệnh sau để vào thư mục dự án và bật môi trường ảo `.venv`:

```cmd
cd /d C:\Users\tule\Documents\GitHub\driver-drowsiness
.venv\Scripts\activate
```

> **Lưu ý:** Nếu thấy `( .venv )` ở đầu dòng lệnh nghĩa là đã kích hoạt thành công.

---

## Bước 2: Đăng ký khuôn mặt tài xế (chỉ cần làm 1 lần cho mỗi người)

Chạy lệnh sau (thay `ten_ban` bằng tên tài xế, không dấu, không khoảng trắng):

```cmd
python enroll.py --user ten_ban
```

Thực hiện theo hướng dẫn trên cửa sổ camera:
1. **10 giây đầu (Đăng ký khuôn mặt):** Nhìn vào camera và xoay nhẹ đầu theo 5 hướng: nhìn thẳng → xoay trái → xoay phải → ngẩng lên → cúi xuống.
2. **8 giây tiếp theo (Hiệu chỉnh mắt):** Giữ đầu thẳng, nhìn thẳng vào camera với mắt mở bình thường để app lưu ngưỡng riêng cho bạn.
3. Muốn đăng ký thêm tài xế khác, chạy lại lệnh trên với tên khác (ví dụ `--user nam`).
4. Muốn bổ sung thêm góc mặt cho tài xế cũ mà không ghi đè, thêm cờ `--add`:
   ```cmd
   python enroll.py --user ten_ban --add
   ```

---

## Bước 3: Chạy ứng dụng giám sát

Bạn có thể chọn **1 trong 2 cách**:

- **Cách nhanh:** Mở thư mục `C:\Users\tule\Documents\GitHub\driver-drowsiness` và **double-click** vào file `run_app.bat`.
- **Cách chạy bằng CMD:**
  ```cmd
  python app.py
  ```

### Điều gì sẽ xảy ra khi app chạy?
1. Trong **3 giây đầu**, app nhìn vào camera để nhận diện bạn là tài xế nào đã đăng ký. Nếu không nhận ra, app dùng hồ sơ `khach` (và yêu cầu hiệu chỉnh 8 giây nếu chưa có).
2. Trên màn hình hiển thị:
   - **Khung khuôn mặt** kèm tên tài xế (hoặc `Nguoi la`).
   - **Viền & khung quanh 2 mắt**: màu **xanh lá** = mắt mở, màu **đỏ** = mắt nhắm.
   - **Thanh trạng thái** và các chỉ số `Nham mat`, `Ha mieng`, `pitch`, `yaw`.
3. **Phím tắt khi đang chạy:**
   - `q`: Thoát ứng dụng.
   - `c`: Hiệu chỉnh lại mắt (nếu đổi vị trí ngồi hoặc ánh sáng).

---

## Bước 4 (Tùy chọn): Tự thu dữ liệu & Train model riêng

App mặc định đã dùng sẵn model của MediaPipe (không cần train vẫn chạy tốt). Nếu muốn train thêm model riêng cho bản thân:

### 4.1. Thu dữ liệu bằng webcam
```cmd
python collect_data.py --user ten_ban
```
- Nhấn phím số để chọn trạng thái trước khi ghi:
  - `0` (**Tỉnh táo**): Mắt **mở rõ**, nhìn thẳng, kết hợp quay đầu trái/phải, nhìn gương, nhìn lên/xuống để model học rằng quay mặt không phải buồn ngủ.
  - `1` (**Buồn ngủ**): Nhắm mắt lâu, lim dim, hoặc gật đầu (không ghi nhãn 1 khi mắt đang mở).
  - `2` (**Ngáp**): Ngáp thật hoặc há miệng ngáp.
- Nhấn `SPACE` để **bật/tắt** ghi (mỗi lần bật/tắt là 1 phiên; nên ghi ít nhất **3 phiên** cho mỗi nhãn).
- Nhấn `q` để thoát.

### 4.2. Train model
```cmd
python train.py
```
Lệnh này đọc dữ liệu từ `monitor.db` và tạo ra `model_window.joblib` (và `model_frame.joblib`). Lần sau chạy `python app.py`, app sẽ tự động kết hợp model của bạn với MediaPipe.

---

## Bước 5: Các tùy chọn chỉnh ngưỡng khi cần

Nếu chạy thực tế thấy quá nhạy hoặc chưa nhạy, bạn có thể truyền thêm tham số cho `app.py`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--user ten_ban` | Tự nhận diện | Chỉ định cố định tài xế, bỏ qua bước nhận diện đầu |
| `--camera 1` | `0` | Đổi sang webcam khác nếu máy có nhiều camera |
| `--blink-thresh 0.6` | `0.5` | Tăng lên nếu mở mắt hay bị báo nhầm; giảm xuống `0.4` nếu nhắm mắt mà không báo |
| `--open-veto 0.25` | `0.25` | Ngưỡng mắt mở rõ để bác bỏ cảnh báo sai của model tự train |
| `--warn-after 1.0` | `1.0` | Số giây nhắm mắt liên tục trước khi cảnh báo nhẹ |
| `--alarm-after 2.0` | `2.0` | Số giây nhắm mắt liên tục trước khi báo động mạnh |
| `--yaw-thresh 0.25` | `0.25` | Ngưỡng góc quay đầu (`~25 độ`), vượt mức này coi là đang nhìn chỗ khác |
| `--distract-after 3.0` | `3.0` | Số giây quay mặt liên tục trước khi nhắc nhìn về phía trước (`0` để tắt) |
| `--recalibrate` | Tắt | Ép hiệu chỉnh lại mắt ngay khi mở app |
| `--no-faceid` | Tắt | Tắt nhận diện khuôn mặt tài xế |

Ví dụ chạy với camera số 1 và tăng ngưỡng nhắm mắt:
```cmd
python app.py --camera 1 --blink-thresh 0.6
```

