# 📋 BẢNG ĐÁNH GIÁ MỨC ĐỘ SẴN SÀNG MÔ PHỎNG VLA (SIMULATION READINESS CHECKLIST)
## Dự án: GEAR-SONIC + NVIDIA GR00T N1.7 trên MuJoCo 3D (Robot Unitree G1)

> **Cập nhật ngày:** 23/09/2026  
> **Trạng thái tổng quan:** **~99% Hoàn thiện** | **Đã tích hợp xong ONNX Decoder, Môi trường Runtime, Phím nóng tương tác & Colab PolicyServer** | **Sẵn sàng chạy thực tế**

---

## 🎯 BẢNG CHECKLIST TỔNG HỢP (TRACKING CHECKLIST)

### 1. Trọng số & Checkpoints mô hình (Weights)
- [x] **NVIDIA GR00T-N1.7-3B Checkpoint** *(Đã tải $6.5\text{ GB}$ về `./checkpoints/GR00T-N1.7-3B`)*
  - [x] Trọng số chính: `model-00001-of-00002.safetensors` ($4.7\text{ GB}$) & `model-00002-of-00002.safetensors` ($1.8\text{ GB}$)
  - [x] Metadata & cấu hình: `config.json`, `embodiment_id.json`, `processor_config.json`, `statistics.json`
- [x] **NVIDIA GEAR-SONIC WBC Checkpoint** *(Đã tải $3.8\text{ GB}$ về `./checkpoints/GEAR-SONIC`)*
  - [x] Bộ giải mã ONNX: `model_decoder.onnx` ($40\text{ MB}$), `low_latency/model_decoder.onnx` ($143\text{ MB}$), `sonic_v1_1/model_decoder.onnx` ($143\text{ MB}$)
  - [x] Mô hình mã hóa & kế hoạch: `model_encoder.onnx`, `planner_sonic.onnx` ($739\text{ MB}$)
  - [x] PyTorch Checkpoints: `sonic_release/last.pt` ($448\text{ MB}$), `low_latency/last.pt` ($1.1\text{ GB}$), `sonic_v1_1/last.pt` ($1.1\text{ GB}$)
- [ ] **Cosmos-Reason2-2B Backbone** *(Tùy chọn: chỉ cần nếu chạy full Hugging Face in-process cục bộ không qua PolicyServer)*

---

### 2. Môi trường Mô phỏng MuJoCo & Tài nguyên 3D (Simulation Core)
- [x] **Scene phân loại lon nước 3D**: `sim/unitree_g1/scene_pepsi_coke_sorting.xml`
- [x] **Mô hình Robot Unitree G1 43-DOF**: `sim/unitree_g1/g1_with_hands.xml` (kèm 2 bàn tay linh hoạt DexHands)
- [x] **Hệ thống Camera Sensors**:
  - [x] Camera góc nhìn thứ nhất của robot: `ego_view` ($448 \times 672$)
  - [x] Camera quan sát toàn cảnh bàn làm việc: `front_camera`
- [x] **MuJoCo Simulation Driver**: `sim/g1_scene.py` (Reset, step physics 500 Hz, trích xuất sensor, kinematics)
- [x] **Giao diện tương tác 3D thời gian thực**: `launch_vla_simulation.py` (MuJoCo Passive Viewer, phím tắt `[1]`, `[2]`, `[SPACE]`, `[R]`, `[S]`, tham số `--prompt`)

---

### 3. Cầu nối điều khiển & Bộ giải mã (Bridge & Controllers)
- [x] **Vòng lặp điều khiển khép kín đa tần số**: `bridge/closed_loop_pipeline.py` (500 Hz Physics $\rightarrow$ 50 Hz WBC $\rightarrow$ 2.5–5 Hz VLA)
- [x] **Script tải dữ liệu tự động**: `download_checkpoint.py` (Tải từ Hugging Face qua token xác thực)
- [x] **Kết nối ONNX Decoder thật vào `SonicWBCController`**:
  - Đã nạp thành công `./checkpoints/GEAR-SONIC/model_decoder.onnx` qua ONNX Runtime.
  - Tự động duy trì cửa sổ trượt lịch sử 10 frames ($994\text{ dims}$) từ proprioception và giải mã thành $29\text{ DOFs}$ góc khớp thân.
  - Tự động fallback về Kinematic WBC nếu không tìm thấy file ONNX.
- [x] **Kiến trúc PolicyServer trên Google Colab GPU**:
  - Script server: [`colab_gr00t_server.py`](file:///home/thanhhung27/Code/GEAR-SONIC-GR00T-N1.7/colab_gr00t_server.py)
  - Jupyter Notebook Colab: [`colab_gr00t_server.ipynb`](file:///home/thanhhung27/Code/GEAR-SONIC-GR00T-N1.7/colab_gr00t_server.ipynb)
- [x] **Đồng bộ hóa Action Space**: Khớp output 78 chiều của GR00T Policy (`[64 latent token, 14 hand DOFs]`) với input 994 chiều của GEAR-SONIC ONNX decoder và 43 motor actuators của robot G1.

---

### 4. Môi trường Thực thi & Thư viện Python (Dependencies) — **ĐÃ HOÀN TẤT 100%**
- [x] **Môi trường ảo Python**: Đã tạo môi trường chuyên biệt `.venv/` (Python 3.10.12)
- [x] **Cài đặt thư viện mô phỏng**:
  - `mujoco` (v3.13.0, hỗ trợ EGL headless render)
- [x] **Cài đặt thư viện AI & Inference**:
  - `torch` (v2.14.0+cu130), `torchvision` (v0.29.0)
  - `onnxruntime` (v1.23.2, đã kiểm tra nạp thành công)
  - `transformers` (v5.17.0), `accelerate` (v1.15.0), `safetensors` (v0.8.0)
- [x] **Thư viện phụ trợ**:
  - `pyzmq` (v27.2.0), `pyyaml` (v6.0.3), `scipy` (v1.15.3), `numpy` (v2.2.6)

---

## 🔍 CHI TIẾT CÁC RÀO CẢN CẦN GIẢI QUYẾT

### Rào cản 1: Cập nhật `SonicWBCController` dùng ONNX thật
* **Vấn đề:** 
  Hàm `SonicWBCController.decode(...)` hiện tại đang gán góc khớp theo công thức thủ công:
  ```python
  l_shoulder_pitch = 0.2 + 0.9 * motion_token[15]
  ```
  GR00T N1.7 sinh ra latent token nằm trong không gian phân phối tiềm ẩn (learned latent space), không thể điều khiển robot đúng nếu không đi qua mạng nơ-ron giải mã.
* **Trạng thái:** ✅ **ĐÃ GIẢI QUYẾT 100%**
  Đã tích hợp mạng nơ-ron sâu `model_decoder.onnx` thông qua ONNX Runtime trong `bridge/sonic_wbc_controller.py`. Bộ đệm trượt 10 frame (994 chiều) hoạt động trơn tru ở 50 Hz.

---

### Rào cản 2: Kiến trúc chạy VLA (Google Colab GPU + Local MuJoCo)
* **Trạng thái:** ✅ **ĐÃ GIẢI QUYẾT 100%**
  Đã hoàn thiện giải pháp chạy VLA trên GPU Google Colab và kết nối về MuJoCo trên máy tính cá nhân qua ZeroMQ (ZMQ) và Ngrok TCP Tunnel:
  1. File Notebook Colab: [`colab_gr00t_server.ipynb`](file:///home/thanhhung27/Code/GEAR-SONIC-GR00T-N1.7/colab_gr00t_server.ipynb)
  2. Script Colab Server: [`colab_gr00t_server.py`](file:///home/thanhhung27/Code/GEAR-SONIC-GR00T-N1.7/colab_gr00t_server.py)

---

### Rào cản 3: Thiết lập môi trường Python & Dependencies
* **Trạng thái:** ✅ **ĐÃ GIẢI QUYẾT 100%**
  Đã tạo môi trường ảo Python 3.10 tại `.venv/` và cài đặt đầy đủ tất cả các gói thư viện (`mujoco`, `torch`, `torchvision`, `onnxruntime`, `transformers`, `pyzmq`, `pyyaml`...).

---

## 🎮 HƯỚNG DẪN GIAO NHIỆM VỤ (TASK) CHO ROBOT

### Cách 1: Giao nhiệm vụ qua tham số dòng lệnh (`--prompt`)
Bạn có thể ra lệnh bằng văn bản tiếng Anh tự nhiên bất kỳ khi khởi chạy:
```bash
.venv/bin/python launch_vla_simulation.py --prompt "Pick up the Coca-Cola can and place it in the red tray"
```
hoặc:
```bash
.venv/bin/python launch_vla_simulation.py --prompt "Pick up the Pepsi can and place it in the blue tray"
```

### Cách 2: Tương tác thời gian thực bằng phím tắt trong cửa sổ 3D MuJoCo Viewer
Khi cửa sổ mô phỏng 3D đang hiển thị, bạn có thể điều khiển trực tiếp bằng bàn phím:
* `[1]` : Giao nhiệm vụ robot gắp lon **Coca-Cola** đưa vào khay đỏ (sử dụng tay phải).
* `[2]` : Giao nhiệm vụ robot gắp lon **Pepsi** đưa vào khay xanh (sử dụng tay trái).
* `[SPACE]` : **Tạm dừng** / **Tiếp tục** dòng thời gian mô phỏng vật lý.
* `[R]` : **Reset** robot và các lon nước về lại vị trí ban đầu trên bàn.
* `[S]` : **Chụp ảnh snapshot** từ cả 2 góc camera (`ego_view` và `front_camera`) và lưu vào thư mục `./snapshots/`.
* `[ESC]` : Thoát khỏi chương trình an toàn.

---

## 🚀 HƯỚNG DẪN CHẠY KẾT NỐI VỚI GOOGLE COLAB GPU

1. **Trên Google Colab:**
   * Mở trình duyệt vào [Google Colab](https://colab.research.google.com/).
   * Tải file [`colab_gr00t_server.ipynb`](file:///home/thanhhung27/Code/GEAR-SONIC-GR00T-N1.7/colab_gr00t_server.ipynb) lên Colab.
   * Chọn Menu `Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ Chọn **T4 GPU** (hoặc A100 / L4).
   * Chạy tuần tự các bước 1, 2, 3.
   * Tại **Bước 4**: Điền Ngrok AuthToken của bạn (lấy miễn phí tại [dashboard.ngrok.com](https://dashboard.ngrok.com/)). Chạy cell để lấy địa chỉ host và port của đường hầm TCP.
   * Chạy **Bước 5 & 6** để server GPU bắt đầu lắng nghe.

2. **Trên máy Local (Terminal của bạn):**
   * Dán câu lệnh được Colab in ra ở Bước 4, ví dụ:
   ```bash
   .venv/bin/python launch_vla_simulation.py --policy-mode remote --server-host 0.tcp.ap.ngrok.io --server-port 12345
   ```
   * Cửa sổ MuJoCo 3D sẽ xuất hiện, truyền ảnh camera lên Colab GPU để suy luận và nhận hành động điều khiển robot gắp vật thể mượt mà!
