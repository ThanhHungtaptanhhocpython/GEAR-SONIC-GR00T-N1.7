# HỆ THỐNG ĐIỀU KHIỂN ROBOT TOÀN THÂN: GEAR-SONIC + NVIDIA GR00T N1.7 TRÊN MUJOCO 3D
## Tích Hợp Mô Hình VLA Đa Phương Thức, Bộ Giải Mã Toàn Thân WBC & Môi Trường Vật Lý Unitree G1

Dự án này tích hợp hoàn chỉnh hệ thống điều khiển robot hình người **Unitree G1 (29 DoF toàn thân + 14 DoF hai bàn tay linh hoạt)** thực hiện bài toán thao tác phân loại lon nước ngọt (**Coca-Cola** đỏ và **Pepsi** xanh) trên nền tảng mô phỏng vật lý **MuJoCo 3D**.

Hệ thống kết nối trực tiếp 3 trụ cột công nghệ tiên tiến nhất của NVIDIA Robotics:
1. **NVIDIA GR00T N1.7 VLA**: Tiếp nhận câu lệnh ngôn ngữ tự nhiên và ảnh camera góc nhìn người thứ nhất (`ego_view` $448 \times 672$) từ đầu robot, sinh chuỗi Action Chunk 40 bước ($64\text{ dims}$ latent motion token + $14\text{ dims}$ bàn tay).
2. **GEAR-SONIC Whole-Body Controller (WBC)**: Giải mã $64\text{ dims}$ latent motion token ở tần số **$50\text{ Hz}$** ($dt=20\text{ ms}$) thành góc quay cho toàn bộ 43 motor robot, duy trì thăng bằng trọng tâm và điều phối hai tay vươn tới mục tiêu.
3. **Môi Trường Mô Phỏng Vật Lý MuJoCo 3D**: Render camera góc nhìn robot thời gian thực, mô phỏng va chạm vật lý ở tần số **$500\text{ Hz}$**, hỗ trợ cửa sổ tương tác 3D trực quan.

---

## 🏗️ SƠ ĐỒ KIẾN TRÚC ĐA TẦN SỐ (CLOSED-LOOP ARCHITECTURE)

```
                     ┌────────────────────────────────────────────────────────┐
                     │          NVIDIA GR00T N1.7 VLA POLICY                  │
                     │  • Lệnh ngôn ngữ: "Pick up Coke/Pepsi can..."           │
                     │  • Camera Ego-View: 448x672 RGB (Step 02 Preprocess)   │
                     │  • Trạng thái 43 khớp + Vector trọng trường chiếu      │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
                                                │ Suy luận @ 2.5 Hz - 5 Hz
                                                │ Action Chunk (T=40 x 78 dims):
                                                │ - 64-dim Latent Motion Token
                                                │ - 14-dim Hand Joints (7L + 7R)
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │          GEAR-SONIC WHOLE-BODY CONTROLLER              │
                     │  • Giải mã 64-dim latent tokens @ 50 Hz (20 ms)        │
                     │  • Giữ thăng bằng trọng tâm 12 khớp chân               │
                     │  • Định hướng thân hông 3 khớp waist                   │
                     │  • Điều khiển 14 khớp cánh tay vươn tới lon nước       │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
                                                │ Xung điều khiển 43 motor @ 50 Hz
                                                │ (Bộ lọc làm mịn Low-Pass Filter)
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                      MÔI TRƯỜNG MÔ PHỎNG VẬT LÝ MUJOCO 3D                                 │
│  • Robot: Unitree G1 43-DOF có bàn tay linh hoạt nhiều ngón                               │
│  • Bàn thao tác (X=0.48m, Z=0.68m) với Lon Coca-Cola (đỏ), Lon Pepsi (xanh), 2 khay chứa  │
│  • Bước vật lý: mj_step @ 500 Hz (10 substeps mỗi chu kỳ WBC)                             │
│  • Sensor: Camera ego_view (448x672), camera front_view, IMU Pelvis, Encoders khớp        │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN

```
GEAR-SONIC-GR00T-N1.7/
├── launch_vla_simulation.py         # Script chính khởi chạy cửa sổ mô phỏng tương tác 3D
├── verify_pipeline.py               # Bộ kiểm thử tự động toàn diện (Scene, WBC, VLA, Pipeline)
├── README_VN.md                     # Tài liệu hướng dẫn Tiếng Việt
├── README.md                        # Master Documentation (English)
│
├── sim/                             # Module mô phỏng MuJoCo
│   ├── g1_scene.py                  # Wrapper điều khiển MuJoCo, camera rendering và trích xuất sensor
│   ├── preview_ego_view.png         # Ảnh chụp mẫu góc nhìn camera thứ nhất của robot (448x672)
│   ├── preview_front_view.png       # Ảnh chụp mẫu góc nhìn toàn cảnh người thứ ba
│   └── unitree_g1/                  # Assets, mesh STL, và file cấu hình XML:
│       ├── scene_pepsi_coke_sorting.xml  # Scene tích hợp robot, bàn, lon nước, khay và camera
│       └── g1_with_hands.xml        # Định nghĩa 43 khớp và camera ego_view
│
├── bridge/                          # Cầu nối tích hợp VLA & SONIC
│   ├── sonic_wbc_controller.py      # Bộ giải mã GEAR-SONIC WBC (64 dims -> 43 motor targets @ 50Hz)
│   ├── gr00t_policy_client.py       # Giao diện VLA Policy GR00T N1.7 (ZMQ Remote + Standalone)
│   └── closed_loop_pipeline.py      # Vòng lặp điều phối đa tần số khép kín (2.5Hz -> 50Hz -> 500Hz)
│
└── GR00T-WholeBodyControl/          # Kho lưu trữ gốc GEAR-SONIC từ NVIDIA NVlabs
    ├── gear_sonic/                  # Mã nguồn lõi SONIC
    ├── gear_sonic_deploy/           # Cấu hình deploy và weights ONNX
    ├── sonic_release/last.pt        # Checkpoint PyTorch SONIC (469 MB)
    └── bones_seed_smpl/             # Dữ liệu huấn luyện cử động người SMPL
```

---

## 🚀 HƯỚNG DẪN KHỞI CHẠY NHANH (QUICK START)

### 1. Kiểm thử tự động hệ thống (Không cần giao diện đồ họa)

Chạy script kiểm tra tự động để xác nhận toàn bộ 4 thành phần (MuJoCo, SONIC WBC, GR00T Policy, và Closed-loop Pipeline) hoạt động hoàn hảo:

```bash
cd c:\Users\Lenovo\Documents\GitHub\NVIDIA\GEAR-SONIC-GR00T-N1.7
python verify_pipeline.py
```

*Kết quả mong đợi*: Cả 4 bài kiểm tra đều vượt qua với thông báo `ALL TESTS PASSED!`.

---

### 2. Khởi chạy cửa sổ mô phỏng tương tác 3D

Chạy trực tiếp môi trường mô phỏng 3D trên MuJoCo:

```bash
# Mặc định: Chế độ Standalone, bắt đầu với nhiệm vụ gắp lon Coca-Cola
python launch_vla_simulation.py

# Hoặc bắt đầu ngay với nhiệm vụ gắp lon Pepsi:
python launch_vla_simulation.py --task 2
```

### 🎮 Phím nóng điều khiển trong cửa sổ mô phỏng:
| Phím | Chức Năng |
| :---: | :--- |
| **`[1]`** | Chuyển ngay sang nhiệm vụ: **"Gắp lon Coca-Cola sang khay đỏ"** (Dùng tay phải) |
| **`[2]`** | Chuyển ngay sang nhiệm vụ: **"Gắp lon Pepsi sang khay xanh"** (Dùng tay trái) |
| **`[SPACE]`** | Tạm dừng (Pause) / Tiếp tục (Resume) mô phỏng |
| **`[R]`** | Đặt lại (Reset) robot và 2 lon nước về vị trí ban đầu |
| **`[ESC]`** | Thoát chương trình |
| **Chuột trái** | Giữ và rê để **Xoay góc nhìn 3D** (Rotate) |
| **Chuột phải** | Giữ và rê để **Di chuyển góc nhìn** (Pan) |
| **Cuộn chuột** | **Phóng to / Thu nhỏ** khung nhìn (Zoom) |
| **Ctrl + Chuột phải** | Tương tác lực vật lý trực tiếp: Kéo rê lon nước hoặc tay robot trong thời gian thực |

---

### 3. Kết nối với Isaac-GR00T PolicyServer thực tế (Tùy chọn)

Nếu bạn đã khởi chạy `PolicyServer` của mô hình `nvidia/GR00T-N1.7-3B` trên máy chủ GPU từ kho lưu trữ `Thanh-Hung---Isaac-GR00T`:

```bash
# Bước 1: Khởi động PolicyServer (từ thư mục Thanh-Hung---Isaac-GR00T)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag UNITREE_G1_SONIC \
    --port 5550

# Bước 2: Kết nối MuJoCo với PolicyServer qua ZMQ
python launch_vla_simulation.py --policy-mode remote --server-host localhost --server-port 5550
```

---

## 📊 THÔNG SỐ KỸ THUẬT KHÔNG GIAN HÀNH ĐỘNG & TRẠNG THÁI

### Không Gian Trạng Thái Đầu Vào (State Space - 46 dimensions):
- `left_leg` (6 dims): 3 hông (pitch, roll, yaw), 1 đầu gối, 2 cổ chân (pitch, roll).
- `right_leg` (6 dims): 3 hông, 1 đầu gối, 2 cổ chân.
- `waist` (3 dims): yaw, roll, pitch hông.
- `left_arm` (7 dims): 3 vai, 1 khuỷu tay, 3 cổ tay.
- `right_arm` (7 dims): 3 vai, 1 khuỷu tay, 3 cổ tay.
- `left_hand` (7 dims): 3 ngón cái, 2 ngón giữa, 2 ngón trỏ.
- `right_hand` (7 dims): 3 ngón cái, 2 ngón trỏ, 2 ngón giữa.
- `projected_gravity` (3 dims): Vector trọng lực $[g_x, g_y, g_z]$ chiếu vào thân robot.

### Không Gian Hành Động VLA (Action Space - 78 dimensions x 40 steps):
- `motion_token` (64 dims): Vector tiềm ẩn điều khiển toàn thân cho SONIC.
- `left_hand_joints` (7 dims): Góc điều khiển bàn tay trái.
- `right_hand_joints` (7 dims): Góc điều khiển bàn tay phải.
