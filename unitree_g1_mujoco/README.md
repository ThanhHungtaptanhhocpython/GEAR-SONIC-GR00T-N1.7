# Mô Phỏng Unitree G1 - Phân Loại Lon Pepsi & Coca-Cola trên MuJoCo

Thư mục này chứa môi trường mô phỏng 3D hoàn chỉnh của robot hình người **Unitree G1** (có 2 bàn tay linh hoạt nhiều ngón) cùng bài toán phân loại lon nước ngọt (**Coca-Cola** và **Pepsi**) trên nền tảng **MuJoCo**.

---

## 📁 Cấu trúc thư mục

- `run_simulation.py`: Script Python khởi chạy cửa sổ mô phỏng 3D tương tác thời gian thực.
- `preview_g1_scene.png`: Ảnh chụp góc nhìn 3D của môi trường mô phỏng.
- `unitree_g1/`:
  - `scene_pepsi_coke_sorting.xml`: File XML chứa robot G1 đứng trước bàn, 2 lon nước và 2 khay phân loại.
  - `g1_with_hands.xml`: Định nghĩa mô hình robot Unitree G1 (kèm 2 bàn tay linh hoạt) từ Google DeepMind MuJoCo Menagerie.
  - `assets/`: Toàn bộ file 3D mesh (.STL / .OBJ) của các khớp và ngón tay G1.

---

## 🚀 Cách khởi chạy

Mở Terminal hoặc PowerShell và chạy lệnh:

```bash
cd c:\Users\Lenovo\Documents\GitHub\NVIDIA\unitree_g1_mujoco
python run_simulation.py
```

### 🎮 Điều khiển chuột trên cửa sổ MuJoCo 3D:
- **Chuột trái:** Giữ và rê chuột để **xoay góc nhìn** (Rotate).
- **Chuột phải:** Giữ và rê chuột để **di chuyển góc nhìn** (Pan).
- **Cuộn chuột:** **Phóng to / Thu nhỏ** (Zoom).
- **Giữ phím Ctrl + Chuột phải:** Kéo trực tiếp cánh tay robot hoặc lon nước để thử nghiệm va chạm vật lý.
- **Phím SPACE:** Tạm dừng (Pause) / Tiếp tục mô phỏng.
- **Phím ESC:** Thoát cửa sổ mô phỏng.
