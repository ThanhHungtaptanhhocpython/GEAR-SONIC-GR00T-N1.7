# Báo Cáo Kỹ Thuật: Phân Định Quy Trình Xử Lý Dữ Liệu BONES-SEED & NVIDIA SONIC

> **Tài liệu tham khảo chính**:  
> * Paper *SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control* (NVIDIA, arXiv:2511.07820v4) — Mục **3.1 Dataset Construction** (Trang 13–14) & References [54, 55, 56, 57].  
> * Dataset công khai: [Bones Studio BONES-SEED trên Hugging Face](https://huggingface.co/datasets/bones-studio/seed).

---

## 1. Tổng Quan Phân Định Trách Nhiệm Xử Lý Dữ Liệu

Chuỗi xử lý dữ liệu từ phòng thu mocap đến khi đưa vào huấn luyện mô hình điều khiển toàn thân robot hình người (**SONIC**) được phân định rõ ràng thành hai phần việc độc lập:

```text
══════════════════════════════════════════════════════════════════════════════════════════════════════
PHẦN I: QUY TRÌNH TIỀN XỬ LÝ CỦA BONES STUDIO (TRƯỚC KHI PHÁT HÀNH BẢN PUBLIC BONES-SEED)
══════════════════════════════════════════════════════════════════════════════════════════════════════
  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ GIAI ĐOẠN 1: THU NHẬN & LÀM SẠCH QUANG HỌC (OPTICAL CLEANUP)                                   │
  │ • Thu nhận tín hiệu camera Vicon 120 FPS từ 522 diễn viên                                      │
  │ • Khử điểm ảo (Ghost markers) & Sửa hoán đổi nhãn (Label swapping)                             │
  │ • Lấp đầy khoảng trống che khuất (Rigid-body Inpainting) & Lọc mượt Butterworth                │
  └───────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                  │
                                                  ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ GIAI ĐOẠN 2: KHỚP MÔ HÌNH NHÂN TRẮC HỌC NGƯỜI (HUMAN MESH FITTING)                             │
  │ • Khớp toàn thân vào khung xương SMPL (24 khớp)                                                │
  │ • Khớp chi tiết chuyển động bàn tay và cơ thể vào chuẩn SOMA (77 khớp)                         │
  │ ──► ĐÓNG GÓI & PHÁT HÀNH CÔNG KHAI BONES-SEED (142,220 sequences, ~288 giờ)                   │
  └───────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                  │
══════════════════════════════════════════════════╪═══════════════════════════════════════════════════
PHẦN II: QUY TRÌNH XỬ LÝ & ĐỒNG HÓA CỦA NVIDIA / SONIC (CHUẨN BỊ TRƯỚC KHI HUẤN LUYỆN ROBOT)
══════════════════════════════════════════════════╪═══════════════════════════════════════════════════
                                                  │
                                                  ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ GIAI ĐOẠN 3: MOTION RETARGETING SANG ROBOT G1 (GENERAL MOTION RETARGETING)                     │
  │ • Đơn vị thực hiện: Đội ngũ nghiên cứu NVIDIA SONIC                                            │
  │ • Thuật toán GMR + Thư viện PyRoki giải bài toán Động học nghịch (IK Optimization)             │
  │ • Bảo toàn trọng tâm (CoM), bảo toàn tiếp xúc chân, ép giới hạn cơ khí 29 DOF Unitree G1       │
  │ • Kiểm tra lưới va chạm tự thân (Self-Collision Avoidance) trong môi trường MuJoCo             │
  └───────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                  │
                                                  ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ GIAI ĐOẠN 4: LỌC BỎ CHUYỂN ĐỘNG PHI VẬT LÝ CHO ROBOT (PHYSICAL FILTERING)                      │
  │ • Đơn vị thực hiện: Đội ngũ nghiên cứu NVIDIA SONIC                                            │
  │ • Loại bỏ các hành vi dựa dẫm ngoại vật: leo cầu thang, tựa lưng ghế, đu xà...                 │
  │ • Tinh giản dung lượng: 700 giờ thô ban đầu ──► 611 giờ chuyển động khả thi trên sàn phẳng     │
  └───────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                  │
                                                  ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ GIAI ĐOẠN 5: TĂNG CƯỜNG DỮ LIỆU & ĐÓNG GÓI HUẤN LUYỆN (AUGMENTATION & CURATION)                │
  │ • Đơn vị thực hiện: Đội ngũ nghiên cứu NVIDIA SONIC                                            │
  │ • Tái lấy mẫu tần số: Hạ từ 120 FPS về 50 Hz (Δt = 0.02 s) đồng bộ chu kỳ điều khiển          │
  │ • Lật đối xứng gương (Sagittal Plane Mirroring) nhân đôi dữ liệu, khử định kiến thuận chi      │
  │ • Phân loại Taxonomy 33 nhóm, 8,447 nhãn con & Phân chia Train / Test-Content / Test-Repetition│
  │ ──► TỔNG CỘNG HƠN 100 TRIỆU FRAME SẴN SÀNG NẠP VÀO HUẤN LUYỆN POLICY SONIC                     │
  └────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. PHẦN I: Các Giai Đoạn Do Bones Studio Thực Hiện (Trước Khi Release)

Toàn bộ giai đoạn này được thực hiện bởi phòng thu chuyên nghiệp **Bones Studio** nhằm tạo ra tập dữ liệu chuyển động người chất lượng cao nhất, sạch hoàn toàn lỗi quang học.

### Giai Đoạn 1: Thu Nhận & Làm Sạch Dữ Liệu Quang Học (Optical Cleanup)
1. **Khử điểm ảo (Ghost Marker Removal)**:
   * Loại bỏ các tọa độ $3\text{D}$ ngoại lai do hiện tượng phản xạ ánh sáng hồng ngoại từ mặt sàn, đạo cụ kim loại hoặc tán xạ ngoài trường nhìn camera ($V_{\text{capture}}$).
2. **Sửa lỗi hoán đổi nhãn (Label Swapping Correction)**:
   * Khi diễn viên thực hiện các động tác hai chi chạm sát nhau (khoanh tay, đấm bốc, bắt chéo chân), hệ thống Vicon dễ gán nhầm định danh (ID) giữa marker trái và marker phải.
   * Áp dụng thuật toán **Theo dõi quỹ đạo liên tục (Continuous Trajectory Tracking)** kết hợp ràng buộc khoảng cách xương bất biến:
     $$\|p_i(t) - p_j(t)\| \approx L_{ij} \quad (\forall t)$$
3. **Lấp khoảng trống mất dấu (Gap-Filling / Inpainting)**:
   * Khi marker bị che khuất trong khoảng $5 \div 30\text{ frames}$, áp dụng mô hình **Nội suy vật thể cứng (Rigid-Body Interpolation)** từ cụm marker lân cận cùng đoạn xương:
     $$p_{\text{lost}}(t) = R_{\text{rigid}}(t) \cdot p_{\text{lost}}(t_0) + T_{\text{rigid}}(t)$$
   * Kết hợp tiên nghiệm động học (**Kinematic Priors**) đảm bảo quỹ đạo chuyển động liên tục, không nhảy vọt gia tốc.
4. **Lọc mượt tín hiệu (Smoothing Filter)**:
   * Bộ lọc số thông thấp Butterworth bậc 4 không lệch pha ($f_c \approx 6 \div 10\text{ Hz}$):
     $$|H(f)|^2 = \frac{1}{1 + \left(\frac{f}{f_c}\right)^{2n}} \quad (n = 4)$$
   * Triệt tiêu hoàn toàn rung lắc tần số cao dưới milimét, đảm bảo đạo hàm vận tốc $\dot{p}(t)$ và gia tốc $\ddot{p}(t)$ khả vi liên tục.

### Giai Đoạn 2: Khớp Khung Xương Người (Human Mesh Fitting) & Phát Hành BONES-SEED
1. **Khớp mô hình SMPL (24 khớp)**:
   * Toàn bộ chuyển động cơ thể được khớp vào chuẩn $\text{SMPL}$ gồm tọa độ gốc $p_{\text{root}} \in \mathbb{R}^3$, hướng gốc $R_{\text{root}} \in \text{SO}(3)$ và góc quay của $23$ khớp cơ thể $\theta_k \in \text{SO}(3)$.
2. **Khớp mô hình SOMA (77 khớp)**:
   * Để hỗ trợ điều khiển thao tác khéo léo (**Dexterous Manipulation**), dữ liệu được khớp vào chuẩn **$\text{SOMA}$ ($77$ khớp)** của Saito et al. [57], bao gồm cử động độc lập của $10$ ngón tay ($30$ bậc tự do bàn tay), khớp mắt cá chân đa trục và cử động đầu.
3. **Đóng gói phát hành công khai**:
   * Bones Studio phát hành gói **BONES-SEED** trên Hugging Face gồm **$142{,}220$ chuỗi chuyển động ($\sim 288$ giờ)** của $522$ diễn viên ở định dạng SOMA và SMPL.

---

## 3. PHẦN II: Các Giai Đoạn Do NVIDIA / SONIC Thực Hiện (Trước Khi Huấn Luyện)

Sau khi tiếp nhận dữ liệu chuyển động người đã làm sạch từ Bones Studio, nhóm nghiên cứu **NVIDIA** tiến hành đồng hóa dữ liệu này sang không gian điều khiển của robot hình người Unitree G1.

### Giai Đoạn 3: Motion Retargeting Sang Robot G1 (29 DOF)
Chuyển động người không thể nạp trực tiếp vào robot do sự bất đồng cấu trúc cơ khí (**Morphological Discrepancy**). NVIDIA sử dụng **GMR (General Motion Retargeting)** [54] kết hợp **PyRoki** [55] để giải bài toán tối ưu động học nghịch (IK):

$$\min_{q_t} \left[ \mathcal{L}_{\text{pos}}(q_t) + \lambda_{\text{rot}} \mathcal{L}_{\text{rot}}(q_t) + \lambda_{\text{CoM}} \mathcal{L}_{\text{CoM}}(q_t) + \lambda_{\text{contact}} \mathcal{L}_{\text{contact}}(q_t) + \lambda_{\text{reg}} \|q_t - q_{t-1}\|^2 \right]$$

Các ràng buộc nghiêm ngặt được áp đặt:
* **Bảo toàn khối tâm**: $\mathcal{L}_{\text{CoM}}(q_t) = \|\mathbf{c}(q_t) - s \cdot \mathbf{c}_{\text{human}}\|^2$ (giữ cân bằng động, chống đổ nhào).
* **Bảo toàn tiếp xúc chân**: Khi chân người chạm đất ($v_{\text{foot}} \approx 0$), chân robot bị khóa vận tốc tiếp tuyến để triệt tiêu hiện tượng trượt chân giả tạo (*foot sliding*).
* **Giới hạn cơ khí & Chống tự va chạm**:
  $$q_{\min} \le q_t \le q_{\max}, \quad |\dot{q}_t| \le \dot{q}_{\max}, \quad \text{dist}_{\text{collision}}(q_t) > d_{\text{safe}} > 0$$
  *(kiểm tra va chạm các khối mắt xích Collision Meshes của G1 trong môi trường MuJoCo).*

### Giai Đoạn 4: Lọc Bỏ Dữ Liệu Phi Vật Lý Cho Robot (Physical Filtering)
* Con người có thể thực hiện nhiều tư thế nhờ điểm tựa ngoại cảnh (tựa lưng vào tường, ngồi ghế, nắm tay vịn cầu thang, đu xà). Robot G1 hoạt động độc lập trên mặt sàn phẳng không thể thực hiện các động tác này.
* Nhóm nghiên cứu NVIDIA đã lọc bỏ toàn bộ các chuỗi: **Leo cầu thang (Stair climbing)**, **Ngồi ghế / dựa lưng (Seated activities)**, **Đu bám ngoại vật**.
* Kho dữ liệu được tinh tuyển từ **$\sim 700$ giờ thô ban đầu xuống còn $611$ giờ dữ liệu khả thi về mặt vật lý**.

### Giai Đoạn 5: Tăng Cường Dữ Liệu & Chuẩn Bị Huấn Luyện (Augmentation & Curation)
1. **Tái lấy mẫu thời gian (Temporal Downsampling)**:
   * Hạ tần số từ $120\text{ FPS}$ (chuẩn quay mocap) về $50\text{ Hz}$ ($\Delta t = 0.02\text{ s}$), khớp chính xác với chu kỳ điều khiển của Policy mạng nơ-ron $\rightarrow$ đạt trên **$100$ triệu frames**.
2. **Lật đối xứng gương (Sagittal Plane Mirroring)**:
   * Với mỗi quỹ đạo $q(t)$, tạo chuỗi đối xứng $q^{\text{mirrored}}(t) = \mathcal{M}_{\text{sagittal}}(q(t))$ để nhân đôi dữ liệu và triệt tiêu thói quen thuận tay/chân của diễn viên.
3. **Phân tầng Taxonomy & Phân chia tập dữ liệu**:
   * Phân loại thành **$33$ nhóm chính** và **$8{,}447$ danh mục con độc nhất** (Locomotion, Acting, Combat, Gestures, Manipulation...).
   * **Train Split**: $317{,}189$ clips ($611$ giờ), bao quát $8{,}447$ danh mục con.
   * **Test-Content Split ($15$ giờ)**: Gồm $182$ danh mục con **hoàn toàn mới lạ** để kiểm thử khả năng zero-shot generalization.
   * **Test-Repetition Split ($12$ giờ)**: Gồm các lần diễn mới của cùng danh mục quen thuộc do diễn viên khác thực hiện.

---

## 4. Bảng Đối Chiếu Ranh Giới Trách Nhiệm

| Hạng Mục | Phần Việc Của Bones Studio (Trước Khi Release) | Phần Việc Của NVIDIA / SONIC (Trước Khi Huấn Luyện) |
| :--- | :--- | :--- |
| **Mục tiêu cốt lõi** | Thu nhận và làm sạch dữ liệu quang học con người chuẩn xác | Chuyển đổi và tương thích hóa dữ liệu người sang Robot Unitree G1 |
| **Không gian trạng thái** | Tọa độ Marker $3\text{D} \rightarrow$ Khung xương người SMPL (24 khớp) / SOMA (77 khớp) | Không gian góc khớp robot Unitree G1 (29 DOF) |
| **Xử lý hình học** | Sửa nhãn chéo, lấp khuyết điểm marker, lọc Butterworth | Tối ưu hóa IK (GMR + PyRoki), chống tự va chạm mắt xích cơ khí |
| **Xử lý vật lý** | Đảm bảo tính liên tục giải phẫu học của con người | Lọc bỏ tương tác phi vật lý (loại bỏ leo thang, ngồi dựa ghế) |
| **Tần số lấy mẫu** | Giữ nguyên tần số thu nhận gốc $120\text{ FPS}$ | Downsample về $50\text{ Hz}$ ($\Delta t = 0.02\text{ s}$) đồng bộ điều khiển robot |
| **Tăng cường dữ liệu** | Không áp dụng (giữ nguyên tính chân thực lượt diễn) | Lật gương đối xứng (Sagittal Mirroring) nhân đôi dữ liệu |
| **Sản phẩm đầu ra** | **BONES-SEED Dataset công khai (142,220 clips, ~288 giờ)** | **611 giờ huấn luyện (100+ triệu frame @ 50 Hz, 8,447 classes)** |
