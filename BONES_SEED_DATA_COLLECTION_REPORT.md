# Báo Cáo Kỹ Thuật: Quy Trình Thu Thập & Chuẩn Hóa Dữ Liệu BONES-SEED

> **Tài liệu tham khảo**: [BONES-SEED Dataset](https://huggingface.co/datasets/bones-studio/seed) | [Bones Studio](https://bones.studio) | [NVIDIA GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)  
> **Đơn vị thực hiện**: Bones Studio  
> **Ứng dụng chính**: Huấn luyện mô hình điều khiển toàn thân robot hình người (Humanoid Whole-Body Control) & Mô hình hành vi (VLA)

---

## 1. Tổng Quan (Executive Summary)

**BONES-SEED** (*Skeletal Everyday Embodiment Dataset*) là một trong những bộ dữ liệu ghi nhận chuyển động người (Motion Capture - Mocap) quang học quy mô lớn nhất hiện nay được thiết kế chuyên biệt cho robot hình người (Humanoid Robotics) và AI sinh chuyển động.

```text
┌─────────────────┐     ┌────────────────────────────────────┐     ┌──────────────────────────────────────┐
│  522 Diễn viên  │ ──► │  Studio Optical Mocap (120 FPS)    │ ──► │  Làm sạch & Gán nhãn đa phương thức  │
└─────────────────┘     └────────────────────────────────────┘     └──────────────────┬───────────────────┘
                                                                                      │
                                                                                      ▼
                                                                   ┌──────────────────────────────────────┐
                                                                   │  Chuyển đổi khung xương (Retargeting) │
                                                                   └───────┬──────────────┬───────────────┘
                                                                           │              │              │
                                                                           ▼              ▼              ▼
                                                                    ┌─────────────┐┌─────────────┐┌──────────────┐
                                                                    │   SOMA 77   ││   SMPL 24   ││  Unitree G1  │
                                                                    │    Khớp     ││    Khớp     ││    29 DOF    │
                                                                    └─────────────┘└─────────────┘└──────────────┘
```

### Các chỉ số định lượng chính:
* **Tổng số chuỗi chuyển động**: **142.220 chuỗi** (gồm 71.132 chuỗi gốc + 71.088 chuỗi lật đối xứng gương).
* **Tổng thời lượng thu**: **~288 giờ** (từ tổng kho dữ liệu gốc hơn 700 giờ của Bones Studio).
* **Tần số lấy mẫu**: **120 FPS** đồng bộ thời gian thực.
* **Số lượng diễn viên**: **522 diễn viên** (253 Nữ, 269 Nam).

---

## 2. Thiết Lập Phần Cứng & Phòng Thu (Hardware & Studio Setup)

Toàn bộ dữ liệu được thực hiện trong môi trường phòng thu quang học đạt chuẩn điện ảnh và công nghiệp game (AAA):

### 2.1. Hệ thống Camera quang học (Optical Motion Capture)
* **Thương hiệu**: Sử dụng hệ thống camera quang học hồng ngoại cao cấp của **Vicon**.
* **Độ phân giải & Tần số**: Ghi nhận tọa độ 3D với tần số **120 khung hình/giây (FPS)**, độ trễ cực thấp và sai số dưới mức milimét (**sub-millimeter precision**).
* **Vùng không gian bắt chuyển động (Capture Volume)**: Thể tích sàn phòng thu rộng cho phép diễn viên thực hiện các động tác tự do như chạy nước rút, xoay người 360 độ, nhào lộn, trườn bò hoặc tương tác với chướng ngại vật mà không bị mất vùng nhìn (blind spots).

### 2.2. Trang phục & Điểm phản quang (Suit & Markers)
* Diễn viên mặc đồ ôm sát cơ thể chuyên dụng được đính các điểm phản quang (retro-reflective markers).
* **Phủ kín toàn thân**: Markers được bố trí theo sơ đồ khớp giải phẫu học chi tiết, bao gồm:
  * Khung xương trục (đầu, cổ, ngực, cột sống, xương chậu).
  * Chi trên và chi dưới (vai, khuỷu tay, cổ tay, hông, đầu gối, mắt cá chân).
  * **Khớp bàn tay và ngón tay chi tiết**: Bắt trọn các cử động cầm nắm ngón tay tinh vi phục vụ cho các tác vụ khéo léo (Dexterous Manipulation).

### 2.3. Thu thập đa phương thức đồng bộ (Multimodal In-Take)
Khác với các tập dữ liệu ghép nối từ nhiều nguồn, Bones Studio thu thập dữ liệu theo phương thức **Single-Take**:
* Dữ liệu 3D Mocap được khóa pha (genlock) với hệ thống **camera RGB độ phân giải cao** từ nhiều góc độ.
* Đồng bộ trực tiếp với kênh âm thanh micro phòng thu, hỗ trợ tối đa cho các mô hình đa phương thức (Audio-conditioned / Video-conditioned Motion Generation).

---

## 3. Độ Đa Dạng Thể Trạng Diễn Viên (Performer Diversity)

Để bộ điều khiển robot hình người có khả năng thích ứng (generalize) vượt trội trên nhiều kiểu dáng vận động khác nhau, tập dữ liệu quy tụ 522 diễn viên/vận động viên với phổ nhân khẩu học cực kỳ phong phú:

| Tiêu chí | Dải phân bố | Ý nghĩa kỹ thuật đối với Robot |
| :--- | :--- | :--- |
| **Giới tính** | 253 Nữ (48.5%) / 269 Nam (51.5%) | Cân bằng trọng tâm và biên độ bước đi giữa nam và nữ |
| **Độ tuổi** | **17 – 71 tuổi** | Học được cả phong cách đi nhanh nhẹn của thanh niên và chậm rãi, vững chắc của người lớn tuổi |
| **Chiều cao** | **145 cm – 199 cm** | Giúp policy robot học cách co giãn sải chân và hạ thấp trọng tâm linh hoạt |
| **Cân nặng** | **38 kg – 145 kg** | Cung cấp dữ liệu về quán tính bước đi và độ rung cơ thể ở các mức tải trọng khác nhau |

---

## 4. Phân Loại Tác Vụ & Kịch Bản Động Tác (Motion Taxonomy)

142.220 động tác trong BONES-SEED được cấu trúc thành **8 danh mục lớn** và hơn 20 danh mục con:

| Danh mục (Package) | Số lượng | Các dạng động tác điển hình |
| :--- | :---: | :--- |
| **Locomotion** | **74.488** | Đi bộ thường, đi nhanh, chạy bộ, leo trèo, nhảy lò cò, trườn bò, xoay hướng 360°, bước lùi, vấp ngã |
| **Communication** | **21.493** | Vẫy tay chào, chỉ trỏ vật thể, gật đầu, lắc đầu, biểu cảm hình thể, nhún vai |
| **Interactions** | **14.643** | Cúi người nhặt đồ, bê vác vật nặng, chuyển đồ vật giữa hai tay, thao tác với công cụ |
| **Dances** | **11.006** | Nhảy Hip-hop, vũ đạo Macarena, múa ba-lê, xoay người theo điệu nhạc |
| **Gaming** | **8.700** | Động tác chiến đấu, nhào lộn, luồn lách, thế thủ mô phỏng nhân vật hành động |
| **Everyday** | **5.816** | Đứng lên, ngồi ghế, uống nước, với tay lấy sách, quét dọn sinh hoạt đời sống |
| **Sport** | **3.993** | Các động tác bóng rổ, sút bóng, nhảy xa, đấm bốc, khởi động thể thao |
| **Other** | **2.081** | Động tác đóng thế (stunts), ngã tự do, tư thế bước đi dị thường |

> [!TIP]
> Việc sở hữu tỷ lệ rất cao các động tác **Locomotion (52.3%)** và **Interactions (10.3%)** biến BONES-SEED thành nguồn tài nguyên lý tưởng nhất để huấn luyện bài toán **Tracking Controller** kết hợp **Mobile Manipulation** trên robot hình người.

---

## 5. Quy Trình Hậu Kỳ, Gán Nhãn & Tăng Cường Dữ Liệu

Sau khi ghi nhận tín hiệu đám mây điểm (point cloud) thô từ camera Vicon, dữ liệu trải qua quy trình tiền xử lý nghiêm ngặt:

```text
┌─────────────────────────────────────────────────────────────┐
│  Raw Marker Cloud (Đám mây điểm thô từ camera Vicon 120Hz)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Làm Sạch & Khử Rung (Bù đắp frame mất dấu, khử rung quang) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Khớp Khung Xương (Khớp vào mô hình nhân trắc học hoàn chỉnh)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Gán Nhãn Ngữ Nghĩa (Gán 6 câu mô tả ngôn ngữ & timestamp)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Tăng Cường Đối Xứng (Lật gương trục dọc - 71.088 mirrored)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Xuất Đa Định Dạng: SOMA (77D) / SMPL (24D) / Unitree G1    │
└─────────────────────────────────────────────────────────────┘
```

### 5.1. Khử nhiễu & Khớp khung xương (Marker Cleaning & Solving)
* Lọc bỏ nhiễu quang học, bù đắp các frame bị mất dấu (occlusions) khi diễn viên giấu tay hoặc ngồi xổm.
* Khớp dữ liệu marker vào mô hình khung xương nhân trắc học hoàn chỉnh, đảm bảo không có hiện tượng gãy khớp hay biến dạng cơ học phi tự nhiên.

### 5.2. Gán nhãn ngữ nghĩa (Semantic & Language Annotation)
* **Natural Language Captions**: Mỗi chuỗi động tác được gán **lên tới 6 câu mô tả ngôn ngữ tự nhiên chi tiết** (ví dụ: *"Một người bước nhanh về phía trước 3 bước, cúi người nhặt chiếc chai nhựa bằng tay phải rồi đứng thẳng dậy"*).
* **Temporal Segmentation**: Đánh dấu chính xác mốc thời gian (*Timestamp*) của từng phân đoạn hành động (chuẩn bị $\rightarrow$ bắt đầu chuyển động $\rightarrow$ thực hiện tương tác $\rightarrow$ hồi phục vị trí cân bằng ban đầu).

### 5.3. Tăng cường dữ liệu (Mirror Augmentation)
* Từ 71.132 clip gốc, hệ thống tự động sinh thêm **71.088 clip đối xứng qua trục dọc (Mirroring)**.
* Đảm bảo tính cân bằng tuyệt đối giữa các hành động thuận tay trái và thuận tay phải, loại bỏ thiên lệch (bias) khi huấn luyện AI.

---

## 6. Chuyển Đổi Sang Khung Xương Robot (Retargeting)

Dữ liệu cuối cùng được cung cấp ở 3 định dạng phục vụ trực tiếp cho AI và Robotics:

1. **SOMA Proportional & Uniform (BVH)**:
   * Chuẩn khung xương 77 khớp (bao gồm bàn tay và cơ mặt) do NVIDIA phát triển.
   * Bản *Proportional* giữ nguyên tỷ lệ cơ thể diễn viên; bản *Uniform* chuẩn hóa kích thước xương để đưa vào mạng nơ-ron xử lý theo batch.
2. **SMPL (Standard Human Mesh Model)**:
   * Chuẩn 24 khớp kinh điển trong thị giác máy tính, dùng làm đầu vào cho `Human Encoder` trong GEAR-SONIC.
3. **Unitree G1 Format (CSV / MuJoCo)**:
   * Chuyển vị hình học và động học nghịch (IK) từ chuyển động người sang góc quay của **29 khớp động cơ trên robot Unitree G1**.
   * Đã được kiểm tra giới hạn góc quay vật lý (joint limits) và vận tốc quay an toàn để robot không bị tự va chạm (self-collision) khi mô phỏng trong MuJoCo / Isaac Lab.

---

## 7. Tổng Kết: Vai Trò Trong Dự Án GEAR-SONIC

Dữ liệu BONES-SEED giải quyết điểm nghẽn lớn nhất trong việc phát triển robot hình người: **sự thiếu hụt dữ liệu chuyển động tự nhiên quy mô lớn**. 

Bằng cách huấn luyện mô hình **SONIC Universal Tokenizer** trên 142.220 chuyển động này, robot không cần phải học từng kỹ năng rời rạc (như bộ điều khiển đi bộ riêng, leo cầu thang riêng, cúi nhặt đồ riêng), mà sở hữu một **không gian tiềm ẩn thống nhất (Universal Latent Space)** biểu diễn toàn bộ hành vi vận động của con người một cách tự nhiên và bền bỉ.
