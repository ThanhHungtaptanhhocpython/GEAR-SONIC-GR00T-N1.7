# Báo cáo paper: Real-Time Execution of Action Chunking Flow Policies (RTC)

- **Tác giả:** Kevin Black, Manuel Y. Galliker, Sergey Levine (Physical Intelligence, UC Berkeley)
- **Hội nghị:** NeurIPS 2025. arXiv [2506.07339v2](https://arxiv.org/abs/2506.07339) (bản 05/12/2025)
- **Code (phần mô phỏng):** https://github.com/Physical-Intelligence/real-time-chunking-kinetix
- **Video:** https://pi.website/research/real_time_chunking
- **File gốc trong repo:** `2506.07339v2.pdf`

---

## 1. Tóm tắt nhanh

VLA lớn (π0, π0.5, OpenVLA, GR00T...) mất từ vài chục đến vài trăm ms cho mỗi lần suy luận, lâu hơn nhiều so với chu kỳ điều
khiển (20 ms ở 50 Hz). Action chunking giải quyết một phần, nhưng để lại hai vấn đề:

1. **Chạy đồng bộ** (chạy hết chunk, dừng chờ model sinh chunk mới): robot bị khựng giữa các chunk, chậm hơn và lệch phân bố
   so với dữ liệu train.
2. **Chạy bất đồng bộ kiểu ngây thơ** (sinh chunk mới song song rồi đổi ngay khi có): chunk mới có thể chọn một "kiểu" (mode)
   khác chunk cũ, gây nhảy đột ngột ở điểm nối.

**Real-time chunking (RTC)** coi việc sinh chunk mới là bài toán **inpainting**. Các action chắc chắn sẽ được thực thi trong lúc
model đang suy luận thì được **"đóng băng"** theo chunk cũ. Phần còn lại được sinh sao cho nối tiếp mượt với phần đã đóng băng.
Việc này làm bằng guidance lúc suy luận trong vòng lặp flow matching, **không cần train lại**, áp dụng được cho mọi policy dạng
diffusion hoặc flow.

Kết quả chính:
- Trên benchmark mô phỏng mới (12 task động trong Kinetix), RTC tốt nhất ở mọi mức trễ, và càng trễ thì khoảng cách càng lớn.
- Trên robot thật với π0.5 (6 task hai tay, 480 episode, khoảng 28 giờ chạy robot), RTC có throughput cao nhất và **không suy
  giảm** khi thêm trễ +100 ms, +200 ms. Chạy đồng bộ giảm tuyến tính; temporal ensembling thì rung tới mức robot dừng bảo vệ.
- Chuyển động nhanh hơn chạy đồng bộ khoảng 20%, và mượt hơn mọi baseline (Hình 1). Làm được cả task cần độ chính xác cao như
  quẹt diêm thắp nến, với độ trễ trên 300 ms.

---

## 2. Bối cảnh và bài toán

### 2.1 Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $\pi(\mathbf{A}_t \mid \mathbf{o}_t)$ | Policy action chunking, $\mathbf{A}_t = [\mathbf{a}_t, \dots, \mathbf{a}_{t+H-1}]$ |
| $H$ | *Prediction horizon*: độ dài chunk model sinh ra |
| $s$ | *Execution horizon*: số action thực sự chạy trước khi đổi sang chunk mới, $s \le H$ (thường $s \approx H/2$) |
| $\Delta t$ | Chu kỳ điều khiển (20 ms ở 50 Hz) |
| $\delta$ | Thời gian model sinh một chunk |
| $d = \lfloor \delta / \Delta t \rfloor$ | *Inference delay*: số bước điều khiển trôi qua từ lúc nhận quan sát tới lúc có chunk mới |
| $\mathbf{a}_{t' \mid t}$ | Action thứ $t'-t$ của chunk $\mathbf{A}_t$ sinh từ $\mathbf{o}_t$ |

Policy được train bằng conditional flow matching. Sinh chunk bằng cách lấy noise $\mathbf{A}^0_t \sim \mathcal{N}(0, I)$ rồi
tích phân trường vận tốc từ $\tau = 0$ tới $1$ trong $n$ bước:

$$\mathbf{A}_t^{\tau + 1/n} = \mathbf{A}_t^{\tau} + \tfrac{1}{n}\, \mathbf{v}_\pi(\mathbf{A}_t^{\tau}, \mathbf{o}_t, \tau)$$

### 2.2 Vì sao độ trễ là vấn đề

- Nếu $\delta \le \Delta t$ thì không có vấn đề gì, nhưng VLA hiện đại hầu như không đạt được. Ví dụ trong paper: π0 (3B) chỉ
  riêng prefill KV cache đã mất 46 ms trên RTX 4090, trong khi mục tiêu là 50 Hz. Suy luận từ xa cộng thêm 13 ms mạng trong
  điều kiện lý tưởng, thực tế có thể trên 20 ms. OpenVLA 7B đã tối ưu vẫn mất ít nhất 321 ms trên A100.
- **Chạy đồng bộ** (mặc định ở π0, π0.5, OpenVLA-OFT...): dừng ở cuối mỗi chunk để chờ. Robot vừa chậm vừa có động lực học khác
  lúc train (khựng, tăng tốc lại).
- **Chunk dài** thì phản ứng chậm với thay đổi. **Chunk ngắn** thì hay nhảy mode ở điểm nối.
- **Bất đồng bộ ngây thơ:** bắt đầu suy luận ở bước $s-d$ để chunk mới sẵn sàng đúng lúc cần. Điều kiện thời gian thực được đảm
  bảo khi $d \le H - s$. Nhưng policy không biết robot sẽ làm gì trong $d$ bước đó, nên điểm nối giữa $\mathbf{a}_{s-1|0}$ và
  $\mathbf{a}_{s|s-d}$ có thể gián đoạn tuỳ ý. Trễ càng lớn càng tệ.
- **Temporal ensembling** (lấy trung bình các chunk chồng nhau, Zhao et al. / ACT): giảm gia tốc nhưng **trung bình của hai
  action hợp lệ không nhất thiết hợp lệ**. Ví dụ ở Hình 2: chunk cũ định đi *trên* vật cản, chunk mới định đi *dưới*, trung bình
  thì đâm thẳng vào vật cản.

---

## 3. Phương pháp

### 3.1 Inpainting lúc suy luận với flow matching

Ý tưởng: $d$ action đầu của chunk mới **không thể dùng được**, vì tới lúc chunk mới xong thì robot đã chạy qua các bước đó theo
chunk cũ. Vậy nên cố định chúng bằng giá trị của chunk cũ và "tô" phần còn lại cho khớp, giống inpainting ảnh.

Tác giả dùng phương pháp inpainting không cần train của Pokle et al., dựa trên pseudoinverse guidance (ΠGDM). Ở mỗi bước khử
nhiễu, cộng thêm một số hạng gradient vào trường vận tốc:

$$\mathbf{v}_{\Pi GDM}(\mathbf{A}^\tau_t, \mathbf{o}_t, \tau) = \mathbf{v}(\mathbf{A}^\tau_t, \mathbf{o}_t, \tau) + \min\!\left(\beta, \frac{1-\tau}{\tau \cdot r_\tau^2}\right) \left(\mathbf{Y} - \widehat{\mathbf{A}^1_t}\right)^\top \mathrm{diag}(\mathbf{W})\, \frac{\partial \widehat{\mathbf{A}^1_t}}{\partial \mathbf{A}^\tau_t}$$

trong đó
- $\widehat{\mathbf{A}^1_t} = \mathbf{A}^\tau_t + (1-\tau)\,\mathbf{v}(\mathbf{A}^\tau_t, \mathbf{o}_t, \tau)$: ước lượng chunk sạch cuối cùng
  (nhảy thẳng một bước tới $\tau = 1$),
- $r_\tau^2 = \dfrac{(1-\tau)^2}{\tau^2 + (1-\tau)^2}$,
- $\mathbf{Y}$: mục tiêu (phần còn lại của chunk cũ, pad phải cho đủ độ dài $H$), $\mathbf{W}$: mask/trọng số theo thời gian.

Số hạng guidance là một **vector-Jacobian product**, tính bằng một lần backprop qua mạng vận tốc ở mỗi bước khử nhiễu.

**Đóng góp riêng: kẹp trọng số guidance bằng $\beta$.** Hệ số $\frac{1-\tau}{\tau r_\tau^2}$ tiến tới vô cùng khi $\tau \to 0$.
Inpainting ảnh thường dùng khoảng 100 bước khử nhiễu nên không sao. Điều khiển robot chỉ dùng khoảng 5 bước, và không kẹp thì
chunk bị phân kỳ (Phụ lục A.2). Paper chọn $\beta = 5$. Với $n = 5$, mọi $\beta \ge 4.25$ chỉ kẹp đúng bước đầu tiên
($\tau = 0$).

### 3.2 Soft masking để nối chunk liền mạch hơn

Nếu chỉ đóng băng $d$ bước đầu (**hard masking**, $\mathbf{W}$ gồm 0 và 1) thì khi $d$ nhỏ, tín hiệu guidance yếu. Chunk mới
vẫn có thể đổi chiến lược ngay sau vùng đóng băng (Hình 4: đường xanh dương lệch khỏi chunk cũ và đổi hướng nhanh).

Giải pháp: dùng tất cả $H - s$ action chồng lấn, với trọng số giảm dần theo hàm mũ:

$$\mathbf{W}_i = \begin{cases} 1 & i < d \\ c_i \dfrac{e^{c_i} - 1}{e - 1} & d \le i < H - s \\ 0 & i \ge H - s \end{cases}, \qquad c_i = \frac{H - s - i}{H - s - d + 1}$$

Chunk mới được chia làm 3 vùng (Hình 3):
1. **Vùng đóng băng** ($i < d$): trọng số 1, chắc chắn sẽ được thực thi theo chunk cũ.
2. **Vùng trung gian** ($d \le i < H-s$): trọng số giảm dần từ 1 về 0. Action ở xa trong tương lai thì bám chunk cũ ít hơn, vì
   chunk cũ ít chắc chắn về chúng hơn và quan sát mới có thể cần sửa.
3. **Vùng mới** ($i \ge H-s$): trọng số 0, sinh tự do vì chunk cũ không có dữ liệu ở đây.

### 3.3 Hệ thống RTC hoàn chỉnh (Algorithm 1)

Hai luồng dùng chung trạng thái, bảo vệ bằng mutex:

- **`GetAction(o_next)`**: bộ điều khiển gọi mỗi $\Delta t$. Tăng $t$, lưu quan sát mới nhất, báo cho luồng suy luận, trả về
  `A_cur[t-1]`.
- **`InferenceLoop`** (luồng nền):
  1. Chờ tới khi đã chạy ít nhất $s_{\min}$ action của chunk hiện tại.
  2. $s = t$; `A_prev` = phần chưa chạy của chunk hiện tại.
  3. **Dự báo trễ một cách thận trọng:** $d = \max(Q)$, với $Q$ là hàng đợi $b$ độ trễ gần nhất.
  4. Nhả mutex, gọi `GuidedInference(π, o, A_prev, d, s)`.
  5. Đổi sang chunk mới ngay khi có, đặt lại $t \leftarrow t - s$ để chỉ số khớp với chunk mới, ghi độ trễ thực tế vào $Q$.
- **`GuidedInference`**: tính $\mathbf{W}$ theo công thức ở 3.2, pad `A_prev` cho đủ $H$, khởi tạo noise. Ở mỗi bước khử nhiễu:
  tính $\widehat{\mathbf{A}^1}$, sai số có trọng số $\mathbf{e} = (\mathbf{A}_{prev} - \widehat{\mathbf{A}^1})^\top \mathrm{diag}(\mathbf{W})$,
  VJP $\mathbf{g}$ bằng autodiff, rồi cập nhật Euler với vận tốc đã cộng guidance.

Execution horizon thay đổi theo từng chunk: $s = \max(d, s_{\min})$, ràng buộc $d \le s \le H - d$.

---

## 4. Thí nghiệm

Ba câu hỏi: (1) RTC so với các phương pháp khác thế nào trong môi trường động, khi trễ tăng dần? (2) Soft masking quan trọng tới
đâu? (3) RTC ảnh hưởng thế nào tới hiệu năng *và* tốc độ của robot thật?

### 4.1 Benchmark mô phỏng (Kinetix)

**Lý do làm benchmark mới:** các benchmark imitation learning phổ biến gần như tĩnh, chạy chunk đồng bộ với horizon đủ dài đã
đạt gần 100%. Kinetix dùng điều khiển bằng lực, nên không có khái niệm "giữ nguyên tư thế" trong lúc chờ. Trễ suy luận vì vậy
*bắt buộc* phải chạy bất đồng bộ.

**Thiết lập:**
- 12 môi trường (10 có sẵn, 2 mới): ném, bắt, giữ thăng bằng, di chuyển... Mục tiêu: đưa vật xanh lá chạm vật xanh dương. Cộng
  noise Gaussian vào action để mô phỏng cơ cấu chấp hành không hoàn hảo, nên sửa sai vòng kín là bắt buộc.
- Dữ liệu: train expert bằng RPO (6 seed mỗi môi trường), sinh 1M transition.
- Policy: flow matching, $H = 8$, MLP-Mixer 4 lớp, 32 epoch. Mỗi điểm dữ liệu là 2048 rollout, trễ mô phỏng $d \in [0, 4]$.

**Baseline:**
- *Naive async*: không quan tâm chunk cũ, có chunk mới là đổi.
- *Bidirectional Decoding (BID)*: rejection sampling, lấy batch $N = 32$, mode size $K = 3$, cần thêm một model yếu.
- *Temporal Ensembling (TE)*: trung bình tất cả dự đoán cho cùng một timestep.

**Kết quả (Hình 5):**
- **TE kém ở mọi mức, kể cả $d = 0$.** Môi trường đa mode, trung bình các action hợp lệ không cho action hợp lệ.
- **RTC bền nhất với trễ**, hơn BID, và khoảng cách tăng khi $d$ tăng. BID còn tốn tính toán hơn nhiều (64 chunk mỗi bước: 32
  từ model mạnh, 32 từ model yếu).
- **Hard masking kém hơn soft masking**, rõ nhất khi $d$ nhỏ, đúng như lập luận ở 3.2.
- **Theo execution horizon ($d = 1$ cố định):** chỉ RTC và BID tăng đều khi giảm $s$, tức tận dụng được việc cập nhật thường
  xuyên để sửa sai vòng kín. Naive async lại tệ đi khi $s$ quá nhỏ, vì nối chunk nhiều lần hơn.

### 4.2 Robot thật

**Thiết lập:**
- Base policy π0.5, $H = 50$, $\Delta t = 20$ ms (50 Hz), $n = 5$ bước khử nhiễu. Robot hai tay, mỗi tay 6 DoF, gripper hàm
  song song. Điều khiển vị trí.
- Latency model: 76 ms (baseline), 97 ms (RTC). Suy luận qua LAN thêm 10–20 ms, nên $d \approx 6$ với RTC. Thêm trễ nhân tạo
  +100 ms ($d \approx 11$) và +200 ms ($d \approx 16$) để giả lập model lớn hơn hoặc server cloud ở xa.

**6 task** (điểm = số bước con hoàn thành):

| Task | Số bước con | Giới hạn thời gian | Nội dung |
|---|---|---|---|
| Light candle | 5 | 40 s | Cầm diêm và hộp diêm, quẹt diêm, thắp nến, bỏ que diêm vào bát |
| Plug ethernet | 6 | 120 s | Cầm đầu cáp, xoay lại, cắm vào rack server, lặp với đầu còn lại |
| Make bed (mobile) | 3 | 200 s | Kéo góc chăn và 2 gối từ cuối giường lên đầu giường |
| Shirt folding | 1 | 300 s | Gấp áo sơ mi đã trải phẳng |
| Batch folding | 4 | 300 s | Lấy đồ nhàu từ thùng, trải phẳng, gấp, xếp lên chồng |
| Dishes in sink (mobile) | 8 | 300 s | Chuyển 4 đồ vật khác nhau từ quầy vào bồn rửa |

10 lần thử cho mỗi task × phương pháp × mức trễ: **480 episode, khoảng 28 giờ chạy robot**.

**Baseline:**
- *Synchronous*: chạy $s = 25$ action rồi dừng chờ chunk mới. Đây là mặc định của π0, π0.5, OpenVLA-OFT, FAST.
- *TE, sparse*: chạy $s = 25$ và suy luận song song, có ensemble phần chồng lấn.
- *TE, dense*: suy luận liên tục với $s = d$, gần với TE gốc của ACT nhất.
- Không so với BID ngoài đời thật vì trong mô phỏng BID thua RTC mà tốn tính toán hơn nhiều. Với π0.5 và batch 16, BID có
  latency gấp 2.3 lần RTC.

**Kết quả (Hình 6):**
- **Throughput** (tỷ lệ task hoàn thành chia thời gian episode, đo cả tốc độ lẫn chất lượng): RTC cao nhất ở mọi mức trễ, có ý
  nghĩa thống kê ở +100 ms và +200 ms.
- **RTC không suy giảm khi thêm trễ** (throughput khoảng 0.8–0.9 task/phút ở cả 3 mức). Synchronous giảm tuyến tính (khoảng
  0.72 xuống 0.6 xuống 0.5). **Cả hai biến thể TE không chạy được ở +100/+200 ms**: rung mạnh làm robot kích hoạt dừng bảo vệ.
- **Nhanh hơn không chỉ nhờ bỏ thời gian chờ.** Vẽ tiến độ theo *số bước điều khiển* (đã loại thời gian dừng chờ), RTC vẫn hoàn
  thành nhiều hơn sớm hơn: ít sai và ít phải thử lại hơn.
- **Light candle** (task cần chính xác nhất và là task duy nhất không cho thử lại): RTC hơn rõ về điểm cuối cùng. **Bed making**
  (task khó nhất, khó ở phần xử lý gối) cũng vậy.

### 4.3 Phụ lục: ablation và đo latency

**Kẹp $\beta$ (A.2, Hình 7):**
- Mô phỏng: tăng $\beta$ quá 5 không cải thiện thêm.
- Robot thật với $d = 15$, $n = 5$: $\beta$ càng lớn thì chunk càng phân kỳ khỏi chunk cũ, gia tốc cực đại tăng từ khoảng 1.3
  rad/s² ($\beta = 5$) lên khoảng 3 rad/s² ($\beta = 150$). Với $n = 100$ bước thì không có vấn đề này.
- Cách khác để tránh trọng số vô cùng ở $\tau = 0$ là bắt đầu khử nhiễu từ $\tau > 0$ (như Pokle et al.). Tác giả chưa thử.

**Latency trên RTX 4090, bf16, $n = 5$ (A.3):**

| Phương pháp | Latency |
|---|---|
| π0.5 gốc | 76 ms |
| **RTC** | **97 ms** |
| BID N=16 (không forward model) | 115 ms |
| BID N=16 (dùng chung backbone) | 169 ms |
| BID N=16 (đầy đủ) | 223 ms |

Phân rã: SigLIP 18 ms, prefill Gemma 2B 44 ms, 5 bước khử nhiễu 14 ms (gốc) hoặc **35 ms (RTC)**. Tức RTC làm mỗi bước khử nhiễu
chậm khoảng 2.5 lần vì phải backprop qua action expert. Tính cả hệ thống: khoảng 109 ms trên workstation (mạng 7 ms), khoảng
139 ms trên robot di động (mạng 21 ms, resize ảnh 11 ms trên CPU NUC).

**Lịch trọng số soft mask (A.4, Hình 8):** exponential decay tốt nhất, linear decay sát nút. Không decay và hard masking kém hơn.

**So với inpainting kiểu Diffuser** (ghi đè trực tiếp phần action đã biết ở mỗi bước khử nhiễu, không cần gradient): rẻ hơn và
vẫn có ích so với naive async, nhưng thua cách dùng guidance của RTC.

**Siêu tham số (A.5):**

| Tham số | Ý nghĩa | Mô phỏng | Robot thật |
|---|---|---|---|
| $n$ | Số bước khử nhiễu | 5 | 5 |
| $H$ | Prediction horizon | 8 | 50 |
| $s_{\min}$ | Execution horizon tối thiểu | – | 25 |
| $\beta$ | Ngưỡng kẹp guidance | 5 | 5 |
| $b$ | Kích thước buffer độ trễ | – | 10 |

**Tài nguyên:** mô phỏng dùng tối đa 6×H100 (train expert khoảng 4 giờ, train policy khoảng 1.5 giờ). Fine-tune π0.5 khoảng
24 giờ trên 8×H100. Suy luận ngoài đời thật trên một RTX 4090.

---

## 5. Các hướng liên quan (tóm tắt mục 5)

- **Giảm latency** (consistency policy, streaming diffusion policy, parallel decoding cho OpenVLA, tối ưu transformer): hữu ích
  nhưng không giảm được dưới một lần forward. Nếu một forward vẫn lâu hơn $\Delta t$ thì vẫn cần RTC.
- **Inpainting/guidance** (ΠGDM, Pokle et al., Diffuser): RTC là công trình đầu tiên đưa inpainting/guidance vào *điều khiển
  thời gian thực*.
- **MPC:** cũng lập kế hoạch trên horizon trượt và dùng kế hoạch cũ để khởi động nóng, nhưng cần mô hình động lực học và hàm
  chi phí tường minh.
- **VLA phân cấp System 1 / System 2** (GR00T N1, Figure Helix): System 2 lớn chạy chậm, System 1 nhẹ chạy nhanh. Hướng này
  trực giao với RTC và phải đánh đổi: System 1 phải nhỏ, cần công thức train riêng.
- **Bidirectional Decoding (BID):** gần nhất. Giữ liên tục bằng rejection sampling, không xét độ trễ, tốn tính toán hơn nhiều.

---

## 6. Hạn chế (theo tác giả)

- Tăng chi phí tính toán so với lấy mẫu trực tiếp từ policy (thêm VJP ở mỗi bước khử nhiễu, khoảng +21 ms với π0.5).
- Chỉ dùng được cho policy diffusion hoặc flow; policy autoregressive (FAST, OpenVLA) không dùng được.
- Thí nghiệm thật chưa có môi trường động hơn như đi bằng chân (có trong mô phỏng, không có ngoài đời thật).

---

## 7. Nhận xét

**Điểm mạnh**
- Ý tưởng gọn, không cần train lại, cắm được vào mọi VLA dạng flow hoặc diffusion. Rất thực dụng.
- Chỉ ra đúng lý do temporal ensembling thất bại: dữ liệu đa mode, trung bình hai mode không phải một mode hợp lệ.
- Thí nghiệm thật quy mô lớn (480 episode), dùng metric throughput kết hợp tốc độ và chất lượng, có phân tích theo cả thời gian
  thật lẫn số bước điều khiển để tách ảnh hưởng của việc bỏ thời gian chờ.
- Các chi tiết kỹ thuật (kẹp $\beta$, soft mask, dự báo trễ bằng max buffer) có ablation hoặc lý giải rõ.

**Điểm cần lưu ý**
- Chi phí VJP làm mỗi bước khử nhiễu chậm khoảng 2.5 lần. Với action head lớn hơn hoặc nhiều bước khử nhiễu hơn thì tăng đáng
  kể.
- Cần truy cập vào vòng lặp khử nhiễu của model (phải sửa phía server). Không làm được nếu chỉ có API "đưa quan sát, nhận chunk".
- Mức độ bám chunk cũ bị khoá bởi lịch $\mathbf{W}$ và $\beta$. Nếu chunk cũ đang *sai* (ví dụ sắp đụng vật), RTC cũng làm
  chậm việc sửa sai ở vùng trung gian. Paper không phân tích trường hợp này.
- Baseline thật chỉ có synchronous và TE. Không so với các hướng giảm latency hoặc System 1/2.

---

## 8. Liên hệ với dự án GEAR-SONIC + GR00T N1.7

### 8.1 RTC nhắm đúng lỗi đang gặp

Memo mục 5.2g ghi nhận B-9000 hỏng ở lon Coca nằm quanh ranh giới đổi kiểu gắp (yaw khoảng 90–100°): **tay đổi ý giữa các
chunk**, từ tiếp cận ngang sang hạ từ trên xuống, đụng lon làm rơi. Nguyên nhân đã chẩn đoán: *flow matching lấy noise mới ở mỗi
lần gọi model (mỗi 0.3 s), nên các chunk liên tiếp có thể rơi vào hai kiểu khác nhau*. Đây chính là hiện tượng "mode jumping"
mà RTC xử lý (Hình 2 của paper).

- `--replan-steps 30` không giúp (7/10 so với 9/10): đúng như paper dự đoán, execution horizon dài chỉ giảm *số lần* đổi mode,
  không loại bỏ nó, mà còn làm sửa sai chậm đi.
- Ý tưởng "cố định seed noise" trong memo là một cách giảm đổi mode, nhưng không đảm bảo: quan sát mới vẫn có thể đẩy cùng một
  noise sang mode khác. RTC thì ép chunk mới bám phần chồng lấn của chunk cũ.

### 8.2 Cơ chế trộn hiện tại chưa phải RTC

[bridge/closed_loop_pipeline.py:237-249](../bridge/closed_loop_pipeline.py#L237-L249) có chú thích "RTC Smooth Inpainting",
nhưng thực chất là **crossfade hậu kỳ 4 bước** (trọng số $\cos^2$) giữa chunk cũ và chunk mới *sau khi* model đã sinh xong. Về
bản chất đây là một dạng temporal ensembling trên đoạn ngắn:
- Làm mượt được điểm nối 4 bước đầu (0.08 s ở 50 Hz).
- **Không** ngăn chunk mới chọn kiểu gắp khác. Sau 4 bước, robot vẫn đi theo mode mới. Trong đoạn trộn, trung bình của hai mode
  có thể là action không hợp lệ (paper: TE kém ngay cả khi $d = 0$).

Nên đổi tên chú thích cho khỏi nhầm, và nếu muốn chống đổi mode thì phải can thiệp *bên trong* vòng khử nhiễu.

### 8.3 Nếu cài RTC cho GR00T N1.7: những điểm cần xử lý

1. **Phải sửa ở server**, trong vòng lặp flow matching của action head (DiT) của GR00T, vì cần $\mathbf{v}_\pi$ ở từng bước và
   cần backprop để lấy VJP. Client chỉ cần gửi thêm `A_prev` (phần còn lại của chunk cũ) và $d$, $s$ theo mỗi request.
2. **Không gian action:** $\mathbf{Y}$ phải ở **đúng không gian model sinh ra**, tức action đã chuẩn hoá và ở dạng *relative*
   (embodiment `real_g1_relative_eef_relative_joints`). Server hiện đổi relative sang absolute trước khi trả về, nên phải:
   lấy phần còn lại của chunk cũ ở dạng absolute, đổi lại thành relative **theo state của quan sát mới**, rồi chuẩn hoá. Nếu làm
   sai bước này, guidance sẽ kéo về một mục tiêu sai.
3. **Đổi đơn vị thời gian:** chunk 40 khung ở 25 Hz, còn bộ điều khiển chạy 50 Hz và nội suy tuyến tính. Replan mỗi 15 bước 50 Hz
   = 7.5 khung model, không chia hết. Nên chọn `replan_steps` chẵn (ví dụ 14 hoặc 16) để $s$ nguyên theo khung model, và tính
   $d$, $s$, $H$ theo khung 25 Hz ($H = 40$).
4. **Sim hiện chạy đồng bộ** (sim dừng chờ `_predict_chunk`), nên trong sim $d = 0$. RTC vẫn có ích nhờ soft masking: vùng
   trung gian $0 \le i < H - s$ vẫn bám chunk cũ với trọng số giảm dần, đúng thứ cần để chống đổi kiểu gắp. Với $d = 0$ và
   $s = 8$ khung: 32 khung đầu có trọng số từ khoảng 1 giảm về 0, 8 khung cuối sinh tự do.
5. **$\beta$ và số bước khử nhiễu:** GR00T dùng ít bước (GR00T N1 mặc định 4; repo này không cấu hình, cần xem
   `num_inference_timesteps` trên server). Với $n = 4$, hệ số $\frac{1-\tau}{\tau r_\tau^2}$ ở
   $\tau = 0.25, 0.5, 0.75$ lần lượt khoảng 3.3, 2.0, 3.3, nên $\beta = 5$ chỉ kẹp bước $\tau = 0$, giống paper. Cần kiểm tra quy
   ước $\tau$ của GR00T (0 là noise, 1 là action sạch) trùng với paper trước khi dùng công thức.
6. **Chi phí:** backprop qua action head ở mỗi bước khử nhiễu. Theo paper, khoảng +2.5 lần thời gian phần khử nhiễu. Backbone
   VLM (Eagle) không bị ảnh hưởng vì chỉ backprop qua action head, với điều kiện cache đặc trưng VLM và tách khỏi đồ thị gradient.
7. **Đánh đổi cần đo:** bám chunk cũ làm giảm đổi mode, nhưng cũng làm chậm sửa sai ở vùng trung gian (ví dụ 100049, tay quét đổ
   lon bên cạnh). Nên A/B trên cùng tập `ik_test` và `swap_test`, báo cả tỷ lệ thành công, số lần đổi kiểu gắp, và gia tốc cực
   đại của khớp (như Hình 7 phải dưới của paper).

### 8.4 Thứ tự thử đề xuất

1. Bản rẻ trước: **inpainting kiểu Diffuser** (ghi đè phần đầu chunk bằng chunk cũ, đã thêm noise đúng mức $\tau$, ở mỗi bước khử
   nhiễu). Không cần gradient nên dễ cài vào server. Paper cho thấy bản này vẫn hơn naive async.
2. Nếu thấy có tác dụng với lỗi đổi kiểu gắp thì cài RTC đầy đủ (ΠGDM + soft mask + $\beta = 5$), so sánh với bản 1.
3. Khi chạy trên robot thật hoặc server Colab qua tunnel (trễ mạng lớn), thêm luồng suy luận bất đồng bộ như Algorithm 1, dự báo
   $d$ bằng max của buffer độ trễ.
