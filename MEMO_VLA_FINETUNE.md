# Memo: GR00T N1.7 thật + fine-tune cho nhiệm vụ phân loại lon (Unitree G1, MuJoCo)

Ghi lại các quyết định, phát hiện và trạng thái công việc trong các phiên làm việc ngày 22–23/09/2026
(phiên 23/09: IK online cho `collect_demos.py`, mục 4.2).
Chưa có commit nào cho các thay đổi dưới đây (nhánh `feat/smooth-vla-trajectory`).

---

## 1. Xuất phát điểm

- Chế độ `--policy-mode remote` **chưa từng gọi GR00T thật**. `_init_model()` cũ nạp model bằng `AutoModelForVision2Seq`
  (luôn lỗi rồi âm thầm chuyển sang giả lập), và `predict()` luôn trả về quỹ đạo mẫu `scripted_sorting_chunk`.
- Checkpoint `GR00T-N1.7-3B` (base) **không có embodiment SONIC**. Embodiment G1 duy nhất có thống kê là
  `real_g1_relative_eef_relative_joints` (tag `REAL_G1`):
  - video `ego_view`, 2 khung hình (`delta_indices [-20, 0]`);
  - state: `left/right_wrist_eef_9d` (9+9), `left/right_hand` (7+7), `left/right_arm` (7+7), `waist` (3);
  - action: như state + `base_height_command` (1) + `navigate_command` (3), chunk 40 bước. EEF và tay dùng action relative.
- `UNITREE_G1_SONIC` (action = motion token 64 chiều) là tag posttrain, cần checkpoint fine-tune riêng. NVIDIA chưa công bố
  checkpoint hay dataset nào cho tag này.
- Máy local không có GPU, nên server chạy trên Colab.

Hai hướng đã thống nhất: **A** chạy zero-shot để kiểm chứng đường ống; **B** tự sinh dữ liệu sim rồi fine-tune.

---

## 2. Hướng A: chạy GR00T thật zero-shot (xong)

### Đã làm
- `bridge/real_g1_adapter.py` (mới): tạo state REAL_G1 từ sim (pose cổ tay 9D trong khung pelvis, góc tay/ngón/eo) và đổi
  action của model thành 43 góc điều khiển, nội suy lên 50 Hz. Có thêm `action_from_ctrl` để ghi dataset.
- `colab_gr00t_server.py`: `--policy gr00t` (mặc định) nạp model thật qua `Gr00tPolicy` của Isaac-GR00T; `--policy scripted`
  giữ quỹ đạo mẫu cũ. Có `--embodiment-tag` (mặc định REAL_G1; checkpoint fine-tune dùng `new_embodiment`). Có warm-up.
  Tự tắt flash-attn trên GPU trước Ampere (T4). Nạp model lỗi thì dừng hẳn, không âm thầm giả lập.
- Client (`bridge/gr00t_policy_client.py`): gửi 2 khung hình JPEG 256×384, gửi list/bytes thay vì ndarray (local numpy 2.2,
  Isaac-GR00T ghim numpy 1.26). Server model thật lỗi thì robot giữ tư thế, không lẫn quỹ đạo mẫu.
- Pipeline (`bridge/closed_loop_pipeline.py`): lịch sử khung hình, chunk góc khớp, `--record-vla` ghi state/action/ảnh ra `.npz`.
- WBC (`bridge/sonic_wbc_controller.py`): `track_joint_targets` (bám góc khớp trực tiếp, chân giữ tư thế đứng) và
  `last_target` (mục tiêu trước bộ lọc làm mịn).
- Launcher: `--server-timeout-ms`, `--vla-action-hz`, `--record-vla`.
- **Camera `ego_view` nghiêng 24° → 50°** (`sim/unitree_g1/g1_with_hands.xml`). Góc cũ chỉ thấy sàn và mép lon, ảnh lệch hẳn
  phân phối huấn luyện; đây là nguyên nhân chính khiến robot giơ tay kỳ lạ khi chạy zero-shot.

### Quy ước suy ra từ `statistics.json` (chưa có tài liệu xác nhận)
1. Thứ tự 7 khớp ngón tay theo alphabet: `index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2`.
2. Khung EEF = khung `wrist_yaw_link` nhân thêm phép xoay −90° quanh trục y (`EEF_ROT_OFFSET`); rot6d = 2 hàng đầu ma trận quay
   (đã xác nhận quy ước rot6d trong code Isaac-GR00T).
3. Tần số dữ liệu REAL_G1 khoảng 20 Hz (suy từ DROID `[-15, 0]` ở 15 Hz).

### Lỗi đã gặp trên Colab và cách sửa
| Lỗi | Nguyên nhân | Sửa |
|---|---|---|
| `.venv/bin/python: No such file` | trình cài uv đặt `uv` ở `/usr/local/bin`, notebook gọi cứng `~/.local/bin/uv` | gọi `uv` qua `PATH` |
| `Invalid zip file structure` (torchcodec wheel) | clone với `GIT_LFS_SKIP_SMUDGE=1` nên wheel chỉ là con trỏ LFS | `git lfs pull --include="scripts/deployment/dgpu/wheels/*"` |
| `Cannot download ... Cosmos-Reason2-2B` | backbone gated | xin quyền trên HuggingFace; Bước 3 kiểm tra token và quyền ngay |
| Output `%%bash` không hiện | Colab chỉ in output `%%bash` khi cell chạy xong | train chạy nền + cell theo dõi |

### Bảo mật
Notebook cũ hard-code HF token `hf_JbmJ...`, token này đã nằm trong commit `26398c5` và repo có remote GitHub. Đã gỡ khỏi notebook;
**token phải được thu hồi trên HuggingFace**. Notebook mới đọc token từ Colab Secrets (`HF_TOKEN`).

---

## 3. Nguồn dữ liệu đã khảo sát

- Không có dataset mở nào đúng bài toán "G1 phân loại lon Coca/Pepsi".
- `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` → `gr1_arms_only.CanSort`: đúng nhiệm vụ (1000 episode sim, 20 fps,
  1.34 GB) nhưng là robot **Fourier GR1**, action không dùng cho G1 được.
- `unitreerobotics/G1_Dex3_*` (PickBottle 202 ep, ObjectPlacement 210 ep, …): đúng robot G1 + Dex3, 28 khớp, 30 fps, ảnh thật,
  định dạng LeRobot **v3.0** (GR00T cần v2). Có thể chuyển sang layout REAL_G1 (đổi thứ tự ngón, tính EEF bằng FK).
- Nhánh SONIC: encoder `checkpoints/GEAR-SONIC/model_encoder.onnx` (vào 1762 chiều, ra token 64) ở chế độ `g1` chỉ cần quỹ đạo
  29 khớp + hướng thân, nên về nguyên tắc tự tính được motion token. **Chưa kiểm chứng** cách ghép vector 1762 chiều
  (cần đọc code GR00T-WholeBodyControl). Tạm gác.
- Kết luận: nguồn chính là **demo sim tự sinh** (không cần robot thật), lúc đầu từ quỹ đạo mẫu, nay bằng IK online (mục 4.2).

---

## 4. Hướng B: sinh dữ liệu sim

### 4.1. Mốc thử: quỹ đạo mẫu cố định (xong, đã fine-tune ở mục 5)
- `collect_demos.py` (bản đầu): chạy quỹ đạo mẫu, ghi ở **25 Hz**. Mỗi khung gồm:
  - ảnh `ego_view` 384×256 h264;
  - state 49 chiều;
  - action 53 chiều: mục tiêu khớp trước bộ lọc làm mịn, EEF tính bằng FK.

  Xuất ra GR00T LeRobot v2, kèm `meta/modality.json` và `meta/collection.jsonl` (seed, cfg).
  - Ngẫu nhiên hoá: lon ±2 cm, khay đích ±2 cm, nhiễu khớp tay σ = 0.05 rad.
  - Chỉ giữ episode thành công **và** phát lại thành công qua đường chạy model (loại các lần gắp sát mép).
  - Mỗi tiến trình chạy một episode rồi thoát (`maxtasksperchild=1`), vì tạo pipeline lặp lại rò rỉ ~150 MB/episode và từng làm tràn RAM.
- `verify_demos.py`: kiểm tra định dạng và phát lại action qua `action_to_ctrl` → `track_joint_targets` → sim.
- Quỹ đạo mẫu chỉ chịu được lon lệch **±1–2 cm**, vì chạy mù theo waypoint cố định. Tỉ lệ thành công khoảng 70% (Coca), 65% (Pepsi).

| Dataset | Episode | Seed | Kiểm tra |
|---|---|---|---|
| `datasets/g1_can_sort_pilot` (train, 85 MB, zip 87 MB) | 100 (50 Coca + 50 Pepsi, xen kẽ) | 0–143 | định dạng OK, phát lại 100/100 |
| `datasets/g1_can_sort_test` (test, 34 MB, zip 35 MB) | 40 (20 + 20, xen kẽ) | 100001–100056 | định dạng OK, phát lại 40/40, không trùng seed train |

Episode chẵn là Coca, lẻ là Pepsi. Biểu đồ vị trí lon: `datasets/g1_can_sort_train_test_positions.png`.
**Hạn chế:** vùng lon chỉ là ô 4×4 cm và action gần như không phụ thuộc vị trí lon, nên model học thuộc quỹ đạo hơn là học nhìn lon.

### 4.2. IK online, lon ở vùng rộng, đứng hoặc nằm nhiều hướng (xong code 23/09, chưa sinh dataset lớn)
`collect_demos.py` giờ thu demo bằng `sim/pick_place_planner.py` (IK trong `sim/arm_ik.py`) và bỏ hẳn quỹ đạo mẫu.
"IK online" nghĩa là giải động học ngược cho từng episode, từ vị trí và hướng lon đo trong sim, thay cho waypoint góc khớp chép cứng.

**Ngẫu nhiên hoá** (`sample_randomization`, tái tạo được từ task + seed + cfg):
- Tâm lon lấy trong x 0.26–0.37 m, |y| 0.03–0.19 m, mỗi lon ở phía tay gắp nó (Coca bên phải, Pepsi bên trái).
  Vùng này đúng bằng tầm với: vai ở x 0.03 m, eo cúi tối đa 0.45 rad, lon đứng chỉ với tới x ≈ 0.37.
- Mỗi lon đứng hoặc nằm (`--lying-prob`, mặc định 0.5). Lon nằm có hướng trục ngẫu nhiên 360°, nên nắp có thể quay về bất kỳ phía nào.
- Hai lon cách nhau ít nhất `--can-gap` (3 cm) và cách thành khay ≥ 1 cm. Khay đích lệch ±2 cm, nhiễu khớp tay σ 0.05 rad.
- Camera `ego_view` thấy trọn vùng này. Riêng góc gần-ngoài bị bàn tay ở tư thế ready che một phần.
- Lon nằm đứng yên trên bàn (trôi ~2 mm trong 3 s), không cần sửa XML hay đổi `condim`.

**Cách gắp** (khung cổ tay `wrist_yaw_link`: x = hướng ngón, y = pháp tuyến lòng bàn tay, z = trục mà ngón cuộn quanh):
- **Lon đứng:** gắp ngang như quỹ đạo cũ, z cổ tay thẳng đứng, ngón hướng theo hướng vai → lon, tiếp cận từ phía ngoài.
- **Lon nằm:** z cổ tay trùng trục lon, bàn tay úp từ trên xuống nhưng **xoay 35° quanh trục lon** về phía ngón tay.
  Ngón cái lúc mở chĩa theo pháp tuyến lòng bàn tay, dài tới 11.4 cm, nên úp thẳng thì đầu ngón cái nằm dưới mặt bàn 3 cm và tay bị bàn chặn lại.
  Cho phép lệch thêm tối đa 40° khi không với tới.
- **Khi đặt:** bàn tay chỉ xoay quanh trục đứng so với lúc gắp, nên lon đứng vẫn đứng, lon nằm vẫn nằm.
  Điểm đặt lệch từ tâm khay về phía robot 2 cm để đỡ phải vươn xa.

**Các đoạn** (tổng 375 bước 50 Hz = 7.5 s = 188 khung):
vươn tay 45 → hạ xuống điểm chờ 20 → tiến vào lon 35 → khép ngón 20 → nhấc 30 → mang tới khay 65 → hạ 25 → mở ngón 15
→ rút tay lên 30 → về điểm cao trên ready 35 → hạ về ready 20 → đứng yên 35.
- Đoạn hạ xuống và tiến vào lon được lập **ngược từ tư thế gắp**, nên luôn kết thúc đúng nhánh nghiệm IK của tư thế gắp.
- Đoạn tiến vào lon và hạ lon chạy bằng **servo**: mỗi bước giải lại IK theo pose pelvis đo được.
- Các đoạn đi trong khoảng trống thử nội suy khớp trước. Nếu va chạm thì chuyển sang đường Descartes có cung nhô 5 cm, rồi 12 cm.

**IK** (`ArmIK`): damped least squares trên 7 khớp tay cộng `waist_pitch`, có các cơ chế sau:
- active set cho giới hạn khớp (chỉ kẹp góc thì bộ giải kẹt ở nghiệm sai);
- thử nhiều điểm khởi đầu và phạt các khớp nằm sát giới hạn;
- số hạng tư thế của eo nặng gấp 5 lần khớp tay, để IK ưu tiên dùng tay.

**Các vấn đề đã gặp khi làm và cách sửa:**
| Hiện tượng | Nguyên nhân | Sửa |
|---|---|---|
| Ngón tay quét đổ lon lúc vươn tay | nội suy khớp giữa ready và điểm chờ đi thấp qua lon | kết thúc vươn tay ở điểm cao; kiểm tra va chạm với lon/bàn/khay |
| Tay trái không tới lon, bị chặn giữa đường | IK chọn vai khép vào trong, bắp tay lồng vào thân | giới hạn vai giạng ≥ 0.05 rad; kiểm tra tự va chạm (khép quá 0.1 rad thì gần như luôn va) |
| Khâu cổ tay chạm nhau | \|wrist_pitch\| ≥ 1.5 rad | giới hạn 1.45 rad; bỏ qua cặp wrist_roll–wrist_yaw khi kiểm tra (lồng nhau < 1 mm) |
| Tay rảnh đè lên khay/lon bên kia | eo cúi làm tay rảnh hạ ~10 cm | bảng bù khớp tay rảnh theo góc eo (giữ cổ tay đứng yên, nâng dần khi eo cúi) |
| Lon nằm: tay dừng cách lon 3–5 cm | robot ngả người làm tay lệch; ngón cái đâm xuống bàn | servo theo pose pelvis; xoay tay 35° quanh trục lon |
| Toàn thân dao động (pelvis −2° ↔ +12°, lệch 16 cm) | servo chỉnh eo để bù độ ngả, chính chuyển động eo lại làm thân lắc | servo giữ nguyên eo, chỉ chỉnh khớp tay |
| Tay kéo lon ra khỏi khay lúc về ready | mép bàn tay ở tư thế ready rất gần lon trong khay | rút tay lên 12 cm, về điểm cao trên ready rồi mới hạ thẳng xuống |
| Kiểm tra va chạm báo sai khi về ready | robot còn ngả 8° nên tính tư thế ready với pelvis đang ngả thấy tay trúng lon | đoạn về ready chỉ kiểm tra va chạm với thân robot |
| Lon đặt cách khay ~20 cm | mang lon nằm sang ngang nhanh, robot ngả 9°, hông chạm mép bàn, bàn chân trượt xoay 18° | mang chậm hơn (65 bước), đặt lệch về phía robot 2 cm; ca còn sót bị bộ lọc loại |

**Kết quả** (quét 36 seed mỗi loại, cùng vùng và cfg mặc định):
| | Coca (tay phải) | Pepsi (tay trái) |
|---|---|---|
| Lon đứng | 36/36 | 36/36 |
| Lon nằm | 28/36 (78%) | 25/36 (69%) |

- Lỗi còn lại ở lon nằm: gắp trượt; robot ngả chạm mép bàn hoặc trượt chân khi mang lon; vài hướng không có tư thế gắp.
  Episode lỗi bị loại, nên dataset hơi lệch về các hướng lon nằm dễ gắp.
  Muốn tỉ lệ lon nằm trong dataset gần 50% thì tăng `--lying-prob` lên khoảng 0.6.
- Thu thử 20 episode (12 tiến trình, 90 s): thành công 40/44 lần. `verify_demos.py` báo định dạng OK, phát lại 20/20,
  lệch khớp tay trung bình 0.031 rad. Video trông hợp lý.
- `cfg` cũ (có khoá `can_jitter`) vẫn được `sample_randomization` tái tạo đúng từng số, nên dataset pilot/test cũ vẫn verify
  (thử 6/6) và vẽ được.
- `tools/plot_can_positions.py` vẽ lon đứng thành chấm, lon nằm thành đoạn thẳng theo trục lon, kèm vùng lấy mẫu và
  histogram hướng của lon nằm.
- `collection.jsonl` ghi thêm `lying` và `palm_tilt_deg` của lon cần gắp. Cuối lần thu có in thống kê theo lon × đứng/nằm và lý do thất bại.

**Giới hạn gốc:** WBC (`track_joint_targets`) giữ chân ở góc cố định và không có điều khiển thăng bằng, nên robot ngả
theo tải ở tay (3–9°). Khi chạy model, robot cũng ngả như vậy, nên dữ liệu và triển khai vẫn khớp nhau.
Nếu cần lon ở xa hơn hoặc gắp ổn định hơn thì phải thêm điều khiển thăng bằng cho chân (điều chỉnh cổ chân/hông theo IMU).

---

## 5. Fine-tune trên Colab (xong lần thử đầu)

### Notebook `colab_gr00t_finetune.ipynb` (sinh bởi `tools/build_colab_notebooks.py`)
| Bước | Nội dung |
|---|---|
| 1–3 | GPU, cài Isaac-GR00T (ghim commit `51d4c89`), HF token |
| 4 | Mount Drive, biến môi trường (`MAX_STEPS=5000`, `SAVE_STEPS=1000`, `GLOBAL_BATCH_SIZE=32`) |
| 5–7 | Giải nén dataset, ghi `g1_can_sort_config.py`, `stats.py` + nạp thử bằng data loader (~5 phút) |
| 8 / 8b / 8c | Train **chạy nền** (`subprocess.Popen(start_new_session=True)`, log `train.log`) / theo dõi ETA + loss + VRAM / chờ xong rồi bíp |
| 9 | Đánh giá open-loop episode 0, 1 (`MPLBACKEND=Agg`) |
| 10 | Chép checkpoint mới nhất về `MyDrive/gr00t/checkpoints/g1_can_sort_pilot/` |
| 11 | Đánh giá open-loop 40 episode test + 40 episode train, bảng so sánh theo lon |

- Dừng train chủ động: `os.killpg(train_proc.pid, signal.SIGTERM)` (bấm ■ không dừng được tiến trình nền).
- Bước 11 chạy được trên runtime mới (tự lấy checkpoint trên Drive, tự tạo `stats.json`); **L4 là đủ**, không dùng T4.

### Quyết định
- **Tag `NEW_EMBODIMENT`**, không dùng `REAL_G1`: checkpoint base đã có thống kê REAL_G1 nên `set_statistics(override=False)` bỏ qua
  thống kê của dữ liệu sim mà chỉ in cảnh báo. Cái giá: projector riêng cho robot (ngăn số 10) học lại từ đầu.
- `finetune/g1_can_sort_config.py` giữ layout REAL_G1 (video `[-20, 0]`, action 40 bước, EEF/tay relative) nên client/server dùng lại.
- Phần được train (mặc định): VL LayerNorm + VL self-attention (~0.2 tỉ), DiT (~1.09 tỉ), projector (state/action encoder, decoder).
  Đóng băng: vision encoder và LLM của Cosmos-Reason2-2B. Augmentation mặc định: color jitter, state dropout 20%, crop 95%.
- `--save-only-model`: checkpoint gọn, không train tiếp được.

### Loss
Flow matching: t = (1 − s) × 0.999 với s ~ Beta(1.5, 1.0); x_t = (1 − t)ε + t·a; mục tiêu v = a − ε;
L = Σ M ⊙ (v̂ − v)² / Σ M (M che phần đệm lên 132 chiều). Action chuẩn hoá theo q01/q99 về [−1, 1]; suy luận 4 bước Euler.
Với dữ liệu gần tất định, loss có thể giảm rất sâu; loss thấp chủ yếu nói lên model học thuộc demo.

### Kết quả
- Loss ở bước 314: 0.214 (đang giảm bình thường).
- Train xong `checkpoint-5000`. Open-loop trên episode 0 (Coca) và 1 (Pepsi) — **đều là episode train**:

| Episode | MSE | MAE |
|---|---|---|
| 0 | 0.00035 | 0.0056 |
| 1 | 0.00019 | 0.0044 |
| Trung bình | 0.00027 | 0.0050 (~0.3°) |

- Bước 11 (chạy trên L4, khoảng 20 phút), cùng `checkpoint-5000`, 40 episode mỗi tập:

| Tập | Lon | MSE TB | MAE TB | MAE lớn nhất |
|---|---|---|---|---|
| train | Coca (20) | 0.000303 | 0.00509 | 0.00592 |
| train | Pepsi (20) | 0.000264 | 0.00509 | 0.00580 |
| train | cả hai (40) | 0.000284 | 0.00509 | 0.00592 |
| test | Coca (20) | 0.000320 | 0.00537 | 0.00647 |
| test | Pepsi (20) | 0.000267 | 0.00516 | 0.00600 |
| test | cả hai (40) | 0.000294 | 0.00526 | 0.00647 |

  Episode test sai nhiều nhất: 20 (Coca, 0.0065), 32 (Coca), 33 (Pepsi), 10 (Coca), 22 (Coca), đều khoảng 0.006.

  **Nhận định:** khoảng cách train–test chỉ ~3%, sai số đều giữa các episode, nên model không học thuộc trong vùng ±2 cm.
  Nhưng quỹ đạo mẫu chạy mù (action gần như không phụ thuộc vị trí lon), nên open-loop **chưa chứng minh model biết nhìn lon**;
  cần chạy closed-loop và dữ liệu có vùng lon rộng hơn.

---

## 6. Việc tiếp theo

1. ~~Chạy Bước 11~~ (xong, xem mục 5: không học thuộc trong vùng ±2 cm).
2. **Chạy thử closed-loop trong sim** với checkpoint fine-tune:
   - Colab: `colab_gr00t_server.ipynb`, Bước 5 bắt buộc chạy lại (file server có `--embodiment-tag`); Bước 6
     `MODEL_PATH = "/content/drive/MyDrive/gr00t/checkpoints/g1_can_sort_pilot/checkpoint-5000"`, `EMBODIMENT_TAG = "new_embodiment"`.
   - Local: `.venv/bin/python launch_vla_simulation.py --policy-mode remote --server-host <host> --server-port <port> --server-timeout-ms 10000 --vla-action-hz 25 --record-vla vla_log_ft.npz`
3. **Mở rộng dữ liệu**: ~~IK online, lon đứng/nằm nhiều hướng~~ (xong code, mục 4.2). Việc còn lại:
   - Sinh dataset ~400 episode: `collect_demos.py --out datasets/g1_can_sort_ik --episodes-per-task 200`, khoảng 30 phút với 12 tiến trình.
   - Sinh tập test: `--out datasets/g1_can_sort_ik_test --episodes-per-task 20 --seed 100000`.
   - Chạy `verify_demos.py`, `plot_can_positions.py`, nén zip, rồi fine-tune lại. Notebook fine-tune (Bước 4: `DATASET_ZIP`, `DATASET_DIR`, `EXPERIMENT`) đang gắn tên `g1_can_sort_pilot`, cần đổi hoặc thêm tham số.
   - Test closed-loop: `launch_vla_simulation.py` vẫn đặt lon ở vị trí mặc định. Cần thêm tuỳ chọn đặt lon theo
     `sample_randomization` (seed), nếu không sẽ không đo được model có biết nhìn lon hay không.
   - Tuỳ chọn: cho lon đổi phía (Coca bên trái…) để model phải phân biệt bằng màu chứ không bằng vị trí. Việc này cần thêm động tác gắp chéo người.
4. Tuỳ kết quả: bật `--tune-visual` (cần A100 80 GB), thử fine-tune dưới tag REAL_G1 với override thống kê, nhánh SONIC.

---

## 7. Tra cứu nhanh

### Lệnh local
```bash
.venv/bin/python verify_pipeline.py                                                   # 5 test của project
.venv/bin/python collect_demos.py --out datasets/<tên> --episodes-per-task 50 [--seed N] \
    [--lying-prob 0.5] [--can-x 0.26 0.37] [--can-y 0.03 0.19] [--can-gap 0.03] [--workers 12]
.venv/bin/python verify_demos.py datasets/<tên>
.venv/bin/python tools/plot_can_positions.py datasets/g1_can_sort_pilot --test-dataset datasets/g1_can_sort_test
.venv/bin/python tools/build_colab_notebooks.py                                       # sau khi sửa server/config
.venv/bin/python -c "import shutil; shutil.make_archive('datasets/<tên>', 'zip', 'datasets', '<tên>')"
```

### File chính
| File | Vai trò |
|---|---|
| `bridge/real_g1_adapter.py` | state/action REAL_G1 ↔ actuator sim |
| `colab_gr00t_server.py` | PolicyServer ZMQ (model thật hoặc quỹ đạo mẫu) |
| `colab_gr00t_server.ipynb`, `colab_gr00t_finetune.ipynb` | notebook Colab (sinh tự động, đừng sửa tay) |
| `finetune/g1_can_sort_config.py` | cấu hình modality cho fine-tune |
| `collect_demos.py`, `verify_demos.py` | sinh và kiểm tra dataset |
| `sim/arm_ik.py` | IK một tay G1 + waist_pitch (active set, kiểm tra va chạm, cố định khớp) |
| `sim/pick_place_planner.py` | quỹ đạo gắp-đặt IK online (generator 50 Hz, lập kế hoạch theo trạng thái sim) |
| `tools/build_colab_notebooks.py`, `tools/plot_can_positions.py` | sinh notebook, vẽ vị trí/hướng lon |

### Gói đã cài thêm vào `.venv`
`pyarrow`, `av` (ghi parquet/mp4), `matplotlib` (vẽ biểu đồ).
