# Memo: GR00T N1.7 thật + fine-tune cho nhiệm vụ phân loại lon (Unitree G1, MuJoCo)

Ghi lại các quyết định, phát hiện và trạng thái công việc trong các phiên làm việc ngày 22–24/09/2026.
- 23/09: IK online cho `collect_demos.py` (mục 4.2).
- 23–24/09: chạy closed-loop checkpoint pilot (mục 5.1), đo biên dư kẹp (mục 4.3), sinh dataset IK (mục 4.4),
  công cụ đánh giá closed-loop (mục 5.2), tham số hoá notebook fine-tune, lập plan mở rộng vùng gắp (mục 6).
- 24/09 (tiếp): lon đổi phía, gắp chéo người bằng `waist_yaw` (mục 4.5).

Nhánh `feat/smooth-vla-trajectory`. Các commit:
- `1579e2c`: đường ống GR00T thật, IK demo, công cụ fine-tune.
- `db8767c`: `tools/eval_closed_loop.py`, `setup_episode(pipeline_kwargs)`.

Chưa commit: tham số hoá notebook (`tools/build_colab_notebooks.py` và hai notebook), lon đổi phía (mục 4.5) và memo này.

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
Notebook cũ hard-code HF token `hf_JbmJ...`. Token này nằm trong commit `26398c5`, và commit đó **đã có trên `origin/main` ở GitHub**
(nên coi như đã lộ công khai). Commit `1579e2c` đã gỡ token khỏi notebook, nhưng token vẫn còn trong lịch sử git.
**Token phải được thu hồi trên HuggingFace** (Settings → Access Tokens); thu hồi là đủ, không cần viết lại lịch sử git.
Notebook mới đọc token từ Colab Secrets (`HF_TOKEN`). Tới 24/09 chưa ghi nhận đã thu hồi.

---

## 3. Nguồn dữ liệu đã khảo sát

- Không có dataset mở nào đúng bài toán "G1 phân loại lon Coca/Pepsi".
- `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` → `gr1_arms_only.CanSort`: đúng nhiệm vụ (1000 episode sim, 20 fps,
  1.34 GB) nhưng là robot **Fourier GR1**, action không dùng cho G1 được.
- `unitreerobotics/G1_Dex3_*` (PickBottle 202 ep, ObjectPlacement 210 ep, …): đúng robot G1 + Dex3, 28 khớp, 30 fps, ảnh thật,
  định dạng LeRobot **v3.0** (GR00T cần v2). Có thể chuyển sang layout REAL_G1 (đổi thứ tự ngón, tính EEF bằng FK).
- Nhánh SONIC: encoder `checkpoints/GEAR-SONIC/model_encoder.onnx` (vào 1762 chiều, ra token 64) ở chế độ `g1` chỉ cần quỹ đạo
  29 khớp + hướng thân, nên về nguyên tắc tự tính được motion token. **Chưa kiểm chứng** cách ghép vector 1762 chiều
  (cần đọc code GR00T-WholeBodyControl). Tạm gác. Các file ONNX khác đã kiểm tra shape (24/09):
  - Decoder `model_decoder.onnx`: vào 994 = token 64 + lịch sử 10 khung của gyro (30), góc khớp (290), vận tốc khớp (290),
    action trước (290), trọng lực (30); ra 29 action. Khớp với cách `SonicWBCController.decode` đang dựng.
  - `planner_sonic.onnx` (sinh chuyển động đi): vào `context_mujoco_qpos [1,4,36]`, `target_vel`, `mode`, `movement_direction`,
    `facing_direction`, `height`, `specific_target_positions/headings`…; ra `mujoco_qpos [1,64,36]` (64 khung qpos toàn thân).
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

### 4.2. IK online, lon ở vùng rộng, đứng hoặc nằm nhiều hướng (xong code 23/09, dataset ở mục 4.4)
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
Nếu cần lon ở xa hơn hoặc gắp ổn định hơn thì phải thêm điều khiển thăng bằng cho chân (plan ở mục 6).

**Kết thúc episode:** sau khi đặt lon, tay luôn về **tư thế ready chuẩn** (`q_ready` = keyframe `ready_table`), không về
tư thế lúc bắt đầu episode. Tư thế bắt đầu có nhiễu khớp σ 0.05 rad, nên cổ tay lúc kết thúc lệch so với lúc bắt đầu trung bình
2.4 cm, tối đa 6.7 cm (đo trên 400 episode IK). **Đã quyết định giữ như vậy** (24/09): về đúng tư thế lúc bắt đầu là mục tiêu
model không học được, vì cuối episode model chỉ thấy state hiện tại và 2 khung hình gần nhất.

### 4.2b. Sửa tư thế gắp lon nằm bị dị dạng (24/09, sau khi đã sinh `g1_can_sort_ik` và `g1_can_sort_swap`)
**Hiện tượng:** với một số lon nằm, cổ tay vừa gập hết cỡ (`wrist_pitch` 1.45 = giới hạn) vừa xoắn (`wrist_yaw` khoảng ±1.2–1.6),
bàn tay móc vòng rồi gắp vào đầu lon. Trong `g1_can_sort_ik`, 15/181 episode lon nằm bị như vậy (ví dụ episode 16, 168, 195).
Tất cả thuộc nhóm lon nằm chéo 20–60° so với hướng vai → lon; lon đứng chỉ 2/219 episode.

**Nguyên nhân:** planner bắt trục cuộn ngón tay (z cổ tay) trùng khít trục lon. Lon nằm chéo thì cả hai chiều bàn tay đều
buộc cổ tay G1 vừa gập vừa xoắn.

**Sửa** (`sim/pick_place_planner.py`):
- `LYING_AXIS_OFFSETS` = 0, ±15°, ±30°: trục cuộn ngón được lệch khỏi trục lon (ngón ôm chéo thân lon), phạt
  `LYING_AXIS_COST` = 1.5 mỗi rad lệch.
- `WRIST_TWIST_COST` = 2.0 × |wrist_pitch × wrist_yaw| cộng vào chi phí nghiệm khi gắp lon nằm.
- `planner.info["grasp_q"]` lưu tư thế gắp (để phân tích).

**Kết quả** (150 episode lon nằm cùng phía, seed 7000–7074, cùng seed cho hai planner):
| | Planner cũ | Planner mới |
|---|---|---|
| Thành công | 115/150 (77%) | **136/150 (91%)** |
| \|pitch × yaw\| cổ tay lúc gắp, TB | 0.48 | **0.11** |
| Gắp trượt | 18 | 8 |

Render lại episode 16 và 168: bàn tay úp từ trên xuống, ngón ôm quanh thân lon; episode 3 (vốn tự nhiên) giữ nguyên tư thế.

**Hệ quả:** `g1_can_sort_ik` (dùng cho checkpoint A) sinh bằng planner cũ nên vẫn có các tư thế dị dạng này; giữ nguyên làm
mốc so sánh. Dataset cũ vẫn verify được vì phát lại dùng action đã ghi, không chạy lại planner. `g1_can_sort_swap` đã được
sinh lại bằng planner mới (mục 4.5).

### 4.3. Biên dư kẹp lon (đo 23/09)
Sim chỉ bật weld kẹp khi điểm gắp trong lòng bàn tay (`grasp_point` = (0.13, ±0.05, 0) trong khung `wrist_yaw_link`) cách tâm lon
**≤ `GRASP_RADIUS` = 5.0 cm** (`sim/g1_scene.py`), tay khép quá `GRIP_CLOSED_RAD` và bàn tay chạm lon.
"Biên dư" là khoảng tay còn được phép lệch thêm, tính từ khoảng cách lúc tay khép tới ngưỡng 5.0 cm.

| Nguồn | Khoảng cách lúc tay khép | Biên dư |
|---|---|---|
| Quỹ đạo mẫu (dataset pilot), lon ở vị trí mặc định | 4.3 cm | **0.7 cm** |
| Model pilot `checkpoint-5000`, closed-loop (mục 5.1) | 5.2–5.7 cm | âm, không kẹp được |
| IK online, lon đứng (14 episode) | TB 1.2 cm, tối đa 2.4 cm | 2.6–4.5 cm |
| IK online, lon nằm (6 episode) | TB 0.8 cm, tối đa 1.3 cm | 3.7–4.5 cm |

- Phát lại action IK (ghi 25 Hz → nội suy 50 Hz → `track_joint_targets`) lệch ≤ 1 mm so với lúc planner chạy trực tiếp.
- **Không nới `GRASP_RADIUS`** để model "đạt": làm vậy chỉ che lỗi và sim không còn khớp với dữ liệu.

### 4.4. Dataset IK online (sinh 23–24/09)
Lệnh: `collect_demos.py --out datasets/g1_can_sort_ik --episodes-per-task 200 --lying-prob 0.6`, tập test thêm `--seed 100000`
(20 + 20). Dùng `--lying-prob 0.6` để bù việc lon nằm bị loại nhiều hơn.

| Dataset | Episode | Khung | Lon cần gắp đứng / nằm | Seed | Verify | Zip |
|---|---|---|---|---|---|---|
| `datasets/g1_can_sort_ik` (316 MB) | 400 (200 + 200, xen kẽ) | 75 200 | 219 / 181 | 0–881 | định dạng OK, phát lại **400/400**, lệch khớp TB 0.045 rad | 325 MB |
| `datasets/g1_can_sort_ik_test` (32 MB) | 40 (20 + 20) | 7 520 | 18 / 22 | từ 100000 | định dạng OK, phát lại **40/40**, lệch khớp TB 0.049 rad | 32 MB |

Tỉ lệ thành công khi thu (episode lỗi đã bị loại khỏi dataset):
| | Lon đứng | Lon nằm | Lỗi chính ở lon nằm |
|---|---|---|---|
| Coca (train) | 214/221 (97%) | 168/220 (76%) | gắp trượt 27, đặt không thành công 14 |
| Pepsi (train) | 213/215 (99%) | 164/226 (73%) | gắp trượt 39, đặt không thành công 14 |
| test | 31/31 | Coca 23/30, Pepsi 20/27 | gắp trượt 8 |

- Lệch khớp lớn nhất khi phát lại là 0.774 rad (train), chỉ ở vài khung của một episode, và episode đó vẫn phát lại thành công.
  Nhiều khả năng do ngón tay bị thân lon chặn.
- Biểu đồ `datasets/g1_can_sort_ik_can_positions.png`: lon trải đều vùng lấy mẫu, hướng lon nằm phủ đều 360°.
  Tập test nằm xen trong cùng vùng.
- **Vị trí lon mặc định của scene (x = 0.36) nằm ở mép xa của vùng dữ liệu** (giới hạn 0.37). Chạy
  `launch_vla_simulation.py` với vị trí mặc định là ca khó, nên đánh giá bằng `tools/eval_closed_loop.py` trên tập test.
- Thời gian: tập train chạy khoảng 35 phút thực tế (882 lần thử, 12 tiến trình), tập test 161 s. Máy từng tự ngủ (suspend) giữa chừng;
  với việc chạy dài, bọc lệnh trong `systemd-inhibit --what=sleep:idle ...`.

### 4.5. Lon đổi phía: gắp chéo người (code và dataset xong 24/09)
**Vấn đề:** ở mục 4.2, Coca luôn nằm bên phải (gần khay Coca), Pepsi luôn bên trái. Vì vậy model có thể học lối tắt
"lệnh Coca → tay phải → lon bên phải" mà không cần phân biệt màu. Cho lon đổi phía thì model buộc phải nhìn màu.

**Quyết định (24/09):**
- Mỗi lon luôn do tay cùng phía với khay của nó gắp (Coca tay phải, Pepsi tay trái), kể cả khi lon nằm ở nửa bàn bên kia.
  Tay trái không mang được Coca sang khay phải: khoảng cách ngang 0.55 m, dài hơn cánh tay 0.49 m.
  Đã loại hai phương án chuyển tay (handover) và đặt tạm giữa bàn rồi gắp lại.
- Cờ `--swap-prob` (mặc định 0): với xác suất này **cả hai lon đổi phía**. Lon cần gắp khi đó có |y| trong
  `--cross-can-y` (mặc định 0.03–0.15). Lon còn lại vẫn lấy trong `--can-y`.
- `cfg` cũ không có `swap_prob` thì `sample_randomization` không rút thêm số ngẫu nhiên, nên dataset cũ vẫn tái tạo đúng
  (đã thử `g1_can_sort_ik_test`: phát lại 8/8).

**Tầm với chéo người** (chỉ giải tư thế gắp, tay phải với lon ở nửa trái):
| | Lon đứng | Lon nằm |
|---|---|---|
| IK cũ (không xoay eo) | tới y ≈ +0.06–0.09 | tới y ≈ +0.03–0.06 |
| `waist_yaw` 0.5 rad | tới +0.15 | tới +0.12–0.15 |

**Code:**
- `ArmIK(use_waist_yaw=True)`: thêm khớp `waist_yaw` (giới hạn ±0.6 rad, trọng số tư thế ×3), đặt trước `waist_pitch`
  (`waist_pitch` vẫn là khớp cuối). Có `waist_mask`.
- `PickPlacePlanner(waist_yaw=None)`: **chỉ bật `waist_yaw` khi lon ở nửa bàn bên kia**. Nếu bật cả khi lon ở phía mình,
  lon nằm gắp trượt nhiều hơn: 80/119 so với 91/119 khi không bật, đo trên cùng seed.
  Servo giữ cố định cả hai khớp eo. Có thêm điểm khởi đầu IK cho lon ở nửa bàn bên kia (eo xoay 0.4 rad).
- `plan_grasps` trả về nhiều pose gắp. Với lon đứng đổi phía, hướng ngón tay được tính từ vai, từ điểm giữa và từ tâm thân.
  `targets()` lấy pose đầu tiên mà đường vươn tay, hạ xuống/tiến vào lon và **nhấc lon (kiểm tra trước)** đều không va chạm.
- `ArmIK.collides(attached=...)`: tính cả **lon đang cầm** như một khâu của bàn tay, áp dụng từ lúc nhấc tới lúc thả.
  Trước đó đường mang lon đi ngang trên lon còn lại và hất lon kia đổ hoặc rơi xuống sàn.
- `OTHER_CAN_MARGIN = 0.01`: khoảng hở 1 cm quanh lon còn lại, đặt trên một bản sao model chỉ dùng cho IK (vật lý sim
  không đổi). So trên 120 episode đổi phía: 0 / 1 / 2 cm được 39 / 40 / 36 episode, số lần xô lon kia là 12 / 5 / 4.
- `collection.jsonl` ghi thêm `swapped`. `eval_closed_loop.py` có thêm `--swap-prob`/`--cross-can-y` và nhóm "đổi phía /
  cùng phía". Bước 11 của notebook có thêm bảng theo đổi phía. `plot_can_positions.py` vẽ vùng lấy mẫu đổi phía (gạch chấm).

**Tỉ lệ thành công khi thu** (cfg mặc định, `--swap-prob 1`, seed 3000–3059):
| | Lon đứng | Lon nằm |
|---|---|---|
| Đổi phía | 15/82 (18%) | 25/38 (66%) |
| Cùng phía (đo riêng, xem ghi chú) | 39/41 (95%) | 91/119 (76%) |

Ghi chú về dòng cùng phía: số lon đứng đo lúc `waist_yaw` còn bật mọi lúc; số lon nằm đo khi `waist_yaw` tắt (giống bản
cuối). Hai số khớp với dataset IK cũ (97–99% / 73–76%). Kiểm tra va chạm của lon đang cầm và khoảng hở quanh lon còn lại
cũng áp dụng cho ca cùng phía, nhưng chưa đo lại riêng sau khi thêm.

- Lon đứng đổi phía khó vì hai lý do. Thứ nhất, cổ tay ở độ cao gắp (thấp hơn nắp lon kia 5 cm) và cẳng tay luôn nằm giữa
  hai lon. Thứ hai, lon còn lại có |y| < 0.07 thì gần như không bao giờ thành công (0/21); ở xa (|y| 0.13–0.16) cũng chỉ
  khoảng 37%. Lỗi chính là "tay va chạm khi tiến vào lon". Episode lỗi bị loại nên dữ liệu vẫn sạch, chỉ có hai hệ quả:
  - phần đổi phía của dataset lệch về lon nằm, và về các bố cục có lon còn lại ở xa hành lang của cánh tay;
  - tỉ lệ đổi phía trong dataset thấp hơn `--swap-prob`. Với 0.5 thì chỉ khoảng 1/3 dataset là đổi phía; muốn khoảng 1/2
    thì dùng khoảng 0.7.
- Chạy thử trọn đường ống (`--episodes-per-task 6 --swap-prob 0.5`): định dạng OK, phát lại 12/12.
- Khi gắp chéo, camera `ego_view` (gắn trên thân) xoay theo eo, nên model thấy khung hình xoay.
**Dataset (sinh lại 24/09 chiều bằng planner đã sửa tư thế gắp lon nằm, mục 4.2b; bản đầu buổi sáng đã xoá):**
`collect_demos.py --out datasets/g1_can_sort_swap --episodes-per-task 200 --swap-prob 0.7`,
tập test thêm `--episodes-per-task 20 --seed 100000`. Để mặc định `--lying-prob 0.5`: lon đứng đổi phía bị loại nhiều nên
lon nằm đã tự chiếm đa số.

| Dataset | Episode | Khung | Cùng phía đứng / nằm | Đổi phía đứng / nằm | Verify | Zip |
|---|---|---|---|---|---|---|
| `datasets/g1_can_sort_swap` (329 MB) | 400 (200 + 200) | 75 200 | 121 / 77 | 53 / 149 | định dạng OK, phát lại **400/400**, lệch khớp TB 0.047 rad | 339 MB |
| `datasets/g1_can_sort_swap_test` (34 MB) | 40 (20 + 20), seed 100000–100079 | 7 520 | 8 / 6 | 4 / 22 | định dạng OK, phát lại **40/40**, lệch khớp TB 0.057 rad | 34 MB |

- Tỉ lệ thành công khi thu tập train (882 lần thử, 26 phút, 12 tiến trình); trong ngoặc là số của bản đầu (planner cũ):
  | | Lon đứng | Lon nằm |
  |---|---|---|
  | Coca cùng phía | 88/89 | 52/60 (49/60) |
  | Pepsi cùng phía | 71/74 | 52/55 (38/55) |
  | Coca đổi phía | 40/159 (25%) | 85/133 (84/133) |
  | Pepsi đổi phía | 34/153 (22%) | 121/159 (90/159) |

  Số lon đứng giống hệt bản đầu, vì cùng seed và phần gắp lon đứng không đổi. Lỗi chính của lon đứng đổi phía vẫn là "tay
  va chạm khi tiến vào lon" (164/312 lần thử).
- Không còn episode lon nằm nào có cổ tay vừa gập > 1.2 vừa xoắn > 1.0 rad (0/226; bản đầu và `g1_can_sort_ik` có).
- Tập test chỉ có 4 episode lon đứng đổi phía, quá ít để kết luận riêng cho nhóm này.
- Biểu đồ `datasets/g1_can_sort_swap_can_positions.png`. `plot_can_positions.py` tách mặt bàn thành hai ô cùng phía / đổi
  phía nếu dataset có episode đổi phía; dataset cũ vẫn vẽ như trước.
- Cùng số episode và số khung với `g1_can_sort_ik`, nên fine-tune được với cùng `MAX_STEPS`/batch để so sánh trực tiếp.
- Hướng cải thiện nếu cần thêm lon đứng đổi phía: gắp lon đứng cao hơn (nâng `UPRIGHT_GRASP_HEIGHT` khi gắp chéo), hoặc
  thêm ràng buộc lấy mẫu để lon còn lại tránh hành lang của cánh tay.

---

## 5. Fine-tune trên Colab (xong lần thử đầu)

### Notebook `colab_gr00t_finetune.ipynb` (sinh bởi `tools/build_colab_notebooks.py`)
| Bước | Nội dung |
|---|---|
| 1–3 | GPU, cài Isaac-GR00T (ghim commit `51d4c89`), HF token |
| 4 | Mount Drive, `DATASET_NAME` / `TEST_DATASET_NAME` (mặc định `g1_can_sort_ik` / `g1_can_sort_ik_test`), biến môi trường (`MAX_STEPS=5000`, `SAVE_STEPS=1000`, `GLOBAL_BATCH_SIZE=32`) |
| 5–7 | Giải nén dataset, ghi `g1_can_sort_config.py`, `stats.py` + nạp thử bằng data loader (~5 phút) |
| 8 / 8b / 8c | Train **chạy nền** (`subprocess.Popen(start_new_session=True)`, log `train.log`) / theo dõi ETA + loss + VRAM / chờ xong rồi bíp |
| 8d | Biểu đồ loss (thang log + trung bình trượt), grad norm, learning rate theo bước. Đọc `train.log`; mất log thì đọc `trainer_state.json` trong checkpoint. Lưu `loss_curve.png` ở `$OUTPUT_DIR` và cạnh checkpoint trên Drive |
| 9 | Đánh giá open-loop episode 0, 1 (`MPLBACKEND=Agg`) |
| 10 | Chép checkpoint mới nhất về `MyDrive/gr00t/checkpoints/<DATASET_NAME>/` |
| 11 | Đánh giá open-loop 40 episode test + 40 episode train, bảng theo lon và theo lon đứng/nằm (đọc `lying` trong `collection.jsonl`) |

- Tham số hoá dataset ngày 24/09. Từ Bước 4, `DATASET_NAME` quyết định zip, thư mục giải nén, tên experiment và thư mục checkpoint,
  nên checkpoint IK không đè checkpoint pilot. Muốn chạy lại pilot thì đặt `"g1_can_sort_pilot"` / `"g1_can_sort_test"`.
- Dừng train chủ động: `os.killpg(train_proc.pid, signal.SIGTERM)` (bấm ■ không dừng được tiến trình nền).
- Bước 11 chạy được trên runtime mới (tự lấy checkpoint trên Drive, tự tạo `stats.json`); **L4 là đủ**, không dùng T4.
- **Số bước và batch cho dataset IK** (75 200 khung, gấp 4 lần pilot; với 5000 × 32 thì chỉ đi qua dữ liệu khoảng 2 lần):
  - A100 40 GB: giữ batch 32 (ghi chú notebook ước khoảng 35 GB), tăng `MAX_STEPS=10000`.
  - A100 80 GB: batch 64 và giữ 5000 bước.
  - Hai phương án cho tổng số mẫu như nhau. Xem VRAM ở Bước 8b để quyết định. Learning rate giữ mặc định.

- **Bước 7 tính thống kê song song (24/09):** `stats.py` gốc tính thống kê action relative tuần tự bằng Python. Riêng 2 khoá EEF
  mất khoảng 0.9 s/episode: 400 episode tốn khoảng 7 phút trên Colab, 12.7 phút trên máy local.
  - Bước 7 giờ thay `calculate_stats_for_key` bằng bản chạy `Pool(os.cpu_count())` theo episode, vẫn gọi đúng
    `load_relative_actions` và giữ thứ tự.
  - Trên máy local mất 127 s. So với bản gốc trên `g1_can_sort_swap_hold`, `stats.json` và `relative_stats.json` **giống hệt**.
  - Chưa đo trên Colab.

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

### 5.0b. Open-loop checkpoint IK `g1_can_sort_ik/checkpoint-5000` (24/09, Bước 11)
| Tập | Nhóm | Số ep | MSE TB | MAE TB | MAE lớn nhất |
|---|---|---|---|---|---|
| train | Coca / Pepsi | 20 / 20 | 0.00516 / 0.00385 | 0.0250 / 0.0239 | 0.0443 / 0.0307 |
| train | cả hai | 40 | 0.00451 | **0.0244** | 0.0443 |
| test | Coca / Pepsi | 20 / 20 | 0.00559 / 0.00560 | 0.0267 / 0.0272 | 0.0368 / 0.0507 |
| test | cả hai | 40 | 0.00559 | **0.0269** | 0.0507 |
| train | đứng / nằm | 21 / 19 | 0.00335 / 0.00578 | 0.0216 / 0.0276 | 0.0245 / 0.0443 |
| test | đứng / nằm | 18 / 22 | 0.00394 / 0.00694 | 0.0230 / 0.0302 | 0.0325 / 0.0507 |

Episode test sai nhiều nhất: 27 (Pepsi, nằm, 0.0507), 37 (Pepsi, nằm, 0.0501), 10 (Coca, nằm, 0.0368), 21 (Pepsi, nằm).

**Nhận định:**
- MAE cao gấp khoảng 5 lần pilot (0.0050). Điều này là dự kiến: action IK phụ thuộc thật vào vị trí và hướng lon, trong khi
  quỹ đạo mẫu của pilot gần như cố định. Hai con số không so sánh trực tiếp được.
- Khoảng cách train–test nhỏ (0.0244 so với 0.0269, khoảng 10%), nên model không học thuộc tập train.
- Sai số trên chính tập train còn cao, nên nhiều khả năng model **chưa học đủ** (underfit). Với 5000 × 32 mẫu, model mới đi
  qua 75 200 khung khoảng 2 lần (mục 5). Nếu loss ở Bước 8d vẫn đang giảm thì train dài hơn (10 000 bước) có thể giúp.
- Lon nằm khó hơn lon đứng (test 0.0302 so với 0.0230), và cả 4 episode sai nhất đều là lon nằm.
- Open-loop không nói được tỉ lệ gắp thành công (mục 5.1: MAE pilot 0.005 vẫn chỉ đạt 3/10). Bước tiếp theo là chạy
  closed-loop `eval_closed_loop.py`.

### 5.0c. Checkpoint IK `g1_can_sort_ik/checkpoint-10000` (`MAX_STEPS=10000`, batch **64** trên A100 80 GB, 24/09)
- 10000 × 64 = 640k mẫu, khoảng 8.5 lần đi qua 75 200 khung. (Bản ghi đầu tiên của mục này ghi nhầm batch 32.)
- Loss 1.33 → 0.0041 ở bước 10000 (trung bình trượt; 0.0046 ở bước 9100, khoảng 0.011 ở bước 5000). Grad norm đạt đỉnh khoảng 1.6 lúc warmup (bước ~400) rồi giảm đều
  về khoảng 0.2, không có spike. LR warmup 500 bước lên 1e-4, cosine còn khoảng 2e-6 ở bước 9100.
- Train ổn định. Loss vẫn giảm ở cuối, nhưng một phần là do LR giảm về 0, nên 900 bước còn lại chỉ giảm thêm rất ít.
- Train xong 10000 bước; loss ở 900 bước cuối chỉ giảm thêm một chút (0.0046 → 0.0041).

Open-loop (Bước 9: episode 0 / 1 có MAE 0.0129 / 0.0145). Bước 11, 40 episode mỗi tập, so với checkpoint-5000 (mục 5.0b):

| Tập | Nhóm | MSE TB (5000 → 10000) | MAE TB (5000 → 10000) | MAE lớn nhất (10000) |
|---|---|---|---|---|
| train | cả hai | 0.00451 → 0.00395 | 0.0244 → **0.0149** (−39%) | 0.0268 |
| test | cả hai | 0.00559 → 0.00537 | 0.0269 → **0.0189** (−30%) | 0.0493 |
| train | đứng / nằm | 0.00309 / 0.00490 | 0.0128 / 0.0173 | 0.0143 / 0.0268 |
| test | đứng / nằm | 0.00392 / 0.00656 | 0.0152 / 0.0219 | 0.0232 / 0.0493 |
| test | Coca / Pepsi | 0.00513 / 0.00561 | 0.0186 / 0.0192 | 0.0282 / 0.0493 |

Episode test sai nhiều nhất vẫn là các episode cũ: 27 (Pepsi, nằm, 0.0493; trước là 0.0507), 37 (Pepsi, nằm, 0.0392; trước 0.0501),
21 (Pepsi, nằm), 10 (Coca, nằm).

**Nhận định:**
- Checkpoint 10000 tốt hơn 5000 rõ ràng trên cả hai tập, nên checkpoint 5000 đúng là chưa học đủ. Dùng 10000 cho closed-loop.
- Khoảng cách train–test tăng từ 10% lên 27%. Model khớp tập train nhanh hơn khả năng tổng quát hoá, nhưng test vẫn cải thiện
  nên chưa gọi là overfit có hại. Nếu train dài hơn nữa thì phải theo dõi MAE test.
- MAE giảm 30–39%, nhưng MSE gần như không đổi (test −4%). Sai số thông thường nhỏ đi, còn các sai số lớn (có thể là lúc khép/mở
  ngón hoặc đầu mỗi chunk) vẫn còn nguyên. Cần tách lỗi theo khớp và theo thời gian để biết chính xác.
- Lon nằm vẫn là điểm yếu: MAE test gấp 1.44 lần lon đứng (ở 5000 là 1.31 lần), và episode 27 gần như không cải thiện.

### 5.0d. Open-loop checkpoint B `g1_can_sort_swap_hold/checkpoint-5000` (24/09, Bước 11, train đang chạy tiếp)
Train loss ở khoảng bước 5000 là 0.0077 (trung bình trượt; A-5000 khoảng 0.011). Grad norm khoảng 0.3–0.4, không có spike.
Tập đánh giá là `g1_can_sort_swap_hold(_test)`, không phải tập của A.

| Tập | Nhóm | Số ep | MSE TB | MAE TB | MAE lớn nhất |
|---|---|---|---|---|---|
| train | Coca / Pepsi | 20 / 20 | 0.00129 / 0.00183 | 0.0150 / 0.0148 | 0.0304 / 0.0234 |
| train | cả hai | 40 | 0.00156 | **0.0149** | 0.0304 |
| test | Coca / Pepsi | 20 / 20 | 0.00163 / 0.00224 | 0.0166 / 0.0162 | 0.0265 / 0.0236 |
| test | cả hai | 40 | 0.00194 | **0.0164** | 0.0265 |
| train | đứng / nằm | 19 / 21 | 0.00095 / 0.00212 | 0.0137 / 0.0160 | 0.0208 / 0.0304 |
| test | đứng / nằm | 12 / 28 | 0.00102 / 0.00233 | 0.0135 / 0.0176 | 0.0168 / 0.0265 |
| train | cùng / đổi phía | 18 / 22 | 0.00092 / 0.00209 | 0.0134 / 0.0161 | 0.0208 / 0.0304 |
| test | cùng / đổi phía | 14 / 26 | 0.00150 / 0.00217 | 0.0151 / 0.0171 | 0.0202 / 0.0265 |

Episode test sai nhiều nhất: 8 (Coca, 0.0264), 9, 1, 29, 17 (Pepsi, khoảng 0.022). Cả 5 đều là lon nằm và đổi phía.

**Nhận định:**
- Con số tốt hơn A-5000 (MAE test 0.0269) và nhỉnh hơn A-10000 (0.0189). MSE test chỉ bằng khoảng 1/3 của A (0.0019 so với
  0.0054), MAE lớn nhất 0.0265 so với 0.0493.
- **Không so trực tiếp được.** Tập test khác nhau, và `--steps 400` phủ cả 2.5 s đứng yên (khoảng 19% số khung), vốn dễ đoán.
  Giả sử khung đứng yên sai số bằng 0, thì MAE trên phần gắp-đặt vào khoảng 0.0164 / 0.81 ≈ 0.020, tức ngang A-10000 và vẫn
  tốt hơn A-5000.
- Khoảng cách train–test 10%, giống A-5000, nên chưa học thuộc.
- Đổi phía khó hơn cùng phía 1.14 lần, lon nằm khó hơn lon đứng 1.31 lần. Hai nhóm này chồng lên nhau nên chưa tách được
  riêng từng yếu tố.
- Quyết định: train tiếp tới hết `MAX_STEPS` (checkpoint `--save-only-model` không train tiếp được). So công bằng bằng cách
  chạy chéo: A-10000 trên `swap_hold_test`, B trên `ik_test`, rồi closed-loop.

### 5.0e. Open-loop checkpoint B `g1_can_sort_swap_hold/checkpoint-7000` (24/09, Bước 11)
| Tập | Nhóm | Số ep | MSE TB | MAE TB (5000 → 7000) | MAE lớn nhất |
|---|---|---|---|---|---|
| train | cả hai | 40 | 0.00069 | 0.0149 → **0.0100** (−33%) | 0.0159 |
| test | cả hai | 40 | 0.00108 | 0.0164 → **0.0118** (−28%) | 0.0221 |
| test | Coca / Pepsi | 20 / 20 | 0.00106 / 0.00110 | 0.0119 / 0.0117 | 0.0221 / 0.0192 |
| train | đứng / nằm | 19 / 21 | 0.00063 / 0.00075 | 0.0096 / 0.0104 | 0.0159 / 0.0153 |
| test | đứng / nằm | 12 / 28 | 0.00077 / 0.00121 | 0.0102 / 0.0125 | 0.0145 / 0.0221 |
| train | cùng / đổi phía | 18 / 22 | 0.00060 / 0.00077 | 0.0093 / 0.0106 | 0.0159 / 0.0153 |
| test | cùng / đổi phía | 14 / 26 | 0.00095 / 0.00114 | 0.0109 / 0.0123 | 0.0157 / 0.0221 |

Episode test sai nhiều nhất: 8 (Coca, nằm, đổi phía, 0.0221; ở 5000 là 0.0264), 1 (Pepsi, nằm, đổi phía, 0.0192),
30 (Coca, nằm), 20 (Coca, nằm, đổi phía), 13 (Pepsi, đứng, 0.0145).

**Nhận định:**
- Vẫn tiến bộ mạnh: MAE test −28%, MSE test −44%. Ước lượng bỏ khung đứng yên: 0.0118 / 0.81 ≈ 0.0146, tốt hơn A-10000
  (0.0189) khoảng 23%, dù tập test khác nhau.
- Khoảng cách train–test tăng từ 10% lên 18% (A cũng tăng 10% → 27%). Test vẫn giảm nên chưa phải overfit có hại; từ đây theo
  dõi MAE test ở mỗi checkpoint.
- Lon nằm / lon đứng trên test giảm từ 1.31 xuống 1.22 lần, nhưng trên train chỉ 1.08 lần. Lỗi lon nằm chủ yếu là lỗi tổng quát hoá.
  Đổi phía / cùng phía giữ ở 1.13 lần, và đổi phía tổng quát hoá không kém cùng phía (test/train đều khoảng 1.16).
- **Episode 8 không nằm ngoài phân bố:** lon nằm cách episode train gần nhất 0.4 cm, có 21 episode train cùng nhóm trong bán
  kính 3 cm, yaw lệch 5°. Trung vị khoảng cách tới láng giềng gần nhất của cả tập test là 0.8 cm.
- **Lon nằm có hai kiểu gắp:** planner thử hai chiều trục lon (`s = ±1`), 5 độ lệch và 2 trọng số rồi chọn chi phí nhỏ nhất. Ở
  khung khép tay (luôn là khung 55), wrist pitch rơi vào hai cụm: khoảng +1.0..1.4 (78/225 episode train) hoặc khoảng
  −0.5..−0.7, cách nhau khoảng 2 rad. Trong 28 episode test lon nằm, 12 episode có 6 láng giềng gần nhất (vị trí + yaw gập theo
  180°) lẫn cả hai kiểu; lon đứng là 0/12. Ví dụ: test 20 và train 60 lệch 1 cm và 9° nhưng khác nhau 1.9 rad. Ngay trong cùng
  một kiểu, láng giềng vẫn lệch 0.15–0.6 rad.
  Kết luận: MAE open-loop của lon nằm có một mức sàn mà train thêm không xoá được. Flow matching vẫn học được phân bố nhiều
  kiểu, nên closed-loop không nhất thiết bị ảnh hưởng, nhưng nên xem closed-loop có bị lưỡng lự giữa hai kiểu không. Nếu cần
  thì sửa planner để chọn kiểu một cách liên tục theo yaw (ví dụ cố định dấu `s` theo vùng yaw), rồi sinh lại dữ liệu.

### 5.1. Closed-loop với checkpoint pilot (23/09)
Server Colab chạy `g1_can_sort_pilot/checkpoint-5000`, tag `new_embodiment`, qua tunnel Pinggy. Local dùng `--vla-action-hz 25`
(bắt buộc cho checkpoint fine-tune: mặc định 20 Hz làm chunk chậm 25% và lấy khung hình quá khứ lùi 1.0 s thay vì 0.8 s).
Lon ở vị trí mặc định, tức giữa vùng train của pilot.

- **Đường ống closed-loop chạy đúng**: đã có 2 lần gắp-đặt trọn vẹn (Coca và Pepsi), tay về ready, robot ngả tối đa 5.6°,
  không có timeout.
- **Tỉ lệ thành công 3/10.** Mọi lần hỏng đều giống nhau: tay khép đúng lúc (khoảng 2.4 s) nhưng cách lon 5.2–5.7 cm, vượt
  `GRASP_RADIUS` vài mm, nên không kẹp được. Lon bị đẩy tới trước hoặc bị quét đổ (Z = 0.712 tức lon nằm).
  Lần chạy GUI đầu tiên (`vla_log_ft_2.npz`) cũng hỏng theo kiểu này.
- Nguyên nhân: model lệch **có hệ thống** khoảng 1 cm so với demo (lùi về phía robot 7 mm, ra ngoài 9 mm; điểm gắp
  (0.335, −0.183) so với demo (0.342, −0.174), lặp lại giữa các lần chạy ± 2 mm). Trong khi đó demo pilot chỉ có biên dư 0.7 cm
  (mục 4.3). Khớp cổ tay sai 0.02–0.03 rad, qua cánh tay đòn khoảng 13 cm tới điểm gắp, là thành 1 cm; open-loop MAE 0.005 rad
  không cho thấy điều này.
- Model bám quỹ đạo demo rất sát (khớp tay và eo lệch khoảng 0.02 rad, cả thời điểm khép và mở ngón). Model đang **phát lại
  quỹ đạo đã học thuộc**. Sau khi xong một vòng, model lặp lại động tác gắp.
- Kết luận: cách sửa đúng là dữ liệu IK (biên dư 2.6–4.5 cm, vùng lon rộng), không phải nới ngưỡng kẹp.

### 5.2. Công cụ đánh giá closed-loop `tools/eval_closed_loop.py` (23/09, commit `db8767c`)
- Đặt lon, khay và nhiễu khớp theo seed như `collect_demos.py`: lấy từ `--dataset DIR` (đọc `meta/collection.jsonl`) hoặc từ
  `--seed/--episodes-per-task` với cfg giống `collect_demos`. Dùng `collect_demos.setup_episode(..., pipeline_kwargs=...)`.
- Chấm bằng `is_success` như lúc thu demo. Mỗi episode còn ghi: có kẹp được không, khoảng cách lúc khép, độ nhấc lon, lon còn lại
  có bị xô không, độ ngả tối đa, số lần gọi và số lần server lỗi, lý do thất bại. Có riêng lý do "đã đặt xong nhưng sau đó xô/gắp lại
  lon", để bắt trường hợp model lặp lại task.
- Kết quả ghi ra `results.jsonl`, `summary.txt` (bảng theo lon × đứng/nằm) và `videos/` (camera `front_camera`, `--save-video fail|all|none`).
- **Dừng hẳn** nếu server không phải model thật (`action_space != real_g1`), hoặc 3 lần gọi đầu đều lỗi.
  Lý do: client `Gr00tN17Policy` **tự chuyển sang standalone (quỹ đạo mẫu) mà không báo** nếu không kết nối được ZMQ.
- Mỗi episode chạy trong một tiến trình riêng (`maxtasksperchild=1`); `--workers 2` mặc định.
- Tunnel **Pinggy miễn phí hết hạn sau khoảng 60 phút**. Khi server không phản hồi thì chạy lại Bước 6 của notebook server để lấy
  host/port mới; server chạy nền nên không phải nạp lại model (từ 24/09, mục 5.3).

### 5.2b. Closed-loop checkpoint IK-10000, bản thử 4 episode test (24/09, `eval/ik10k_smoke`)
Server chạy bản cũ nên `server_model` = null; dựa vào hành vi thì đây đúng là checkpoint-10000. Video của lần chạy model base
(xem 5.3) đã chuyển sang `eval/base_smoke/videos/`.

| Seed | Lon | Kết quả | K/c lúc khép | Kẹp lúc | Ngả tối đa |
|---|---|---|---|---|---|
| 100044 | Pepsi đứng | **OK** (đặt xong lúc 5.5 s) | 3.4 cm | 2.32 s | 11.2° |
| 100001 | Coca nằm | **OK** (5.5 s) | 3.4 cm | 2.14 s | 4.9° |
| 100002 | Coca nằm | FAIL: khép tay nhưng không kẹp được | **8.1 cm** | – | 11.3° |
| 100046 | Pepsi đứng | **OK** | 2.0 cm | – | 6.3° |

- **3/4**, trong khi pilot chỉ được 3/10 và chỉ ở vị trí mặc định. Model IK gắp được cả lon nằm và tay đi đúng tới lon.
- Ca hỏng 100002 (lon nằm ở x = 0.329, y = −0.095): tay khép khi còn cách lon 8 cm. Sau đó tay kéo lê lon (không nhấc lên,
  độ nhấc −0.1 cm) sang phía khay, rồi lon rơi khỏi bàn (z cuối 0.032). `ok_any` = true lúc 5.5 s chỉ là lúc lon trượt qua khay.
- 4 episode là quá ít để kết luận; bước tiếp theo là chạy đủ 40 episode test.

### 5.2c. Eval cho phép thử lại (24/09)
- Ý tưởng: không cắt ở 10 s rồi kết luận thất bại mà chạy dài hơn (`--seconds 30`) để model tự thử lại. Model chưa từng thấy
  thử lại trong demo (episode lỗi đã bị loại), nhưng ca 100051 cho thấy model tự bắt đầu lại task khi tay đã về ready và lon
  không nằm trong khay; thử lại có thể đến từ chính hành vi này.
- Giới hạn: trong các ca hụt lon nằm, lon thường bị kéo lê rồi rơi khỏi bàn (100002, 100049, 100008), nên không thể thử lại.
  Chạy dài hơn cũng làm tăng lỗi "đặt xong rồi gắp lại".
- Vì vậy báo cả ba số: thành công ngay lần đầu (`t_success` ≤ ~7.5 s), từng đặt xong trong T giây (`ok_any`), và trạng thái
  cuối (`ok`).
- `eval_closed_loop.py` thêm các trường:
  - `attempts`: số lần khép tay, tính theo cạnh lên.
  - `can_fell`: z cuối thấp hơn lúc đầu 0.2 m; lý do thất bại mới là "lon rơi khỏi bàn", được xét trước các lý do khác.
  - Summary in số lần khép TB/tối đa, số lon rơi, và thời điểm đặt xong lần đầu (trung vị, muộn nhất).

### 5.2d. Closed-loop 27/40 episode test và quyết định dataset tiếp theo (24/09)
Bộ chạy đủ 40 episode dừng ở 27/40 (`eval/ik10k_smoke/results.jsonl`; `summary.txt` trong thư mục đó vẫn là của bản thử 4 episode).
- **Thành công 19/27 (70%)**, từng đặt xong 21/27 (78%), server không lỗi lần nào.
- Theo nhóm: Coca nằm 6/10, Coca đứng 3/4, Pepsi nằm 5/6, Pepsi đứng 5/7 (Pepsi đứng từng đặt xong 7/7).
- Lỗi:
  - Khép tay nhưng không kẹp được: 5 ca, 4 ca lon nằm; k/c lúc khép 5.8–7.1 cm, trong khi ca thành công là 1.0–4.8 cm.
  - Đặt xong rồi gắp lại: 2 ca (100051, 100053), đều bắt đầu ở mốc khoảng 7.5 s, tức lúc hết dữ liệu demo.
  - Xô lon còn lại: 1 ca.
- Lỗi hụt lon nằm **không tụ theo hướng lon** (yaw 27–168°; với cùng yaw 98–108° có ca OK, có ca hỏng). Đây là lỗi độ chính
  xác, không phải thiếu vùng dữ liệu, nên không đổi phân bố dữ liệu để sửa.
- **Quyết định:** sinh lại **`g1_can_sort_swap`** (dataset train tiếp theo, chưa train lần nào) với `hold` dài hơn. **Giữ
  nguyên** `g1_can_sort_ik` để mốc A (checkpoint-10000) còn tái lập được. Chưa làm episode "việc đã xong" (lon nằm sẵn trong khay,
  robot chỉ đứng yên); làm sau nếu eval 30 s vẫn thấy gắp lại.
- Code:
  - `collect_demos.py --hold-seconds` (mặc định **2.5 s**, ghi vào `cfg.hold_seconds`; cfg không có khoá này thì 0.7 s như cũ).
  - `PickPlacePlanner(hold_steps=...)`.
  - Episode dài 7.5 → **9.3 s (233 khung)**.
  - Notebook fine-tune Bước 9/11 đổi `--steps 200` → 400.
  - Thử 4 episode (swap 0.7): verify OK, phát lại 4/4; action trong 2.5 s cuối không đổi, state dao động ≤ 0.03 rad.
  - Open-loop MAE của dataset mới sẽ thấp hơn một chút do có thêm khung đứng yên, nên không so trực tiếp với A.

### 5.2c. Closed-loop checkpoint IK-10000 trên tập test, 27/40 episode (24/09, `eval/ik10k_smoke`)
Lần chạy 40 episode ghi đè lên `eval/ik10k_smoke` và dừng ở 27/40 lúc 15:54, không ghi summary, không rõ lý do (Ctrl+C
hoặc server mất phản hồi làm cả script dừng). `summary.txt` được viết lại từ 27 kết quả.

| Nhóm | Thành công | Từng đặt xong (`ok_any`) | K/c lúc khép TB |
|---|---|---|---|
| tất cả (27) | **19 (70%)** | 21 (78%) | 4.4 cm |
| lon đứng (11) | 8 | **10 (91%)** | ~3.6 cm |
| lon nằm (16) | 11 (69%) | 11 (69%) | ~5 cm |

Lý do hỏng:
- 5 lần khép tay nhưng không kẹp được, **khoảng cách lúc khép 5.8–7.1 cm**, trong khi ca thành công là 2–4.8 cm.
  4/5 ca là lon nằm, 1 ca là lon đứng ở y = −0.153 (sát rìa vùng).
- 2 lần **đặt xong rồi gắp lại**, đều là lon đứng (100051, 100053). Video 100051: đặt xong lúc 5.0 s, về ready lúc 6.2 s,
  vươn tay lại lúc 6.8 s, gắp lon ra khỏi khay rồi đặt về bàn.
  - Nguyên nhân: `hold` chỉ 35 bước 50 Hz = **0.7 s** (khoảng 17/188 khung mỗi demo), còn eval chạy 10 s, dài hơn demo 7.5 s.
  - Ở tư thế ready, state gần như giống lúc bắt đầu episode; chỉ ảnh (lon đã nằm trong khay) cho biết việc đã xong.
- 1 lần xô lon còn lại (lon nằm, tay khép cách lon 15 cm).

Đề xuất (chờ người dùng duyệt):
- Kéo dài `hold` lên 2.5–3 s, và thêm 10–15% episode "đã xong": lon nằm sẵn trong khay, action là đứng yên.
- Gộp vào lần sinh dataset đổi phía để khỏi phải train hai lần.

Công cụ (24/09): `summary.txt` có thêm cột "từng đặt xong". `eval_closed_loop.py --resume` giữ kết quả cũ và chỉ chạy episode còn
thiếu. Script luôn ghi summary khi bị dừng giữa chừng, kèm dòng "CHƯA ĐỦ" (đã thử với server không phản hồi).
Lưu ý: kiểm tra xem eval còn chạy không bằng `pgrep -f` với một pattern có chứa lệnh eval sẽ khớp luôn chính lệnh kiểm tra,
nên nhìn thời điểm sửa của `results.jsonl` thì đáng tin hơn.

### 5.2d-bis. Dataset `g1_can_sort_swap_hold` (sinh 24/09, 16:24–17:04)
Lệnh như `g1_can_sort_swap` (`--episodes-per-task 200 --swap-prob 0.7`, cùng seed bắt đầu), thêm `--hold-seconds 2.5` (mặc
định mới). Tập test: `--episodes-per-task 20 --seed 100000`.

| Dataset | Episode | Khung | Đứng / nằm | Đổi phía | Verify | Zip |
|---|---|---|---|---|---|---|
| `g1_can_sort_swap_hold` (378 MB) | 400 (200 + 200) | 93 200 (233/ep) | 175 / 225 | 200 | định dạng OK, phát lại **400/400**, lệch khớp TB 0.046 rad, max 0.407 | 389 MB |
| `g1_can_sort_swap_hold_test` (39 MB) | 40 | 9 320 | 12 / 28 | 26 | định dạng OK, phát lại **40/40**, lệch khớp TB 0.057 rad | 39 MB |

- Phân bố gần như trùng bộ `swap` cũ (174 / 226, 202 đổi phía). Khác vài episode, có thể do `is_success` được xét sau khi
  robot đứng yên thêm 1.8 s.
- Số khung tăng 24% nhưng đều là khung đứng yên. Muốn phần gắp-đặt được học đủ như mốc A (10000 bước × batch 64 trên
  75 200 khung) thì cần khoảng **12 500 bước với batch 64**, hoặc khoảng 6 200 bước với batch 128 (cùng số mẫu, ít lần cập nhật
  hơn). Giữ batch 64 để so A/B chỉ khác dữ liệu.
  Một phương án gọn khác là batch 80 × 10 000 bước = 800k mẫu (khoảng 8.6 lần đi qua dữ liệu, gần bằng A); giữ LR 1e-4.
- Biểu đồ vị trí lon: `datasets/g1_can_sort_swap_hold_can_positions.png`.

### 5.2e. GUI: `--stop-on-success`, phím 2 làm mất robot (24/09)
- **Phím 2 làm robot biến mất:** phím 0–5 của viewer MuJoCo bật/tắt hiển thị geom group, mà mesh robot G1 (`class="visual"`)
  nằm ở group 2. Mô phỏng không lỗi gì. Sửa: launcher tự bật lại group 1 và 2 mỗi vòng lặp.
- **Đổi phím sang F8–F12 (24/09):** viewer MuJoCo dùng hết 26 chữ cái (phím tắt vis/render), 0–5 (geom group) và F1–F7.
  Phím mới: F8 Coca, F9 Pepsi, F10 reset (thay R), F11 cảnh tiếp (thay N), F12 snapshot (thay S); SPACE vẫn là tạm dừng.
  Đã bỏ phần tự đặt lại geom group/island vì không còn trùng phím. Thử bằng viewer giả: 1/2/N/R không còn tác dụng, F8–F12
  chạy đúng. Chưa thử trong viewer thật.
- **Phím N làm robot và lon thành màu vàng:** N là phím tắt `mjVIS_ISLAND` của viewer, tô màu theo constraint island. Sửa: đặt
  lại cờ này về giá trị ban đầu mỗi vòng lặp, như cách làm với geom group. Phím R (phản chiếu) và S (bóng) vẫn bị viewer bật/tắt,
  vì Python binding không cho truy cập `mjvScene.flags`; hai phím này chỉ ảnh hưởng hình ảnh.
- **R / N:** đã thử vòng lặp chính với viewer giả; sau R, qpos trùng tuyệt đối với lúc bắt đầu, `reset()` đưa mọi trạng thái
  về như pipeline mới.
- **`--stop-on-success`** (mặc định bật; tắt bằng `--no-stop-on-success`): khi lon của nhiệm vụ hiện tại nằm trong khay và tay
  đã nhả, chờ cổ tay rút xa lon thêm 6 cm so với lúc nhả (tối đa 2 s), rồi ngừng gọi model.
  - Robot đi theo `PickPlacePlanner.return_to_ready` (nhấc tay 12 cm, về điểm cao trên ready, hạ xuống) rồi giữ tư thế ready,
    tới khi bấm F10 / F11 / F8 / F9.
  - Pipeline có thêm `hold_ready(path)`; `reset()` và đổi nhiệm vụ đều huỷ chế độ giữ.
  - Hai phương án đã thử và bỏ:
    - Chờ tay về trong 0.2 rad quanh ready: không bao giờ tới, vì "ready" của model lệch khoảng 0.4 rad, và sau khoảng 2 s
      model đã vươn tay gắp lại.
    - Nội suy khớp thẳng về ready: tay quét lon Pepsi đặt sát mép khay ra ngoài 3–4 cm.
- Đoạn "về ready" của planner được tách thành `_return_path` (dùng chung). Đã kiểm tra: action sinh ra giống hệt
  `g1_can_sort_swap_hold` (Δ = 0).
- Thử với server thật (checkpoint-10000, viewer giả):
  - Episode 1 (Pepsi): dừng lúc 6.1 s, không gọi model thêm, tay về ready, lon vẫn trong khay.
  - Episode 0: đặt Coca, bấm 2 lúc 10 s, gắp Pepsi, giữ; cả hai lon ở trong khay tới 19 s.

### 5.2f. Chân tự cân bằng: `--lower-body balance` (24/09)
- **Nguồn:** `third_party/GR00T-WholeBodyControl` (NVlabs, commit `b042411`), sparse checkout gồm `decoupled_wbc/sim2mujoco`,
  `control/policy`, `control/main/teleop/configs`. Thư mục này nằm trong `.gitignore`.
  - Máy chưa có git-lfs, nên hai file policy được tải từ
    `https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/<commit>/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-{Balance,Walk}.onnx`
    (1.9 MB mỗi file, sha256 khớp con trỏ LFS).
  - Mesh STL vẫn là con trỏ LFS; không cần.
- **Model robot của NVIDIA (`g1_gear_wbc.xml`) so với sim này:** trùng tổng khối lượng (34.39 kg), khối lượng từng khâu,
  armature, độ cao pelvis 0.793 m và thứ tự 29 khớp. Khác: frictionloss 0.1 so với 0.3, và actuator (họ dùng motor + PD
  trong Python, sim này dùng position KP 500).
- **`bridge/lower_body_policy.py` (`BalancePolicy`)** làm theo `run_mujoco_gear_wbc.py` và `g1_decoupled_whole_body_policy.py`:
  - Quan sát 86 chiều × 6 bước, output 15 (chân + eo), mục tiêu `0.25·a + default`.
  - KP/KD của NVIDIA chỉ áp cho 15 actuator chân + eo, frictionloss 0.1. Tay và bàn tay giữ KP 500.
  - Eo: góc eo VLA ra lệnh được đổi (forward kinematics) thành rpy của torso so với pelvis và đưa vào lệnh rpy của
    policy; policy điều khiển eo.
  - Lệnh đi = 0, luôn dùng policy Balance. `height_cmd` = 0.82, robot đứng ở ~0.77 m (sim cũ 0.79). Mức 0.74 của NVIDIA
    cho 0.75 m và nghiêng nhiều hơn.
  - `init_pose()`: đặt chân vào tư thế mặc định của policy và hạ pelvis để bàn chân chạm sàn. Nếu bắt đầu từ chân thẳng,
    bàn chân trượt khoảng 3 cm.
  - `settle()`: cho robot đứng 2 s ở tư thế ready rồi dời về x, y ban đầu. Lý do: policy đẩy pelvis lùi khoảng 7 cm để giữ
    trọng tâm (eo cúi, tay đưa ra trước); không dời lại thì lon ra ngoài tầm với.
- **Pipeline:** `ClosedLoopVLAPipeline(lower_body="fixed"|"balance", balance_height=0.82)`. `_lower_body()` thay mục tiêu
  chân + eo trước mọi `sim.step` (chạy thường, lúc giữ ready, lúc tạm dừng). `reset()` chạy lại `init_pose` + `settle` (0.1 s);
  reset hai lần cho qpos giống hệt.
- **Cờ:** `launch_vla_simulation.py --lower-body balance`, `tools/eval_closed_loop.py --lower-body balance` (ghi `lower_body`
  vào `results.jsonl`). Mặc định `fixed`; thu demo, verify và phát lại không đổi. `verify_pipeline.py` qua đủ 5 test.
- **Kết quả đo:**
  - Đứng yên 30 s: nghiêng cuối 0.4° (chân khoá cứng 2.9°), trôi dưới 0.5 cm sau khi ổn định.
  - Planner IK, 15 seed × 2 lon, cùng seed:

| Chế độ | Thành công đứng / nằm | Nghiêng max TB đứng / nằm | Pelvis dịch max TB |
|---|---|---|---|
| fixed | 16/17 / 12/13 | 5.4° / 7.2° (max 11.4°) | 4.8 / 7.3 cm (max 12.8) |
| balance | 16/17 / 13/13 | 2.3° / 3.1° (max 4.3°) | 2.3 / 2.5 cm |

- **Chưa thử với model thật** (tunnel hết hạn). A/B được train với chân khoá cứng, pelvis ở 0.79 m. Bật balance thì pelvis
  thấp hơn khoảng 2 cm và còn dịch vài cm khi vươn tay, nên ảnh camera khác một chút. Cần chạy
  `eval_closed_loop --lower-body balance` và so với `fixed` trên cùng tập test.

### 5.2g. Closed-loop B-9000 trên `ik_test`: lon Coca nằm hỏng nhiều hơn (25/09, `eval/B9000_ik_test`, đang chạy)
14/40 episode đầu: 12 thành công. Lon Coca nằm 4/6, Coca đứng 1/1, Pepsi đứng 5/5, Pepsi nằm 2/2. A-10000 trên 27 episode:
Coca nằm 6/10, Pepsi nằm 5/6. Mẫu còn nhỏ, nhưng Coca nằm là nhóm yếu nhất ở cả A lẫn B.
- B sửa được các ca của A: 100008 (A hụt 6.3 cm, B kẹp ở 1.8 cm), 100051 và 100053 (A đặt xong rồi gắp lại; `hold` 2.5 s có tác dụng).
- B hỏng đúng hai cảnh A cũng hỏng: 100002 và 100009, đều là Coca nằm, yaw gập theo 180° là **103° và 100°**. Kiểu hỏng
  nặng hơn A: khép tay cách lon 14.4 / 9.5 cm, lon rơi khỏi bàn (A: hụt 7 cm, lon vẫn trên bàn).
- **Video:** tay tiếp cận thấp từ bên cạnh, rồi giữa chừng (1.4 s ở 100002, 3.2 s ở 100009) chuyển sang giơ cao và hạ từ trên
  xuống. Đó là hai kiểu gắp lon nằm ở mục 5.0e. Tay đổi ý giữa các chunk, đụng lon làm lon rơi, rồi vẫn làm động tác đặt với
  tay rỗng.
- **Dữ liệu:** với Coca nằm cùng phía (tay phải), kiểu gắp đổi theo yaw ngay quanh 90–100°.

  | Yaw gập | 0–20 | 20–40 | 40–60 | 60–80 | 80–100 | 100–120 | 120–140 | 140–160 | 160–180 |
  |---|---|---|---|---|---|---|---|---|---|
  | A (`g1_can_sort_ik`) kiểu +/tổng | 4/7 | 2/4 | 6/7 | 15/15 | 13/13 | 5/7 | 0/13 | 1/10 | 1/16 |
  | B (`swap_hold`) kiểu +/tổng | 2/6 | 4/7 | 3/3 | 4/4 | **0/0** | 0/3 | 0/4 | 0/6 | 0/6 |

  - B chỉ có 39 episode Coca nằm cùng phía (A có 92), vì 70% episode là đổi phía. B không có episode nào ở 80–100°, và các
    láng giềng gần nhất của 100002/100009 lẫn cả hai kiểu (`---++-`, `--+--+`).
  - Cảnh thử rơi đúng vào chỗ đổi kiểu và cũng là chỗ thiếu dữ liệu. Flow matching lấy noise mới ở mỗi lần gọi model (mỗi
    0.3 s), nên các chunk liên tiếp có thể rơi vào hai kiểu khác nhau.
- **100014** (Coca nằm, yaw gập 121°, cũng gần ranh giới, B chỉ có 7 episode ở 100–140°): hỏng vì lần gắp đầu, không phải vì
  video ngắn.
  - 0.5–2.9 s: tay lượn quanh lon, lưỡng lự.
  - Khoảng 3.7 s: khép tay cách lon 11.7 cm, đẩy lon 17 cm tới (0.513, 0.008), ra ngoài vùng train (x 0.26–0.37).
  - 4.5–5.3 s: đặt tay rỗng vào khay rồi về ready.
  - Từ 6.9 s: vươn lại về phía lon, tới 10 s thì hết giờ.
  - Chạy dài hơn (`--seconds 20`) có thể cho thêm `ok_any`, nhưng lần thử lại bắt đầu từ vị trí lon ngoài phân bố. Nên báo
    riêng "thành công lần đầu" và "thành công sau khi thử lại" (mục 5.2c).
- **Kết quả 28/40 (dừng giữa chừng, chạy tiếp bằng `--resume`): 20/28 = 71%** (khoảng tin cậy 95%: 53–85%).
  - Coca 8/14 (57%), Pepsi 12/14 (86%). Pepsi đứng 7/7, Coca đứng 2/4.
  - Lý do hỏng: 4 lon rơi khỏi bàn, 4 khép tay hụt (5–14 cm).
  - So cặp với A-10000 trên 27 episode chung: A 19/27, B 20/27. Nhưng hai checkpoint hỏng ở các episode khác nhau: 5 episode
    chỉ A thành công (100012, 100014, 100017, 100018, 100057), 6 episode chỉ B thành công (100008, 100010, 100015, 100049,
    100051, 100053). McNemar p = 1.0, tức chưa phân biệt được A với B trên các ca cùng phía.
  - Kết quả mỗi episode khó thường như tung đồng xu, nên cần nhiều episode hoặc chạy lặp.
  - B giữ được mức của A ở ca cùng phía (đúng tiêu chí ở mục 7), dù chỉ có khoảng 40% dữ liệu cùng phía so với A. Ưu thế của
    B phải đo trên tập đổi phía (`g1_can_sort_swap_hold_test`).
- **Chạy 20 s (`eval/B9000_ik_test_20s`, 8 episode đầu): `ok` 4/8, `ok_any` 7/8.** Chạy 10 s trên cùng 8 episode: 7/8.
  - **100046** (Pepsi đứng): xong lúc 5.5 s. Tới 7.7 s model gắp lon ra khỏi khay rồi đặt nằm trên bàn, sau đó đẩy lon ra
    mép. Khoảng 16 s tay phải còn đi tìm lon Coca. Kết thúc với 3 lần khép tay, lon Pepsi rơi khỏi bàn.
  - **100004** (Coca nằm): xong lúc 7.0 s. Khoảng 9–11 s tay thò vào khay định gắp lại, sau đó lon Pepsi bị xô 17.9 cm.
  - **100002** (Coca nằm): lần này xong lúc 6.0 s (ở lượt 10 s thì hỏng), robot đứng yên, nhưng lon còn lại lệch 3.2 cm, vượt
    ngưỡng 3 cm của `is_success`. Thực chất là thành công, trượt ngưỡng 2 mm.
  - **100049** (Pepsi nằm, sát lon Coca): lần đầu cánh tay trái quét đổ lon Coca, cả hai lon rơi khỏi bàn trước 7.6 s. 12 s
    còn lại robot khua tay trên bàn trống, không thể thử lại.
  - Kết luận:
    - Trong 8 episode này, thời gian thêm không cứu được ca nào nhờ thử lại.
    - 3/4 ca hỏng xảy ra **sau khi đã xong**. Model không biết việc đã xong khi đứng quá 2.5 s `hold` ở tư thế ready, nên bắt
      đầu lại task, đúng như dự đoán ở mục 5.2d.
    - Muốn đo thử lại phải kết thúc episode khi thành công.
  - **Công cụ:** thêm `eval_closed_loop.py --stop-on-success`, kết thúc episode ngay khi `is_success` đúng lần đầu, và ghi
    `stop_on_success` vào kết quả. Lượt 20 s nên chạy với cờ này.
  - Muốn model tự dừng thì cần thêm dữ liệu "đã xong" (lon nằm sẵn trong khay, action đứng yên) như đề xuất ở mục 5.2d. Khi
    triển khai thì dùng bộ phát hiện xong như `--stop-on-success` của GUI.
- **Việc làm được ngay, không cần train lại:** thêm cờ `--replan-steps` (mặc định 15 = 0.3 s) cho `eval_closed_loop.py` và
  `launch_vla_simulation.py`, và ghi `replan_steps` vào `results.jsonl`. Chunk dài 40 khung ở 25 Hz, tức 1.6 s = 80 bước 50 Hz.
  Thử 30 (0.6 s) để giảm số lần đổi ý; đổi lại, model sửa sai chậm hơn.
- **Thử `--replan-steps 30` (`eval/B9000_ik_test_replan30`, 10 episode đầu): 7/10**, trong khi bản 15 bước đạt 9/10 trên
  cùng 10 episode. Không giúp được gì.
  - 100002 hụt gần hơn (5.6 cm so với 14.4 cm), nhưng vẫn không kẹp được.
  - 100004 (Coca nằm, yaw 98°): vẫn đổi kiểu gắp lúc 2.4 s, từ tiếp cận ngang sang hạ từ trên xuống. Lon bị đẩy lăn rồi rơi.
  - 100049 (Pepsi nằm, sát lon Coca): cánh tay quét đổ lon Coca lúc 1.9–2.4 s. Chunk sai phải chạy tới 0.6 s mới được sửa.
  - Lý do: giãn khoảng gọi chỉ giảm số lần lấy noise mới (17 lần thay vì 34), mỗi lần vẫn có thể rơi vào kiểu kia; mà sửa
    sai lại chậm hơn. Giữ mặc định 15.
- **Tuỳ chọn khác:**
  - Server cố định seed noise, để hai lần gọi liên tiếp với quan sát gần nhau cho ra cùng một kiểu gắp.
  - Sửa planner cho kiểu gắp lon nằm nhất quán theo yaw, và phủ dày dữ liệu cùng phía quanh ranh giới đổi kiểu.

### 5.2h. Closed-loop B-9000 trên tập đổi phía, 20 s + `--stop-on-success` (25/09, `eval/B9000_swap_test_20s_stop`, đang chạy)
- 7 episode đầu: 4 thành công. Hỏng: 100046 (Pepsi) và 100002 (Coca) làm lon rơi khỏi bàn; 100001 (Coca) xô lon còn lại.
- **coke 100001** (Coca nằm, đổi phía, hai lon sát nhau ở giữa bàn):
  - Tay phải vươn chéo người, kẹp rất chuẩn (cách lon 0.75 cm lúc 2.4 s) và đặt đúng vào khay đỏ.
  - Nhưng lúc 0.7–3.7 s cánh tay đè qua lon Pepsi đang nằm, đẩy nó lăn. Lon Pepsi lệch tổng cộng 30.3 cm.
  - `is_success` yêu cầu lon còn lại lệch dưới 3 cm (demo nào xô lon đều đã bị loại), nên tính là hỏng. Bản cũ ghi nhầm lý do
    là "kẹp được nhưng rơi hoặc đặt sai chỗ".
  - Với `--stop-on-success`, episode chạy hết 20 s vì điều kiện thành công không bao giờ đúng.
- **Sửa công cụ:**
  - `collect_demos.is_success(..., check_other=False)` bỏ điều kiện lon còn lại; mặc định không đổi.
  - `eval_closed_loop` ghi thêm `in_tray` (lon nhiệm vụ nằm trong khay, đã nhả, lúc kết thúc), và có lý do hỏng mới "đặt
    đúng khay nhưng xô lon còn lại".
  - Đã thử `is_success` / `in_tray` bằng cách đặt lon tay vào khay. Chỉ áp dụng cho các lượt chạy sau.
- Xô lon còn lại là điểm khó riêng của ca đổi phía: tay đi chéo sát lon kia. Planner chỉ chừa được khoảng 1 cm (mục 4.5), nên
  model lệch khoảng 1 cm là đủ đụng lon.

### 5.2i. `--lower-body balance` làm B-9000 hỏng nặng: quỹ đạo demo phụ thuộc vào việc thân robot ngả (25/09)
- Closed-loop, 20 s + `--stop-on-success`, balance so với fixed:
  - `ik_test`: **18/40** (fixed 31/40).
  - `swap_test`, 15 episode chung: **5/15** (fixed 9/15). Mọi ca đổi phía đều hỏng (0/9); ca cùng phía 5/6.
  - Khoảng cách lúc khép tay TB 11.7 cm (fixed 4.3), nghiêng max 4.5° (fixed 14°). Lượt swap balance dừng ở 34/40 lúc 10:55
    vì file XML cảnh bị sửa hỏng (xem dưới).
- Balance vẫn bám góc eo mà VLA ra lệnh: đứng yên 3 s với lệnh waist_yaw ±0.6 thì eo thực ±0.63–0.64, pitch thấp hơn
  khoảng 0.02–0.04.
- **Phép thử quyết định:** phát lại nguyên action của 40 demo `g1_can_sort_swap_hold_test`, không dùng model, trên bản HEAD sạch:

  | Chế độ | Cùng phía | Đổi phía | Lon nằm / đứng | Pelvis dịch TB (max) |
  |---|---|---|---|---|
  | fixed | 14/14 | 26/26 | 28/28 / 12/12 | 5–7 cm (13.4) |
  | balance | **8/14** | **9/26** | 13/28 / 4/12 | 2.3–2.5 cm (6.6) |

  Balance xô lon còn lại ở 10/26 ca đổi phía.
- **Kết luận:**
  - Ở chế độ fixed, chân khoá nhưng thân vẫn ngả và pelvis trôi 5–13 cm khi vươn tay. Planner IK online đã bù chuyển động
    này, nên góc khớp trong demo gắn với cách thân ngả của chế độ fixed.
  - Balance giữ thân đứng yên, nên cùng góc khớp đưa tay tới chỗ lệch vài cm: hụt lon, và ở ca đổi phía thì quét trúng lon kia.
  - Model train trên dữ liệu fixed không thể chạy balance. Lần thử ở mục 5.2f đạt kết quả tốt chỉ vì planner chạy online và
    tự bù.
  - Muốn dùng balance thì phải **thu lại demo với `--lower-body balance`** rồi train lại. Đây là Phase 1 của plan mục 6.
  - Tới lúc đó, mọi eval dùng `fixed`.
- **Cảnh báo, 25/09 10:55:** `sim/unitree_g1/scene_pepsi_coke_sorting.xml` bị sửa ngoài phiên này. File có hai dòng
  `<body>` cho mỗi lon và khay, vị trí và ma sát khác, mất weld kẹp `coke_grasp`/`pepsi_grasp`, thêm weld
  `pelvis_standing_support`, và **không parse được** (dòng 89). `colab_gr00t_server.py` (bản cũ, action_space
  `sonic_tokens`) và `SIMULATION_READINESS_CHECKLIST.md` cũng đổi. **Đã khôi phục** cả ba file về bản commit `a5e70aa`
  theo yêu cầu người dùng (`git checkout --`): XML parse lại được, `verify_pipeline.py` qua 5/5 test.

### 5.2j. GUI: tay dị dạng khi gắp lon Coca nằm (25/09, tunnel `rxyom-…:36755`)
- Ảnh chụp khớp với `--scene-seed 200010 --swap-prob 0.5`: Coca nằm ở (0.360, −0.099), yaw gập 59°, cùng phía; Pepsi đứng
  ở (0.282, 0.077). Lon ở sát mép xa của vùng train (x tối đa 0.37). Trong ảnh, tay model thấp sát bàn và lon đã bị đẩy lệch.
- **Planner trên cùng cảnh:** gắp gọn, tay giơ cao rồi úp từ trên xuống (kiểu "+"), thành công.
- **Dữ liệu train quanh cảnh này** (Coca nằm cùng phía):
  - A (`g1_can_sort_ik`, planner cũ): 3/6 láng giềng gần nhất (ep 236, 346, 358) có cổ tay gập hết cỡ và xoắn
    (pitch 1.45, yaw −1.2..−1.56), đúng tư thế dị dạng ở mục 4.2b. Cả dataset có 15/181 episode lon nằm như vậy (Coca 8/92).
  - B (`g1_can_sort_swap_hold`, planner đã sửa): 0/225 tư thế dị dạng. Nhưng không có episode nào trong vòng 3 cm và 20°
    quanh cảnh này; gần nhất lệch 2.7 cm/21° và 4.5 cm/11°. Chỉ 10/39 episode Coca nằm cùng phía có x > 0.35.
- Chạy lại model trên cảnh này thì server không phản hồi (tunnel hết hạn), nên chưa xác định được checkpoint lúc đó.
  - Nếu là A-10000: model học đúng tư thế dị dạng có trong dữ liệu của A.
  - Nếu là B-9000: model phải ngoại suy ở mép vùng train, nơi dữ liệu thưa.
- Lần sau chạy GUI với `--record-vla vla_log_<tên>.npz` để lưu góc khớp model ra lệnh.

### 5.2k. Bước 0b: kiểm chứng nhận xét về dữ liệu B bằng `tools/demo_quality.py` (25/09)
- Công cụ mới:
  - `sim/episode_metrics.py` (`EpisodeMetrics`): khoảng hở nhỏ nhất (mj_geomDistance, geom va chạm của cánh tay gắp + lon đang
    cầm) tới lon còn lại theo pha trước kẹp / đang kẹp / sau nhả; chạm lon còn lại; ngả, trượt, yaw pelvis; thân dưới chạm bàn;
    vector lệch điểm gắp lúc khép tay trong khung pelvis.
  - `collect_demos.replay_actions(..., observer=...)` gọi observer sau mỗi bước sim.
  - `tools/demo_quality.py DATASET [--eval DIR ...] [--reuse]`: phát lại toàn bộ dataset (400 episode khoảng 3 phút, 12
    worker), ghi `eval/demo_quality/<dataset>.jsonl/.txt`, nối với eval theo (task, seed).
  - `tools/eval_closed_loop.py` ghi thêm `offset_at_close_cm` (lệch điểm gắp so với tâm lon trong khung pelvis) để lần eval
    sau xem được model lệch theo hướng nào. Các eval cũ chưa có trường này.
- Dataset train B (`g1_can_sort_swap_hold`, phát lại 400/400, chân fixed). Khoảng hở tính bằng cm: trung vị / tỉ lệ < 2 cm:

  | Nhóm | n | Trước kẹp | Đang kẹp | Chạm lon kia | Thân chạm bàn |
  |---|---|---|---|---|---|
  | Cùng phía đứng | 123 | 11.4 / 0% | 11.5 / 0% | 0 | 0 |
  | Cùng phía nằm | 77 | 7.2 / 10% | 8.4 / 4% | 0 | 17 |
  | Đổi phía đứng | 52 | 0.8 / 92% | 2.0 / 46% | 6 | 0 |
  | Đổi phía nằm | 148 | 2.1 / 49% | 1.7 / 55% | 13 | 39 |

  - Khớp với nhận xét: đổi phía 1.4 / 60% trước kẹp, 1.9 / 53% đang kẹp; 52/200 đổi phía là lon đứng; yaw ≥ 7.5° ở 10% ca đổi
    phía, max 12.6°; 19 demo (10% ca đổi phía) chạm hẳn lon kia.
  - Thân chạm bàn: **56** episode (nhận xét đếm 49, có lẽ chỉ tính pelvis), đều là lon nằm, trượt trung vị 13.2 cm, ngả 12.5°.
  - Demo khép tay cách tâm lon trung vị 0.9 cm. Ngả cuối episode khoảng 2.9°.
- Nối với B-9000, `swap_hold_test`, fixed (`eval/B9000_swap_test_20s_stop`), ca đổi phía: model ok 10 ep, demo đang kẹp
  trung vị 5.4 cm và không ca nào < 2 cm; model hỏng 16 ep, trung vị 2.6 cm, 44% < 2 cm. Lỗi "đặt đúng khay nhưng xô lon còn lại"
  (8 ep) có demo trung vị 1.6 cm. Lon đứng đổi phía: demo trước kẹp 0.4 cm, model 0/4.
- **Không phải do hai lon đặt sát nhau:** chia ca đổi phía theo khoảng hở hai lon lúc đầu: 0–5 cm model 2/4, 5–10 cm 7/12,
  ≥ 10 cm **1/10** (demo đang kẹp chỉ 2.6 cm, 50% < 2 cm). Khoảng hở nhỏ đến từ hình học đường mang lon, không phải từ cảnh. Vì vậy
  nới `--can-gap`/`--cross-can-y` không giúp được; bước 4 của plan (mang lon qua phía trên) là đúng hướng.
- **Tì bàn không hỏng chắc chắn:** trên `swap_hold_test` có 4 demo chạm bàn, model hỏng 3; trên `ik_test` (demo của dataset A)
  có 9 demo chạm bàn, B-9000 vẫn làm được 7. Nhận xét "4/4 hỏng" không đúng khi tính cả `ik_test`.
- Cùng phía: demo không sát lon kia (trung vị 8–11 cm) ở cả ca ok lẫn hỏng. Ca hỏng là do độ chính xác hoặc do không biết dừng.

### 5.2l. Bước 0: mốc A/B ở chế độ fixed, 20 s + `--stop-on-success` (25/09)
`eval/A10000_{ik,swap}_test_20s_stop` (server `/content/ckpt/g1_can_sort_ik_checkpoint-10000`, 3 và 2 lần không phản hồi),
so với `eval/B9000_{ik,swap}_test_20s_stop`. Cùng 40 cảnh mỗi tập, chân fixed, `--workers 4`.

| Tập / nhóm | n | A-10000 | B-9000 |
|---|---|---|---|
| `ik_test` tất cả | 40 | 30 (75%) | 31 (78%) |
| — lon đứng | 18 | 17 | 18 |
| — lon nằm | 22 | 13 | 13 |
| `swap_hold_test` tất cả | 40 | 14 (35%) | 23 (58%) |
| — cùng phía đứng | 8 | 8 | 8 |
| — cùng phía nằm | 6 | 6 | 5 |
| — đổi phía đứng | 4 | **0** | **0** |
| — đổi phía nằm | 22 | **0** | 10 |

- **A: lối tắt "lệnh → tay → lon cùng phía" đúng như dự đoán.** Ca đổi phía 0/26: 20 ca xô lon còn lại (tay đi về lon nằm bên
  mình), khép tay trung bình 6.8 lần mỗi episode. Ca cùng phía 14/14.
- Trên `ik_test` A và B ngang nhau (30 và 31) nhưng hỏng ở các episode khác nhau: cả hai ok 26, chỉ A 4, chỉ B 5, cả hai hỏng 5.
  Học thêm ca đổi phía không làm B kém đi ở ca cùng phía.
- Lon đứng đổi phía: cả A và B đều 0/4. Với B, demo nhóm này cách lon kia chỉ 0.4 cm trước khi kẹp (mục 5.2k).
- **Hướng lệch lúc khép tay** (trường mới `offset_at_close_cm`, A trên `ik_test`, trừ đi độ lệch của chính demo cùng cảnh,
  khung pelvis): model khép tay **thiếu tầm 1–3 cm theo x** (Coca nằm −3.1, Coca đứng −2.2, Pepsi đứng −1.3, Pepsi nằm −1.1 cm)
  và lệch ra phía ngoài 0.4–1.8 cm theo y. Độ lệch có hướng cố định, không phải nhiễu.
  - Không do thân ngả khác demo: pitch max lúc eval 7.1° so với demo 7.1°, tương quan giữa độ thiếu tầm và chênh pitch là 0.03
    (38 ep).
  - Giả thuyết còn lại: model khép tay **sớm** khi tay vẫn đang tiến vào lon (hướng tiến là +x). Kiểm tra được bằng cách ghi
    khoảng cách tay–lon vài bước sau lúc khép, hoặc xem video.

### 5.2m. Bước 1: `collect_demos.py --lower-body` và ghi chỉ số chất lượng lúc thu (25/09)
- `collect_demos.py`:
  - `--lower-body {fixed,balance}` (mặc định `fixed`), ghi `"lower_body"` vào `cfg`.
  - `setup_episode` lấy chế độ chân từ `cfg.get("lower_body", "fixed")`; `pipeline_kwargs` ghi đè được.
  - `run_episode` và `replay_actions` bước sim qua `pipe._lower_body(pipe.wbc.track_joint_targets(target))`.
  - Mỗi episode có `EpisodeMetrics` chạy cùng planner. `collection.jsonl` có thêm khoá `"quality"`. File mới
    `meta/rejected.jsonl` ghi mọi lần thử bị bỏ (task, seed, đứng/nằm, đổi phía, lý do, chỉ số tới lúc dừng).
  - `replay_actions(..., pipeline_kwargs=...)` để phát lại với chế độ chân khác.
- `verify_demos.py --lower-body`, `tools/eval_closed_loop.py --lower-body`, `launch_vla_simulation.py --lower-body`: mặc định
  `None` = theo `cfg` của episode (dataset cũ: fixed). **Đổi hành vi:** trước đây mặc định cứng là `fixed`; để eval checkpoint C
  trên tập test cũ với balance phải truyền `--lower-body balance`. GUI lấy theo episode đầu của `--scene-dataset`.
- Kiểm tra:
  - `verify_pipeline.py` 5/5.
  - `verify_demos.py datasets/g1_can_sort_swap_hold`: 400/400, lệch khớp tay TB 0.046 rad (max 0.407).
  - Sinh thử 4 episode balance (`--swap-prob 0.5`, 10 lần thử, 1 lần bỏ vì "đặt lon không thành công" với khoảng hở lúc đang kẹp
    −0.2 cm, tức lon đang cầm đè vào lon kia) rồi phát lại: 4/4, lệch khớp TB 0.047 rad, bằng mức của dataset fixed. Phát lại
    với balance vẫn tất định đủ dùng.
- **Cần xem ở bước 2:** với balance, ngả và trượt nhỏ hơn fixed (ngả 2–5°, trượt 2–5 cm), nhưng **yaw pelvis lệch 7.4–9.6°** ở
  3/3 ca đổi phía (fixed: trung vị 1.8°). Policy thăng bằng nhận góc eo yaw thành hướng thân mong muốn, có thể xoay cả pelvis
  thay vì chỉ xoay eo. Vượt ngưỡng 3° của bước 2.

### 5.2n. Bước 2: chạy thử planner với balance, so với fixed trên cùng seed (25/09, `eval/planner_trial_bal`)
- Công cụ mới `tools/planner_trial.py`: chạy `collect_demos.run_episode` trên seed cố định (Coca và Pepsi, seed 0–39) cho từng
  chế độ chân, không ghi dataset. In tỉ lệ thu được theo 4 nhóm, lý do bỏ, bảng chất lượng của `demo_quality`. 160 lần thử
  mất 6.5 phút (12 worker). `--swap-prob 0.5`, các tham số khác như dataset B.
- `EpisodeMetrics` có thêm `final_yaw_deg`, `max_foot_yaw_deg`, `max_foot_slide_cm` (bàn chân `*_ankle_roll_link`).

| Nhóm | fixed | balance |
|---|---|---|
| Tất cả | 54/80 (68%) | **61/80 (76%)** |
| Cùng phía đứng | 20/20 | 20/20 |
| Cùng phía nằm | 18/20 | 19/20 |
| Đổi phía đứng | 7/24 (29%) | **12/24 (50%)** |
| Đổi phía nằm | 9/16 | 10/16 |

| Chỉ số (episode thu được) | fixed (54) | balance (61) |
|---|---|---|
| Thân dưới chạm bàn | 12 (22%) | **0** |
| Ngả max, trung vị / max | 6.1° / 13.5° | **2.6° / 5.7°** |
| Pelvis trượt, trung vị / max | 5.5 / 13.5 cm | **2.2 / 5.3 cm** |
| Ngả cuối episode, trung vị | 2.9° | **0.4°** |
| Bàn chân xoay, trung vị / max | 1.4° / 10.8° | **1.5° / 4.2°** |
| Bàn chân trượt, max | 3.1 cm | **0.7 cm** |
| Yaw pelvis max, ca đổi phía trung vị / max | 1.5° / 10.4° | 8.3° / 11.3° |
| Yaw pelvis cuối, max | 5.2° | 4.0° |
| Đổi phía: khoảng hở trước kẹp, trung vị / < 2 cm | 1.7 cm / 62% | 1.2 cm / 77% |
| Chạm lon kia | 1 | 4 (3 ở lon đứng đổi phía) |

- **Yaw pelvis lớn ở balance không phải trượt chân.** Truy vết một episode đổi phía: eo xoay +18° thì pelvis xoay −8° qua khớp
  hip yaw (`hip_yaw_l` 5 → 12°), bàn chân xoay dưới 1.7° và dời dưới 4 mm; eo về 0 thì pelvis về −1.9°. Policy thăng bằng xoay
  pelvis ngược chiều eo để giữ mô-men, rồi tự trả lại. Planner bù theo pose pelvis đo được nên demo vẫn nhất quán. Tiêu chí
  "yaw < 3°" của plan đổi thành **bàn chân xoay < 3°** (trượt thật). NVIDIA không có bộ giữ hướng: muốn ra lệnh yaw rate thì
  policy chuyển sang Walk (`|cmd| ≥ 0.05`, `g1_gear_wbc_policy.py`), nên không sửa theo hướng đó.
- **Điều kiện đi tiếp: đạt**, với vài ngoại lệ:
  - Tỉ lệ thu được của balance ≥ fixed ở mọi nhóm; phát lại 61/61; không chạm bàn.
  - Ngả > 5°: 3/61 (max 5.7°). Bàn chân xoay > 3°: 3/61 (max 4.2°). Yaw pelvis cuối > 3°: 2/61.
  - Giữ ngưỡng 5° và 3° làm bộ lọc ở bước 7, mất khoảng 8% episode, quota sẽ sinh bù.
- **Ca đổi phía còn sát lon kia hơn** so với fixed: trước khi kẹp 77% dưới 2 cm, 3/12 ca lon đứng chạm lon kia. Bước 4 càng cần.
- Lon đứng đổi phía: balance thu được 12/24. 11/12 lần bỏ là "tay va chạm khi tiến vào lon", đầu vào cho bước 5.

### 5.2o. Bước 4: chừa khoảng hở với lon còn lại (25/09, `eval/planner_trial_step4{,b,c,d}`)
- Đo theo đoạn (nhãn mới `PickPlacePlanner.phase`: reach/approach/close/lift/carry/lower/retreat), ca đổi phía, balance,
  trước khi sửa: sát lon kia ở **hai** chỗ, không chỉ lúc mang lon:
  - Vươn tay + tiến vào lon: cổ tay/ngón tay cách lon kia trung vị 1.1 cm (lon đứng). Lon đứng tiếp cận theo pháp tuyến lòng
    bàn tay; khi gắp chéo người, điểm chờ lùi 7 cm về phía lon kia.
  - Mang lon: lon đang cầm (lon nằm) hoặc cổ tay (lon đứng) cách lon kia trung vị 1.6–3.0 cm, vì đáy lon đang cầm (~10 cm) thấp
    hơn đỉnh lon kia (11.8 cm), cả ở điểm trên khay.
- Sửa `sim/pick_place_planner.py`:
  - `APPROACH_MARGINS` (5, 4, 3, 2, 1 cm) cho vươn tay/hạ xuống/tiến vào; `CARRY_MARGINS` (6, 4, 2, 1 cm) cho mang lon. Thử từ lớn
    tới nhỏ, nên không mất episode. Đường nhấc lon và hạ lon xuống khay vẫn 1 cm (xuất phát/kết thúc sát lon kia là do cảnh).
  - Lon đứng có thêm 2 điểm chờ (`UPRIGHT_PRE_OFFSETS`: lùi ít theo lòng bàn tay, nhiều theo ngón tay hoặc lên cao).
  - Lon kia nằm gần đường mang lon (`CARRY_CORRIDOR` 15 cm): nhấc lon tới khi đáy cao hơn đỉnh lon kia `CARRY_CLEARANCE` 4 cm
    (nhấc 14–16 cm thay vì 10), điểm trên khay ở cùng độ cao. Số bước nhấc và hạ tăng theo quãng đường.
  - Khoảng hở > 1 cm đo bằng `mj_geomDistance` (`_too_close`). **`geom_margin` với mesh bàn tay chỉ đúng một phần:** lập với
    margin 5 cm, khoảng cách thật trên kế hoạch chỉ 3.2 cm, và kết quả chạy không đổi so với margin 3 cm.
  - Lon nằm: ở các mức khoảng hở > 1 cm chỉ xét 2 pose gắp đầu. Xét cả pose xếp sau thì "gắp trượt" tăng (lon nằm đổi phía thu
    được 12 → 9).
  - `planner.info` và `plan` trong kết quả `run_episode` / `collection.jsonl` ghi mức khoảng hở đã chọn, điểm chờ, độ cao nhấc,
    điểm trên khay.
- **Tay chạy thật lệch khỏi kế hoạch 2–3 cm** (khớp tay bám chậm theo PD giữa đoạn vươn tay, cổ tay lệch xuống ~2 cm), ở cả fixed
  lẫn balance. Kế hoạch 3 cm thì lúc chạy còn trung vị 1.5 cm. Vì vậy mức đầu của các bậc lớn hơn khoảng hở mong muốn.
- Kết quả, balance, 80 seed (40 mỗi lon, `--swap-prob 0.5`), ca đổi phía:

  | | Trước bước 4 | Sau (`step4d`) |
  |---|---|---|
  | Thu được, đổi phía | 22/40 | **31/40** (đứng 20/24, nằm 11/16) |
  | Thu được, tất cả | 61/80 | **70/80** |
  | Trước khi kẹp: trung vị / < 2 cm | 1.2 cm / 77% | **2.9 cm / 39%** |
  | Đang kẹp: trung vị / < 2 cm | 2.1 cm / 41% | **5.1 cm / 10%** |
  | Đoạn mang lon, lon đứng: < 2 cm | 25% | **0%** |
  | Chạm lon kia | 3 | 2 |

  Ca cùng phía giữ nguyên (khoảng hở ≥ 7 cm, không ca nào < 2 cm).
- Chưa đạt ngưỡng plan (sau kẹp < 2 cm dưới 5%, trước kẹp dưới 15%). Phần còn lại:
  - Đoạn tiến vào lon đứng (30% < 2 cm): tay bám chậm, và 4/20 ca phải hạ xuống mức 2 cm mới lập được.
  - Đoạn nhấc lon ngay sau khi kẹp: sát lon kia khi hai lon đặt gần nhau (hở 3.2 cm).
  - **Đề xuất lọc ở bước 7:** bỏ episode có min(trước kẹp, đang kẹp) < 1 cm hoặc chạm lon kia: mất 7/31 ca đổi phía (23%). Sau
    lọc, tỉ lệ thu được ở ca đổi phía còn khoảng 24/40, vẫn cao hơn trước bước 4 (22/40, chưa lọc).
- `verify_pipeline.py` 5/5.
- **Bộ lọc chất lượng đưa vào lúc thu** (thay cho lọc riêng ở bước 7): `collect_demos.py --quality-filter` (mặc định bật, ghi
  `"quality_filter"` vào `cfg`). `run_episode` gọi `EpisodeMetrics.violation()` sau mỗi bước và dừng sớm, lý do
  `"chất lượng: ..."` vào `rejected.jsonl`. Ngưỡng ở đầu `sim/episode_metrics.py`:
  - chạm lon còn lại, hoặc khoảng hở < 1 cm (`MIN_OTHER_CLEAR`), trước khi kẹp / đang kẹp;
  - chỉ với chân balance: thân dưới chạm bàn, ngả > 5°, bàn chân xoay > 3°.
- Planner: `LYING_CLEAR_GRASPS = 4` (lon nằm, số pose gắp xét ở mức khoảng hở > 1 cm). So với 2 thì sát lon ít hơn (5 → 2 ca)
  nhưng ngả > 5° nhiều hơn (0 → 3); giữ 4.
- Chạy thử có lọc (`eval/planner_trial_filter4`, balance, 80 seed): thu được **59/80** (cùng phía đứng 19/20, cùng phía nằm
  17/20, đổi phía đứng 16/24, **đổi phía nằm 7/16**). Mọi episode giữ lại: không chạm lon kia, không chạm bàn, đang kẹp không ca
  nào < 2 cm (trung vị đổi phía 5.8 cm), trước kẹp đổi phía < 2 cm 22% (trung vị 3.8 cm), ngả max 4.3°.
  - Lon nằm đổi phía giờ là nhóm khó thu nhất (44%): quota ở bước 7 phải thử khoảng 2.3 lần; cần xem vùng phủ ở bước 5.

### 5.2p. Bước 3: kiểu gắp lon nằm theo quy tắc cố định (25/09, `eval/planner_trial_modes_{cost,rule}`)
- Quét 300 cảnh lon nằm (`plan_grasps`, không chạy sim): hai kiểu gắp = hai chiều trục cuộn ngón dọc trục lon (`finger_axis`,
  góc φ trong world). Với góc tham chiếu ψ, kiểu ưu tiên là kiểu có cos(φ − ψ) > 0. Quét ψ theo từng tay:
  - Tay phải (Coca) ψ = 125°: chi phí tăng TB 0.06 (cùng phía) / 0.33 (đổi phía), 2 + 2 cảnh không có kiểu ưu tiên.
  - Tay trái (Pepsi) ψ = 215°: tăng 0.15 / 0.20, 0 + 1 cảnh. Pepsi vốn gần như theo quy tắc; Coca chọn lẫn lộn trong φ 0–225°.
- `sim/pick_place_planner.py`: `LYING_MODE_REF_DEG = {"right": 125, "left": 215}`. Pose gắp chia hai nhóm theo kiểu, nhóm ưu
  tiên thử trước (với mọi mức khoảng hở và điểm chờ), nhóm kia chỉ khi nhóm ưu tiên không lập được. `plan_grasps` trả thêm
  `finger_axis`, `cost`. `planner.info`/`plan` ghi `wrist_pitch`, `finger_axis_deg`, `mode_fallback`.
- Công cụ: `tools/grasp_modes.py` (bảng kiểu gắp theo tay × cùng/đổi phía × yaw 20°, số lần đổi kiểu khi xếp theo yaw; dataset
  cũ thì xấp xỉ bằng wrist pitch ở khung 55), `tools/planner_trial.py --lying-mode {rule,cost}` và `--lying-prob`.
- So sánh 300 cảnh lon nằm (seed 1000–1149 mỗi lon, `--swap-prob 0.5 --lying-prob 1.0`, balance, có lọc):

  | | Theo chi phí | Theo quy tắc |
  |---|---|---|
  | Thu được | 190/300 (63%) | 174/300 (58%) |
  | — cùng phía / đổi phía | 115/154 / 75/146 | 108/154 / 66/146 |
  | Ô yaw 20° có ≥ 90% cùng kiểu | 25/33 | **28/33** |
  | Số lần đổi kiểu theo yaw: Coca cùng / Coca đổi / Pepsi cùng / Pepsi đổi | 6 / 8 / 14 / 4 | **2 / 6 / 6 / 4** |
  | Dùng kiểu còn lại (`mode_fallback`) | – | 5 |

  - "Đổi kiểu" tính cả chỗ nối 180° ≡ 0° (định nghĩa kiểu "+" đảo dấu ở đó), nên 2 là mức tối thiểu: một ranh giới thật.
  - Coca cùng phía (nhóm hỏng ở closed-loop B, mục 5.2g) giờ chỉ đổi kiểu một lần ở yaw ~40°; vùng 90–120° hết lẫn.
  - Ô còn lẫn: ranh giới của Pepsi cùng phía (120–140°, do góc tham chiếu 215°) và các ca `mode_fallback`.
- Tỉ lệ thu được giảm 5.3 điểm (ngưỡng plan 5): kiểu ưu tiên lập được nhưng lúc chạy hỏng nhiều hơn (gắp trượt, sát lon kia,
  ngả > 5°), chủ yếu ở đổi phía yaw 30–90°. **Giữ quy tắc**: lỗi đổi kiểu giữa chừng đã làm hỏng closed-loop thật, còn số
  lượng thì quota bù được. Vùng thiếu dữ liệu xử lý ở bước 5.
- **Ngưỡng bàn chân xoay nâng 3° → 5°** (`MAX_FOOT_YAW_DEG`): với balance, p90 = 3.0°, lúc không lọc max 4.2°. Ngưỡng 3° loại
  21–28/300 episode lon nằm bình thường. Trượt chân thật ở chân fixed tới 10.8°.

### 5.2q. Bước 5: vùng phủ ca đổi phía (25/09, `eval/planner_trial_{swaplying,swaplying2,swapup,swapup4,step5b}`)
- Công cụ `tools/trial_coverage.py TRIAL [--swapped|--same-side]`: tỉ lệ thu được theo x, |y|, khoảng hở hai lon, yaw lon nằm;
  lý do loại; ảnh `coverage.png` (vị trí thu được / bị loại, histogram yaw).
- **Lon nằm đổi phía, yaw 30–120° gần như bị loại hết** (3/20 ở 90–120°):
  - Coca: phần lớn do ngưỡng bàn chân xoay 3° (đã nâng lên 5°, mục 5.2p).
  - Pepsi và Coca còn lại: "gắp trượt", bàn tay khép cách tâm lon 5–7 cm, lon gần như không dời (< 1.3 cm). Nguyên nhân: servo đoạn
    tiến vào lon giữ nguyên góc eo, pelvis xoay ngược chiều eo ~9° (balance), tay vươn chéo người gần giới hạn không bù nổi; IK
    không tới ở cả 10 bước cuối (ca thành công: 0/10) nên quay về tư thế đã lập, bàn tay lệch ~5 cm.
  - Sửa `_servo`: khi chỉ khớp tay không tới, giải lại cho **waist_yaw** tham gia (waist_pitch vẫn giữ). 120 cảnh lon nằm đổi
    phía: 71 → **92/120**, gắp trượt 24 → 4, yaw 60–90° 10/21 → 18/21, 90–120° 7/15 → 14/15. Yaw pelvis lúc gắp chéo tăng (trung
    vị 8.3 → 11.1°), vẫn tự trở lại (yaw cuối ≤ 3.9°), bàn chân xoay max 4.2°.
- **Lon đứng đổi phía, hai lon cách nhau 5–10 cm: 0/50** (≥ 15 cm: 35/42), tổng 47/120. Gắp lon đứng là bàn tay dựng đứng, lòng
  bàn tay hướng vào lon; gắp chéo người thì bàn tay chen giữa hai lon và mọi điểm chờ đều lùi ngang về phía lon còn lại.
  - Thử pose bàn tay lật ngược (lòng bàn tay quay sang phía kia): IK tới được tư thế gắp nhưng không episode nào dùng được
    (điểm chờ lùi ra ngoài tầm với). Đã bỏ.
  - Thêm 3 điểm chờ không lùi ngang vào `UPRIGHT_PRE_OFFSETS`: hạ thẳng từ trên (0, 0, 9 cm), tiến dọc ngón tay (0, 8, 2 cm),
    kết hợp (1.5, 5, 5 cm). 120 cảnh: 47 → **84/120**, hai lon cách 5–10 cm 0/50 → 24/50, |y| < 7 cm 9/58 → 30/58. 59/84 episode
    thu được dùng điểm chờ hạ thẳng từ trên. Lon đứng cùng phía vẫn dùng điểm chờ cũ (19/19).
- Kiểm tra lại 80 seed trộn (`eval/planner_trial_step5b`, balance, có lọc): **68/80** (trước bước 5: 59/80).

  | Nhóm | Trước bước 4 (không lọc) | Sau bước 4 + lọc | Sau bước 3 + 5 |
  |---|---|---|---|
  | Cùng phía đứng | 20/20 | 19/20 | 19/20 |
  | Cùng phía nằm | 19/20 | 17/20 | 17/20 |
  | Đổi phía đứng | 12/24 | 16/24 | **19/24** |
  | Đổi phía nằm | 10/16 | 7/16 | **13/16** |

  Phát lại 68/68; không chạm lon kia, không chạm bàn; đang kẹp ≥ 1 cm (đổi phía < 2 cm: 6%); ngả max 4.8°.
- Còn lại: hai lon cách nhau < 10 cm ở lon đứng đổi phía vẫn chỉ thu được khoảng một nửa (loại vì sát lon kia < 1 cm lúc chạy).
- `verify_pipeline.py` 5/5.

### 5.2r. Bước 6: episode "đã xong" (25/09)
- `collect_demos.py --done-prob P` (ghi `done_prob` vào `cfg`): với xác suất P, lon nhiệm vụ nằm sẵn trong khay đích (lệch tâm
  khay ≤ 3 cm, giữ đứng/nằm và hướng đã lấy mẫu), robot về tư thế ready từ tư thế tay có nhiễu rồi đứng yên `DONE_SECONDS` = 4 s
  (100 khung). Mục đích: model thấy lon đã trong khay thì không gắp lại (lỗi 2/3 ca cùng phía hỏng của B, mục 5.2g).
- `sample_randomization` bốc cờ `done` bằng bộ sinh số riêng `default_rng([seed, 7])` sau mọi số ngẫu nhiên khác, nên cảnh của
  dataset cũ và của episode thường không đổi. `apply_randomization` đặt lon vào khay khi `done`, nên phát lại, eval và GUI
  (`--scene-dataset`) dựng đúng cảnh. `collection.jsonl` có khoá `"done"`.
- Kiểm tra: sinh 6 episode `--done-prob 1.0` (balance): phát lại 6/6, lệch khớp 0; ảnh khung đầu/cuối đúng (lon trong khay).

### 5.2s. Bước 7: quota theo nhóm và sinh dataset C (25/09)
- `collect_demos.py --group-quota N [--done-per-task K --done-prob P]`: mỗi lon N episode cho từng nhóm (`GROUPS`: cùng/đổi
  phía × đứng/nằm) cộng K episode "đã xong". Nhóm của seed tính trước bằng `scene_group()` (từ `sample_randomization`), nên chỉ
  chạy seed thuộc nhóm còn thiếu; số lần thử mỗi đợt ước theo tỉ lệ thu được của từng nhóm. Không có `--group-quota` thì chạy
  như cũ (`--episodes-per-task`). Thống kê cuối in theo (lon, nhóm).
- Thử `--group-quota 2 --done-per-task 1`: 18 episode, 46 lần thử, 107 s (12 worker), phát lại 18/18.
- Lệnh sinh (log `eval/collect_bal.log`, `eval/collect_bal_test.log`):
  ```bash
  collect_demos.py --out datasets/g1_can_sort_bal --group-quota 50 --done-per-task 20 --done-prob 0.1 \
      --swap-prob 0.5 --lying-prob 0.5 --lower-body balance --seed 0 --workers 12          # 440 episode
  collect_demos.py --out datasets/g1_can_sort_bal_test --group-quota 5 --done-per-task 2 --done-prob 0.1 \
      --swap-prob 0.5 --lying-prob 0.5 --lower-body balance --seed 100000 --workers 12     # 44 episode
  ```
  Mặc định đã bật `--quality-filter`, `--hold-seconds 2.5`.
- Đợt đầu giả định tỉ lệ thu được 0.5 nên chạy 1 110 lần thử cho 440 episode (41 phút); đã sửa giả định thành 0.7.
- **`g1_can_sort_bal`: 440 episode, 99 343 khung**, 224 đứng / 216 nằm, 214 đổi phía, 40 "đã xong". Zip 434 MB.
  **`g1_can_sort_bal_test`: 44 episode, 9 953 khung** (4 "đã xong"). Zip 43 MB.
  - Tỉ lệ thu được (111 lần thử mỗi nhóm): cùng phía đứng 107 / 107, cùng phía nằm 87 / 92, đổi phía nằm 94 / 90, đổi phía đứng
    **72 / 77** (Coca / Pepsi). Đổi phía đứng bị loại vì sát lon kia (38) và không với tới tư thế gắp (32).
  - Phát lại 440/440 và 44/44 (lệch khớp TB 0.036 / 0.033 rad).
  - `demo_quality`: không chạm lon kia, không chạm bàn; ngả max 4.9°; ngả cuối trung vị 0.4°; bàn chân xoay max 4.7°. Đổi phía:
    trước kẹp trung vị 3.8 cm (13% < 2 cm), đang kẹp 5.1 cm (6% < 2 cm). Cùng phía ≥ 8.5 cm.
  - `grasp_modes` (bỏ episode "đã xong"): cả 4 nhóm lon nằm đổi kiểu đúng 2 lần (một ranh giới + chỗ nối 180° ≡ 0°); ô đạt 33/36,
    ô chưa đạt là ô chứa ranh giới.
  - Vùng phủ đổi phía (`datasets/g1_can_sort_bal_swap_coverage.png`, tính cả lần bị loại): lon đứng khi hai lon cách < 5 cm chỉ
    1/27 (khoảng 2% mọi cảnh), 5–10 cm 19/49; lon ở xa (x > 0.33 hoặc |y| > 0.11) thu được 40–50%. Các vùng khác ≥ 70%.
  - Ảnh vị trí lon: `datasets/g1_can_sort_bal_can_positions.png`.
- Train C: B thực tế chạy **batch 80 × 10 000 bước** (800k mẫu / 93 200 khung ≈ 8.6 lần đi qua dữ liệu; batch theo người dùng,
  số bước giả định). Giữ batch 80 và cùng số lần đi qua dữ liệu: **`MAX_STEPS` ≈ 8.6 × 99 343 / 80 ≈ 10 700**, `SAVE_STEPS` 1000,
  LR 1e-4 mặc định, từ base `nvidia/GR00T-N1.7-3B`. GPU H100 80 GB (batch 80 vừa như A100 80 GB).

### 5.2t. Notebook fine-tune: chép mọi checkpoint về Drive, đánh giá mọi checkpoint (25/09)
- Trước đây Bước 10 chỉ chép checkpoint mới nhất, Bước 11 chỉ đánh giá checkpoint mới nhất. `examples/finetune.sh` của
  Isaac-GR00T còn đặt `--save_total_limit 5`, nên trên `/content` checkpoint cũ bị xoá trong lúc train (10 700 bước: mất 5000).
- `tools/build_colab_notebooks.py` (notebook sinh lại, `verify_pipeline.py` 5/5):
  - **Bước 8a (mới, chạy ngay sau Bước 8, trước 8c):** luồng nền mỗi phút chép checkpoint từ `SYNC_FROM_STEP` (mặc định 5000)
    về `MyDrive/gr00t/checkpoints/<DATASET_NAME>/`. Checkpoint "xong" khi có `trainer_state.json` và 60 s không file nào đổi.
    Chép vào `_partial_checkpoint-N` rồi đổi tên; kiểm tra chỗ trống Drive trước khi chép. Khi train dừng thì đợi 65 s rồi chép
    lượt cuối (checkpoint cuối). Chạy lại cell để xem nhật ký; runtime khởi động lại thì chạy lại để chép nốt.
  - **Bước 10:** `COPY_STEPS` = `"latest"` / `"all"` / danh sách bước; bỏ qua checkpoint đã có trên Drive.
  - **Bước 11:** đánh giá mọi checkpoint từ `EVAL_FROM_STEP` trên toàn bộ tập test (ưu tiên bản trên ổ tạm, không còn thì lấy
    trên Drive), in bảng MSE/MAE theo checkpoint, chọn checkpoint MAE test thấp nhất, rồi đánh giá nó thêm trên 40 episode
    train và in các bảng chi tiết như trước.
  - Mặc định Bước 4 đổi sang C: `g1_can_sort_bal`, batch 80, `MAX_STEPS` 10 700.
- Thử luồng chép ở máy local với thư mục giả: chép đúng 5000/6000/10700, bỏ 4000 (dưới mốc) và checkpoint vừa ghi (< 60 s).
- Dung lượng: mỗi checkpoint khoảng 10–15 GB; 5000 → 10700 là 7 checkpoint.
- **GPU G4 của Colab (25/09):** NVIDIA RTX PRO 6000 Blackwell Server Edition, 97 887 MiB, driver 580.82 (CUDA 13.0). Trong
  `.venv` của Isaac-GR00T: torch 2.9.0+cu128 có `sm_120` trong arch list; `flash_attn_func` chạy được trên GPU (bf16).
- **Kế hoạch train C trên G4: batch 80 × 10 700 bước**, `SAVE_STEPS` 1000, `SYNC_FROM_STEP`/`EVAL_FROM_STEP` 5000, LR 1e-4 (mặc
  định notebook). G4 có 96 GB nên batch 80 (đã vừa trên A100 80 GB khi train B) chắc chắn vừa, và C chỉ khác B ở dữ liệu.
  - Đã cân nhắc batch 128 × 6 700 bước (cùng ~8.6 lần đi qua dữ liệu). Cùng tổng số mẫu nên chỉ nhanh hơn chút ít, nhưng số lần
    cập nhật ít hơn 37% với cùng LR; C sẽ khác B cả dữ liệu lẫn cách train, nên nếu C chỉ hơn/kém B một ít thì không tách được
    nguyên nhân. Chỉ dùng khi thiếu thời gian GPU (mốc 3000; hết VRAM thì 112 × 7 600 hoặc 96 × 8 900, mốc 4000), và ghi rõ khi
    so kết quả.

### 5.3. Notebook server: chép checkpoint về đĩa local, server chạy nền (24/09)
- Lần đầu chạy `checkpoint-10000`: nạp thẳng từ Drive mất **13 phút** chỉ riêng `Loading checkpoint shards` (3 shard, 260 s/shard,
  khoảng 15 MB/s). Lúc đó cell server chiếm notebook, nên tunnel (mở từ trước) đã tiêu gần nửa thời hạn 60 phút; muốn mở tunnel
  mới thì phải dừng server rồi nạp lại từ đầu.
- Sửa trong `tools/build_colab_notebooks.py`, thứ tự cell mới:
  - Bước 4 (`%%writefile`) ghi file server.
  - Bước 5 chọn `MODEL_PATH`/`EMBODIMENT_TAG`/`POLICY`. Checkpoint trên Drive được chép về `/content/ckpt/<exp>_<checkpoint>`
    (bỏ qua file đã đủ kích thước, in MB/s từng shard). Server chạy nền (`start_new_session`, log `/content/server.log`).
    Cell chờ tới dòng "ĐANG LẮNG NGHE", nếu server thoát thì in cuối log.
  - Bước 6 mở tunnel **sau** khi server sẵn sàng. Chạy lại được bất cứ lúc nào: đóng tunnel cũ, in giờ hết hạn, in sẵn lệnh
    `eval_closed_loop.py` và lệnh GUI.
  - Bước 7 in trạng thái (port, số request, suy luận TB, số Traceback, cuối log). Bước 8 dừng server (`pkill -f`),
    chỉ khi đặt `STOP_SERVER = True`: bản đầu dừng vô điều kiện nên "Chạy tất cả" tắt luôn server vừa khởi động.
- Lần chạy đầu trên Colab: server nạp xong và lắng nghe (warm-up 3.2 s); tổng thời gian chép + nạp chưa ghi lại.
- **Sự cố 24/09:** notebook sinh lại đưa `MODEL_PATH` về mặc định `nvidia/GR00T-N1.7-3B` (REAL_G1), nên server chạy **model base**.
  Bản thử 4 episode `eval/ik10k_smoke` (seed 100044/100001/100002/100046) cả 4 đều "không khép tay": hai tay giơ lên vẫy,
  một lần robot ngã (ngả 87.4°). Kết quả này **không phải** của checkpoint-10000. `eval_closed_loop` không bắt được lỗi này,
  vì model base cũng trả `action_space = real_g1`.
- Chống lặp lại:
  - Bước 5 giờ mặc định là `g1_can_sort_ik/checkpoint-10000` + `new_embodiment`, và in model đang nạp; Bước 7 cũng in model.
  - Server gửi `model` = {`model_path`, `embodiment_tag`} trong mỗi phản hồi; client lưu vào `server_model`.
  - `eval_closed_loop.py --expect-embodiment` (mặc định `new_embodiment`, `''` để tắt): dừng nếu server báo embodiment khác.
    Script ghi `server_model` vào `results.jsonl`, in ra sau episode đầu và ghi vào `summary.txt`.
  - Đã thử với server giả: embodiment REAL_G1 thì script dừng kèm thông báo, `new_embodiment` thì chạy bình thường;
    `verify_pipeline.py` qua cả 5 test. Server bản cũ không gửi `model` thì script chỉ cảnh báo.
- Tốc độ chép Drive → local trên Colab chưa đo; xem số MB/s in ra ở Bước 5 lần tới.
- **25/09: Bước 5 chạy rất lâu mà không in gì, lần trước còn lỗi `OSError: [Errno 107] Transport endpoint is not connected`.**
  - Chậm: bản cũ chỉ in sau khi chép xong từng shard. Với khoảng 15 MB/s, mỗi shard vài GB mất vài phút, nên cell đứng im
    ở dòng "Mounted at /content/drive". Checkpoint vừa ghi từ runtime train phải tải lại từ cloud nên không có cache.
  - Errno 107: FUSE của Drive bị rớt khi đọc file lớn. Bản cũ dừng hẳn, phải mount lại tay và chép lại từ đầu.
  - Sửa (`tools/build_colab_notebooks.py`, sinh lại `colab_gr00t_server.ipynb`):
    - In tổng dung lượng lúc đầu, rồi in tiến độ, MB/s và thời gian còn lại mỗi 30 s.
    - Chép theo khối 64 MB vào `.part` và nối tiếp nếu bị ngắt.
    - Gặp Errno 107 hoặc EIO thì `drive.mount(force_remount=True)` rồi chép tiếp, tối đa 5 lần.
  - Đã thử local với Drive giả, bị rớt 2 lần giữa shard: chép tiếp đúng, md5 khớp.

---

## 6. Mở rộng vùng gắp: phân tích tầm với và plan (24/09, chưa làm)

### Vì sao lon chỉ ở x 0.26–0.37 m
Số đo từ model sim:
- Vai phải nằm ở (0.03, −0.10) so với pelvis, cao hơn mặt bàn 0.40 m (mặt bàn cao 0.68 m, bàn đặt ở x = 0.48 nên mép bàn ở x = 0.18).
- Cánh tay trên 0.21 + cẳng tay 0.10 + cổ tay tới điểm gắp 0.18 = **0.49 m**.
- Vai cao hơn tâm lon đứng khoảng 0.35 m, nên phần vươn ngang tối đa là √(0.49² − 0.35²) ≈ 0.35 m tính từ vai (≈ 0.38 m từ pelvis).
  Thực tế ngắn hơn: phải xoay đúng hướng bàn tay, tư thế duỗi hết là điểm kỳ dị, eo chỉ cúi tới 0.45 rad (khớp cho phép 0.52),
  và robot không có thăng bằng.
- Giới hạn dưới 0.26: tay gập sát thân, khuỷu và bàn tay dễ va vào thân robot hoặc mép bàn.

**Lưu ý:** đứng sát bàn và bước tới **không tăng tầm với của tay**, chỉ dời vùng với vào sâu hơn trên mặt bàn. Hông cao 0.79 m,
cao hơn mặt bàn, nên chỉ tiến thêm được khoảng 8 cm là hông chạm mép bàn. Tầm với theo chiều sâu chỉ tăng nhờ cúi người có giữ
thăng bằng. Bước đi chủ yếu mở rộng vùng gắp **dọc theo bàn**.

### Plan đã duyệt (chi tiết: [`docs/PLAN_REACH_EXPANSION.md`](docs/PLAN_REACH_EXPANSION.md))
Mục tiêu: gắp được lon **sâu hơn** trên bàn và **rộng hơn** dọc theo bàn. Làm **sau** lần fine-tune trên dataset IK,
để có mốc đối chứng. Mọi phase đổi cách robot chuyển động, nên dataset IK hiện tại chỉ hợp lệ với WBC cũ.

1. **Phase 1 – Thăng bằng bằng policy chân RL của NVIDIA** (decoupled WBC trong repo `NVlabs/GR00T-WholeBodyControl`, dùng cho GR00T N1.5/N1.6):
   - `decoupled_wbc/control/policy/g1_gear_wbc_policy.py`: quan sát 86 chiều mỗi khung (có lịch sử):
     - `navigate_cmd` (3), `base_height_command` (1), `torso_orientation_rpy` (3);
     - gyro (3), trọng lực (3);
     - 29 góc khớp và 29 vận tốc khớp (tức policy "thấy" cả tay);
     - 15 action trước.
   - Output 15 action (chân + eo), `q = action · action_scale + default_angles`. Hai ONNX: đứng (|cmd| < 0.05) và đi.
   - `decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml`: KP chân 150/150/150/200/40/40, eo 250, tay 100/100/40/40/20/20/20;
     KD 2–5; góc chân mặc định (−0.1, 0, 0, 0.3, −0.2, 0); SIMULATE_DT 0.005. Sim hiện tại dùng **kp = 500 cho mọi khớp**, nên phải chỉnh.
   - Việc cần làm:
     - Sparse clone vào `third_party/`, tìm file ONNX, chốt scale và thứ tự khớp.
     - Viết `bridge/lower_body_policy.py`.
     - Thêm cờ `lower_body="fixed"|"gear_wbc"` cho WBC và sim (mặc định `fixed` để dataset cũ vẫn verify).
     - Đưa `base_height_command` và `navigate_command` từ action REAL_G1 vào WBC; hiện adapter bỏ qua và ghi hằng số.
     - Ghi khoá `"wbc"` vào `cfg` của `collection.jsonl`.
   - Kiểm tra: đứng yên 30 s; planner 36 seed; mục tiêu ngả dưới 2°.
2. **Phase 2 – Gắp sâu hơn**:
   - `tools/reach_map.py` (heatmap tầm với theo `table_dx`, giới hạn eo, `base_height`; thêm kiểm tra va chạm thân dưới với bàn).
   - Khoá `table_dx` trong `cfg` (mặc định 0).
   - Mở `waist_pitch_range`, rồi sinh dữ liệu với `--can-x` mới.
3. **Phase 3 – Bước đi dọc bàn**:
   - Robot xuất phát lệch, lon và khay trải rộng theo |y|.
   - Planner có đoạn "đi tới chỗ đứng" (bộ điều khiển P trên pose pelvis sinh `navigate_command`).
   - `navigate_command` khác 0 được ghi vào dataset; kiểm tra lại `stats.json`.
- Rủi ro chính:
  - Policy NVIDIA được huấn luyện trên MJCF và hệ số PD của họ; sim khác (bàn tay Dex3, `timestep` 0.002) có thể làm robot mất ổn định.
    Dự phòng: dùng MJCF G1 của họ.
  - KP tay giảm làm tay bám kém hơn khi có tải, nên phải đo lại biên dư kẹp.
  - File ONNX có thể phải tải từ LFS/HF.

---

## 7. Việc tiếp theo

- **25/09: plan sinh lại dữ liệu với balance (checkpoint C), đã duyệt; bước 0 (5.2l), 0b (5.2k), 1 (5.2m), 2 (5.2n, đạt), 3 (5.2p), 4 (5.2o), 5 (5.2q), 6 (5.2r), 7 (5.2s) xong, tiếp theo bước 8 (train C trên Colab):** [`docs/PLAN_BALANCE_DATASET.md`](docs/PLAN_BALANCE_DATASET.md).
  Gồm: eval A với cấu hình 20 s + stop; kiểm chứng số liệu nhận xét bằng `tools/demo_quality.py` (0b); `collect_demos
  --lower-body` kèm chỉ số chất lượng và lý do bỏ; chạy thử planner (điều kiện: không chạm bàn, ngả < 5°, yaw < 3°); kiểu gắp
  lon nằm nhất quán; mang lon qua **phía trên** lon kia (`CARRY_CLEARANCE` ≈ 4 cm, không tăng `OTHER_CAN_MARGIN` lên 4 cm);
  đủ dữ liệu lon đứng đổi phía; episode "đã xong" (tuỳ chọn); quota 50/nhóm/lon (đứng/nằm × cùng/đổi phía) và lọc demo chạm
  bàn; train C, so trên cùng cảnh test với A/B, tách theo nhóm.
  - Nhận xét 25/09 về dữ liệu B (chưa kiểm chứng): demo đổi phía cách lon kia trung vị 1.4 cm trước và 1.8 cm sau khi kẹp
    (`LIFT_HEIGHT` 0.10 để đáy lon đang cầm thấp hơn đỉnh lon kia); 49/400 demo hông tì bàn (đều lon nằm); yaw thân lệch tới
    12.6° ở ca đổi phía; state không có tư thế pelvis. Đưa độ ngả vào state để sau khi có kết quả C.

0. **Thu hồi HF token `hf_JbmJ...`** trên HuggingFace (mục 2, Bảo mật).
1. ~~Chạy Bước 11 pilot~~; ~~chạy closed-loop pilot~~ (mục 5.1); ~~sinh, verify, vẽ, nén dataset IK~~ (mục 4.4);
   ~~tham số hoá notebook~~; ~~công cụ đánh giá closed-loop~~ (mục 5.2).
2. **Fine-tune trên dataset IK**:
   - Tải `datasets/g1_can_sort_ik.zip` và `datasets/g1_can_sort_ik_test.zip` lên `MyDrive/gr00t/`.
   - Mở lại `colab_gr00t_finetune.ipynb` bản mới, chọn batch/steps theo mục 5, chạy tới Bước 11.
3. **Đánh giá closed-loop** checkpoint IK (mốc đối chứng cho mục 6):
   - Colab: `colab_gr00t_server.ipynb` Bước 5 (server) rồi Bước 6 (tunnel) với `MODEL_PATH = ".../checkpoints/g1_can_sort_ik/checkpoint-<N>"`,
     `EMBODIMENT_TAG = "new_embodiment"`.
   - Local: `tools/eval_closed_loop.py --host <host> --port <port> --dataset datasets/g1_can_sort_ik_test --out eval/ik_test`.
   - Có thể chạy cùng lệnh với checkpoint pilot để đối chứng (dự đoán gần 0% với lon nằm hoặc lon xa vị trí mặc định).
   - Chú ý lỗi "đã đặt xong nhưng sau đó gắp lại". Nếu hay gặp thì kéo dài đoạn `hold` cuối episode rồi sinh lại dataset.
4. ~~`launch_vla_simulation.py`: đặt cảnh theo seed~~ (24/09): `--scene-seed N` (lấy mẫu như `collect_demos`, kèm
   `--swap-prob`/`--lying-prob`) hoặc `--scene-dataset DIR --scene-episode K` (dựng lại đúng episode, kể cả nhiệm vụ).
   Phím R reset về cảnh hiện tại; phím N chuyển sang cảnh kế tiếp. Mọi chữ cái đều là phím tắt hiển thị của viewer
   MuJoCo, nên N cũng bật/tắt hiển thị constraint (R, S cũ cũng vậy). Phần đặt cảnh nằm ở `collect_demos.apply_randomization`,
   đặt khay tuyệt đối nên reset nhiều lần không bị lệch dần; đã kiểm tra `qpos`/`body_pos` giống hệt `setup_episode`. Việc
   đo vẫn dùng `eval_closed_loop`.
5. **Fine-tune trên dataset lon đổi phía** (mục 4.5; dataset đã sinh, verify và nén zip):
   - Train **từ base** `nvidia/GR00T-N1.7-3B`, không từ checkpoint cũ. Checkpoint lưu `--save-only-model` nên không train
     tiếp được. Checkpoint pilot học thuộc quỹ đạo mù. Thống kê action cũng khác: `waist_yaw` trong dataset IK luôn bằng 0
     (q01 = q99 = 0), còn trong dataset đổi phía dao động tới ±0.6 rad.
   - Để lần fine-tune `g1_can_sort_ik` (đang chạy 24/09) chạy xong; checkpoint đó là mốc đối chứng A.
   - **Sinh lại** với `hold` 2.5 s (mục 5.2d) thành `g1_can_sort_swap_hold` / `g1_can_sort_swap_hold_test`. Tải zip lên
     `MyDrive/gr00t/`, đặt `DATASET_NAME`/`TEST_DATASET_NAME` tương ứng, batch 80, `MAX_STEPS` 10 000 (checkpoint B, xem mục 5.2d-bis;
     memo cũ ghi nhầm batch 64 × 12 500).
   - So closed-loop A và B trên cả `g1_can_sort_ik_test` lẫn `g1_can_sort_swap_test`. Dự đoán A hỏng ở ca đổi phía (lối tắt
     "lệnh Coca → tay phải → lon bên phải"). B cần giữ được kết quả ở ca cùng phía.
6. Plan mở rộng vùng gắp (mục 6): Phase 1 → 2 → 3.
7. Tuỳ chọn:
   - `--tune-visual` (A100 80 GB).
   - Fine-tune dưới tag REAL_G1 với override thống kê.
   - Nhánh SONIC.

---

## 8. Tra cứu nhanh

### Lệnh local
```bash
.venv/bin/python verify_pipeline.py                                                   # 5 test của project
.venv/bin/python collect_demos.py --out datasets/<tên> --episodes-per-task 50 [--seed N] \
    [--lying-prob 0.5] [--can-x 0.26 0.37] [--can-y 0.03 0.19] [--can-gap 0.03] [--workers 12] \
    [--swap-prob 0.7] [--cross-can-y 0.03 0.15]
.venv/bin/python verify_demos.py datasets/<tên>
.venv/bin/python tools/plot_can_positions.py datasets/g1_can_sort_ik --test-dataset datasets/g1_can_sort_ik_test
.venv/bin/python tools/build_colab_notebooks.py                                       # sau khi sửa server/config/notebook
.venv/bin/python -c "import shutil; shutil.make_archive('datasets/<tên>', 'zip', 'datasets', '<tên>')"
# Đánh giá closed-loop (server Colab phải đang chạy model thật)
.venv/bin/python tools/eval_closed_loop.py --host <host> --port <port> --dataset datasets/g1_can_sort_ik_test \
    --out eval/<tên> [--save-video fail|all|none] [--seconds 10] [--workers 2] [--max-episodes N]
# Xem trực quan trong GUI (checkpoint fine-tune bắt buộc --vla-action-hz 25). Mặc định lon ở vị trí cố định trong XML;
# thêm --scene-seed N [--swap-prob 0.5] hoặc --scene-dataset datasets/g1_can_sort_swap_test --scene-episode K
.venv/bin/python launch_vla_simulation.py --policy-mode remote --server-host <host> --server-port <port> \
    --server-timeout-ms 10000 --vla-action-hz 25 --record-vla vla_log_<tên>.npz [--scene-seed 7]
# Việc chạy lâu: chặn máy tự ngủ
systemd-inhibit --what=sleep:idle --who=me --why="collect" <lệnh>
```

### Lưu ý khi chạy
- Tunnel Pinggy miễn phí hết hạn sau khoảng 60 phút.
- GUI MuJoCo: thoát bằng ESC để `--record-vla` được ghi (hoặc gửi SIGINT; nhánh `finally` vẫn gọi `save_vla_log`).
- `collect_demos.py` chỉ in tiến độ sau mỗi đợt; đợt đầu gần như gồm toàn bộ số lần thử, nên log trống lâu là bình thường.
  Số episode đã có xem trong `<out>/_tmp/*.npz`.

### File chính
| File | Vai trò |
|---|---|
| `bridge/real_g1_adapter.py` | state/action REAL_G1 ↔ actuator sim |
| `colab_gr00t_server.py` | PolicyServer ZMQ (model thật hoặc quỹ đạo mẫu) |
| `colab_gr00t_server.ipynb`, `colab_gr00t_finetune.ipynb` | notebook Colab (sinh tự động, đừng sửa tay) |
| `finetune/g1_can_sort_config.py` | cấu hình modality cho fine-tune |
| `collect_demos.py`, `verify_demos.py` | sinh và kiểm tra dataset |
| `sim/arm_ik.py` | IK một tay G1 + waist_pitch (+ waist_yaw khi gắp chéo); active set, kiểm tra va chạm (kể cả lon đang cầm), cố định khớp |
| `sim/pick_place_planner.py` | quỹ đạo gắp-đặt IK online (generator 50 Hz, lập kế hoạch theo trạng thái sim, gắp chéo người khi lon đổi phía) |
| `tools/build_colab_notebooks.py`, `tools/plot_can_positions.py` | sinh notebook (tham số `DATASET_NAME`), vẽ vị trí/hướng lon |
| `tools/eval_closed_loop.py` | đánh giá closed-loop headless theo seed, chấm như lúc thu demo |
| `tools/demo_quality.py`, `sim/episode_metrics.py` | phát lại dataset, đo khoảng hở tới lon kia, thân ngả/trượt/chạm bàn |
| `checkpoints/GEAR-SONIC/` | encoder/decoder SONIC (3 biến thể), `planner_sonic.onnx`, `README.md` |

### Dataset và checkpoint
| Tên | Vị trí | Ghi chú |
|---|---|---|
| `g1_can_sort_pilot` / `g1_can_sort_test` | `datasets/` + zip, Drive `MyDrive/gr00t/` | quỹ đạo mẫu, lon ±2 cm (mục 4.1) |
| `g1_can_sort_ik` / `g1_can_sort_ik_test` | `datasets/` + zip, Drive (đang fine-tune 24/09) | IK online, lon đứng/nằm, vùng rộng (mục 4.4) |
| `g1_can_sort_swap` / `g1_can_sort_swap_test` | `datasets/` + zip (chưa lên Drive tới 24/09) | như IK + khoảng 1/2 episode lon đổi phía (mục 4.5) |
| `g1_can_sort_swap_hold` / `..._test` | `datasets/` + zip (24/09) | như swap, hold 2.5 s, 233 khung/ep (mục 5.2d-bis); dataset train checkpoint B |
| `g1_can_sort_pilot/checkpoint-5000` | Drive `MyDrive/gr00t/checkpoints/` | closed-loop 3/10 ở vị trí mặc định (mục 5.1) |
| `vla_log_ft_2.npz` | gốc repo | log GUI closed-loop pilot (lon Coca bị quét đổ) |

### Gói đã cài thêm vào `.venv`
`pyarrow`, `av` (ghi parquet/mp4), `matplotlib` (vẽ biểu đồ).
