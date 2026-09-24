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

### 5.0c. Checkpoint IK `g1_can_sort_ik/checkpoint-10000` (`MAX_STEPS=10000`, batch 32, 24/09)
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
  - Bước 7 in trạng thái (port, số request, suy luận TB, số Traceback, cuối log). Bước 8 dừng server (`pkill -f`).
- Tốc độ chép Drive → local trên Colab chưa đo; xem số MB/s in ra ở Bước 5 lần tới.

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
   - Tải `g1_can_sort_swap.zip` và `g1_can_sort_swap_test.zip` lên `MyDrive/gr00t/`, đặt `DATASET_NAME = "g1_can_sort_swap"`,
     `TEST_DATASET_NAME = "g1_can_sort_swap_test"`, cùng `MAX_STEPS`/batch với lần IK (checkpoint B).
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
| `checkpoints/GEAR-SONIC/` | encoder/decoder SONIC (3 biến thể), `planner_sonic.onnx`, `README.md` |

### Dataset và checkpoint
| Tên | Vị trí | Ghi chú |
|---|---|---|
| `g1_can_sort_pilot` / `g1_can_sort_test` | `datasets/` + zip, Drive `MyDrive/gr00t/` | quỹ đạo mẫu, lon ±2 cm (mục 4.1) |
| `g1_can_sort_ik` / `g1_can_sort_ik_test` | `datasets/` + zip, Drive (đang fine-tune 24/09) | IK online, lon đứng/nằm, vùng rộng (mục 4.4) |
| `g1_can_sort_swap` / `g1_can_sort_swap_test` | `datasets/` + zip (chưa lên Drive tới 24/09) | như IK + khoảng 1/2 episode lon đổi phía (mục 4.5) |
| `g1_can_sort_pilot/checkpoint-5000` | Drive `MyDrive/gr00t/checkpoints/` | closed-loop 3/10 ở vị trí mặc định (mục 5.1) |
| `vla_log_ft_2.npz` | gốc repo | log GUI closed-loop pilot (lon Coca bị quét đổ) |

### Gói đã cài thêm vào `.venv`
`pyarrow`, `av` (ghi parquet/mp4), `matplotlib` (vẽ biểu đồ).
