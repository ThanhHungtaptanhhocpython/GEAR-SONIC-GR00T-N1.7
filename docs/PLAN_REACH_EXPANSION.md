<!-- Plan đã duyệt ngày 24/09/2026. Trạng thái: chưa bắt đầu (làm sau lần fine-tune trên dataset IK). Tóm tắt và tiến độ: MEMO_VLA_FINETUNE.md mục 6–7. -->

# Plan: Mở rộng vùng gắp của G1 (thăng bằng → đứng sát bàn → bước đi)

## Context

Hiện robot chỉ gắp được lon có tâm ở x 0.26–0.37 m trước pelvis. Nguyên nhân:
- Cánh tay tới điểm gắp dài 0.49 m, vai cao hơn lon khoảng 0.35 m, nên phần vươn ngang tối đa chỉ khoảng 0.35 m tính từ vai.
- `SonicWBCController.track_joint_targets` khoá cứng 12 khớp chân (`target_ctrl[0:12] = default_ctrl[0:12]`). Robot không có thăng bằng, ngả 3–9° theo tải ở tay, hông chạm mép bàn, bàn chân trượt khi mang lon nằm.
- Action REAL_G1 có `base_height_command` và `navigate_command`, nhưng adapter đang bỏ qua (ghi hằng số, không đưa vào WBC).

Người dùng muốn gắp được lon ở **sâu hơn trên bàn** và **rộng hơn dọc theo bàn**. Các quyết định đã chốt:
- **Fine-tune trên dataset IK hiện có trước** (để có mốc đối chứng), rồi mới làm plan này.
- Dùng **policy RL chân của NVIDIA** (decoupled WBC trong repo GR00T-WholeBodyControl, dùng cho GR00T N1.5/N1.6).

Đã khảo sát repo NVIDIA:
- `decoupled_wbc/control/policy/g1_gear_wbc_policy.py`: quan sát 86 chiều mỗi khung, xếp chồng theo lịch sử.
  - [0:3] `navigate_cmd` (vx, vy, yaw rate), [3] `base_height_command`, [4:7] `torso_orientation_rpy`.
  - [7:10] gyro, [10:13] hướng trọng lực, [13:42] 29 góc khớp, [42:71] 29 vận tốc khớp, [71:86] action trước.
  - Output 15 action (chân và eo): `q = action * action_scale + default_angles`.
  - Hai ONNX: đứng thăng bằng khi |cmd| < 0.05, đi khi lớn hơn.
- `decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml`: có `MOTOR_KP` và `MOTOR_KD` cho từng khớp (chân 150/200/40, eo 250, tay 100/40/20), góc mặc định chân (−0.1, 0, 0, 0.3, −0.2, 0), SIMULATE_DT 0.005. Sim hiện tại dùng kp = 500 cho mọi khớp, nên phải chỉnh cho khớp.

Đính chính so với đề xuất ban đầu: đứng sát bàn và bước tới **không tăng tầm với của tay**. Hông cao 0.79 m, mặt bàn cao 0.68 m, nên chỉ tiến thêm được khoảng 8 cm trước khi hông chạm mép bàn. Tầm với theo chiều sâu chỉ tăng nhờ cúi người có giữ thăng bằng. Bước đi chủ yếu giúp mở rộng vùng gắp **dọc theo bàn**.

## Bước 0: làm trước plan này (ngoài phạm vi, đã thống nhất)
Fine-tune trên `g1_can_sort_ik`, sau đó chạy `tools/eval_closed_loop.py --dataset datasets/g1_can_sort_ik_test` để có mốc đối chứng. Toàn bộ plan dưới đây sẽ đổi cách robot chuyển động, nên dataset hiện tại chỉ hợp lệ với WBC cũ.

## Phase 1: Tích hợp policy chân của NVIDIA (thăng bằng khi đứng)

1. **Tải và đọc chi tiết** (sparse clone GR00T-WholeBodyControl vào `third_party/`, ghi commit ghim vào memo):
   - Tìm file ONNX của hai policy đứng và đi trong `decoupled_wbc/.../resources/robots/g1/` (có thể nằm trong LFS hoặc trên HF).
   - Chốt các thông số chính xác: `obs_history_len`, `action_scale`, `cmd_scale`, `ang_vel_scale`, `dof_pos_scale`, `dof_vel_scale`, thứ tự 29 khớp, 15 action gồm những khớp nào, tần số chạy policy.
   - Đọc `g1_decoupled_whole_body_policy.py` để biết waist của phần thân trên được giao cho policy chân như thế nào (qua `torso_orientation_rpy` hay góc khớp trực tiếp).
2. **Module mới `bridge/lower_body_policy.py`** (`GearWbcLowerBody`):
   - Dựng quan sát từ sim: gyro `imu-pelvis-angular-velocity` (đã có `pelvis_gyro_sid`), `get_projected_gravity()`, góc và vận tốc 29 khớp theo đúng thứ tự policy. Ánh xạ từ thứ tự actuator qua `act_qposadr` và `act_dofadr` của `G1CanSortingSim`.
   - Giữ lịch sử quan sát, chọn policy đứng hay đi theo |nav|, trả về mục tiêu cho chân (và eo).
3. **Chế độ WBC mới, giữ nguyên chế độ cũ**:
   - Thêm cờ `lower_body="fixed" | "gear_wbc"` cho `SonicWBCController` và `G1CanSortingSim`. Chế độ `gear_wbc` áp hệ số KP/KD của NVIDIA cho 29 khớp thân bằng cách sửa `actuator_gainprm`/`biasprm` lúc nạp model; bàn tay giữ nguyên.
   - `track_joint_targets(target, nav_cmd, height_cmd)`: tay và bàn tay lấy từ target; chân (và eo) lấy từ policy.
   - Mặc định vẫn là `fixed`, để các dataset cũ vẫn verify và phát lại được.
4. **Nối action REAL_G1 vào WBC**:
   - `RealG1Adapter.action_to_ctrl` trả thêm `base_height_command` và `navigate_command` theo từng bước.
   - `ClosedLoopVLAPipeline.step_50hz` chuyển hai lệnh này vào WBC.
   - `action_from_ctrl` ghi giá trị lệnh thực tế thay cho hằng số `BASE_HEIGHT_COMMAND` và `NAVIGATE_COMMAND`.
5. **Ghi chế độ vào dataset**: đưa `"wbc": "gear_wbc"` vào `cfg` trong `meta/collection.jsonl`. Hàm `collect_demos.setup_episode` (đã có `pipeline_kwargs`), `replay_actions`, `verify_demos.py` và `tools/eval_closed_loop.py` đọc khoá này để dựng đúng WBC (thiếu khoá thì dùng `fixed`).
6. **Kiểm tra**:
   - Đứng yên 30 s: pelvis nghiêng dưới 1°, trôi dưới 1 cm.
   - Chạy planner IK với vùng lon hiện tại, 36 seed mỗi loại lon. So với bảng trong memo (lon đứng 100%, lon nằm 69–78%, ngả 3–9°). Mục tiêu: ngả dưới 2°, lon nằm không còn lỗi "trượt chân / hông chạm bàn".
   - Nếu policy mất ổn định trong sim này (do khác khối lượng, quán tính, bàn tay Dex3): thử dùng MJCF G1 của NVIDIA (`g1_29dof_with_hand.xml`) cho phần robot.

## Phase 2: Gắp sâu hơn vào bàn

1. **Bản đồ tầm với `tools/reach_map.py`**:
   - Dùng lại `ArmIK` và `ArmIK.collides` (`sim/arm_ik.py`), `plan_grasp` (`sim/pick_place_planner.py`) trên một lưới vị trí lon (đứng, và nằm theo nhiều hướng).
   - Thử các biến thể: khoảng cách bàn (dx), eo cúi tối đa 0.45 → 0.52, hạ `base_height`.
   - Thêm kiểm tra va chạm **thân dưới với bàn** (pelvis, hông, đùi); hiện chỉ kiểm tra tay.
   - Kết quả: ảnh heatmap. Dùng để chọn `table_dx` và dải `--can-x` mới.
2. **Khoảng cách bàn là tham số của episode**:
   - Khoá `table_dx` trong `cfg`, mặc định 0 nên dataset cũ vẫn tái tạo đúng.
   - `setup_episode` dời body `work_table`, hai khay và vùng lấy mẫu lon.
   - `scene_layout()`, `tools/plot_can_positions.py`, `tools/eval_closed_loop.py` và `launch_vla_simulation.py` dùng cùng giá trị này. Launcher được thêm tuỳ chọn đặt cảnh theo seed, như memo mục 6.3.
3. **Planner**:
   - Mở giới hạn `waist_pitch_range` trong `ArmIK` theo kết quả bản đồ.
   - Đưa va chạm thân dưới với bàn vào `_check`.
   - Nếu có lợi, lập `base_height_command` theo đoạn (hạ người khi vươn xa).
   - `_servo` đã giải lại IK theo pose pelvis đo được, nên vẫn đúng khi chân do policy điều khiển.
4. Thu thử 20 episode rồi verify, sau đó sinh dataset đầy đủ với `--can-x` mới.

## Phase 3: Bước đi dọc theo bàn

1. **Thử đi**: gửi `navigate_command` (vx, vy, yaw) trong sim. Đo độ chính xác khi dừng, độ ổn định khi đứng lại, và va chạm chân với chân bàn (các chân bàn ở góc: x 0.23/0.73, y ±0.40).
2. **Ngẫu nhiên hoá mới**:
   - Robot xuất phát lệch theo y (và theo yaw).
   - Lon và khay trải rộng hơn theo |y|, sao cho một phần episode phải bước sang mới gắp hoặc đặt được.
   - Dải cụ thể chọn từ bản đồ tầm với ở Phase 2.
3. **Planner, đoạn mới "đi tới chỗ đứng"**:
   - Tính tư thế đứng sao cho lon nằm trong vùng với của Phase 2.
   - Bộ điều khiển P trên pose pelvis sinh `navigate_command`, có giới hạn vận tốc và gia tốc. Dừng khi sai số dưới ngưỡng, rồi chạy các đoạn gắp cũ.
   - Tương tự khi khay nằm ngoài tầm với. Trong lúc đi, hai tay giữ tư thế ready.
4. **Dataset và fine-tune**:
   - `navigate_command` giờ khác 0 và được ghi vào action. Kiểm tra `stats.json` (chuẩn hoá q01/q99), vì trước đây giá trị này là hằng số.
   - Episode dài hơn: cập nhật `STEPS` và thời lượng mặc định của `eval_closed_loop`.

## File chính cần sửa hoặc tạo
- Mới: `bridge/lower_body_policy.py`, `tools/reach_map.py`, `third_party/GR00T-WholeBodyControl` (sparse, ghim commit).
- Sửa:
  - `bridge/sonic_wbc_controller.py` (chế độ `gear_wbc`)
  - `sim/g1_scene.py` (hệ số KP/KD theo chế độ)
  - `bridge/real_g1_adapter.py` (lệnh height và nav)
  - `bridge/closed_loop_pipeline.py`
  - `collect_demos.py` (khoá `wbc`, `table_dx`, xuất phát lệch)
  - `sim/pick_place_planner.py`, `sim/arm_ik.py` (va chạm thân dưới, đoạn đi)
  - `verify_demos.py`, `tools/eval_closed_loop.py`, `tools/plot_can_positions.py`, `launch_vla_simulation.py`
  - `verify_pipeline.py` (thêm test)
  - `MEMO_VLA_FINETUNE.md`

## Kiểm chứng (mỗi phase)
- `verify_pipeline.py`: toàn bộ test cũ vẫn đạt, cộng test mới cho chế độ `gear_wbc` (đứng yên, và bám được target tay).
- Dataset cũ (`g1_can_sort_ik`, pilot, test): `verify_demos.py` vẫn phát lại đạt, vì mặc định là `fixed`.
- Phase 1: đứng yên 30 s; tỉ lệ thành công của planner trên 36 seed × 2 lon × đứng/nằm; độ ngả tối đa.
- Phase 2: heatmap tầm với trước và sau; tỉ lệ thành công của planner ở dải `--can-x` mới; `verify_demos.py` trên 20 episode thử.
- Phase 3: sai số dừng khi đi; tỉ lệ thành công cho episode cần bước đi; video của `eval_closed_loop --save-video all` cho vài seed.
- Sau mỗi dataset mới: fine-tune, rồi `tools/eval_closed_loop.py` trên tập test tương ứng, so với mốc đối chứng ở Bước 0.

## Rủi ro chính
- Policy của NVIDIA được huấn luyện trên MJCF và hệ số PD của họ. Nếu sim hiện tại lệch nhiều (bàn tay Dex3, khối lượng, `timestep` 0.002 so với 0.005), robot có thể mất ổn định. Cách xử lý: dùng phần robot trong MJCF của họ.
- Hệ số KP của tay giảm (500 → 100/40/20) làm tay bám kém hơn khi có tải. Planner cần servo nhiều hơn, và phải đo lại biên dư gắp (cùng cách đo đã dùng: khoảng cách lúc tay khép).
- File ONNX có thể phải tải riêng (LFS/HF). Cần có mạng lúc làm Phase 1.1.
