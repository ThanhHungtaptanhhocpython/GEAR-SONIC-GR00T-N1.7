# Nối GR00T với SONIC: code NVIDIA so với repo này, và kế hoạch sửa

Ngày: 27/09/2026.

Tham chiếu:
- `third_party/GR00T-WholeBodyControl` @ `b042411`
- `third_party/Isaac-GR00T` @ `51d4c89`
- Paper SONIC (arXiv 2511.07820v4)

Trang giải thích luồng của NVIDIA (có sơ đồ): https://claude.ai/artifact/2BAwB2x32bNVfKPH3ZERii

---

## 0. Tóm tắt

- NVIDIA nối GR00T với SONIC bằng **universal token**:
  - GR00T dự đoán chunk 40 bước, mỗi bước là `motion_token` (64) + `left_hand_joints` (7) + `right_hand_joints` (7).
  - Client gửi một bước mỗi tick 50 Hz qua ZMQ.
  - C++ chép token vào chỗ của encoder và chạy decoder SONIC để ra 29 góc khớp.
  - Tay Dex3 đi đường riêng, không qua SONIC.
- Nhãn `motion_token` để fine-tune chính là **output của encoder SONIC**, ghi lại trong lúc teleop.
- Repo này hiện **không chạy SONIC thật**:
  - VLA dùng action góc khớp theo layout REAL_G1 (`new_embodiment`), đi qua `track_joint_targets`.
  - Chân khoá cứng hoặc do policy thăng bằng `decoupled_wbc` điều khiển.
  - Nhánh token (`SonicWBCController.decode`) chỉ được gọi với quỹ đạo mẫu (`scripted`) và trong `verify_pipeline.py`.
- Chế độ `onnx` của `decode` có gọi decoder thật, nhưng dựng input và đọc output **khác NVIDIA ở 7 điểm**. Ba điểm trong đó đủ làm decoder nhận quan sát sai hoàn toàn: thứ tự khớp, tư thế gốc, action scale.
- Chế độ `kinematic` (mặc định) coi từng chiều token như delta góc khớp. Cách này không có ý nghĩa với token FSQ thật.
- Chưa có encoder, nên chưa tự sinh được nhãn token.

---

## 1. Luồng của NVIDIA

| Tầng | File | Việc làm |
|---|---|---|
| Server | `Isaac-GR00T/gr00t/eval/run_gr00t_server.py`, `gr00t/policy/gr00t_policy.py:380` | Chạy model và giải chuẩn hoá **min-max** theo stats của dataset. Cả 3 nhóm action là `ABSOLUTE`. |
| Cấu hình | `Isaac-GR00T/gr00t/configs/data/embodiment_configs.py:67` (`unitree_g1_sonic`) | video `ego_view` 1 khung. State gồm 7 nhóm khớp + `projected_gravity`. Action 40 bước. |
| Client | `gear_sonic/scripts/run_vla_inference.py` | Worker gọi server 2.5 Hz. Bỏ chunk nếu `max\|token\| > 1.25`. Chỉ số bắt đầu = `round(độ trễ × 50)`. Mỗi tick gửi 1 bước, dừng ở bước 39. Chunk mới thay thẳng chunk cũ, **không trộn, không lọc**. |
| Message | `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py:161` | `"pose"` + header JSON 1024 byte + `token_state` f32[1,64], `frame_index` i64[1], hai tay f32[1,7]. Protocol v4. |
| Nhận token | `gear_sonic_deploy/.../zmq_endpoint_interface.hpp:395` | v4 chỉ được chứa token. Token lưu vào `external_token_state_`. |
| Vòng điều khiển | `gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp:3033` | Kiểm tra kích thước 64, chép nguyên, tắt encoder. 200 ms không có token mới thì cảnh báo và giữ token cuối. |
| Quan sát decoder | `g1_deploy_onnx_ref.cpp:2862`, `:1454–1630`, `checkpoints/GEAR-SONIC/observation_config.yaml` | 994 = token 64 + 10 khung (cũ → mới) của: gyro IMU pelvis (3), `q − default_angles` (29), `dq` (29), action thô trước đó (29), trọng lực `R^T·[0,0,−1]` (3). **Thứ tự IsaacLab.** |
| Ra motor | `g1_deploy_onnx_ref.cpp:3155`, `policy_parameters.hpp` | `q[i] = default_angles[i] + a[isaaclab_to_mujoco[i]] · g1_action_scale[i]`. PD với `kps`/`kds` riêng từng khớp. |
| Tay | `g1_deploy_onnx_ref.cpp:3991` | `dex3_hands_.setAllJointsCommand(...)` trực tiếp. |
| Nhãn | `gear_sonic/scripts/run_data_exporter.py:662` | `action.motion_token = token_state` do encoder sinh lúc teleop. |

---

## 2. Khác biệt chi tiết

Mức độ:
- **Cao**: decoder nhận input sai, robot không thể đứng đúng.
- **Vừa**: hành vi khác NVIDIA nhưng vẫn chạy.
- **Thấp**: an toàn hoặc tiện dụng.

### 2.1. Decoder (`bridge/sonic_wbc_controller.py`, chế độ `onnx`)

| # | Điểm | NVIDIA | Repo này | Mức |
|---|---|---|---|---|
| D1 | Thứ tự 29 khớp trong quan sát | IsaacLab: `body_q[i] = q_hw[mujoco_to_isaaclab[i]]`. Trái/phải/eo xen kẽ. | Ghép theo thứ tự MuJoCo `left_leg, right_leg, waist, left_arm, right_arm`, không đổi thứ tự (`:189-195`). Ví dụ vị trí 1 nhận `L.hipR` thay vì `R.hipP`. | Cao |
| D2 | Thứ tự 29 action đầu ra | Đổi IsaacLab → MuJoCo bằng `isaaclab_to_mujoco`. | Đọc thẳng `action_29[0:12]` là chân, `[12:15]` là eo… (`:233-238`). | Cao |
| D3 | Tư thế gốc (`default_angles`) | Chân gập gối: hip pitch −0.312, gối 0.669, cổ chân −0.363. Vai 0.2/±0.2, khuỷu 0.6. | Lấy từ keyframe sim: chân **thẳng** (toàn 0), eo pitch 0.12, tay ở tư thế ôm lon (`:109`). Quan sát `q − default` lệch khoảng 0.7 rad ở gối ngay khi đứng yên. | Cao |
| D4 | Action scale | Riêng từng khớp, `0.25 · effort / stiffness`: hông/gối 7520-22 là 0.351, hip yaw/eo yaw 7520-14 là 0.548, loại 5020 là 0.439, cổ tay 4010 là 0.075. | `0.25` cho mọi khớp, thêm `clip(±1.5)` (`:231`). | Cao |
| D5 | Chân | Decoder điều khiển đầy đủ. | Ép `default + 0.05 · clip(±0.2)` (`:234`), gần như khoá chân. | Cao |
| D6 | Bộ lọc thông thấp | Không có. | `_smooth` với α = 0.35 áp lên cả 43 khớp (`:302`), làm lệch động học so với lúc train. | Vừa |
| D7 | PD trong sim | `kps` từ armature ở 10 Hz: 99.1 (7520-22), 40.2 (7520-14), 14.3 (5020, cổ chân và eo roll/pitch ×2), 16.8 (4010). `kds` tương ứng. | `kp = 500`, `dampratio = 1` cho mọi actuator (`sim/unitree_g1/g1_with_hands.xml:10`). Policy RL rất nhạy với PD. | Cao |
| D8 | Lịch sử lúc bắt đầu | Lấy từ state logger (state thật của các tick trước). | Khởi tạo toàn 0; trọng lực `[0,0,−1]` (`:141-148`). Mười tick đầu decoder thấy lịch sử giả. | Vừa |
| D9 | Checkpoint | Encoder và decoder cùng checkpoint, cùng checkpoint dùng để sinh nhãn. | Tự chọn `checkpoints/GEAR-SONIC/model_decoder.onnx` (bản `sonic_release`); `low_latency` và `sonic_v1_1` có md5 khác. Chưa có ràng buộc với nhãn. | Vừa |

Hai điểm đã **khớp**: gyro lấy từ IMU pelvis (`sim/g1_scene.py:186`), trọng lực tính từ ma trận quay pelvis (`sim/g1_scene.py:172`).

`last_onnx_action` lưu output thô trước khi nhân scale, đúng như NVIDIA, nhưng vẫn theo thứ tự sai (D2).

### 2.2. Chế độ `kinematic`

| # | Điểm | Mức |
|---|---|---|
| K1 | `motion_token[0:12]` coi là chân, `[12:15]` là eo, `[15:29]` là tay (`:245-270`). Token FSQ không có ngữ nghĩa theo từng chiều như vậy. Chế độ này chỉ đúng với token do chính `scripted_sorting_chunk` tạo ra. | Cao (nếu dùng với model thật) |
| K2 | Không có góc tay thì lấy `token[29:43]` làm độ khép tay (`:276-281`). Với token thật, kết quả là rác. | Vừa |

### 2.3. Thực thi chunk (`bridge/closed_loop_pipeline.py`)

| # | Điểm | NVIDIA | Repo này | Mức |
|---|---|---|---|---|
| C1 | Nối chunk | Chunk mới thay thẳng, bắt đầu từ `round(độ trễ × 50)`. | Trộn 4 bước kiểu RTC (`:237-249`), bắt đầu từ bước 0. | Thấp (sim dừng chờ server nên độ trễ tính theo thời gian sim là 0) |
| C2 | Token vượt biên | Bỏ cả chunk nếu `>1.25`. | Không kiểm tra. | Vừa |
| C3 | Tay | Gửi thẳng tới Dex3. | Clip theo giới hạn Dex3 rồi qua bộ lọc (D6). | Thấp |

### 2.4. Dữ liệu và fine-tune

| # | Điểm | NVIDIA | Repo này | Mức |
|---|---|---|---|---|
| T1 | Action space | `unitree_g1_sonic`: token 64 + tay 7+7, tuyệt đối. | Layout REAL_G1 (`finetune/g1_can_sort_config.py`): EEF/tay relative + góc khớp, tag `new_embodiment`. | Quyết định thiết kế |
| T2 | Nguồn nhãn token | Encoder SONIC lúc teleop. | Không có encoder. Input 1762 chiều chưa được dựng. | Cao (chặn hướng SONIC) |
| T3 | State | 7 nhóm khớp + `projected_gravity`. | 7 nhóm khớp + EEF 9D; không có trọng lực. | Theo T1 |
| T4 | Thứ tự 7 ngón tay | Theo robot model của `gear_sonic` (cần xác minh). | Thứ tự alphabet suy từ `statistics.json` của REAL_G1 (memo §2). | Cần xác minh |

---

## 3. Kế hoạch sửa

Nguyên tắc:
- Đường REAL_G1 hiện tại giữ **mặc định** cho tới khi đường SONIC được chứng minh tốt hơn trên cùng bộ seed.
- Mỗi phase có cờ bật/tắt và tiêu chí nghiệm thu riêng.
- Phase 1 và 2 có giá trị ngay cả khi không chuyển action space: SONIC có thể thay policy thăng bằng `decoupled_wbc` để giữ chân.
- Phase 3 đến 5 chỉ làm nếu quyết định chuyển sang `UNITREE_G1_SONIC`.

### Phase 0: chuẩn bị (nhỏ)

1. Chốt checkpoint `sonic_release` (ONNX ở gốc `checkpoints/GEAR-SONIC/`). Ghi md5 của `model_encoder.onnx` và `model_decoder.onnx` vào memo.
2. Tuỳ chọn, nên làm: dựng sim2sim chính thức của NVIDIA để làm **chuẩn đối chiếu**:
   - `bash install_scripts/install_mujoco_sim.sh`
   - `python gear_sonic/scripts/run_sim_loop.py` + `bash deploy.sh sim`

   Bước này cần build `gear_sonic_deploy`. Có được nó thì ghi lại `obs_buffer_` 994 chiều và 29 action của C++ để so từng số với bản Python (test vàng ở Phase 1).
3. Xác minh T4: đọc `gear_sonic/data/robot_model/instantiation/g1.py`, ghi thứ tự 7 khớp tay của NVIDIA.

### Phase 1: viết lại decoder cho đúng chuẩn (vừa)

File mới `bridge/sonic_decoder.py`. `SonicWBCController.decode` gọi sang file này khi `mode="onnx"`.

1. **Hằng số.** Chép từ `policy_parameters.hpp`, ghi rõ commit:
   - `isaaclab_to_mujoco`, `mujoco_to_isaaclab`
   - `default_angles`, `g1_action_scale`, `kps`, `kds`

   Thêm bảng nối "29 khớp MuJoCo" ↔ "43 actuator sim" bằng `BODY_INDICES`.
2. **Quan sát** (sửa D1, D3, D8):
   - `body_q[i] = q_mj[m2i[i]] − default_angles[m2i[i]]`; `dq` dùng cùng thứ tự.
   - `last_action` là output thô theo thứ tự IsaacLab.
   - Lịch sử 10 khung xếp cũ → mới.
   - Lúc `reset`, lấp lịch sử bằng state thật hiện tại, không lấp bằng 0.
3. **Đầu ra** (sửa D2, D4, D5, D6):
   - `q_mj[i] = default_angles[i] + a[i2m[i]] · g1_action_scale[i]`.
   - Bỏ kẹp chân, bỏ `clip(±1.5)`, bỏ `_smooth` cho 29 khớp thân.
   - Chỉ giữ clip theo giới hạn khớp của MJCF để an toàn.
4. **PD** (sửa D7): đặt `actuator_gainprm`/`biasprm` cho 29 actuator thân theo `kps`/`kds`. Làm giống cách `bridge/lower_body_policy.py:76-81` đang làm, và chỉ bật khi dùng decoder SONIC.
5. **Kiểm tra token** (sửa C2): từ chối token có `max|token| > 1.25`.
6. **Test:**
   - `i2m[m2i[i]] == i`.
   - Tên và shape input/output của ONNX (`obs_dict [1,994]` → `action [1,29]`).
   - Test vàng so với C++ nếu Phase 0.2 đã xong.
7. **Nghiệm thu trong MuJoCo:**
   - Cấp liên tục `LATENT_INITIAL_MOTION_TOKEN` (`gear_sonic/utils/inference/initial_poses.py`) trong 30 s: robot đứng vững, thân ngả dưới 2°.
   - Đẩy nhẹ pelvis: robot không ngã.

**Rủi ro.** SONIC được train trên G1 29 khớp. MJCF của repo có thêm tay Dex3 và `timestep 0.002`, khối lượng tay có thể làm lệch động học. Dự phòng: dùng MJCF G1 của NVIDIA (`gear_sonic/data/robots/g1/`) rồi gắn lại tay.

### Phase 2: encoder và kiểm tra khứ hồi (vừa)

File mới `bridge/sonic_encoder.py`.

1. Dựng input 1762 chiều đúng thứ tự trong `observation_config.yaml`. Tổng các khối đã khớp 1762:

   | Khối | Chiều |
   |---|---|
   | `encoder_mode_4` | 4 |
   | `motion_joint_positions_10frame_step5` | 290 |
   | `motion_joint_velocities_10frame_step5` | 290 |
   | `motion_root_z_position_10frame_step5` | 10 |
   | `motion_root_z_position` | 1 |
   | `motion_anchor_orientation` | 6 |
   | `motion_anchor_orientation_10frame_step5` | 60 |
   | `…lowerbody` (vị trí + vận tốc) | 120 + 120 |
   | `vr_3point_local_target` | 9 |
   | `vr_3point_local_orn_target` | 12 |
   | `smpl_joints_10frame_step1` | 720 |
   | `smpl_anchor_orientation_10frame_step1` | 60 |
   | `motion_joint_positions_wrists_10frame_step1` | 60 |

   - Chế độ `g1` (`mode_id 0`) chỉ cần 4 khối: mode, vị trí khớp và vận tốc khớp 10 khung bước 5, hướng anchor 10 khung bước 5.
   - Cần đọc các hàm `Gather…` trong `g1_deploy_onnx_ref.cpp:560-720` để chốt ba điều: các khối không dùng có lấp 0 không, quy ước quaternion, và khung toạ độ của anchor.
2. **Nhìn trước quỹ đạo.** Chế độ `g1` cần chuyển động tham chiếu của khoảng 1 s tới (10 khung × bước 5 ở 50 Hz). Planner IK (`sim/pick_place_planner.py`) đang sinh online, nên phải chạy trước một đệm 50 khung.
3. **Kiểm tra khứ hồi.** Quỹ đạo planner → encoder → token → decoder (Phase 1) → sim. Đo:
   - MPJPE so với tham chiếu (paper khoảng 25 mm);
   - tỉ lệ không ngã;
   - khoảng hở tới lon (`sim/episode_metrics.py`).

   Mốc đạt: bám được toàn bộ episode của dataset `g1_can_sort_swap_hold`.

Kết quả của Phase 2 là **SONIC làm WBC cho planner hiện tại**. Tới đây đã thay được `decoupled_wbc` mà chưa cần đổi VLA.

### Phase 3: thu dataset `unitree_g1_sonic` (vừa)

1. `collect_demos.py --wbc sonic`: planner → encoder → decoder. Mỗi khung ghi:
   - `ego_view`;
   - state 7 nhóm khớp + `projected_gravity`;
   - `action.motion_token` = token **đúng như đã đưa vào decoder**;
   - `left/right_hand_joints` theo thứ tự đã xác minh ở Phase 0.3.
2. Xuất LeRobot v2, `meta/modality.json` khớp `unitree_g1_sonic`. Ghi `"wbc": "sonic_release"` và md5 checkpoint vào `collection.jsonl`.
3. Ghi dữ liệu ở 50 Hz, để chunk 40 bước tương ứng 0.8 s, giống NVIDIA. Cần xác nhận fps của data exporter NVIDIA.
4. `verify_demos.py`: phát lại token qua decoder và phải tái tạo được episode.

### Phase 4: fine-tune với `UNITREE_G1_SONIC` (vừa)

1. `tools/build_colab_notebooks.py`: thêm tham số `EMBODIMENT_TAG`. Với SONIC thì dùng cấu hình đăng ký sẵn, không cần `g1_can_sort_config.py`.
2. Kiểm tra stats. Checkpoint base không có stats cho tag này, nên stats lấy từ dataset (không lặp lại sự cố `override=False` của REAL_G1).
3. Đánh giá open-loop:
   - MSE token so với dataset;
   - tỉ lệ token vượt 1.25;
   - giải mã token dự đoán trong sim.

### Phase 5: client inference theo chuẩn NVIDIA (nhỏ)

1. `closed_loop_pipeline.py` thêm `action_space == "sonic_token"`:
   - bỏ chunk vượt biên;
   - chỉ số bắt đầu `round(độ trễ × 50)`, có cờ chọn độ trễ theo đồng hồ thật hoặc theo sim;
   - không trộn chunk;
   - lặp bước cuối;
   - tay đi thẳng.
2. Đổi tên chế độ `kinematic` thành `heuristic`, chỉ dùng với `--policy scripted`. In cảnh báo nếu dùng với model thật (sửa K1, K2).

### Phase 6: so sánh

Chạy `tools/eval_closed_loop.py` trên cùng bộ seed với checkpoint B (REAL_G1). So sánh:
- tỉ lệ thành công;
- độ ngả thân;
- va chạm lon;
- độ mượt (jerk).

Chỉ đổi mặc định sang SONIC khi thắng rõ ràng.

---

## 4. Thứ tự và phụ thuộc

```
Phase 0 ─► Phase 1 ─► Phase 2 ─┬─► (dừng ở đây: SONIC làm WBC cho planner)
                               └─► Phase 3 ─► Phase 4 ─► Phase 5 ─► Phase 6
```

Nên làm ngay: **Phase 1**. Phase này nhỏ, sửa đúng những điểm mức Cao, và có tiêu chí nghiệm thu rõ (đứng vững 30 s với token tư thế ban đầu).

## 5. Câu hỏi còn mở

1. Thứ tự 7 khớp tay trong `gear_sonic` (T4).
2. Encoder chế độ `g1`: các khối không dùng có lấp 0 không? Quaternion anchor theo quy ước nào (`wxyz`, khung heading)?
3. Có build được `gear_sonic_deploy` trên máy local (không GPU) để làm chuẩn đối chiếu không? Nếu không, test vàng phải dựa vào số liệu tay.
4. MJCF có tay Dex3 có làm SONIC mất ổn định không (rủi ro của Phase 1).
