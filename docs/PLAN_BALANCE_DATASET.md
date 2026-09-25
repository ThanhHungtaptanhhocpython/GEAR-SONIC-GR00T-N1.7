<!-- Plan soạn ngày 25/09/2026, sửa cùng ngày theo hai lượt nhận xét về dữ liệu train B. Trạng thái: đã duyệt; bước 0–7 xong (memo mục 5.2k–5.2s), tiếp theo bước 8.
Số liệu gốc: MEMO_VLA_FINETUNE.md mục 5.0e, 5.2g–5.2i. -->

# Plan: Sinh lại dữ liệu với chân tự cân bằng (checkpoint C)

## Context

Checkpoint A (`g1_can_sort_ik`) và B (`g1_can_sort_swap_hold`) đều được train trên demo thu với **chân khoá cứng**
(`--lower-body fixed`). Chế độ này không thực tế: chân khoá nhưng thân vẫn ngả tới 14° và pelvis trôi tới 13 cm khi vươn tay.
Planner IK online đã bù chuyển động đó, nên góc khớp trong demo gắn chặt với cách thân ngả của chế độ fixed.

Khi bật policy thăng bằng của NVIDIA (`--lower-body balance`, `bridge/lower_body_policy.py`), thân gần như đứng yên. Cùng một
bộ góc khớp lại đưa tay tới chỗ lệch vài cm:

| Phép thử (tập `g1_can_sort_swap_hold_test`) | fixed | balance |
|---|---|---|
| Phát lại nguyên action của 40 demo, không dùng model | **40/40** | **17/40** (cùng phía 8/14, đổi phía 9/26) |
| B-9000 closed-loop trên `ik_test`, 20 s + `--stop-on-success` | 31/40 | 18/40 |
| B-9000 closed-loop trên tập đổi phía, 15 episode chung | 9/15 | 5/15 (đổi phía 0/9) |
| Khoảng cách tay–lon lúc khép TB | 4.3 cm | 11.7 cm |

Kết luận: model train trên dữ liệu fixed không chạy được với balance. Muốn robot đứng bằng bộ cân bằng thì phải **thu lại
demo với balance và train lại**. Đây cũng là Phase 1 của [`PLAN_REACH_EXPANSION.md`](PLAN_REACH_EXPANSION.md).

Lần sinh lại này cũng là dịp sửa các lỗi đã tìm ra khi đánh giá B, để chỉ phải train một lần:
- **Lon nằm có hai kiểu gắp** (mục 5.0e, 5.2g): planner chọn chiều trục lon `s = ±1` theo chi phí nhỏ nhất. Quanh ranh giới
  (Coca nằm cùng phía, yaw gập khoảng 90–120°), hai cảnh gần giống nhau có wrist pitch lệch khoảng 2 rad. Closed-loop cho thấy
  tay đổi kiểu gắp giữa chừng, đụng lon làm lon rơi.
- **Thiếu dữ liệu cùng phía:** `--swap-prob 0.7` làm B chỉ có 39 episode Coca nằm cùng phía (A có 92). Khoảng yaw 80–100° không
  có episode nào.
- **Không biết lúc nào dừng** (mục 5.2g): quá 2.5 s `hold`, model làm lại task và gắp chính lon trong khay. Trong 3 ca cùng
  phía hỏng ở lần eval B gần nhất, 2 ca là lỗi này.

### Nhận xét về dữ liệu train B (25/09, chưa kiểm chứng, xem bước 0b)

- **Demo đổi phía đi quá sát lon kia.** `OTHER_CAN_MARGIN` = 1 cm. Khoảng cách nhỏ nhất từ cánh tay hoặc lon đang cầm tới lon kia:

  | Demo train | Cùng phía | Đổi phía |
  |---|---|---|
  | Trước khi kẹp, trung vị / tỉ lệ < 2 cm | 9.8 cm / 4% | 1.4 cm / 60% |
  | Sau khi kẹp, trung vị / tỉ lệ < 2 cm | 10.5 cm / 2% | 1.8 cm / 58% |

  Ở tập test đổi phía, demo của episode hỏng có khoảng hở sau khi kẹp trung vị 2.3 cm, của episode thành công 5.8 cm (không ca
  nào < 2 cm). Lon kia nằm sẵn ở phía khay nên đường mang lon đi ngang qua nó. Trong code: `LIFT_HEIGHT = 0.10` tính từ tâm
  lon, nên đáy lon đang cầm cao khoảng 10 cm so với mặt bàn, thấp hơn đỉnh lon đứng (11.8 cm). Đoạn `carry` nội suy trong không
  gian khớp (`_free_path`) nên tay có thể võng xuống.
- **Thiếu lon đứng ở ca đổi phía:** 52/200 demo đổi phía là lon đứng vì planner hay bỏ. Tập test: 0/4.
- **Độ chính xác khi kẹp:** sim kẹp khi điểm gắp cách tâm lon < 5 cm (`GRASP_RADIUS`). Ca thành công có 90% dưới 4.5 cm, lệch
  2.5–4.8 cm, nên chỉ còn 0.5–2 cm dư.
- **Hông tì vào bàn:** 49/400 demo train (đều là lon nằm) thân ngả khoảng 13°, pelvis trượt 13 cm rồi tì mép bàn tới hết
  episode. 4 demo như vậy trong tập test đều hỏng khi eval. Demo bình thường ngả khoảng 6°, trượt 5 cm. Khi về ready vẫn ngả
  khoảng 3°.
- **Bàn chân trượt xoay** khi tay vung chéo người: yaw thân lệch tới 7.5° ở 10% demo đổi phía, max 12.6°.
- **Model không thấy tư thế thân:** state chỉ có tay, bàn tay, eo (`wrist_eef_9d` tính trong khung pelvis). Planner lại bù theo
  pose pelvis đo được, nên action có chứa phần bù mà model chỉ đoán được qua `ego_view` (gắn trên `torso_link`).

Những gì **không** làm theo nhận xét và lý do:
- **Không tăng `OTHER_CAN_MARGIN` lên 4 cm.** Với fixed, 2 cm đã mất 1/4 ca lon đứng. 4 cm sẽ làm mất thêm đúng nhóm lon đứng
  đổi phía đang thiếu. Thay bằng mang lon qua phía trên lon kia (bước 4).
- **Không bỏ demo chỉ theo ngưỡng pelvis trượt 8 cm.** Ngưỡng này hợp với chế độ fixed. Với balance, lọc theo pelvis chạm bàn và
  góc ngả, và phải sinh bù cho đủ quota (bước 7). Chỉ bỏ thì nhóm lon nằm, vốn yếu nhất, còn ít dữ liệu hơn.
- **Chưa đưa độ ngả pelvis vào state** (xem phần tuỳ chọn cuối plan). Projected gravity chỉ cho pitch/roll, không cho độ trượt
  hay yaw của pelvis. Với balance, độ ngả gần như không đổi (mục tiêu < 2°).

## Bước 0: Chốt mốc đối chứng ở chế độ fixed

Server đang sẵn, nên chạy A-10000 với cùng cấu hình như B:
- `eval/A10000_ik_test_20s_stop` và `eval/A10000_swap_test_20s_stop` (`--seconds 20 --stop-on-success`, chân fixed).
- Ghi bảng A/B vào memo. Checkpoint C sẽ được so với bảng này.

## Bước 0b: Kiểm chứng số liệu nhận xét (chạy song song với bước 0)

- Viết `tools/demo_quality.py`. Script phát lại từng demo và đo:
  - khoảng cách nhỏ nhất từ cánh tay và lon đang cầm tới lon kia, tách trước và sau khi kẹp;
  - pelvis có chạm bàn không, góc ngả max, độ trượt của pelvis (xy), độ xoay yaw;
  - nhóm cảnh: đứng/nằm × cùng/đổi phía; khoảng cách giữa hai lon.
- Chạy trên `g1_can_sort_swap_hold` và `g1_can_sort_swap_hold_test`, so với số liệu phần Context.
- Nối với kết quả eval B-9000: khoảng hở sau khi kẹp còn phân biệt được ca thành công và ca hỏng không, khi đã tính tới khoảng cách
  giữa hai lon? Nếu chỉ do hai lon đặt sát nhau thì bước 4 cần xem lại cả phân bố cảnh (`--cross-can-y`, `--can-gap`).
- Kiểm tra sai lệch điểm khép tay khi eval có lệch theo một hướng cố định không (ví dụ luôn thiếu tầm). Nếu có thì nguyên nhân
  là tư thế thân, không phải độ chính xác của model.

## Bước 1: `collect_demos.py` hỗ trợ balance

- Thêm cờ `--lower-body {fixed,balance}` (mặc định `fixed`) và ghi `"lower_body"` vào `cfg` của `meta/collection.jsonl`.
- `setup_episode` đọc `cfg.get("lower_body", "fixed")` rồi truyền vào `ClosedLoopVLAPipeline`. Dataset cũ không có khoá này
  nên vẫn tái tạo đúng như trước.
- `run_episode` và `replay_actions` bước sim qua `pipe._lower_body(pipe.wbc.track_joint_targets(target))`, giống đường chạy
  model. Hiện cả hai hàm gọi thẳng `track_joint_targets`.
- `verify_demos.py`, `tools/eval_closed_loop.py`, `launch_vla_simulation.py --scene-dataset`: lấy chế độ chân từ `cfg` của
  episode nếu không truyền cờ, để tránh chạy nhầm chế độ.
- Ghi các chỉ số của bước 0b vào `collection.jsonl` ngay lúc thu (dùng chung hàm với `tools/demo_quality.py`). Ghi cả lần thử
  bị bỏ kèm lý do: IK, va chạm, đặt lon, phát lại, thân (chạm bàn/ngả).
- Kiểm tra:
  - `verify_pipeline.py` qua 5/5 test.
  - `verify_demos.py` trên dataset cũ vẫn phát lại 400/400.
  - Sinh 4 episode balance và phát lại 4/4.

## Bước 2: Chạy thử planner với balance (điều kiện để đi tiếp)

Lần thử balance ở mục 5.2f chưa có ca đổi phía. Chạy planner trên 40 seed mỗi lon với `--swap-prob 0.5`, báo cáo tách theo 4
nhóm (đứng/nằm × cùng/đổi phía):
- Tỉ lệ thu được và phân loại lý do bỏ.
- Góc ngả max, độ trượt pelvis, yaw lệch, pelvis chạm bàn.
- Phân bố khoảng hở tới lon kia, trước và sau khi kẹp.

**Điều kiện đi tiếp:**
- Tỉ lệ thu được không thấp hơn dataset B quá 10 điểm ở từng nhóm, phát lại đạt 100%.
- Không episode nào có pelvis chạm bàn; góc ngả max < 5°; **bàn chân** xoay < 3°.
  (Sửa sau khi chạy: yaw pelvis không dùng được làm tiêu chí vì với balance pelvis xoay ngược chiều eo tới ~9° qua khớp
  hông rồi tự trả lại, bàn chân đứng yên. Kết quả: đạt, memo mục 5.2n.)

Nếu không đạt: chỉnh `settle`, `height_cmd`, giới hạn eo, hoặc chậm đoạn `carry` (vung tay nhanh làm bàn chân trượt xoay) trước
khi làm các bước sau.

## Bước 3: Kiểu gắp lon nằm nhất quán

- Công cụ đo: bảng kiểu gắp theo yaw như mục 5.2g, dựa trên wrist pitch ở khung khép tay. Đưa đoạn phân tích đó thành
  `tools/grasp_modes.py` để dùng lại.
- Cách sửa, thử theo thứ tự:
  1. Chọn dấu `s` theo quy tắc cố định: trục ngón hướng ra xa thân robot, hoặc cùng chiều một vector tham chiếu của từng tay.
     Chỉ đổi sang `s` còn lại khi chiều ưu tiên không giải được IK.
  2. Nếu quy tắc 1 làm mất nhiều episode, thêm chi phí phạt cố định cho `s` không ưu tiên, thay vì cấm hẳn. Ranh giới đổi
     kiểu khi đó bị đẩy ra xa vùng yaw hay gặp.
- **Đạt khi:** trong mỗi ô yaw 20°, mỗi tay có ít nhất 90% episode cùng một kiểu, và tỉ lệ thu được lon nằm giảm không quá
  5 điểm.
- **Đã làm (memo mục 5.2p):** quy tắc cố định `LYING_MODE_REF_DEG` (tay phải 125°, tay trái 215°), dùng kiểu còn lại khi kiểu ưu
  tiên không lập được. Ô đạt 25/33 → 28/33; Coca cùng phía đổi kiểu 6 → 2 lần (chỉ còn ranh giới quanh yaw 40°); tỉ lệ thu
  được lon nằm 63% → 58% (−5.3 điểm, sát ngưỡng).

## Bước 4: Mang lon qua phía trên lon kia

- Đoạn `carry`: khi lon kia nằm gần đường đi (luôn gặp ở ca đổi phía), thêm điểm trung gian để đáy lon đang cầm cao hơn đỉnh lon
  kia ít nhất `CARRY_CLEARANCE` ≈ 4 cm, rồi mới sang trên khay (`ABOVE_TRAY`). Có thể nâng thẳng lên độ cao đó ngay ở đoạn `lift`
  hoặc thêm một đoạn `_pose_path` riêng, nên chọn cách có IK dễ giải hơn.
- Đoạn tay vắt qua người trước khi kẹp: giữ cẳng tay cao hơn đỉnh lon kia một khoảng tương tự, khi hình học cho phép.
- Sau đó mới quét `OTHER_CAN_MARGIN` 1 / 1.5 / 2 cm trên khoảng 120 episode đổi phía. **Không** tăng lên 4 cm.
- **Đạt khi:**
  - Sau khi kẹp: khoảng hở trung vị ≥ 5 cm, tỉ lệ < 2 cm dưới 5%.
  - Trước khi kẹp: tỉ lệ < 2 cm dưới 15%.
  - Tỉ lệ thu được ở ca đổi phía giảm không quá 10 điểm.
- Rủi ro: tay vươn cao hơn nên IK dễ hỏng hơn và thân ngả nhiều hơn. Đo lại bằng các chỉ số của bước 2.
- **Đã làm (memo mục 5.2o):** bậc khoảng hở cho cả đoạn tiến vào lẫn mang lon, đo bằng `mj_geomDistance`; nhấc lon cao hơn
  lon kia. Đổi phía: thu được 22 → 31/40; đang kẹp < 2 cm 41% → 10%; trước kẹp < 2 cm 77% → 39%. Chưa đạt ngưỡng vì tay chạy
  lệch kế hoạch 2–3 cm; phần còn lại xử lý bằng bộ lọc ở bước 7.

## Bước 5: Đủ dữ liệu lon đứng ở ca đổi phía

- Dựa vào lý do bỏ ghi ở bước 1, tìm vì sao planner hay bỏ nhóm này. Nếu một lý do chiếm đa số thì sửa planner cho lý do đó.
- Vẽ vùng phủ của nhóm bằng `tools/plot_can_positions.py`, kiểm tra demo giữ lại không dồn vào vùng dễ (hai lon cách xa nhau).
  Quota ở bước 7 chỉ bù số lượng, không bù được vùng mà planner không giải được.
- **Đã làm (memo mục 5.2q):** `tools/trial_coverage.py`. Tìm ra hai vùng hổng và sửa planner: lon nằm đổi phía yaw 30–120°
  (servo cho waist_yaw bù pelvis xoay; 59% → 77%), lon đứng đổi phía khi hai lon cách < 10 cm (điểm chờ hạ thẳng từ trên;
  0/50 → 24/50). 80 seed trộn: 59 → 68/80.

## Bước 6 (tuỳ chọn): Episode "đã xong"

- Thêm `--done-prob` (khoảng 0.1): lon nhiệm vụ nằm sẵn trong khay đúng, tay ở ready có nhiễu khớp, action là đứng yên trong
  khoảng 4 s. Ghi `"done": true` trong `collection.jsonl`.
- Mục đích: dạy model rằng thấy lon đã trong khay thì không làm gì, thay vì dựa vào bộ phát hiện xong bên ngoài.
- Kiểm tra: closed-loop 20 s **không** có `--stop-on-success` thì `ok` phải xấp xỉ `ok_any`.
- **Đã làm (memo mục 5.2r):** `--done-prob`, 4 s đứng yên (100 khung), cờ `done` bốc bằng bộ sinh số riêng nên cảnh cũ không đổi.

## Bước 7: Sinh dataset và tập test

- Thêm quota theo nhóm vào `collect_demos.py`: mỗi lon có 4 nhóm (đứng/nằm × cùng/đổi phía), mỗi nhóm 50 episode. Mỗi nhóm sinh
  tiếp cho tới khi đủ quota, thay vì chỉ đặt `--swap-prob`/`--lying-prob`.
- Lọc lúc thu: **đã có** (`collect_demos.py --quality-filter`, mặc định bật, memo mục 5.2o). Loại episode chạm hoặc cách lon
  kia < 1 cm (trước kẹp / đang kẹp), và với balance: chạm bàn, ngả > 5°, bàn chân xoay > 5° (3° loại ~10% lon nằm bình thường,
  memo 5.2p). Thử 80 seed: thu được 59/80, lon nằm đổi phía 7/16. Quota chỉ cần sinh bù. Episode bị bỏ ghi trong
  `meta/rejected.jsonl`.

| Dataset | Lệnh chính | Ghi chú |
|---|---|---|
| `g1_can_sort_bal` | quota 50/nhóm/lon (400 episode), `--lower-body balance` (+ `--done-prob 0.1` nếu làm bước 6) | Cùng seed bắt đầu như B |
| `g1_can_sort_bal_test` | quota 5/nhóm/lon (40 episode), `--seed 100000`, cùng cfg | Tập test open-loop |

- Chạy `verify_demos.py` (phát lại 100%), `tools/demo_quality.py`, `tools/plot_can_positions.py`, và bảng kiểu gắp của bước 3
  trên dataset thật.
- Nén zip rồi tải lên `MyDrive/gr00t/`.
- Tính trước số bước train theo số khung, như mục 5.2d-bis: muốn đi qua dữ liệu khoảng 8.5 lần thì số mẫu ≈ 8.5 × số khung.
- **Đã làm (memo mục 5.2s):** `--group-quota 50 --done-per-task 20` (440 episode, 99 343 khung) và tập test 44 episode; phát lại
  100%; zip 434 MB / 43 MB; `MAX_STEPS` ≈ 13 200.

## Bước 8: Fine-tune và đánh giá checkpoint C

- Train từ base `nvidia/GR00T-N1.7-3B`, cùng cấu hình như B (batch 64, LR 1e-4, tag `NEW_EMBODIMENT`). Chỉ đổi dataset và
  `MAX_STEPS`.
- Open-loop (Bước 11 của notebook) trên `g1_can_sort_bal_test`, mỗi 1000 bước từ 5000 trở đi. Chọn checkpoint có MAE test thấp
  nhất.
- Closed-loop, **cùng các cảnh với A/B** (cảnh dựng lại từ seed nên dùng lại được tập test cũ):

| Lượt | Tập | Chân | Tham số |
|---|---|---|---|
| `C_bal_test_20s_stop` | `g1_can_sort_bal_test` | balance (theo cfg) | `--seconds 20 --stop-on-success --workers 4` |
| `C_ik_test_20s_stop` | `g1_can_sort_ik_test` | balance | `--seconds 20 --stop-on-success --workers 4` |
| `C_swap_test_20s_stop` | `g1_can_sort_swap_hold_test` | balance | như trên |
| `C_ik_test_20s` (nếu làm bước 6) | `g1_can_sort_ik_test` | balance | `--seconds 20`, không `--stop-on-success` |

- Tách kết quả theo 4 nhóm và phân loại lỗi: hụt khi kẹp, xô lon kia, gắp lại, đặt trượt khay.
- **Tiêu chí thành công:**
  - C (balance) ≥ B (fixed) trên cả hai tập; riêng Coca nằm tăng rõ so với 60%.
  - Lon đứng ở ca đổi phía > 0/4.
  - Xô lon kia không còn là lỗi chính ở tập đổi phía.
  - Góc ngả max dưới 5°.

## Tuỳ chọn, sau khi có kết quả C: đưa độ ngả pelvis vào state

- Chỉ làm nếu C vẫn có lỗi liên quan tới thân.
- Dùng projected gravity (`G1Sim.get_projected_gravity`, robot thật có từ IMU), thêm khoá state mới.
- Cái giá: đổi layout state nên mất tương thích REAL_G1, phải sửa `bridge/real_g1_adapter.py`, server, `finetune/g1_can_sort_config.py`
  và `stats.json`, rồi sinh lại hoặc tính lại state. Ở chế độ fixed, độ ngả trùng với pha "đang vươn tay", nên model có thể học lối
  tắt từ proprioception.

## Rủi ro

- **Planner hỏng nhiều hơn với balance:** policy thăng bằng dời pelvis khi tay vươn xa, nên kế hoạch IK tính từ tư thế đầu có
  thể lệch. Bước 2 dùng để phát hiện sớm; planner vốn lập kế hoạch lại từng đoạn từ trạng thái hiện tại.
- **Mang lon cao hơn làm mất episode:** tay vươn cao dễ vượt giới hạn khớp, nhất là ca đổi phía. Nếu mất nhiều thì chỉ nâng khi lon
  kia thật sự nằm gần đường đi.
- **Quy tắc kiểu gắp làm mất episode lon nằm:** chiều ưu tiên có thể không giải được IK ở một số yaw. Phương án phạt mềm ở bước 3
  là dự phòng.
- **Quota che vùng khó:** đủ số lượng nhưng demo dồn vào cảnh dễ. Bước 5 kiểm tra vùng phủ.
- **Phát lại không tất định:** policy thăng bằng là mạng ONNX. Nếu phát lại không cho kết quả giống hệt lúc thu thì phải kiểm
  tra seed và thứ tự tính toán trước khi sinh cả bộ.
- **Không so thẳng được với A/B:** C khác cả dữ liệu lẫn chế độ chân. Vì vậy phải so trên cùng cảnh test, và giữ bảng A/B ở chế
  độ fixed (bước 0) làm mốc.

## Thứ tự và ước lượng thời gian

Thứ tự: 0 và 0b song song → 1 → 2 (điều kiện đi tiếp) → 3 và 4 → 5 → 6 → 7 → 8.

| Việc | Thời gian |
|---|---|
| Bước 0 (eval A) | khoảng 40 phút, chạy song song với code |
| Bước 0b | 1–2 giờ |
| Bước 1 + 2 | 2–3 giờ |
| Bước 3 + 4 | 3–5 giờ (cần thử nghiệm) |
| Bước 5 | 1–2 giờ |
| Bước 6 | khoảng 1 giờ |
| Sinh 400 + 40 episode | 1.5–2 giờ (balance chạy thêm ONNX, quota cần sinh bù) |
| Train + open-loop | như B |
| Closed-loop | khoảng 1 giờ |
