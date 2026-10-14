# How the SONIC checkpoint choice (sonic_release / sonic_v1_1 / low_latency) affects pairing with GR00T N1.7 (`unitree_g1_sonic`)

Scope and conventions
- Facts are as of 2026-10-01. Code is pinned to `third_party/GR00T-WholeBodyControl` @ b042411 (called **WBC**) and `third_party/Isaac-GR00T` @ 51d4c89 (called **IG**). Checkpoint configs are under `checkpoints/GEAR-SONIC/` (called **CK**). The paper is `2511.07820v4.pdf` (pdftotext page numbers).
- Under **Cited Findings**, every bullet is a FACT taken from the cited NVIDIA code, doc, config, paper or GitHub thread. Under **Inferences**, every bullet is an INFERENCE (my reasoning, not stated by NVIDIA). None of this project's own code (bridge/, sim/, tools/) is used as a source.
- Lookahead arithmetic: "N frames × Δt" means encoder input frames at t, t+Δt, …, t+(N−1)Δt. So the span is (N−1)·Δt. NVIDIA's docs count N·Δt instead (for example, "10 frames ≈ 200 ms"). Both conventions are noted where they differ.

## Q1. Token compatibility: what must be redone if the SONIC checkpoint changes?

### Takeaway
Each SONIC checkpoint has its own latent space. Everything downstream of the encoder is therefore tied to one checkpoint: the recorded `action.motion_token` labels, the GR00T action normalization stats, the fine-tuned GR00T weights, and `LATENT_INITIAL_MOTION_TOKEN`. All of it must be regenerated or retrained together. No runtime check catches a mismatch, because all three checkpoints emit the same 64-dim FSQ grid.

### Cited Findings
- FACT: NVIDIA lists switching SONIC checkpoints as a reason to update the initial token: "each checkpoint has its own latent space — the same token produces different poses across checkpoints". — [WBC docs/source/tutorials/vla_inference.md:346-351](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_inference.md)
- FACT: `initial_poses.py` warns that the shipped token "is specific to the SONIC checkpoint used during training … When changing the SONIC checkpoint, you MUST update LATENT_INITIAL_MOTION_TOKEN". The token is a hard-coded 64-float array. The file does not say which checkpoint it belongs to. — [WBC gear_sonic/utils/inference/initial_poses.py:6-11, 16-31](third_party/GR00T-WholeBodyControl/gear_sonic/utils/inference/initial_poses.py)
- FACT: NVIDIA's documented ways to get a new initial token both depend on the checkpoint: (1) read `action.motion_token` at `frame_index=0` of a good demonstration, or (2) put the robot in the pose via teleop and read the latest token published by the C++ deploy. — [WBC vla_inference.md:368-375](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_inference.md)
- FACT: All three checkpoints use the same quantizer settings: `vector_quantize_pytorch.FSQ`, `num_fsq_levels: 32`, `max_num_tokens: 2` (2 tokens × 32 dims = 64). — [CK sonic_release/config.yaml:72-73, 187-189](checkpoints/GEAR-SONIC/sonic_release/config.yaml); [CK sonic_v1_1/config.yaml:74-75, 191-193](checkpoints/GEAR-SONIC/sonic_v1_1/config.yaml); [CK low_latency/config.yaml:74-75, 191-193](checkpoints/GEAR-SONIC/low_latency/config.yaml). The shipped initial token values are all multiples of 0.0625. — [initial_poses.py:20-29](third_party/GR00T-WholeBodyControl/gear_sonic/utils/inference/initial_poses.py)
- FACT: The only sanity check on VLA output in the inference client is a magnitude bound: a chunk is skipped if `max|motion_token| > 1.25`. Nothing checks which checkpoint produced the token. — [WBC gear_sonic/scripts/run_vla_inference.py:290-296](third_party/GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py)
- FACT: At VLA inference the C++ side takes the token directly (ZMQ protocol v4, "token-only streaming", stored as `external_token_state_` for policy input). The SONIC encoder is bypassed and the token goes to the decoder. — [WBC gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp:82-85, 677-681](third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp). An NVIDIA collaborator confirmed: "If you are streaming tokens as in our VLA workflow, then there is no separate encoder or mode selection involved". — [GitHub NVlabs/GR00T-WholeBodyControl#152](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/152)
- FACT: The paper defines the control decoder as a_t = D_c(z, s^p_t), meaning the action depends on the token plus proprioception. The token is the encoder's quantized latent. — [Paper 2511.07820v4, p.15 "Quantizer"/"Decoders"](2511.07820v4.pdf)
- FACT: GR00T normalizes each state/action key with per-key dataset statistics (min/max, or q01/q99 when percentiles are used). These come from the dataset's stats. — [IG gr00t/data/state_action/state_action_processor.py:150-175](third_party/Isaac-GR00T/gr00t/data/state_action/state_action_processor.py)
- FACT: `unitree_g1_sonic` has a 40-step action horizon (`delta_indices=list(range(40))`). Its action keys are `motion_token`, `left_hand_joints` and `right_hand_joints`, all ABSOLUTE / NON_EEF. — [IG gr00t/configs/data/embodiment_configs.py:67-114](third_party/Isaac-GR00T/gr00t/configs/data/embodiment_configs.py)
- FACT: The hand joints (7+7) bypass SONIC: "The hand action is passed directly from the interface to the robot; it is not controlled by the Sonic policy. If you are using the VLA setup, then the VLA controls it directly." — [GitHub NVlabs/GR00T-WholeBodyControl#159](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/159)

### Inferences
- INFERENCE: Checklist when changing the SONIC checkpoint:
  1. Re-label every episode's `action.motion_token`, either by re-collecting with the new checkpoint or by offline re-encoding (see Q2).
  2. Recompute dataset stats. `motion_token` min/max/q01/q99 change with the latent space; the hand-joint stats do not.
  3. Re-fine-tune GR00T. The flow-matching head has learned the old token distribution, and the normalization params stored in the processor/checkpoint belong to the old labels.
  4. Replace `LATENT_INITIAL_MOTION_TOKEN` with a token taken from data labeled by the new checkpoint (for example, the first frame of a demonstration).
  5. Deploy the matching `--cp` / `--obs-config` pair, plus the ankle-gain flags for v1_1 (Q4).
- INFERENCE: The 14 hand dims are SONIC-independent, so only the 64 token dims need relabeling. However, the hand stats and the camera frames still reflect the robot behavior produced by the old checkpoint. Re-collected data is therefore not identical to relabeled data.
- INFERENCE: Mismatches fail silently, which is the main risk. The shapes are identical (64 dims) and every checkpoint emits values on a shared grid within about [−1, 1], so the 1.25 bound will not fire. The decoder will run another checkpoint's tokens and produce a different pose or motion. This is the failure NVIDIA warns about for the initial token, and it applies equally to every token in the chunk.
- INFERENCE: Because the latent spaces differ, you cannot mix labels from different checkpoints in one dataset.

### Gaps
- Which SONIC checkpoint the shipped `LATENT_INITIAL_MOTION_TOKEN` was produced with is not documented anywhere I could find. The file says only "the SONIC checkpoint used during training".
- NVIDIA publishes no measurement of how far apart the three latent spaces are (for example, decoding one checkpoint's tokens with another's decoder).

## Q2. How labels are produced, and what offline labeling of synthetic/sim data needs per checkpoint

### Takeaway
In NVIDIA's pipeline the label is simply the token that the running C++ SONIC encoder produced during teleop. The data exporter records it at 50 Hz as `action.motion_token`. The encoder used is whichever mode teleop is in: SMPL for whole-body PICO, teleop/3-point for VR-3PT. Offline labeling of sim robot trajectories would naturally use the G1 encoder, and its input window differs by checkpoint:
- release and v1_1: 10 frames at a 0.1 s step (t to t+0.9 s), plus anchor orientation. Release uses the full robot base quaternion; v1_1 uses only the robot's heading (yaw).
- low_latency: 10 frames at a 0.02 s step (t to t+0.18 s), plus full base-quaternion orientation.

### Cited Findings
- FACT (label source): The exporter writes `frame_data["action.motion_token"] = proprio["token_state"]`, or zeros(64) if `token_state` is missing. — [WBC gear_sonic/scripts/run_data_exporter.py:662-665](third_party/GR00T-WholeBodyControl/gear_sonic/scripts/run_data_exporter.py). `token_state` is the "Encoder token output" logged by the C++ deploy. — [WBC gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/state_logger.hpp:35, 116](third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/state_logger.hpp); [encoder.hpp:10-20](third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/encoder.hpp)
- FACT: Recording runs at 50 Hz by default (`data_collection_frequency: int = 50`). — [run_data_exporter.py:71-72](third_party/GR00T-WholeBodyControl/gear_sonic/scripts/run_data_exporter.py); [WBC docs/source/tutorials/data_collection.md:276](third_party/GR00T-WholeBodyControl/docs/source/tutorials/data_collection.md). The LeRobot schema stores `action.motion_token` as float64 with shape (64,). — [WBC gear_sonic/data/features_sonic_vla.py:107-111, 264-268](third_party/GR00T-WholeBodyControl/gear_sonic/data/features_sonic_vla.py)
- FACT: The data-collection launcher forwards `--deploy-checkpoint`, `--deploy-obs-config` and `--deploy-motor-kp/kd-scale` to deploy.sh. So the labels come from whatever checkpoint deploy.sh ran. — [data_collection.md:277-282](third_party/GR00T-WholeBodyControl/docs/source/tutorials/data_collection.md). deploy.sh defaults to `policy/release/model` + `policy/release/observation_config.yaml`. — [WBC gear_sonic_deploy/deploy.sh:240-241](third_party/GR00T-WholeBodyControl/gear_sonic_deploy/deploy.sh)
- FACT: The encoder mode is set by `encoder_mode_4`: g1 = 0, teleop = 1, smpl = 2. — [WBC gear_sonic_deploy/policy/release/observation_config.yaml:57-80](third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml). "The released ONNX model includes all three modes… The existing ZMQ interface will select the corresponding mode for you." — [GitHub #152](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/152). In VR-3PT mode the SMPL columns are all zero, because 3PT uses raw VR poses rather than SMPL. — [data_collection.md:508-534](third_party/GR00T-WholeBodyControl/docs/source/tutorials/data_collection.md)
- FACT: The paper describes the encoders: E_r (robot joints over F_r future frames at Δt_r), E_h (human/SMPL over F_h at Δt_h) and E_m (hybrid: 3-point upper body plus lower-body robot motion over F_m at Δt_m). All are quantized into one shared token space. Paper values: F_r = F_h = F_m = 10, Δt_r = Δt_m = 0.1 s, Δt_h = 0.02 s. — [Paper p.15 "Encoders"; Table S1 p.29](2511.07820v4.pdf)
- FACT: Lookahead parameters by checkpoint (training configs):
  - sonic_release: `num_future_frames: 10`, `dt_future_ref_frames: 0.1` (g1/teleop); `smpl_num_future_frames: 10`, `smpl_dt_future_ref_frames: 0.02`. — [CK sonic_release/config.yaml:331-332, 421-422](checkpoints/GEAR-SONIC/sonic_release/config.yaml)
  - sonic_v1_1: identical to release (10 × 0.1 s; SMPL 10 × 0.02 s). — [CK sonic_v1_1/config.yaml:337-338, 427-428](checkpoints/GEAR-SONIC/sonic_v1_1/config.yaml)
  - low_latency: `dt_future_ref_frames: 0.02` with `num_future_frames: 10` (g1/teleop), and `smpl_num_future_frames: 4` with `smpl_dt_future_ref_frames: 0.02`. — [CK low_latency/config.yaml:335-336, 425-426](checkpoints/GEAR-SONIC/low_latency/config.yaml)
- FACT: The deploy observation configs mirror these values:
  - release: `motion_joint_positions_10frame_step5`, `motion_anchor_orientation_10frame_step5`, `smpl_joints_10frame_step1`. — [policy/release/observation_config.yaml:31-55](third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml)
  - low_latency: "g1/teleop lookahead at step1, body-frame orientation", using `motion_joint_positions_10frame_step1`, `smpl_joints_4frame_step1`, and so on. Encoder input dim 1247. — [CK low_latency/observation_config.yaml:1-10, 39-67](checkpoints/GEAR-SONIC/low_latency/observation_config.yaml)
  - v1_1: "G1/teleop references use 10 frames at step5. SMPL and wrist references use 10 frames at step1. Root orientations are normalized by robot heading." Encoder input dim 1751. — [CK sonic_v1_1/observation_config.yaml:1-7](checkpoints/GEAR-SONIC/sonic_v1_1/observation_config.yaml)
- FACT: NVIDIA's doc describes low_latency only by its SMPL change ("4 future reference frames… approximately 80 ms of reference lookahead, compared with 10 frames (approximately 200 ms)… This is reference lookahead, not total end-to-end system latency"). — [vla_inference.md:117-123](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_inference.md); [WBC docs/source/getting_started/download_models.md:86-97](third_party/GR00T-WholeBodyControl/docs/source/getting_started/download_models.md). The config also shortens the g1/teleop step from 0.1 to 0.02 s (config lines above). The docs do not mention this second change.
- FACT (encoder inputs):
  - G1 encoder: release/LL take `command_multi_future_nonflat` + `motion_anchor_ori_b_mf_nonflat`; v1_1 takes `command_multi_future_nonflat` + `motion_anchor_ori_heading_mf_nonflat`.
  - SMPL encoder: release/LL take `smpl_root_ori_b_multi_future`; v1_1 takes `smpl_root_ori_heading_multi_future`.
  - Teleop encoder: release/LL take `motion_anchor_ori_b`; v1_1 takes `motion_anchor_ori_heading`.
  - Sources: [CK sonic_v1_1/config.yaml:76-127](checkpoints/GEAR-SONIC/sonic_v1_1/config.yaml); [CK low_latency/config.yaml:80, 101, 120](checkpoints/GEAR-SONIC/low_latency/config.yaml); diff of [sonic_release/config.yaml:78, 99, 118](checkpoints/GEAR-SONIC/sonic_release/config.yaml)
- FACT (what orientation modes mean): In the C++ encoder input, mode 0 (`_b`) computes conj(robot **full base quaternion**) × reference root rotation. Mode 1 (`_heading`) uses only the robot's **heading** quaternion (yaw). The result is encoded as the first two columns of the rotation matrix, flattened row-wise (6 numbers per frame). The base quaternion is read from the latest logged robot state. — [WBC gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:608-614, 626-628, 664-684](third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp)
- FACT (episode-end padding convention): When the future frame index passes the motion length, the C++ code clamps it to the last frame ("hold final pose"). — [g1_deploy_onnx_ref.cpp:646-655](third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp)
- FACT: Python reference implementations of the training-side observation functions exist in the WBC repo: `command_multi_future` (:570), `motion_anchor_ori_b_mf` (:1022), `motion_anchor_ori_heading_mf` (:1153), `smpl_root_ori_b_mf` (:1625), `smpl_root_ori_heading_mf` (:1668). — [WBC gear_sonic/envs/manager_env/mdp/observations.py](third_party/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/mdp/observations.py). Locally, all three checkpoints ship `last.pt`. v1_1 and low_latency also ship `model_encoder.onnx`/`model_decoder.onnx` + `observation_config.yaml`. `CK sonic_release/` holds only `config.yaml` + `last.pt`; its ONNX files come from `download_from_hf.py` into `gear_sonic_deploy/policy/release/` (directory listing; [download_models.md:45-48](third_party/GR00T-WholeBodyControl/docs/source/getting_started/download_models.md)).
- FACT: No NVIDIA offline token-labeling tool is merged at b042411. A community PR, "Add Unitree G1 SONIC dataset conversion pipeline… encode 64D motion tokens", is still open. — [GitHub NVlabs/GR00T-WholeBodyControl#251](https://github.com/NVlabs/GR00T-WholeBodyControl/pull/251)
- FACT: The low_latency 4-frame SMPL orientation observation is registered only in mode 0 (body-frame). No heading variant exists. — [g1_deploy_onnx_ref.cpp:1761, 1772, 1781](third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp); raised in the open issue [#231](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/231)

### Inferences
- INFERENCE (window per checkpoint for G1-encoder offline labeling):

  | Checkpoint | Encoder window | Orientation input | Frames clamped at episode end |
  |---|---|---|---|
  | release | t..t+0.9 s (frames t+0, +5, …, +45 at 50 Hz) | robot full base quaternion | last 45 frames |
  | v1_1 | same 0.9 s | robot yaw only | last 45 frames |
  | low_latency | t..t+0.18 s (step 1) | robot full base quaternion | last 9 frames |

  The SMPL encoder (only if you have SMPL) covers t..t+0.18 s in release/v1_1 and t..t+0.06 s in low_latency.
- INFERENCE (the labeling encoder does not match NVIDIA's): In NVIDIA's own data, the token comes from the SMPL encoder (whole-body PICO) or the teleop encoder (VR-3PT), not the G1 encoder. Offline labels from the G1 encoder are a different input path into the "shared" space (paper p.15). The token distribution may differ from teleop-collected labels. Mixing the two in one fine-tuning set is not validated anywhere in NVIDIA's docs.
- INFERENCE (easiest/hardest to label offline):
  - **v1_1 is easiest when you only have kinematic or retargeted trajectories with no simulated base dynamics.** Its orientation input needs only the robot's yaw. You can take yaw from the reference itself (assuming good tracking), and robot roll/pitch errors do not enter the label.
  - **low_latency is easiest when you have full physics rollouts with logged base quaternions.** Its short window means only 9 clamped frames per episode end and the least dependence on far-future data. Because it uses the full base quaternion, any roll/pitch you synthesize enters the label.
  - **release is hardest.** It combines the 0.9 s window (45 clamped frames per episode end) with full-quaternion dependence. Its deploy ONNX is also not in the local `CK sonic_release/` folder.
- INFERENCE: For causal (live) labeling, "future" frames must exist when the token is produced. With a 0.9 s window, an offline pipeline must buffer or look ahead 0.9 s of reference, which is trivial offline. Teleop-style live encoding pays that 0.9 s as delay or planner output.
- INFERENCE: The exporter writes zeros(64) when `token_state` is missing (run_data_exporter.py:665). Zero is a legitimate FSQ code, so such frames are silent bad labels. They should be filtered whichever checkpoint is used.

### Gaps
- NVIDIA publishes no recipe or validation for labeling non-teleop or sim data with the encoder, and no statement on whether G1-encoder tokens are distributionally interchangeable with SMPL/teleop-encoder tokens for VLA training.
- I did not run the encoders to check whether ONNX and `last.pt` give identical tokens for the same input in each checkpoint.

## Q3. Does the token's lookahead interact with GR00T's 40-step / 50 Hz chunk and the client's latency compensation? Is there a documented reason low_latency suits VLA execution?

### Takeaway
SONIC's encoder lookahead is not used at VLA inference time, because tokens are streamed straight to the decoder. The "low latency" of the low_latency checkpoint therefore buys nothing in GR00T's inference loop. Closed-loop timing there is set by the 40 × 20 ms chunk, the 0.4 s re-query interval, and index clamping at 39. Lookahead matters only indirectly: it sets how far into the future each label token "sees", and how responsive teleop was when the data was collected. I found no NVIDIA statement that low_latency is better for VLA execution.

### Cited Findings
- FACT: Client defaults: `action_publish_rate = 50` Hz, `action_horizon = 40`, `rate = 1/0.4` (2.5 Hz). — [run_vla_inference.py:72-79](third_party/GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py); [vla_inference.md:299-301](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_inference.md); [WBC docs/source/tutorials/vla_workflow.md:27-32](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_workflow.md) ("2.5 Hz… 64-dim x 40… 50 Hz")
- FACT (latency compensation): When a chunk arrives, the start index is `clip(round(inference_delay × 50), 0, 39)`. `inference_delay` is measured from `inference_start_time`, which is set after the observation is prepared. — [WBC gear_sonic/utils/inference/vla_utils.py:97-114](third_party/GR00T-WholeBodyControl/gear_sonic/utils/inference/vla_utils.py); [run_vla_inference.py:329-332, 645-652](third_party/GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py)
- FACT (re-query schedule): `last_inference_time` is reset when a chunk is **received**. A new inference starts once 0.4 s have passed and the worker is idle. — [run_vla_inference.py:652, 659-671](third_party/GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py); [vla_utils.py:117-138](third_party/GR00T-WholeBodyControl/gear_sonic/utils/inference/vla_utils.py)
- FACT (clamp): The index increments once per 20 ms loop and is clamped at `action_horizon−1` = 39 (`min(action_chunk_index + 1, 39)`), so the last token is held until the next chunk. — [run_vla_inference.py:713-714, 741](third_party/GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py)
- FACT: In the VLA workflow there is no encoder/mode selection; tokens go straight in (#152 comment above; ZMQ protocol v4 at zmq_endpoint_interface.hpp:82-85).
- FACT: NVIDIA describes low_latency as "configured for responsive whole-body **teleoperation**". Its lookahead reduction "is reference lookahead, not total end-to-end system latency", and the checkpoint "does not replace the default top-level deployment policy". — [vla_inference.md:117-123](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_inference.md); [download_models.md:86-97](third_party/GR00T-WholeBodyControl/docs/source/getting_started/download_models.md). The VLA inference doc gives launch commands for low_latency, so it is a supported option, but no rationale is given for VLA. — [vla_inference.md:131-156, 247-266](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_inference.md)
- FACT: The only VLA-specific selection wording in the docs is for v1_1: "Use the `sonic_v1_1/` checkpoint when the VLA policy was trained against the robot-heading-normalized SONIC controller… It is not the low-latency checkpoint." — [vla_inference.md:77-82](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_inference.md)
- FACT: The issue tracker has no thread asking which SONIC checkpoint to use for VLA. I scanned about 278 WBC issues/PRs and 300 Isaac-GR00T issues/PRs via the GitHub API. Related low_latency threads concern teleop/planner mode:
  - [#190](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/190) (closed): LL unstable in VR-3PT Planner mode "for VLA data collection". The user later traced it to their own control-loop slowdown and said "The low-latency mode appears to have no issues".
  - [#231](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/231) (open): LL "loses yaw responsiveness in Planner VR3PT mode".
- FACT: A community question about FSQ reconstruction jitter (a small latent change jumps a quantization level) is open. NVIDIA replied only that FSQ's motivation is a compact shared discrete token space. — [#154](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/154)

### Inferences
- INFERENCE (timing arithmetic, the same for every checkpoint): Let L be the server latency.
  - A chunk starts at index ≈ round(50L) and the next chunk arrives about 0.4 + L s later. By then the index has reached about 20 + 100L.
  - Clamping at 39 starts when L > ~0.19 s. Beyond that, the robot holds the 0.78 s-ahead token until the next chunk. Example: L = 0.25 s gives ~0.12 s of hold.
  - In normal operation (L ≈ 0.1 s), indices ≈ 5…35 are used, that is, tokens 0.1 to 0.7 s ahead of the observation. None of this depends on the SONIC checkpoint.
- INFERENCE (how lookahead does interact): A label at chunk index j encodes reference motion over [t+0.02j, t+0.02j+H].
  - H = 0.9 s for release/v1_1 G1/teleop-encoder labels. The 40-step chunk then implicitly asks GR00T to commit to intent up to ~1.7 s ahead.
  - H ≈ 0.18 s for release/v1_1 SMPL labels and for LL G1/teleop labels, giving ~0.96 s.
  - H ≈ 0.06 s for LL SMPL labels, giving ~0.84 s.
  - A longer H gives smoother, more anticipatory tokens. A shorter H gives tokens that change faster and carry less anticipatory information, which the 40-step chunk itself then has to supply.
  - These are hypotheses about learnability, not measured results.
- INFERENCE: Holding index 39 means repeating one token. With long-horizon tokens this repeats a 0.9 s "plan"; with LL it repeats a ~0.06–0.18 s plan. The physical effect of a held token is undocumented for every checkpoint.
- INFERENCE: The documented benefit of low_latency (shorter reference lookahead, so less teleop lag) acts at **data collection**. The operator sees the robot follow more promptly, which may improve demonstration quality. It does not act at VLA execution. Choosing low_latency "for VLA execution latency" has no documented support.

### Gaps
- NVIDIA publishes no experiment comparing VLA success across the three SONIC checkpoints, and no analysis of how token lookahead interacts with chunk length.
- How the decoder behaves under a repeated (held) token is not documented.

## Q4. Deployment differences relevant to VLA

### Takeaway
All three checkpoints use the same VLA client and the same 78-dim action. They differ in the deploy model/obs-config pair. v1_1 additionally adds ankle-pitch Kp/Kd × 1.5, which NVIDIA says improves observed wrist tracking. The model pair and the gain flags should match between data collection and inference.

### Cited Findings
- FACT: v1_1 deploy: `--cp policy/sonic_v1_1/model --obs-config policy/sonic_v1_1/observation_config.yaml --motor-kp-scale 4,10=1.5 --motor-kd-scale 4,10=1.5`. "This v1.1 tuning scales the left and right ankle-pitch motors (hardware indices 4 and 10). The resulting whole-body stability improves observed wrist tracking; it does not directly scale the wrist motors." — [vla_inference.md:84-103](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_inference.md)
- FACT: The Python launcher equivalents are `--deploy-checkpoint`, `--deploy-obs-config`, `--deploy-motor-kp-scale` and `--deploy-motor-kd-scale`. They are forwarded verbatim to deploy.sh as `--cp`, `--obs-config`, `--motor-kp-scale` and `--motor-kd-scale`. — [WBC gear_sonic/scripts/launch_inference.py:101-119, 319-332](third_party/GR00T-WholeBodyControl/gear_sonic/scripts/launch_inference.py); [vla_inference.md:105-115, 150-156, 307-312](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_inference.md). The data-collection launcher has the same flags. — [data_collection.md:277-282](third_party/GR00T-WholeBodyControl/docs/source/tutorials/data_collection.md)
- FACT: low_latency deploy: `--cp policy/low_latency/model --obs-config policy/low_latency/observation_config.yaml`, with no gain flags. `--cp` is a prefix; deploy.sh appends `_encoder.onnx`/`_decoder.onnx`. — [vla_inference.md:136-146](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_inference.md); [deploy.sh:381-382](third_party/GR00T-WholeBodyControl/gear_sonic_deploy/deploy.sh)
- FACT: The LL obs config says it "Requires the smpl_*_4frame_step1 entries in GetObservationRegistry() + a rebuild". At b042411 those entries exist. — [CK low_latency/observation_config.yaml:9-10](checkpoints/GEAR-SONIC/low_latency/observation_config.yaml); [g1_deploy_onnx_ref.cpp:1761, 1772, 1781](third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp)
- FACT (training differences that change robot behavior under the same token stream):
  - v1_1 adds `randomize_wrist_poses: true` ([CK sonic_v1_1/config.yaml:433](checkpoints/GEAR-SONIC/sonic_v1_1/config.yaml)).
  - v1_1 and LL add a `tracking_vr_2wrists_local_ori` reward ([v1_1 :713](checkpoints/GEAR-SONIC/sonic_v1_1/config.yaml), [LL :710](checkpoints/GEAR-SONIC/low_latency/config.yaml)).
  - v1_1 and LL widen wrist/torso mass randomization from 0.8–1.5× (release) to 0.8–2.5× ([release :856-857](checkpoints/GEAR-SONIC/sonic_release/config.yaml), [v1_1 :880-881](checkpoints/GEAR-SONIC/sonic_v1_1/config.yaml), [LL :877-878](checkpoints/GEAR-SONIC/low_latency/config.yaml)).
  - The training motion mixes differ: release uses bones_seed data; LL uses `train_pyroki_0213_26`; v1_1 uses `train_pyroki_v081_0414_26` ([release :297-298](checkpoints/GEAR-SONIC/sonic_release/config.yaml), [LL :301-302](checkpoints/GEAR-SONIC/low_latency/config.yaml), [v1_1 :301-304](checkpoints/GEAR-SONIC/sonic_v1_1/config.yaml)).
  - Run-name timestamps in the configs are: release 2026-01-18 (resumed 2026-02-04), LL 2026-03-23, v1_1 2026-04-15 (`checkpoint:`/`experiment_dir:` lines 8-13 of each config.yaml).
- FACT: Pressing `i` linearly blends from the last sent token to `LATENT_INITIAL_MOTION_TOKEN` over 1 s at 50 Hz (or snaps if no prior token). — [run_vla_inference.py:440-494](third_party/GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py); [vla_inference.md:377-396](third_party/GR00T-WholeBodyControl/docs/source/tutorials/vla_inference.md)

### Inferences
- INFERENCE: The robot state is a GR00T input: joint groups plus `projected_gravity` ([embodiment_configs.py:72-84](third_party/Isaac-GR00T/gr00t/configs/data/embodiment_configs.py)). The base orientation also feeds the encoder's anchor-orientation term during labeling. Using different ankle gains at collection and at inference shifts both the state distribution and, for teleop-recorded labels, the labels themselves. Use the same gain flags at both stages, especially for v1_1.
- INFERENCE: The initial-pose blend interpolates linearly in token space between two FSQ codes. Intermediate points are off-grid. Whether they decode safely is checkpoint-specific and untested by NVIDIA documentation.
- INFERENCE: Wrist-pose augmentation, the wrist-orientation reward and heavier wrist mass randomization point v1_1 (and partly LL) toward better wrist tracking under varied payloads. This is relevant to manipulation VLA, but NVIDIA has published no VLA-level measurement of it.

### Gaps
- No NVIDIA measurement quantifies the v1_1 ankle-gain effect on VLA task success. The doc states only the qualitative claim "improves observed wrist tracking".

## Q5. Has NVIDIA published a GR00T checkpoint or dataset for `unitree_g1_sonic`, and with which SONIC checkpoint?

### Takeaway
No. NVIDIA has published neither a fine-tuned GR00T checkpoint nor a dataset for `unitree_g1_sonic`. The published N1.7 + SONIC results (Isaac-GR00T example) and the paper's N1.5 VLA results do not say which SONIC checkpoint was used. `UNITREE_G1_SONIC` is a posttrain tag that requires the user's own fine-tuned model.

### Cited Findings
- FACT: Isaac-GR00T says `UNITREE_G1_SONIC` is a posttrain tag that "require[s] a finetuned checkpoint". Its table row links to the GEAR-SONIC VLA workflow instead of a checkpoint. Rows for DROID, LIBERO and SimplerEnv link to `nvidia/GR00T-N1.7-*` checkpoints. — [IG getting_started/policy.md:43, 66-70](third_party/Isaac-GR00T/getting_started/policy.md)
- FACT: The IG README lists the finetuned checkpoints LIBERO, DROID, SimplerEnv-Bridge and SimplerEnv-Fractal, with no SONIC entry. Its SONIC section points to the WBC tutorials and notes the end-to-end collect-finetune-deploy workflow "is only supported for GEAR-SONIC (`UNITREE_G1_SONIC`)". — [IG README.md:275-278, 430-440](third_party/Isaac-GR00T/README.md)
- FACT: The Hugging Face API (queried 2026-10-01) returns these nvidia models matching "GR00T-N1.7": 3B, LIBERO, DROID, SimplerEnv-Bridge, SimplerEnv-Fractal, `GR00T-N1.7-ApplePnP-V1`, `GR00T-H-N1.7`. A search for "sonic" returns only `nvidia/GEAR-SONIC`. `GR00T-N1.7-ApplePnP-V1` is a G1 model whose exported decoder outputs `left_arm, right_arm, left_hand, right_hand, waist, navigate_command, base_height_command…`. That is the decoupled-WBC action space, not SONIC motion tokens. — [HF api/models?author=nvidia&search=GR00T-N1.7](https://huggingface.co/api/models?author=nvidia&search=GR00T-N1.7); [HF api/models?author=nvidia&search=sonic](https://huggingface.co/api/models?author=nvidia&search=sonic); [ApplePnP-V1 exported_leapp.yaml](https://huggingface.co/nvidia/GR00T-N1.7-ApplePnP-V1/raw/main/exported_leapp.yaml)
- FACT: Hugging Face datasets search (nvidia, "sonic") returns nothing. A search for "G1" returns `PhysicalAI-Robotics-GR00T-Teleop-G1`, `Arena-G1-Loco-Manipulation-Task`, `g1_locomanip_dataset` and `Arena-G1-Static-PickNPlace-Task`. None of their READMEs mention SONIC or `motion_token` (grep). — [HF api/datasets?author=nvidia&search=sonic](https://huggingface.co/api/datasets?author=nvidia&search=sonic)
- FACT: The IG example `examples/GR00TWholeBodyControl/README.md` reports real-G1 N1.7 + SONIC results: object pickup 159/216 = 73.6% (about 18k episodes, 60k iterations, batch 256) and soda-can-to-trash 8/12 = 66.7% (about 150 episodes, 20k iterations, batch 32). It says only that "The robot-side controller follows the GR00T Whole-Body Control VLA inference workflow". The SONIC checkpoint is not named. — [IG examples/GR00TWholeBodyControl/README.md:15-16, 62, 100](third_party/Isaac-GR00T/examples/GR00TWholeBodyControl/README.md)
- FACT: The paper's whole-body VLA (GR00T N1.5) predicted a 78-dim action (64 token + 14 hand). It reports 75% average success over 5 tasks, and token-vs-SMPL-action ablations (Tab. 1B). The SONIC checkpoint variant is not named. — [Paper p.9, p.11 Tab. 1, p.33 S1.4](2511.07820v4.pdf). NVIDIA restated the ablation (68% vs 27% average) in [#236](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/236).
- FACT: The N1.7 processor maps embodiment projector slot 11 to `{"unitree_g1_sonic"}`. — [IG gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py:76](third_party/Isaac-GR00T/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py). The config sits under the "Pre-registered posttrain configurations" comment. — [embodiment_configs.py:66-67](third_party/Isaac-GR00T/gr00t/configs/data/embodiment_configs.py)
- FACT: When an NVIDIA collaborator was asked how GR00T integrates with SONIC, the answer was "We change Gr00t's output to latent codes for SONIC", and the thread was later closed by pointing to the released workflow. — [#108](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/108). An Isaac-GR00T issue about running the `PickNApple` (decoupled-WBC) checkpoint with SONIC run_sim_loop shows the action-space mismatch users hit. — [NVIDIA/Isaac-GR00T#661](https://github.com/NVIDIA/Isaac-GR00T/issues/661)

### Inferences
- INFERENCE: There is no public SONIC-token GR00T checkpoint or dataset, so no external artifact forces a particular SONIC checkpoint choice. The team's own data defines the latent space.
- INFERENCE: The defaults (deploy.sh → `policy/release`; launchers default to deploy.sh's defaults) and the timing (the VLA workflow was released 2026-05-07 per #152; the IG README/example) make it plausible that NVIDIA's shipped initial token and its N1.7 results used the default release checkpoint. This is not stated anywhere and should be treated as unverified.

### Gaps
- Which SONIC checkpoint produced (a) the paper's N1.5 VLA results, (b) the IG N1.7 benchmark, and (c) the shipped `LATENT_INITIAL_MOTION_TOKEN` is undocumented.
- Whether the N1.7 base model saw any `unitree_g1_sonic` data in pretraining is not documented; slot 11 is listed among the posttrain tags.

## Q6. Evidence-based criteria for choosing a checkpoint

### Takeaway
Pick one checkpoint and use it for all four stages. Base the choice on four things: (1) the checkpoint any reused labels or tokens came from; (2) how you will label (teleop versus offline from sim, and which state signals you have); (3) whether wrist precision or teleop responsiveness matters more; (4) acceptance of open issues. No NVIDIA evidence ranks the checkpoints by VLA success.

### Cited Findings
- FACT: Latent spaces differ per checkpoint (Q1: vla_inference.md:346-351; initial_poses.py:6-11).
- FACT: NVIDIA's only VLA-directed selection guidance is "Use sonic_v1_1 when the VLA policy was trained against the robot-heading-normalized SONIC controller" (vla_inference.md:77-82). Low_latency is framed as a teleoperation checkpoint that "does not replace the default top-level deployment policy" (download_models.md:86-97).
- FACT: v1_1 is documented as heading-normalized, with wrist-pose augmentation and ankle Kp/Kd × 1.5 for better observed wrist tracking (vla_inference.md:77-103; download_models.md:58). Low_latency shortens the SMPL lookahead to 4 frames (docs) and also the g1/teleop step to 0.02 s (config) (Q2). Release is the deploy.sh default (deploy.sh:240-241).
- FACT: Low_latency has an open yaw-responsiveness issue in Planner VR-3PT mode (#231). Its 4-frame SMPL orientation observation has no heading variant (g1_deploy_onnx_ref.cpp:1781).

### Inferences
- INFERENCE, criteria in order:
  1. **Consistency is non-negotiable.** One checkpoint for collection or labeling, stats, fine-tuning, initial token and deployment, with the same deploy model/obs-config/gain flags at collection and inference.
  2. **Reusing existing labels?** Use the checkpoint that produced them. Otherwise relabel and retrain (Q1 checklist).
  3. **Offline labeling of kinematic/sim data with the G1 encoder:** prefer **v1_1**. Its yaw-only orientation input avoids synthesizing robot roll/pitch, and it is the only variant with explicit VLA wording in the docs. Prefer **low_latency** only if you have full physics rollouts and want minimal end-of-episode clamping (9 vs 45 frames). **Release** is the least convenient (0.9 s window, full-quaternion dependence, ONNX not in the local folder).
  4. **Manipulation needing wrist precision (whole-body pick/place):** **v1_1** is the variant whose documentation and training config target wrist tracking. Remember the `4,10=1.5` gain flags at both collection and inference.
  5. **Teleop responsiveness during collection (dynamic, reactive demos):** **low_latency** reduces reference lookahead (operator-to-robot lag). Weigh this against the open Planner-VR-3PT yaw issue, and against the fact that it gives no VLA-inference latency benefit (Q3).
  6. **Lowest integration risk and matching NVIDIA defaults:** **release** is the default deploy target, so default launch commands work without extra flags. Whether the shipped initial token is release-specific is unverified, so still regenerate it from your own first frames.
- INFERENCE: Before committing, run a cheap empirical check for each candidate. Decode a few labeled episodes open-loop in sim with that checkpoint's decoder (re-encode, then decode, then compare to the source trajectory). Also verify that the regenerated initial token decodes to a safe stand. NVIDIA's Isaac-GR00T example likewise recommends checking "open-loop robot joint trajectory matching" before real deployment ([IG examples/GR00TWholeBodyControl/README.md](third_party/Isaac-GR00T/examples/GR00TWholeBodyControl/README.md), Notes under "Data-Train-Eval Workflow").

### Gaps
- No NVIDIA or community A/B data compares VLA task success across the three checkpoints, so the ranking above rests on documented properties and reasoning, not measured outcomes.
