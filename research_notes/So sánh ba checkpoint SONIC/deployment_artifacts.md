# Deployment artifacts of the three GEAR-SONIC checkpoints (default / sonic_v1_1 / low_latency)

Path shorthands used below (all local, inspected 01/10/2026):
- `CK/` = `/home/thanhhung27/Code/GEAR-SONIC-GR00T-N1.7/checkpoints/GEAR-SONIC/` (HF `nvidia/GEAR-SONIC` download)
- `WBC/` = `/home/thanhhung27/Code/GEAR-SONIC-GR00T-N1.7/third_party/GR00T-WholeBodyControl/` at commit `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0` (2026-09-23; `git log -1`)
- `CPP` = `WBC/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` (4535 lines)
- "Default" = top-level `CK/model_encoder.onnx`, `CK/model_decoder.onnx`, `CK/observation_config.yaml` (PyTorch source `CK/sonic_release/last.pt`; the `CK/sonic_release/` folder contains only `config.yaml` + `last.pt`, no ONNX — `ls -laR CK`).

Inspection commands (binary ONNX files; reproducible):
- I/O, node counts, op histogram, parameter count, linear-layer weight shapes:
  `uv run --no-project --with onnx --with numpy python inspect_onnx.py <file>` where the script does `onnx.load(p, load_external_data=False)`, prints `graph.input` (excluding initializers) / `graph.output` shapes, `len(graph.node)`, `Counter(op_type)`, `sum(prod(init.dims))`, and the first initializer shape feeding each `Gemm`/`MatMul` node. **[CMD-1]**
- Encoder input slicing: dump every non-Constant node with constant inputs resolved (`Slice` starts/ends, `Reshape` targets, quantizer constants). **[CMD-2]**
- Per-mode input sensitivity: `uv run --no-project --with onnxruntime --with numpy --with pyyaml python sensitivity.py` — for each variant, builds the encoder input exactly as the C++ does (blocks concatenated in yaml `encoder_observations` order with registry dims), sets `obs_dict[0]=mode_id`, randomizes one block at a time and records whether `encoded_tokens` changes (>1e-6). **[CMD-3]**
- Weight identity across variants: load all initializers with `onnx.numpy_helper.to_array` and `np.array_equal` per tensor name. **[CMD-4]**
- File sizes: `ls -la`; hashes: `sha256sum`; config identity: `md5sum`. **[CMD-5]**

## Q1. ONNX encoder/decoder I/O, size, parameters and layer structure per variant

### Takeaway
All three variants share the same encoder topology (3 parallel 5-layer SiLU MLP sub-encoders g1/teleop/smpl + identical FSQ quantizer, 64-d token) and the same decoder I/O (`[1,994] -> [1,29]`), but they differ in encoder input width (1762 / 1751 / 1247) and in decoder size: default has a 7-layer 10.2M-param decoder, while sonic_v1_1 and low_latency share a larger 9-layer 37.4M-param decoder architecture (different weights). No weight tensor is shared between any two variants.

### Cited Findings
**File sizes / hashes (FACT, [CMD-5], `ls -la CK`, `sha256sum`)**
| File | Default (top-level) | sonic_v1_1 | low_latency |
|---|---|---|---|
| model_encoder.onnx | 50,100,513 B (sha256 013ab028…) | 50,117,796 B (fb97de22…) | 45,933,505 B (60be4315…) |
| model_decoder.onnx | 40,900,688 B (c7241a12…) | 149,778,638 B (34bae857…) | 149,763,762 B (c4ac2e74…) |
| observation_config.yaml | 2,336 B | 2,475 B | 3,258 B |
| planner_sonic.onnx (shared, top-level only) | 773,952,989 B (39b553e1…) | — | — |

**Graph metadata (FACT, [CMD-1])**: every encoder/decoder is ONNX IR 7, opset 13, producer `pytorch 2.7.0`; all initializers are FP32 (data_type 1) and embedded (0 external initializers). Planner: opset 17, pytorch 2.7.0, 7,932 nodes, 185,097,686 params.

**Encoder I/O and structure (FACT, [CMD-1])**
| | Default | sonic_v1_1 | low_latency |
|---|---|---|---|
| Input | `obs_dict` `[1,1762]` float32 | `obs_dict` `[1,1751]` | `obs_dict` `[1,1247]` |
| Output | `encoded_tokens` `[1,64]` | `[1,64]` | `[1,64]` |
| Nodes / initializers | 241 / 30 | 241 / 30 | 241 / 30 |
| Params | 12,339,392 | 12,339,392 | 11,307,200 |
| g1 sub-encoder Gemm weights | 2048×640, 1024×2048, 512×1024, 512×512, 64×512 | same | same |
| teleop sub-encoder | 2048×267, then 1024×2048, 512×1024, 512×512, 64×512 | same | same |
| smpl sub-encoder | 2048×**840**, then same 4 layers | 2048×**840** | 2048×**336** |
- Op histogram identical for all three encoders: Constant 97, Reshape 25, Mul 19, Gemm 15, Sigmoid 12, Slice 11, Unsqueeze 8, Add 7, Concat 7, Sub 6, Shape 5, Cast 4, ConstantOfShape/Equal/Where/Expand/Tanh/Round/Div 3 each, Gather/ScatterND/Transpose/ReduceSum 1 each ([CMD-1]).
- Activations are SiLU: each hidden Gemm output is multiplied by its own Sigmoid (`/g1/module/module.1/Mul` = Gemm × Sigmoid) ([CMD-2]).
- Quantizer (per sub-encoder, nodes `/quantizer*/…`): reshape 64→[2,32]; `x + 0.032237` → `Tanh` → `× 15.5155` → `− 0.5` → `Round` (straight-through form) → `÷ 16`; identical constants in all three encoders ([CMD-2]). Initializer names are `module.encoders.{g1,teleop,smpl}.module.{0,2,4,6,8}.{weight,bias}` ([CMD-4]).
- Mode selection inside the graph: `Gather(obs_dict, index 0, axis 1)` → Cast → one-hot via `ScatterND`; the three sub-encoder outputs are stacked (`/Concat_5`), multiplied by the one-hot and `ReduceSum`-ed into `encoded_tokens` ([CMD-2]).
- Per-sub-encoder param split (computed from the weight shapes above): g1 4,231,232; teleop 3,467,328; smpl 4,640,832 (default, v1_1) or 3,608,640 (low_latency). Sums reproduce the [CMD-1] totals exactly (12,339,392 and 11,307,200).

**Decoder I/O and structure (FACT, [CMD-1])**
| | Default | sonic_v1_1 | low_latency |
|---|---|---|---|
| Input / output | `obs_dict [1,994]` / `action [1,29]` | same | same |
| Nodes / initializers | 48 / 14 | 56 / 18 | 56 / 18 |
| Linear layers (MatMul weight, in×out) | 994×2048, 2048×2048, 2048×1024, 1024×1024, 1024×512, 512×512, 512×29 | 994×4096, 4096×4096, 4096×2048, 2048×2048, 2048×1024, 1024×1024, 1024×512, 512×512, 512×29 | identical shapes to v1_1 |
| Hidden layers / widths | 6 hidden: 2048-2048-1024-1024-512-512 | 8 hidden: 4096-4096-2048-2048-1024-1024-512-512 | same as v1_1 |
| Params | 10,184,221 | 37,393,949 | 37,393,949 |
- Decoder ops: default MatMul 7, Add 7, Sigmoid 6, Mul 6 (SiLU), Slice 3; v1_1/low_latency MatMul 9, Add 9, Sigmoid 8, Mul 8, Slice 3 ([CMD-1]). Initializer prefix `module.decoders.g1_dyn.module.N` ([CMD-2]).
- Decoder internally splits its input: default `Slice(obs_dict, 0, -930)` + `Slice(obs_dict, -930, end)`; v1_1/low_latency `Slice(obs_dict, 0, 64)` + `Slice(obs_dict, -930, end)` — i.e. 64 token dims + 930 proprioception dims in all three ([CMD-2]). No observation-normalization constants appear in either decoder (only W/b initializers: 14 = 7×2, 18 = 9×2) ([CMD-1]).

**Weight sharing (FACT, [CMD-4])**: identical tensors default-vs-v1_1 0/30 (encoder) and 0/14 (decoder); default-vs-low_latency 0/30, 0/14; v1_1-vs-low_latency 0/30 (encoder), 0/18 (decoder).

**Training-side dims consistent with ONNX (FACT)**: `CK/low_latency/model_config.yaml:301,303` actor_obs 930, tokenizer 1257; `CK/sonic_v1_1/model_config.yaml:301,303` actor_obs 930, tokenizer 1761; `:308` actions_dim 29.

### Inferences
- INFERENCE: the encoder ONNX is roughly ~4× larger in FLOP terms only in the first layer of the smpl branch; the low_latency encoder saves exactly 504×2048 + 0 = 1,032,192 params (smpl input 840→336), so encoder cost is essentially unchanged across variants. The real compute difference is the decoder: v1_1 and low_latency decoders have ~3.7× the parameters of the default (37.4M vs 10.2M), dominated by the 4096×4096 layer (16.8M).
- INFERENCE: v1_1 and low_latency share a newer "large decoder" recipe; the default is an older/smaller recipe. Same decoder shape but 0 identical tensors means separately trained models, not a fine-tune that froze the decoder.
- INFERENCE: FSQ quantizer has 32 levels per dim (Tanh scaled to ±15.5155, minus 0.5 then Round → integers −16…15, ÷16 → token values in [−1, 0.9375] at 1/16 steps), 64 dims; the constants match the standard FSQ even-level formulation (half_l = 31·1.001/2 = 15.5155, shift = atanh(0.5/15.5155) ≈ 0.03224).
- INFERENCE: all three sub-encoders execute on every encoder call (no control flow; selection is a masked sum), so encoder latency is independent of mode.

### Gaps
- TensorRT engine build behaviour (FP16 vs FP32 actually used on Jetson, latency per variant) was not measured; `use_fp16: false` in all three yaml files (see Q2) but runtime timing was not benchmarked.
- The planner ONNX internals (architecture) were not analysed beyond I/O and parameter count.

## Q2. Observation configs: decoder list, encoder observations, modes, frame counts/steps, heading observations

### Takeaway
The decoder (policy) observation list is identical across all three variants (token 64 + 10-frame step1 history, total 994). The encoder configs differ: default keeps 11 unused root-z dims (1762); v1_1 replaces body-frame anchor orientations with robot-heading-normalized ones (1751); low_latency uses step1 G1/teleop references and 4-frame SMPL references (1247). **The low_latency yaml lists the two G1/teleop anchor-orientation blocks in the opposite order from what its ONNX slices, which (by static analysis and an onnxruntime test) mis-feeds the g1 and teleop modes; the smpl mode is aligned.**

### Cited Findings
**Decoder (policy) observations — identical in all three (FACT)**
- Order: `token_state`, `his_base_angular_velocity_10frame_step1`, `his_body_joint_positions_10frame_step1`, `his_body_joint_velocities_10frame_step1`, `his_last_actions_10frame_step1`, `his_gravity_dir_10frame_step1` — `CK/observation_config.yaml:7-23`, `CK/sonic_v1_1/observation_config.yaml:10-21`, `CK/low_latency/observation_config.yaml:14-30`.
- Registry dims: token_state = encoder dimension (64) (`CPP:1710-1711`); his_* 10frame_step1 = 30 / 290 / 290 / 290 / 30 (`CPP:1810-1814`) → 64+930 = 994 = decoder input ([CMD-1]).
- The default yaml header "Total dimension: 436 (64+12+116+116+116+12)" (`CK/observation_config.yaml:3`) describes a 4-frame history (dims of `his_*_4frame_step1`, `CPP:1805-1809`), not the enabled 10-frame list; it is a stale comment (FACT: the enabled entries sum to 994 and the ONNX expects 994). The v1_1 header states "Encoder input: 1751 Decoder input: 994 Token: 64" (`CK/sonic_v1_1/observation_config.yaml:5`); low_latency "Encoder input dimension: 1247 Decoder/policy dimension: 994 token: 64" (`CK/low_latency/observation_config.yaml:6`).
- All three: `encoder.dimension: 64`, `use_fp16: false` (`CK/observation_config.yaml:26-27`; `CK/sonic_v1_1/observation_config.yaml:24-25`; `CK/low_latency/observation_config.yaml:33-34`).

**Encoder observations, in yaml (= C++ concatenation) order, with registry dims**
| # | Default (`CK/observation_config.yaml:29-56`) | dim | sonic_v1_1 (`CK/sonic_v1_1/observation_config.yaml:27-50`) | dim | low_latency (`CK/low_latency/observation_config.yaml:37-68`) | dim |
|---|---|---|---|---|---|---|
| 1 | encoder_mode_4 | 4 | encoder_mode_4 | 4 | encoder_mode_4 | 4 |
| 2 | motion_joint_positions_10frame_step5 | 290 | motion_joint_positions_10frame_step5 | 290 | motion_joint_positions_10frame_**step1** | 290 |
| 3 | motion_joint_velocities_10frame_step5 | 290 | motion_joint_velocities_10frame_step5 | 290 | motion_joint_velocities_10frame_step1 | 290 |
| 4 | motion_root_z_position_10frame_step5 | 10 | motion_anchor_orientation_**heading**_10frame_step5 | 60 | motion_anchor_orientation_10frame_step1 | 60 |
| 5 | motion_root_z_position | 1 | motion_anchor_orientation_**heading** | 6 | motion_anchor_orientation | 6 |
| 6 | motion_anchor_orientation | 6 | lowerbody pos 10f step5 | 120 | lowerbody pos 10f step1 | 120 |
| 7 | motion_anchor_orientation_10frame_step5 | 60 | lowerbody vel 10f step5 | 120 | lowerbody vel 10f step1 | 120 |
| 8 | lowerbody pos 10f step5 | 120 | vr_3point_local_target | 9 | vr_3point_local_target | 9 |
| 9 | lowerbody vel 10f step5 | 120 | vr_3point_local_orn_target | 12 | vr_3point_local_orn_target | 12 |
| 10 | vr_3point_local_target | 9 | smpl_joints_10frame_step1 | 720 | smpl_joints_**4frame**_step1 | 288 |
| 11 | vr_3point_local_orn_target | 12 | smpl_anchor_orientation_**heading**_10frame_step1 | 60 | smpl_anchor_orientation_4frame_step1 | 24 |
| 12 | smpl_joints_10frame_step1 | 720 | motion_joint_positions_wrists_10frame_step1 | 60 | motion_joint_positions_wrists_4frame_step1 | 24 |
| 13 | smpl_anchor_orientation_10frame_step1 | 60 | | | | |
| 14 | motion_joint_positions_wrists_10frame_step1 | 60 | | | | |
| | **Total** | **1762** | | **1751** | | **1247** |
- Registry dims/gatherers (FACT): `CPP:1713` (encoder_mode_4 = 4: mode id + 3 zeros, `CPP:1691-1697`), `CPP:1733-1734` (root_z 1 / 10), `CPP:1732,1737-1738` (anchor orientation 6 / 60 step5 / 60 step1), `CPP:1741-1743` (heading variants, gatherer mode arg `1`), `CPP:1749-1758` (joint pos/vel 290, lowerbody 120), `CPP:1759,1761` (wrists 60 / 24), `CPP:1769,1772` (smpl_joints 720 / 288), `CPP:1779,1781,1783` (smpl anchor 60 / 24 / heading 60), `CPP:1791,1794` (vr 9 / 12).
- Step semantics (FACT): e.g. `motion_joint_positions_10frame_step5` → `GatherMotionJointPositionsMultiFrame(buf, offset, 10, 5)` vs `_step1` → `(…, 10, 1)` (`CPP:1749,1751`).
- Heading semantics (FACT): "Heading-only variants (mode=1): use robot yaw-only quaternion. Matches Python heading setting: quat_mul(quat_inv(get_heading_q(robot_pelvis_quat)), ref_quat)" (`CPP:1739-1741`); the orientation gatherer's mode 0 = full base quaternion (`motion_anchor_ori_b`), mode 1 = robot heading only (`CPP:612-613,667-674`). v1_1 header: "Root orientations are normalized by robot heading" (`CK/sonic_v1_1/observation_config.yaml:7`).
- Encoder modes and required observations (FACT): default g1 = mode 0 {encoder_mode_4, joint pos/vel 10f step5, anchor_orientation_10frame_step5}, teleop = mode 1 {encoder_mode_4, lowerbody pos/vel step5, vr pos, vr orn, motion_anchor_orientation}, smpl = mode 2 {encoder_mode_4, smpl_joints_10f_step1, smpl_anchor_orientation_10f_step1, wrists_10f_step1} (`CK/observation_config.yaml:57-80`). v1_1 same structure with `_heading` orientation names (`CK/sonic_v1_1/observation_config.yaml:51-74`). low_latency with step1 G1/teleop and 4-frame smpl names (`CK/low_latency/observation_config.yaml:69-92`). The root-z observations are listed in the default encoder_observations but are required by no mode (`CK/observation_config.yaml:35-38` vs `:57-80`).
- C++ behaviour (FACT): encoder input is built by concatenating enabled `encoder_observations` in yaml order with registry dims (`CPP:1871-1893`), only the **total** is validated against the ONNX input width (`CPP:1896-1901`); per step, observations not in the current mode's `required_observations` are left at zero (`CPP:2069-2117`, comment "Not required for this mode - leave as zero").
- low_latency yaml notes it "Requires the smpl_*_4frame_step1 entries in GetObservationRegistry() … + a rebuild to load" (`CK/low_latency/observation_config.yaml:9-10`); those entries exist at b042411 (`CPP:1761,1772,1781`, commented "low-latency SONIC").
- Lookahead per manifest (FACT): default and v1_1 SMPL `future_frames: 10, frame_interval_ms: 20, lookahead_ms: 200`; low_latency `future_frames: 4, frame_interval_ms: 20, lookahead_ms: 80` (`CK/config.json:12-16,30-34,49-53`). Model card: low_latency "G1 and teleoperation future-reference observations use step1"; v1_1 and default use step5 (`WBC/docs/source/model_card.md:11-13`).

**What the ONNX graphs actually slice (FACT, [CMD-2]; indices relative to `obs_dict[1:]`, i.e. add 1 for absolute)**
- Default: [3:583] → 10×58 (g1 joint pos+vel); [583:594] **not read**; [594:600] → 6 (teleop anchor ori); [600:660] → 10×6 (g1 anchor ori); [660:900] 240 (teleop lowerbody); [900:909] 9; [909:921] 12; [921:1641] 10×72 (smpl joints); [1641:1701] 10×6; [1701:1761] 10×6 (wrists). g1 input = concat(10×58, 10×6) = 640; teleop = 240+9+12+6 = 267; smpl = 10×84 = 840.
- sonic_v1_1: [3:583] 10×58; [583:643] → 10×6 (g1); [643:649] → 6 (teleop); [649:889] 240; [889:898]; [898:910]; [910:1630] 10×72; [1630:1690]; [1690:1750].
- low_latency: [3:583] 10×58; **[583:589] → 6 (teleop); [589:649] → 10×6 (g1)**; [649:889] 240; [889:898]; [898:910]; [910:1198] 4×72; [1198:1222] 4×6; [1222:1246] 4×6; smpl input 4×84 = 336.
- In all three, `obs_dict[1..3]` (the 3 zero-padding slots of encoder_mode_4) are never read ([CMD-3]).
- Training tokenizer order (FACT): low_latency `group_obs_names.tokenizer`: encoder_index, command_multi_future_nonflat, command_z_multi_future_nonflat, command_z, **motion_anchor_ori_b, motion_anchor_ori_b_mf_nonflat**, command_multi_future_lower_body, vr_3point…, smpl… (`CK/low_latency/model_config.yaml:286-298`), i.e. single-frame before multi-frame. sonic_v1_1: …, command_z, **motion_anchor_ori_heading_mf_nonflat, motion_anchor_ori_heading**, … (`CK/sonic_v1_1/model_config.yaml:286-298`), i.e. multi-frame before single-frame. Tokenizer dims include command_z (10+1) (`CK/low_latency/model_config.yaml:260-264`), which the v1_1/low_latency ONNX inputs drop and the default ONNX keeps but ignores.

**onnxruntime sensitivity test with the C++ layout (FACT, [CMD-3])**
- Default: mode 0 reacts to {joint pos, joint vel, anchor_10f}; mode 1 to {anchor single, lowerbody pos/vel, vr pos, vr orn}; mode 2 to {smpl joints, smpl anchor, wrists}. Exactly equal to the `required_observations` sets; root-z blocks affect nothing.
- sonic_v1_1: same exact match (heading variants).
- low_latency: mode 0 (g1) reacts to `motion_anchor_orientation_10frame_step1` **and** `motion_anchor_orientation` (the latter is not required for g1, so C++ zeroes it); mode 1 (teleop) reacts to `motion_anchor_orientation_10frame_step1` (not required for teleop → zeroed) and **does not react to the required `motion_anchor_orientation`**. Mode 2 (smpl) matches exactly.

### Inferences
- INFERENCE (high confidence, static + numeric evidence, not run on robot/sim): with the shipped `low_latency/observation_config.yaml`, the C++ writes the 60-d multi-frame anchor orientation at absolute [584:644] and the 6-d single-frame one at [644:650], but the ONNX reads the teleop 6-d input at [584:590] and the g1 10×6 input at [590:650]. Consequences: (a) **teleop mode (VR 3-point)** receives zeros as its anchor orientation (the slot holds the non-required, zeroed multi-frame block); (b) **g1 mode** (planner, joint-reference streaming, reference-motion playback) receives a 6-element-shifted orientation sequence (frames 1–9 of the 10-frame block plus a zeroed final frame). The total width (1247) still matches, so the C++ dimension check (`CPP:1896-1901`) does not catch it. SMPL mode — the low-latency variant's headline use — is unaffected. The fix would be to swap yaml lines 45-49 (put `motion_anchor_orientation` before `motion_anchor_orientation_10frame_step1`), which would reproduce the training order in `model_config.yaml:291-292`. The yaml comments "(body, multi-frame FIRST)" / "(single-frame SECOND)" (`CK/low_latency/observation_config.yaml:44,47`) contradict the training config.
- INFERENCE: the frame window spans: step5 → frames 0,5,…,45 (≈0–900 ms at 50 Hz); step1 → frames 0…9 (≈0–180 ms). So low_latency shortens the G1/teleop reference window ~5×, and the SMPL window from 10 to 4 frames (0–60 ms of future frames; manifest calls it 80 ms lookahead).
- INFERENCE: v1_1's heading normalization removes robot roll/pitch from the reference-orientation frame (only yaw is factored out), which is what "heading-stable teleoperation" refers to.

### Gaps
- I did not run the C++ deploy (sim) with low_latency to observe the behavioural effect of the ordering mismatch; it was only verified by graph analysis and onnxruntime. Whether NVIDIA's tested low_latency demos used only SMPL mode (which would hide the bug) is unknown.
- `WBC/gear_sonic_deploy/policy/release/observation_config_sonic_release.yaml` (1751 = encoder_index 4 + command_multi_future_nonflat 580 + motion_anchor_ori_b_mf_nonflat 60 + lower_body 240 + vr 9 + 12 + motion_anchor_ori_b 6 + smpl 720 + 60 + 60; aliases at `CPP:1714-1729`) describes a self-exported sonic_release ONNX ("python gear_sonic/eval_agent_trl.py +checkpoint=sonic_release/last.pt … +export_onnx_only=true", lines 4-6 of that file). Its layout matches none of the three HF ONNX encoders (default is 1762; v1_1 is 1751 but with heading obs and lowerbody at a different offset). I did not export an ONNX to verify that yaml.

## Q3. Deployment flags per variant; is the planner shared?

### Takeaway
Only sonic_v1_1 has a recommended extra flag pair: `--motor-kp-scale 4,10=1.5 --motor-kd-scale 4,10=1.5` (ankle-pitch gains ×1.5), documented as opt-in. low_latency needs only `--cp policy/low_latency/model --obs-config policy/low_latency/observation_config.yaml`. The default is what `deploy.sh` uses with no flags. One planner (`planner_sonic.onnx`) is shared by all variants.

### Cited Findings
- deploy.sh defaults (FACT): `CHECKPOINT_DEFAULT="policy/release/model"`, `OBS_CONFIG_DEFAULT="policy/release/observation_config.yaml"`, `PLANNER_DEFAULT="planner/target_vel/V2/planner_sonic.onnx"` (`WBC/gear_sonic_deploy/deploy.sh:240-242`); encoder/decoder paths are `${CHECKPOINT}_encoder.onnx` / `${CHECKPOINT}_decoder.onnx` (`deploy.sh:381-382`); passed as `g1_deploy_onnx_ref … --obs-config … --encoder-file … --planner-file …` (`deploy.sh:581-584`). Gain flags parsed at `deploy.sh:322-335`, forwarded at `deploy.sh:410-413`.
- `WBC/gear_sonic_deploy/policy/release/observation_config.yaml` is byte-identical to the HF top-level `CK/observation_config.yaml` (both md5 `b6b0607f…`), and the repo's `observation_config_low_latency.yaml` / `observation_config_sonic_v1_1.yaml` are byte-identical to the HF `low_latency/` and `sonic_v1_1/` yaml (md5 `d5e10b4e…`, `362ca247…`) ([CMD-5]).
- Downloader destinations (FACT): default → `policy/release/model_{encoder,decoder}.onnx` + `observation_config.yaml` (`WBC/download_from_hf.py:30-34`); low_latency → `policy/low_latency/…` (`:36-40`); v1_1 → `policy/sonic_v1_1/…` (`:42-49`); planner → `planner/target_vel/V2/planner_sonic.onnx` (`:51`). Variant flags `--low-latency` / `--sonic-v1-1` are mutually exclusive (`:97-117`); the downloader validates file lists against `config.json` (`:179-191, 339-353`). SMPL motion data download is skipped for `--low-latency` (`:342, 373-374`).
- Manifest (FACT): `"shared_files": {"planner": "planner_sonic.onnx", …}` (`CK/config.json:66-67`); each variant lists only encoder/decoder/yaml as `deployment_files` (`CK/config.json:17-21, 35-39, 54-58`). HF README: `--low-latency` = "low-latency policy + planner", `--sonic-v1-1` = "SONIC v1.1 policy + planner" (`CK/README.md:181-184`).
- Planner I/O (FACT, [CMD-1]): inputs `context_mujoco_qpos [1,4,36]`, `target_vel [1]`, `mode [1]` int64, `movement_direction [1,3]`, `facing_direction [1,3]`, `random_seed [1]`, `has_specific_target [1,1]`, `specific_target_positions [1,4,3]`, `specific_target_headings [1,4]`, `allowed_pred_num_tokens [1,11]`, `height [1]`; outputs `mujoco_qpos [1,64,36]`, `num_pred_frames [1]` int32. It outputs robot qpos (not SMPL, not tokens).
- v1_1 gain recommendation (FACT): `--motor-kp-scale 4,10=1.5 --motor-kd-scale 4,10=1.5`; "Motor indices 4 and 10 are the left and right ankle-pitch motors; the additional stiffness improves whole-body stability and observed wrist tracking. Scaling is opt-in." (`WBC/docs/source/model_card.md:101-112`); "They remain explicit so other model variants retain their original gains" (`WBC/docs/source/getting_started/download_models.md:225-228`); same in `CK/README.md:204-211`. VLA launcher equivalents `--deploy-motor-kp-scale 4,10=1.5 --deploy-motor-kd-scale 4,10=1.5` (`model_card.md:134-140`), forwarded by `WBC/gear_sonic/scripts/launch_inference.py:116-119, 329-332` (also `launch_data_collection.py:359-361`).
- low_latency usage (FACT): only `--cp policy/low_latency/model --obs-config policy/low_latency/observation_config.yaml` (+ `--input-type zmq_manager` for real) (`model_card.md:84-93`; `download_models.md:119-136`); no gain flags documented. Python launcher: `--deploy-checkpoint` / `--deploy-obs-config` (`model_card.md:126-132`; `launch_inference.py:101-107, 319-324`).
- Gain-scale mechanics (FACT): after computing `q_target = default_angles + action × g1_action_scale` with the default `kps[i]`/`kds[i]`, `apply_motor_gain_scales` multiplies `kp[index]`/`kd[index]` for each configured hardware index (`CPP:3155-3165`; `WBC/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/motor_gain_scaling.cpp:172-182`). Default ankle-pitch gains are `2.0 * STIFFNESS_5020` / `2.0 * DAMPING_5020` (`WBC/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp:148,154,182,188`).
- `run_vla_inference.py` (777 lines) has no variant-specific code (FACT: `grep -n "low_latency\|sonic_v1_1"` returns nothing); it publishes a 64-d `token_state` (latent protocol v4) (`WBC/gear_sonic/scripts/run_vla_inference.py:7, 138-166`).

### Inferences
- INFERENCE: with v1_1 + gain scaling, ankle-pitch Kp/Kd become 3.0×STIFFNESS_5020 / 3.0×DAMPING_5020 (2.0 × 1.5). Running default or low_latency with these flags would apply gains those policies were not tuned/tested with.
- INFERENCE: because the planner emits G1 qpos that the deploy consumes through encoder mode 0 (see Q5), the shared planner's outputs go through each variant's own g1 sub-encoder — for low_latency that path is affected by the anchor-orientation ordering issue in Q2, and its step1 G1 window looks at only the first 10 of the planner's 64 predicted frames.

### Gaps
- No documented hardware gain tuning for low_latency; no published reason why the default/low_latency do not need it.
- I did not verify which `deploy.sh` flag maps to `--planner-fp16` / `--policy-fp16` (`CPP:45-46`) or whether any variant recommends FP16.

## Q4. Are the variants' encoders/decoders interchangeable? What breaks if mixed?

### Takeaway
Decoders are shape-interchangeable (all `[1,994]→[1,29]`) and the encoders all output `[1,64]`, so mixing an encoder of one variant with a decoder of another will load and run without any C++ error — but it is semantically wrong because each pair (and its token space) was trained jointly and no weights are shared. Encoders are not interchangeable with each other's yaml (input widths 1762/1751/1247; the C++ throws on mismatch). NVIDIA states each encoder/decoder must be used with its matching observation configuration.

### Cited Findings
- I/O shapes: encoders `[1,1762]`, `[1,1751]`, `[1,1247]` → `[1,64]`; decoders all `[1,994]` → `[1,29]` ([CMD-1]).
- 0 identical weight tensors between any two variants (encoder and decoder) ([CMD-4]).
- C++ validates encoder total dim (`CPP:1896-1901` "Encoder observation dimension mismatch") and policy dim (`CPP:1953-1960` "Policy observation dimension mismatch"), and token dim (`CPP:1676-1680`); `deploy.sh` derives both encoder and decoder from one `--cp` prefix (`deploy.sh:381-382`), so mixing requires manual renaming/symlinks.
- "Each ONNX encoder and decoder must be used with its matching observation configuration." (`WBC/docs/source/model_card.md:151-152`); low_latency: "Use its encoder, decoder, and observation config together." (`model_card.md:12`).
- Mixing yaml across encoders: default yaml (1762) with v1_1 encoder (1751) or v1_1 yaml with default encoder → dimension-mismatch exception at init (FACT from widths + `CPP:1896-1901`). v1_1 yaml (1751) with a self-exported sonic_release ONNX (1751) would pass the width check but supply heading-normalized orientations at different offsets (FACT from Q2 layouts).
- External-token path: when a ZMQ/ROS input supplies a token (VLA latent protocol), the C++ copies it into `token_state_data_` and sets `is_using_encoder_ = false` (`CPP:3034-3064`; `GatherTokenState` uses the external token when not using the encoder, `CPP:1648-1653`), i.e. the encoder is bypassed and only the decoder interprets the token.

### Inferences
- INFERENCE: what breaks if mixed: (1) default/v1_1 encoder → v1_1/low_latency decoder (or vice versa): runs, but tokens from one FSQ space are decoded by a decoder trained on another → unpredictable/unsafe actions; (2) a VLA (GR00T latent-token head) trained against one variant's tokens and executed with a different variant's decoder has the same problem, since the encoder is bypassed in that path; model_card explicitly ties v1_1 to "SONIC-backed VLA policies trained against this controller" (`model_card.md:13, 61-63`); (3) the `LATENT_INITIAL_MOTION_TOKEN` idle token used by `run_vla_inference.py` (`:430-454`) is variant-specific in meaning — if it was computed from one variant, it is not guaranteed to be an idle pose for another (not verified).
- INFERENCE: The only thing that is genuinely shared and safe to reuse across variants is the 994-d decoder observation layout (same yaml block) and the planner.

### Gaps
- I did not check which variant `LATENT_INITIAL_MOTION_TOKEN` in `run_vla_inference.py` was derived from.
- No empirical test of cross-variant decoding was run (would require sim).

## Q5. Does the C++ deploy code treat the variants differently?

### Takeaway
No variant-specific branching exists in the C++: behaviour is entirely driven by the yaml (observation names → registry gatherers) and the ONNX shapes. Variant support was added purely as registry entries (heading-normalized gatherers for v1_1, 4-frame SMPL/wrist gatherers for low_latency, name aliases for sonic_release-exported ONNX). Gain scaling is a generic CLI feature, not tied to v1_1.

### Cited Findings
- `grep -rln "low_latency|sonic_v1_1|motor_kp_scale|motor-kp-scale"` over `WBC` (excluding thirdparty) hits only docs, downloader, launchers, `deploy.sh` and one C++ file; inside `CPP` the only variant mentions are comments on registry entries "low-latency SONIC (smpl 4-frame)" (`CPP:1761,1772,1781`) and the sonic_release alias block (`CPP:1714-1729`). (FACT, grep)
- Registry is the "single source of truth mapping observation names … to their dimension and gatherer lambda" for both policy and encoder (`CPP:1699-1706`).
- Heading-normalized gatherers (used by v1_1) call the shared orientation gatherer with mode `1` (`CPP:1741-1743, 1783`); refheading (mode 2) variants also exist but are unused by any released yaml (`CPP:1744-1748, 1785-1787`).
- Encoder mode is chosen by the input source, not the variant: ZMQ protocol 1 → mode 0 (g1), protocols 2/3 → mode 2 (smpl) (`WBC/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp:1706-1711`); VR 3-point enabled → mode 1, disabled → mode 0 (`…/include/input_interface/zmq_manager.hpp:652-664`); planner start forces mode 1→0 (`zmq_manager.hpp:466-473`). If the intended mode's required observations fail to gather, the C++ falls back to other modes (`CPP:2039-2060, 2119-2124`).
- Encoder engine is TensorRT, initialized with the yaml `use_fp16` (`CPP:2381`; header `CPP:157-159`).

### Inferences
- INFERENCE: because the C++ only checks total widths, a yaml whose block order differs from the ONNX's internal slicing (the low_latency case in Q2) is silently accepted — the deploy relies on the yaml being authored correctly for each export.
- INFERENCE: practical per-variant differences on the robot therefore reduce to: (a) which yaml/ONNX prefix is passed, (b) optional gain flags (v1_1), (c) the larger decoder in v1_1/low_latency (≈3.7× decoder params; latency impact not measured), (d) which encoder sub-network the active input source routes to.

### Gaps
- TensorRT engine caching/naming per variant (whether engines for different variants could collide in a cache directory) was not checked.
