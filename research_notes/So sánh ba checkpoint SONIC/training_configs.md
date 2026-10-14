# Training configurations and data of the three GEAR-SONIC checkpoints (sonic_release, sonic_v1_1, low_latency)

Scope: the checkpoint files on disk (downloaded 22/09/2026) plus NVIDIA code in `third_party/GR00T-WholeBodyControl` at commit b042411 (merge dated 2026-09-23). The notes do not use this project's own code.

Path abbreviations used in citations (all relative to `/home/thanhhung27/Code/GEAR-SONIC-GR00T-N1.7/`):
- `REL` = `checkpoints/GEAR-SONIC/sonic_release/config.yaml` (912 lines; this checkpoint ships **no** `model_config.yaml` or ONNX in its folder)
- `V11` = `checkpoints/GEAR-SONIC/sonic_v1_1/config.yaml` (946 lines); `V11M` = `.../sonic_v1_1/model_config.yaml`; `V11O` = `.../sonic_v1_1/observation_config.yaml`
- `LL` = `checkpoints/GEAR-SONIC/low_latency/config.yaml` (935 lines); `LLM` = `.../low_latency/model_config.yaml`; `LLO` = `.../low_latency/observation_config.yaml`
- `WBC` = `third_party/GR00T-WholeBodyControl/` (commit b042411)
- `PT:<name>` = values read from `checkpoints/GEAR-SONIC/<name>/last.pt` with torch 2.14 (`policy_state_dict`, `value_state_dict`, `state`, `optimizer_state_dict`, `env_state_dict`). These were read with a stub unpickler, so there is no line number.

Method: I ran a full `diff` on each pair of config.yaml files and on V11M vs LLM. Every key that does not appear in the diff output is byte-identical across the files. That is the evidence behind each "identical in all three" claim below.

## Q1. Systematic diff: which settings differ and which are shared

### Takeaway
Differences are few and targeted. There are four groups: (a) orientation features, body-frame `*_ori_b` (REL, LL) vs robot-heading-normalized `*_ori_heading` (V11); (b) reference lookahead, where LL uses G1 spacing 0.02 s and 4 SMPL frames while REL and V11 use 0.1 s and 10 frames; (c) network and critic size, where V11 and LL have a 4096-4096-… 8-layer g1_dyn decoder and critic and REL has a 6-layer 2048-… stack; (d) data and regularization: REL uses BONES-SEED paths, V11 and LL use internal "pyroki" training mixes, V11 and LL add `tracking_vr_2wrists_local_ori` (w=0.4) and `energy_consumption` (w=-1e-4), widen the mass-scale DR from 0.8-1.5 to 0.8-2.5, and only V11 turns on `randomize_wrist_poses`. Everything else is identical: PPO hyperparameters, FSQ/token settings, encoder sample probabilities, aux losses, most rewards, terminations, other DR, adaptive sampling, and upper-body augmentation prefixes.

### Cited Findings

**A. Run identity / `checkpoint:` / experiment_dir / logging (all differ)**
- REL: `checkpoint: .../all_modes/base3_noz_g1_wrist_5pt_consistency-20260118_163933/last.pt` — [REL:8]; `experiment_dir: .../all_modes/sonic_sm_3pt_resume-20260204_110722` — [REL:13]; `wandb_group: zen_0122`, `wandb_id: null` — [REL:283-284]; `exp_var: resume` — [REL:908]. No `resume:` key — [diff REL vs V11: `912a938,946`].
- V11: `checkpoint:` and `experiment_dir:` both `.../all_modes/sonic_release_3pt_heading_wrist_81-20260415_051436` — [V11:8, V11:13]; `wandb_group: zen_heading_0223`, `wandb_id: jbpdpb7r` — [V11:287-288]; `exp_var: wrist_81` — [V11:933]; `resume: true` — [V11:946]; `shared_dir: /mnt/amlfs-04/shared` — [V11:945].
- LL: `checkpoint:` and `experiment_dir:` both `.../all_modes/sonic_release_3pt_ultralatency_not_usd-20260323_141755` — [LL:8, LL:13]; `wandb_group: zen_0306`, `wandb_id: 7wypxf8x` — [LL:287-288]; `exp_var: not_usd` — [LL:930]; `resume: true` — [LL:935].
- The run directory saved inside the checkpoints agrees: `args.output_dir` = `sonic_sm_3pt_resume-20260204_110722` (REL), `sonic_release_3pt_heading_wrist_81-20260415_051436` (V11), `sonic_release_3pt_ultralatency_not_usd-20260323_141755` (LL) — [PT:sonic_release / sonic_v1_1 / low_latency, `args`].
- model_config export dirs: V11 `models/0430/...` — [V11M:238-246]; LL `models/0527/...` — [LLM diff vs V11M at line 240: `'0527'`].
- Module import paths: REL uses the older `groot.rl.agents.modules.modules.BaseModule` and V11/LL use `groot.rl.trl.modules.base_module.BaseModule` — [REL:52 vs V11:52; diff shows this for every BaseModule target].
- Eval datasets appear only in V11: `pyroki_v0.8.1_splits/v0.8.1_test_content`, `v0.8.1_test_repetition`; eval_modes g1/smpl/teleop — [V11:938-944].

**B. Training data paths (all differ)**
- REL: `motion_file: data/motion_lib_bones_seed/robot_filtered/`, `smpl_motion_file: data/motion_lib_bones_seed/1215_bones_seed_filtered/` — [REL:297-298].
- V11: `motion_file: [${shared_dir}/wbc_data/training_mixes/train_pyroki_v081_0414_26/]`, `smpl_motion_file: [${shared_dir}/wbc_data/smpl_retarget/1215/]` — [V11:301-304].
- LL: `motion_file: /mnt/amlfs-01/shared/wbc_data/training_mixes/train_pyroki_0213_26/`, `smpl_motion_file: /mnt/amlfs-01/shared/wbc_data/smpl_retarget/1215/` — [LL:301-302].
- The public NVIDIA recipes at b042411 point both `sonic_release` and `sonic_v1_1` at BONES-SEED: `motion_file: data/motion_lib_bones_seed/robot_filtered`, `smpl_motion_file: data/bones_seed_smpl` — [WBC/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml:72-73; .../sonic_v1_1.yaml:71-72]. So the V11 checkpoint's own config (internal pyroki mix) is **not** what the public V11 recipe uses.
- Motion-lib size at save time: the adaptive-sampling buffers have length 2,598,143 (REL), 2,579,711 (V11), 2,514,976 (LL) bins — [PT:*, `env_state_dict.motion_lib.adp_samp_num_episodes`]. Bins are `bin_size` frames per clip: `bin_size: 50` — [V11:315]. Each clip is split into 50-frame bins with a shorter last bin — [WBC/gear_sonic/utils/motion_lib/motion_lib_base.py:2263, 2324, 2348-2350].

**C. Encoder/decoder observation names: `*_ori_b` vs `*_ori_heading`**
- G1 encoder inputs: REL/LL `[command_multi_future_nonflat, motion_anchor_ori_b_mf_nonflat]` — [REL:77-78; LL:79-80]; V11 `[..., motion_anchor_ori_heading_mf_nonflat]` — [V11:79-80].
- Teleop encoder 4th input: REL/LL `motion_anchor_ori_b` — [REL:99; LL:101]; V11 `motion_anchor_ori_heading` — [V11:101].
- SMPL encoder 2nd input: REL/LL `smpl_root_ori_b_multi_future` — [REL:118; LL:120]; V11 `smpl_root_ori_heading_multi_future` — [V11:120].
- g1_kin decoder outputs follow the same pattern: REL/LL `motion_anchor_ori_b_mf_nonflat` — [REL:165; LL:169]; V11 `motion_anchor_ori_heading_mf_nonflat` — [V11:169].
- The matching tokenizer observation terms: V11 `func: ...:motion_anchor_ori_heading_mf` / `motion_anchor_ori_heading` / `smpl_root_ori_heading_mf` — [V11:552-561, 577-585, 601-610]; REL `motion_anchor_ori_b_mf` / `motion_anchor_ori_b` / `smpl_root_ori_b_mf` — [REL:545-547, 570-572, 594-596].
- The critic keeps body-frame `motion_anchor_ori_b` in all three — [V11:497-501; REL:490-492; LL:494-496].
- The public recipes encode the same switch: `sonic_release.yaml` uses tokenizer `unitoken_all_noz` and `*_ori_b` inputs; `sonic_v1_1.yaml` uses `unitoken_all_noz_heading` and `*_ori_heading` inputs — [WBC/.../all_modes/sonic_release.yaml:13, 89-107; .../sonic_v1_1.yaml:13, 88-106].

**D. Future-reference frames (lookahead)**
- G1/teleop reference spacing: REL `dt_future_ref_frames: 0.1`, `num_future_frames: 10` — [REL:331-332]; V11 `0.1`, `10` — [V11:337-338]; LL `0.02`, `10` — [LL:335-336].
- SMPL reference: REL `smpl_num_future_frames: 10`, `smpl_dt_future_ref_frames: 0.02` — [REL:421-422]; V11 `10`, `0.02` — [V11:427-428]; LL `4`, `0.02` — [LL:425-426].
- Code semantics: `frame_skips = dt_future_ref_frames // (1/target_fps)` and `future_time_steps_init = arange(num_future_frames) * frame_skips` — [WBC/gear_sonic/envs/manager_env/mdp/commands.py:346, 360-367]; SMPL is analogous — [commands.py:369-386]. Absolute index = `future_time_steps_init + time_steps + motion_start_time_steps`, clipped to the clip end — [commands.py:3293-3302, 3313-3320]. So frame 0 is the **current** reference frame. `target_fps: 50` — [V11:310].
- Resulting dimensions: LL tokenizer obs `tokenizer: 1257` — [LLM:303, 312]; V11 `tokenizer: 1761` — [V11M:303, 312]; V11 per-term dims, e.g. `smpl_joints_multi_future_local_nonflat: [10, 72]`, `command_multi_future_nonflat: [10, 58]` — [V11M:253-284]. LL SMPL temporal dims are 4 — [LLM diff vs V11M lines 119, 268, 276, 279].
- Deploy-side observation configs agree. V11O: "G1/teleop references use 10 frames at step5. SMPL and wrist references use 10 frames at step1. Root orientations are normalized by robot heading", encoder input 1751 — [V11O:1-7 (from diff)]. LLO: "smpl 4-frame lookahead; g1/teleop lookahead at step1, body-frame orientation", "Encoder input dimension: 1247" — [LLO:1-6]; it uses `motion_joint_positions_10frame_step1`, `smpl_joints_4frame_step1`, etc. — [LLO:40-67].
- NVIDIA model card: Default = SMPL "10 future frames at 20 ms spacing, approximately 200 ms", G1/teleop `step5`; Low-latency = "4 future frames at 20 ms spacing, approximately 80 ms", G1/teleop `step1`; v1.1 = 10×20 ms ≈200 ms, `step5`, "this is not the low-latency model" — [WBC/README.md:91-94]. It also says these lookahead values "are **not** measurements of total end-to-end teleoperation latency" — [WBC/README.md:98].

**E. Network sizes (hidden_dims)**
- Encoders g1/teleop/smpl: `[2048, 1024, 512, 512]`, SiLU, identical in all three — [V11:90-95, 111-116, 132-137; REL:88/109/129 region; LL:90/111/131 region; no diff].
- g1_kin decoder `[2048, 1024, 512, 512]`, identical — [V11:181-185].
- g1_dyn decoder (action head): V11/LL `[4096, 4096, 2048, 2048, 1024, 1024, 512, 512]` — [V11:154-162; LL:154-162]; REL `[2048, 2048, 1024, 1024, 512, 512]` (the diff `152a155,156` shows REL lacks the two 4096 layers) — [REL:152-158].
- Critic: V11/LL `[4096, 4096, 2048, 2048, 1024, 1024, 512, 512]` — [V11:62-70]; REL `[2048, 2048, 1024, 1024, 512, 512]` (diff `62a63,64`) — [REL:62-68].
- The public `sonic_v1_1.yaml` recipe explicitly overrides g1_dyn and critic to the 8-layer 4096 stack — [WBC/.../all_modes/sonic_v1_1.yaml:107-117]. The `sonic_release.yaml` recipe has no such override — [WBC/.../sonic_release.yaml:105-107].
- First-layer shapes confirmed from weights: g1_dyn `module.0.weight` (2048, 994) in REL vs (4096, 994) in V11/LL; critic (2048, 1645) vs (4096, 1645); SMPL encoder (2048, 840) in REL/V11 vs (2048, 336) in LL — [PT:*].

**F. Tokenizer / FSQ / aux losses / encoder sampling (identical in all three)**
- `quantizer: vector_quantize_pytorch.FSQ`, `num_fsq_levels: 32`, `fsq_level_list: 32`, `max_num_tokens: 2` — [V11:75, 191-193; REL:187-189; LL:191-193]. In code, token_dim = `num_fsq_levels`, an int `fsq_level_list` is broadcast to all dims, and `token_total_dim = token_dim * max_num_tokens` — [WBC/gear_sonic/trl/modules/universal_token_modules.py:64-66, 231-249]. So each token is 32-d with 32 levels per dim, 2 tokens, 64-d flattened, matching "token: 64" — [V11O/LLO header; LLO:33].
- `encoder_sample_probs: {g1: 1.0, teleop: 1.0, smpl: 1.0}` (normalized to 1/3 each in code) — [V11:294-297; WBC/.../commands.py:402-411].
- aux losses: g1_recon (MSE, coef 0.01), g1_smpl_latent 1.0, g1_teleop_latent 1.0, teleop_smpl_latent 1.0 (`detach_teleop_target: true`), reencoded_smpl_g1_latent 1.0; `reencode_smpl_g1_recon: true` — [V11:194-217].
- Other command augmentations, all identical: `cat_upper_body_poses: true`, `cat_upper_body_poses_prob: 0.5`, `freeze_frame_aug: true`, `teleop_sample_prob_when_smpl: 0.5` — [V11:429-432].

**G. Wrist-pose augmentation**
- `randomize_wrist_poses: true` appears only in V11 — [V11:433]. It is absent from REL (diff `426a433`) and LL (diff `433d430`). The code default is `False` — [WBC/gear_sonic/envs/manager_env/mdp/commands.py:4223].
- Neither config sets prob/std, so code defaults apply: `randomize_wrist_prob: 0.3`, `randomize_wrist_std: 0.1  # radians (~5.7 degrees)` — [commands.py:4224-4225; passed to the motion lib at commands.py:225-227; motion_lib_base.py:358-362].
- Mechanism: during training only (`not is_evaluation`), with probability 0.3 per loaded clip, Gaussian noise (σ=0.1 rad) is added to the 6 wrist DOFs (MuJoCo DOF idx 19,20,21,26,27,28 = L/R wrist roll/pitch/yaw). This is applied to `pose_aa` **before** forward kinematics, so the reference wrist poses and positions themselves change — [motion_lib_base.py:1863-1883].

**H. Upper-body augmentation prefixes and adaptive sampling (identical)**
- `upper_body_augment_prefixes: ['2025', sonic_mixed_walking_running, sonic_squating_different_height_switching, sonic_running, sonic_walking, sonic_squating_different_height_hold, sonic_walking_other_style, balance]` — [V11:323-331]. The code default is `["2025", "walking_2025", "running_2025", "slow_walk_2025"]`, commented "generated by the kinematic planner" — [motion_lib_base.py:352-356]. With prob 0.5 it splices upper-body joints from a random non-navigation clip into clips whose key matches a prefix — [motion_lib_base.py:1813-1861].
- adaptive_sampling: `enable: true, bin_size: 50, sequence_length_agnostic: true, init_num_failures: 1, uniform_sampling_rate: 0.1, pre_failure_sample_window: 200, use_failure_rate_decay: false, decay_gamma: 0.8, adp_samp_failure_rate_max_over_mean: 200` — [V11:313-322].

**I. Rewards**
- Shared terms and weights (all three): tracking_anchor_pos 0.5 (std 0.3), tracking_anchor_ori 0.5 (std 0.4), tracking_relative_body_pos 1.0 (0.3), tracking_relative_body_ori 1.0 (0.4), tracking_body_linvel 1.0 (1.0), tracking_body_angvel 1.0 (3.14), action_rate_l2 -0.1, joint_limit -10.0, undesired_contacts -0.1 (threshold 1.0; excludes ankles/wrists/elbows), anti_shake_ang_vel -0.005 (threshold 1.5; wrists+head), tracking_vr_5point_local 2.0 (std 0.1), feet_acc -2.5e-6 (ankle joints) — [V11:628-732].
- Only in V11 and LL: `tracking_vr_2wrists_local_ori` weight 0.4, std 0.1, on left/right_wrist_yaw_link — [V11:713-722; LL:710-719]. `energy_consumption` weight -0.0001 — [V11:733-739; LL:730-736]. REL has neither (diff `705a713`, `715a733`).
- Code: `tracking_local_vr_2wrists_ori_error` = exp(-mean(angle²)/std²) on wrist orientation expressed in the anchor (root) frame for both reference and robot — [WBC/gear_sonic/envs/manager_env/mdp/rewards.py:212-248]. `energy_consumption` = Σ|τ·q̇| (mechanical power) — [rewards.py:251-261]. In the public recipes, V11 uses reward preset `tracking/local_feet_acc_energy_5pt` and REL uses `tracking/base_5point_local_feet_acc` — [WBC/.../sonic_v1_1.yaml:18; .../sonic_release.yaml:18]. That preset includes `terms/tracking_vr_2wrists_local_ori` and `terms/energy_consumption` — [WBC/gear_sonic/config/manager_env/rewards/tracking/local_feet_acc_energy_5pt.yaml:15, 17].

**J. Terminations (identical)**
- anchor_pos (`exceeded_anchor_height`, threshold 0.15, adaptive, down_threshold 0.75, root_height_threshold 0.5); anchor_ori_full (threshold 0.2); ee_body_pos (0.15, adaptive; ankles+wrists); time_out; foot_pos_xyz (0.2; ankles) — [V11:741-789].

**K. Domain randomization**
- Identical: friction static 0.3-1.6, dynamic 0.3-1.2, restitution 0-0.5, 64 buckets; joint default pos ±0.01; torso CoM x ±0.025, y/z ±0.05; push_robot every 4-6 s with vel x/y ±0.5, z ±0.2, roll/pitch ±0.52, yaw ±0.78 — [V11:790-869]. Observation noise is also identical, e.g. gravity ±0.05, ang vel ±0.2, joint pos ±0.01, joint vel ±0.5 — [V11:445-478].
- **Differs:** `randomize_rigid_body_mass` on `.*wrist_yaw.*|torso_link`, scale range REL `[0.8, 1.5]` — [REL:855-857] vs V11/LL `[0.8, 2.5]` — [V11:879-882; LL:876-878].

**L. PPO / env hyperparameters (identical in all three)**
- `num_envs: 4096` — [V11:6]; `world_size: 128` — [V11:259]; `num_steps_per_env: 24`, `num_learning_epochs: 5`, `num_mini_batches: 4`, `clip_param 0.2`, `gamma 0.99`, `lam 0.95`, `value_loss_coef 1.0`, `entropy_coef 0.01`, `actor_learning_rate 2e-5`, `critic_learning_rate 1e-3`, `max_grad_norm 0.1`, `schedule: adaptive`, `desired_kl: 0.01`, `adaptive_lr_min 1e-5`, `adaptive_lr_max 2e-4`, `init_noise_std 0.05`, `num_learning_iterations: 100000`, std clamp [0.001, 0.5] — [V11:222-262; REL:225, 230, 243, 255; LL:229, 234, 247, 259].
- Env: `decimation: 4`, `sim_dt: 0.005` (so 50 Hz policy), `episode_length_s: 10.0`, `terrain_type: trimesh`, robot `g1_model_12_dex`, `action_clip_value: 20.0`, actor/critic history length 10 — [V11:891-924, 934-937].
- `needs_policy_atm: true` appears only in V11/LL — [V11:916; LL:913]. The env wrapper defaults it to True when absent — [WBC/gear_sonic/envs/wrapper/manager_env_wrapper.py:200].

**M. Checkpoint training-state (from last.pt)**
- `global_step`: REL 41,550 / max_steps 100,000; V11 100,000/100,000; LL 100,000/100,000 — [PT:*, `state`].
- `tot_timesteps`: REL 522,819,993,600; V11 and LL 1,258,291,200,000 — [PT:*]. 1.2583e12 = 100,000 × 24 × 4096 × 128, and 5.228e11 = 41,550 × 24 × 4096 × 128, so the counter matches num_steps_per_env × num_envs × world_size.
- `tot_time` (wall-clock seconds): REL 522,078 s (≈6.0 days); V11 1,037,953 s (≈12.0 days); LL 2,033,509 s (≈23.5 days) — [PT:*].
- Optimizer LR at save is 2e-5 for both param groups in all three. The lr_scheduler is constant (`lr_scheduler_type: constant`) — [PT:* `optimizer_state_dict`; V11:44]. The pickled `args.learning_rate` is 1e-5 in all three and `seed: 42` — [PT:* `args`], while the config has `seed: 0` — [V11:3].

### Inferences
- The NVIDIA model card describes V11 as "heading-normalized + wrist-pose augmentation". The configs show it also includes the bigger decoder/critic, two new reward terms, wider mass DR, and a different (internal pyroki v0.8.1) dataset. Several of these extras (decoder 4096, 2-wrist-ori reward, energy, mass 2.5) are shared with LL, so they look like a "second-generation recipe" common to both newer checkpoints and not V11-specific.
- LL is essentially "REL observation semantics (`*_ori_b`) + second-generation recipe + shorter lookahead", without wrist augmentation.
- The REL config's data paths (`data/motion_lib_bones_seed/...`) are local relative paths, unlike the internal `/mnt/amlfs-*` paths in V11/LL. Possibly REL's config was sanitized for public release, or REL really was trained on a BONES-SEED-filtered set. This cannot be decided from the files.
- Using the 50-frame bin counts as an upper bound on frames: ≤129.9 M frames (≈≤722 h at 50 fps) for REL, ≤129.0 M (≈≤717 h) for V11, ≤125.7 M (≈≤699 h) for LL. The three motion libraries are therefore of very similar scale (within ~3%) despite different source paths. This is an approximation, because the last bin of each clip is partial.

### Gaps
- No `low_latency` experiment recipe exists in the public repo: `all_modes/` holds only sonic_bones_seed, sonic_h2, sonic_release, sonic_v1_1 — [WBC/gear_sonic/config/exp/manager/universal_token/all_modes/]. LL can only be reconstructed from its saved config.
- The contents of `train_pyroki_0213_26`, `train_pyroki_v081_0414_26`, `smpl_retarget/1215` and `robot_filtered` are not public. Clip counts per dataset are unknown. "pyroki" is not documented in the repo docs I read.
- Why `args.learning_rate` (1e-5) ≠ config/optimizer LR (2e-5) is not resolved.
- REL ships no `model_config.yaml`, so its exported obs dims can only be inferred, from the weights: g1 enc 640, teleop 267, smpl 840, decoder 994.

## Q2. Physical meaning of the differences

### Takeaway
REL and V11 see G1/teleop targets spanning 0.9 s ahead (10 frames × 0.1 s) and SMPL targets spanning 0.18 s. LL sees G1/teleop 0.18 s ahead (10 × 0.02 s) and SMPL only 0.06 s ahead (4 × 0.02 s). NVIDIA's README quotes "≈200 ms" and "≈80 ms", which count N×dt and not (N-1)×dt. V11's heading normalization removes the robot's own pitch/roll from the target-orientation feature. Its wrist augmentation and wrist-orientation reward specifically target expressive wrist teleoperation.

### Cited Findings
- Frame k is at time `t + k·frame_skips/50` for k = 0..N-1, and frame 0 = current frame — [WBC/.../commands.py:346, 360-367, 3293-3302]. So the horizon beyond "now" = (N-1)·dt: REL/V11 G1 = 9 × 0.1 = 0.9 s; LL G1 = 9 × 0.02 = 0.18 s; REL/V11 SMPL = 9 × 0.02 = 0.18 s; LL SMPL = 3 × 0.02 = 0.06 s. Config values: [REL:331-332, 421-422; V11:337-338, 427-428; LL:335-336, 425-426].
- NVIDIA's own figures: "10 future frames at 20 ms spacing, approximately 200 ms" and "4 future frames at 20 ms spacing, approximately 80 ms" — [WBC/README.md:92-94]. "step5"/"step1" in deploy obs names = 5 or 1 control steps of 20 ms between frames — [V11O / LLO:40-67; WBC/README.md:92-94].
- `motion_anchor_ori_b_mf`: reference-vs-robot root rotation difference "in body-local frame. Uses the robot's full orientation (including pitch/roll)" — [WBC/gear_sonic/envs/manager_env/mdp/observations.py:1022-1043]. `motion_anchor_ori_b` = `subtract_frame_transforms(robot_anchor, ref_anchor)` as a 6D rotation — [observations.py:943-970].
- `motion_anchor_ori_heading_mf`: "normalized by robot heading (yaw) only ... preserves the reference motion's pitch/roll relative to gravity while removing the heading difference" — [observations.py:1153-1173]. Single-frame `motion_anchor_ori_heading` is likewise "canonicalized by robot heading (yaw)" — [observations.py:992-1008].
- The README describes V11 as "Intended for heading-stable whole-body teleoperation and SONIC-backed VLA policies" — [WBC/README.md:94]. The LL README entry says "Intended for more responsive whole-body teleoperation and VLA execution" — [WBC/README.md:93].
- Wrist augmentation: σ = 0.1 rad (≈5.7°) Gaussian on wrist roll/pitch/yaw, applied to 30% of training clip loads before FK — [WBC/.../commands.py:4223-4225; motion_lib_base.py:1863-1871]. The 2-wrist local orientation reward (w=0.4, std 0.1 rad) rewards matching wrist orientation relative to the pelvis — [rewards.py:212-248; V11:713-722].
- Energy penalty Σ|τ·q̇| with weight -1e-4 — [rewards.py:251-261; V11:733-739].
- Mass DR on wrists and torso widened to ×0.8-2.5 — [V11:870-882] vs ×0.8-1.5 — [REL:855-857].
- Encoder input width changes: SMPL encoder input shrinks from 840 (10×84) to 336 (4×84) in LL — [PT:low_latency weights; LLM:303]. The G1 encoder input is still 640 (10×64) because LL keeps 10 frames and only shrinks the spacing — [PT:*; LL:336].

### Inferences
- Body-frame (`_b`) target orientation entangles the robot's current tilt with the command. If the robot pitches or rolls, the same world-frame target looks different to the encoder. Heading-only normalization makes the gravity-relative target stable, which plausibly explains NVIDIA's "heading-stable" wording. This is my interpretation; the docstrings are fact.
- The 0.9 s → 0.18 s G1 lookahead in LL means the encoder gets much less information about upcoming motion. That should make it react faster to sudden teleop changes, because less of the token is conditioned on future reference that may be stale. Agile or anticipatory motions may suffer. I found no NVIDIA ablation numbers in the files to confirm the trade-off.
- Wrist noise (≈±6°) together with the wrist-orientation reward and the larger mass range (heavier hands/torso up to 2.5×) are consistent with a goal of robust, expressive wrist and hand teleoperation and with carrying payloads (e.g. VLA manipulation). That is inference from the combination of settings.
- The energy penalty is tiny (1e-4 × watts). Its practical effect is likely mild smoothing or less jitter, but it is not quantified here.

### Gaps
- No NVIDIA-published quantitative comparison (tracking error, latency) between the three checkpoints was found in the repo files examined.
- Whether `step5` deploy frames are exactly 0.1 s (5 × 20 ms) depends on deploy code, which is not examined here. The training side is exact (`frame_skips = 0.1 // 0.02 = 5`; floating-point floor division could yield 4.0 in principle, and the warning only triggers when dt×fps is non-integer) — [commands.py:346-358]. This edge case was not verified at runtime.

## Q3. Training lineage (fact vs inference)

### Takeaway
FACT: REL's run `sonic_sm_3pt_resume-20260204_110722` lists a different parent run (`base3_noz_g1_wrist_5pt_consistency-20260118_163933`) in `checkpoint:` and was saved at global_step 41,550. V11 (`sonic_release_3pt_heading_wrist_81-20260415_051436`) and LL (`sonic_release_3pt_ultralatency_not_usd-20260323_141755`) each list **their own** run directory as `checkpoint:` with `resume: true`, and both reached 100,000 iterations. INFERENCE: REL is a fine-tune or resume of a January 2026 base run. V11 and LL are separate runs from March and April 2026 that were self-resumed (preemption restarts), and their initial weights cannot be determined from configs.

### Cited Findings
- REL: parent `base3_noz_g1_wrist_5pt_consistency-20260118_163933` — [REL:8]; own dir `sonic_sm_3pt_resume-20260204_110722` — [REL:13]; `exp_var: resume` — [REL:908]; wandb group `zen_0122` — [REL:283]; global_step 41,550, eval_step 40,000 — [PT:sonic_release].
- LL: self-referencing `checkpoint` = `experiment_dir` = `...ultralatency_not_usd-20260323_141755` — [LL:8, 13]; `resume: true` — [LL:935]; wandb group `zen_0306`, id `7wypxf8x` — [LL:287-288]; data `train_pyroki_0213_26` — [LL:301]; model export dir `models/0527` — [LLM:240 via diff].
- V11: self-referencing `...heading_wrist_81-20260415_051436` — [V11:8, 13]; `resume: true` — [V11:946]; wandb group `zen_heading_0223`, id `jbpdpb7r` — [V11:287-288]; data `train_pyroki_v081_0414_26`, eval on `pyroki_v0.8.1_splits` — [V11:302, 938-940]; model export dir `models/0430` — [V11M:240].
- Public release dates: low-latency checkpoint 2026-06-16, SONIC v1.1 2026-07-23 — [WBC/README.md:35, 37].
- NVIDIA docs: "Use `sonic_release` for finetuning and evaluation"; "Use `sonic_v1_1` with `sonic_v1_1/last.pt`" — [WBC/docs/source/user_guide/training.md:23-29]. The finetune example uses `+checkpoint=sonic_release/last.pt` — [training.md:107-112].
- Run names of V11 and LL both start with `sonic_release_3pt_` — [V11:13; LL:13]. REL's own name is `sonic_sm_3pt_resume` — [REL:13].
- Weight shapes: the encoders (g1 640→2048, teleop 267→2048, smpl 840→2048) and g1_kin are identical in REL and V11, while g1_dyn and the critic differ (2048 vs 4096 first layer, 6 vs 8 layers) — [PT:sonic_release, PT:sonic_v1_1].

### Inferences
- Timeline from run-name timestamps: base parent 18/01/2026 → REL run 04/02/2026 → LL run 23/03/2026 (data mix dated "0213", wandb group "0306") → V11 run 15/04/2026 (data mix "0414", pyroki v0.8.1; wandb group "zen_heading_0223", which suggests the heading line of experiments started around 23/02) → export 30/04 (V11) and 27/05 (LL) → public release 16/06 (LL) and 23/07 (V11). The run-name timestamps are FACT; reading them as dates (YYYYMMDD, MMDD) is INFERENCE.
- The `sonic_release_3pt_*` prefix on V11 and LL suggests they were launched from the `sonic_release` experiment config family, i.e. "release recipe + variant" (inference from naming). They could not have loaded REL's g1_dyn decoder or critic weights directly, because the layer shapes differ. At most, the encoders and g1_kin could have been warm-started from a REL-like run (shapes match). The heading-input change in V11 would also make a warm-start of the G1/teleop/SMPL encoders semantically mismatched. Nothing on disk records V11's or LL's original initialization, because `checkpoint:` was overwritten by the self-resume path.
- REL's global_step 41,550 may be the cumulative count carried over from the parent or just the resumed run's own count. TRL trainer resume semantics were not checked, so the original release model may have trained more or fewer total iterations than 41,550. Whatever the count means, REL's `tot_timesteps` (5.2e11) is ≈41.5% of V11's and LL's (1.26e12).
- LL's wall-clock (≈23.5 days) is about double V11's (≈12 days) for the same 100k iterations. This could reflect multiple preemptions and restarts with accumulated time, or slower hardware. It is not determinable.
- `exp_var` names hint at the purpose: `wrist_81` most likely means "wrist augmentation, pyroki v0.8.1 data". `not_usd` possibly means the robot asset is not loaded from USD (the config uses MJCF `g1_29dof_rev_1_0.xml` — [V11:306-307]). `ultralatency` = the low-lookahead variant. All three readings are speculative.

### Gaps
- There is no changelog or paper text in the repo tying REL to a specific training duration or dataset version.
- I did not resolve whether REL's `base3_noz_g1_wrist_5pt_consistency` parent also used BONES-SEED or internal data.

## Q4. Approximate parameter counts implied by hidden_dims

### Takeaway
Actor (encoders + decoders, from actual weights): REL ≈25.9 M, V11 ≈53.1 M, LL ≈52.0 M. Critic: REL ≈11.5 M, V11/LL ≈40.0 M. Almost all of the growth is the g1_dyn action decoder (10.2 M → 37.4 M). Only the SMPL encoder differs between V11 and LL (4.64 M vs 3.61 M, because of its 840 vs 336 input).

### Cited Findings (counts from `PT:*` state dicts; architecture from configs)
| Module | REL | V11 | LL | hidden_dims source |
|---|---|---|---|---|
| encoder g1 (in 640) | 4,231,232 | 4,231,232 | 4,231,232 | [V11:90-95] |
| encoder teleop (in 267) | 3,467,328 | 3,467,328 | 3,467,328 | [V11:111-116] |
| encoder smpl (in 840 / 840 / 336) | 4,640,832 | 4,640,832 | 3,608,640 | [V11:132-137; LL:425] |
| decoder g1_dyn (in 994 → 29) | 10,184,221 | 37,393,949 | 37,393,949 | [REL:152-158] vs [V11:154-162] |
| decoder g1_kin (in 64) | 3,347,072 | 3,347,072 | 3,347,072 | [V11:181-185] |
| **policy_state_dict total** | **25,870,714** | **53,080,442** | **52,048,250** | [PT:*] |
| **critic (value_state_dict, in 1645)** | **11,506,396** | **40,049,372** | **40,049,372** | [REL:62-68] vs [V11:62-70] |

- Input dims tie out with config: actor proprioception `actor_obs: 930` = 10-step history × (gravity 3 + ang vel 3 + joint pos 29 + joint vel 29 + last action 29) — [V11M:301; V11:445-482, 934-935]. Decoder input 994 = 930 + 64 token — [V11O header "Decoder input: 994 Token: 64"]. `critic_obs: 1645` — [V11M:302]. `actions_dim: 29` — [V11M:308].
- Checkpoint file sizes (they include optimizer state): REL `last.pt` 469 MB, V11 1.14 GB, LL 1.13 GB. ONNX: decoder ≈150 MB for both V11 and LL; encoder 50.1 MB (V11) vs 45.9 MB (LL) — [`ls -la checkpoints/GEAR-SONIC/*/`].

### Inferences
- The deployed path (encoder + g1_dyn, no g1_kin, no critic) is ≈ 4.2+3.5+4.6+10.2 ≈ 22.5 M params for REL and ≈ 49.7 M (V11) / ≈ 48.7 M (LL). Only one encoder runs per step, so the per-step compute is dominated by g1_dyn: ≈10 M MACs for REL vs ≈37 M for V11/LL, roughly 3.7× more FLOPs per control step in the decoder. ONNX decoder size ≈150 MB ≈ 37.4 M × 4 bytes (fp32) is consistent with this.
- The g1_dyn growth alone is (994·4096 + 4096·4096 + 4096·2048) − (994·2048 + 2048·2048) + … ≈ +27.2 M params, which matches 37.39 M − 10.18 M = 27.21 M.

### Gaps
- REL's ONNX models (top-level `model_encoder.onnx` / `model_decoder.onnx` on HF) are not in the local `sonic_release/` folder, so their size and export dims could not be cross-checked on disk.
