# Public evidence on the three GEAR-SONIC checkpoints (`sonic_release`/default, `low_latency`, `sonic_v1_1`)

Status: current as of 2026-10-01. Labels: **FACT** = stated verbatim or shown directly in a source; **INFERENCE** = my reading of facts, not stated by NVIDIA.

Sources checked:
- Hugging Face model card (local copy `checkpoints/GEAR-SONIC/README.md`, matches HF `lastModified` 2026-08-26), the HF commit API, and HF `config.json`.
- GitHub `NVlabs/GR00T-WholeBodyControl`: README, docs (local checkout at `b042411`, 2026-09-23), commit patches, and issues/PRs. Issues were read through the API until the unauthenticated rate limit ran out, then from the issue HTML pages.
- SONIC paper arXiv 2511.07820v4 (13 Aug 2026).
- Project page https://nvlabs.github.io/GEAR-SONIC/.
- Isaac-GR00T README and `examples/GR00TWholeBodyControl` (local checkout `51d4c89`, 2026-08-20).

URL shorthands:
- HF = https://huggingface.co/nvidia/GEAR-SONIC
- HFC = https://huggingface.co/nvidia/GEAR-SONIC/commits/main
- GH = https://github.com/NVlabs/GR00T-WholeBodyControl
- DOCS = https://nvlabs.github.io/GR00T-WholeBodyControl

Author roles come from the GitHub `author_association` field. ZhengyiLuo, chenran-nvidia, fcastanedaga and KevinMi2023p are COLLABORATOR, which means maintainers. All four are also co-authors of the paper (Zhengyi Luo, Chenran Li, Fernando Castañeda) or NVIDIA staff. The HF uploader is `Zhengyiluo`.

---

## 1. Release date and stated purpose of each checkpoint (exact wording)

### Takeaway
- **Default (`sonic_release`)**: ONNX released 2026-02-19 and the PyTorch checkpoint on 2026-04-10. Described as the "original general-purpose" controller.
- **`low_latency`**: uploaded 2026-06-16. It has a 4-frame (~80 ms) SMPL lookahead and is aimed at "more responsive whole-body teleoperation and VLA execution."
- **`sonic_v1_1`**: uploaded 2026-07-23 (US time; 2026-07-24 UTC). It is robot-heading-normalized, trained with wrist-pose augmentation, and aimed at "heading-stable whole-body teleoperation and SONIC-backed VLA policies." Its wording was edited on 2026-08-20 from "3-point" to "whole-body" teleoperation.
- NVIDIA describes the checkpoints only qualitatively, through observation and lookahead specs. There are no GitHub Releases or tags.

### Cited Findings

**Default SONIC (`sonic_release`, top-level ONNX)**
- FACT: GitHub News, 2026-02-19: "🎉 Released GEAR-SONIC: pretrained checkpoints, C++ inference, VR teleoperation, and documentation." — [GH README News](https://github.com/NVlabs/GR00T-WholeBodyControl#news)
- FACT: HF history. On 2026-02-17 the first ONNX files were uploaded as `model_releasev1-1_encoder.onnx` / `model_releasev1-1_decoder.onnx`. On 2026-02-19 they were renamed to `model_encoder.onnx` / `model_decoder.onnx`, and on the same day the decoder was replaced: "Fix model_decoder.onnx: upload correct decoder (was duplicate of encoder)". — [HFC](https://huggingface.co/nvidia/GEAR-SONIC/commits/main)
- FACT: GitHub News, 2026-04-10: "🚀 Released **SONIC training code and checkpoint** on HuggingFace." HF commits on 2026-04-10 read "Add SONIC release checkpoint" and "Add SONIC release training config". On 2026-04-11 HF added "SMPL retargeted motion data (bones_seed filtered, 131K sequences)". — [GH README News](https://github.com/NVlabs/GR00T-WholeBodyControl#news); [HFC](https://huggingface.co/nvidia/GEAR-SONIC/commits/main)
- FACT: Model card row (GitHub README / docs): "Default general-purpose SONIC controller for motion tracking, planning, teleoperation, and compatibility with existing deployments. G1 and teleoperation future-reference observations use `step5`." SMPL input: "10 future frames at 20 ms spacing, approximately 200 ms of reference lookahead". — [DOCS model card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html)
- FACT: HF model card row: "General-purpose motion tracking, planning, and teleoperation. G1 and teleoperation references use `step5`." — [HF](https://huggingface.co/nvidia/GEAR-SONIC)
- FACT: `config.json` manifest: `"default": {"display_name": "Default SONIC", "description": "Original general-purpose SONIC whole-body controller."}`. The manifest was added 2026-08-25. — [HF config.json](https://huggingface.co/nvidia/GEAR-SONIC/blob/main/config.json)
- FACT: "Choosing a Model": "Use **Default SONIC** when you want the original release, the broadest compatibility with existing deployment setups, or the standard motion-tracking and planning controller." — [DOCS model card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html)

**Low-latency (`low_latency/`)**
- FACT: HF commit "Add low-latency SONIC checkpoint", 2026-06-16T06:21Z. Commit body: "Adds the low-latency SONIC variant from sonic_release_3pt_ultralatency_not_usd model_step_100000. The final PyTorch checkpoint is exposed as low_latency/last.pt." HF docs commits followed on 2026-06-17. — [HFC](https://huggingface.co/nvidia/GEAR-SONIC/commits/main)
- FACT: GitHub News, 2026-06-16: "**Low-latency teleoperation checkpoint** — released a SONIC checkpoint with 4-frame SMPL reference lookahead for more responsive whole-body teleoperation." The C++ observation support landed just before, in commit `5933a12` "Add low-latency SONIC observation support" (2026-06-12). — [GH README News](https://github.com/NVlabs/GR00T-WholeBodyControl#news); [GH commits](https://github.com/NVlabs/GR00T-WholeBodyControl/commits/main)
- FACT: Model card row: "Intended for more responsive whole-body teleoperation and VLA execution. G1 and teleoperation future-reference observations use `step1`. Use its encoder, decoder, and observation config together." SMPL input: "4 future frames at 20 ms spacing, approximately 80 ms of reference lookahead". — [DOCS model card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html)
- FACT: The download guide says: "Its SMPL encoder uses **4 future reference frames**, compared with **10 frames** in the default release… reduces SMPL reference lookahead from approximately **200 ms to 80 ms**. This is the controller's reference lookahead, not a measurement of total end-to-end system latency. The checkpoint does not replace the default top-level deployment policy." — [DOCS download_models](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html#low-latency-teleoperation-checkpoint)
- FACT: "Choosing a Model": "Use **Low-latency teleoperation** when responsiveness to streamed SMPL, VR, or VLA commands is the priority. Its shorter reference horizon reduces commanded motion lookahead, but it does not remove latency elsewhere in the system." — [DOCS model card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html)
- FACT: Manifest: "Whole-body teleoperation controller with a shorter SMPL reference horizon." — [HF config.json](https://huggingface.co/nvidia/GEAR-SONIC/blob/main/config.json)
- FACT: The released `low_latency/observation_config.yaml` header reads "(smpl 4-frame lookahead; g1/teleop lookahead at step1, body-frame orientation). Encoder input dimension: 1247 Decoder/policy dimension: 994 token: 64". — [HF low_latency/observation_config.yaml](https://huggingface.co/nvidia/GEAR-SONIC/blob/main/low_latency/observation_config.yaml)

**SONIC v1.1 (`sonic_v1_1/`)**
- FACT: HF commits:
  - 2026-07-24T02:16Z: "Add heading and wrist teleoperation checkpoint"
  - 2026-07-27: "Rename checkpoint release to sonic_v1_1" and "Update SONIC v1.1 observation config label"
  - 2026-08-26: "Document SONIC v1.1 deployment tuning"

  — [HFC](https://huggingface.co/nvidia/GEAR-SONIC/commits/main)
- FACT: GitHub News entry dated 2026-07-23, added in commit `54d0b10` on 2026-07-28. Original wording: "released a robot-heading-normalized SONIC controller trained with wrist-pose augmentation for 3-point teleoperation and SONIC-backed VLA execution." Commit `a0732b6` ("v1.1", 2026-08-20) changed "3-point teleoperation" to "whole-body teleoperation". The same commit changed the caption "3-point teleoperation with wrist-pose tracking, kneeling, ground pickup, and dynamic whole-body control" to "Whole-body teleoperation mode with expressive wrist motion and dynamic movement." — [commit 54d0b10](https://github.com/NVlabs/GR00T-WholeBodyControl/commit/54d0b102bb8876a54c9d41796bd9f221c9e042d9); [commit a0732b6](https://github.com/NVlabs/GR00T-WholeBodyControl/commit/a0732b642c0333077e127a2f56ab0014c196bca4)
- FACT: Current model card row: "Uses robot-heading-normalized target orientation and was trained with wrist-pose augmentation. Intended for heading-stable whole-body teleoperation and SONIC-backed VLA policies that use this controller. G1 and teleoperation future-reference observations use `step5`; this is not the low-latency model." — [DOCS model card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html)
- FACT: "Choosing a Model": "Use **SONIC v1.1** for robot-heading-normalized whole-body teleoperation or a SONIC-backed VLA policy trained against this controller. It retains the 10-frame SMPL horizon and was trained with wrist-pose augmentation." — [DOCS model card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html)
- FACT: Manifest: "Whole-body teleoperation controller with robot-heading-normalized targets and wrist-pose augmentation." — [HF config.json](https://huggingface.co/nvidia/GEAR-SONIC/blob/main/config.json)
- FACT: The released `sonic_v1_1/observation_config.yaml` header reads "Encoder input: 1751 Decoder input: 994 Token: 64. G1/teleop references use 10 frames at step5. SMPL and wrist references use 10 frames at step1. Root orientations are normalized by robot heading." The recipe `sonic_v1_1.yaml` (added in `54d0b10`) contains `randomize_wrist_poses: true` and uses `*_heading` anchor-orientation observations for all three encoders. — [HF sonic_v1_1/observation_config.yaml](https://huggingface.co/nvidia/GEAR-SONIC/blob/main/sonic_v1_1/observation_config.yaml); [commit 54d0b10](https://github.com/NVlabs/GR00T-WholeBodyControl/commit/54d0b102bb8876a54c9d41796bd9f221c9e042d9)

**Common to all three**
- FACT: "All three models use the SONIC universal-token controller, produce 64-dimensional latent motion tokens, run the controller at 50 Hz, and support SMPL pose, G1 motion reference, and teleoperation inputs." Also: "Each ONNX encoder must be used with the decoder and observation config from the same variant." — [DOCS model card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html); [HF](https://huggingface.co/nvidia/GEAR-SONIC)
- FACT: Under "Limitations and Safety": "SONIC v1.1 is not a low-latency checkpoint"; "These checkpoints target the Unitree G1 embodiment"; "Test in simulation before deployment and keep a safety operator ready to stop a physical robot." — [DOCS model card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html)
- FACT: The repository has no GitHub Releases ("There aren't any releases") and no tags. The README News section is the only release-notes channel. — [GH releases](https://github.com/NVlabs/GR00T-WholeBodyControl/releases)
- FACT: On 2026-10-01 HF reports 2,575 downloads and 77 likes for the whole repo, not broken down per variant. — [HF API](https://huggingface.co/api/models/nvidia/GEAR-SONIC)

### Inferences
- INFERENCE: The Feb-2026 filenames `model_releasev1-1_*.onnx` refer to the *default* release. They are unrelated to the July "SONIC v1.1". Anyone citing "v1.1" should say which one is meant.
- INFERENCE: The internal run names in the released `config.yaml` files suggest training order and dates:
  - default: resumed run `sonic_sm_3pt_resume-20260204` from `base3_noz_g1_wrist_5pt_consistency-20260118`
  - low_latency: `sonic_release_3pt_ultralatency_not_usd-20260323`
  - v1.1: `sonic_release_3pt_heading_wrist_81-20260415`, wandb_group `zen_heading_0223`

  This means low_latency and v1.1 were trained months before they were released. The names "sonic_release_3pt_*" suggest both derive from the release recipe. They do not prove that either was fine-tuned from the release weights. (Source: released `*/config.yaml` on [HF](https://huggingface.co/nvidia/GEAR-SONIC/tree/main).)
- INFERENCE: #221 (2026-07-21) asks about a "better squatting + wrist tracking" checkpoint teased on X by Zhengyi Luo. v1.1, uploaded two days later with "wrist-pose augmentation", is probably that checkpoint. No maintainer confirmed this.

### Gaps
- No official changelog lists exactly what changed in v1.1 beyond "heading-normalized + wrist-pose augmentation". Reward changes visible in commit `54d0b10`, such as `tracking_vr_2wrists_local_ori` and `local_feet_acc_energy_5pt`, are not described in prose anywhere.
- I could not read the X post (https://x.com/zhengyiluo/status/2078235641469702379) directly. Its content is known only from the issue author's paraphrase in [#221](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/221).

---

## 2. Does any source compare the checkpoints quantitatively (success rate, MPJPE, latency)?

### Takeaway
No. I found no NVIDIA source (model card, docs, paper, HF, GitHub) that gives success rate, MPJPE, or measured latency for any of the three released checkpoints, or that compares them with each other. The only numbers given are design specs: 10 vs 4 future SMPL frames, ~200 vs ~80 ms reference lookahead. NVIDIA explicitly says these are not end-to-end latency measurements. The v1.1 gain tuning is supported only by qualitative claims ("improves… observed wrist tracking").

### Cited Findings
- FACT: "The lookahead values describe the reference horizon presented to the controller. They are **not** measurements of total end-to-end teleoperation latency, which also includes sensing, networking, preprocessing, and inference." — [GH README Model Card](https://github.com/NVlabs/GR00T-WholeBodyControl#model-card)
- FACT: v1.1 tuning claim, with no numbers: "This is the tested v1.1 deployment tuning. Hardware indices `4` and `10` are the left and right ankle-pitch motors; increasing their gains improves whole-body stability and observed wrist tracking." — [HF](https://huggingface.co/nvidia/GEAR-SONIC)
- FACT: The paper's quantitative results never name a released checkpoint. The words "v1.1" and "low latency" do not appear in the v4 PDF. The reported numbers are:
  - Scaling: "the largest model achieved 99.6% success with 23.8 mm MPJPE-L on test-content, compared to 98.0% success with 27.7 mm for the smallest (1.2M)"
  - MuJoCo cross-tracker: "98.5%/99.2%/97.2% success on test-content/test-repetition/PHUMA"; 23.7 mm MPJPE-L
  - Real robot: "99.2% success rate compared to 100% in simulation, with an overall MPJPE-L of 25.7 mm (vs. 22.3 mm in sim)" on 124 motions

  — [arXiv 2511.07820v4](https://arxiv.org/abs/2511.07820)
- FACT: The test splits are internal: test-content has 7,016 clips (15 h, 182 sub-categories absent from training) and test-repetition has 9,395 clips (12 h). The training split is 317,189 clips (611 h). — [arXiv 2511.07820v4, Table 2](https://arxiv.org/abs/2511.07820)
- FACT: Third-party benchmark (community, not NVIDIA): rhoyn's G1 disturbance-walking benchmark reports `sonic` "47 / 24 %" completion (MuJoCo/PhysX) vs `gr00t_wbc` "77 / 78 %". The author writes "I still read that as a deployment bug on my side". The post does not say which SONIC checkpoint was used; it mentions an "Encoder 1762-dim" input, which matches none of the released configs (1751 for default and v1.1, 1247 for low_latency). No maintainer replied. — [#269](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/269) (rhoyn, NONE, 2026-09-07)
- FACT: Community reproduction reports only. A user training on "~130k boneseed" says success on training data is "close to that of your released model" but "noticeably inferior" on AMASS/LAFAN. As of the last comment (2026-09-07, pinging @tingwuwang) there is no maintainer answer. — [#136](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/136) (stevenxmy, NONE, 2026-05-05)

### Inferences
- INFERENCE: Anyone who needs a numeric comparison of the three checkpoints must run it themselves, for example with `eval_agent_trl.py` on the same motion set. Public material contains none.

### Gaps
- No per-checkpoint MPJPE, success, or latency numbers exist publicly. No NVIDIA benchmark ties a VLA result to a specific released checkpoint.

---

## 3. Which checkpoint does NVIDIA recommend for VLA (GR00T N1.7) use? Does the paper's model correspond to a released checkpoint?

### Takeaway
NVIDIA does not name one checkpoint as *the* recommended one for N1.7. The guidance is conditional:
- The **default** is what the plain launcher and `deploy.sh` use.
- **low_latency** is labelled for "VLA execution" when responsiveness matters.
- **v1.1** is to be used "when the VLA policy was trained against" it.

The implied rule is to deploy with the same controller used to collect and train the VLA data, because the VLA predicts that controller's latent tokens.

The paper never maps its model to a released checkpoint. Its VLA results use **GR00T N1.5**, not N1.7. Its Table S1 matches the default checkpoint on SMPL horizon, but it matches low_latency/v1.1 (not the default) on decoder width.

### Cited Findings
- FACT: Default is the implicit default. `deploy.sh` has `CHECKPOINT_DEFAULT="policy/release/model"` and `OBS_CONFIG_DEFAULT="policy/release/observation_config.yaml"`. The model card shows the "default Python VLA launcher" with no checkpoint flags: `python gear_sonic/scripts/launch_inference.py --camera-host … --prompt "pick up the cup"`. — [GH deploy.sh](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/deploy.sh); [GH README Model Card](https://github.com/NVlabs/GR00T-WholeBodyControl#model-card)
- FACT: VLA Inference guide, v1.1: "Use the `sonic_v1_1/` checkpoint when the VLA policy was trained against the robot-heading-normalized SONIC controller. It uses a 10-frame SMPL/wrist reference horizon and was trained with wrist-pose augmentation. It is not the low-latency checkpoint." Added 2026-07-28. — [DOCS vla_inference](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html)
- FACT: VLA Inference guide, low-latency: "The `low_latency/` checkpoint is configured for responsive whole-body teleoperation… This is reference lookahead, not total end-to-end system latency." The model card row says "Intended for more responsive whole-body teleoperation and VLA execution." — [DOCS vla_inference](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html)
- FACT: The VLA workflow is token-based. The VLA streams 64-dim tokens to the SONIC decoder (diagram: "VLA Model 2.5 Hz → latent tokens 64-dim x 40 → SONIC Decoder (C++) 50 Hz"). Maintainer chenran-nvidia (COLLABORATOR, 2026-05-18): "If you are streaming tokens as in our VLA workflow, then there is no separate encoder or mode selection involved, and it should work directly." — [DOCS vla_workflow](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html); [#152](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/152)
- FACT: ZhengyiLuo (COLLABORATOR, 2026-04-14): "We change Gr00t's output to latent codes for SONIC". — [#108](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/108)
- FACT: fcastanedaga (COLLABORATOR, 2026-06-05) answered whether the arXiv v3 loco-manipulation results are reproducible with released code: "Yes, the pipeline released in this repository—combined with the updates made in the Isaac-GR00T repository to support the `UNITREE_G1_SONIC` embodiment tag—is fully sufficient to reproduce the whole-body loco-manipulation results detailed in our arXiv v3 paper… No additional unreleased components or hidden scripts are required." He did not name a checkpoint. — [#171](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/171)
- FACT: Paper VLA results: "A GR00T N1.5 model, fine-tuned on teleoperated data, is evaluated across five whole-body loco-manipulation tasks… the VLA achieved 75% average success using the universal token action space". Token vs SMPL action-space ablation: 68% vs 27%. The project page also says "We connect a VLA foundation model (GR00T N1.5)". — [arXiv 2511.07820v4 Table 1](https://arxiv.org/abs/2511.07820); [project page](https://nvlabs.github.io/GEAR-SONIC/)
- FACT: KevinMi2023p (COLLABORATOR, 2026-08-13) restated the same paper ablation: "FSQ motion tokens average 68% success versus 27% for explicit SMPL supervision". — [#236](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/236)
- FACT: Project page: "All models shown in the videos will be released." — [project page](https://nvlabs.github.io/GEAR-SONIC/)
- FACT: Paper Table S1 values:
  - "Future frames F_r = F_h = F_m = 10 frames"
  - "Frame interval Δt_r = Δt_m = 0.1s, Δt_h = 0.02s", so a 200 ms SMPL horizon
  - "Decoder (actions) MLP hidden = [4096, 4096, 2048, 2048, 1024, 1024, 512, 512]"

  — [arXiv 2511.07820v4, Table S1](https://arxiv.org/abs/2511.07820)
- FACT: Decoder widths in the released configs:
  - `sonic_release/config.yaml`: g1_dyn decoder hidden_dims `[2048, 2048, 1024, 1024, 512, 512]`
  - `low_latency/config.yaml` and `sonic_v1_1/config.yaml`: `[4096, 4096, 2048, 2048, 1024, 1024, 512, 512]`

  Decoder ONNX file sizes match this: 40.9 MB for the default, ~150 MB for low_latency and v1.1. — [HF tree](https://huggingface.co/nvidia/GEAR-SONIC/tree/main)
- FACT: Isaac-GR00T's N1.7 real-G1 benchmark page (added 2026-06-24) reports 73.6% (159/216) on walk-to-table pickup and 66.7% (8/12) on soda-can-to-trash. It does not say which SONIC checkpoint was used. — [Isaac-GR00T examples/GR00TWholeBodyControl](https://github.com/NVIDIA/Isaac-GR00T/tree/main/examples/GR00TWholeBodyControl)

### Inferences
- INFERENCE: Each checkpoint has its own token space, because the encoder, decoder, and observation configs differ and must be used together. A VLA fine-tuned on tokens collected with one controller should therefore be deployed with that same controller. This is the practical meaning of "SONIC-backed VLA policies trained against this controller."
- INFERENCE: On 2026-06-05, when fcastanedaga said the released pipeline suffices to reproduce the arXiv v3 results, only the default checkpoint was public. low_latency came on 06-16 and v1.1 on 07-23. That statement therefore implicitly vouches for the default checkpoint, or at least does not require the later ones.
- INFERENCE: By decoder size, the paper's Table S1 architecture matches the low_latency and v1.1 configs, not the default release config. By SMPL horizon (10 frames @ 20 ms), it matches the default and v1.1 but not low_latency. **No released checkpoint can be called "the paper model" from public evidence.** v1.1 is the closest on both dimensions, but this is unconfirmed, and its training data path is internal (see Q5).
- INFERENCE: The Isaac-GR00T N1.7 benchmark is dated 2026-06-24, before v1.1 existed. If it used a released checkpoint, it used the default or low_latency, but the page does not say.

### Gaps
- No NVIDIA statement says which checkpoint was used for the N1.7 demos or benchmark, or for the paper's tables.
- No statement explains why the default release has a smaller decoder than Table S1.

---

## 4. Known issues reported per checkpoint, and fixes

### Takeaway
- **Default**: most reports are hardware-fit problems (ankle/hip overheating, forward lean on older white/grey-pelvis G1, heavy hands). Maintainers attribute these to robot version, reference posture, or calibration. There is also a fixed sim2sim obs-alias bug.
- **low_latency**: two community reports. One (#190, unstable planner walking) turned out to be the user's own control-loop bug, and the user then reported "no issues". The other (#231, weak yaw in Planner VR3PT, plausibly from the missing heading-only observation variant for 4-frame) is open with no reply.
- **v1.1**: no public bug reports found. NVIDIA itself ships an opt-in ankle-pitch Kp/Kd ×1.5 "tested v1.1 deployment tuning", released 2026-08-31 "to reduce stumbling".

### Cited Findings

**Fix of 2026-08-31 (per-motor Kp/Kd)**
- FACT: Commit `4c6c948` (zhengyiluo, 2026-08-31) "Add SONIC control and adaptive sampling fixes". It originally added the News line "**SONIC fixes** — added per-motor Kp/Kd scaling to reduce stumbling and fixed adaptive sampling." Commit `6dc961c` changed this the same day to the current "**SONIC deployment update** — added per-motor Kp/Kd scaling to reduce stumbling." The commit added `motor_gain_scaling.{hpp,cpp}`, the `--motor-kp-scale/--motor-kd-scale` deploy flags, and `attribute_pre_reset_cursor: true` to the `sonic_release`, `sonic_v1_1` and `sonic_bones_seed` recipes. — [commit 4c6c948](https://github.com/NVlabs/GR00T-WholeBodyControl/commit/4c6c9480909bfb5db89f62d5f9e3c4b4af072f4f); [commit 6dc961c](https://github.com/NVlabs/GR00T-WholeBodyControl/commit/6dc961c412c8bcbd2ae80283d0cc068de1687f85)
- FACT: The gain tuning is documented only for v1.1: "The gain settings are the tested v1.1 hardware tuning. Indices `4` and `10` refer to the left and right ankle-pitch motors; the change improves whole-body stability and observed wrist tracking. They remain explicit so other model variants retain their original gains." The VLA guide adds that it "does not directly scale the wrist motors." HF documented this on 2026-08-26, before the GitHub commit. — [DOCS download_models](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html#sonic-v11-checkpoint); [DOCS vla_inference](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html); [HFC](https://huggingface.co/nvidia/GEAR-SONIC/commits/main)
- FACT: Adaptive-sampling fix (training side): "failures are attributed to the motion cursor tracked before environment reset. Legacy checkpoints still load their policy and optimizer state, while old incompatible adaptive-sampling counters are discarded once." — [HF](https://huggingface.co/nvidia/GEAR-SONIC)
- FACT: Earlier, on 2026-06-02, chenran-nvidia (COLLABORATOR) advised against PD tuning for ankle overheating: "As for tuning the PD gains, I don't think that would directly fix the issue. It could instead make the policy operate under an even more misaligned condition." — [#174](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/174#issuecomment-4607591850)

**Default / `sonic_release`**
- FACT: Ankle-roll overheating (#174, opened 2026-06-02, still open). chenran-nvidia (COLLABORATOR, 2026-06-04): "The SONIC policy is mainly built for the newer version. If you are using the white-pelvis version, the behavior may not be ideal." On 2026-06-08: "Our current policy was trained with the PD gains and armature parameters corresponding to the newer motor. To optimize performance on the older G1 version, you can fine-tune the policy using the PD gains and armature parameters of the older G1 hardware." A 2026-09-22 community log, using the "default `policy/release` checkpoint", shows left ankle roll at 49→80 °C within 90 s of standing still. — [#174](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/174)
- FACT: Similar community reports with no maintainer reply: forward lean and ankle overheating on an "old model, metallic pelvis" G1 ([#217](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/217), 2026-07-14), and a "High Temperature" warning while standing ([#247](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/247), 2026-08-06). On #247 a user reports that taking one Planner step stabilized temperatures (2026-09-14).
- FACT: Sim2sim with fine-tuned `sonic_release` ONNX failed: "8 of 10 encoder observations missing from C++ registry" ([#233](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/233), Harshita-82, CONTRIBUTOR, 2026-08-03). The fix, PR #232 "Add sonic_release observation aliases to encoder registry", was merged 2026-09-15 (`7f14429`). — [GH commits](https://github.com/NVlabs/GR00T-WholeBodyControl/commits/main)
- FACT: Base sway in sim2sim (#185): chenran-nvidia (2026-07-06) attributed it to user error: "it seems like you didn't press "9" on the sim side to release the robot". — [#185](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/185)
- FACT: Fine-tuning `sonic_release` for 23-DoF G1 shows a "death curve" (community, no maintainer reply). — [#168](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/168)
- FACT: G1 joint-encoder latency on the default checkpoint. A user computed ~1 s of reference window for `10frame_step5`. ZhengyiLuo (2026-03-03): "Yes, retraining would be necessary. That's for the teleoperation encoder (human pose) we uses the lower latency variant." chenran-nvidia (2026-03-09): "we recommend performing teleoperation using our **SMPL encoder**", plus an unsupported workaround with the warning "We **do not recommend** using the policy in this way." ZhengyiLuo (2026-03-09): "We are also training a lower latency model. Stay tuned!" — [#60](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/60)

**low_latency**
- FACT: #190 (LEECHOONGHO, 2026-06-23) reported that in VR3pt Planner mode, fast arm motion caused falls, Slow_Walk/Walk "does not function well", and temperature warnings appeared. It also asked for a planner matched to the low-latency decoder. chenran-nvidia blamed the heavier hands (Inspire RH56E2, ~790 g) and the older G1: "Since you are using the older version G1 robot, the performance drop is expected." The user later found their own bug, a hand/glove comm inside the control loop: "After moving that part out of the loop, the control cycle started working properly. The low‑latency mode appears to have no issues." Closed 2026-07-24. — [#190](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/190)
- FACT: #231 (LEECHOONGHO, 2026-07-30, open, 0 comments): "Low-latency model loses yaw responsiveness in Planner VR3PT mode". The report says the standard model's yaw "works well in both Pose and Planner", while for low-latency only exaggerated inputs turn the robot. The reporter's analysis: "the 4-frame (low-latency) variant only registers mode 0" (full quaternion; no heading/refheading anchor variant). — [#231](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/231)
- FACT: #198 (2026-06-29) reported a compliance warning and an unstable sim robot with both the low_latency and default deploy commands. chenran-nvidia: "compliance warning is expected", and suggested checking with the keyboard reference first ("The issue may stem from your Pico settings or the TensorRT version"). — [#198](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/198)
- FACT: The low-latency observation config notes that it "Requires the smpl_*_4frame_step1 entries in GetObservationRegistry() … + a rebuild to load", so older deploy binaries cannot load it. — [HF low_latency/observation_config.yaml](https://huggingface.co/nvidia/GEAR-SONIC/blob/main/low_latency/observation_config.yaml)

**v1.1**
- FACT: No GitHub issue mentions "v1.1", "v1_1" or "sonic_v1_1" (search on 2026-10-01 returned 0 results for `"v1.1"` and `v1_1`). The only v1.1-specific "issue" is NVIDIA's own ankle-pitch gain tuning described above. — [GH issue search](https://github.com/NVlabs/GR00T-WholeBodyControl/issues?q=v1_1)
- FACT: Wrist-related community issue that predates v1.1: wrist roll not calibrated in SMPL encoder mode ([#103](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/103), closed). There is also a request for 5-tracker (wrist tracker) PICO input. KevinMi2023p (COLLABORATOR, 2026-08-13): "We support headset plus 2 controllers and two ankle trackers for full body input." — [#227](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/227)

### Inferences
- INFERENCE: The "stumbling" that the 2026-08-31 Kp/Kd scaling addresses is documented only in the context of v1.1. The flags are opt-in, and the docs say the other variants keep their original gains. Applying them to the default or low_latency checkpoints is untested publicly.
- INFERENCE: Many "default checkpoint" complaints (overheating, forward lean) are tied to G1 hardware revision or calibration rather than to the checkpoint itself. No public evidence shows that v1.1 or low_latency behave better on older G1s.
- INFERENCE: #231 suggests that low_latency does not get the heading-isolated anchor observation that v1.1 introduces systematically. For workflows with heavy yaw or planner use, this is a plausible functional difference between the two. NVIDIA has not confirmed it.

### Gaps
- There are no maintainer responses on #231, #217, #247 or #269.
- No public data (numbers or video) supports "improves whole-body stability and observed wrist tracking" for the v1.1 gain tuning.
- No public bug report exists for v1.1 yet. It may be lightly used, or users may not name it.

---

## 5. Maintainer statements on GitHub about differences, training data (BONES-SEED vs internal), and which to use

### Takeaway
- Maintainers said little about checkpoint differences beyond the model card. In March 2026 they recommended the SMPL encoder for teleop and announced that a lower-latency model was in training (#60). In June they said the policy targets the newer black-pelvis G1 (#174). In June they said the released pipeline reproduces the paper's VLA results (#171).
- **No maintainer stated what data each released checkpoint was trained on.** The paper says it trained on a 100M-frame internal mocap set, of which BONES-SEED is "a substantial portion".
- The released configs point to BONES-SEED paths for the default checkpoint, but to internal NVIDIA paths (`wbc_data/training_mixes/train_pyroki_*`) for low_latency and v1.1.

### Cited Findings
- FACT: Paper: "SONIC, trained on 100 million frames of motion over 21k GPU hours (128 GPUs over 7 days)…" Also: "A substantial portion of our motion-capture dataset has been publicly released as the BONES-SEED dataset… 142,220 annotated motion sequences (288 hours) from 522 actors". The paper's training split is 317,189 clips (611 h). — [arXiv 2511.07820v4](https://arxiv.org/abs/2511.07820)
- FACT: HF commit 2026-04-11: "Add SMPL retargeted motion data (bones_seed filtered, 131K sequences)", shared by all variants as `bones_seed_smpl/` in the manifest. — [HFC](https://huggingface.co/nvidia/GEAR-SONIC/commits/main); [HF config.json](https://huggingface.co/nvidia/GEAR-SONIC/blob/main/config.json)
- FACT: Data paths in the released training configs:
  - `sonic_release/config.yaml`: `motion_file: data/motion_lib_bones_seed/robot_filtered/`, `smpl_motion_file: data/motion_lib_bones_seed/1215_bones_seed_filtered/`
  - `low_latency/config.yaml`: `motion_file: /mnt/amlfs-01/shared/wbc_data/training_mixes/train_pyroki_0213_26/`, `smpl_motion_file: /mnt/amlfs-01/shared/wbc_data/smpl_retarget/1215/`
  - `sonic_v1_1/config.yaml`: `motion_file: - ${shared_dir}/wbc_data/training_mixes/train_pyroki_v081_0414_26/`, `smpl_motion_file: - ${shared_dir}/wbc_data/smpl_retarget/1215/`

  The public `sonic_v1_1.yaml` recipe in the repo uses `data/motion_lib_bones_seed/robot_filtered` and `data/bones_seed_smpl` instead. — [HF tree](https://huggingface.co/nvidia/GEAR-SONIC/tree/main); [commit 54d0b10](https://github.com/NVlabs/GR00T-WholeBodyControl/commit/54d0b102bb8876a54c9d41796bd9f221c9e042d9)
- FACT: Maintainer statements on which to use / differences:
  - ZhengyiLuo (2026-03-03, #60): "That's for the teleoperation encoder (human pose) we uses the lower latency variant."
  - ZhengyiLuo (2026-03-09, #60): "We are also training a lower latency model. Stay tuned!"
  - chenran-nvidia (2026-03-09, #60): "we recommend performing teleoperation using our **SMPL encoder**, as G1 joint-based teleoperation may introduce additional latency and errors due to the quality of online retargeting."
  - chenran-nvidia (2026-05-18, #152): "The released ONNX model includes all three modes. `encoder_mode_4` is used to control which mode you want to run."
  - chenran-nvidia (2026-06-04, #174): "The SONIC policy is mainly built for the newer version [black pelvis]."
  - ZhengyiLuo (2026-06-06, #142): "we are reporting training on the LAFAN dataset but eval on AMASS here; this way the evaluation dataset is the same and out-of-distribution from the training set."

  — [#60](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/60); [#152](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/152); [#174](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/174); [#142](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/142)
- FACT: Unanswered questions about training data and release configs:
  - #136 asks for "the detailed training configuration used for the current 'sonic release'" (2026-05-05; pinged @tingwuwang 2026-09-07, no answer).
  - #174 (2026-06-09) asks "Which exact Isaac Lab robot asset/config was used to train the released `sonic_release/last.pt` checkpoint?" with no maintainer answer.
  - #221 asks what the squat/wrist checkpoint improves (no answer).

  — [#136](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/136); [#174](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/174); [#221](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/221)
- FACT: Community claim, not a maintainer: "`sonic_release` was trained with a simplified fakehand/Dex3 configuration, not Inspire" (rylinjames, NONE, 2026-05-06; unverified). — [#121](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/121)

### Inferences
- INFERENCE: The `config.yaml` files for low_latency and v1.1 reference internal NVIDIA training mixes (`train_pyroki_0213_26`, `train_pyroki_v081_0414_26`), not the public BONES-SEED path. Those two checkpoints were therefore most likely trained on an internal data mix, probably a superset of BONES-SEED as the paper describes. Their behavior may not be reproducible from public data alone.
- INFERENCE: The default config's BONES-SEED paths may have been sanitized for release, since the same file references internal run directories, so it is not conclusive either. No maintainer has confirmed what data any checkpoint used.
- INFERENCE: Users training from scratch on BONES-SEED report gaps against the released model on AMASS/LAFAN (#136). That fits, but does not prove, the idea that the released weights benefited from data or recipe choices that are not public.

### Gaps
- No explicit maintainer statement compares default vs low_latency vs v1.1 (pros/cons, measured differences). Nor is there an explicit recommendation for GR00T N1.7 data collection.
- No statement confirms the training data or base initialization for any released checkpoint.
- The GitHub API rate limit (unauthenticated) stopped full comment retrieval for a few low-relevance threads. Issues #227, #247, #142, #136, #236, #128, #37, #108, #168 and #269 were read from HTML instead. Other PRs, such as #180 (open, community "Add low-latency SONIC observation support"), were not read in full.
