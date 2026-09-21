# WHOLE-BODY HUMANOID CONTROL: GEAR-SONIC + NVIDIA GR00T N1.7 ON MUJOCO 3D
## End-to-End Integration of Multimodal VLA Policy, Whole-Body Controller & Unitree G1 Physics Simulation

This repository provides an integrated, closed-loop simulation and deployment platform for the **Unitree G1 humanoid robot (29 whole-body DoFs + 14 dexterous hand DoFs)** performing dual soda can sorting (**Coca-Cola** red and **Pepsi** blue) on **MuJoCo 3D**.

The system establishes a direct bridge connecting the three most advanced components in NVIDIA Robotics:
1. **NVIDIA GR00T N1.7 VLA Policy**: Takes natural language instructions and real-time first-person RGB camera frames (`ego_view` $448 \times 672$), predicting 40-step action chunks (**64-dim latent motion tokens** + **14-dim hand joints**).
2. **GEAR-SONIC Whole-Body Controller (WBC)**: Decodes the 64-dim latent tokens at **$50\text{ Hz}$** ($dt=20\text{ ms}$) into position setpoints across all 43 motors, maintaining balance, center-of-mass stability, and coordinated dual-arm reaching.
3. **MuJoCo 3D Physics Simulation**: Steps multi-contact dynamics at **$500\text{ Hz}$**, renders camera sensors offscreen, and provides an interactive real-time 3D GUI viewer.

---

## 🏗️ CLOSED-LOOP MULTI-FREQUENCY ARCHITECTURE

```
                     ┌────────────────────────────────────────────────────────┐
                     │          NVIDIA GR00T N1.7 VLA POLICY                  │
                     │  • Task Prompt: "Pick up Coke/Pepsi can..."            │
                     │  • Ego-View Camera: 448x672 RGB (Step 02 Preprocess)   │
                     │  • 43 Joint States + Projected Gravity Vector          │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
                                                │ Inference @ 2.5 Hz - 5 Hz
                                                │ Action Chunk (T=40 x 78 dims):
                                                │ - 64-dim Latent Motion Token
                                                │ - 14-dim Hand Joints (7L + 7R)
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │          GEAR-SONIC WHOLE-BODY CONTROLLER              │
                     │  • Decodes 64-dim latent tokens @ 50 Hz (20 ms)        │
                     │  • Stabilizes CoM across 12 leg joints                 │
                     │  • Orients torso via 3 waist joints                    │
                     │  • Steers 14 arm joints towards the soda can           │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
                                                │ 43 Motor Position Commands @ 50 Hz
                                                │ (Low-Pass Smoothing Filter)
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                           MUJOCO 3D PHYSICS SIMULATION                                    │
│  • Robot: Unitree G1 43-DOF with multi-finger dexterous hands                             │
│  • Work table (X=0.48m, Z=0.68m) with Coca-Cola (red), Pepsi (blue), and sorting trays     │
│  • Dynamics stepping: mj_step @ 500 Hz (10 substeps per WBC tick)                         │
│  • Sensors: Camera ego_view (448x672), front_view, Pelvis IMU, Joint Encoders             │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 REPOSITORY STRUCTURE

```
GEAR-SONIC-GR00T-N1.7/
├── launch_vla_simulation.py         # Main entrypoint launching interactive 3D simulation
├── verify_pipeline.py               # Comprehensive automated test suite (Scene, WBC, VLA, Pipeline)
├── README.md                        # Master Documentation (English)
├── README_VN.md                     # Documentation (Vietnamese)
│
├── sim/                             # MuJoCo simulation package
│   ├── g1_scene.py                  # Simulation wrapper, camera rendering and state extraction
│   ├── preview_ego_view.png         # Snapshot of robot first-person camera view (448x672)
│   ├── preview_front_view.png       # Snapshot of third-person workspace camera view
│   └── unitree_g1/                  # Assets, STL meshes, and XML scene models:
│       ├── scene_pepsi_coke_sorting.xml  # Scene containing robot, table, cans, trays & cameras
│       └── g1_with_hands.xml        # G1 robot kinematics and ego_view head camera definition
│
├── bridge/                          # Integration bridge modules
│   ├── sonic_wbc_controller.py      # GEAR-SONIC WBC decoder (64 dims -> 43 motor targets @ 50Hz)
│   ├── gr00t_policy_client.py       # GR00T N1.7 VLA interface (ZMQ Remote + Standalone)
│   └── closed_loop_pipeline.py      # Closed-loop orchestrator (2.5Hz -> 50Hz -> 500Hz)
│
└── GR00T-WholeBodyControl/          # Upstream GEAR-SONIC repository from NVIDIA NVlabs
    ├── gear_sonic/                  # Core SONIC controllers
    ├── gear_sonic_deploy/           # Deployment scripts and ONNX configs
    ├── sonic_release/last.pt        # PyTorch SONIC model checkpoint (469 MB)
    └── bones_seed_smpl/             # SMPL motion dataset for training
```

---

## 🚀 QUICK START

### 1. Run Automated Verification (Headless)

Run the verification suite to validate all 4 components (MuJoCo, SONIC WBC, GR00T Policy, and Closed-Loop Pipeline):

```bash
cd c:\Users\Lenovo\Documents\GitHub\NVIDIA\GEAR-SONIC-GR00T-N1.7
python verify_pipeline.py
```

*Expected output*: All 4 test stages pass with `ALL TESTS PASSED!`.

---

### 2. Launch Interactive 3D Simulation Viewer

Launch the interactive MuJoCo 3D visualization:

```bash
# Default: Standalone mode, begins with Coca-Cola sorting task
python launch_vla_simulation.py

# Or launch directly with Pepsi sorting task:
python launch_vla_simulation.py --task 2
```

### 🎮 Keyboard & Mouse Shortcuts in 3D Viewer:
| Key | Action |
| :---: | :--- |
| **`[1]`** | Switch to task: **"Pick up the Coca-Cola can and place it in the red tray"** (Right arm) |
| **`[2]`** | Switch to task: **"Pick up the Pepsi can and place it in the blue tray"** (Left arm) |
| **`[SPACE]`** | Pause / Resume physics simulation |
| **`[R]`** | Reset robot and soda cans to initial positions |
| **`[ESC]`** | Exit simulation |
| **Left Click + Drag** | **Rotate 3D view** |
| **Right Click + Drag** | **Pan camera** |
| **Scroll Wheel** | **Zoom in / out** |
| **Ctrl + Right Click** | Apply external physical perturbation / drag objects in real time |

---

### 3. Connect to Live Isaac-GR00T PolicyServer (Optional)

If running the GPU-backed `PolicyServer` serving `nvidia/GR00T-N1.7-3B` from `Thanh-Hung---Isaac-GR00T`:

```bash
# Step 1: Start PolicyServer (from Thanh-Hung---Isaac-GR00T directory)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag UNITREE_G1_SONIC \
    --port 5550

# Step 2: Connect MuJoCo simulation to PolicyServer via ZMQ
python launch_vla_simulation.py --policy-mode remote --server-host localhost --server-port 5550
```

---

## 📊 INPUT / OUTPUT SPECIFICATIONS

### Robot Observation Space (46 dimensions):
- `left_leg` (6 dims): 3 hip (pitch, roll, yaw), 1 knee, 2 ankle (pitch, roll).
- `right_leg` (6 dims): 3 hip, 1 knee, 2 ankle.
- `waist` (3 dims): waist yaw, roll, pitch.
- `left_arm` (7 dims): 3 shoulder, 1 elbow, 3 wrist.
- `right_arm` (7 dims): 3 shoulder, 1 elbow, 3 wrist.
- `left_hand` (7 dims): 3 thumb, 2 middle, 2 index.
- `right_hand` (7 dims): 3 thumb, 2 index, 2 middle.
- `projected_gravity` (3 dims): Unit gravity vector $[g_x, g_y, g_z]$ projected into pelvis frame.

### VLA Action Space (78 dimensions x 40 steps):
- `motion_token` (64 dims): Latent whole-body action representation decoded by SONIC.
- `left_hand_joints` (7 dims): Position targets for left hand fingers.
- `right_hand_joints` (7 dims): Position targets for right hand fingers.
