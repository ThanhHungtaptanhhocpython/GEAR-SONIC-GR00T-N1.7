"""
Module bridge.lower_body_policy: policy giữ thăng bằng cho chân và eo của NVIDIA GR00T-WholeBodyControl (decoupled WBC
dùng cho GR00T N1.5/N1.6), bản đứng yên `GR00T-WholeBodyControl-Balance.onnx`.

Tham chiếu (third_party/GR00T-WholeBodyControl, commit b042411):
  - decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py và resources/robots/g1/g1_gear_wbc.yaml: quan sát, scale, PD.
  - decoupled_wbc/control/policy/g1_decoupled_whole_body_policy.py: eo do policy chân điều khiển; tư thế eo mà thân trên
    muốn được đổi thành hướng thân (torso so với pelvis, bỏ yaw của pelvis) và đưa vào lệnh rpy của policy.

Quan sát mỗi bước 86 chiều, xếp chồng 6 bước (cũ -> mới, đầu episode đệm 0):
  [0:3] lệnh đi (vx, vy, yaw rate) * cmd_scale, [3] chiều cao pelvis, [4:7] rpy thân mong muốn,
  [7:10] vận tốc góc pelvis (khung pelvis) * 0.5, [10:13] trọng lực trong khung pelvis,
  [13:42] 29 góc khớp thân - góc mặc định, [42:71] 29 vận tốc khớp * 0.05, [71:86] action trước.
Ra 15 action (12 khớp chân + 3 khớp eo): mục tiêu = 0.25 * action + góc mặc định, bám bằng PD KP/KD của NVIDIA.

Chỉ dùng khi đứng yên (lệnh đi = 0). Tay và bàn tay vẫn do VLA điều khiển với gain cũ của sim.
"""

import collections
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

POLICY_DIR = (Path(__file__).resolve().parent.parent / "third_party" / "GR00T-WholeBodyControl" / "decoupled_wbc"
              / "sim2mujoco" / "resources" / "robots" / "g1" / "policy")

# g1_gear_wbc.yaml
KPS = np.array([150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250], dtype=np.float64)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5], dtype=np.float64)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0],
                          dtype=np.float32)
ANG_VEL_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.5, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
HISTORY_LEN, OBS_DIM = 6, 86
DEFAULT_HEIGHT_CMD = 0.74
JOINT_FRICTIONLOSS = 0.1  # g1_gear_wbc.xml; sim này để 0.3

# 29 khớp thân theo thứ tự của policy (chân, eo, tay trái, tay phải) = chỉ số actuator trong G1CanSortingSim
BODY_ACTUATORS = list(range(0, 22)) + list(range(29, 36))
NUM_ACTIONS = 15  # actuator 0..14: 12 khớp chân + 3 khớp eo


class BalancePolicy:
    """Điều khiển 12 khớp chân và 3 khớp eo để robot tự đứng vững trong khi VLA điều khiển tay."""

    def __init__(self, sim, height_cmd: float = DEFAULT_HEIGHT_CMD, onnx_path: Optional[str] = None):
        import onnxruntime as ort

        path = Path(onnx_path) if onnx_path else POLICY_DIR / "GR00T-WholeBodyControl-Balance.onnx"
        if not path.exists() or path.stat().st_size < 10000:
            raise FileNotFoundError(
                f"Không có file policy {path} (hoặc mới là con trỏ Git LFS). Tải từ repo NVlabs/GR00T-WholeBodyControl, "
                "xem MEMO_VLA_FINETUNE.md mục 5.2f.")
        self.session = ort.InferenceSession(str(path))
        self.input_name = self.session.get_inputs()[0].name
        self.sim = sim
        self.height_cmd = float(height_cmd)
        m = sim.model
        self.qadr = sim.act_qposadr[BODY_ACTUATORS]
        self.dadr = sim.act_dofadr[BODY_ACTUATORS]
        pelvis_joint = m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
        self.base_qadr = int(m.jnt_qposadr[pelvis_joint])
        self.base_dadr = int(m.jnt_dofadr[pelvis_joint])
        self.defaults29 = np.zeros(29, dtype=np.float32)
        self.defaults29[:NUM_ACTIONS] = DEFAULT_ANGLES
        # Hướng thân mong muốn từ góc eo: forward kinematics trên bản sao dữ liệu, pelvis ở gốc tọa độ
        self._fk = mujoco.MjData(m)
        self._waist_qadr = sim.act_qposadr[12:15]
        self._pelvis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._torso_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.reset()

    def apply_gains(self):
        """Đổi 15 actuator chân và eo sang PD của NVIDIA (actuator position của sim: gain = kp, bias = -kp*q - kd*qdot)."""
        m = self.sim.model
        for i in range(NUM_ACTIONS):
            m.actuator_gainprm[i, 0] = KPS[i]
            m.actuator_biasprm[i, 1] = -KPS[i]
            m.actuator_biasprm[i, 2] = -KDS[i]
            j = m.actuator_trnid[i, 0]
            m.dof_frictionloss[m.jnt_dofadr[j]] = JOINT_FRICTIONLOSS

    def reset(self):
        self.action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.history = collections.deque([np.zeros(OBS_DIM, dtype=np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)

    def init_pose(self):
        """Đặt 12 khớp chân vào tư thế mặc định của policy (gối gập) và hạ pelvis để bàn chân vẫn chạm sàn, giữ nguyên
        x, y của pelvis. Gọi ngay sau sim.reset(): bắt đầu từ chân thẳng thì lúc chuyển tư thế bàn chân trượt ~3 cm và
        pelvis lùi ~7 cm khỏi bàn."""
        m, d = self.sim.model, self.sim.data
        feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")]
        mujoco.mj_kinematics(m, d)
        z_before = min(d.xpos[b][2] for b in feet)
        d.qpos[self.qadr[:12]] = DEFAULT_ANGLES[:12]
        mujoco.mj_kinematics(m, d)
        d.qpos[self.base_qadr + 2] -= min(d.xpos[b][2] for b in feet) - z_before
        d.qvel[:] = 0.0
        d.ctrl[:12] = DEFAULT_ANGLES[:12]
        mujoco.mj_forward(m, d)

    def settle(self, upper_ctrl: np.ndarray, seconds: float = 2.0, substeps: int = 10):
        """Cho robot tự đứng vững với thân trên ở `upper_ctrl` (43 actuator) rồi dời cả robot về x, y ban đầu của pelvis.

        Ở tư thế ready (eo cúi, hai tay đưa ra trước) policy đẩy pelvis lùi ~7 cm để giữ trọng tâm trên bàn chân; không dời
        lại thì lon ra ngoài tầm với so với dữ liệu thu với chân khoá cứng."""
        m, d = self.sim.model, self.sim.data
        xy0 = d.qpos[self.base_qadr:self.base_qadr + 2].copy()
        target = np.asarray(upper_ctrl, dtype=np.float32).copy()
        for _ in range(int(seconds * 50)):
            target[:NUM_ACTIONS] = self(target[12:15])
            self.sim.step(target, n_substeps=substeps)
        d.qpos[self.base_qadr:self.base_qadr + 2] = xy0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    def torso_rpy(self, waist_target: np.ndarray) -> np.ndarray:
        """rpy (quy ước Pinocchio: R = Rz(y) Ry(p) Rx(r)) của torso so với pelvis khi eo ở góc `waist_target`."""
        m, d = self.sim.model, self._fk
        d.qpos[:] = m.qpos0
        d.qpos[self._waist_qadr] = waist_target
        mujoco.mj_kinematics(m, d)
        r = d.xmat[self._pelvis_bid].reshape(3, 3).T @ d.xmat[self._torso_bid].reshape(3, 3)
        return np.array([np.arctan2(r[2, 1], r[2, 2]), np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0])),
                         np.arctan2(r[1, 0], r[0, 0])], dtype=np.float32)

    def observation(self, waist_target: np.ndarray) -> np.ndarray:
        d = self.sim.data
        quat = d.qpos[self.base_qadr + 3:self.base_qadr + 7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[3] = self.height_cmd  # [0:3] lệnh đi = 0
        obs[4:7] = self.torso_rpy(waist_target)
        obs[7:10] = d.qvel[self.base_dadr + 3:self.base_dadr + 6] * ANG_VEL_SCALE  # free joint: khung pelvis
        obs[10:13] = rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        obs[13:42] = d.qpos[self.qadr] - self.defaults29
        obs[42:71] = d.qvel[self.dadr] * DOF_VEL_SCALE
        obs[71:86] = self.action
        return obs

    def __call__(self, waist_target: np.ndarray) -> np.ndarray:
        """Chạy policy một bước (50 Hz) từ trạng thái sim hiện tại; trả về mục tiêu 15 actuator chân + eo."""
        self.history.append(self.observation(np.asarray(waist_target, dtype=np.float64)))
        obs = np.concatenate(self.history)[None].astype(np.float32)
        self.action = self.session.run(None, {self.input_name: obs})[0][0].astype(np.float32)
        return self.action * ACTION_SCALE + DEFAULT_ANGLES
