"""
Module bridge.real_g1_adapter: Chuyển đổi giữa mô phỏng MuJoCo G1 và embodiment REAL_G1 của GR00T N1.7-3B
(tag nội bộ `real_g1_relative_eef_relative_joints`).

State gửi lên model (mỗi key một vector float32):
    left/right_wrist_eef_9d (9+9): vị trí cổ tay trong khung pelvis + rot6d (2 hàng đầu của ma trận quay)
    left/right_hand (7+7), left/right_arm (7+7), waist (3)
Action model trả về (đã đổi từ relative sang absolute ở server): các key như state, cộng thêm
base_height_command (1) và navigate_command (3). Ở đây chỉ dùng góc khớp tay, bàn tay, eo; bỏ qua EEF
(cần IK), base_height_command và navigate_command (chân giữ tư thế đứng).

Các quy ước dưới đây suy ra từ statistics.json của checkpoint, không có tài liệu chính thức:
  - Thứ tự 7 khớp bàn tay theo alphabet (index, middle, thumb): khớp với khoảng giá trị từng chiều.
  - Khung EEF = khung wrist_yaw_link xoay thêm EEF_ROT_OFFSET: rot6d của tư thế ready trong sim chỉ gần
    với giá trị trung bình của dataset khi có phép xoay này.
  - Tần số dữ liệu ~20 Hz: video dùng delta_indices [-20, 0], tương tự DROID dùng [-15, 0] ở 15 Hz (1 giây).
"""

from typing import Dict

import mujoco
import numpy as np

EMBODIMENT_TAG = "real_g1_relative_eef_relative_joints"

# Thứ tự khớp bàn tay trong dữ liệu REAL_G1 (khác thứ tự actuator của scene)
HAND_JOINT_ORDER = ["index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2"]
ARM_JOINT_ORDER = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw",
]
WAIST_JOINT_ORDER = ["waist_yaw", "waist_roll", "waist_pitch"]

# Khung EEF của dataset = khung wrist_yaw_link nhân phải với ma trận này (xoay -90° quanh trục y)
EEF_ROT_OFFSET = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

DEFAULT_ACTION_HZ = 20.0
# Độ lệch thời gian của khung hình quá khứ trong video, tính theo bước dữ liệu (delta_indices [-20, 0])
VIDEO_HISTORY_STEPS = 20

# Thứ tự key trong vector state/action của dataset (modality_keys của REAL_G1 trong checkpoint)
STATE_DIMS = {
    "left_wrist_eef_9d": 9, "right_wrist_eef_9d": 9, "left_hand": 7, "right_hand": 7,
    "left_arm": 7, "right_arm": 7, "waist": 3,
}
ACTION_DIMS = {**STATE_DIMS, "base_height_command": 1, "navigate_command": 3}
STATE_KEYS = list(STATE_DIMS)
ACTION_KEYS = list(ACTION_DIMS)
# Robot đứng yên tại chỗ: chiều cao thân ở mức đứng (giá trị q99/max của dữ liệu REAL_G1), không di chuyển
BASE_HEIGHT_COMMAND = 0.74
NAVIGATE_COMMAND = (0.0, 0.0, 0.0)


class RealG1Adapter:
    """Ánh xạ state/action REAL_G1 <-> 43 actuator của scene Unitree G1."""

    def __init__(self, model: mujoco.MjModel):
        self._model = model
        act_joint = {model.joint(int(model.actuator_trnid[i, 0])).name: i for i in range(model.nu)}

        def act_idx(names):
            return np.array([act_joint[f"{n}_joint"] for n in names], dtype=np.int64)

        self.arm_idx = {s: act_idx([f"{s}_{j}" for j in ARM_JOINT_ORDER]) for s in ("left", "right")}
        self.hand_idx = {s: act_idx([f"{s}_hand_{j}" for j in HAND_JOINT_ORDER]) for s in ("left", "right")}
        self.waist_idx = act_idx(WAIST_JOINT_ORDER)

        self.pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.wrist_bid = {
            s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link") for s in ("left", "right")
        }

        limited = model.actuator_ctrllimited.astype(bool)
        self.ctrl_lo = np.where(limited, model.actuator_ctrlrange[:, 0], -np.inf).astype(np.float32)
        self.ctrl_hi = np.where(limited, model.actuator_ctrlrange[:, 1], np.inf).astype(np.float32)

    def wrist_eef_9d(self, data: mujoco.MjData, side: str) -> np.ndarray:
        """Vị trí + rot6d của cổ tay trong khung pelvis."""
        p_rot = data.xmat[self.pelvis_bid].reshape(3, 3)
        w_rot = data.xmat[self.wrist_bid[side]].reshape(3, 3)
        pos = p_rot.T @ (data.xpos[self.wrist_bid[side]] - data.xpos[self.pelvis_bid])
        rot = p_rot.T @ w_rot @ EEF_ROT_OFFSET
        return np.concatenate([pos, rot[:2].reshape(6)]).astype(np.float32)

    def state(self, data: mujoco.MjData, act_qposadr: np.ndarray) -> Dict[str, np.ndarray]:
        """State REAL_G1 từ MjData (act_qposadr: địa chỉ qpos của khớp do từng actuator điều khiển)."""
        q = data.qpos[act_qposadr].astype(np.float32)
        state = {}
        for s in ("left", "right"):
            state[f"{s}_wrist_eef_9d"] = self.wrist_eef_9d(data, s)
            state[f"{s}_hand"] = q[self.hand_idx[s]]
            state[f"{s}_arm"] = q[self.arm_idx[s]]
        state["waist"] = q[self.waist_idx]
        return state

    def action_from_ctrl(
        self, data: mujoco.MjData, act_qposadr: np.ndarray, target_ctrl: np.ndarray, scratch: mujoco.MjData
    ) -> Dict[str, np.ndarray]:
        """Action REAL_G1 tuyệt đối từ 43 mục tiêu actuator: nghịch đảo của action_to_ctrl.

        Pose cổ tay mục tiêu là động học thuận của góc khớp mục tiêu (tay + eo), tính trên scratch (một MjData
        riêng) để không đụng tới trạng thái mô phỏng.
        """
        target_ctrl = np.asarray(target_ctrl, dtype=np.float32)
        scratch.qpos[:] = data.qpos
        scratch.qpos[act_qposadr] = target_ctrl
        mujoco.mj_kinematics(self._model, scratch)
        action = {}
        for s in ("left", "right"):
            action[f"{s}_wrist_eef_9d"] = self.wrist_eef_9d(scratch, s)
            action[f"{s}_hand"] = target_ctrl[self.hand_idx[s]]
            action[f"{s}_arm"] = target_ctrl[self.arm_idx[s]]
        action["waist"] = target_ctrl[self.waist_idx]
        action["base_height_command"] = np.array([BASE_HEIGHT_COMMAND], dtype=np.float32)
        action["navigate_command"] = np.array(NAVIGATE_COMMAND, dtype=np.float32)
        return action

    def action_to_ctrl(
        self,
        action: Dict[str, np.ndarray],
        base_ctrl: np.ndarray,
        action_hz: float,
        control_hz: float,
    ) -> np.ndarray:
        """Đổi action chunk REAL_G1 (mỗi key [T, D]) thành [N, 43] mục tiêu actuator ở tần số điều khiển.

        Khớp không có trong action (chân) giữ giá trị base_ctrl. Chunk được nội suy tuyến tính từ action_hz
        sang control_hz để giữ đúng tốc độ chuyển động của dữ liệu huấn luyện.
        """
        horizon = len(action["left_arm"])
        ctrl = np.repeat(np.asarray(base_ctrl, dtype=np.float32)[None], horizon, axis=0)
        for s in ("left", "right"):
            ctrl[:, self.arm_idx[s]] = action[f"{s}_arm"]
            ctrl[:, self.hand_idx[s]] = action[f"{s}_hand"]
        ctrl[:, self.waist_idx] = action["waist"]
        ctrl = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)

        # Nội suy tuyến tính: bước điều khiển k ứng với vị trí k * action_hz / control_hz trong chunk
        n_out = int(np.floor((horizon - 1) * control_hz / action_hz)) + 1
        pos = np.arange(n_out) * action_hz / control_hz
        i0 = np.minimum(pos.astype(np.int64), horizon - 1)
        i1 = np.minimum(i0 + 1, horizon - 1)
        w = (pos - i0)[:, None].astype(np.float32)
        return (1.0 - w) * ctrl[i0] + w * ctrl[i1]
