"""
Module sim.g1_scene: Trình điều khiển môi trường mô phỏng MuJoCo Unitree G1.
Tác vụ: Robot G1 phân loại Lon Coca-Cola và Pepsi vào hai khay chuyên biệt.
Tích hợp camera ego_view góc nhìn robot (448x672) và trích xuất sensor phục vụ GR00T N1.7 VLA.
"""

import os
from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Any

# Cấu hình backend OpenGL offscreen mặc định là EGL nếu chưa được chỉ định
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np

DEFAULT_SCENE_XML = Path(__file__).parent / "unitree_g1" / "scene_pepsi_coke_sorting.xml"


def ready_key_id(model: mujoco.MjModel) -> int:
    """ID keyframe tư thế sẵn sàng (ưu tiên ready_table, sau đó stand); -1 nếu không có."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready_table")
    if key_id < 0:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    return key_id


def load_ready_ctrl(scene_xml_path: Optional[str] = None) -> np.ndarray:
    """Đọc 43 giá trị ctrl của tư thế sẵn sàng từ keyframe trong scene XML."""
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path or DEFAULT_SCENE_XML))
    key_id = ready_key_id(model)
    if key_id < 0:
        return np.zeros(model.nu, dtype=np.float32)
    return np.array(model.key_ctrl[key_id], dtype=np.float32)


class G1CanSortingSim:
    """Môi trường mô phỏng Unitree G1 trên MuJoCo hỗ trợ VLA GR00T N1.7 & SONIC WBC."""

    JOINT_GROUPS = {
        "left_leg": [0, 1, 2, 3, 4, 5],
        "right_leg": [6, 7, 8, 9, 10, 11],
        "waist": [12, 13, 14],
        "left_arm": [15, 16, 17, 18, 19, 20, 21],
        "left_hand": [22, 23, 24, 25, 26, 27, 28],
        "right_arm": [29, 30, 31, 32, 33, 34, 35],
        "right_hand": [36, 37, 38, 39, 40, 41, 42],
    }

    def __init__(
        self,
        scene_xml_path: Optional[str] = None,
        camera_width: int = 672,
        camera_height: int = 448,
    ):
        if scene_xml_path is None:
            scene_xml_path = str(DEFAULT_SCENE_XML)

        self.scene_path = Path(scene_xml_path)
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file scene XML tại: {self.scene_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        self.camera_width = camera_width
        self.camera_height = camera_height

        # Khởi tạo Offscreen Renderer (khởi tạo lười khi thực sự cần render)
        self.renderer = None

        # Lấy ID của các khớp và vật thể
        self.coke_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "coke_joint")
        self.pepsi_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pepsi_joint")
        self.coke_adr = self.model.jnt_qposadr[self.coke_jid]
        self.pepsi_adr = self.model.jnt_qposadr[self.pepsi_jid]

        # Khớp floating base và sensor của robot pelvis
        self.pelvis_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.pelvis_gyro_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-angular-velocity"
        )

        # Địa chỉ qpos/qvel của khớp do từng actuator điều khiển (thứ tự actuator khác thứ tự qpos ở bàn tay phải)
        act_jids = self.model.actuator_trnid[:, 0]
        self.act_qposadr = self.model.jnt_qposadr[act_jids].copy()
        self.act_dofadr = self.model.jnt_dofadr[act_jids].copy()

        # Cấu hình gắp cho từng lon: (lon, cổ tay, slice ctrl bàn tay, weld kẹp giữ, điểm gắp trong khung cổ tay)
        self.grasps = {
            "coke": self._make_grasp("coke_can", "right", slice(36, 43), "coke_grasp"),
            "pepsi": self._make_grasp("pepsi_can", "left", slice(22, 29), "pepsi_grasp"),
        }
        self.r_wrist_bid = self.grasps["coke"]["wrist_bid"]
        self.l_wrist_bid = self.grasps["pepsi"]["wrist_bid"]
        self.coke_bid = self.grasps["coke"]["can_bid"]
        self.pepsi_bid = self.grasps["pepsi"]["can_bid"]

        # Geom thân lon và material gốc, để đổi màu hai lon cho nhau (thử model có nhìn màu không)
        self._can_body_gids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                               for g in ("coke_body", "pepsi_body")]
        self._can_body_matids = self.model.geom_matid[self._can_body_gids].copy()
        self.can_colors_swapped = False

        # Tư thế sẵn sàng mặc định lấy từ ctrl của keyframe (ready_table, hoặc stand)
        self.stand_key_id = ready_key_id(self.model)
        self.default_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        if self.stand_key_id >= 0:
            self.default_ctrl = np.array(
                self.model.key_ctrl[self.stand_key_id], dtype=np.float32
            )

        # Khởi tạo trạng thái ban đầu
        self.reset()

    def reset(self):
        """Khôi phục toàn bộ trạng thái robot G1 và các vật thể về vị trí ban đầu."""
        # qpos0: pelvis và 2 lon ở đúng vị trí body khai báo trong XML, mọi weld kẹp giữ tắt
        mujoco.mj_resetData(self.model, self.data)

        # Đặt các khớp vào tư thế sẵn sàng suy ra từ ctrl của keyframe
        self.data.qpos[self.act_qposadr] = self.default_ctrl
        self.data.ctrl[:] = self.default_ctrl

        # Chạy forward kinematics
        mujoco.mj_forward(self.model, self.data)

    def _make_grasp(self, can_body: str, side: str, hand_ctrl: slice, weld_name: str) -> Dict[str, Any]:
        m = self.model
        wrist_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
        hand_bids = {
            b for b in range(m.nbody)
            if b == wrist_bid or m.body(b).name.startswith(f"{side}_hand")
        }
        return {
            "can_bid": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, can_body),
            "wrist_bid": wrist_bid,
            "hand_bids": hand_bids,
            "hand_ctrl": hand_ctrl,
            "eq_id": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name),
            # Tâm vùng nắm giữa ngón cái và các ngón còn lại (ngón cái nằm phía +y ở tay phải, -y ở tay trái)
            "grasp_point": np.array([0.13, 0.05 if side == "right" else -0.05, 0.0]),
        }

    def set_can_colors_swapped(self, swapped: bool):
        """Đổi material thân lon: lon Coca mang màu xanh Pepsi và ngược lại (vị trí, vật lý, khay giữ nguyên).

        Chỉ đổi model (không đụng MjData) nên giữ nguyên qua reset(); áp dụng cho cả viewer lẫn ảnh render_camera.
        """
        if min(self._can_body_gids) < 0:
            raise ValueError(f"Scene {self.scene_path.name} không có geom coke_body / pepsi_body để đổi màu lon")
        matids = self._can_body_matids[::-1] if swapped else self._can_body_matids
        self.model.geom_matid[self._can_body_gids] = matids
        self.can_colors_swapped = bool(swapped)

    def is_grasped(self, can: str) -> bool:
        return bool(self.data.eq_active[self.grasps[can]["eq_id"]])

    @property
    def is_coke_grasped(self) -> bool:
        return self.is_grasped("coke")

    @property
    def is_pepsi_grasped(self) -> bool:
        return self.is_grasped("pepsi")

    def is_holding_any(self) -> bool:
        return any(self.is_grasped(can) for can in self.grasps)

    def get_projected_gravity(self) -> np.ndarray:
        """Tính vector trọng trường g = [0, 0, -1] chiếu vào hệ quy chiếu của robot pelvis."""
        # Lấy ma trận quay của pelvis (3x3)
        rot_mat = self.data.xmat[self.pelvis_bid].reshape(3, 3)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        # Chiếu vào hệ pelvis: R^T * g
        proj_g = rot_mat.T @ world_gravity
        return proj_g.astype(np.float32)

    def get_state(self) -> Dict[str, np.ndarray]:
        """Trích xuất vector trạng thái khớp robot chuẩn định dạng UNITREE_G1_SONIC."""
        # 43 giá trị vị trí khớp điều khiển, theo thứ tự actuator
        ctrl_qpos = self.data.qpos[self.act_qposadr].copy()

        # Trích xuất vận tốc góc từ Gyro Pelvis (rad/s)
        if hasattr(self, "pelvis_gyro_sid") and self.pelvis_gyro_sid >= 0:
            adr = self.model.sensor_adr[self.pelvis_gyro_sid]
            base_ang_vel = self.data.sensordata[adr : adr + 3].copy().astype(np.float32)
        else:
            base_ang_vel = self.data.qvel[3:6].copy().astype(np.float32)

        # 43 giá trị vận tốc khớp, theo thứ tự actuator
        ctrl_qvel = self.data.qvel[self.act_dofadr].copy().astype(np.float32)
        body_joint_indices = [
            0, 1, 2, 3, 4, 5,       # left leg
            6, 7, 8, 9, 10, 11,     # right leg
            12, 13, 14,             # waist
            15, 16, 17, 18, 19, 20, 21,  # left arm
            29, 30, 31, 32, 33, 34, 35   # right arm
        ]
        body_joint_vel = ctrl_qvel[body_joint_indices]

        state_dict = {
            "left_leg": ctrl_qpos[self.JOINT_GROUPS["left_leg"]].astype(np.float32),
            "right_leg": ctrl_qpos[self.JOINT_GROUPS["right_leg"]].astype(np.float32),
            "waist": ctrl_qpos[self.JOINT_GROUPS["waist"]].astype(np.float32),
            "left_arm": ctrl_qpos[self.JOINT_GROUPS["left_arm"]].astype(np.float32),
            "right_arm": ctrl_qpos[self.JOINT_GROUPS["right_arm"]].astype(np.float32),
            "left_hand": ctrl_qpos[self.JOINT_GROUPS["left_hand"]].astype(np.float32),
            "right_hand": ctrl_qpos[self.JOINT_GROUPS["right_hand"]].astype(np.float32),
            "projected_gravity": self.get_projected_gravity(),
            "base_ang_vel": base_ang_vel,
            "ctrl_qvel": ctrl_qvel,
            "body_joint_vel": body_joint_vel,
        }
        return state_dict

    def get_flat_state_vector(self) -> np.ndarray:
        """Ghép toàn bộ trạng thái thành một vector 1D phẳng (46 dimensions)."""
        state = self.get_state()
        return np.concatenate([
            state["left_leg"],
            state["right_leg"],
            state["waist"],
            state["left_arm"],
            state["right_arm"],
            state["left_hand"],
            state["right_hand"],
            state["projected_gravity"],
        ])

    def render_camera(self, camera_name: str = "ego_view") -> np.ndarray:
        """Render khung hình RGB từ camera (kích thước chuẩn 448x672 cho GR00T N1.7)."""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model, height=self.camera_height, width=self.camera_width
            )
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render().copy()

    def get_observation(self) -> Dict[str, Any]:
        """Cung cấp toàn bộ quan sát đa phương thức cho VLA Policy."""
        return {
            "video": {
                "ego_view": self.render_camera("ego_view"),
                "front_view": self.render_camera("front_camera"),
            },
            "state": self.get_state(),
            "sim_time": self.data.time,
            "coke_pos": self.data.qpos[self.coke_adr : self.coke_adr + 3].copy(),
            "pepsi_pos": self.data.qpos[self.pepsi_adr : self.pepsi_adr + 3].copy(),
        }

    # Độ khép bàn tay (trung bình |góc| của 6 khớp ngón, bỏ thumb_0 xoay gốc ngón cái), đơn vị rad
    GRIP_CLOSED_RAD = 0.30
    GRIP_OPEN_RAD = 0.15
    # Khoảng cách tối đa từ điểm gắp trên bàn tay tới tâm lon để được coi là lon nằm trong lòng bàn tay
    GRASP_RADIUS = 0.05

    def _hand_touches(self, grasp: Dict[str, Any]) -> bool:
        for con in self.data.contact[: self.data.ncon]:
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            if (b1 == grasp["can_bid"] and b2 in grasp["hand_bids"]) or (
                b2 == grasp["can_bid"] and b1 in grasp["hand_bids"]
            ):
                return True
        return False

    def _update_grasp(self, grasp: Dict[str, Any], action_ctrl: np.ndarray):
        """Bật/tắt weld cổ tay-lon theo độ khép bàn tay và vị trí lon trong lòng bàn tay."""
        closure = float(np.mean(np.abs(action_ctrl[grasp["hand_ctrl"]][1:])))
        eq_id = grasp["eq_id"]

        if self.data.eq_active[eq_id]:
            if closure < self.GRIP_OPEN_RAD:
                self.data.eq_active[eq_id] = 0
            return

        if closure <= self.GRIP_CLOSED_RAD:
            return
        w_pos = self.data.xpos[grasp["wrist_bid"]]
        w_rot = self.data.xmat[grasp["wrist_bid"]].reshape(3, 3)
        c_pos = self.data.xpos[grasp["can_bid"]]
        if np.linalg.norm(w_pos + w_rot @ grasp["grasp_point"] - c_pos) > self.GRASP_RADIUS:
            return
        if not self._hand_touches(grasp):
            return

        # Giữ nguyên tư thế tương đối hiện tại của lon trong khung cổ tay (không dịch chuyển lon)
        w_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(w_quat_inv, self.data.xquat[grasp["wrist_bid"]])
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, w_quat_inv, self.data.xquat[grasp["can_bid"]])
        self.model.eq_data[eq_id, 0:3] = 0.0
        self.model.eq_data[eq_id, 3:6] = w_rot.T @ (c_pos - w_pos)
        self.model.eq_data[eq_id, 6:10] = rel_quat
        self.data.eq_active[eq_id] = 1

    def step(self, action_ctrl: np.ndarray, n_substeps: int = 10):
        """Áp dụng tín hiệu điều khiển tới 43 actuators và tính toán bước vật lý."""
        assert len(action_ctrl) == self.model.nu, f"Kích thước action {len(action_ctrl)} != nu {self.model.nu}"
        self.data.ctrl[:] = action_ctrl

        # 1. Kẹp giữ / nhả lon bằng weld equality (lực giữ và trọng lượng lon tác động thật lên robot)
        for grasp in self.grasps.values():
            self._update_grasp(grasp, action_ctrl)

        # 2. Bước lặp vật lý
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def close(self):
        """Giải phóng tài nguyên."""
        try:
            self.renderer.close()
        except Exception:
            pass
