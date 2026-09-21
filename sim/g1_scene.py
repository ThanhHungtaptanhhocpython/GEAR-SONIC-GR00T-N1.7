"""
Module sim.g1_scene: Trình điều khiển môi trường mô phỏng MuJoCo Unitree G1.
Tác vụ: Robot G1 phân loại Lon Coca-Cola và Pepsi vào hai khay chuyên biệt.
Tích hợp camera ego_view góc nhìn robot (448x672) và trích xuất sensor phục vụ GR00T N1.7 VLA.
"""

from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Any
import mujoco
import numpy as np


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
            scene_xml_path = str(
                Path(__file__).parent / "unitree_g1" / "scene_pepsi_coke_sorting.xml"
            )

        self.scene_path = Path(scene_xml_path)
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file scene XML tại: {self.scene_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        self.camera_width = camera_width
        self.camera_height = camera_height

        # Khởi tạo Offscreen Renderer cho camera ego_view và front_camera
        self.renderer = mujoco.Renderer(
            self.model, height=self.camera_height, width=self.camera_width
        )

        # Lấy ID của các khớp và vật thể
        self.coke_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "coke_joint")
        self.pepsi_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pepsi_joint")
        self.coke_adr = self.model.jnt_qposadr[self.coke_jid]
        self.pepsi_adr = self.model.jnt_qposadr[self.pepsi_jid]

        # Khớp floating base của robot pelvis
        self.pelvis_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        # Lưu tư thế đứng mặc định
        self.stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.default_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        if self.stand_key_id >= 0:
            self.default_ctrl = np.array(
                self.model.key_ctrl[self.stand_key_id], dtype=np.float32
            )

        # Lưu vị trí ban đầu của 2 lon nước
        self.coke_init_qpos = self.data.qpos[self.coke_adr : self.coke_adr + 7].copy()
        self.pepsi_init_qpos = self.data.qpos[self.pepsi_adr : self.pepsi_adr + 7].copy()

        # Khởi tạo trạng thái ban đầu
        self.reset()

    def reset(self):
        """Khôi phục toàn bộ trạng thái robot G1 và các vật thể về vị trí ban đầu."""
        mujoco.mj_resetData(self.model, self.data)

        if self.stand_key_id >= 0:
            robot_dofs = len(self.model.key_ctrl[self.stand_key_id])
            self.data.qpos[:robot_dofs] = self.model.key_qpos[self.stand_key_id][:robot_dofs]
            self.data.ctrl[:robot_dofs] = self.model.key_ctrl[self.stand_key_id]

        # Đặt lại vị trí 2 lon nước
        self.data.qpos[self.coke_adr : self.coke_adr + 7] = self.coke_init_qpos
        self.data.qpos[self.pepsi_adr : self.pepsi_adr + 7] = self.pepsi_init_qpos

        # Chạy forward kinematics
        mujoco.mj_forward(self.model, self.data)

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
        # 43 giá trị vị trí khớp điều khiển
        ctrl_qpos = self.data.qpos[7:50].copy()  # Bỏ qua 7 dof floating base (pos + quat)

        state_dict = {
            "left_leg": ctrl_qpos[self.JOINT_GROUPS["left_leg"]].astype(np.float32),
            "right_leg": ctrl_qpos[self.JOINT_GROUPS["right_leg"]].astype(np.float32),
            "waist": ctrl_qpos[self.JOINT_GROUPS["waist"]].astype(np.float32),
            "left_arm": ctrl_qpos[self.JOINT_GROUPS["left_arm"]].astype(np.float32),
            "right_arm": ctrl_qpos[self.JOINT_GROUPS["right_arm"]].astype(np.float32),
            "left_hand": ctrl_qpos[self.JOINT_GROUPS["left_hand"]].astype(np.float32),
            "right_hand": ctrl_qpos[self.JOINT_GROUPS["right_hand"]].astype(np.float32),
            "projected_gravity": self.get_projected_gravity(),
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

    def step(self, action_ctrl: np.ndarray, n_substeps: int = 10):
        """Áp dụng tín hiệu điều khiển tới 43 actuators và tính toán bước vật lý."""
        assert len(action_ctrl) == self.model.nu, f"Kích thước action {len(action_ctrl)} != nu {self.model.nu}"
        self.data.ctrl[:] = action_ctrl
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def close(self):
        """Giải phóng tài nguyên."""
        self.renderer.close()
