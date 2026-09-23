"""
Module bridge.closed_loop_pipeline: Vòng lặp điều khiển kín đa tần số (Multi-frequency Closed-Loop Pipeline).
Kết nối:
  1. MuJoCo Simulation (500 Hz physics, offscreen camera ego_view)
  2. GR00T N1.7 VLA Policy (~2.5 Hz - 5 Hz, dự đoán 40-step action chunk)
  3. GEAR-SONIC Whole-Body Controller (50 Hz decoding, giữ cân bằng và điều phối toàn thân)
"""

import time
from typing import Dict, Optional, Tuple, Any
import numpy as np

from sim.g1_scene import G1CanSortingSim
from bridge.sonic_wbc_controller import SonicWBCController
from bridge.gr00t_policy_client import Gr00tN17Policy


class ClosedLoopVLAPipeline:
    """Bộ điều phối vòng lặp khép kín VLA + WBC + MuJoCo cho Unitree G1."""

    def __init__(
        self,
        task_prompt: str = "Pick up the Coca-Cola can and place it in the red tray",
        policy_mode: str = "standalone",
        server_host: str = "localhost",
        server_port: int = 5550,
        wbc_smoothing: float = 0.35,
        wbc_mode: str = "kinematic",
        substeps_per_wbc: int = 10,
        policy_replan_interval: int = 15,  # Số bước WBC (50Hz) giữa các lần gọi VLA (15 bước = 0.3s ~ 3.33Hz)
    ):
        self.task_prompt = task_prompt
        self.policy_replan_interval = policy_replan_interval
        self.substeps_per_wbc = substeps_per_wbc

        # 1. Khởi tạo môi trường MuJoCo G1
        self.sim = G1CanSortingSim()

        # 2. Khởi tạo bộ giải mã GEAR-SONIC WBC
        self.wbc = SonicWBCController(
            default_stand_ctrl=self.sim.default_ctrl,
            smoothing_alpha=wbc_smoothing,
            mode=wbc_mode,
        )

        # 3. Khởi tạo mô hình VLA Policy GR00T N1.7
        self.policy = Gr00tN17Policy(
            mode=policy_mode,
            server_host=server_host,
            server_port=server_port,
        )

        # Trạng thái vòng lặp
        self.current_action_chunk: Optional[Dict[str, np.ndarray]] = None
        self.chunk_step_idx = 0
        self.total_steps_executed = 0
        self.is_paused = False

    def set_task_prompt(self, new_prompt: str):
        """Thay đổi câu lệnh nhiệm vụ ngôn ngữ trong thời gian thực."""
        self.task_prompt = new_prompt
        self.chunk_step_idx = 0
        self.current_action_chunk = None
        print(f"\n[VLA Pipeline] Đã chuyển đổi nhiệm vụ sang: '{self.task_prompt}'")

    def reset(self):
        """Khôi phục toàn bộ hệ thống về trạng thái ban đầu."""
        self.sim.reset()
        self.wbc.reset()
        self.current_action_chunk = None
        self.chunk_step_idx = 0
        self.total_steps_executed = 0

    def step_50hz(self) -> Dict[str, Any]:
        """Thực thi một chu kỳ điều khiển 50 Hz (20 ms).
        
        Returns:
            Dict chứa ảnh camera, góc khớp điều khiển, và thông tin trạng thái lon nước.
        """
        if self.is_paused:
            # Nếu tạm dừng, duy trì vị trí hiện tại
            self.sim.step(self.wbc.last_ctrl, n_substeps=self.substeps_per_wbc)
            return {"paused": True}

        # 1. Thu nhận trạng thái khớp @ 50 Hz (siêu nhanh, không tốn chi phí render)
        robot_state = self.sim.get_state()

        # 2. Kiểm tra chu kỳ suy luận VLA (Re-planning @ 2.5 - 5 Hz)
        need_replan = (
            self.current_action_chunk is None
            or self.chunk_step_idx >= self.policy_replan_interval
            or self.chunk_step_idx >= self.policy.action_horizon
        )

        ego_img = None
        if need_replan:
            # Render camera ego_view góc nhìn robot (448x672) khi VLA cần suy luận
            ego_img = self.sim.render_camera("ego_view")
            # Tính tiến trình thời gian của nhiệm vụ (0.0 -> 1.0)
            progress = (self.total_steps_executed % 350) / 350.0

            # Dự đoán Action Chunk 40 bước từ GR00T N1.7
            new_chunk = self.policy.predict_action_chunk(
                task_description=self.task_prompt,
                ego_view_image=ego_img,
                robot_state=robot_state,
                step_progress=progress,
            )

            # Cơ chế hòa trộn mượt RTC Smooth Inpainting với chunk trước đó
            if self.current_action_chunk is not None and self.chunk_step_idx < self.policy.action_horizon:
                # Trộn mượt 4 bước giao thoa
                overlap = min(4, self.policy.action_horizon - self.chunk_step_idx)
                for k in range(overlap):
                    blend_weight = np.cos(np.pi * k / (2.0 * overlap)) ** 2
                    old_idx = self.chunk_step_idx + k
                    if old_idx < self.policy.action_horizon:
                        new_chunk["motion_token"][k] = (
                            blend_weight * self.current_action_chunk["motion_token"][old_idx]
                            + (1.0 - blend_weight) * new_chunk["motion_token"][k]
                        )

            self.current_action_chunk = new_chunk
            self.chunk_step_idx = 0

        # 3. Lấy hành động tại bước thời gian hiện tại trong chunk
        curr_token = self.current_action_chunk["motion_token"][self.chunk_step_idx]
        curr_l_hand = self.current_action_chunk["left_hand_joints"][self.chunk_step_idx]
        curr_r_hand = self.current_action_chunk["right_hand_joints"][self.chunk_step_idx]

        # 4. GEAR-SONIC WBC giải mã 64-dim token thành 43 motor targets
        motor_targets = self.wbc.decode(
            motion_token=curr_token,
            left_hand_joints=curr_l_hand,
            right_hand_joints=curr_r_hand,
            robot_state=robot_state,
        )

        # 5. MuJoCo bước 10 substeps (10 x 0.002s = 0.02s ~ 50 Hz)
        self.sim.step(motor_targets, n_substeps=self.substeps_per_wbc)

        self.chunk_step_idx += 1
        self.total_steps_executed += 1

        return {
            "paused": False,
            "sim_time": self.sim.data.time,
            "motor_targets": motor_targets,
            "coke_pos": self.sim.data.qpos[self.sim.coke_adr : self.sim.coke_adr + 3].copy(),
            "pepsi_pos": self.sim.data.qpos[self.sim.pepsi_adr : self.sim.pepsi_adr + 3].copy(),
            "ego_view": ego_img,
            "chunk_idx": self.chunk_step_idx,
            "total_steps": self.total_steps_executed,
        }

    def close(self):
        """Giải phóng tài nguyên."""
        self.sim.close()
