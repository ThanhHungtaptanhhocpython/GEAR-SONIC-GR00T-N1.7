"""
Module bridge.closed_loop_pipeline: Vòng lặp điều khiển kín đa tần số (Multi-frequency Closed-Loop Pipeline).
Kết nối:
  1. MuJoCo Simulation (500 Hz physics, offscreen camera ego_view)
  2. GR00T N1.7 VLA Policy (~2.5 Hz - 5 Hz, dự đoán 40-step action chunk)
  3. GEAR-SONIC Whole-Body Controller (50 Hz decoding, giữ cân bằng và điều phối toàn thân)
Với model GR00T thật (embodiment REAL_G1), action là góc khớp tay/bàn tay/eo đưa thẳng vào actuator,
bỏ qua bộ giải mã motion token.
"""

import collections
import contextlib
import time
from typing import Callable, ContextManager, Dict, Optional, Tuple, Any
import numpy as np

from sim.g1_scene import G1CanSortingSim
from bridge.sonic_wbc_controller import SonicWBCController
from bridge.gr00t_policy_client import Gr00tN17Policy
from bridge.real_g1_adapter import DEFAULT_ACTION_HZ, VIDEO_HISTORY_STEPS, RealG1Adapter

# Số bước 50 Hz trộn từ mục tiêu điều khiển hiện tại sang đường về ready khi bắt đầu giữ (hold_ready)
HOLD_BLEND_STEPS = 50


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
        vla_action_hz: float = DEFAULT_ACTION_HZ,  # Tần số dữ liệu của action chunk REAL_G1
        server_timeout_ms: int = 3000,
        record_vla_path: Optional[str] = None,  # File .npz ghi state/action/ảnh của mỗi lần gọi model GR00T thật
    ):
        self.task_prompt = task_prompt
        self.policy_replan_interval = policy_replan_interval
        self.substeps_per_wbc = substeps_per_wbc
        self.vla_action_hz = vla_action_hz

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
            timeout_ms=server_timeout_ms,
        )
        self.g1_adapter = RealG1Adapter(self.sim.model)

        # Tần số điều khiển WBC (50 Hz) và độ lệch của khung hình quá khứ gửi cho VLA, tính theo bước điều khiển
        self.control_hz = 1.0 / (self.sim.model.opt.timestep * substeps_per_wbc)
        self.video_history_offset = int(round(VIDEO_HISTORY_STEPS * self.control_hz / vla_action_hz))
        # Khung hình ego_view đã render ở các lần replan: (bước điều khiển, ảnh)
        self.ego_history: collections.deque = collections.deque(maxlen=16)
        self.record_vla_path = record_vla_path
        self.vla_log: list = []

        # Trạng thái vòng lặp
        self.current_action_chunk: Optional[Dict[str, np.ndarray]] = None
        self.chunk_step_idx = 0
        self.total_steps_executed = 0  # Tổng số bước 50 Hz kể từ lần reset (không bị đổi nhiệm vụ làm lại từ đầu)
        self.task_step = 0             # Số bước kể từ khi bắt đầu nhiệm vụ hiện tại (quyết định tiến trình quỹ đạo)
        self.pending_prompt: Optional[str] = None
        self.render_failures = 0
        self.is_paused = False
        # Giữ tư thế ready (không gọi model) sau khi xong nhiệm vụ: None khi không giữ
        self.hold_from: Optional[np.ndarray] = None
        self.hold_path: list = []
        self.hold_step = 0

    def set_task_prompt(self, new_prompt: str):
        """Thay đổi câu lệnh nhiệm vụ ngôn ngữ trong thời gian thực.

        Chọn lại đúng nhiệm vụ đang chạy thì bỏ qua (chỉ chạy lại khi nhiệm vụ đó đã xong). Nếu robot đang
        cầm lon, nhiệm vụ mới được hoãn tới khi xong lần gắp-đặt hiện tại để lon được đặt vào khay trước,
        không bị thả rơi giữa đường.
        """
        if new_prompt == self.task_prompt:
            if self.pending_prompt is not None:
                self.pending_prompt = None
                print(f"\n[VLA Pipeline] Hủy chuyển nhiệm vụ, tiếp tục: '{self.task_prompt}'")
            elif self.task_done:
                self._apply_task_prompt(new_prompt)
            return

        if self.sim.is_holding_any():
            self.pending_prompt = new_prompt
            print(f"\n[VLA Pipeline] Robot đang cầm lon, sẽ chuyển sang '{new_prompt}' khi đặt xong lon.")
            return

        self._apply_task_prompt(new_prompt)

    @property
    def task_done(self) -> bool:
        """Đã chạy hết một lần gắp-đặt của nhiệm vụ hiện tại."""
        return self.task_step >= self.policy.CYCLE_STEPS

    def _apply_task_prompt(self, new_prompt: str):
        self.task_prompt = new_prompt
        self.pending_prompt = None
        self.chunk_step_idx = 0
        self.task_step = 0
        self.current_action_chunk = None
        self.hold_from = None
        print(f"\n[VLA Pipeline] Đã chuyển đổi nhiệm vụ sang: '{self.task_prompt}'")

    def reset(self):
        """Khôi phục toàn bộ hệ thống về trạng thái ban đầu."""
        self.sim.reset()
        self.wbc.reset()
        self.ego_history.clear()
        self.current_action_chunk = None
        self.chunk_step_idx = 0
        self.total_steps_executed = 0
        self.task_step = 0
        self.hold_from = None
        if self.pending_prompt is not None:
            self._apply_task_prompt(self.pending_prompt)

    @property
    def holding_ready(self) -> bool:
        return self.hold_from is not None

    def hold_ready(self, path: Optional[list] = None):
        """Ngừng gọi model và đi theo `path` (mục tiêu actuator 50 Hz, ví dụ PickPlacePlanner.return_to_ready) rồi giữ
        mục tiêu cuối, tới khi reset hoặc đổi nhiệm vụ. Không có path thì nội suy thẳng về tư thế ready. Đầu đường đi
        được trộn dần từ mục tiêu hiện tại trong HOLD_BLEND_STEPS bước."""
        self.hold_from = self.wbc.last_ctrl.copy()
        self.hold_path = list(path) if path else [self.sim.default_ctrl.copy()]
        self.hold_step = 0

    def step_50hz(self, physics_lock: Optional[Callable[[], ContextManager]] = None) -> Dict[str, Any]:
        """Thực thi một chu kỳ điều khiển 50 Hz (20 ms).

        Args:
            physics_lock: Hàm trả về context manager (ví dụ viewer.lock) chỉ được giữ trong lúc ghi vào
                MjData. Suy luận policy, truy vấn mạng và render camera chạy bên ngoài lock.

        Returns:
            Dict chứa góc khớp điều khiển và thông tin trạng thái lon nước.
        """
        lock = physics_lock or contextlib.nullcontext

        if self.is_paused:
            # Nếu tạm dừng, duy trì vị trí hiện tại
            with lock():
                self.sim.step(self.wbc.last_ctrl, n_substeps=self.substeps_per_wbc)
            return {"paused": True}

        # Nhiệm vụ bị hoãn vì đang cầm lon được áp dụng khi nhiệm vụ hiện tại xong
        if self.pending_prompt is not None and self.task_done:
            self._apply_task_prompt(self.pending_prompt)

        if self.hold_from is not None:
            w = min(self.hold_step / HOLD_BLEND_STEPS, 1.0)
            w = w * w * (3.0 - 2.0 * w)
            target = self.hold_path[min(self.hold_step, len(self.hold_path) - 1)]
            motor_targets = self.wbc.track_joint_targets((1.0 - w) * self.hold_from + w * target)
            with lock():
                self.sim.step(motor_targets, n_substeps=self.substeps_per_wbc)
            self.hold_step += 1
            self.total_steps_executed += 1
            return {
                "paused": False,
                "holding": True,
                "sim_time": self.sim.data.time,
                "motor_targets": motor_targets,
                "coke_pos": self.sim.data.qpos[self.sim.coke_adr : self.sim.coke_adr + 3].copy(),
                "pepsi_pos": self.sim.data.qpos[self.sim.pepsi_adr : self.sim.pepsi_adr + 3].copy(),
                "chunk_idx": self.chunk_step_idx,
                "total_steps": self.total_steps_executed,
            }

        # 1. Thu nhận trạng thái khớp @ 50 Hz (siêu nhanh, không tốn chi phí render)
        robot_state = self.sim.get_state()

        # 2. Kiểm tra chu kỳ suy luận VLA (Re-planning @ 2.5 - 5 Hz)
        need_replan = (
            self.current_action_chunk is None
            or self.chunk_step_idx >= self.policy_replan_interval
            or self.chunk_step_idx >= len(self.current_action_chunk[self._chunk_key(self.current_action_chunk)])
        )

        if need_replan:
            new_chunk = self._predict_chunk(robot_state)
            if new_chunk is None:
                # Server lỗi / không phản hồi: giữ tư thế hiện tại tới lần replan sau
                new_chunk = self._hold_chunk()

            # Cơ chế hòa trộn mượt RTC Smooth Inpainting với chunk trước đó (chỉ khi cùng không gian action)
            old_chunk = self.current_action_chunk
            key = self._chunk_key(new_chunk)
            if old_chunk is not None and self._chunk_key(old_chunk) == key:
                # Trộn mượt 4 bước giao thoa
                old_len = len(old_chunk[key])
                overlap = min(4, old_len - self.chunk_step_idx, len(new_chunk[key]))
                for k in range(overlap):
                    blend_weight = np.cos(np.pi * k / (2.0 * overlap)) ** 2
                    new_chunk[key][k] = (
                        blend_weight * old_chunk[key][self.chunk_step_idx + k]
                        + (1.0 - blend_weight) * new_chunk[key][k]
                    )

            self.current_action_chunk = new_chunk
            self.chunk_step_idx = 0

        # 3-4. Lấy hành động tại bước hiện tại trong chunk và đổi thành 43 motor targets
        chunk = self.current_action_chunk
        if chunk["action_space"] == "joint_targets":
            # Góc khớp tuyệt đối từ model GR00T thật: bỏ qua bộ giải mã motion token
            motor_targets = self.wbc.track_joint_targets(chunk["joint_ctrl"][self.chunk_step_idx])
        else:
            # GEAR-SONIC WBC giải mã 64-dim token thành 43 motor targets
            motor_targets = self.wbc.decode(
                motion_token=chunk["motion_token"][self.chunk_step_idx],
                left_hand_joints=chunk["left_hand_joints"][self.chunk_step_idx],
                right_hand_joints=chunk["right_hand_joints"][self.chunk_step_idx],
                robot_state=robot_state,
            )

        # 5. MuJoCo bước 10 substeps (10 x 0.002s = 0.02s ~ 50 Hz)
        with lock():
            self.sim.step(motor_targets, n_substeps=self.substeps_per_wbc)

        self.chunk_step_idx += 1
        self.total_steps_executed += 1
        self.task_step += 1

        return {
            "paused": False,
            "sim_time": self.sim.data.time,
            "motor_targets": motor_targets,
            "coke_pos": self.sim.data.qpos[self.sim.coke_adr : self.sim.coke_adr + 3].copy(),
            "pepsi_pos": self.sim.data.qpos[self.sim.pepsi_adr : self.sim.pepsi_adr + 3].copy(),
            "chunk_idx": self.chunk_step_idx,
            "total_steps": self.total_steps_executed,
        }

    @staticmethod
    def _chunk_key(chunk: Dict[str, Any]) -> str:
        return "joint_ctrl" if chunk["action_space"] == "joint_targets" else "motion_token"

    def _hold_chunk(self) -> Dict[str, Any]:
        """Chunk giữ nguyên mục tiêu điều khiển hiện tại trong một chu kỳ replan."""
        return {
            "action_space": "joint_targets",
            "joint_ctrl": np.repeat(self.wbc.last_ctrl[None], self.policy_replan_interval, axis=0),
        }

    def _ego_frames(self) -> Optional[np.ndarray]:
        """Render ego_view hiện tại, ghép với khung hình quá khứ gần nhất với video_history_offset bước trước.

        Trả về [2, H, W, 3] theo thứ tự cũ -> mới (delta_indices [-20, 0] của REAL_G1); None nếu không render được.
        """
        try:
            frame = self.sim.render_camera("ego_view")
        except Exception as e:
            self.render_failures += 1
            if self.render_failures == 1 or self.render_failures % 50 == 0:
                print(
                    f"\n⚠️ [VLA Pipeline] Không render được camera ego_view ({e!r}), "
                    f"{self.render_failures} lần. Policy dùng quỹ đạo cục bộ thay cho server."
                )
            return None
        now = self.total_steps_executed
        self.ego_history.append((now, frame))
        # Đầu episode chưa có khung hình đủ cũ thì dùng khung hình cũ nhất hiện có
        _, past = min(self.ego_history, key=lambda item: abs(item[0] - (now - self.video_history_offset)))
        return np.stack([past, frame])

    def _predict_chunk(self, robot_state: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
        """Gọi VLA policy và đổi output REAL_G1 thành chunk mục tiêu actuator ở tần số điều khiển."""
        ego_frames = None
        g1_state = None
        # Chỉ render camera và tính state REAL_G1 khi chạy remote (cần gửi tới Policy Server)
        if self.policy.mode == "remote":
            ego_frames = self._ego_frames()
            g1_state = self.g1_adapter.state(self.sim.data, self.sim.act_qposadr)

        # Tiến trình thời gian của quỹ đạo mẫu (0.0 -> 1.0, giữ 1.0 khi nhiệm vụ đã xong)
        progress = min(self.task_step / self.policy.CYCLE_STEPS, 1.0)
        chunk = self.policy.predict_action_chunk(
            task_description=self.task_prompt,
            ego_view_image=ego_frames,
            robot_state=robot_state,
            step_progress=progress,
            g1_state=g1_state,
        )
        if chunk is None or chunk["action_space"] != "real_g1":
            return chunk
        if self.record_vla_path is not None:
            self.vla_log.append({
                "step": self.total_steps_executed,
                "prompt": self.task_prompt,
                "state": g1_state,
                "action": chunk["g1_action"],
                "frame": ego_frames[-1][::2, ::2].copy(),  # nửa độ phân giải cho nhẹ file
            })
        return {
            "action_space": "joint_targets",
            "joint_ctrl": self.g1_adapter.action_to_ctrl(
                chunk["g1_action"], self.sim.default_ctrl, self.vla_action_hz, self.control_hz
            ),
            "g1_action": chunk["g1_action"],
        }

    def save_vla_log(self):
        """Ghi các lần gọi model GR00T thật ra file .npz (mỗi key state_*/action_* xếp chồng theo lần gọi)."""
        if self.record_vla_path is None or not self.vla_log:
            return
        log = self.vla_log
        arrays = {
            "step": np.array([e["step"] for e in log]),
            "prompt": np.array([e["prompt"] for e in log]),
            "frame": np.stack([e["frame"] for e in log]),
        }
        for k in log[0]["state"]:
            arrays[f"state_{k}"] = np.stack([e["state"][k] for e in log])
        for k in log[0]["action"]:
            arrays[f"action_{k}"] = np.stack([e["action"][k] for e in log])
        np.savez_compressed(self.record_vla_path, **arrays)
        print(f"\n[VLA Pipeline] Đã ghi {len(log)} lần gọi model vào {self.record_vla_path}")

    def close(self):
        """Giải phóng tài nguyên."""
        self.save_vla_log()
        self.sim.close()
