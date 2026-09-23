"""
Module bridge.sonic_wbc_controller: Bộ điều khiển toàn thân GEAR-SONIC WBC cho robot Unitree G1.
Tiếp nhận 64-dimensional latent motion tokens từ GR00T N1.7 VLA Policy và giải mã
thành mục tiêu vị trí 43 khớp (29 khớp toàn thân + 14 khớp hai bàn tay) ở tần số 50 Hz.
Hỗ trợ giải mã nơ-ron học sâu qua model_decoder.onnx chính thức từ NVIDIA GEAR-SONIC
kèm cơ chế dự phòng Kinematic Heuristic Decoder.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np


class SonicWBCController:
    """Bộ giải mã toàn thân GEAR-SONIC Whole-Body Controller cho Unitree G1."""

    # 43 Actuators của Unitree G1 theo đúng thứ tự trong scene_pepsi_coke_sorting.xml
    ACTUATOR_NAMES = [
        # Chân trái (0-5)
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        # Chân phải (6-11)
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        # Thân / Hông (12-14)
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        # Tay trái (15-21)
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        # Bàn tay trái (22-28)
        "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        "left_hand_index_0_joint", "left_hand_index_1_joint",
        # Tay phải (29-35)
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        # Bàn tay phải (36-42)
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    ]

    # Chỉ số 29 khớp thân điều khiển bởi bộ giải mã GEAR-SONIC ONNX
    BODY_INDICES = [
        0, 1, 2, 3, 4, 5,       # Chân trái (6)
        6, 7, 8, 9, 10, 11,     # Chân phải (6)
        12, 13, 14,             # Hông / Thân (3)
        15, 16, 17, 18, 19, 20, 21,  # Tay trái (7)
        29, 30, 31, 32, 33, 34, 35,  # Tay phải (7)
    ]

    # Giới hạn góc quay chuẩn của từng khớp (rad)
    JOINT_LIMITS = {
        # Tay
        "shoulder_pitch": (-3.08, 2.67),
        "shoulder_roll": (-1.58, 2.25),
        "shoulder_yaw": (-2.61, 2.61),
        "elbow": (-1.04, 2.09),
        "wrist_roll": (-1.97, 1.97),
        "wrist_pitch": (-1.61, 1.61),
        "wrist_yaw": (-1.61, 1.61),
        # Hông
        "waist_yaw": (-2.61, 2.61),
        "waist_roll": (-0.52, 0.52),
        "waist_pitch": (-0.52, 0.52),
    }

    # Giới hạn 7 khớp mỗi bàn tay Dex3 theo thứ tự actuator (22-28 trái, 36-42 phải).
    # Chiều gập ngón ngược dấu giữa hai tay, nên không thể clip chung một khoảng [0, x].
    LEFT_HAND_LIMITS = np.array([
        (-1.0472, 1.0472), (-0.724312, 1.0472), (0.0, 1.74533),   # thumb 0-2
        (-1.5708, 0.0), (-1.74533, 0.0),                            # middle 0-1
        (-1.5708, 0.0), (-1.74533, 0.0),                            # index 0-1
    ], dtype=np.float32)
    RIGHT_HAND_LIMITS = np.array([
        (-1.0472, 1.0472), (-1.0472, 0.724312), (-1.74533, 0.0),   # thumb 0-2
        (0.0, 1.5708), (0.0, 1.74533),                              # index 0-1
        (0.0, 1.5708), (0.0, 1.74533),                              # middle 0-1
    ], dtype=np.float32)

    # Tư thế bàn tay khép ôm lon (rad); tay trái là ảnh gương của tay phải
    RIGHT_HAND_CLOSED = np.array([0.0, -0.5, -0.6, 0.7, 0.7, 0.7, 0.7], dtype=np.float32)
    LEFT_HAND_CLOSED = -RIGHT_HAND_CLOSED

    def __init__(
        self,
        default_stand_ctrl: Optional[np.ndarray] = None,
        onnx_model_path: Optional[str] = None,
        smoothing_alpha: float = 0.35,
        history_len: int = 10,
        mode: str = "kinematic",
    ):
        """Khởi tạo SONIC WBC Controller.
        
        Args:
            default_stand_ctrl: Mảng 43 phần tử biểu diễn tư thế sẵn sàng; mặc định đọc từ keyframe của scene XML.
            onnx_model_path: Đường dẫn tới model_decoder.onnx nếu có.
            smoothing_alpha: Hệ số lọc thông thấp làm mịn tín hiệu điều khiển (0 < alpha <= 1).
            history_len: Độ dài cửa sổ lịch sử quan sát cho ONNX Decoder (chuẩn 10 frames).
            mode: 'kinematic' (đứng vững & vươn tay gắp lon chuẩn xác) hoặc 'onnx' (mạng nơ-ron GEAR-SONIC).
        """
        self.mode = mode.lower()
        if default_stand_ctrl is None:
            from sim.g1_scene import load_ready_ctrl
            default_stand_ctrl = load_ready_ctrl()
        self.default_ctrl = np.asarray(default_stand_ctrl, dtype=np.float32).copy()

        self.default_body_ctrl = self.default_ctrl[self.BODY_INDICES].copy()
        self.smoothing_alpha = smoothing_alpha
        self.history_len = history_len
        self.last_ctrl = self.default_ctrl.copy()
        self.last_target = self.default_ctrl.copy()

        # Tự động tìm model_decoder.onnx nếu chưa chỉ định đường dẫn
        if onnx_model_path is None:
            candidates = [
                Path(__file__).parent.parent / "checkpoints" / "GEAR-SONIC" / "model_decoder.onnx",
                Path(__file__).parent.parent / "checkpoints" / "GEAR-SONIC" / "low_latency" / "model_decoder.onnx",
                Path(__file__).parent.parent / "checkpoints" / "GEAR-SONIC" / "sonic_v1_1" / "model_decoder.onnx",
            ]
            for cand in candidates:
                if cand.exists():
                    onnx_model_path = str(cand)
                    break

        # Khởi tạo ONNX Runtime Inference Session
        self.onnx_session = None
        self.onnx_model_path = onnx_model_path
        if onnx_model_path and Path(onnx_model_path).exists():
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(str(onnx_model_path))
                print(f"[SONIC WBC] Đã nạp thành công ONNX Decoder từ: {onnx_model_path}")
            except Exception as e:
                print(f"[SONIC WBC] Cảnh báo: Không thể nạp ONNX ({e}). Sử dụng Kinematic WBC Decoder.")

        # Khởi tạo bộ đệm lịch sử quan sát cho ONNX Decoder
        self._init_history()

    def _init_history(self):
        """Khởi tạo hàng đợi lịch sử 10 frames cho 5 kênh cảm biến proprioception."""
        self.his_ang_vel = [np.zeros(3, dtype=np.float32) for _ in range(self.history_len)]
        self.his_joint_pos = [np.zeros(29, dtype=np.float32) for _ in range(self.history_len)]
        self.his_joint_vel = [np.zeros(29, dtype=np.float32) for _ in range(self.history_len)]
        self.his_last_actions = [np.zeros(29, dtype=np.float32) for _ in range(self.history_len)]
        self.his_gravity = [np.array([0.0, 0.0, -1.0], dtype=np.float32) for _ in range(self.history_len)]
        self.last_onnx_action = np.zeros(29, dtype=np.float32)

    def reset(self):
        """Đặt lại bộ lọc làm mịn và bộ đệm lịch sử về trạng thái mặc định."""
        self.last_ctrl = self.default_ctrl.copy()
        self.last_target = self.default_ctrl.copy()
        self._init_history()

    def decode(
        self,
        motion_token: np.ndarray,
        left_hand_joints: Optional[np.ndarray] = None,
        right_hand_joints: Optional[np.ndarray] = None,
        robot_state: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Giải mã 64-dim latent motion token và 14-dim hand targets thành 43 motor commands.
        
        Args:
            motion_token: Vector 64 chiều biểu diễn quỹ đạo toàn thân do VLA dự đoán.
            left_hand_joints: Vector 7 chiều điều khiển bàn tay trái (ngón cái, trỏ, giữa).
            right_hand_joints: Vector 7 chiều điều khiển bàn tay phải.
            robot_state: Trạng thái hiện tại của robot (left_leg, waist, arm, ...).
            
        Returns:
            np.ndarray: Mảng 43 giá trị điều khiển vị trí motor (rad) ở tần số 50 Hz.
        """
        assert len(motion_token) == 64, f"Motion token phải có 64 chiều, nhận được {len(motion_token)}"

        # Khởi tạo từ tư thế đứng ổn định
        target_ctrl = self.default_ctrl.copy()

        # =========================================================================
        # PHƯƠNG PHÁP 1: GIẢI MÃ NƠ-RON QUA ONNX RUNTIME (GEAR-SONIC WBC)
        # =========================================================================
        if self.mode == "onnx" and self.onnx_session is not None and robot_state is not None:
            # 1. Trích xuất proprioception từ robot_state
            ang_vel = robot_state.get("base_ang_vel", np.zeros(3, dtype=np.float32)).astype(np.float32)
            gravity = robot_state.get("projected_gravity", np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

            # Ghép 29 góc khớp thân hiện tại
            body_pos = np.concatenate([
                robot_state["left_leg"],
                robot_state["right_leg"],
                robot_state["waist"],
                robot_state["left_arm"],
                robot_state["right_arm"],
            ]).astype(np.float32)

            # Độ lệch vị trí khớp so với tư thế đứng mặc định
            body_pos_rel = (body_pos - self.default_body_ctrl).astype(np.float32)
            body_vel = robot_state.get("body_joint_vel", np.zeros(29, dtype=np.float32)).astype(np.float32)

            # 2. Cập nhật FIFO rolling history buffers (độ dài 10 frames)
            self.his_ang_vel.pop(0)
            self.his_ang_vel.append(ang_vel)

            self.his_joint_pos.pop(0)
            self.his_joint_pos.append(body_pos_rel)

            self.his_joint_vel.pop(0)
            self.his_joint_vel.append(body_vel)

            self.his_last_actions.pop(0)
            self.his_last_actions.append(self.last_onnx_action.copy())

            self.his_gravity.pop(0)
            self.his_gravity.append(gravity)

            # 3. Tạo vector obs_dict 994 chiều: [64, 30, 290, 290, 290, 30]
            obs_dict = np.concatenate([
                motion_token.astype(np.float32),
                np.concatenate(self.his_ang_vel),
                np.concatenate(self.his_joint_pos),
                np.concatenate(self.his_joint_vel),
                np.concatenate(self.his_last_actions),
                np.concatenate(self.his_gravity),
            ]).reshape(1, -1).astype(np.float32)

            # 4. Thực thi mạng nơ-ron giải mã ONNX kèm action scaling
            action_29 = self.onnx_session.run(["action"], {"obs_dict": obs_dict})[0][0]
            self.last_onnx_action = action_29.copy()

            # Action scale chuẩn 0.25 để tránh giật khớp
            action_scaled = 0.25 * np.clip(action_29, -1.5, 1.5)

            # Giữ vững chân tuyệt đối ở tư thế đứng (ổn định trọng tâm)
            target_ctrl[0:12] = self.default_ctrl[0:12] + 0.05 * np.clip(action_scaled[0:12], -0.2, 0.2)
            # Thân trên (hông + 2 tay):
            target_ctrl[12:15] = self.default_ctrl[12:15] + action_scaled[12:15]
            target_ctrl[15:22] = self.default_ctrl[15:22] + action_scaled[15:22]
            target_ctrl[29:36] = self.default_ctrl[29:36] + action_scaled[22:29]

        # =========================================================================
        # PHƯƠNG PHÁP 2: KINEMATIC HEURISTIC WBC DECODER (CHUẨN ĐỨNG VỮNG CHO PHÂN LOẠI)
        # =========================================================================
        else:
            # 1. Chân & Trọng tâm (Giữ tư thế đứng vững vàng)
            leg_damping = 0.05 * np.clip(motion_token[0:12], -1.0, 1.0)
            target_ctrl[0:12] = self.default_ctrl[0:12] + leg_damping

            # 2. Hông & Thân trên
            waist_yaw_delta = 0.45 * np.clip(motion_token[12], -1.0, 1.0)
            waist_roll_delta = 0.15 * np.clip(motion_token[13], -1.0, 1.0)
            waist_pitch_delta = 0.25 * np.clip(motion_token[14], -1.0, 1.0)
            target_ctrl[12] = np.clip(self.default_ctrl[12] + waist_yaw_delta, -0.6, 0.6)
            target_ctrl[13] = np.clip(self.default_ctrl[13] + waist_roll_delta, -0.3, 0.3)
            target_ctrl[14] = np.clip(self.default_ctrl[14] + waist_pitch_delta, -0.2, 0.5)

            # 3-4. Hai cánh tay: token dùng quy ước đối xứng gương, cùng một token cho tay trái và tay phải
            # sinh ra chuyển động đối xứng qua mặt phẳng dọc thân (roll/yaw vai của tay phải đảo dấu).
            arm_gains = np.array([0.9, 0.6, 0.8, 0.8, 1.2, 0.7, 0.7], dtype=np.float32)
            right_sign = np.array([1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            left_lo = np.array([-1.8, -0.2, -1.5, 0.1, -1.8, -1.2, -1.2], dtype=np.float32)
            left_hi = np.array([1.5, 1.8, 1.5, 2.0, 1.8, 1.2, 1.2], dtype=np.float32)
            right_lo = np.array([-1.8, -1.8, -1.5, 0.1, -1.8, -1.2, -1.2], dtype=np.float32)
            right_hi = np.array([1.5, 0.2, 1.5, 2.0, 1.8, 1.2, 1.2], dtype=np.float32)

            target_ctrl[15:22] = np.clip(
                self.default_ctrl[15:22] + arm_gains * motion_token[15:22], left_lo, left_hi
            )
            target_ctrl[29:36] = np.clip(
                self.default_ctrl[29:36] + right_sign * arm_gains * motion_token[22:29], right_lo, right_hi
            )

        # =========================================================================
        # 5. ĐIỀU KHIỂN BÀN TAY (DEXTEROUS HANDS: 7L + 7R)
        # =========================================================================
        if left_hand_joints is None or len(left_hand_joints) != 7:
            # Không có góc khớp tay: token [-1, 1] -> độ khép [0, 1] của tư thế ôm lon
            left_hand_joints = np.clip(0.5 * (motion_token[29:36] + 1.0), 0.0, 1.0) * self.LEFT_HAND_CLOSED
        target_ctrl[22:29] = np.clip(left_hand_joints, self.LEFT_HAND_LIMITS[:, 0], self.LEFT_HAND_LIMITS[:, 1])

        if right_hand_joints is None or len(right_hand_joints) != 7:
            right_hand_joints = np.clip(0.5 * (motion_token[36:43] + 1.0), 0.0, 1.0) * self.RIGHT_HAND_CLOSED
        target_ctrl[36:43] = np.clip(right_hand_joints, self.RIGHT_HAND_LIMITS[:, 0], self.RIGHT_HAND_LIMITS[:, 1])

        # =========================================================================
        # 6. BỘ LỌC LÀM MỊN KHÔNG GIAN KHỚP (LOW-PASS FILTER / SMOOTHING)
        # =========================================================================
        return self._smooth(target_ctrl)

    def track_joint_targets(self, target_ctrl: np.ndarray) -> np.ndarray:
        """Bám theo 43 góc khớp tuyệt đối do VLA xuất trực tiếp (embodiment REAL_G1, không qua motion token).

        Chân giữ tư thế đứng mặc định vì embodiment này không điều khiển chân; bàn tay được clip theo giới hạn
        Dex3 rồi toàn bộ đi qua cùng bộ lọc làm mịn với decode().
        """
        target_ctrl = np.asarray(target_ctrl, dtype=np.float32).copy()
        target_ctrl[0:12] = self.default_ctrl[0:12]
        target_ctrl[22:29] = np.clip(target_ctrl[22:29], self.LEFT_HAND_LIMITS[:, 0], self.LEFT_HAND_LIMITS[:, 1])
        target_ctrl[36:43] = np.clip(target_ctrl[36:43], self.RIGHT_HAND_LIMITS[:, 0], self.RIGHT_HAND_LIMITS[:, 1])
        return self._smooth(target_ctrl)

    def _smooth(self, target_ctrl: np.ndarray) -> np.ndarray:
        # Mục tiêu trước bộ lọc: là action ghi vào dataset demo, vì lúc chạy model action lại đi qua bộ lọc này
        self.last_target = np.asarray(target_ctrl, dtype=np.float32).copy()
        smoothed_ctrl = (1.0 - self.smoothing_alpha) * self.last_ctrl + self.smoothing_alpha * target_ctrl
        self.last_ctrl = smoothed_ctrl.copy()
        return smoothed_ctrl.astype(np.float32)
