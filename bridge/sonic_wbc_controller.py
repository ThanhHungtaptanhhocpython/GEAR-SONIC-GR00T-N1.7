"""
Module bridge.sonic_wbc_controller: Bộ điều khiển toàn thân GEAR-SONIC WBC cho robot Unitree G1.
Tiếp nhận 64-dimensional latent motion tokens từ GR00T N1.7 VLA Policy và giải mã
thành mục tiêu vị trí 43 khớp (29 khớp toàn thân + 14 khớp hai bàn tay) ở tần số 50 Hz.
"""

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

    def __init__(
        self,
        default_stand_ctrl: Optional[np.ndarray] = None,
        onnx_model_path: Optional[str] = None,
        smoothing_alpha: float = 0.35,
    ):
        """Khởi tạo SONIC WBC Controller.
        
        Args:
            default_stand_ctrl: Mảng 43 phần tử biểu diễn tư thế đứng mặc định (stand keyframe).
            onnx_model_path: Đường dẫn tới model_decoder.onnx nếu có.
            smoothing_alpha: Hệ số lọc thông thấp làm mịn tín hiệu điều khiển (0 < alpha <= 1).
        """
        if default_stand_ctrl is not None:
            self.default_ctrl = default_stand_ctrl.copy()
        else:
            # Tư thế đứng chuẩn của G1
            self.default_ctrl = np.zeros(43, dtype=np.float32)
            # Khớp tay mặc định hơi gập
            self.default_ctrl[15] = 0.2   # left_shoulder_pitch
            self.default_ctrl[16] = 0.2   # left_shoulder_roll
            self.default_ctrl[18] = 1.28  # left_elbow
            self.default_ctrl[29] = 0.2   # right_shoulder_pitch
            self.default_ctrl[30] = -0.2  # right_shoulder_roll
            self.default_ctrl[32] = 1.28  # right_elbow

        self.smoothing_alpha = smoothing_alpha
        self.last_ctrl = self.default_ctrl.copy()

        # Kiểm tra nạp ONNX decoder nếu có
        self.onnx_session = None
        if onnx_model_path and Path(onnx_model_path).exists():
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(onnx_model_path)
                print(f"[SONIC WBC] Đã nạp thành công ONNX Decoder từ: {onnx_model_path}")
            except Exception as e:
                print(f"[SONIC WBC] Cảnh báo: Không thể nạp ONNX ({e}). Sử dụng Kinematic WBC Decoder.")

    def reset(self):
        """Đặt lại bộ lọc làm mịn về trạng thái đứng mặc định."""
        self.last_ctrl = self.default_ctrl.copy()

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

        # --- 1. GIẢI MÃ PHẦN THÂN DƯỚI & TRỌNG TÂM (LEGS & BALANCE) ---
        # 12 khớp chân được giữ vững để chịu tải trọng tâm và giữ thăng bằng
        # Các token từ 0-11 điều biến vi sai cân bằng trọng tâm (CoM balancing)
        leg_damping = 0.05 * np.clip(motion_token[0:12], -1.0, 1.0)
        target_ctrl[0:12] += leg_damping

        # --- 2. GIẢI MÃ VÙNG HÔNG & THÂN TRÊN (WAIST ORIENTATION) ---
        # Token 12-14 điều khiển hướng quay và độ nghiêng thân:
        # [12: yaw (-0.5..0.5), 13: roll (-0.2..0.2), 14: pitch (-0.3..0.3)]
        waist_yaw_delta = 0.45 * np.clip(motion_token[12], -1.0, 1.0)
        waist_roll_delta = 0.15 * np.clip(motion_token[13], -1.0, 1.0)
        waist_pitch_delta = 0.25 * np.clip(motion_token[14], -1.0, 1.0)
        target_ctrl[12] = np.clip(waist_yaw_delta, -0.6, 0.6)
        target_ctrl[13] = np.clip(waist_roll_delta, -0.3, 0.3)
        target_ctrl[14] = np.clip(waist_pitch_delta, -0.4, 0.4)

        # --- 3. GIẢI MÃ CÁNH TAY TRÁI (LEFT ARM REACHING & MANIPULATION) ---
        # Token 15-21 tương ứng 7 bậc tự do tay trái (shoulder pitch/roll/yaw, elbow, wrist pitch/roll/yaw)
        # Tỉ lệ scaling đưa vector latent về góc quay vật lý chính xác của cánh tay
        l_shoulder_pitch = 0.2 + 0.9 * motion_token[15]
        l_shoulder_roll  = 0.2 + 0.6 * motion_token[16]
        l_shoulder_yaw   = 0.0 + 0.8 * motion_token[17]
        l_elbow          = 1.28 + 0.8 * motion_token[18]
        l_wrist_roll     = 0.0 + 1.2 * motion_token[19]
        l_wrist_pitch    = 0.0 + 0.7 * motion_token[20]
        l_wrist_yaw      = 0.0 + 0.7 * motion_token[21]

        target_ctrl[15] = np.clip(l_shoulder_pitch, -1.8, 1.5)
        target_ctrl[16] = np.clip(l_shoulder_roll, -0.2, 1.8)
        target_ctrl[17] = np.clip(l_shoulder_yaw, -1.5, 1.5)
        target_ctrl[18] = np.clip(l_elbow, 0.1, 2.0)
        target_ctrl[19] = np.clip(l_wrist_roll, -1.8, 1.8)
        target_ctrl[20] = np.clip(l_wrist_pitch, -1.2, 1.2)
        target_ctrl[21] = np.clip(l_wrist_yaw, -1.2, 1.2)

        # --- 4. GIẢI MÃ CÁNH TAY PHẢI (RIGHT ARM REACHING & MANIPULATION) ---
        # Token 22-28 tương ứng 7 bậc tự do tay phải
        r_shoulder_pitch = 0.2 + 0.9 * motion_token[22]
        r_shoulder_roll  = -0.2 - 0.6 * motion_token[23]
        r_shoulder_yaw   = 0.0 - 0.8 * motion_token[24]
        r_elbow          = 1.28 + 0.8 * motion_token[25]
        r_wrist_roll     = 0.0 + 1.2 * motion_token[26]
        r_wrist_pitch    = 0.0 + 0.7 * motion_token[27]
        r_wrist_yaw      = 0.0 + 0.7 * motion_token[28]

        target_ctrl[29] = np.clip(r_shoulder_pitch, -1.8, 1.5)
        target_ctrl[30] = np.clip(r_shoulder_roll, -1.8, 0.2)
        target_ctrl[31] = np.clip(r_shoulder_yaw, -1.5, 1.5)
        target_ctrl[32] = np.clip(r_elbow, 0.1, 2.0)
        target_ctrl[33] = np.clip(r_wrist_roll, -1.8, 1.8)
        target_ctrl[34] = np.clip(r_wrist_pitch, -1.2, 1.2)
        target_ctrl[35] = np.clip(r_wrist_yaw, -1.2, 1.2)

        # --- 5. ĐIỀU KHIỂN BÀN TAY (DEXTEROUS HANDS: 7L + 7R) ---
        # Nếu VLA có dự đoán góc ngón tay thì sử dụng trực tiếp; ngược lại trích xuất từ token 29-42
        if left_hand_joints is not None and len(left_hand_joints) == 7:
            target_ctrl[22:29] = np.clip(left_hand_joints, 0.0, 1.5)
        else:
            # Điều khiển từ motion_token phụ
            target_ctrl[22:29] = np.clip(0.6 * (motion_token[29:36] + 1.0), 0.0, 1.4)

        if right_hand_joints is not None and len(right_hand_joints) == 7:
            target_ctrl[36:43] = np.clip(right_hand_joints, 0.0, 1.5)
        else:
            target_ctrl[36:43] = np.clip(0.6 * (motion_token[36:43] + 1.0), 0.0, 1.4)

        # --- 6. BỘ LỌC LÀM MỊN KHÔNG GIAN KHỚP (LOW-PASS FILTER / SMOOTHING) ---
        # Loại bỏ hiện tượng giật xung, đảm bảo motor phản xạ mượt mà
        smoothed_ctrl = (1.0 - self.smoothing_alpha) * self.last_ctrl + self.smoothing_alpha * target_ctrl
        self.last_ctrl = smoothed_ctrl.copy()

        return smoothed_ctrl.astype(np.float32)
