"""
Module bridge.gr00t_policy_client: Giao diện mô hình VLA NVIDIA GR00T N1.7 cho Unitree G1.
Tiếp nhận câu lệnh ngôn ngữ, ảnh camera ego_view (448x672) và trạng thái robot;
sinh chuỗi Action Chunk 40 bước (64-dim motion_token + 14-dim hand joints = 78 dims).
Hỗ trợ cả kết nối ZMQ tới PolicyServer thực tế và bộ sinh chính sách tự lập (Standalone).
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np


class Gr00tN17Policy:
    """Mô hình chính sách VLA GR00T N1.7 hỗ trợ embodiment UNITREE_G1_SONIC."""

    def __init__(
        self,
        mode: str = "standalone",
        server_host: str = "localhost",
        server_port: int = 5550,
        action_horizon: int = 40,
    ):
        """Khởi tạo Policy Client.
        
        Args:
            mode: 'standalone' (chạy trực tiếp không cần server) hoặc 'remote' (kết nối ZMQ).
            server_host: Địa chỉ IP máy chủ PolicyServer.
            server_port: Cổng ZMQ PolicyServer.
            action_horizon: Độ dài chuỗi hành động dự đoán (mặc định 40 bước thời gian).
        """
        self.mode = mode
        self.action_horizon = action_horizon
        self.server_host = server_host
        self.server_port = server_port

        self.zmq_client = None
        if self.mode == "remote":
            self._init_zmq()

    def _init_zmq(self):
        """Khởi tạo kết nối ZMQ REQ tới GR00T PolicyServer."""
        try:
            import zmq
            self.zmq_context = zmq.Context()
            self.zmq_client = self.zmq_context.socket(zmq.REQ)
            # Thiết lập timeout gửi/nhận tối đa 3 giây để tránh bị treo vĩnh viễn
            self.zmq_client.setsockopt(zmq.RCVTIMEO, 3000)
            self.zmq_client.setsockopt(zmq.SNDTIMEO, 3000)
            self.zmq_client.setsockopt(zmq.LINGER, 0)
            self.zmq_client.connect(f"tcp://{self.server_host}:{self.server_port}")
            print(f"[GR00T N1.7] Đã kết nối ZMQ PolicyServer tại: tcp://{self.server_host}:{self.server_port}")
        except Exception as e:
            print(f"[GR00T N1.7] Không thể kết nối ZMQ ({e}). Tự động chuyển sang chế độ Standalone.")
            self.mode = "standalone"

    def predict_action_chunk(
        self,
        task_description: str,
        ego_view_image: np.ndarray,
        robot_state: Dict[str, np.ndarray],
        step_progress: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Dự đoán Action Chunk 40 bước từ thông tin đa phương thức.
        
        Returns:
            Dict chứa:
                'motion_token': Mảng [40, 64] (latent motion tokens cho SONIC)
                'left_hand_joints': Mảng [40, 7] (góc khớp bàn tay trái)
                'right_hand_joints': Mảng [40, 7] (góc khớp bàn tay phải)
                'full_action': Mảng [40, 78] ghép nối đầy đủ
        """
        if self.mode == "remote" and self.zmq_client is not None:
            return self._query_remote_server(task_description, ego_view_image, robot_state, step_progress)
        else:
            return self._generate_standalone_trajectory(task_description, robot_state, step_progress)

    def _query_remote_server(
        self,
        task_description: str,
        ego_view_image: np.ndarray,
        robot_state: Dict[str, np.ndarray],
        step_progress: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Truy vấn PolicyServer từ xa thông qua giao thức ZMQ (có timeout an toàn)."""
        import pickle
        import zmq
        payload = {
            "video": {"ego_view": ego_view_image},
            "state": robot_state,
            "language": {"annotation.human.task_description": task_description},
        }

        try:
            self.zmq_client.send(pickle.dumps(payload))
            raw_response = pickle.loads(self.zmq_client.recv())
        except (zmq.Again, Exception) as e:
            print(f"\n⚠️ [GR00T N1.7] Không nhận được phản hồi từ Colab Server ({e}).")
            print("   -> Tự động dùng bộ sinh hành động cục bộ để robot tiếp tục gắp lon mượt mà!")
            # Tái tạo lại socket ZMQ để tránh trạng thái hỏng của REQ-REP
            try:
                self.zmq_client.close()
                self.zmq_client = self.zmq_context.socket(zmq.REQ)
                self.zmq_client.setsockopt(zmq.RCVTIMEO, 3000)
                self.zmq_client.setsockopt(zmq.SNDTIMEO, 3000)
                self.zmq_client.setsockopt(zmq.LINGER, 0)
                self.zmq_client.connect(f"tcp://{self.server_host}:{self.server_port}")
            except Exception:
                pass
            return self._generate_standalone_trajectory(task_description, robot_state, step_progress)

        # Chuẩn hóa phản hồi từ PolicyServer về cấu trúc chuẩn của GR00T-N1.7
        if isinstance(raw_response, dict):
            if "action" in raw_response:
                full_act = raw_response["action"]
            elif "full_action" in raw_response:
                full_act = raw_response["full_action"]
            elif "motion_token" in raw_response:
                return raw_response
            else:
                full_act = next(iter(raw_response.values()))
        else:
            full_act = raw_response

        if hasattr(full_act, "detach"):
            full_act = full_act.detach().cpu().numpy()
        full_act = np.asarray(full_act, dtype=np.float32)

        if full_act.ndim == 3:  # Trường hợp có batch dimension: [1, 40, 78] -> [40, 78]
            full_act = full_act[0]

        T = full_act.shape[0]
        return {
            "motion_token": full_act[:, :64],
            "left_hand_joints": full_act[:, 64:71] if full_act.shape[1] >= 71 else np.zeros((T, 7), dtype=np.float32),
            "right_hand_joints": full_act[:, 71:78] if full_act.shape[1] >= 78 else np.zeros((T, 7), dtype=np.float32),
            "full_action": full_act,
        }

    def _generate_standalone_trajectory(
        self,
        task_description: str,
        robot_state: Dict[str, np.ndarray],
        progress: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Bộ sinh chính sách Standalone mô phỏng chính xác hành vi phân loại lon của VLA.
        
        Tự động phân giải mục tiêu từ câu lệnh:
        - "Coca-Cola" / "Coke" / "red tray": Điều khiển cánh tay phải và hông hướng về lon Coke.
        - "Pepsi" / "blue tray": Điều khiển cánh tay trái và hông hướng về lon Pepsi.
        """
        T = self.action_horizon
        motion_tokens = np.zeros((T, 64), dtype=np.float32)
        left_hand = np.zeros((T, 7), dtype=np.float32)
        right_hand = np.zeros((T, 7), dtype=np.float32)

        desc_lower = task_description.lower()
        is_coke = "coca" in desc_lower or "coke" in desc_lower or "red" in desc_lower
        is_pepsi = "pepsi" in desc_lower or "blue" in desc_lower or not is_coke

        # Sinh 40 bước thời gian trong action chunk
        for t in range(T):
            # Tiến trình thời gian cục bộ (phase t) kết hợp tiến trình toàn cục (progress)
            phase = (progress + t * 0.015) % 1.0

            if is_coke:
                # TÁC VỤ 1: GẮP LON COCA-COLA (BÊN PHẢI) VÀ ĐẶT VÀO KHAY ĐỎ (BÊN PHẢI)
                # Giai đoạn 1 (0.0 -> 0.35): Vươn tay phải về phía lon Coca (Y = -0.12, Z = 0.74)
                if phase < 0.35:
                    reach_f = phase / 0.35
                    # Thân xoay nhẹ sang phải
                    motion_tokens[t, 12] = -0.15 * reach_f  # waist yaw
                    motion_tokens[t, 14] = 0.25 * reach_f   # waist pitch nghiêng xuống bàn
                    # Tay phải vươn tới lon
                    motion_tokens[t, 22] = -0.45 * reach_f  # r_shoulder_pitch
                    motion_tokens[t, 23] = -0.15 * reach_f  # r_shoulder_roll
                    motion_tokens[t, 25] = -0.65 * reach_f  # r_elbow duỗi ra
                    # Bàn tay mở rộng đón lon
                    right_hand[t, :] = 0.05
                # Giai đoạn 2 (0.35 -> 0.50): Khép các ngón tay kẹp chặt thân lon Coke
                elif phase < 0.50:
                    grasp_f = (phase - 0.35) / 0.15
                    motion_tokens[t, 12] = -0.15
                    motion_tokens[t, 14] = 0.25
                    motion_tokens[t, 22] = -0.45
                    motion_tokens[t, 25] = -0.65
                    # Ngón cái và ngón trỏ khép chặt quanh lon
                    right_hand[t, :] = 0.05 + 0.95 * grasp_f
                # Giai đoạn 3 (0.50 -> 0.75): Nhấc lon lên và xoay người sang khay đỏ (Y = -0.32)
                elif phase < 0.75:
                    lift_f = (phase - 0.50) / 0.25
                    # Hông quay sâu hơn sang phải hướng về khay đỏ
                    motion_tokens[t, 12] = -0.15 - 0.30 * lift_f
                    motion_tokens[t, 14] = 0.25 - 0.15 * lift_f  # Đứng thẳng hơn
                    # Nhấc khuỷu tay lên cao
                    motion_tokens[t, 22] = -0.45 + 0.20 * lift_f
                    motion_tokens[t, 25] = -0.65 + 0.35 * lift_f
                    # Giữ chặt lon
                    right_hand[t, :] = 1.0
                # Giai đoạn 4 (0.75 -> 1.0): Hạ lon vào khay đỏ và mở ngón tay
                else:
                    place_f = (phase - 0.75) / 0.25
                    motion_tokens[t, 12] = -0.45
                    motion_tokens[t, 14] = 0.20 * place_f
                    motion_tokens[t, 22] = -0.35
                    motion_tokens[t, 25] = -0.40
                    # Mở ngón tay nhả lon vào khay
                    right_hand[t, :] = 1.0 * (1.0 - place_f)

            else:
                # TÁC VỤ 2: GẮP LON PEPSI (BÊN TRÁI) VÀ ĐẶT VÀO KHAY XANH (BÊN TRÁI)
                if phase < 0.35:
                    reach_f = phase / 0.35
                    motion_tokens[t, 12] = 0.15 * reach_f   # waist yaw sang trái
                    motion_tokens[t, 14] = 0.25 * reach_f   # waist pitch
                    motion_tokens[t, 15] = -0.45 * reach_f  # l_shoulder_pitch
                    motion_tokens[t, 16] = 0.15 * reach_f   # l_shoulder_roll
                    motion_tokens[t, 18] = -0.65 * reach_f  # l_elbow duỗi ra
                    left_hand[t, :] = 0.05
                elif phase < 0.50:
                    grasp_f = (phase - 0.35) / 0.15
                    motion_tokens[t, 12] = 0.15
                    motion_tokens[t, 14] = 0.25
                    motion_tokens[t, 15] = -0.45
                    motion_tokens[t, 18] = -0.65
                    left_hand[t, :] = 0.05 + 0.95 * grasp_f
                elif phase < 0.75:
                    lift_f = (phase - 0.50) / 0.25
                    motion_tokens[t, 12] = 0.15 + 0.30 * lift_f
                    motion_tokens[t, 14] = 0.25 - 0.15 * lift_f
                    motion_tokens[t, 15] = -0.45 + 0.20 * lift_f
                    motion_tokens[t, 18] = -0.65 + 0.35 * lift_f
                    left_hand[t, :] = 1.0
                else:
                    place_f = (phase - 0.75) / 0.25
                    motion_tokens[t, 12] = 0.45
                    motion_tokens[t, 14] = 0.20 * place_f
                    motion_tokens[t, 15] = -0.35
                    motion_tokens[t, 18] = -0.40
                    left_hand[t, :] = 1.0 * (1.0 - place_f)

        # Ghép thành Action Chunk kích thước [40, 78]
        full_action = np.concatenate([motion_tokens, left_hand, right_hand], axis=1)

        return {
            "motion_token": motion_tokens,
            "left_hand_joints": left_hand,
            "right_hand_joints": right_hand,
            "full_action": full_action,
        }
