"""
Module bridge.gr00t_policy_client: Giao diện mô hình VLA NVIDIA GR00T N1.7 cho Unitree G1.
Tiếp nhận câu lệnh ngôn ngữ, ảnh camera ego_view (448x672) và trạng thái robot, trả về action chunk:
  - Standalone: quỹ đạo mẫu cục bộ dạng motion token (64-dim motion_token + 14-dim hand joints = 78 dims).
  - Remote: truy vấn PolicyServer qua ZMQ. Server chạy model GR00T thật (embodiment REAL_G1) trả về góc khớp
    tay/bàn tay/eo; server ở chế độ scripted trả về quỹ đạo mẫu dạng motion token như standalone.
"""

import io
from typing import Dict, Optional, Tuple, Any
import numpy as np

from bridge.sonic_wbc_controller import SonicWBCController


# Quỹ đạo phân loại lon mẫu (Standalone). Mỗi waypoint là token cho tay gắp lon Coca-Cola (tay phải):
#   [waist_yaw, waist_pitch, s_pitch, s_roll, s_yaw, elbow, wr_roll, wr_pitch, wr_yaw]
# Token là độ lệch so với tư thế ready_table theo quy ước đối xứng gương của SonicWBCController.
# Các giá trị được giải IK sao cho điểm gắp trên bàn tay tới lon/khay, bàn tay giữ trục đứng (lon luôn thẳng)
# và các ngón tay cách mặt bàn. waist_yaw giữ 0: xoay eo khi đang vươn tay làm bàn chân trượt xoay và
# cánh tay còn lại quét trúng lon kia.
WAYPOINTS = {
    "ready":   [0.0,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],
    "pre":     [0.0,  0.303, -0.037, -0.492,  0.001,  0.235, -0.243, -0.473, -0.878],
    "grasp":   [0.0,  0.834, -0.304, -0.601, -0.091,  0.342, -0.372, -0.495, -1.233],
    "lift":    [0.0, -0.314, -0.143, -0.449, -0.177, -0.195, -0.308,  0.293, -0.994],
    "carry":   [0.0,  0.206, -0.374, -0.150,  0.194,  0.105, -0.253,  0.231, -1.512],
    "place":   [0.0,  1.000, -0.517, -0.270,  0.244,  0.430, -0.261, -0.236, -1.748],
    "retreat": [0.0, -0.424, -0.262,  0.027,  0.176, -0.125, -0.208,  0.617, -1.192],
}

# Lịch trình một chu kỳ: (phase, waypoint, độ khép bàn tay 0..1); giữa hai mốc nội suy smoothstep
SCHEDULE = [
    (0.00, "ready", 0.0),
    (0.18, "pre", 0.0),
    (0.30, "grasp", 0.0),
    (0.38, "grasp", 1.0),   # khép ngón ôm lon
    (0.50, "lift", 1.0),
    (0.64, "carry", 1.0),
    (0.74, "place", 1.0),
    (0.80, "place", 0.0),   # mở ngón nhả lon vào khay
    (0.88, "retreat", 0.0),
    (1.00, "ready", 0.0),
]

# Tay trái (Pepsi) là ảnh gương của tay phải: chỉ đảo dấu waist_yaw, wrist_roll, wrist_yaw
PEPSI_MIRROR = np.array([-1, 1, 1, 1, 1, 1, -1, 1, -1], dtype=np.float32)

# Chỉ số motion token của [waist_yaw, waist_pitch] + 7 khớp tay
RIGHT_ARM_TOKEN_IDX = [12, 14, 22, 23, 24, 25, 26, 27, 28]
LEFT_ARM_TOKEN_IDX = [12, 14, 15, 16, 17, 18, 19, 20, 21]

_KEY_PHASES = np.array([k[0] for k in SCHEDULE], dtype=np.float32)
_KEY_POSES = np.array([WAYPOINTS[k[1]] for k in SCHEDULE], dtype=np.float32)
_KEY_GRIPS = np.array([k[2] for k in SCHEDULE], dtype=np.float32)


def scripted_sorting_chunk(task_description: str, progress: float, horizon: int, cycle_steps: int) -> np.ndarray:
    """Sinh action chunk [horizon, 78] của quỹ đạo phân loại lon mẫu bắt đầu tại tiến trình `progress`.

    progress chạy từ 0 đến 1 trong một lần gắp-đặt; hết chu kỳ thì robot giữ tư thế ready (nhiệm vụ đã xong)
    thay vì lặp lại và hạ tay trúng lon đã nằm trong khay.
    """
    desc = task_description.lower()
    is_coke = "coca" in desc or "coke" in desc or "red" in desc

    phase = np.minimum(progress + np.arange(horizon, dtype=np.float32) / cycle_steps, 1.0)
    seg = np.clip(np.searchsorted(_KEY_PHASES, phase, side="right") - 1, 0, len(SCHEDULE) - 2)
    t = np.clip((phase - _KEY_PHASES[seg]) / (_KEY_PHASES[seg + 1] - _KEY_PHASES[seg]), 0.0, 1.0)
    w = (t * t * (3.0 - 2.0 * t))[:, None]
    pose = (1.0 - w) * _KEY_POSES[seg] + w * _KEY_POSES[seg + 1]
    grip = (1.0 - w[:, 0]) * _KEY_GRIPS[seg] + w[:, 0] * _KEY_GRIPS[seg + 1]

    action = np.zeros((horizon, 78), dtype=np.float32)
    if is_coke:
        action[:, RIGHT_ARM_TOKEN_IDX] = pose
        action[:, 71:78] = grip[:, None] * SonicWBCController.RIGHT_HAND_CLOSED
    else:
        action[:, LEFT_ARM_TOKEN_IDX] = pose * PEPSI_MIRROR
        action[:, 64:71] = grip[:, None] * SonicWBCController.LEFT_HAND_CLOSED
    return action


class Gr00tN17Policy:
    """Client chính sách VLA GR00T N1.7.

    Mỗi action chunk là dict có key 'action_space':
      - 'sonic_tokens': motion_token [T, 64], left/right_hand_joints [T, 7], full_action [T, 78]
      - 'real_g1': g1_action {key: [T, D]} là output của model GR00T thật (embodiment REAL_G1)
    """

    # Độ dài một chu kỳ gắp-đặt của quỹ đạo mẫu: 350 bước (7.0 giây @ 50 Hz)
    CYCLE_STEPS = 350

    def __init__(
        self,
        mode: str = "standalone",
        server_host: str = "localhost",
        server_port: int = 5550,
        action_horizon: int = 40,
        timeout_ms: int = 3000,
        image_short_edge: int = 256,
        jpeg_quality: int = 90,
    ):
        """Khởi tạo Policy Client.

        Args:
            mode: 'standalone' (chạy trực tiếp không cần server) hoặc 'remote' (kết nối ZMQ).
            server_host: Địa chỉ IP máy chủ PolicyServer.
            server_port: Cổng ZMQ PolicyServer.
            action_horizon: Độ dài chuỗi hành động của quỹ đạo mẫu (mặc định 40 bước thời gian).
            timeout_ms: Thời gian chờ tối đa một lần gửi/nhận ZMQ.
            image_short_edge: Cạnh ngắn của ảnh gửi lên server (model GR00T tự resize cạnh ngắn về 256).
            jpeg_quality: Chất lượng nén JPEG của ảnh gửi lên server.
        """
        self.mode = mode
        self.action_horizon = action_horizon
        self.server_host = server_host
        self.server_port = server_port
        self.timeout_ms = timeout_ms
        self.image_short_edge = image_short_edge
        self.jpeg_quality = jpeg_quality

        # Không gian action của phản hồi gần nhất từ server (None khi chưa nhận được phản hồi nào)
        self.server_action_space: Optional[str] = None
        self.zmq_client = None
        if self.mode == "remote":
            self._init_zmq()

    def _init_zmq(self):
        """Khởi tạo kết nối ZMQ REQ tới GR00T PolicyServer."""
        try:
            import zmq
            self.zmq_context = zmq.Context()
            self._connect()
            print(f"[GR00T N1.7] Đã kết nối ZMQ PolicyServer tại: tcp://{self.server_host}:{self.server_port}")
        except Exception as e:
            print(f"[GR00T N1.7] Không thể kết nối ZMQ ({e}). Tự động chuyển sang chế độ Standalone.")
            self.mode = "standalone"

    def _connect(self):
        import zmq
        self.zmq_client = self.zmq_context.socket(zmq.REQ)
        # Timeout gửi/nhận để không bị treo vĩnh viễn khi mất kết nối
        self.zmq_client.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.zmq_client.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.zmq_client.setsockopt(zmq.LINGER, 0)
        self.zmq_client.connect(f"tcp://{self.server_host}:{self.server_port}")

    def predict_action_chunk(
        self,
        task_description: str,
        ego_view_image: np.ndarray,
        robot_state: Dict[str, np.ndarray],
        step_progress: float = 0.0,
        g1_state: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Dự đoán action chunk từ thông tin đa phương thức.

        Args:
            ego_view_image: Ảnh [H, W, 3] hoặc chuỗi khung hình [T, H, W, 3] theo thứ tự cũ -> mới.
            g1_state: State REAL_G1 (bridge.real_g1_adapter) cho model GR00T thật ở chế độ remote.

        Returns:
            Action chunk (xem docstring lớp), hoặc None khi server lỗi / không phản hồi và không có quỹ đạo
            dự phòng phù hợp; khi đó pipeline giữ nguyên tư thế và thử lại ở lần replan sau.
        """
        # Không có ảnh camera thì không gửi request thiếu ảnh tới server mà dùng quỹ đạo cục bộ
        if self.mode == "remote" and self.zmq_client is not None and ego_view_image is not None:
            return self._query_remote_server(task_description, ego_view_image, robot_state, step_progress, g1_state)
        else:
            return self._generate_standalone_trajectory(task_description, robot_state, step_progress)

    def _encode_frames(self, frames: np.ndarray) -> list:
        """Thu nhỏ ảnh về cạnh ngắn image_short_edge rồi nén JPEG để giảm băng thông qua tunnel."""
        from PIL import Image

        frames = np.asarray(frames)
        if frames.ndim == 3:
            frames = frames[None]
        encoded = []
        for frame in frames:
            img = Image.fromarray(frame)
            scale = self.image_short_edge / min(img.size)
            if scale < 1.0:
                img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BOX)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality)
            encoded.append(buf.getvalue())
        return encoded

    def _query_remote_server(
        self,
        task_description: str,
        ego_view_image: np.ndarray,
        robot_state: Dict[str, np.ndarray],
        step_progress: float = 0.0,
        g1_state: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Truy vấn PolicyServer từ xa thông qua giao thức ZMQ (có timeout an toàn)."""
        import pickle

        # Chỉ gửi kiểu thuần Python (bytes, list, float): môi trường Isaac-GR00T dùng numpy 1.26,
        # không unpickle được ndarray tạo bởi numpy 2.x ở máy local.
        payload = {
            "video": {"ego_view": self._encode_frames(ego_view_image)},
            "video_encoding": "jpeg",
            "g1_state": {k: np.asarray(v, dtype=np.float32).tolist() for k, v in (g1_state or {}).items()},
            "language": {"annotation.human.task_description": task_description},
            "step_progress": float(step_progress),
        }

        try:
            self.zmq_client.send(pickle.dumps(payload))
            response = pickle.loads(self.zmq_client.recv())
        except Exception as e:
            print(f"\n⚠️ [GR00T N1.7] Không nhận được phản hồi từ Policy Server ({e}).")
            # Socket REQ hỏng trạng thái sau timeout: tạo lại socket
            try:
                self.zmq_client.close()
                self._connect()
            except Exception:
                pass
            return self._fallback(task_description, robot_state, step_progress)

        if not isinstance(response, dict):
            response = {"action": response}
        if "error" in response:
            print(f"\n⚠️ [GR00T N1.7] Policy Server báo lỗi: {response['error']}")
            return self._fallback(task_description, robot_state, step_progress)

        self.server_action_space = response.get("action_space", "sonic_tokens")
        if self.server_action_space == "real_g1":
            return {
                "action_space": "real_g1",
                "g1_action": {k: np.asarray(v, dtype=np.float32) for k, v in response["action"].items()},
            }
        return self._parse_token_chunk(response)

    def _fallback(
        self, task_description: str, robot_state: Dict[str, np.ndarray], step_progress: float
    ) -> Optional[Dict[str, Any]]:
        """Chỉ thay bằng quỹ đạo mẫu cục bộ khi server cũng đang phục vụ quỹ đạo mẫu.

        Với model GR00T thật thì trả None (robot giữ tư thế), để quỹ đạo mẫu không lẫn vào kết quả kiểm thử VLA.
        """
        if self.server_action_space == "sonic_tokens":
            print("   -> Dùng quỹ đạo mẫu cục bộ thay cho server.")
            return self._generate_standalone_trajectory(task_description, robot_state, step_progress)
        print("   -> Robot giữ nguyên tư thế, thử lại ở lần replan sau.")
        return None

    @staticmethod
    def _parse_token_chunk(response: Dict[str, Any]) -> Dict[str, Any]:
        """Chuẩn hóa phản hồi dạng motion token [T, 78] của server."""
        if "motion_token" in response:
            return {"action_space": "sonic_tokens", **response}
        full_act = response.get("action", response.get("full_action"))

        if hasattr(full_act, "detach"):
            full_act = full_act.detach().cpu().numpy()
        full_act = np.asarray(full_act, dtype=np.float32)

        if full_act.ndim == 3:  # Trường hợp có batch dimension: [1, 40, 78] -> [40, 78]
            full_act = full_act[0]

        T = full_act.shape[0]
        return {
            "action_space": "sonic_tokens",
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
    ) -> Dict[str, Any]:
        """Bộ sinh chính sách Standalone: quỹ đạo gắp-đặt mẫu nội suy smoothstep qua SCHEDULE.

        Reach (0.00 -> 0.30), Grasp (0.30 -> 0.38), Lift & Carry (0.38 -> 0.64),
        Place & Release (0.64 -> 0.80), Retract (0.80 -> 1.00).
        """
        full_action = scripted_sorting_chunk(task_description, progress, self.action_horizon, self.CYCLE_STEPS)
        return {
            "action_space": "sonic_tokens",
            "motion_token": full_action[:, :64],
            "left_hand_joints": full_action[:, 64:71],
            "right_hand_joints": full_action[:, 71:78],
            "full_action": full_action,
        }
