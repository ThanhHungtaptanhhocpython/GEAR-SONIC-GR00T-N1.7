"""
Script colab_gr00t_server.py: ZMQ Policy Server cho NVIDIA GR00T N1.7 trên Google Colab GPU.
Nhận ảnh camera ego_view + câu lệnh ngôn ngữ + trạng thái robot từ máy local và trả về action chunk.

Hai chế độ (--policy):
  - gr00t (mặc định): model GR00T N1.7 thật qua API Gr00tPolicy của Isaac-GR00T. Checkpoint base dùng embodiment
    REAL_G1 (real_g1_relative_eef_relative_joints); checkpoint fine-tune bằng finetune/g1_can_sort_config.py dùng
    --embodiment-tag new_embodiment với cùng layout. Trả về góc khớp tay/bàn tay/eo tuyệt đối, action_space "real_g1".
    Cần chạy bằng Python của môi trường Isaac-GR00T và HF_TOKEN có quyền truy cập nvidia/Cosmos-Reason2-2B.
  - scripted: quỹ đạo phân loại lon mẫu dạng motion token [40, 78], action_space "sonic_tokens".
"""

import argparse
import io
import os
import traceback
import pickle
import sys
import time
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False


# Quỹ đạo phân loại lon mẫu: bản sao của bridge/gr00t_policy_client.py (server chạy độc lập trên Colab,
# không import được package bridge). Khi đổi waypoint, lịch trình hay tư thế bàn tay phải sửa cả hai nơi.
# Waypoint cho tay phải: [waist_yaw, waist_pitch, s_pitch, s_roll, s_yaw, elbow, wr_roll, wr_pitch, wr_yaw]
WAYPOINTS = {
    "ready":   [0.0,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],
    "pre":     [0.0,  0.303, -0.037, -0.492,  0.001,  0.235, -0.243, -0.473, -0.878],
    "grasp":   [0.0,  0.834, -0.304, -0.601, -0.091,  0.342, -0.372, -0.495, -1.233],
    "lift":    [0.0, -0.314, -0.143, -0.449, -0.177, -0.195, -0.308,  0.293, -0.994],
    "carry":   [0.0,  0.206, -0.374, -0.150,  0.194,  0.105, -0.253,  0.231, -1.512],
    "place":   [0.0,  1.000, -0.517, -0.270,  0.244,  0.430, -0.261, -0.236, -1.748],
    "retreat": [0.0, -0.424, -0.262,  0.027,  0.176, -0.125, -0.208,  0.617, -1.192],
}
SCHEDULE = [
    (0.00, "ready", 0.0), (0.18, "pre", 0.0), (0.30, "grasp", 0.0), (0.38, "grasp", 1.0),
    (0.50, "lift", 1.0), (0.64, "carry", 1.0), (0.74, "place", 1.0), (0.80, "place", 0.0),
    (0.88, "retreat", 0.0), (1.00, "ready", 0.0),
]
CYCLE_STEPS = 350
PEPSI_MIRROR = np.array([-1, 1, 1, 1, 1, 1, -1, 1, -1], dtype=np.float32)
RIGHT_ARM_TOKEN_IDX = [12, 14, 22, 23, 24, 25, 26, 27, 28]
LEFT_ARM_TOKEN_IDX = [12, 14, 15, 16, 17, 18, 19, 20, 21]
RIGHT_HAND_CLOSED = np.array([0.0, -0.5, -0.6, 0.7, 0.7, 0.7, 0.7], dtype=np.float32)
LEFT_HAND_CLOSED = -RIGHT_HAND_CLOSED

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
        action[:, 71:78] = grip[:, None] * RIGHT_HAND_CLOSED
    else:
        action[:, LEFT_ARM_TOKEN_IDX] = pose * PEPSI_MIRROR
        action[:, 64:71] = grip[:, None] * LEFT_HAND_CLOSED
    return action


# Embodiment của checkpoint base GR00T-N1.7-3B ứng với robot G1 (xem bridge/real_g1_adapter.py ở máy local)
DEFAULT_EMBODIMENT_TAG = "real_g1_relative_eef_relative_joints"


class ScriptedBackend:
    """Quỹ đạo phân loại lon mẫu, không dùng model."""

    action_space = "sonic_tokens"

    def predict(self, task_description: str, frames: list, g1_state: dict, step_progress: float):
        return scripted_sorting_chunk(task_description, step_progress, 40, CYCLE_STEPS).tolist()


def disable_flash_attn_on_pre_ampere():
    """flash-attn 2 chỉ chạy trên GPU từ Ampere (sm80) trở lên. Trên T4 (sm75) chặn import flash_attn để
    backbone Qwen3-VL tự chuyển sang attention sdpa. Phải gọi trước khi import gr00t."""
    if HAS_TORCH and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        sys.modules["flash_attn"] = None
        print(
            "[Colab Server] GPU trước Ampere (ví dụ T4): tắt flash-attn, dùng sdpa. bfloat16 trên GPU này chạy "
            "giả lập nên suy luận chậm; nên chọn runtime L4 hoặc A100."
        )


class Gr00tBackend:
    """Model GR00T N1.7 thật qua Gr00tPolicy của Isaac-GR00T."""

    action_space = "real_g1"

    def __init__(self, model_path: str, device: str, embodiment_tag: str = DEFAULT_EMBODIMENT_TAG):
        disable_flash_attn_on_pre_ampere()
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        if not os.path.isdir(model_path):
            from huggingface_hub import snapshot_download
            print(f"[Colab Server] Đang tải checkpoint {model_path} từ HuggingFace...")
            model_path = snapshot_download(model_path)

        print(f"[Colab Server] Đang nạp GR00T từ {model_path} (embodiment {embodiment_tag}) vào {device}...")
        self.policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device=device)
        self.embodiment_tag = self.policy.embodiment_tag.value

        cfg = self.policy.get_modality_config()
        self.video_keys = cfg["video"].modality_keys
        self.video_horizon = len(cfg["video"].delta_indices)
        self.state_keys = cfg["state"].modality_keys
        self.language_keys = cfg["language"].modality_keys
        print(f"[Colab Server] Video {self.video_keys} x {self.video_horizon} khung hình | State {self.state_keys}")
        print(f"[Colab Server] Action {cfg['action'].modality_keys}")

    def predict(self, task_description: str, frames: list, g1_state: dict, step_progress: float):
        missing = [k for k in self.state_keys if k not in g1_state]
        if missing:
            raise ValueError(f"Thiếu state {missing}. Client phải gửi g1_state theo embodiment {self.embodiment_tag}.")
        if not frames:
            raise ValueError("Request không có khung hình ego_view.")

        # Thiếu khung hình quá khứ (đầu episode) thì lặp lại khung hình cũ nhất
        frames = list(frames[-self.video_horizon:])
        frames = [frames[0]] * (self.video_horizon - len(frames)) + frames
        video = np.stack(frames)[None].astype(np.uint8)  # (B=1, T, H, W, 3)

        observation = {
            "video": {k: video for k in self.video_keys},
            "state": {k: np.asarray(g1_state[k], dtype=np.float32).reshape(1, 1, -1) for k in self.state_keys},
            "language": {k: [[task_description]] for k in self.language_keys},
        }
        action, _ = self.policy.get_action(observation)
        return {k: v[0].tolist() for k, v in action.items()}  # bỏ batch: {key: [T, D]}

    def warmup(self):
        """Chạy một lần suy luận giả để lần gọi đầu từ client không bị timeout vì khởi tạo CUDA kernel."""
        frame = np.zeros((256, 384, 3), dtype=np.uint8)
        norm_params = self.policy.processor.state_action_processor.norm_params[self.embodiment_tag]["state"]
        state = {}
        for k in self.state_keys:
            dim = norm_params[k]["dim"].item()
            state[k] = np.zeros(dim, dtype=np.float32)
            if k.endswith("eef_9d"):
                state[k][3:] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # rot6d của ma trận đơn vị
        t0 = time.time()
        self.predict("warm up", [frame], state, 0.0)
        print(f"[Colab Server] Warm-up xong trong {time.time() - t0:.1f}s.")


def decode_frames(payload: dict) -> list:
    """Giải mã khung hình ego_view (JPEG hoặc mảng uint8) thành list ảnh [H, W, 3] theo thứ tự cũ -> mới."""
    frames = payload.get("video", {}).get("ego_view")
    if frames is None:
        return []
    if payload.get("video_encoding") == "jpeg":
        from PIL import Image
        return [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in frames]
    frames = np.asarray(frames, dtype=np.uint8)
    return list(frames[None] if frames.ndim == 3 else frames)


class Gr00tColabServer:
    """Server suy luận VLA chính sách GR00T-N1.7 qua giao thức ZMQ."""

    def __init__(self, backend, host: str = "0.0.0.0", port: int = 5550):
        if not HAS_ZMQ:
            raise RuntimeError("Thiếu thư viện pyzmq! Vui lòng chạy: pip install pyzmq")
        self.backend = backend
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{self.host}:{self.port}")

    def serve(self):
        """Vòng lặp lắng nghe kết nối ZMQ từ máy Local."""
        print("\n" + "=" * 65)
        print(f"🚀 GR00T N1.7 POLICY SERVER ({self.backend.action_space}) ĐANG LẮNG NGHE TẠI: tcp://{self.host}:{self.port}")
        print("   Sẵn sàng nhận ảnh camera và câu lệnh từ máy Local...")
        print("=" * 65 + "\n")

        step_count = 0
        while True:
            try:
                msg_bytes = self.socket.recv()
            except KeyboardInterrupt:
                print("\n[Colab Server] Đang dừng server theo yêu cầu người dùng.")
                break

            # Mọi request đều phải được trả lời (REQ-REP), kể cả khi lỗi
            try:
                payload = pickle.loads(msg_bytes)
                task_desc = payload.get("language", {}).get("annotation.human.task_description", "")
                frames = decode_frames(payload)
                g1_state = payload.get("g1_state", {})
                step_progress = payload.get("step_progress", 0.0)

                step_count += 1
                t0 = time.time()
                action = self.backend.predict(task_desc, frames, g1_state, step_progress)
                dt_ms = (time.time() - t0) * 1000
                img_shape = frames[-1].shape if frames else None
                print(
                    f"[{step_count:04d}] '{task_desc[:40]}' | {len(frames)} khung hình {img_shape} | "
                    f"suy luận {dt_ms:.0f} ms"
                )
                response = {
                    "action": action,
                    "action_space": self.backend.action_space,
                    "inference_time_ms": dt_ms,
                    "step": step_count,
                }
            except Exception as e:
                traceback.print_exc()
                response = {"error": f"{type(e).__name__}: {e}"}
            self.socket.send(pickle.dumps(response))

        self.socket.close()
        self.context.term()
        print("[Colab Server] Đã đóng ZMQ socket an toàn.")


def start_pinggy_tunnel(port: int):
    """Mở đường hầm TCP qua Pinggy (Miễn phí, 0 token, 0 cần đăng ký)."""
    import subprocess
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-p", "443",
        f"-R0:localhost:{port}",
        "tcp@free.pinggy.io",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        print("\n[Pinggy Tunnel] Đang kết nối mạng Pinggy TCP (Không cần tài khoản/token)...")
        pub_url = None
        for _ in range(30):
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.2)
                continue
            if "tcp://" in line:
                parts = line.strip().split("tcp://")
                pub_url = parts[1].split()[0].replace("/", "")
                break

        if pub_url and ":" in pub_url:
            host, remote_port = pub_url.split(":")
            print("\n" + "=" * 65)
            print("🌐 ĐƯỜNG HẦM PINGGY TCP THIẾT LẬP THÀNH CÔNG (KHÔNG CẦN NGROK)!")
            print(f"   Public Address : tcp://{pub_url}")
            print(f"   Host           : {host}")
            print(f"   Port           : {remote_port}")
            print("=" * 65)
            print("👉 LỆNH CHẠY TRÊN MÁY LOCAL CỦA BẠN:")
            print(f"   .venv/bin/python launch_vla_simulation.py \\")
            print(f"       --policy-mode remote \\")
            print(f"       --server-host {host} \\")
            print(f"       --server-port {remote_port}")
            print("=" * 65 + "\n")
        else:
            print("[Pinggy Tunnel] Đang duy trì SSH Pinggy tunnel ở chế độ nền.")
    except Exception as e:
        print(f"[Cảnh báo Pinggy] Không thể mở Pinggy tunnel: {e}")


def main():
    parser = argparse.ArgumentParser(description="Chạy GR00T N1.7 Policy Server trên Colab.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Địa chỉ host")
    parser.add_argument("--port", type=int, default=5550, help="Cổng TCP ZMQ")
    parser.add_argument(
        "--policy",
        type=str,
        default="gr00t",
        choices=["gr00t", "scripted"],
        help="gr00t: model GR00T thật (REAL_G1); scripted: quỹ đạo mẫu dạng motion token",
    )
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.7-3B", help="Model repo hoặc path")
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default=DEFAULT_EMBODIMENT_TAG,
        help="Embodiment của checkpoint: mặc định REAL_G1 cho checkpoint base; new_embodiment cho checkpoint fine-tune",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if HAS_TORCH and torch.cuda.is_available() else "cpu",
        help="Thiết bị chạy model",
    )
    parser.add_argument("--ngrok-token", type=str, default=None, help="Ngrok authtoken để mở tunnel")
    parser.add_argument("--pinggy", action="store_true", help="Sử dụng Pinggy TCP tunnel (không cần tài khoản/token)")
    args = parser.parse_args()

    # 1. Mở Pinggy tunnel nếu được yêu cầu (hoặc không có ngrok token)
    if args.pinggy:
        start_pinggy_tunnel(args.port)
    # 2. Mở Ngrok tunnel nếu có token
    elif args.ngrok_token or os.environ.get("NGROK_AUTHTOKEN"):
        try:
            from pyngrok import ngrok
            token = args.ngrok_token or os.environ.get("NGROK_AUTHTOKEN")
            ngrok.set_auth_token(token)
            tunnel = ngrok.connect(args.port, "tcp")
            pub_url = tunnel.public_url.replace("tcp://", "")
            host, port = pub_url.split(":")
            print("\n" + "=" * 65)
            print("🌐 ĐƯỜNG HẦM NGROK TCP ĐÃ THIẾT LẬP THÀNH CÔNG!")
            print(f"   Public Address: {pub_url}")
            print("=" * 65)
            print("👉 LỆNH CHẠY TRÊN MÁY LOCAL CỦA BẠN:")
            print(f"   .venv/bin/python launch_vla_simulation.py \\")
            print(f"       --policy-mode remote \\")
            print(f"       --server-host {host} \\")
            print(f"       --server-port {port}")
            print("=" * 65 + "\n")
        except Exception as e:
            print(f"[Cảnh báo Ngrok] Không thể mở ngrok tunnel: {e}")

    # Nạp model trước khi mở cổng: lỗi nạp model dừng server ngay, không âm thầm chuyển sang quỹ đạo mẫu
    if args.policy == "gr00t":
        backend = Gr00tBackend(args.model_path, args.device, args.embodiment_tag)
        backend.warmup()
    else:
        backend = ScriptedBackend()

    server = Gr00tColabServer(backend, host=args.host, port=args.port)
    server.serve()


if __name__ == "__main__":
    main()

