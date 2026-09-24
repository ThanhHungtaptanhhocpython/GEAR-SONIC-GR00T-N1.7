"""
Script launch_vla_simulation.py: Điểm khởi chạy mô phỏng 3D tương tác của GEAR-SONIC-GR00T-N1.7.
Tích hợp:
  - Robot Unitree G1 đứng trước bàn với 2 lon nước (Coca-Cola & Pepsi) và 2 khay phân loại.
  - VLA Policy GR00T N1.7 tiếp nhận lệnh ngôn ngữ và ảnh camera ego_view.
  - GEAR-SONIC Whole-Body Controller giải mã hành động và ổn định thăng bằng toàn thân ở 50 Hz.
  - Cửa sổ đồ họa tương tác 3D MuJoCo Viewer hỗ trợ phím nóng điều khiển trong thời gian thực:
      [1]     : Chuyển sang nhiệm vụ Coca-Cola
      [2]     : Chuyển sang nhiệm vụ Pepsi
      [SPACE] : Tạm dừng / Tiếp tục mô phỏng
      [R]     : Đặt lại (Reset) robot và lon nước về vị trí ban đầu (cảnh ngẫu nhiên hiện tại nếu có)
      [N]     : Cảnh ngẫu nhiên kế tiếp (seed + 1, hoặc episode kế tiếp của --scene-dataset)
      [S]     : Chụp và lưu ảnh snapshot từ camera ego_view & front_camera
      [ESC]   : Thoát chương trình

Mặc định lon ở vị trí cố định trong scene XML. Muốn đặt cảnh ngẫu nhiên như lúc thu demo (collect_demos.py):
  --scene-seed N                        : lấy mẫu vị trí/hướng lon, khay đích, nhiễu khớp tay theo seed N
                                          (--swap-prob, --lying-prob chỉnh cách lấy mẫu)
  --scene-dataset DIR --scene-episode K : dựng lại đúng cảnh của episode K trong dataset (kể cả nhiệm vụ)
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

# Thiết lập bảng mã UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import mujoco
import mujoco.viewer
from bridge.closed_loop_pipeline import ClosedLoopVLAPipeline
from bridge.real_g1_adapter import DEFAULT_ACTION_HZ
# Import sau mujoco: collect_demos đặt MUJOCO_GL=egl (nếu chưa đặt), chỉ có tác dụng với lần import mujoco đầu tiên
import collect_demos


TASK_PROMPTS = {
    "1": "Pick up the Coca-Cola can and place it in the red tray",
    "2": "Pick up the Pepsi can and place it in the blue tray",
}
TASK_OF_KEY = {"1": "coke", "2": "pepsi"}


class SceneSource:
    """Nguồn cảnh ngẫu nhiên: theo seed (cfg như collect_demos.py) hoặc theo từng episode của một dataset."""

    def __init__(self, args):
        self.index = 0
        self.episodes = None
        if args.scene_dataset:
            lines = (Path(args.scene_dataset) / "meta" / "collection.jsonl").read_text().splitlines()
            self.episodes = [json.loads(line) for line in lines if line.strip()]
            self.index = args.scene_episode
        self.seed = args.scene_seed
        self.cfg = {"can_x": [0.26, 0.37], "can_y": [0.03, 0.19], "lying_prob": args.lying_prob, "can_gap": 0.03,
                    "tray_jitter": 0.02, "joint_noise": 0.05, "swap_prob": args.swap_prob,
                    "cross_can_y": [0.03, 0.15]}

    def current(self, task: str):
        """(task, seed, cfg) của cảnh hiện tại; dataset thì nhiệm vụ lấy theo episode."""
        if self.episodes is not None:
            e = self.episodes[self.index % len(self.episodes)]
            return e["task"], e["seed"], e["cfg"]
        return task, self.seed, self.cfg

    def advance(self):
        if self.episodes is not None:
            self.index += 1
        else:
            self.seed += 1

    def describe(self, task: str) -> str:
        task, seed, cfg = self.current(task)
        rand = collect_demos.sample_randomization(task, seed, cfg)
        can = rand["cans"][task]
        where = f"episode {self.index % len(self.episodes)} " if self.episodes is not None else ""
        return (f"{where}seed {seed}: lon {task} {'nằm' if can['lying'] else 'đứng'} tại "
                f"({can['pos'][0]:.3f}, {can['pos'][1]:.3f}){', đổi phía' if rand['swapped'] else ''}")


def save_snapshots(sim):
    """Lưu ảnh chụp từ cả 2 camera (ego_view và front_camera)."""
    try:
        from PIL import Image
        snap_dir = Path(ROOT_DIR) / "snapshots"
        snap_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        ego_img = sim.render_camera("ego_view")
        front_img = sim.render_camera("front_camera")

        ego_path = snap_dir / f"ego_view_{ts}.png"
        front_path = snap_dir / f"front_view_{ts}.png"

        Image.fromarray(ego_img).save(ego_path)
        Image.fromarray(front_img).save(front_path)

        print(f"\n📸 [PHÍM S] Đã lưu snapshot thành công:")
        print(f"   -> {ego_path}")
        print(f"   -> {front_path}\n")
    except Exception as e:
        print(f"\n[!] Lỗi khi lưu snapshot: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Mô phỏng tương tác 3D GEAR-SONIC-GR00T-N1.7 trên MuJoCo."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="1",
        choices=["1", "2"],
        help="Nhiệm vụ ban đầu: 1 (Coca-Cola -> Khay Đỏ), 2 (Pepsi -> Khay Xanh)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Câu lệnh nhiệm vụ tùy ý (ghi đè --task)",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="standalone",
        choices=["standalone", "remote"],
        help="Chế độ Policy: standalone (chạy trực tiếp) hoặc remote (kết nối ZMQ PolicyServer)",
    )
    parser.add_argument(
        "--server-host",
        type=str,
        default="localhost",
        help="Địa chỉ host của PolicyServer nếu dùng mode remote",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=5550,
        help="Cổng port của PolicyServer nếu dùng mode remote",
    )
    parser.add_argument(
        "--ssh-tunnel",
        type=str,
        default=None,
        help="Lệnh SSH từ Colab/Tmate (Ví dụ: 'ssh xxx@nyc1.tmate.io'). Tự động mở đường hầm SSH và kết nối remote.",
    )
    parser.add_argument(
        "--server-timeout-ms",
        type=int,
        default=5000,
        help="Thời gian chờ tối đa một lần truy vấn PolicyServer (ms). Tăng lên nếu GPU Colab chậm (ví dụ T4).",
    )
    parser.add_argument(
        "--vla-action-hz",
        type=float,
        default=DEFAULT_ACTION_HZ,
        help="Tần số dữ liệu của action chunk từ model GR00T thật (REAL_G1); dùng để nội suy về 50 Hz.",
    )
    parser.add_argument(
        "--record-vla",
        type=str,
        default=None,
        help="Ghi state/action/ảnh của mỗi lần gọi model GR00T thật ra file .npz khi thoát (để phân tích).",
    )
    parser.add_argument("--scene-seed", type=int, default=None,
                        help="Đặt cảnh ngẫu nhiên theo seed như collect_demos.py (mặc định: vị trí cố định trong XML)")
    parser.add_argument("--scene-dataset", type=str, default=None,
                        help="Dựng lại cảnh theo meta/collection.jsonl của dataset này (ghi đè --task)")
    parser.add_argument("--scene-episode", type=int, default=0, help="Episode của --scene-dataset")
    parser.add_argument("--swap-prob", type=float, default=0.5,
                        help="Với --scene-seed: xác suất hai lon đổi phía (như collect_demos.py)")
    parser.add_argument("--lying-prob", type=float, default=0.5,
                        help="Với --scene-seed: xác suất mỗi lon nằm")
    parser.add_argument(
        "--wbc-mode",
        type=str,
        default="kinematic",
        choices=["kinematic", "onnx"],
        help="Chế độ giải mã WBC: kinematic (đứng vững & gắp lon ổn định, mặc định) hoặc onnx (mạng nơ-ron GEAR-SONIC).",
    )
    args = parser.parse_args()

    # Tự động mở SSH tunnel nếu có --ssh-tunnel
    ssh_proc = None
    if args.ssh_tunnel:
        args.policy_mode = "remote"
        args.server_host = "localhost"
        ssh_target = args.ssh_tunnel.strip()
        if ssh_target.startswith("ssh "):
            ssh_target = ssh_target[4:].strip()

        print("\n" + "=" * 70)
        print(f"🔐 [SSH TUNNEL] Đang tự động kết nối SSH port forwarding tới {ssh_target}...")
        import subprocess
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-L", f"{args.server_port}:localhost:{args.server_port}",
            "-N",
            ssh_target,
        ]
        try:
            ssh_proc = subprocess.Popen(ssh_cmd)
            time.sleep(2)
            print(f"✅ [SSH TUNNEL] Đường hầm SSH đã sẵn sàng: localhost:{args.server_port} <--> Colab:{args.server_port}")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"❌ [SSH TUNNEL] Lỗi khởi động SSH: {e}")

    scenes = SceneSource(args) if args.scene_seed is not None or args.scene_dataset else None
    current_task = TASK_OF_KEY[args.task]
    if scenes is not None and scenes.episodes is not None:
        current_task = scenes.current(current_task)[0]
    key_of_task = {t: k for k, t in TASK_OF_KEY.items()}
    init_prompt = args.prompt if args.prompt else TASK_PROMPTS[key_of_task[current_task]]

    print("\n" + "=" * 70)
    print("  KHỞI CHẠY MÔ PHỎNG TƯƠNG TÁC: GEAR-SONIC-GR00T-N1.7 TRÊN MUJOCO 3D")
    print("=" * 70)
    print(f"  * Nhiệm vụ ban đầu : {init_prompt}")
    print(f"  * Chế độ Policy    : {args.policy_mode.upper()}")
    print(f"  * Chế độ WBC       : {args.wbc_mode.upper()} (Đứng vững & Thao tác)")
    if args.policy_mode == "remote":
        print(f"  * PolicyServer     : tcp://{args.server_host}:{args.server_port}")
    print("=" * 70)

    # 1. Khởi tạo Pipeline khép kín
    pipeline = ClosedLoopVLAPipeline(
        task_prompt=init_prompt,
        policy_mode=args.policy_mode,
        server_host=args.server_host,
        server_port=args.server_port,
        wbc_mode=args.wbc_mode,
        vla_action_hz=args.vla_action_hz,
        server_timeout_ms=args.server_timeout_ms,
        record_vla_path=args.record_vla,
    )

    def apply_scene():
        """Reset rồi đặt cảnh ngẫu nhiên hiện tại (nếu có); dataset thì chuyển luôn sang nhiệm vụ của episode."""
        nonlocal current_task
        pipeline.reset()
        if scenes is None:
            return
        task, seed, cfg = scenes.current(current_task)
        if task != current_task:
            current_task = task
            pipeline.set_task_prompt(TASK_PROMPTS[key_of_task[task]])
        collect_demos.apply_randomization(pipeline, task, collect_demos.sample_randomization(task, seed, cfg))
        print(f"\n🎲 Cảnh {scenes.describe(current_task)}")

    apply_scene()

    print("\n🎮 PHÍM NÓNG ĐIỀU KHIỂN TRONG CỬA SỔ 3D:")
    print("  [1]     : Chuyển sang nhiệm vụ 'Gắp lon Coca-Cola sang khay đỏ' (Dùng tay phải)")
    print("  [2]     : Chuyển sang nhiệm vụ 'Gắp lon Pepsi sang khay xanh' (Dùng tay trái)")
    print("  [SPACE] : Tạm dừng (Pause) / Tiếp tục (Resume) mô phỏng")
    print("  [R]     : Đặt lại (Reset) robot và lon nước về vị trí ban đầu (cảnh ngẫu nhiên hiện tại nếu có)")
    if scenes is not None:
        print("  [N]     : Cảnh ngẫu nhiên kế tiếp (phím N cũng bật/tắt hiển thị constraint của viewer)")
    print("  [S]     : Chụp và lưu ảnh snapshot từ camera robot vào ./snapshots/")
    print("  [ESC]   : Thoát chương trình")
    print("  Chuột trái: Xoay 3D | Chuột phải: Di chuyển | Cuộn chuột: Phóng to/Thu nhỏ\n")

    # Biến cờ hiệu điều khiển để đảm bảo Thread Safety tuyệt đối giữa GLFW thread và Simulation thread
    pending_prompt = None
    reset_requested = False
    next_scene_requested = False
    snapshot_requested = False

    # Hàm callback xử lý phím bấm trong cửa sổ MuJoCo 3D Viewer (Chạy trên GLFW GUI thread)
    def key_callback(keycode: int):
        nonlocal pending_prompt, reset_requested, snapshot_requested, next_scene_requested, current_task
        # Phím '1' (ASCII 49): Coca-Cola
        if keycode in (ord("1"), 49):
            pending_prompt = TASK_PROMPTS["1"]
            current_task = "coke"
            print(f"\n👉 [PHÍM 1] ĐÃ CHUYỂN NHIỆM VỤ: '{pending_prompt}'")
        # Phím '2' (ASCII 50): Pepsi
        elif keycode in (ord("2"), 50):
            pending_prompt = TASK_PROMPTS["2"]
            current_task = "pepsi"
            print(f"\n👉 [PHÍM 2] ĐÃ CHUYỂN NHIỆM VỤ: '{pending_prompt}'")
        # Phím N / n: cảnh ngẫu nhiên kế tiếp
        elif scenes is not None and keycode in (ord("n"), ord("N"), 78, 110):
            next_scene_requested = True
        # Phím SPACE (ASCII 32): Pause / Resume
        elif keycode in (ord(" "), 32):
            pipeline.is_paused = not pipeline.is_paused
            st = "TẠM DỪNG (PAUSED)" if pipeline.is_paused else "TIẾP TỤC (RESUMED)"
            print(f"\n⏸️  [SPACE] Trạng thái mô phỏng: {st}")
        # Phím R / r (ASCII 82, 114): Reset
        elif keycode in (ord("r"), ord("R"), 82, 114):
            reset_requested = True
        # Phím S / s (ASCII 83, 115): Chụp ảnh
        elif keycode in (ord("s"), ord("S"), 83, 115):
            snapshot_requested = True

    model = pipeline.sim.model
    data = pipeline.sim.data

    try:
        # 2. Khởi chạy MuJoCo Passive Viewer kèm bộ lắng nghe phím bấm key_callback
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            # Căn chỉnh camera góc nhìn trực diện vào Robot G1 và bàn làm việc
            try:
                viewer.cam.lookat[0] = 0.35
                viewer.cam.lookat[1] = 0.0
                viewer.cam.lookat[2] = 0.75
                viewer.cam.distance = 2.1
                viewer.cam.azimuth = 145.0
                viewer.cam.elevation = -20.0
            except Exception:
                pass

            last_time = time.time()
            print_timer = 0.0

            while viewer.is_running():
                step_start = time.time()

                # viewer.lock() chỉ được giữ khi ghi vào MjData (reset, bước vật lý); suy luận policy,
                # truy vấn ZMQ và render camera chạy ngoài lock để cửa sổ 3D không bị đứng hình.
                if pending_prompt is not None:
                    pipeline.set_task_prompt(pending_prompt)
                    pending_prompt = None

                if reset_requested:
                    reset_requested = False
                    with viewer.lock():
                        apply_scene()
                    print("\n🔄 [PHÍM R] Đã reset an toàn môi trường robot và lon nước về vị trí ban đầu.")

                if next_scene_requested:
                    next_scene_requested = False
                    scenes.advance()
                    with viewer.lock():
                        apply_scene()

                if snapshot_requested:
                    snapshot_requested = False
                    save_snapshots(pipeline.sim)

                # Bước 1 chu kỳ 50 Hz (20 ms)
                res = pipeline.step_50hz(physics_lock=viewer.lock)

                # Đồng bộ hiển thị giao diện 3D Viewer
                viewer.sync()

                # In thông tin trạng thái định kỳ mỗi 1.0 giây
                if not res.get("paused", False):
                    print_timer += 0.02
                    if print_timer >= 1.0:
                        print_timer = 0.0
                        coke_z = res["coke_pos"][2]
                        pepsi_z = res["pepsi_pos"][2]
                        print(
                            f"[T={res['sim_time']:05.2f}s | Step {res['total_steps']:04d}] "
                            f"Chunk #{res['chunk_idx']:02d} | "
                            f"Coke Z={coke_z:.3f}m, Pepsi Z={pepsi_z:.3f}m | "
                            f"Task: {pipeline.task_prompt[:35]}..."
                        )

                # Điều tiết tốc độ thực tế (Real-time pacing @ 50 Hz)
                dt_step = 0.02
                elapsed = time.time() - step_start
                sleep_time = dt_step - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        pipeline.close()
        if ssh_proc is not None:
            print("[SSH Tunnel] Đang đóng đường hầm SSH...")
            ssh_proc.terminate()
            ssh_proc.wait()
        print("\nĐã đóng mô phỏng MuJoCo.")


if __name__ == "__main__":
    main()
