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
      [R]     : Đặt lại (Reset) robot và lon nước về vị trí ban đầu
      [S]     : Chụp và lưu ảnh snapshot từ camera ego_view & front_camera
      [ESC]   : Thoát chương trình
"""

import argparse
import datetime
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


TASK_PROMPTS = {
    "1": "Pick up the Coca-Cola can and place it in the red tray",
    "2": "Pick up the Pepsi can and place it in the blue tray",
}


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

    init_prompt = args.prompt if args.prompt else TASK_PROMPTS[args.task]

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
    )

    print("\n🎮 PHÍM NÓNG ĐIỀU KHIỂN TRONG CỬA SỔ 3D:")
    print("  [1]     : Chuyển sang nhiệm vụ 'Gắp lon Coca-Cola sang khay đỏ' (Dùng tay phải)")
    print("  [2]     : Chuyển sang nhiệm vụ 'Gắp lon Pepsi sang khay xanh' (Dùng tay trái)")
    print("  [SPACE] : Tạm dừng (Pause) / Tiếp tục (Resume) mô phỏng")
    print("  [R]     : Đặt lại (Reset) robot và lon nước về vị trí ban đầu")
    print("  [S]     : Chụp và lưu ảnh snapshot từ camera robot vào ./snapshots/")
    print("  [ESC]   : Thoát chương trình")
    print("  Chuột trái: Xoay 3D | Chuột phải: Di chuyển | Cuộn chuột: Phóng to/Thu nhỏ\n")

    # Hàm callback xử lý phím bấm trong cửa sổ MuJoCo 3D Viewer
    def key_callback(keycode: int):
        # Phím '1' (ASCII 49): Coca-Cola
        if keycode in (ord("1"), 49):
            prompt = TASK_PROMPTS["1"]
            pipeline.set_task_prompt(prompt)
            print(f"\n👉 [PHÍM 1] ĐÃ CHUYỂN NHIỆM VỤ: '{prompt}'")
        # Phím '2' (ASCII 50): Pepsi
        elif keycode in (ord("2"), 50):
            prompt = TASK_PROMPTS["2"]
            pipeline.set_task_prompt(prompt)
            print(f"\n👉 [PHÍM 2] ĐÃ CHUYỂN NHIỆM VỤ: '{prompt}'")
        # Phím SPACE (ASCII 32): Pause / Resume
        elif keycode in (ord(" "), 32):
            pipeline.is_paused = not pipeline.is_paused
            st = "TẠM DỪNG (PAUSED)" if pipeline.is_paused else "TIẾP TỤC (RESUMED)"
            print(f"\n⏸️  [SPACE] Trạng thái mô phỏng: {st}")
        # Phím R / r (ASCII 82, 114): Reset
        elif keycode in (ord("r"), ord("R"), 82, 114):
            pipeline.reset()
            print(f"\n🔄 [PHÍM R] Đã reset môi trường robot và lon nước về vị trí ban đầu.")
        # Phím S / s (ASCII 83, 115): Chụp ảnh
        elif keycode in (ord("s"), ord("S"), 83, 115):
            save_snapshots(pipeline.sim)

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

                # Bước 1 chu kỳ 50 Hz (20 ms)
                res = pipeline.step_50hz()

                # Đồng bộ giao diện 3D Viewer
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
