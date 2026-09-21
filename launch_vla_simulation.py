"""
Script launch_vla_simulation.py: Điểm khởi chạy mô phỏng 3D tương tác của GEAR-SONIC-GR00T-N1.7.
Tích hợp:
  - Robot Unitree G1 đứng trước bàn với 2 lon nước (Coca-Cola & Pepsi) và 2 khay phân loại.
  - VLA Policy GR00T N1.7 tiếp nhận lệnh ngôn ngữ và ảnh camera ego_view.
  - GEAR-SONIC Whole-Body Controller giải mã hành động và ổn định thăng bằng toàn thân ở 50 Hz.
  - Cửa sổ đồ họa tương tác 3D MuJoCo Viewer hỗ trợ phím nóng điều khiển trong thời gian thực.
"""

import argparse
import os
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
    args = parser.parse_args()

    init_prompt = TASK_PROMPTS[args.task]

    print("\n" + "=" * 70)
    print("  KHỞI CHẠY MÔ PHỎNG TƯƠNG TÁC: GEAR-SONIC-GR00T-N1.7 TRÊN MUJOCO 3D")
    print("=" * 70)
    print(f"  * Nhiệm vụ ban đầu : {init_prompt}")
    print(f"  * Chế độ Policy    : {args.policy_mode.upper()}")
    print("=" * 70)

    # 1. Khởi tạo Pipeline khép kín
    pipeline = ClosedLoopVLAPipeline(
        task_prompt=init_prompt,
        policy_mode=args.policy_mode,
    )

    print("\nPHÍM NÓNG ĐIỀU KHIỂN TRONG CỬA SỔ 3D:")
    print("  [1]     : Chuyển sang nhiệm vụ 'Gắp lon Coca-Cola sang khay đỏ'")
    print("  [2]     : Chuyển sang nhiệm vụ 'Gắp lon Pepsi sang khay xanh'")
    print("  [SPACE] : Tạm dừng / Tiếp tục mô phỏng")
    print("  [R]     : Đặt lại (Reset) robot và lon nước về vị trí ban đầu")
    print("  [S]     : Chụp và lưu ảnh snapshot từ camera ego_view của robot")
    print("  [ESC]   : Thoát chương trình")
    print("  Chuột trái: Xoay 3D | Chuột phải: Di chuyển | Cuộn chuột: Phóng to/Thu nhỏ\n")

    model = pipeline.sim.model
    data = pipeline.sim.data

    # 2. Khởi chạy MuJoCo Passive Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
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

    pipeline.close()
    print("\nĐã đóng mô phỏng MuJoCo.")


if __name__ == "__main__":
    main()
