"""
Script inspect_robot_trajectory.py: Công cụ can thiệp và giám sát động học chuyển động robot Unitree G1 trong MuJoCo.
Trích xuất dữ liệu đo đạc (telemetry) chi tiết theo từng mili-giây:
- Tọa độ 3D của cổ tay / bàn tay robot (End-effector X, Y, Z)
- Tọa độ 3D của lon nước Coca-Cola / Pepsi
- Góc các khớp quan trọng: Hông (Waist), Vai (Shoulder), Khuỷu tay (Elbow), Cổ tay (Wrist)
- Trạng thái kẹp giữ lon (Grasped: True / False)
- Tự động xuất ảnh chụp các giai đoạn chính vào ./snapshots/inspection_task{1,2}/
"""

import argparse
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bridge.closed_loop_pipeline import ClosedLoopVLAPipeline


def inspect_task(task_id: str = "1", save_images: bool = True):
    is_coke = (task_id == "1")
    task_name = "Gắp lon Coca-Cola -> Khay đỏ" if is_coke else "Gắp lon Pepsi -> Khay xanh"
    prompt = (
        "Pick up the Coca-Cola can and place it in the red tray"
        if is_coke
        else "Pick up the Pepsi can and place it in the blue tray"
    )

    print("=" * 80)
    print(f"📊 GIÁM SÁT ĐỘNG HỌC MUJOCO: {task_name.upper()}")
    print("=" * 80)

    pipeline = ClosedLoopVLAPipeline(
        task_prompt=prompt,
        policy_mode="standalone",
        wbc_mode="kinematic",
    )
    sim = pipeline.sim

    target_adr = sim.coke_adr if is_coke else sim.pepsi_adr
    target_name = "Coca-Cola" if is_coke else "Pepsi"
    wrist_bid = sim.r_wrist_bid if is_coke else sim.l_wrist_bid
    wrist_name = "Tay phải (Right Wrist)" if is_coke else "Tay trái (Left Wrist)"

    snap_dir = Path(ROOT_DIR) / "snapshots" / f"inspection_task{task_id}"
    if save_images:
        snap_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        (0,   "01_READY",      "Tư thế sẵn sàng ban đầu trên mặt bàn"),
        (70,  "02_REACH",      "Vươn tay mở rộng đón thân lon"),
        (120, "03_GRASP",      "Ngón tay khép chặt ôm khít lon nước"),
        (200, "04_LIFT_CARRY", "Nhấc bổng lon và xoay người sang khay"),
        (270, "05_PLACE",      "Hạ thấp lon và mở ngón tay đặt vào khay"),
        (340, "06_RETRACT",    "Thu tay về lại tư thế sẵn sàng ban đầu"),
    ]

    header = (
        f"{'Giai đoạn':<18} | {'Bước':<5} | {'Thời gian':<7} | "
        f"{'Cổ tay (X, Y, Z) m':<22} | {f'{target_name} (X, Y, Z) m':<22} | {'Kẹp giữ':<7}"
    )
    print(header)
    print("-" * len(header))

    stage_idx = 0
    total_steps = 350

    for step in range(total_steps):
        res = pipeline.step_50hz()

        if stage_idx < len(stages) and step == stages[stage_idx][0]:
            s_step, s_code, s_desc = stages[stage_idx]
            hand_pos = sim.data.xpos[wrist_bid]
            obj_pos = sim.data.qpos[target_adr : target_adr + 3]
            is_grasped = sim.is_coke_grasped if is_coke else sim.is_pepsi_grasped

            hand_str = f"[{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]"
            obj_str = f"[{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]"
            grasp_str = "CÓ (YES)" if is_grasped else "CHƯA"

            print(
                f"{s_code:<18} | {step:<5d} | {res['sim_time']:04.2f}s  | "
                f"{hand_str:<22} | {obj_str:<22} | {grasp_str:<7}"
            )

            if save_images:
                img = sim.render_camera("front_camera")
                img_path = snap_dir / f"{s_code.lower()}.png"
                Image.fromarray(img).save(img_path)

            stage_idx += 1

    print("-" * len(header))
    final_pos = sim.data.qpos[target_adr : target_adr + 3]
    print(f"\n🎯 TỔNG KẾT:")
    print(f"  * Tọa độ cuối của {target_name}: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    tray_pos = "[0.40, -0.32, 0.685]" if is_coke else "[0.40, +0.32, 0.685]"
    print(f"  * Tọa độ khay chỉ định      : {tray_pos}")
    if save_images:
        print(f"  * Đã chụp và lưu 6 ảnh góc nhìn trực diện vào: {snap_dir}/\n")

    pipeline.close()


def main():
    parser = argparse.ArgumentParser(description="Can thiệp và đọc dữ liệu động học robot G1 trong MuJoCo.")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["1", "2", "all"],
        help="Nhiệm vụ cần giám sát: 1 (Coca-Cola), 2 (Pepsi), all (cả 2)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Không lưu ảnh chụp màn hình",
    )
    args = parser.parse_args()

    save_img = not args.no_images

    if args.task in ("1", "all"):
        inspect_task("1", save_images=save_img)
    if args.task in ("2", "all"):
        inspect_task("2", save_images=save_img)


if __name__ == "__main__":
    main()

