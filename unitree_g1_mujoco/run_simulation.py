"""
Mô phỏng Robot Unitree G1 và bài toán phân loại Lon Pepsi & Coca-Cola trên MuJoCo.
Tác giả: Hỗ trợ tự động bởi Antigravity.
"""

import time
import math
import mujoco
import mujoco.viewer
from pathlib import Path

def main():
    # 1. Đường dẫn file scene
    scene_path = Path(__file__).parent / "unitree_g1" / "scene_pepsi_coke_sorting.xml"
    if not scene_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file scene tại: {scene_path}")

    print(f"Đang nạp mô hình từ: {scene_path}...")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # 2. Khởi tạo vị trí ban đầu:
    # - Lưu vị trí ban đầu của lon Coca và Pepsi (các khớp freejoint)
    coke_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "coke_joint")
    pepsi_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pepsi_joint")
    coke_adr = model.jnt_qposadr[coke_jid]
    pepsi_adr = model.jnt_qposadr[pepsi_jid]
    
    coke_init_qpos = data.qpos[coke_adr:coke_adr+7].copy()
    pepsi_init_qpos = data.qpos[pepsi_adr:pepsi_adr+7].copy()

    # - Áp dụng tư thế đứng chuẩn (stand keyframe) cho Unitree G1
    stand_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_key_id >= 0:
        robot_dofs = len(model.key_ctrl[stand_key_id])
        data.qpos[:robot_dofs] = model.key_qpos[stand_key_id][:robot_dofs]
        data.ctrl[:robot_dofs] = model.key_ctrl[stand_key_id]
        
        # Đảm bảo 2 lon nước nằm đúng trên mặt bàn
        data.qpos[coke_adr:coke_adr+7] = coke_init_qpos
        data.qpos[pepsi_adr:pepsi_adr+7] = pepsi_init_qpos
        print("-> Đã thiết lập Unitree G1 đứng trước bàn với 2 lon Pepsi & Coca-Cola.")
    else:
        mujoco.mj_resetData(model, data)

    # Khớp vai và khuỷu tay
    right_shoulder_pitch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch_joint")
    left_shoulder_pitch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch_joint")
    right_elbow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_elbow_joint")
    left_elbow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_elbow_joint")

    print("\n" + "="*65)
    print(" CỬA SỔ MÔ PHỎNG MUJOCO 3D ĐANG KHỞI CHẠY...")
    print(" - Chuột trái: Xoay góc nhìn 3D (Rotate)")
    print(" - Chuột phải: Di chuyển góc nhìn (Pan)")
    print(" - Cuộn chuột: Phóng to / Thu nhỏ (Zoom)")
    print(" - Giữ Ctrl + Chuột phải: Kéo rê trực tiếp lon nước hoặc tay robot")
    print(" - Nhấn phím SPACE để tạm dừng / tiếp tục mô phỏng")
    print(" - Nhấn phím ESC trên cửa sổ 3D để thoát")
    print("="*65 + "\n")

    # 3. Khởi chạy cửa sổ tương tác MuJoCo
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            sim_time = data.time

            # Chuyển động minh họa nhẹ của 2 cánh tay vươn về phía bàn
            wave_r = 0.12 * math.sin(sim_time * 1.2)
            wave_l = 0.12 * math.cos(sim_time * 1.2)
            
            if right_shoulder_pitch >= 0:
                data.ctrl[right_shoulder_pitch] = model.key_ctrl[stand_key_id][right_shoulder_pitch] + wave_r
            if left_shoulder_pitch >= 0:
                data.ctrl[left_shoulder_pitch] = model.key_ctrl[stand_key_id][left_shoulder_pitch] + wave_l
            if right_elbow >= 0:
                data.ctrl[right_elbow] = model.key_ctrl[stand_key_id][right_elbow] + 0.08 * math.sin(sim_time * 1.5)
            if left_elbow >= 0:
                data.ctrl[left_elbow] = model.key_ctrl[stand_key_id][left_elbow] + 0.08 * math.cos(sim_time * 1.5)

            # Tính toán bước vật lý
            mujoco.mj_step(model, data)

            # Đồng bộ hình ảnh hiển thị
            viewer.sync()

            # Giữ đúng tốc độ real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("Đã đóng mô phỏng MuJoCo.")

if __name__ == "__main__":
    main()
