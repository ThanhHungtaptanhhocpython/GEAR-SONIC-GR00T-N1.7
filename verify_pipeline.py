"""
Script verify_pipeline.py: Kiểm thử tự động toàn diện hệ thống GEAR-SONIC-GR00T-N1.7.
Kiểm tra:
1. Nạp mô hình MuJoCo Unitree G1 và scene phân loại lon Coca-Cola & Pepsi.
2. Render camera ego_view (448x672) và front_camera.
3. Trích xuất vector trạng thái 43 khớp và projected gravity.
4. Bộ giải mã GEAR-SONIC WBC giải mã 64-dim motion token thành 43 motor targets.
5. VLA Policy GR00T N1.7 sinh Action Chunk [40, 78] cho cả 2 tác vụ Coke và Pepsi.
6. Chạy vòng lặp khép kín 100 bước (2 giây mô phỏng ở 50 Hz) mượt mà không lỗi.
7. Adapter REAL_G1: state/action của model GR00T thật ánh xạ đúng actuator; notebook Colab khớp server.
"""

import os
import sys

# Configure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

# Thêm thư mục hiện tại vào PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sim.g1_scene import G1CanSortingSim
from bridge.sonic_wbc_controller import SonicWBCController
from bridge.gr00t_policy_client import Gr00tN17Policy
from bridge.closed_loop_pipeline import ClosedLoopVLAPipeline
from bridge.real_g1_adapter import RealG1Adapter


def test_mujoco_scene():
    print("\n[TEST 1] Kiểm tra nạp MuJoCo G1 Scene & Render Camera...")
    sim = G1CanSortingSim()
    obs = sim.get_observation()
    
    assert "ego_view" in obs["video"], "Thiếu camera ego_view"
    assert "front_view" in obs["video"], "Thiếu camera front_view"
    
    ego_shape = obs["video"]["ego_view"].shape
    front_shape = obs["video"]["front_view"].shape
    
    assert ego_shape == (448, 672, 3), f"Sai kích thước ego_view: {ego_shape} != (448, 672, 3)"
    assert front_shape == (448, 672, 3), f"Sai kích thước front_view: {front_shape} != (448, 672, 3)"
    
    print(f"  -> Ego-view camera render OK: {ego_shape}")
    print(f"  -> Front camera render OK: {front_shape}")
    print(f"  -> Projected Gravity: {obs['state']['projected_gravity']}")
    print(f"  -> Vị trí lon Coke: {obs['coke_pos']}")
    print(f"  -> Vị trí lon Pepsi: {obs['pepsi_pos']}")
    
    sim.close()
    print("  -> TEST 1 PASSED!")


def test_sonic_wbc():
    print("\n[TEST 2] Kiểm tra bộ giải mã toàn thân GEAR-SONIC WBC...")
    wbc = SonicWBCController()
    
    token = np.random.randn(64).astype(np.float32)
    l_hand = np.zeros(7, dtype=np.float32)
    r_hand = np.ones(7, dtype=np.float32) * 0.7
    
    ctrl = wbc.decode(token, l_hand, r_hand)
    assert len(ctrl) == 43, f"Số lượng motor điều khiển {len(ctrl)} != 43"
    assert not np.isnan(ctrl).any(), "Phát hiện NaN trong tín hiệu điều khiển"
    
    print(f"  -> Giải mã 64-dim token thành 43 motor targets thành công!")
    print(f"  -> Sample arm joints: {ctrl[15:22]}")
    print("  -> TEST 2 PASSED!")


def test_gr00t_policy():
    print("\n[TEST 3] Kiểm tra chính sách VLA GR00T N1.7...")
    policy = Gr00tN17Policy(mode="standalone")
    
    fake_img = np.zeros((448, 672, 3), dtype=np.uint8)
    fake_state = {"left_leg": np.zeros(6), "waist": np.zeros(3)}
    
    # 1. Tác vụ Coca-Cola
    res_coke = policy.predict_action_chunk(
        "Pick up the Coca-Cola can and place it in the red tray",
        fake_img,
        fake_state,
        step_progress=0.2,
    )
    assert res_coke["full_action"].shape == (40, 78), f"Sai shape chunk: {res_coke['full_action'].shape}"
    assert res_coke["motion_token"].shape == (40, 64)
    assert res_coke["right_hand_joints"].shape == (40, 7)
    print(f"  -> Dự đoán Action Chunk Coca-Cola OK: shape {res_coke['full_action'].shape}")
    
    # 2. Tác vụ Pepsi
    res_pepsi = policy.predict_action_chunk(
        "Pick up the Pepsi can and place it in the blue tray",
        fake_img,
        fake_state,
        step_progress=0.4,
    )
    assert res_pepsi["full_action"].shape == (40, 78)
    print(f"  -> Dự đoán Action Chunk Pepsi OK: shape {res_pepsi['full_action'].shape}")

    # 3. Bản sao quỹ đạo mẫu trong colab_gr00t_server.py phải khớp với client
    import colab_gr00t_server
    from bridge.gr00t_policy_client import scripted_sorting_chunk
    for prompt in ("Pick up the Coca-Cola can and place it in the red tray",
                   "Pick up the Pepsi can and place it in the blue tray"):
        for progress in (0.0, 0.37, 0.95):
            assert np.allclose(
                colab_gr00t_server.scripted_sorting_chunk(prompt, progress, 40, Gr00tN17Policy.CYCLE_STEPS),
                scripted_sorting_chunk(prompt, progress, 40, Gr00tN17Policy.CYCLE_STEPS),
            ), "Quỹ đạo mẫu của colab_gr00t_server.py đã lệch so với bridge/gr00t_policy_client.py"
    print("  -> Quỹ đạo mẫu của Colab server khớp với client")
    print("  -> TEST 3 PASSED!")


def test_closed_loop_pipeline():
    print("\n[TEST 4] Kiểm tra vòng lặp kín Closed-Loop Pipeline (100 bước @ 50 Hz)...")
    pipeline = ClosedLoopVLAPipeline(
        task_prompt="Pick up the Coca-Cola can and place it in the red tray",
        policy_mode="standalone",
    )
    
    for step in range(100):
        info = pipeline.step_50hz()
        assert not np.isnan(info["motor_targets"]).any(), f"NaN tại step {step}"
        if step % 25 == 0:
            print(f"  -> Step {step:03d}: sim_time={info['sim_time']:.2f}s, chunk_idx={info['chunk_idx']}, total_steps={info['total_steps']}")
            
    # Chuyển đổi nhiệm vụ sang Pepsi
    pipeline.set_task_prompt("Pick up the Pepsi can and place it in the blue tray")
    for step in range(50):
        info = pipeline.step_50hz()
        assert not np.isnan(info["motor_targets"]).any()
        
    print(f"  -> Hoàn thành 150 bước điều khiển khép kín (3 giây vật lý) không lỗi!")
    pipeline.close()
    print("  -> TEST 4 PASSED!")


def test_real_g1_adapter():
    print("\n[TEST 5] Kiểm tra adapter REAL_G1 cho model GR00T thật...")
    sim = G1CanSortingSim()
    adapter = RealG1Adapter(sim.model)
    state = adapter.state(sim.data, sim.act_qposadr)

    dims = {k: len(v) for k, v in state.items()}
    expected = {"left_wrist_eef_9d": 9, "right_wrist_eef_9d": 9, "left_hand": 7, "right_hand": 7,
                "left_arm": 7, "right_arm": 7, "waist": 3}
    assert dims == expected, f"Sai số chiều state REAL_G1: {dims}"
    assert all(v.dtype == np.float32 for v in state.values())

    # Action tuyệt đối bằng đúng state hiện tại phải cho lại góc khớp hiện tại ở mọi actuator tay/bàn tay/eo
    horizon = 40
    action = {k: np.repeat(v[None], horizon, axis=0) for k, v in state.items() if "eef" not in k}
    ctrl = adapter.action_to_ctrl(action, sim.default_ctrl, action_hz=20.0, control_hz=50.0)
    assert ctrl.shape == (98, 43), f"Sai shape chunk sau nội suy: {ctrl.shape}"
    q = sim.data.qpos[sim.act_qposadr]
    assert np.allclose(ctrl[:, 12:], q[12:], atol=1e-5), "Ánh xạ state -> actuator của REAL_G1 bị lệch"
    assert np.allclose(ctrl[:, :12], sim.default_ctrl[:12]), "Chân phải giữ tư thế đứng mặc định"
    # Dữ liệu REAL_G1 xếp khớp bàn tay theo alphabet: chiều 0 là index_0
    assert sim.model.actuator(int(adapter.hand_idx["right"][0])).name == "right_hand_index_0_joint"
    print(f"  -> State REAL_G1 OK: {dims}")
    print(f"  -> Action chunk [40 @ 20 Hz] -> {ctrl.shape} @ 50 Hz, ánh xạ actuator khớp")
    sim.close()

    # Cell %%writefile trong các notebook Colab phải khớp với file gốc trong repo
    import json
    for notebook, source in (("colab_gr00t_server.ipynb", "colab_gr00t_server.py"),
                             ("colab_gr00t_finetune.ipynb", "finetune/g1_can_sort_config.py")):
        nb = json.load(open(os.path.join(ROOT_DIR, notebook), encoding="utf-8"))
        src = open(os.path.join(ROOT_DIR, source), encoding="utf-8").read()
        cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        writefile = [c.split("\n", 1)[1] for c in cells if c.startswith("%%writefile")]
        assert writefile and writefile[0].rstrip("\n") == src.rstrip("\n"), (
            f"Cell %%writefile trong {notebook} đã lệch so với {source}: chạy tools/build_colab_notebooks.py"
        )
        print(f"  -> {notebook} khớp với {source}")
    print("  -> TEST 5 PASSED!")


def main():
    print("=" * 65)
    print("  BẮT ĐẦU KIỂM THỬ TỰ ĐỘNG HỆ THỐNG GEAR-SONIC-GR00T-N1.7")
    print("=" * 65)
    
    test_mujoco_scene()
    test_sonic_wbc()
    test_gr00t_policy()
    test_closed_loop_pipeline()
    test_real_g1_adapter()
    
    print("\n" + "=" * 65)
    print("  TẤT CẢ CÁC BÀI KIỂM THỬ ĐÃ VƯỢT QUA 100% (ALL TESTS PASSED)!")
    print("=" * 65)


if __name__ == "__main__":
    main()
