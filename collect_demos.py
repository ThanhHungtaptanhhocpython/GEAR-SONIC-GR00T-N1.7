"""
Script collect_demos.py: Thu demo phân loại lon trong MuJoCo và xuất dataset GR00T LeRobot v2 theo layout
state/action của embodiment REAL_G1 (bridge/real_g1_adapter.py), để fine-tune GR00T N1.7.

Mỗi episode: đặt hai lon ở vị trí ngẫu nhiên trong vùng tay với tới (mỗi lon ở phía tay gắp nó), đứng hoặc nằm với
hướng ngẫu nhiên; lệch nhẹ khay đích và tư thế tay ban đầu. Quỹ đạo gắp-đặt được lập bằng IK online
(sim/pick_place_planner.py) từ vị trí/hướng lon đo được, ghi ở ACTION_HZ Hz: ảnh ego_view, state đo được, action là
mục tiêu khớp trước bộ lọc làm mịn của WBC. Chỉ giữ episode thành công (lon nằm trong đúng khay, đã nhả, lon kia không
bị xô lệch) và phát lại action đã ghi qua đường chạy model (nội suy 25 -> 50 Hz, track_joint_targets) cũng thành công.

Dataset cũ (g1_can_sort_pilot/test) thu bằng quỹ đạo mẫu cố định với lon lệch ±2 cm; cfg của chúng (có khoá
can_jitter) vẫn được sample_randomization tái tạo đúng để verify_demos.py / plot_can_positions.py dùng tiếp.

Ví dụ:
    .venv/bin/python collect_demos.py --out datasets/g1_can_sort_ik --episodes-per-task 200
"""

import argparse
import collections
import contextlib
import functools
import io
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from bridge.real_g1_adapter import ACTION_DIMS, ACTION_KEYS, STATE_DIMS, STATE_KEYS
from sim.pick_place_planner import CAN_HALF_HEIGHT, CAN_RADIUS

TASKS = {
    "coke": "Pick up the Coca-Cola can and place it in the red tray",
    "pepsi": "Pick up the Pepsi can and place it in the blue tray",
}
# Tần số ghi dataset: cứ 2 bước điều khiển 50 Hz ghi một khung (khi chạy model: --vla-action-hz 25)
ACTION_HZ = 25
CONTROL_STEPS_PER_FRAME = 2
# Ảnh lưu vào video: cùng cách thu nhỏ với client khi chạy model (cạnh ngắn 256, lọc BOX)
IMAGE_SIZE = (384, 256)  # (rộng, cao)
VIDEO_KEY = "observation.images.ego_view"

# Lon nằm: nửa chiều dài thân lon tính cả nắp (m), dùng cho vùng chiếm chỗ trên mặt bàn khi lấy mẫu vị trí
CAN_HALF_LENGTH = 0.061
# Khoảng hở tối thiểu giữa lon và thành khay (m)
TRAY_MARGIN = 0.01


def _task_can(task: str):
    return ("coke", "pepsi") if task == "coke" else ("pepsi", "coke")


@functools.lru_cache(maxsize=None)
def scene_layout() -> dict:
    """Vị trí mặc định của lon, khay (body_pos), nửa kích thước khay và độ cao mặt bàn, đọc từ scene XML."""
    import mujoco

    from sim.g1_scene import DEFAULT_SCENE_XML

    m = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE_XML))
    body = lambda name: m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)].copy()
    top = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    tray_base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "coke_tray_base")
    return {
        "can": {c: body(f"{c}_can") for c in TASKS},
        "tray": {c: body(f"{c}_tray") for c in TASKS},
        "tray_half": float(m.geom_size[tray_base][0]),
        "table_top": float(body("work_table")[2] + m.geom_pos[top][2] + m.geom_size[top][2]),
    }


def can_quat(lying: bool, yaw: float) -> np.ndarray:
    """Quaternion lon: đứng, hoặc nằm với trục lon (từ đáy lên nắp) hướng yaw trên mặt bàn."""
    if not lying:
        return np.array([1.0, 0.0, 0.0, 0.0])
    # Xoay 90° quanh y (trục lon z -> x) rồi xoay yaw quanh z
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    qy = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qy
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _footprint(x: float, y: float, lying: bool, yaw: float) -> np.ndarray:
    """Các điểm trên trục lon chiếu xuống mặt bàn; vùng chiếm chỗ = các điểm này nới thêm bán kính lon."""
    half = CAN_HALF_LENGTH if lying else 0.0
    t = np.linspace(-half, half, 7)[:, None]
    return np.array([x, y]) + t * np.array([np.cos(yaw), np.sin(yaw)])


def _clear_of_tray(points: np.ndarray, tray_xy: np.ndarray, tray_half: float) -> bool:
    gap = np.maximum(np.abs(points - tray_xy) - tray_half, 0.0)
    return bool(np.all(np.linalg.norm(gap, axis=1) > CAN_RADIUS + TRAY_MARGIN))


def sample_randomization(task: str, seed: int, cfg: dict) -> dict:
    """Trạng thái đầu ngẫu nhiên của episode (tái tạo được từ task + seed + cfg).

    Returns:
        tray: độ lệch x/y của khay đích (m); cans: {lon: {pos, quat, lying, yaw}} tuyệt đối trong khung world;
        joints: nhiễu khớp hai tay ban đầu (rad).
    """
    rng = np.random.default_rng(seed)
    layout = scene_layout()
    can, other = _task_can(task)
    tray = rng.uniform(-cfg["tray_jitter"], cfg["tray_jitter"], 2)

    cans = {}
    if "can_jitter" in cfg:
        # Dataset cũ: lon đứng, lệch nhẹ quanh vị trí mặc định (giữ đúng thứ tự gọi rng để tái tạo được)
        for c in (can, other):
            pos = layout["can"][c].copy()
            pos[:2] += rng.uniform(-cfg["can_jitter"], cfg["can_jitter"], 2)
            cans[c] = {"pos": pos, "quat": can_quat(False, 0.0), "lying": False, "yaw": 0.0}
    else:
        trays = [layout["tray"][c][:2] + (tray if c == can else 0.0) for c in TASKS]
        placed = []
        for c in (can, other):
            side = -1.0 if c == "coke" else 1.0
            for _ in range(1000):
                x = rng.uniform(*cfg["can_x"])
                y = side * rng.uniform(*cfg["can_y"])
                lying = bool(rng.random() < cfg["lying_prob"])
                yaw = float(rng.uniform(-np.pi, np.pi))
                pts = _footprint(x, y, lying, yaw)
                if not all(_clear_of_tray(pts, t, layout["tray_half"]) for t in trays):
                    continue
                if any(np.min(np.linalg.norm(pts[:, None] - p[None], axis=2)) < 2 * CAN_RADIUS + cfg["can_gap"]
                       for p in placed):
                    continue
                break
            else:
                raise RuntimeError(f"Không tìm được chỗ đặt lon {c} (seed {seed}); nới rộng vùng --can-x/--can-y")
            placed.append(pts)
            z = layout["table_top"] + (CAN_RADIUS if lying else CAN_HALF_HEIGHT) + 0.0005
            cans[c] = {"pos": np.array([x, y, z]), "quat": can_quat(lying, yaw), "lying": lying, "yaw": yaw}

    joints = {side: rng.normal(0.0, cfg["joint_noise"], 7) for side in ("left", "right")}
    return {"tray": tray, "cans": cans, "joints": joints}


def setup_episode(task: str, seed: int, cfg: dict, pipeline_kwargs: Optional[dict] = None):
    """Tạo pipeline và áp dụng ngẫu nhiên hoá của episode (tái tạo được từ task + seed + cfg).

    pipeline_kwargs: tham số thêm cho ClosedLoopVLAPipeline (ví dụ policy_mode="remote" khi đánh giá model).
    """
    import mujoco

    from bridge.closed_loop_pipeline import ClosedLoopVLAPipeline

    rand = sample_randomization(task, seed, cfg)
    kwargs = {"policy_mode": "standalone", **(pipeline_kwargs or {})}
    with contextlib.redirect_stdout(io.StringIO()):
        pipe = ClosedLoopVLAPipeline(task_prompt=TASKS[task], **kwargs)
    sim, adapter = pipe.sim, pipe.g1_adapter
    m, d = sim.model, sim.data
    can, other = _task_can(task)
    can_adr = {"coke": sim.coke_adr, "pepsi": sim.pepsi_adr}
    tray_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{can}_tray")

    # Ngẫu nhiên hoá: vị trí/hướng hai lon, khay đích, tư thế hai tay ban đầu
    m.body_pos[tray_bid][:2] += rand["tray"]
    for c in (can, other):
        d.qpos[can_adr[c]:can_adr[c] + 3] = rand["cans"][c]["pos"]
        d.qpos[can_adr[c] + 3:can_adr[c] + 7] = rand["cans"][c]["quat"]
    for side in ("left", "right"):
        d.qpos[sim.act_qposadr[adapter.arm_idx[side]]] += rand["joints"][side]
    mujoco.mj_forward(m, d)
    ctx = {"can": can, "other": other, "can_adr": can_adr, "tray_bid": tray_bid,
           "other_start": d.qpos[can_adr[other]:can_adr[other] + 3].copy()}
    return pipe, ctx


def is_success(pipe, ctx) -> bool:
    """Lon nằm trong đúng khay, đã nhả, và lon còn lại không bị xô lệch quá 3 cm."""
    d = pipe.sim.data
    can_adr, can, other = ctx["can_adr"], ctx["can"], ctx["other"]
    can_pos = d.qpos[can_adr[can]:can_adr[can] + 3]
    tray_pos = d.xpos[ctx["tray_bid"]]
    return bool(
        np.all(np.abs(can_pos[:2] - tray_pos[:2]) < 0.08)
        and can_pos[2] < tray_pos[2] + 0.1
        and not pipe.sim.is_grasped(can)
        and np.linalg.norm(d.qpos[can_adr[other]:can_adr[other] + 3] - ctx["other_start"]) < 0.03
    )


def replay_actions(task: str, seed: int, cfg: dict, actions: np.ndarray):
    """Phát lại action [N, 53] qua đường chạy model từ cùng trạng thái đầu của episode.

    Returns:
        (thành công, state đo được tại mỗi khung [N, 49]).
    """
    pipe, ctx = setup_episode(task, seed, cfg)
    sim, adapter = pipe.sim, pipe.g1_adapter
    layout, _ = modality_layout(ACTION_KEYS, ACTION_DIMS)
    action_dict = {k: actions[:, v["start"]:v["end"]] for k, v in layout.items()}
    ctrl = adapter.action_to_ctrl(action_dict, sim.default_ctrl, ACTION_HZ, pipe.control_hz)
    # Giữ mục tiêu cuối thêm một bước cho đủ độ dài episode gốc (chunk nội suy ngắn hơn 1 bước)
    ctrl = np.concatenate([ctrl, ctrl[-1:]])
    states = []
    for i, target in enumerate(ctrl):
        if i % CONTROL_STEPS_PER_FRAME == 0:
            state = adapter.state(sim.data, sim.act_qposadr)
            states.append(np.concatenate([state[k] for k in STATE_KEYS]))
        sim.step(pipe.wbc.track_joint_targets(target), n_substeps=pipe.substeps_per_wbc)
    ok = is_success(pipe, ctx)
    pipe.close()
    return ok, np.asarray(states, np.float32)


def run_episode(job):
    """Chạy một episode.

    Returns:
        dict có "ok"; thành công thì kèm metadata (video + npz ở thư mục tạm), thất bại thì kèm "reason".
    """
    import mujoco
    from PIL import Image

    from sim.pick_place_planner import PickPlacePlanner, PlanningError

    task, seed, cfg, tmp_dir = job
    pipe, ctx = setup_episode(task, seed, cfg)
    sim, adapter = pipe.sim, pipe.g1_adapter
    d = sim.data
    scratch = mujoco.MjData(sim.model)
    planner = PickPlacePlanner(sim, ctx["can"])
    frames, states, actions = [], [], []
    base = {"task": task, "seed": seed, "lying": sample_randomization(task, seed, cfg)["cans"][ctx["can"]]["lying"]}
    try:
        # Generator lập kế hoạch từng đoạn từ trạng thái hiện tại: mục tiêu của bước k tính từ trạng thái trước bước k
        for step, target in enumerate(planner.targets()):
            record = step % CONTROL_STEPS_PER_FRAME == 0
            if record:
                img = Image.fromarray(sim.render_camera("ego_view")).resize(IMAGE_SIZE, Image.BOX)
                frames.append(np.asarray(img))
                state = adapter.state(d, sim.act_qposadr)
                states.append(np.concatenate([state[k] for k in STATE_KEYS]))
            sim.step(pipe.wbc.track_joint_targets(target), n_substeps=pipe.substeps_per_wbc)
            if record:
                action = adapter.action_from_ctrl(d, sim.act_qposadr, pipe.wbc.last_target, scratch)
                actions.append(np.concatenate([action[k] for k in ACTION_KEYS]))
    except PlanningError as e:
        pipe.close()
        return {**base, "ok": False, "reason": f"IK/gắp: {e}"}

    success = is_success(pipe, ctx)
    pipe.close()
    if not success:
        return {**base, "ok": False, "reason": "đặt lon không thành công"}
    if not replay_actions(task, seed, cfg, np.asarray(actions, np.float32))[0]:
        return {**base, "ok": False, "reason": "phát lại thất bại"}

    stem = Path(tmp_dir) / f"{task}_{seed:06d}"
    write_video(frames, stem.with_suffix(".mp4"))
    np.savez(stem.with_suffix(".npz"), state=np.asarray(states, np.float32), action=np.asarray(actions, np.float32))
    return {**base, "ok": True, "stem": str(stem), "length": len(frames),
            "palm_tilt_deg": round(planner.info.get("palm_tilt_deg", 0.0), 1)}


def write_video(frames, path: Path):
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=ACTION_HZ)
        stream.width, stream.height = IMAGE_SIZE
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "g": "10"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_episode_parquet(path: Path, state, action, episode_index: int, task_index: int, global_start: int):
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(state)
    table = pa.table({
        "observation.state": pa.array(list(state), type=pa.list_(pa.float32())),
        "action": pa.array(list(action), type=pa.list_(pa.float32())),
        "timestamp": pa.array(np.arange(n, dtype=np.float32) / ACTION_HZ),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
        "index": pa.array(np.arange(global_start, global_start + n, dtype=np.int64)),
        "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
        "annotation.human.task_description": pa.array(np.full(n, task_index, dtype=np.int64)),
        "next.reward": pa.array(np.zeros(n, dtype=np.float32)),
        "next.done": pa.array(np.arange(n) == n - 1),
    })
    pq.write_table(table, path)


def modality_layout(keys, dims):
    layout, start = {}, 0
    for k in keys:
        layout[k] = {"start": start, "end": start + dims[k]}
        start += dims[k]
    return layout, start


def write_meta(out: Path, episodes, cfg: dict):
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    task_names = list(TASKS.values())
    state_layout, state_dim = modality_layout(STATE_KEYS, STATE_DIMS)
    action_layout, action_dim = modality_layout(ACTION_KEYS, ACTION_DIMS)

    with open(meta / "tasks.jsonl", "w") as f:
        for i, t in enumerate(task_names):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    with open(meta / "episodes.jsonl", "w") as f:
        for i, ep in enumerate(episodes):
            f.write(json.dumps({"episode_index": i, "tasks": [TASKS[ep["task"]]], "length": ep["length"]}) + "\n")

    # Seed và cấu hình ngẫu nhiên hoá của từng episode, để tái tạo trạng thái đầu khi phát lại (verify_demos.py);
    # kèm tư thế lon cần gắp và độ nghiêng lòng bàn tay lúc gắp để tiện phân tích
    with open(meta / "collection.jsonl", "w") as f:
        for i, ep in enumerate(episodes):
            f.write(json.dumps({
                "episode_index": i, "task": ep["task"], "seed": ep["seed"], "cfg": cfg,
                "lying": ep["lying"], "palm_tilt_deg": ep["palm_tilt_deg"],
            }) + "\n")

    modality = {
        "state": state_layout,
        "action": action_layout,
        "video": {"ego_view": {"original_key": VIDEO_KEY}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=4))

    def names(layout):
        return [f"{k}.{i}" for k, v in layout.items() for i in range(v["end"] - v["start"])]

    scalar = lambda dtype: {"dtype": dtype, "shape": [1], "names": None}
    info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1_dex3",
        "total_episodes": len(episodes),
        "total_frames": int(sum(ep["length"] for ep in episodes)),
        "total_tasks": len(task_names),
        "total_videos": len(episodes),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": ACTION_HZ,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim], "names": names(state_layout)},
            "action": {"dtype": "float32", "shape": [action_dim], "names": names(action_layout)},
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [IMAGE_SIZE[1], IMAGE_SIZE[0], 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": IMAGE_SIZE[1], "video.width": IMAGE_SIZE[0], "video.codec": "h264",
                    "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": ACTION_HZ,
                    "video.channels": 3, "has_audio": False,
                },
            },
            "timestamp": scalar("float32"),
            "frame_index": scalar("int64"),
            "episode_index": scalar("int64"),
            "index": scalar("int64"),
            "task_index": scalar("int64"),
            "annotation.human.task_description": scalar("int64"),
            "next.reward": scalar("float32"),
            "next.done": scalar("bool"),
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=4))


def main():
    parser = argparse.ArgumentParser(description="Thu demo phân loại lon và xuất dataset GR00T LeRobot v2.")
    parser.add_argument("--out", type=str, default="datasets/g1_can_sort_pilot", help="Thư mục dataset đầu ra")
    parser.add_argument("--episodes-per-task", type=int, default=50, help="Số episode thành công cho mỗi lon")
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 2)),
                        help="Số tiến trình (mỗi tiến trình cần tới ~1 GB RAM)")
    parser.add_argument("--seed", type=int, default=0, help="Seed khởi đầu")
    parser.add_argument("--can-x", type=float, nargs=2, default=[0.26, 0.37], metavar=("MIN", "MAX"),
                        help="Khoảng x (về phía trước) của tâm lon (m)")
    parser.add_argument("--can-y", type=float, nargs=2, default=[0.03, 0.19], metavar=("MIN", "MAX"),
                        help="Khoảng |y| của tâm lon, mỗi lon ở phía tay gắp nó (m)")
    parser.add_argument("--lying-prob", type=float, default=0.5, help="Xác suất mỗi lon nằm (hướng ngẫu nhiên)")
    parser.add_argument("--can-gap", type=float, default=0.03, help="Khoảng hở tối thiểu giữa hai lon (m)")
    parser.add_argument("--tray-jitter", type=float, default=0.02, help="Lệch tối đa vị trí khay đích (m)")
    parser.add_argument("--joint-noise", type=float, default=0.05, help="Độ lệch chuẩn nhiễu khớp tay ban đầu (rad)")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"Thư mục {out} đã tồn tại; chọn --out khác hoặc xoá thư mục cũ.")
    tmp_dir = out / "_tmp"
    tmp_dir.mkdir(parents=True)
    cfg = {"can_x": args.can_x, "can_y": args.can_y, "lying_prob": args.lying_prob, "can_gap": args.can_gap,
           "tray_jitter": args.tray_jitter, "joint_noise": args.joint_noise}

    # Chạy xen kẽ hai nhiệm vụ; số lần thử mỗi đợt ước theo tỉ lệ thành công đã đo để bù episode thất bại
    kept = {t: [] for t in TASKS}
    attempts = {t: 0 for t in TASKS}
    # Thống kê theo (lon, đứng/nằm): số lần thử, số lần thành công, lý do thất bại
    stats = collections.defaultdict(lambda: {"tries": 0, "ok": 0, "fail": collections.Counter()})
    seed = args.seed
    t0 = time.time()
    # Mỗi tiến trình chỉ chạy một episode rồi thoát: tạo pipeline lặp lại trong cùng tiến trình rò rỉ
    # ~150 MB mỗi episode (model MuJoCo, renderer EGL, ONNX session) và làm tràn RAM
    with Pool(args.workers, maxtasksperchild=1) as pool:
        while any(len(kept[t]) < args.episodes_per_task for t in TASKS):
            jobs = []
            for t in TASKS:
                need = args.episodes_per_task - len(kept[t])
                rate = len(kept[t]) / attempts[t] if attempts[t] >= 20 else 0.5
                for _ in range(int(np.ceil(need / max(rate, 0.2) * 1.1)) if need > 0 else 0):
                    jobs.append((t, seed, cfg, str(tmp_dir)))
                    seed += 1
            for res in pool.imap(run_episode, jobs):
                attempts[res["task"]] += 1
                st = stats[(res["task"], "nằm" if res["lying"] else "đứng")]
                st["tries"] += 1
                if not res["ok"]:
                    st["fail"][res["reason"].split(" (")[0]] += 1
                    continue
                st["ok"] += 1
                if len(kept[res["task"]]) < args.episodes_per_task:
                    kept[res["task"]].append(res)
            print(
                f"[{time.time() - t0:5.0f}s] "
                + " | ".join(f"{t}: {len(kept[t])}/{args.episodes_per_task} (thử {attempts[t]})" for t in TASKS)
            )

    # Đánh số episode liên tục, xen kẽ hai nhiệm vụ
    episodes = [ep for pair in zip(*(sorted(kept[t], key=lambda e: e["seed"]) for t in TASKS)) for ep in pair]
    data_dir = out / "data" / "chunk-000"
    video_dir = out / "videos" / "chunk-000" / VIDEO_KEY
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    task_index = {t: i for i, t in enumerate(TASKS)}
    global_start = 0
    for i, ep in enumerate(episodes):
        arrays = np.load(ep["stem"] + ".npz")
        write_episode_parquet(
            data_dir / f"episode_{i:06d}.parquet", arrays["state"], arrays["action"], i,
            task_index[ep["task"]], global_start,
        )
        shutil.move(ep["stem"] + ".mp4", video_dir / f"episode_{i:06d}.mp4")
        global_start += ep["length"]

    sample = np.load(episodes[0]["stem"] + ".npz")
    assert sample["state"].shape[1] == sum(STATE_DIMS.values())
    assert sample["action"].shape[1] == sum(ACTION_DIMS.values())
    write_meta(out, episodes, cfg)
    shutil.rmtree(tmp_dir)

    print(f"\nĐã ghi {len(episodes)} episode ({global_start} khung hình) vào {out}")
    lying = sum(ep["lying"] for ep in episodes)
    print(f"Lon cần gắp: {len(episodes) - lying} đứng, {lying} nằm")
    print("Tỉ lệ thành công khi thu:")
    for (t, pose), st in sorted(stats.items()):
        print(f"  {t:5s} {pose}: {st['ok']}/{st['tries']}")
        for reason, n in st["fail"].most_common():
            print(f"      {n:4d} x {reason}")


if __name__ == "__main__":
    main()
