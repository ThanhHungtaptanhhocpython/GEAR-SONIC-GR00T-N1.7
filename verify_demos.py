"""
Script verify_demos.py: Kiểm tra dataset demo do collect_demos.py tạo ra trước khi đem đi fine-tune.

1. Định dạng: số dòng parquet = số khung video = độ dài trong episodes.jsonl, số chiều state/action khớp
   modality.json, cột index/timestamp liên tục.
2. Phát lại (replay): tái tạo trạng thái đầu của episode từ seed, đưa action trong dataset qua đúng đường chạy model
   (RealG1Adapter.action_to_ctrl nội suy lên 50 Hz -> SonicWBCController.track_joint_targets -> sim) và kiểm tra lon
   vẫn được đặt vào đúng khay. Replay thất bại nghĩa là kể cả model học hoàn hảo action trong dataset cũng không
   làm được nhiệm vụ.

Ví dụ:
    .venv/bin/python verify_demos.py datasets/g1_can_sort_pilot
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from collect_demos import VIDEO_KEY, replay_actions


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_format(root: Path) -> list:
    import av
    import pyarrow.parquet as pq

    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    modality = json.loads((meta / "modality.json").read_text())
    episodes = read_jsonl(meta / "episodes.jsonl")
    n_tasks = len(read_jsonl(meta / "tasks.jsonl"))
    state_dim = max(v["end"] for v in modality["state"].values())
    action_dim = max(v["end"] for v in modality["action"].values())
    errors = []

    if info["total_episodes"] != len(episodes):
        errors.append(f"info.total_episodes={info['total_episodes']} != {len(episodes)} dòng episodes.jsonl")
    next_index = 0
    for ep in episodes:
        i, length = ep["episode_index"], ep["length"]
        table = pq.read_table(root / info["data_path"].format(episode_chunk=0, episode_index=i)).to_pydict()
        state = np.asarray(table["observation.state"], dtype=np.float32)
        action = np.asarray(table["action"], dtype=np.float32)
        video_path = root / info["video_path"].format(episode_chunk=0, video_key=VIDEO_KEY, episode_index=i)
        with av.open(str(video_path)) as container:
            frames = [f for f in container.decode(video=0)]
        h, w = frames[0].height, frames[0].width

        if not (len(state) == len(frames) == length):
            errors.append(f"episode {i}: parquet {len(state)} dòng, video {len(frames)} khung, episodes.jsonl {length}")
        if state.shape[1] != state_dim or action.shape[1] != action_dim:
            errors.append(f"episode {i}: state {state.shape[1]}/{state_dim}, action {action.shape[1]}/{action_dim} chiều")
        if [h, w, 3] != info["features"][VIDEO_KEY]["shape"]:
            errors.append(f"episode {i}: video {h}x{w} khác info.json")
        if table["index"] != list(range(next_index, next_index + len(state))):
            errors.append(f"episode {i}: cột index không liên tục")
        if not all(0 <= t < n_tasks for t in table["task_index"]):
            errors.append(f"episode {i}: task_index ngoài tasks.jsonl")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            errors.append(f"episode {i}: có NaN/inf")
        next_index += len(state)
    if info["total_frames"] != next_index:
        errors.append(f"info.total_frames={info['total_frames']} != {next_index}")
    return errors


def replay_episode(job):
    """Phát lại action của một episode qua đường chạy model; trả về (episode, thành công, lệch khớp tay lớn nhất)."""
    import pyarrow.parquet as pq

    root, rec, lower_body = job
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    modality = json.loads((root / "meta" / "modality.json").read_text())
    table = pq.read_table(
        root / info["data_path"].format(episode_chunk=0, episode_index=rec["episode_index"])
    ).to_pydict()
    action = np.asarray(table["action"], dtype=np.float32)
    state = np.asarray(table["observation.state"], dtype=np.float32)

    kwargs = {"lower_body": lower_body} if lower_body else None
    ok, replayed = replay_actions(rec["task"], rec["seed"], rec["cfg"], action, pipeline_kwargs=kwargs)
    arm = np.r_[modality["state"]["left_arm"]["start"]:modality["state"]["right_arm"]["end"]]
    return rec["episode_index"], ok, float(np.abs(replayed[:, arm] - state[:, arm]).max())


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra định dạng và phát lại dataset demo.")
    parser.add_argument("dataset", type=str, help="Thư mục dataset (tạo bởi collect_demos.py)")
    parser.add_argument("--replay", type=int, default=-1, help="Số episode phát lại (-1 = tất cả, 0 = bỏ qua)")
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 2)),
                        help="Số tiến trình (mỗi tiến trình cần tới ~1 GB RAM)")
    parser.add_argument("--lower-body", choices=["fixed", "balance"], default=None,
                        help="Ghi đè chế độ chân khi phát lại (mặc định: theo cfg lúc thu, dataset cũ là fixed)")
    args = parser.parse_args()
    root = Path(args.dataset)

    errors = check_format(root)
    print("1. Định dạng: " + ("OK" if not errors else f"{len(errors)} lỗi"))
    for e in errors[:20]:
        print("   - " + e)

    records = read_jsonl(root / "meta" / "collection.jsonl")
    if args.replay != 0:
        records = records if args.replay < 0 else records[: args.replay]
        # Mỗi tiến trình chỉ chạy một episode rồi thoát: tạo pipeline lặp lại trong cùng tiến trình rò rỉ
        # ~150 MB mỗi episode (model MuJoCo, renderer EGL, ONNX session) và làm tràn RAM
        with Pool(args.workers, maxtasksperchild=1) as pool:
            results = pool.map(replay_episode, [(str(root), r, args.lower_body) for r in records], chunksize=1)
        failed = [i for i, ok, _ in results if not ok]
        devs = np.array([dev for _, _, dev in results])
        print(f"2. Phát lại: {len(results) - len(failed)}/{len(results)} episode thành công | "
              f"lệch góc khớp tay so với dữ liệu: trung bình {devs.mean():.3f} rad, lớn nhất {devs.max():.3f} rad")
        if failed:
            print(f"   Episode thất bại: {failed[:20]}")
        errors += [f"replay thất bại: {failed}"] if failed else []

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
