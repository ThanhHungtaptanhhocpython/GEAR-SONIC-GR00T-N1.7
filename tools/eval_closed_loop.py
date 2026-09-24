"""
Đánh giá closed-loop model GR00T (PolicyServer ZMQ) trong sim, không mở cửa sổ 3D.

Mỗi episode đặt lon/khay/tư thế tay theo đúng ngẫu nhiên hoá của collect_demos.py (task + seed + cfg), chạy model
trong --seconds giây rồi chấm bằng cùng tiêu chí thành công như lúc thu demo. Ngoài thành công còn đo:
khoảng cách điểm gắp - tâm lon lúc tay khép (ngưỡng kẹp của sim là GRASP_RADIUS), có kẹp được lon không,
lon lên cao nhất bao nhiêu, thân robot ngả tối đa bao nhiêu độ, và số lần server không phản hồi.

Nguồn episode:
  - --dataset DIR: lấy task/seed/cfg từ meta/collection.jsonl (ví dụ tập test), đánh giá đúng các cảnh đó.
  - hoặc --seed N --episodes-per-task K với cfg từ các tham số --can-x/--can-y/... (mặc định như collect_demos.py).

Ví dụ:
    .venv/bin/python tools/eval_closed_loop.py --host <host> --port <port> --dataset datasets/g1_can_sort_ik_test \\
        --out eval/ik_test --save-video fail
    .venv/bin/python tools/eval_closed_loop.py --host <host> --port <port> --seed 200000 --episodes-per-task 10 \\
        --out eval/quick
"""

import argparse
import collections
import contextlib
import io
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import collect_demos as cd  # noqa: E402

CONTROL_HZ = 50


def grasp_distance(sim, grasp, can_adr) -> float:
    """Khoảng cách từ điểm gắp trong lòng bàn tay tới tâm lon (m), cùng cách sim quyết định bật weld kẹp."""
    d = sim.data
    w_rot = d.xmat[grasp["wrist_bid"]].reshape(3, 3)
    point = d.xpos[grasp["wrist_bid"]] + w_rot @ grasp["grasp_point"]
    return float(np.linalg.norm(point - d.qpos[can_adr:can_adr + 3]))


def pelvis_pitch_deg(qpos) -> float:
    w, x, y, z = qpos[3:7]
    return float(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))


def failure_reason(r: dict) -> str:
    if r["ok"]:
        return "thành công"
    if r["server_errors"] and r["server_errors"] >= r["policy_calls"]:
        return "server không phản hồi"
    if r["other_moved_cm"] >= 3.0 and not r["grasped"]:
        return "xô lon còn lại"
    if not r["grasped"]:
        if r["dist_at_close_cm"] is None:
            return "không khép tay"
        return "khép tay nhưng không kẹp được lon"
    if r["ok_any"]:
        return "đã đặt xong nhưng sau đó xô/gắp lại lon"
    return "kẹp được nhưng rơi hoặc đặt sai chỗ"


def run_episode(job) -> dict:
    task, seed, cfg, args = job
    from PIL import Image

    pipe_kwargs = {"policy_mode": "remote", "server_host": args["host"], "server_port": args["port"],
                   "vla_action_hz": args["hz"], "server_timeout_ms": args["timeout_ms"]}
    pipe, ctx = cd.setup_episode(task, seed, cfg, pipeline_kwargs=pipe_kwargs)
    sim, policy = pipe.sim, pipe.policy
    if policy.mode != "remote":
        pipe.close()
        raise RuntimeError(f"Không kết nối được PolicyServer tcp://{args['host']}:{args['port']}")

    # Đếm lần server lỗi (client trả None, robot giữ tư thế) và lần gọi model
    counts = collections.Counter()
    fallback, query = policy._fallback, policy._query_remote_server

    def counted_fallback(*a, **k):
        counts["server_errors"] += 1
        return fallback(*a, **k)

    def counted_query(*a, **k):
        counts["policy_calls"] += 1
        return query(*a, **k)

    policy._fallback, policy._query_remote_server = counted_fallback, counted_query

    can, can_adr = ctx["can"], ctx["can_adr"][ctx["can"]]
    grasp = sim.grasps[can]
    rand = cd.sample_randomization(task, seed, cfg)
    lying, swapped = rand["cans"][can]["lying"], rand["swapped"]
    can_start = sim.data.qpos[can_adr:can_adr + 3].copy()
    d_close = t_grasp = t_success = None
    can_max_z, max_pitch = -np.inf, 0.0
    frames = []
    for step in range(int(args["seconds"] * CONTROL_HZ)):
        dist = grasp_distance(sim, grasp, can_adr)
        was_grasped = sim.is_grasped(can)
        with contextlib.redirect_stdout(io.StringIO()):
            res = pipe.step_50hz()
        if counts["policy_calls"] == 1 and policy.server_action_space not in (None, "real_g1"):
            pipe.close()
            raise RuntimeError(f"Server không chạy model GR00T thật (action_space={policy.server_action_space})")
        if counts["policy_calls"] >= 3 and counts["server_errors"] == counts["policy_calls"]:
            pipe.close()
            raise RuntimeError(f"PolicyServer tcp://{args['host']}:{args['port']} không phản hồi 3 lần liền, dừng đánh giá")
        closure = float(np.mean(np.abs(res["motor_targets"][grasp["hand_ctrl"]][1:])))
        if d_close is None and closure > sim.GRIP_CLOSED_RAD:
            d_close = dist
        if t_grasp is None and not was_grasped and sim.is_grasped(can):
            t_grasp = step / CONTROL_HZ
        can_max_z = max(can_max_z, float(sim.data.qpos[can_adr + 2]))
        max_pitch = max(max_pitch, abs(pelvis_pitch_deg(sim.data.qpos)))
        if t_success is None and step % 25 == 0 and cd.is_success(pipe, ctx):
            t_success = step / CONTROL_HZ
        if args["save_video"] != "none" and step % cd.CONTROL_STEPS_PER_FRAME == 0:
            img = Image.fromarray(sim.render_camera("front_camera")).resize(cd.IMAGE_SIZE, Image.BOX)
            frames.append(np.asarray(img))

    d = sim.data
    other_adr = ctx["can_adr"][ctx["other"]]
    r = {
        "task": task, "seed": seed, "lying": bool(lying), "swapped": bool(swapped),
        "ok": cd.is_success(pipe, ctx), "ok_any": t_success is not None, "t_success": t_success,
        "grasped": t_grasp is not None, "t_grasp": t_grasp,
        "dist_at_close_cm": None if d_close is None else round(d_close * 100, 2),
        "can_start": can_start.round(3).tolist(), "can_final": d.qpos[can_adr:can_adr + 3].round(3).tolist(),
        "can_max_lift_cm": round(float(can_max_z - can_start[2]) * 100, 1),
        "other_moved_cm": round(float(np.linalg.norm(d.qpos[other_adr:other_adr + 3] - ctx["other_start"])) * 100, 1),
        "max_pitch_deg": round(max_pitch, 1),
        "policy_calls": counts["policy_calls"], "server_errors": counts["server_errors"],
    }
    r["reason"] = failure_reason(r)
    pipe.close()
    if frames and (args["save_video"] == "all" or not r["ok"]):
        video = Path(args["out"]) / "videos" / f"{task}_{seed:06d}_{'ok' if r['ok'] else 'fail'}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        cd.write_video(frames, video)
        r["video"] = str(video)
    return r


def load_jobs(args) -> list:
    if args.dataset:
        lines = (Path(args.dataset) / "meta" / "collection.jsonl").read_text().splitlines()
        eps = [json.loads(line) for line in lines if line.strip()]
        if args.max_episodes:
            eps = eps[: args.max_episodes]
        return [(e["task"], e["seed"], e["cfg"]) for e in eps]
    cfg = {"can_x": args.can_x, "can_y": args.can_y, "lying_prob": args.lying_prob, "can_gap": args.can_gap,
           "tray_jitter": args.tray_jitter, "joint_noise": args.joint_noise, "swap_prob": args.swap_prob,
           "cross_can_y": args.cross_can_y}
    return [(t, args.seed + i, cfg) for i in range(args.episodes_per_task) for t in args.tasks]


def summarize(results: list) -> str:
    lines = ["", f"{'nhóm':22s} {'thành công':>12s} {'kẹp được':>10s} {'k/c lúc khép (cm)':>18s}"]
    groups = collections.OrderedDict()
    for r in results:
        keys = ["tất cả", r["task"], f"{r['task']} {'nằm' if r['lying'] else 'đứng'}"]
        if any(x.get("swapped") for x in results):
            keys.append(f"{r['task']} {'đổi phía' if r.get('swapped') else 'cùng phía'}")
        for key in keys:
            groups.setdefault(key, []).append(r)
    for key in ["tất cả"] + sorted(k for k in groups if k != "tất cả"):
        g = groups[key]
        ok = sum(r["ok"] for r in g)
        grasped = sum(r["grasped"] for r in g)
        dists = [r["dist_at_close_cm"] for r in g if r["dist_at_close_cm"] is not None]
        dist = f"{np.mean(dists):.1f} (max {np.max(dists):.1f})" if dists else "-"
        lines.append(f"{key:22s} {ok:4d}/{len(g):<3d} {100 * ok / len(g):3.0f}% {grasped:6d}/{len(g):<3d} {dist:>18s}")
    lines.append("\nLý do thất bại:")
    for reason, n in collections.Counter(r["reason"] for r in results if not r["ok"]).most_common():
        lines.append(f"  {n:4d}  {reason}")
    errors = sum(r["server_errors"] for r in results)
    calls = sum(r["policy_calls"] for r in results)
    lines.append(f"\nServer: {calls} lần gọi, {errors} lần không phản hồi")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True, help="Host của PolicyServer")
    p.add_argument("--port", type=int, required=True, help="Port của PolicyServer")
    p.add_argument("--out", required=True, help="Thư mục kết quả (results.jsonl, summary.txt, videos/)")
    p.add_argument("--dataset", help="Lấy task/seed/cfg từ meta/collection.jsonl của dataset này")
    p.add_argument("--max-episodes", type=int, default=0, help="Chỉ lấy N episode đầu của --dataset (0 = tất cả)")
    p.add_argument("--tasks", nargs="+", default=list(cd.TASKS), choices=list(cd.TASKS))
    p.add_argument("--seed", type=int, default=200000, help="Seed đầu tiên khi không dùng --dataset")
    p.add_argument("--episodes-per-task", type=int, default=10)
    p.add_argument("--can-x", type=float, nargs=2, default=[0.26, 0.37])
    p.add_argument("--can-y", type=float, nargs=2, default=[0.03, 0.19])
    p.add_argument("--lying-prob", type=float, default=0.5)
    p.add_argument("--can-gap", type=float, default=0.03)
    p.add_argument("--swap-prob", type=float, default=0.0, help="Xác suất hai lon đổi phía (như collect_demos)")
    p.add_argument("--cross-can-y", type=float, nargs=2, default=[0.03, 0.15])
    p.add_argument("--tray-jitter", type=float, default=0.02)
    p.add_argument("--joint-noise", type=float, default=0.05)
    p.add_argument("--seconds", type=float, default=10.0, help="Thời gian chạy mỗi episode (demo dài 7.5 s)")
    p.add_argument("--hz", type=float, default=cd.ACTION_HZ, help="Tần số action chunk của model (dataset: 25)")
    p.add_argument("--timeout-ms", type=int, default=10000)
    p.add_argument("--workers", type=int, default=2,
                   help="Số episode chạy song song (server xử lý tuần tự, 2 giúp sim chạy trong lúc chờ GPU)")
    p.add_argument("--save-video", choices=["none", "fail", "all"], default="fail",
                   help="Ghi video camera front_camera cho episode thất bại / tất cả")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shared = {"host": args.host, "port": args.port, "hz": args.hz, "timeout_ms": args.timeout_ms,
              "seconds": args.seconds, "save_video": args.save_video, "out": str(out)}
    jobs = [(t, s, cfg, shared) for t, s, cfg in load_jobs(args)]
    print(f"Đánh giá {len(jobs)} episode trên tcp://{args.host}:{args.port} -> {out}")

    results, t0 = [], time.time()
    # Mỗi tiến trình một episode (tạo pipeline lặp lại trong cùng tiến trình bị rò rỉ bộ nhớ)
    with Pool(args.workers, maxtasksperchild=1) as pool, open(out / "results.jsonl", "w") as f:
        for r in pool.imap_unordered(run_episode, jobs):
            results.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{time.time() - t0:5.0f}s] {len(results):3d}/{len(jobs)} {r['task']:5s} seed {r['seed']:6d} "
                  f"{'nằm ' if r['lying'] else 'đứng'}{' đổi phía' if r['swapped'] else ''} "
                  f"{'OK  ' if r['ok'] else 'FAIL'} | {r['reason']} | "
                  f"k/c khép {r['dist_at_close_cm']} cm | lỗi server {r['server_errors']}", flush=True)

    summary = summarize(results)
    (out / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
