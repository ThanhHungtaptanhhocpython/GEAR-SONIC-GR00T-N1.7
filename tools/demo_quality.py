"""
Đo chất lượng demo trong dataset bằng cách phát lại action (như verify_demos.py) và ghi chỉ số của
sim/episode_metrics.py cho từng episode: khoảng hở tới lon còn lại theo pha, chạm lon còn lại, thân ngả/trượt/xoay,
thân dưới chạm bàn, độ lệch điểm gắp lúc khép tay. Xem docs/PLAN_BALANCE_DATASET.md bước 0b.

Có thể nối với kết quả eval closed-loop (tools/eval_closed_loop.py) theo (task, seed) để xem chỉ số của demo có
phân biệt episode model làm được và làm hỏng không.

Ví dụ:
    .venv/bin/python tools/demo_quality.py datasets/g1_can_sort_swap_hold_test \\
        --eval eval/B9000_swap_test_20s_stop
    .venv/bin/python tools/demo_quality.py datasets/g1_can_sort_swap_hold --workers 12
    # Chỉ in lại bảng từ file đã đo
    .venv/bin/python tools/demo_quality.py datasets/g1_can_sort_swap_hold_test --reuse --eval eval/B9000_swap_test_20s_stop
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import collect_demos as cd  # noqa: E402


def can_gap_cm(task: str, seed: int, cfg: dict) -> float:
    """Khoảng hở nhỏ nhất giữa thân hai lon lúc đầu episode (cm), theo hình chiếu trục lon trên mặt bàn."""
    rand = cd.sample_randomization(task, seed, cfg)
    pts = [cd._footprint(c["pos"][0], c["pos"][1], c["lying"], c["yaw"]) for c in rand["cans"].values()]
    return round(float(np.min(np.linalg.norm(pts[0][:, None] - pts[1][None], axis=2)) - 2 * cd.CAN_RADIUS) * 100, 2)


def measure(job) -> dict:
    import pyarrow.parquet as pq

    from sim.episode_metrics import EpisodeMetrics

    root, rec = job
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    table = pq.read_table(root / info["data_path"].format(episode_chunk=0, episode_index=rec["episode_index"]))
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)

    metrics = {}

    def observer(pipe, ctx):
        metrics["m"] = EpisodeMetrics(pipe.sim, ctx["can"], ctx["other"])
        return metrics["m"].step

    ok, _ = cd.replay_actions(rec["task"], rec["seed"], rec["cfg"], action, observer=observer)
    rand = cd.sample_randomization(rec["task"], rec["seed"], rec["cfg"])
    can = cd._task_can(rec["task"])[0]
    return {"episode_index": rec["episode_index"], "task": rec["task"], "seed": rec["seed"],
            "lying": bool(rand["cans"][can]["lying"]), "swapped": bool(rand["swapped"]),
            "can_gap_cm": can_gap_cm(rec["task"], rec["seed"], rec["cfg"]), "replay_ok": bool(ok),
            **metrics["m"].result()}


# ---------------------------------------------------------------------------------------------------------------- bảng

def group_name(r: dict) -> str:
    return f"{'đổi phía' if r['swapped'] else 'cùng phía'} {'nằm' if r['lying'] else 'đứng'}"


def groups_of(rows: list) -> dict:
    out = {"tất cả": rows}
    for side in ("cùng phía", "đổi phía"):
        out[side] = [r for r in rows if group_name(r).startswith(side)]
        for pose in ("đứng", "nằm"):
            out[f"{side} {pose}"] = [r for r in rows if group_name(r) == f"{side} {pose}"]
    return {k: v for k, v in out.items() if v}


def _vals(rows, get):
    v = [get(r) for r in rows]
    return np.array([x for x in v if x is not None], dtype=float)


def _med_low(rows, get, low=2.0) -> str:
    v = _vals(rows, get)
    if not len(v):
        return f"{'-':>13s}"
    return f"{np.median(v):5.1f} {100 * np.mean(v < low):4.0f}%<{low:g}"


def _med_max(rows, get, fmt="4.1f") -> str:
    v = _vals(rows, get)
    return f"{'-':>11s}" if not len(v) else f"{np.median(v):{fmt}} /{np.max(v):{fmt}}"


def summarize(rows: list) -> str:
    lines = [
        "Khoảng hở tới lon còn lại (cánh tay gắp + lon đang cầm), cm: trung vị và tỉ lệ dưới 2 cm.",
        "Thân: trung vị / max. Chạm bàn = số episode thân dưới (pelvis, chân, eo, torso) chạm bàn.",
        "",
        f"{'nhóm':18s} {'n':>4s} {'phát lại':>9s} {'trước kẹp':>13s} {'đang kẹp':>13s} {'chạm lon':>9s} "
        f"{'chạm bàn':>9s} {'ngả max °':>11s} {'trượt cm':>11s} {'yaw °':>11s} {'ngả cuối °':>11s} {'k/c khép':>11s} "
        f"{'yaw cuối °':>11s} {'chân xoay °':>11s} {'chân trượt':>11s}",
    ]
    for name, g in groups_of(rows).items():
        touch = sum((r["other_touch_steps"]["pre"] + r["other_touch_steps"]["hold"]) > 0 for r in g)
        table = sum(r["body_table_steps"] > 0 for r in g)
        lines.append(
            f"{name:18s} {len(g):4d} {sum(r['replay_ok'] for r in g):4d}/{len(g):<4d} "
            f"{_med_low(g, lambda r: r['clear_arm_cm']['pre'])} {_med_low(g, lambda r: r['clear_arm_cm']['hold'])} "
            f"{touch:4d} {100 * touch / len(g):3.0f}% {table:4d} {100 * table / len(g):3.0f}% "
            f"{_med_max(g, lambda r: r['max_tilt_deg'])} {_med_max(g, lambda r: r['max_slide_cm'])} "
            f"{_med_max(g, lambda r: r['max_yaw_deg'])} {_med_max(g, lambda r: r['final_tilt_deg'])} "
            f"{_med_max(g, lambda r: r['close'] and r['close']['dist_cm'])} "
            f"{_med_max(g, lambda r: abs(r['final_yaw_deg']) if r.get('final_yaw_deg') is not None else None)} "
            f"{_med_max(g, lambda r: r.get('max_foot_yaw_deg'))} {_med_max(g, lambda r: r.get('max_foot_slide_cm'))}"
        )
    yaw = _vals([r for r in rows if r["swapped"]], lambda r: r["max_yaw_deg"])
    if len(yaw):
        lines.append(f"\nĐổi phía: yaw thân lệch ≥ 7.5° ở {np.mean(yaw >= 7.5) * 100:.0f}% episode, "
                     f"p90 {np.percentile(yaw, 90):.1f}°, max {yaw.max():.1f}°")
    tbl = [r for r in rows if r["body_table_steps"] > 0]
    if tbl:
        lines.append(f"Chạm bàn: {len(tbl)} episode ({sum(r['lying'] for r in tbl)} lon nằm), "
                     f"trượt trung vị {np.median([r['max_slide_cm'] for r in tbl]):.1f} cm, "
                     f"ngả trung vị {np.median([r['max_tilt_deg'] for r in tbl]):.1f}°")
    return "\n".join(lines)


def summarize_eval(rows: list, results: list, name: str) -> str:
    res = {(r["task"], r["seed"]): r for r in results}
    joined = [(r, res[(r["task"], r["seed"])]) for r in rows if (r["task"], r["seed"]) in res]
    if not joined:
        return f"\n{name}: không có episode chung với dataset"
    lines = [f"\nNối với eval {name} ({len(joined)} episode chung). Chỉ số là của demo, ok/hỏng là của model.",
             f"{'nhóm':18s} {'kết quả':>7s} {'n':>3s} {'trước kẹp':>13s} {'đang kẹp':>13s} {'hở 2 lon':>11s} "
             f"{'chạm bàn':>8s} {'k/c khép model':>14s}"]
    for side in ("cùng phía", "đổi phía"):
        for pose in ("", " đứng", " nằm"):
            key = side + pose
            g = [(r, e) for r, e in joined if group_name(r).startswith(key)]
            for label, want in (("ok", True), ("hỏng", False)):
                sub = [r for r, e in g if e["ok"] == want]
                ev = [e for r, e in g if e["ok"] == want]
                if not sub:
                    continue
                dist = _vals(ev, lambda e: e.get("dist_at_close_cm"))
                lines.append(
                    f"{key:18s} {label:>7s} {len(sub):3d} {_med_low(sub, lambda r: r['clear_arm_cm']['pre'])} "
                    f"{_med_low(sub, lambda r: r['clear_arm_cm']['hold'])} {_med_max(sub, lambda r: r['can_gap_cm'])} "
                    f"{sum(r['body_table_steps'] > 0 for r in sub):8d} "
                    f"{(f'{np.median(dist):.1f}' if len(dist) else '-'):>14s}")
    # Tách ảnh hưởng của khoảng cách hai lon: chia ca đổi phía theo khoảng hở 2 lon
    sw = [(r, e) for r, e in joined if r["swapped"]]
    if sw:
        lines.append("\nĐổi phía, chia theo khoảng hở giữa hai lon lúc đầu:")
        lines.append(f"{'hở 2 lon':>12s} {'n':>3s} {'model ok':>9s} {'demo đang kẹp':>14s}")
        for lo, hi in ((-1, 5), (5, 10), (10, 100)):
            g = [(r, e) for r, e in sw if lo <= r["can_gap_cm"] < hi]
            if g:
                lines.append(f"{f'{max(lo, 0)}–{hi} cm' if hi < 100 else f'≥ {lo} cm':>12s} {len(g):3d} "
                             f"{sum(e['ok'] for _, e in g):4d}/{len(g):<4d} "
                             f"{_med_low([r for r, _ in g], lambda r: r['clear_arm_cm']['hold'])}")
    reasons = {}
    for r, e in joined:
        if not e["ok"]:
            reasons.setdefault(e["reason"], []).append(r)
    if reasons:
        lines.append("\nLý do hỏng của model và chỉ số demo tương ứng (trung vị khoảng hở lúc đang kẹp, cm):")
        for reason, g in sorted(reasons.items(), key=lambda x: -len(x[1])):
            hold = _vals(g, lambda r: r["clear_arm_cm"]["hold"])
            lines.append(f"  {len(g):3d}  {reason:45s} {np.median(hold) if len(hold) else float('nan'):5.1f}  "
                         f"(chạm bàn {sum(r['body_table_steps'] > 0 for r in g)})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Đo chất lượng demo bằng cách phát lại dataset.")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--out", type=str, default=None,
                        help="File jsonl kết quả (mặc định eval/demo_quality/<tên dataset>.jsonl)")
    parser.add_argument("--eval", type=str, nargs="*", default=[], help="Thư mục eval closed-loop để nối theo (task, seed)")
    parser.add_argument("--reuse", action="store_true", help="Đọc lại file --out đã có, không phát lại")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 2)),
                        help="Số tiến trình (mỗi tiến trình cần tới ~1 GB RAM)")
    args = parser.parse_args()
    root = Path(args.dataset)
    out = Path(args.out or ROOT / "eval" / "demo_quality" / f"{root.name}.jsonl")

    if args.reuse:
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    else:
        records = [json.loads(line) for line in (root / "meta" / "collection.jsonl").read_text().splitlines()
                   if line.strip()]
        if args.max_episodes:
            records = records[: args.max_episodes]
        # Mỗi tiến trình một episode rồi thoát (pipeline rò rỉ RAM khi tạo lại nhiều lần, xem verify_demos.py)
        with Pool(args.workers, maxtasksperchild=1) as pool:
            rows = []
            for i, row in enumerate(pool.imap(measure, [(str(root), r) for r in records], chunksize=1)):
                rows.append(row)
                if (i + 1) % 20 == 0 or i + 1 == len(records):
                    print(f"  đã đo {i + 1}/{len(records)}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    text = f"Dataset {root} ({len(rows)} episode)\n\n" + summarize(rows)
    for ev in args.eval:
        results = [json.loads(line) for line in (Path(ev) / "results.jsonl").read_text().splitlines() if line.strip()]
        text += "\n" + summarize_eval(rows, results, ev)
    print(text)
    out.with_suffix(".txt").write_text(text + "\n")
    print(f"\nĐã ghi {out} và {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
