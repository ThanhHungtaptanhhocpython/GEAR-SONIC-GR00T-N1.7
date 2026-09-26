"""
Chạy thử planner (collect_demos.run_episode) trên một số seed cố định, ở một hoặc nhiều chế độ chân, để đo tỉ lệ thu
được và chất lượng demo theo nhóm (đứng/nằm × cùng/đổi phía) trước khi sinh cả dataset.
Xem docs/PLAN_BALANCE_DATASET.md bước 2.

Khác collect_demos.py: không chạy tới khi đủ số episode mà chạy đúng các seed đã cho, nên hai chế độ chân so được trên
cùng cảnh. Không ghi dataset; kết quả ở <out>/trial.jsonl và <out>/summary.txt.

Ví dụ:
    .venv/bin/python tools/planner_trial.py --out eval/planner_trial_bal --seeds-per-task 40 --swap-prob 0.5 \\
        --lower-body fixed balance
"""

import argparse
import collections
import json
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import collect_demos as cd  # noqa: E402
from tools.demo_quality import summarize  # noqa: E402

GROUPS = ("cùng phía đứng", "cùng phía nằm", "đổi phía đứng", "đổi phía nằm")


def run(job) -> dict:
    task, seed, cfg, tmp_dir = job
    r = cd.run_episode((task, seed, cfg, tmp_dir))
    for ext in (".mp4", ".npz"):  # không giữ video/npz của episode thành công
        if r.get("stem"):
            Path(r["stem"] + ext).unlink(missing_ok=True)
    return {"task": task, "seed": seed, "cfg": cfg, "lower_body": cfg["lower_body"], "ok": r["ok"],
            "reason": r.get("reason"), "lying": bool(r["lying"]), "swapped": bool(r["swapped"]), "plan": r.get("plan"),
            **(r.get("quality") or {})}


def group(r) -> str:
    return f"{'đổi phía' if r['swapped'] else 'cùng phía'} {'nằm' if r['lying'] else 'đứng'}"


def reason_kind(reason: str) -> str:
    if reason.startswith("chất lượng"):
        return reason
    if reason.startswith("IK/gắp"):
        return "IK: " + reason.split(": ", 1)[1].split(" (")[0].split(":")[0][:40]
    return reason


def summarize_trial(rows: list, modes: list) -> str:
    lines = ["Tỉ lệ thu được theo nhóm (cùng seed cho mọi chế độ chân):", "",
             f"{'nhóm':18s} " + " ".join(f"{m:>14s}" for m in modes)]
    for g in ("tất cả",) + GROUPS:
        cells = []
        for m in modes:
            sub = [r for r in rows if r["lower_body"] == m and (g == "tất cả" or group(r) == g)]
            ok = sum(r["ok"] for r in sub)
            cells.append(f"{ok:4d}/{len(sub):<3d} {100 * ok / max(len(sub), 1):3.0f}%" if sub else f"{'-':>14s}")
        lines.append(f"{g:18s} " + " ".join(cells))
    for m in modes:
        lines.append(f"\nLý do bỏ, {m}:")
        for g in GROUPS:
            c = collections.Counter(reason_kind(r["reason"]) for r in rows
                                    if r["lower_body"] == m and not r["ok"] and group(r) == g)
            if c:
                lines.append(f"  {g}: " + "; ".join(f"{n} × {k}" for k, n in c.most_common()))
    for m in modes:
        kept = [{**r, "replay_ok": True} for r in rows if r["lower_body"] == m and r["ok"]]
        if kept:
            lines += ["", f"=== Chất lượng các episode thu được, {m} ===", summarize(kept)]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Chạy thử planner trên seed cố định, so các chế độ chân.")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds-per-task", type=int, default=40)
    p.add_argument("--seed", type=int, default=0, help="Seed đầu; Coca dùng seed, seed+1, ...; Pepsi cũng vậy")
    p.add_argument("--lower-body", nargs="+", default=["fixed", "balance"], choices=["fixed", "balance"])
    p.add_argument("--swap-prob", type=float, default=0.5)
    p.add_argument("--lying-prob", type=float, default=0.5)
    p.add_argument("--can-x", type=float, nargs=2, default=[0.26, 0.37])
    p.add_argument("--can-y", type=float, nargs=2, default=[0.03, 0.19])
    p.add_argument("--cross-can-y", type=float, nargs=2, default=[0.03, 0.15])
    p.add_argument("--can-gap", type=float, default=0.03)
    p.add_argument("--hold-seconds", type=float, default=2.5)
    p.add_argument("--quality-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="Loại demo theo ngưỡng chất lượng như collect_demos.py")
    p.add_argument("--lying-mode", choices=["rule", "cost"], default="rule",
                   help="Kiểu gắp lon nằm: theo quy tắc LYING_MODE_REF_DEG (mặc định) hay chi phí nhỏ nhất (như trước 25/09)")
    p.add_argument("--reuse", action="store_true", help="Chỉ in lại bảng từ <out>/trial.jsonl")
    p.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 2)))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.lying_mode == "cost":
        import sim.pick_place_planner as ppm  # tiến trình con (fork) thừa hưởng giá trị này
        ppm.LYING_MODE_REF_DEG = None

    if args.reuse:
        rows = [json.loads(line) for line in open(out / "trial.jsonl") if line.strip()]
    else:
        base = {"can_x": args.can_x, "can_y": args.can_y, "lying_prob": args.lying_prob, "can_gap": args.can_gap,
                "tray_jitter": 0.02, "joint_noise": 0.05, "swap_prob": args.swap_prob,
                "cross_can_y": args.cross_can_y, "hold_seconds": args.hold_seconds,
                "quality_filter": args.quality_filter}
        rows = []
        with tempfile.TemporaryDirectory() as tmp, Pool(args.workers, maxtasksperchild=1) as pool:
            jobs = [(t, args.seed + i, {**base, "lower_body": m}, tmp)
                    for m in args.lower_body for t in cd.TASKS for i in range(args.seeds_per_task)]
            for i, r in enumerate(pool.imap_unordered(run, jobs)):
                rows.append(r)
                if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                    print(f"  {i + 1}/{len(jobs)}", flush=True)
        rows.sort(key=lambda r: (r["lower_body"], r["task"], r["seed"]))
        (out / "trial.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    text = summarize_trial(rows, args.lower_body)
    print(text)
    (out / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
