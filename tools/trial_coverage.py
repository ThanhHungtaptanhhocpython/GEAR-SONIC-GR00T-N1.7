"""
Vùng phủ của các episode planner thu được so với bị loại (docs/PLAN_BALANCE_DATASET.md bước 5).

Quota theo nhóm chỉ bù số lượng. Nếu planner (hoặc bộ lọc chất lượng) hay loại một vùng cảnh, dataset sẽ thiếu đúng vùng đó
dù đủ số episode. Script chia cảnh theo vị trí lon cần gắp (x, |y|), hướng lon nằm và khoảng hở giữa hai lon, in tỉ lệ thu
được từng ô và vẽ vị trí thu được / bị loại.

Nguồn: <out>/trial.jsonl của tools/planner_trial.py (mọi lần thử, kể cả bị loại).

Ví dụ:
    .venv/bin/python tools/trial_coverage.py eval/planner_trial_cover/trial.jsonl --swapped
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import collect_demos as cd  # noqa: E402
from tools.demo_quality import can_gap_cm  # noqa: E402

KEPT, REJECTED, INK, MUTED = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e"


def load(path: Path, swapped) -> list:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = []
    for r in rows:
        rand = cd.sample_randomization(r["task"], r["seed"], r["cfg"])
        if swapped is not None and rand["swapped"] != swapped:
            continue
        can = rand["cans"][cd._task_can(r["task"])[0]]
        out.append({"task": r["task"], "ok": r["ok"], "reason": r.get("reason") or "", "lying": can["lying"],
                    "swapped": rand["swapped"], "x": float(can["pos"][0]), "ay": float(abs(can["pos"][1])),
                    "sy": float(np.sign(can["pos"][1]) or 1.0),
                    "yaw": float(np.degrees(can["yaw"]) % 180), "gap": can_gap_cm(r["task"], r["seed"], r["cfg"])})
    return out


def rate_table(rows, key, edges, label) -> list:
    lines = [f"  {label}: " + "  ".join(
        f"{a:g}–{b:g}: {sum(r['ok'] for r in sub)}/{len(sub)}"
        for a, b in zip(edges[:-1], edges[1:])
        for sub in [[r for r in rows if a <= r[key] < b]] if sub)]
    return lines


def summarize(rows) -> str:
    lines = []
    for lying in (False, True):
        g = [r for r in rows if r["lying"] == lying]
        if not g:
            continue
        name = "lon nằm" if lying else "lon đứng"
        lines.append(f"\n{name}: thu được {sum(r['ok'] for r in g)}/{len(g)}")
        lines += rate_table(g, "x", [0.26, 0.297, 0.333, 0.371], "x (m)")
        lines += rate_table(g, "ay", [0.0, 0.07, 0.11, 0.2], "|y| (m)")
        lines += rate_table(g, "gap", [0, 5, 10, 15, 100], "hở 2 lon (cm)")
        if lying:
            lines += rate_table(g, "yaw", list(range(0, 181, 30)), "yaw gập (°)")
        reasons = collections.Counter(r["reason"].split(" (")[0] for r in g if not r["ok"])
        lines.append("  lý do loại: " + "; ".join(f"{n} × {k}" for k, n in reasons.most_common()))
    return "\n".join(lines)


def plot(rows, out: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, lying in zip(axes[:2], (False, True)):
        if not any(r["lying"] == lying for r in rows):
            ax.text(0.5, 0.5, f"không có lon {'nằm' if lying else 'đứng'}", ha="center", va="center", color=MUTED,
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        for ok, color, marker, label in ((False, REJECTED, "x", "bị loại"), (True, KEPT, "o", "thu được")):
            g = [r for r in rows if r["lying"] == lying and r["ok"] == ok]
            if not g:
                continue
            # Nhìn từ trên xuống: ngang = |y| (xa dần khỏi giữa bàn), dọc = x (ra phía trước)
            ax.scatter([r["ay"] for r in g], [r["x"] for r in g], s=26, marker=marker, color=color, lw=1.2,
                       label=f"{label} ({len(g)})")
            if lying:
                for r in g:
                    # Trục lon (cos yaw, sin yaw) trong (x, y); trục ngang là |y| nên đổi dấu thành phần y khi y < 0
                    d = 0.02 * np.array([r["sy"] * np.sin(np.radians(r["yaw"])), np.cos(np.radians(r["yaw"]))])
                    ax.plot([r["ay"] - d[0], r["ay"] + d[0]], [r["x"] - d[1], r["x"] + d[1]], color=color, lw=1.4,
                            solid_capstyle="round")
        ax.set_title(f"{'Lon nằm' if lying else 'Lon đứng'}: vị trí lon cần gắp", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("|y| tâm lon (m), 0 = giữa bàn", color=MUTED)
        ax.set_ylabel("x tâm lon (m), ra phía trước", color=MUTED)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax = axes[2]
    g = [r for r in rows if r["lying"]]
    edges = np.arange(0, 181, 20)
    ok_n = np.histogram([r["yaw"] for r in g if r["ok"]], edges)[0]
    all_n = np.histogram([r["yaw"] for r in g], edges)[0]
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, all_n - ok_n, width=17, bottom=ok_n, color=REJECTED, label="bị loại")
    ax.bar(centers, ok_n, width=17, color=KEPT, label="thu được")
    ax.set_title("Lon nằm: hướng trục lon (gập 0–180°)", fontsize=11, color=INK, loc="left")
    ax.set_xlabel("yaw (°), 0 = trục lon hướng ra trước", color=MUTED)
    ax.set_ylabel("số lần thử", color=MUTED)
    ax.set_xticks(edges[::3])
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="#fcfcfb")


def main():
    p = argparse.ArgumentParser(description="Vùng phủ của episode thu được / bị loại trong planner_trial.")
    p.add_argument("trial", type=str)
    side = p.add_mutually_exclusive_group()
    side.add_argument("--swapped", action="store_true", help="Chỉ ca đổi phía")
    side.add_argument("--same-side", action="store_true", help="Chỉ ca cùng phía")
    p.add_argument("--out", type=str, default=None, help="Ảnh (mặc định <thư mục trial>/coverage.png)")
    args = p.parse_args()
    path = Path(args.trial)
    swapped = True if args.swapped else (False if args.same_side else None)
    rows = load(path, swapped)
    text = summarize(rows)
    print(text)
    out = Path(args.out) if args.out else path.parent / "coverage.png"
    which = "đổi phía" if swapped else ("cùng phía" if swapped is False else "mọi cảnh")
    plot(rows, out, f"{path.parent.name}, {which}: {sum(r['ok'] for r in rows)}/{len(rows)} thu được")
    (out.with_suffix(".txt")).write_text(text + "\n")
    print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
