"""
Vẽ trạng thái ban đầu của lon Coca/Pepsi trong dataset demo (collect_demos.py): vị trí trên mặt bàn nhìn từ trên
xuống (lon đứng là chấm, lon nằm là đoạn thẳng theo trục lon) và phân bố hướng của lon nằm.

  - Train: hai lon ở đầu mọi episode (tính lại từ seed trong meta/collection.jsonl).
  - Test: tập test độc lập (--test-dataset, seed không trùng tập train), vẽ màu đen.
  - Vị trí mặc định trong scene XML (ngôi sao): nơi lon nằm khi chạy launch_vla_simulation.py không có --scene-seed /
    --scene-dataset.

Ví dụ:
    .venv/bin/python tools/plot_can_positions.py datasets/g1_can_sort_pilot --test-dataset datasets/g1_can_sort_test
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from collect_demos import CAN_HALF_LENGTH, TASKS, sample_randomization, scene_layout  # noqa: E402

COLORS = {"coke": "#e34948", "pepsi": "#2a78d6"}
NAMES = {"coke": "Coca", "pepsi": "Pepsi"}


def load_cans(dataset):
    """Trạng thái đầu của hai lon ở mọi episode: {lon: list (pos, lying, yaw, swapped)}, cfg của episode đầu, số
    episode. swapped: episode có hai lon đổi phía (collect_demos.py --swap-prob)."""
    lines = [json.loads(line) for line in (Path(dataset) / "meta" / "collection.jsonl").read_text().splitlines()
             if line.strip()]
    cans = {c: [] for c in TASKS}
    for r in lines:
        rand = sample_randomization(r["task"], r["seed"], r["cfg"])
        for c in TASKS:
            v = rand["cans"][c]
            cans[c].append((v["pos"][:2], v["lying"], v["yaw"], rand["swapped"]))
    return cans, lines[0]["cfg"], len(lines)


def xy(p):
    """Toạ độ bàn (x tiến, y sang trái) -> ngang = sang phải robot, dọc = phía trước."""
    p = np.atleast_2d(p)
    return -p[:, 1], p[:, 0]


def draw_cans(ax, cans, color, marker, label):
    upright = np.array([p for p, lying, *_ in cans if not lying]).reshape(-1, 2)
    lying = [(p, yaw) for p, is_lying, yaw, *_ in cans if is_lying]
    if len(upright):
        ax.scatter(*xy(upright), s=14, color=color, marker=marker, lw=0.8,
                   label=f"{label}: đứng ({len(upright)})")
    if lying:
        segs = []
        for p, yaw in lying:
            d = CAN_HALF_LENGTH * np.array([np.cos(yaw), np.sin(yaw)])
            a, b = xy(p - d), xy(p + d)
            segs.append([(a[0][0], a[1][0]), (b[0][0], b[1][0])])
        ax.add_collection(LineCollection(segs, colors=color, linewidths=1.2, alpha=0.7,
                                         label=f"{label}: nằm ({len(lying)})"))


def draw_table(ax, layout, cfg, train, test, swapped, title):
    """Mặt bàn nhìn từ trên xuống với các lon của những episode có cờ đổi phía bằng `swapped`."""
    ax.add_patch(Rectangle((-0.45, 0.18), 0.90, 0.60, facecolor="#f1efe9", edgecolor="#b5b3ad"))
    half = layout["tray_half"]
    for c in TASKS:
        x, y = xy(layout["tray"][c][:2])
        ax.add_patch(Rectangle((x[0] - half, y[0] - half), 2 * half, 2 * half, fill=False, edgecolor="#8a8883",
                               ls="--", lw=1))
        ax.text(x[0], y[0] + half + 0.01, f"khay {NAMES[c]}", ha="center", fontsize=9, color="#55534f")
        if "can_x" in cfg:
            # Vùng lấy mẫu tâm lon: mỗi lon ở phía khay của nó, hoặc ở nửa bàn bên kia khi đổi phía
            side = (-1.0 if c == "coke" else 1.0) * (-1.0 if swapped else 1.0)
            (x0, x1), (y0, y1) = cfg["can_x"], cfg["can_y"]
            ys = sorted([-side * y0, -side * y1])
            ax.add_patch(Rectangle((ys[0], x0), ys[1] - ys[0], x1 - x0, fill=False, edgecolor="#8a8883", ls=":",
                                   lw=1))
        if not swapped:
            x, y = xy(layout["can"][c][:2])
            ax.scatter(x, y, s=150, marker="*", color="#2b2a28", zorder=5,
                       label="vị trí mặc định (GUI không --scene-*)" if c == "coke" else None)
    pick = lambda cans: [e for e in cans if e[3] == swapped]
    for c in TASKS:
        draw_cans(ax, pick(train[c]), COLORS[c], "o", f"{NAMES[c]} train")
        if test is not None:
            draw_cans(ax, pick(test[c]), "#2b2a28", "x", f"{NAMES[c]} test")
    ax.plot(0, 0.03, "k^", ms=10)
    ax.text(0, 0.06, "robot", ha="center", fontsize=9)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.02, 0.82)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("sang phải của robot (m)")
    ax.set_ylabel("về phía trước (m)")
    ax.legend(loc="lower left", fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Vẽ trạng thái ban đầu của lon trong dataset demo.")
    parser.add_argument("dataset", type=str, help="Thư mục dataset (tạo bởi collect_demos.py)")
    parser.add_argument("--test-dataset", type=str, default=None, help="Thư mục tập test (tạo bởi collect_demos.py)")
    parser.add_argument("--out", type=str, default=None, help="File ảnh đầu ra (mặc định <dataset>_can_positions.png)")
    args = parser.parse_args()
    root = Path(args.dataset)
    out = args.out or f"{root.as_posix().rstrip('/')}_can_positions.png"

    layout = scene_layout()
    train, cfg, n_train = load_cans(root)
    test, _, n_test = load_cans(args.test_dataset) if args.test_dataset else (None, None, 0)
    # Số episode đổi phía (mỗi episode có một mục cho mỗi lon, đếm theo lon Coca)
    n_swap = lambda cans: sum(e[3] for e in cans["coke"]) if cans is not None else 0
    has_swap = n_swap(train) + n_swap(test) > 0

    # Bản đồ mặt bàn: dataset có đổi phía thì tách hai ô cùng phía / đổi phía cho khỏi chồng lên nhau
    if has_swap:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), gridspec_kw={"width_ratios": [1.5, 1.5, 1]})
        test_note = lambda n: f", test {n}" if test is not None else ""
        draw_table(axes[0], layout, cfg, train, test, False,
                   f"Cùng phía: lon ở nửa bàn phía khay của nó (train {n_train - n_swap(train)}"
                   f"{test_note(n_test - n_swap(test))} episode)")
        draw_table(axes[1], layout, cfg, train, test, True,
                   f"Đổi phía: lon ở nửa bàn phía khay kia (train {n_swap(train)}{test_note(n_swap(test))} episode)")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.5, 1]})
        draw_table(axes[0], layout, cfg, train, test, False,
                   "Mặt bàn nhìn từ trên xuống (vùng chấm: vùng lấy mẫu tâm lon)")

    # Phân bố hướng của lon nằm: góc trục lon (đáy -> nắp) trên mặt bàn, 0° = hướng về phía trước
    ax = axes[-1]
    bins = np.arange(-180, 181, 30)
    any_lying = False
    for c in TASKS:
        yaws = np.degrees([yaw for _, lying, yaw, _ in train[c] if lying])
        if len(yaws):
            any_lying = True
            ax.hist(yaws, bins=bins, histtype="step", lw=2, color=COLORS[c], label=f"{NAMES[c]} ({len(yaws)})")
    if any_lying:
        ax.set_xticks(bins[::2])
        ax.set_xlabel("hướng trục lon nằm (độ, 0 = về phía trước, dương = sang trái robot)")
        ax.set_ylabel("số lon")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Không có lon nằm trong dataset", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title("Hướng của lon nằm (train, cả hai lon mỗi episode)")

    title = f"Trạng thái ban đầu của lon: train {root.name} ({n_train} episode)"
    if args.test_dataset:
        title += f", test {Path(args.test_dataset).name} ({n_test} episode, màu đen)"
    if has_swap:
        title += "; vùng chấm: vùng lấy mẫu tâm lon"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
