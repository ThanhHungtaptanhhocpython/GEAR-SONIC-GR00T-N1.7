"""Vẽ lại các hình minh hoạ dữ liệu dùng trong slide, từ dataset g1_can_sort_swap_hold.

    assets/can_positions.png      vị trí ban đầu hai lon trong 400 episode, tách cùng phía / đổi phía (slide 08)
    assets/coke_stand_same_0.png  khung camera ego_view, cảnh cùng phía, lon đứng (slide 02)
    assets/pepsi_lying_swap_1.png khung camera ego_view, cảnh đổi phía, tay vươn chéo (slide 10)

Cần: numpy, pandas, matplotlib, pillow, mujoco (collect_demos dùng), imageio + imageio-ffmpeg (đọc video).
Dùng:  python report/slides/tools/make_figures.py [đường_dẫn_dataset]
"""
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
ASSETS = REPO / "report/slides/assets"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from collect_demos import CAN_HALF_LENGTH  # noqa: E402
from plot_can_positions import load_cans, xy  # noqa: E402

DATASET = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "datasets/g1_can_sort_swap_hold"
COLORS = {"coke": "#D6453D", "pepsi": "#2F6FD1"}  # đã qua kiểm tra tách màu cho người mù màu
INK, MUTED, BG = "#1B3A2D", "#6B7A72", "#F4F7F5"


def plot_can_positions():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "text.color": INK})
    cans, _, _ = load_cans(DATASET)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), facecolor=BG)
    for ax, swapped, title in zip(axes, [False, True], ["Cùng phía", "Đổi phía"]):
        ax.set_facecolor(BG)
        for side in (-1, 1):
            ax.add_patch(Rectangle((side * 0.32 - 0.06, 0.30), 0.12, 0.2, fill=False, ec="#9AA8A0", ls="--", lw=1.5))
        for c in ("coke", "pepsi"):
            rows = [r for r in cans[c] if r[3] == swapped]
            upright = np.array([p for p, lying, *_ in rows if not lying]).reshape(-1, 2)
            ax.scatter(*xy(upright), s=16, color=COLORS[c], alpha=.8, lw=0)
            segs = []
            for p, lying, yaw, _ in rows:
                if lying:
                    d = CAN_HALF_LENGTH * np.array([np.cos(yaw), np.sin(yaw)])
                    a, b = xy(np.array(p) - d), xy(np.array(p) + d)
                    segs.append([(a[0][0], a[1][0]), (b[0][0], b[1][0])])
            ax.add_collection(LineCollection(segs, colors=COLORS[c], lw=1.3, alpha=.55))
        k = sum(1 for r in cans["coke"] if r[3] == swapped)
        ax.set_title(f"{title}  ·  {k} episode", fontsize=20, color=INK, loc="left", pad=12, fontweight="bold")
        ax.text(-0.32, 0.52, "khay Pepsi", ha="center", color=MUTED, fontsize=14)
        ax.text(0.32, 0.52, "khay Coca", ha="center", color=MUTED, fontsize=14)
        ax.plot(0, 0.04, marker="^", ms=16, color=INK)
        ax.text(0, 0.075, "robot", ha="center", color=MUTED, fontsize=14)
        ax.set_xlim(-0.42, 0.42)
        ax.set_ylim(0, 0.58)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    legend = [Line2D([], [], marker="o", ls="", color=COLORS["coke"], ms=9, label="Coca"),
              Line2D([], [], marker="o", ls="", color=COLORS["pepsi"], ms=9, label="Pepsi"),
              Line2D([], [], marker="o", ls="", color=MUTED, ms=7, label="chấm = lon đứng"),
              Line2D([], [], color=MUTED, lw=2, label="vạch = lon nằm")]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=15)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ASSETS / "can_positions.png", dpi=130, facecolor=BG)


def extract_frames():
    import imageio.v3 as iio
    from PIL import Image

    meta = [json.loads(line) for line in (DATASET / "meta/collection.jsonl").read_text().splitlines() if line.strip()]

    def first(task, lying, swapped):
        return next(m["episode_index"] for m in meta
                    if m["task"] == task and m["lying"] == lying and m["swapped"] == swapped)

    # (tên file, episode, vị trí khung trong episode tính theo tỉ lệ độ dài)
    picks = [("coke_stand_same_0", first("coke", False, False), 0.0),
             ("pepsi_lying_swap_1", first("pepsi", True, True), 0.3)]
    for name, ep, t in picks:
        frames = iio.imread(DATASET / f"videos/chunk-000/observation.images.ego_view/episode_{ep:06d}.mp4")
        Image.fromarray(frames[int(t * (len(frames) - 1))]).save(ASSETS / f"{name}.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    plot_can_positions()
    extract_frames()
    print(f"Đã ghi hình vào {ASSETS}")
