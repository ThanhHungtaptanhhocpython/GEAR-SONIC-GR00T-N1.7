"""Vẽ các hình thống kê bộ dữ liệu Bones-SEED từ file metadata (seed_metadata_v004.parquet).

    report/figures/bones_seed/package.png    số clip và số giờ theo 8 package
    report/figures/bones_seed/category.png   số clip theo 20 category
    report/figures/bones_seed/duration.png   phân bố độ dài clip (giây)
    report/figures/bones_seed/actors.png     tuổi, chiều cao, cân nặng, giới tính, nghề, số clip mỗi diễn viên
    report/figures/bones_seed/movement.png   loại chuyển động và tư thế cơ thể phổ biến nhất
    report/figures/bones_seed/flags.png      tỉ lệ clip theo các cờ nội dung
    report/figures/bones_seed/timeline.png   số clip theo tháng ghi hình

Số clip tính cả bản lật gương (is_mirror), trừ hình diễn viên tính theo từng người.
Hình không có tiêu đề chung vì caption nằm trong report/Nvidia_groot.tex.
Cần: pandas, pyarrow, matplotlib.
Dùng:  uv run --no-project --with pandas --with pyarrow --with matplotlib \
           python tools/plot_bones_seed_metadata.py [đường_dẫn_parquet] [--out THƯ_MỤC]
"""
import argparse
import pathlib

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
FPS = 120  # Bones-SEED ghi mocap ở 120 khung hình mỗi giây
W = 6.3  # inch, bằng bề rộng chữ của report (16 cm), để chèn \textwidth không phải co chữ
BAR, INK, MUTED, GRID, BG = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def vn(v, digits=0):
    """Định dạng số kiểu Việt Nam: dấu chấm ngăn hàng nghìn, dấu phẩy thập phân."""
    return f"{v:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(v, total):
    p = 100 * v / total
    return "<0,1%" if 0 < p < 0.05 else f"{vn(p, 1)}%"


THOUSANDS = FuncFormatter(lambda v, _: vn(v))


def style():
    # Times New Roman cài riêng cho người dùng (cùng font với report) không nằm trong cache của matplotlib
    for f in pathlib.Path.home().glob(".local/share/fonts/**/times*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": BG, "axes.facecolor": BG,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def hbar(ax, counts, total=None, digits=0, unit=""):
    """Cột ngang, giá trị lớn nhất ở trên cùng, nhãn số đặt cuối cột (kèm % nếu có total)."""
    counts = counts[::-1]
    ax.barh(counts.index.astype(str), counts.values, color=BAR, height=0.7)
    for y, v in enumerate(counts.values):
        label = vn(v, digits) + unit
        if total:
            label += f" ({pct(v, total)})"
        ax.text(v, y, " " + label, va="center", fontsize=8, color=MUTED)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_xlim(0, counts.max() * 1.35)


def hist(ax, values, bins, xlabel):
    ax.hist(values, bins=bins, color=BAR, edgecolor=BG, lw=0.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    med = values.median()
    ax.axvline(med, color=INK, lw=0.9, ls="--")
    ax.text(med, ax.get_ylim()[1] * 0.97, f" trung vị {vn(med, 1).removesuffix(',0')}", fontsize=8, va="top")


def save(fig, out, name):
    fig.savefig(out / name)
    plt.close(fig)
    print("  ", out / name)


def plot_package(df, out):
    counts = df.package.value_counts()
    hours = (df.groupby("package").move_duration_frames.sum() / FPS / 3600).reindex(counts.index)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
    hbar(a, counts, total=len(df))
    a.set_title(f"Số clip (tổng {vn(len(df))})")
    hbar(b, hours, total=hours.sum(), digits=1, unit=" giờ")
    b.set_title(f"Số giờ chuyển động (tổng {vn(hours.sum(), 1)} giờ)")
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "package.png")


def plot_category(df, out):
    fig, ax = plt.subplots(figsize=(W, 3.9))
    hbar(ax, df.category.value_counts(), total=len(df))
    save(fig, out, "category.png")


def plot_duration(df, out):
    sec = df.move_duration_frames / FPS
    cap = 30
    fig, ax = plt.subplots(figsize=(W, 2.4))
    hist(ax, sec.clip(upper=cap), bins=60, xlabel="Độ dài clip (giây)")
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set_ylabel("Số clip")
    n_long = int((sec > cap).sum())
    ax.text(cap, ax.get_ylim()[1] * 0.55, f"{vn(n_long)} clip dài hơn {cap} s\ngộp vào cột cuối ",
            ha="right", fontsize=8, color=MUTED)
    save(fig, out, "duration.png")


def plot_actors(df, out):
    actors = df.drop_duplicates("actor_uid").set_index("actor_uid")
    per_actor = df.groupby("actor_uid").size()
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.2))
    hist(axes[0, 0], actors.actor_age_yr, bins=range(15, 75, 3), xlabel="Tuổi")
    hist(axes[0, 1], actors.actor_height_cm, bins=range(144, 204, 4), xlabel="Chiều cao (cm)")
    hist(axes[0, 2], actors.actor_weight_kg, bins=range(35, 150, 7), xlabel="Cân nặng (kg)")
    hbar(axes[1, 0], actors.actor_gender.map({"F": "Nữ", "M": "Nam"}).value_counts(), total=len(actors))
    axes[1, 0].set_xlabel("Giới tính")
    hbar(axes[1, 1], actors.actor_profession.value_counts(), total=len(actors))
    axes[1, 1].set_xlabel("Nghề")
    hist(axes[1, 2], per_actor, bins=range(0, 950, 50), xlabel="Số clip của một diễn viên")
    for ax, tag in zip(axes.flat, "abcdef"):
        ax.set_title(f"({tag})", fontweight="normal")
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    save(fig, out, "actors.png")


def plot_movement(df, out):
    moves, pos = df.content_type_of_movement, df.content_body_position
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.4), gridspec_kw={"width_ratios": [1, 1]})
    hbar(a, moves.value_counts().head(12), total=len(df))
    a.set_title(f"(a) 12 kiểu chuyển động nhiều nhất / {moves.nunique()}")
    hbar(b, pos.value_counts().head(8), total=len(df))
    b.set_title(f"(b) 8 tư thế nhiều nhất / {pos.nunique()}")
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "movement.png")


def plot_flags(df, out):
    flags = pd.Series({
        "Di chuyển theo phương ngang": df.content_horizontal_move.mean(),
        "Di chuyển theo phương dọc": df.content_vertical_move.mean(),
        "Clip lật gương": df.is_mirror.mean(),
        "Có đạo cụ hoặc đồ vật": (df.content_props.astype(str) != "0").mean(),
        "Phong cách khác neutral": (df.content_uniform_style != "neutral").mean(),
        "Hành động phức hợp": df.content_complex_action.mean(),
        "Hành động lặp lại": df.content_repeated_action.mean(),
    }) * 100
    fig, ax = plt.subplots(figsize=(W, 2.3))
    hbar(ax, flags.sort_values(ascending=False), digits=1, unit="%")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlim(0, 100)
    save(fig, out, "flags.png")


def plot_timeline(df, out):
    month = pd.to_datetime(df.take_date.astype(str), format="%y%m%d").dt.to_period("M")
    counts = month.value_counts().sort_index()
    counts = counts.reindex(pd.period_range(counts.index.min(), counts.index.max(), freq="M"), fill_value=0)
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.bar(range(len(counts)), counts.values, color=BAR, width=0.8)
    ticks = [i for i, p in enumerate(counts.index) if p.month in (1, 7)]
    ax.set_xticks(ticks, [counts.index[i].strftime("%m/%Y") for i in ticks])
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set_ylabel("Số clip")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out, "timeline.png")


def summary(df):
    sec = df.move_duration_frames / FPS
    print(f"{len(df):,} clip × {df.shape[1]} cột, {df.actor_uid.nunique()} diễn viên")
    print(f"tổng {sec.sum() / 3600:.1f} giờ ({sec[~df.is_mirror].sum() / 3600:.1f} giờ không tính bản lật gương)")
    print(f"clip: trung bình {sec.mean():.1f} s, trung vị {sec.median():.1f} s, từ {sec.min():.1f} đến {sec.max():.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="?", default=REPO / "seed_metadata_v004.parquet", type=pathlib.Path)
    ap.add_argument("--out", default=REPO / "report/figures/bones_seed", type=pathlib.Path)
    args = ap.parse_args()
    df = pd.read_parquet(args.parquet)
    args.out.mkdir(parents=True, exist_ok=True)
    summary(df)
    style()
    for plot in (plot_package, plot_category, plot_duration, plot_actors, plot_movement, plot_flags, plot_timeline):
        plot(df, args.out)


if __name__ == "__main__":
    main()
