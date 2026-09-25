"""
Bảng kiểu gắp lon nằm theo hướng lon (docs/PLAN_BALANCE_DATASET.md bước 3).

Kiểu gắp: trục cuộn ngón (z cổ tay) theo chiều này hay chiều kia của trục lon. Có "plan" (planner sau 25/09): kiểu "+" khi
trục cuộn ngón hướng về nửa y > 0 so với trục lon, tức cos(finger_axis − yaw gập) > 0. Dataset cũ: xấp xỉ bằng wrist pitch
ở khung khép tay > MODE_SPLIT (hai kiểu cách nhau ~2 rad, MEMO mục 5.0e; không tách hẳn nên chỉ là ước lượng). Mỗi ô là (số episode kiểu "+" / tổng) theo tay, cùng/đổi phía và yaw của trục lon
gập về 0–180°; ô đạt khi một kiểu chiếm ≥ 90%.

Nguồn: <out>/trial.jsonl của tools/planner_trial.py hoặc meta/collection.jsonl của dataset thu sau 25/09 (có "plan"); dataset
cũ không có "plan" thì đọc wrist pitch ở khung khép tay từ parquet.

Ví dụ:
    .venv/bin/python tools/grasp_modes.py eval/planner_trial_modes_rule/trial.jsonl
    .venv/bin/python tools/grasp_modes.py datasets/g1_can_sort_swap_hold
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import collect_demos as cd  # noqa: E402

MODE_SPLIT = 0.25  # wrist pitch (rad) ở giữa hai cụm (+1.0..1.4 và −0.7..−0.2)
BINS = np.arange(0, 181, 20)
CLOSE_FRAME = 55   # khung khép tay của dataset trước 25/09 (độ dài các đoạn cố định)


def load(path: Path) -> list:
    """Danh sách (task, lying, swapped, yaw gập °, wrist pitch hoặc None, mode_fallback)."""
    if path.is_dir():
        rows = [json.loads(line) for line in (path / "meta" / "collection.jsonl").read_text().splitlines() if line.strip()]
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r.get("ok")]
    out = []
    for r in rows:
        rand = cd.sample_randomization(r["task"], r["seed"], r["cfg"])
        can = rand["cans"][cd._task_can(r["task"])[0]]
        if not can["lying"] or rand["done"]:  # episode "đã xong" không có thao tác gắp
            continue
        plan = r.get("plan") or {}
        yaw = float(np.degrees(can["yaw"]) % 180)
        if plan.get("finger_axis_deg") is not None:
            mode = float(np.cos(np.radians(plan["finger_axis_deg"] - yaw)))  # > 0: kiểu "+"
        else:
            pitch = plan.get("wrist_pitch")
            if pitch is None and path.is_dir():
                pitch = wrist_pitch_from_parquet(path, r)
            mode = None if pitch is None else pitch - MODE_SPLIT
        out.append((r["task"], rand["swapped"], yaw, mode, plan.get("mode_fallback")))
    return out


def wrist_pitch_from_parquet(root: Path, rec: dict) -> float:
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    modality = json.loads((root / "meta" / "modality.json").read_text())
    arm = modality["state"]["right_arm" if rec["task"] == "coke" else "left_arm"]
    table = pq.read_table(root / info["data_path"].format(episode_chunk=0, episode_index=rec["episode_index"]))
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    return float(state[min(CLOSE_FRAME, len(state) - 1), arm["start"] + 5])  # khớp tay thứ 6: wrist_pitch


def table(rows: list) -> str:
    lines = [f"{'nhóm':20s} " + " ".join(f"{f'{a}–{b}':>7s}" for a, b in zip(BINS[:-1], BINS[1:])) +
             "   ô đạt   số lần đổi kiểu theo yaw"]
    for task in ("coke", "pepsi"):
        for sw in (False, True):
            g = [r for r in rows if r[0] == task and r[1] == sw and r[3] is not None]
            if not g:
                continue
            cells, good, total = [], 0, 0
            for a, b in zip(BINS[:-1], BINS[1:]):
                sub = [r for r in g if a <= r[2] < b]
                plus = sum(r[3] > 0 for r in sub)
                if sub:
                    total += 1
                    good += max(plus, len(sub) - plus) >= 0.9 * len(sub)
                cells.append(f"{plus}/{len(sub)}" if sub else "-")
            name = f"{task} {'đổi phía' if sw else 'cùng phía'}"
            # Số lần đổi kiểu khi xếp theo yaw (vòng 0–180°): ranh giới rõ thì 2 (kiểu đổi một lần và đổi lại ở 180 ≡ 0)
            seq = [r[3] > 0 for r in sorted(g, key=lambda r: r[2])]
            switches = sum(a != b for a, b in zip(seq, seq[1:] + seq[:1]))
            lines.append(f"{name:20s} " + " ".join(f"{c:>7s}" for c in cells) + f"   {good}/{total}   đổi kiểu {switches}")
    fb = [r for r in rows if r[4]]
    lines.append(f"\nÔ: số episode kiểu '+' / tổng. Episode lon nằm: {len(rows)}; phải dùng kiểu còn lại "
                 f"(mode_fallback): {len(fb)}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Bảng kiểu gắp lon nằm theo yaw.")
    p.add_argument("source", nargs="+", help="trial.jsonl của planner_trial.py hoặc thư mục dataset")
    args = p.parse_args()
    for src in args.source:
        rows = load(Path(src))
        print(f"\n=== {src}")
        print(table(rows))


if __name__ == "__main__":
    main()
