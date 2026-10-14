"""
Đánh giá closed-loop model GR00T (PolicyServer ZMQ) trong sim, không mở cửa sổ 3D.

Mỗi episode đặt lon/khay/tư thế tay theo đúng ngẫu nhiên hoá của collect_demos.py (task + seed + cfg), chạy model
trong --seconds giây rồi chấm bằng cùng tiêu chí thành công như lúc thu demo. Ngoài thành công còn đo:
khoảng cách điểm gắp - tâm lon lúc tay khép (ngưỡng kẹp của sim là GRASP_RADIUS), có kẹp được lon không,
lon lên cao nhất bao nhiêu, thân robot ngả tối đa bao nhiêu độ, và số lần server không phản hồi.

Bảng kết quả tách theo nhóm cảnh của collect_demos.GROUPS (cùng/đổi phía × đứng/nằm). Episode "đã xong" (lon nằm sẵn
trong khay, đã nhả và lon còn lại không bị xô; chúng có dòng riêng và không tính vào "tất cả".

Lon còn lại "bị xô" khi lệch >= 3 cm tính từ lúc robot hoặc lon nhiệm vụ chạm nó lần đầu (other_moved_after_touch_cm):
lon nằm tự lăn chậm trên bàn trong sim (tới 7 cm sau 20 s), nên chỉ đo độ lệch từ đầu episode sẽ tính nhầm. Kết quả chạy trước khi có other_touch_steps
chỉ chấm được theo độ lệch; chạy lại các ca có thể bị nhầm bằng --resume --rerun-rolling --dataset DIR.

Model có phân biệt Coca/Pepsi bằng màu không: cột "chọn đúng lon" (lần khép tay đầu tiên, bàn tay gần lon nhiệm vụ hơn
lon còn lại; "lon nhiệm vụ" là lon Coca/Pepsi thật, không theo màu đang hiển thị). Chạy lại cùng tập cảnh với
--swap-can-colors (đổi màu hai lon trong sim; vị trí, vật lý, khay giữ nguyên) rồi so với lượt màu gốc ở ca đổi phía:
model nhìn màu thì tỉ lệ chọn đúng lon tụt về gần 0 (đi về lon mang màu của lon nhiệm vụ); model chỉ dựa vào vị trí thì
gần như không đổi. Lon trong sim không có logo, ngoài vị trí chỉ có màu để phân biệt.

Nguồn episode:
  - --dataset DIR: lấy task/seed/cfg từ meta/collection.jsonl (ví dụ tập test), đánh giá đúng các cảnh đó.
  - hoặc --seed N --episodes-per-task K với cfg từ các tham số --can-x/--can-y/... (mặc định như collect_demos.py).

Ví dụ:
    .venv/bin/python tools/eval_closed_loop.py --host <host> --port <port> --dataset datasets/g1_can_sort_ik_test \\
        --out eval/ik_test --save-video fail
    .venv/bin/python tools/eval_closed_loop.py --host <host> --port <port> --seed 200000 --episodes-per-task 10 \\
        --out eval/quick
    # Thử model có nhìn màu không: cùng tập cảnh, đổi màu hai lon, so với lượt chạy màu gốc
    .venv/bin/python tools/eval_closed_loop.py --host <host> --port <port> --dataset datasets/g1_can_sort_swap_hold_test \\
        --out eval/swap_test_colors --swap-can-colors --seconds 20 --stop-on-success
    # Tính lại bảng từ results.jsonl đã có, ghi ra summary_regrouped.txt (giữ nguyên summary.txt; kết quả cũ chưa ghi
    # "done" thì lấy nhóm cảnh từ --dataset)
    .venv/bin/python tools/eval_closed_loop.py --summarize-only --out eval/ik_test --dataset datasets/g1_can_sort_ik_test
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


def grasp_offset_pelvis_cm(sim, grasp, can_adr) -> list:
    """Vector từ tâm lon tới điểm gắp trong khung pelvis (cm; x trước, y trái, z lên), để xem model lệch theo hướng nào."""
    d = sim.data
    w_rot = d.xmat[grasp["wrist_bid"]].reshape(3, 3)
    point = d.xpos[grasp["wrist_bid"]] + w_rot @ grasp["grasp_point"]
    p_rot = d.xmat[sim.pelvis_bid].reshape(3, 3)
    return (p_rot.T @ (point - d.qpos[can_adr:can_adr + 3]) * 100).round(2).tolist()


def pelvis_pitch_deg(qpos) -> float:
    w, x, y, z = qpos[3:7]
    return float(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))


OTHER_MOVED_M = 0.03  # ngưỡng lon còn lại bị xô, như collect_demos.is_success


class OtherCanContact:
    """Đếm số bước 50 Hz robot hoặc lon nhiệm vụ chạm lon còn lại, và lon còn lại lệch bao xa kể từ lần chạm đầu.

    Lon nằm tự lăn chậm trên bàn trong sim (khoảng 3 cm sau 20 s, max 7 cm, không có gì chạm), nên chỉ đo độ lệch thì
    tính nhầm là robot xô. Nó còn có thể lăn tới chạm bàn tay đang đứng yên (test coke 100001 với robot đứng yên: lăn
    5.5 cm rồi chạm ngón tay phải ở giây 18.9). Vì vậy lon còn lại chỉ bị coi là bị xô khi lệch >= 3 cm tính từ vị trí
    lúc robot/lon nhiệm vụ chạm nó lần đầu; phần tự lăn trước đó không tính.
    """

    def __init__(self, sim, ctx):
        import mujoco
        from sim.episode_metrics import _subtree

        m = sim.model
        self.sim, self.ctx = sim, ctx
        self.robot = _subtree(m, "pelvis")
        self.other_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{ctx['other']}_can")
        self.can_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{ctx['can']}_can")
        self.adr = ctx["can_adr"][ctx["other"]]
        self.steps = 0
        self.pos_at_touch = None  # vị trí lon còn lại lúc bị chạm lần đầu

    def step(self):
        m, d = self.sim.model, self.sim.data
        for con in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if self.other_bid in (b1, b2):
                b = b2 if b1 == self.other_bid else b1
                if self.robot[b] or b == self.can_bid:
                    self.steps += 1
                    if self.pos_at_touch is None:
                        self.pos_at_touch = d.qpos[self.adr:self.adr + 3].copy()
                    return

    def moved_m(self) -> float:
        return float(np.linalg.norm(self.sim.data.qpos[self.adr:self.adr + 3] - self.ctx["other_start"]))

    def moved_after_touch_m(self) -> float:
        if self.pos_at_touch is None:
            return 0.0
        return float(np.linalg.norm(self.sim.data.qpos[self.adr:self.adr + 3] - self.pos_at_touch))

    def pushed(self) -> bool:
        return self.moved_after_touch_m() >= OTHER_MOVED_M


def other_pushed(r: dict) -> bool:
    if "other_moved_after_touch_cm" in r:
        return r["other_moved_after_touch_cm"] >= OTHER_MOVED_M * 100
    return r["other_moved_cm"] >= OTHER_MOVED_M * 100  # kết quả cũ chưa đếm va chạm: chỉ còn độ lệch


def failure_reason(r: dict) -> str:
    if r["ok"]:
        return "thành công"
    if r["server_errors"] and r["server_errors"] >= r["policy_calls"]:
        return "server không phản hồi"
    if r.get("done"):
        if r.get("can_fell") or not r.get("in_tray"):
            return "đã xong sẵn nhưng lấy lon ra khỏi khay"
        if other_pushed(r):
            return "đã xong sẵn nhưng xô lon còn lại"
        return "đã xong sẵn nhưng vẫn cầm lon"
    if r.get("can_fell"):
        return "lon rơi khỏi bàn"
    if r.get("in_tray") and other_pushed(r):
        return "đặt đúng khay nhưng xô lon còn lại"
    if other_pushed(r) and not r["grasped"]:
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

    # Chế độ chân: theo --lower-body nếu có, không thì theo cfg của episode (dataset cũ: fixed)
    lower_body = args["lower_body"] or cfg.get("lower_body", "fixed")
    pipe_kwargs = {"policy_mode": "remote", "server_host": args["host"], "server_port": args["port"],
                   "vla_action_hz": args["hz"], "server_timeout_ms": args["timeout_ms"], "lower_body": lower_body,
                   "policy_replan_interval": args["replan_steps"]}
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

    if args["swap_can_colors"]:
        sim.set_can_colors_swapped(True)

    can, can_adr = ctx["can"], ctx["can_adr"][ctx["can"]]
    other_adr = ctx["can_adr"][ctx["other"]]
    grasp = sim.grasps[can]
    rand = cd.sample_randomization(task, seed, cfg)
    lying, swapped, done = rand["cans"][can]["lying"], rand["swapped"], bool(rand["done"])
    can_start = sim.data.qpos[can_adr:can_adr + 3].copy()
    d_close = d_other_close = offset_close = t_grasp = t_success = None
    attempts, was_closed = 0, False  # số lần khép tay (cạnh lên), để thấy model có thử lại không
    can_max_z, max_pitch = -np.inf, 0.0
    contact = OtherCanContact(sim, ctx)
    # Thành công: lon trong khay, đã nhả, và lon còn lại không bị robot xô (lệch do tự lăn không tính)
    success = lambda: cd.is_success(pipe, ctx, check_other=False) and not contact.pushed()
    frames = []
    for step in range(int(args["seconds"] * CONTROL_HZ)):
        dist = grasp_distance(sim, grasp, can_adr)
        dist_other = grasp_distance(sim, grasp, other_adr)  # cùng bàn tay, tới lon còn lại
        offset = grasp_offset_pelvis_cm(sim, grasp, can_adr)  # cùng thời điểm với dist
        was_grasped = sim.is_grasped(can)
        with contextlib.redirect_stdout(io.StringIO()):
            res = pipe.step_50hz()
        contact.step()
        if counts["policy_calls"] == 1 and policy.server_action_space not in (None, "real_g1"):
            pipe.close()
            raise RuntimeError(f"Server không chạy model GR00T thật (action_space={policy.server_action_space})")
        if (counts["policy_calls"] == 1 and args["expect_embodiment"] and policy.server_model
                and policy.server_model.get("embodiment_tag") != args["expect_embodiment"]):
            pipe.close()
            raise RuntimeError(
                f"Server đang chạy {policy.server_model.get('model_path')} (embodiment "
                f"{policy.server_model.get('embodiment_tag')}), không phải checkpoint fine-tune "
                f"(embodiment {args['expect_embodiment']}). Sửa MODEL_PATH/EMBODIMENT_TAG ở Bước 5 của notebook server, "
                f"hoặc thêm --expect-embodiment '' nếu cố ý đánh giá checkpoint base.")
        if counts["policy_calls"] >= 3 and counts["server_errors"] == counts["policy_calls"]:
            pipe.close()
            raise RuntimeError(f"PolicyServer tcp://{args['host']}:{args['port']} không phản hồi 3 lần liền, dừng đánh giá")
        closure = float(np.mean(np.abs(res["motor_targets"][grasp["hand_ctrl"]][1:])))
        if d_close is None and closure > sim.GRIP_CLOSED_RAD:
            d_close, d_other_close = dist, dist_other
            offset_close = offset
        attempts += closure > sim.GRIP_CLOSED_RAD and not was_closed
        was_closed = closure > sim.GRIP_CLOSED_RAD
        if t_grasp is None and not was_grasped and sim.is_grasped(can):
            t_grasp = step / CONTROL_HZ
        can_max_z = max(can_max_z, float(sim.data.qpos[can_adr + 2]))
        max_pitch = max(max_pitch, abs(pelvis_pitch_deg(sim.data.qpos)))
        if t_success is None and step % 25 == 0 and success():
            t_success = step / CONTROL_HZ
            if args["stop_on_success"] and not done:  # cảnh "đã xong" thành công ngay từ đầu: chạy đủ để xem có phá không
                break
        if args["save_video"] != "none" and step % cd.CONTROL_STEPS_PER_FRAME == 0:
            img = Image.fromarray(sim.render_camera("front_camera")).resize(cd.IMAGE_SIZE, Image.BOX)
            frames.append(np.asarray(img))

    d = sim.data
    r = {
        "task": task, "seed": seed, "lying": bool(lying), "swapped": bool(swapped), "done": done,
        "group": cd.scene_group(task, seed, cfg),
        "ok": success(), "ok_any": t_success is not None, "t_success": t_success,
        "in_tray": cd.is_success(pipe, ctx, check_other=False),
        "grasped": t_grasp is not None, "t_grasp": t_grasp,
        "dist_at_close_cm": None if d_close is None else round(d_close * 100, 2),
        "offset_at_close_cm": offset_close,
        "other_dist_at_close_cm": None if d_other_close is None else round(d_other_close * 100, 2),
        # Lon model chọn: lần khép tay đầu tiên, bàn tay của nhiệm vụ gần lon nào hơn (None: chưa khép tay lần nào)
        "chose_target": None if d_close is None else bool(d_close < d_other_close),
        "can_colors_swapped": args["swap_can_colors"],
        "can_start": can_start.round(3).tolist(), "can_final": d.qpos[can_adr:can_adr + 3].round(3).tolist(),
        "can_max_lift_cm": round(float(can_max_z - can_start[2]) * 100, 1),
        "can_fell": bool(d.qpos[can_adr + 2] < can_start[2] - 0.2),
        "attempts": int(attempts),
        "other_moved_cm": round(contact.moved_m() * 100, 1),
        "other_touch_steps": contact.steps,
        "other_moved_after_touch_cm": round(contact.moved_after_touch_m() * 100, 1),
        "other_lying": bool(rand["cans"][ctx["other"]]["lying"]),
        "max_pitch_deg": round(max_pitch, 1),
        "policy_calls": counts["policy_calls"], "server_errors": counts["server_errors"],
        "server_model": policy.server_model, "lower_body": lower_body, "replan_steps": args["replan_steps"],
        "stop_on_success": args["stop_on_success"],
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


def group_of(r: dict) -> str:
    """Nhóm cảnh của một kết quả (kết quả cũ chưa ghi "group" thì suy ra từ lying/swapped/done)."""
    if r.get("group"):
        return r["group"]
    if r.get("done"):
        return "đã xong"
    return f"{'đổi phía' if r['swapped'] else 'cùng phía'} {'nằm' if r['lying'] else 'đứng'}"


def condition_of(r: dict) -> str:
    """Màu lon trong sim của một kết quả (kết quả cũ: màu gốc)."""
    return "màu lon đổi" if r.get("can_colors_swapped") else "màu lon gốc"


def summarize(results: list) -> str:
    # "từng đặt xong": có lúc lon đã nằm trong khay (ok_any), kể cả khi sau đó model gắp lại / xô ra.
    # Episode "đã xong sẵn" chỉ nằm ở dòng riêng, không tính vào "tất cả" và các nhóm gắp.
    pick = [r for r in results if group_of(r) != "đã xong"]
    done = [r for r in results if group_of(r) == "đã xong"]
    has_swap = any(r["swapped"] for r in pick)
    side = lambda r: "đổi phía" if r["swapped"] else "cùng phía"
    pose = lambda r: "nằm" if r["lying"] else "đứng"
    # Mỗi khối là một danh sách (tên dòng, hàm chọn episode); khối cách nhau một dòng trống
    blocks = [[("tất cả", lambda r: True)]]
    if has_swap:
        blocks.append([(g, lambda r, g=g: group_of(r) == g) for g in cd.GROUPS])
        blocks.append([(k, lambda r, k=k: side(r) == k) for k in ("cùng phía", "đổi phía")])
    blocks.append([(k, lambda r, k=k: pose(r) == k) for k in ("đứng", "nằm")])
    blocks.append([(t, lambda r, t=t: r["task"] == t) for t in cd.TASKS])
    sub = cd.GROUPS if has_swap else ("đứng", "nằm")
    key_of = group_of if has_swap else pose
    blocks.append([(f"{t} {k}", lambda r, t=t, k=k: r["task"] == t and key_of(r) == k) for t in cd.TASKS for k in sub])

    def row(key, g):
        ok = sum(r["ok"] for r in g)
        ok_any = sum(r["ok_any"] for r in g)
        grasped = sum(r["grasped"] for r in g)
        dists = [r["dist_at_close_cm"] for r in g if r["dist_at_close_cm"] is not None]
        dist = f"{np.mean(dists):.1f} (max {np.max(dists):.1f})" if dists else "-"
        chose = [r["chose_target"] for r in g if r.get("chose_target") is not None]
        choice = f"{sum(chose)}/{len(chose)} {100 * sum(chose) / len(chose):3.0f}%" if chose else "-"
        return (f"{key:24s} {ok:4d}/{len(g):<3d} {100 * ok / len(g):3.0f}% {ok_any:6d}/{len(g):<3d} "
                f"{100 * ok_any / len(g):3.0f}% {grasped:4d}/{len(g):<3d} {dist:>18s} {choice:>13s}")

    conditions = sorted({condition_of(r) for r in results})
    lines = [f"Điều kiện: {', '.join(conditions)}"] if conditions != [condition_of({})] else []
    lines += [f"\n{'nhóm':24s} {'thành công':>12s} {'từng đặt xong':>14s} {'kẹp được':>10s} {'k/c lúc khép (cm)':>18s} "
              f"{'chọn đúng lon':>13s}"]
    for i, block in enumerate(blocks):
        rows = [row(key, g) for key, sel in block if (g := [r for r in pick if sel(r)])]
        lines += ([""] if i and rows else []) + rows

    if done:
        ok = sum(r["ok"] for r in done)
        # Kết quả trước khi ghi "done": cảnh đã xong cũng dừng ngay ở bước đầu khi --stop-on-success
        legacy = any(r.get("legacy_done") and r.get("stop_on_success") for r in done)
        note = " (kết quả cũ: dừng ngay ở bước đầu nên luôn giữ nguyên)" if legacy else ""
        lines.append(f"\nĐã xong sẵn (lon nằm sẵn trong khay, không tính ở trên): {ok}/{len(done)} giữ nguyên{note}")
        lines += [f"  {r['task']} seed {r['seed']}: {r['reason']}" for r in done if not r["ok"]]

    t_succ = [r["t_success"] for r in pick if r["t_success"] is not None]
    tries = [r["attempts"] for r in pick if "attempts" in r]
    if tries:
        lines.append(f"\nSố lần khép tay TB {np.mean(tries):.1f} (tối đa {max(tries)}); "
                     f"lon rơi khỏi bàn {sum(bool(r.get('can_fell')) for r in pick)}/{len(pick)}")
    if t_succ:
        lines.append(f"Đặt xong lần đầu lúc: trung vị {np.median(t_succ):.1f} s, muộn nhất {max(t_succ):.1f} s")
    if any(r.get("chose_target") is not None for r in pick):
        lines.append("Chọn đúng lon: lần khép tay đầu tiên, bàn tay gần lon nhiệm vụ hơn lon còn lại (chỉ tính episode có "
                     "khép tay). Ca cùng phía chọn đúng cả khi không nhìn màu; ca đổi phía mới cho biết model có nhìn màu.")
    lines.append("\nLý do thất bại (không tính đã xong sẵn):")
    for reason, n in collections.Counter(r["reason"] for r in pick if not r["ok"]).most_common():
        lines.append(f"  {n:4d}  {reason}")
    errors = sum(r["server_errors"] for r in results)
    calls = sum(r["policy_calls"] for r in results)
    lines.append(f"\nServer: {calls} lần gọi, {errors} lần không phản hồi")
    models = {json.dumps(r.get("server_model"), ensure_ascii=False) for r in results}
    lines.append("Model: " + " | ".join(sorted(models)))
    return "\n".join(lines)


def fill_legacy(results: list, dataset) -> None:
    """Kết quả chạy trước khi eval ghi "done"/"group": lấy nhóm cảnh từ cfg trong collection.jsonl của --dataset."""
    if all("group" in r for r in results):
        return
    meta = {}
    if dataset:
        for line in (Path(dataset) / "meta" / "collection.jsonl").read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                meta[(e["task"], e["seed"])] = e["cfg"]
    missing = 0
    for r in results:
        if "group" in r:
            continue
        cfg = meta.get((r["task"], r["seed"]))
        if cfg is None:
            missing += 1
            continue
        r["group"] = cd.scene_group(r["task"], r["seed"], cfg)
        r["done"] = r["group"] == "đã xong"
        r["legacy_done"] = r["done"]
        r["reason"] = failure_reason(r)
    if missing:
        print(f"⚠️ {missing} kết quả cũ không có nhóm cảnh (thiếu --dataset hoặc không khớp task/seed): coi như "
              f"không phải 'đã xong', nhóm suy ra từ lying/swapped.")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", help="Host của PolicyServer")
    p.add_argument("--port", type=int, help="Port của PolicyServer")
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
    p.add_argument("--seconds", type=float, default=10.0,
                   help="Thời gian chạy mỗi episode (demo dài 7.5 s). Tăng (ví dụ 30) để model có thời gian thử lại")
    p.add_argument("--hz", type=float, default=cd.ACTION_HZ, help="Tần số action chunk của model (dataset: 25)")
    p.add_argument("--timeout-ms", type=int, default=10000)
    p.add_argument("--stop-on-success", action="store_true",
                   help="Kết thúc episode ngay khi đặt xong (như --stop-on-success của launch_vla_simulation.py). Dùng khi "
                        "tăng --seconds để đo thử lại: không tính lỗi gắp lại lon sau khi đã xong, và chạy nhanh hơn")
    p.add_argument("--replan-steps", type=int, default=15,
                   help="Số bước 50 Hz chạy theo một action chunk trước khi gọi model lần sau (15 = 0.3 s). Tăng (ví dụ "
                        "30) để model ít đổi ý giữa các chunk, như lúc tay lưỡng lự giữa hai kiểu gắp lon nằm")
    p.add_argument("--lower-body", choices=["fixed", "balance"], default=None,
                   help="fixed: chân khoá cứng; balance: policy thăng bằng của NVIDIA điều khiển chân + eo. Mặc định theo "
                        "cfg của episode (lower_body lúc thu demo; dataset trước 25/09: fixed)")
    p.add_argument("--expect-embodiment", default="new_embodiment",
                   help="Dừng nếu server báo embodiment khác (mặc định: checkpoint fine-tune). '' để bỏ kiểm tra")
    p.add_argument("--workers", type=int, default=2,
                   help="Số episode chạy song song (server xử lý tuần tự, 2 giúp sim chạy trong lúc chờ GPU)")
    p.add_argument("--swap-can-colors", action="store_true",
                   help="Đổi màu hai lon trong sim (lon Coca xanh, lon Pepsi đỏ; vị trí và khay giữ nguyên). Model nhìn "
                        "màu thì ở mọi cảnh sẽ đi về lon kia. Dùng --out riêng")
    p.add_argument("--save-video", choices=["none", "fail", "all"], default="fail",
                   help="Ghi video camera front_camera cho episode thất bại / tất cả")
    p.add_argument("--resume", action="store_true",
                   help="Giữ kết quả đã có trong <out>/results.jsonl, chỉ chạy các episode (task, seed) còn thiếu")
    p.add_argument("--rerun-rolling", action="store_true",
                   help="Kèm --resume --dataset: chạy lại các kết quả cũ (trước khi đếm va chạm) có thể bị tính nhầm "
                        "'xô lon còn lại' do lon nằm tự lăn: episode 'đã xong' và ca thất bại có lon còn lại nằm lệch >= 3 cm. "
                        "Bản cũ của results.jsonl lưu ở results.before_rerun.jsonl")
    p.add_argument("--summarize-only", action="store_true",
                   help="Không chạy sim: tính lại bảng từ <out>/results.jsonl, ghi ra <out>/summary_regrouped.txt (không "
                        "đụng summary.txt của lượt chạy; kết quả cũ thì kèm --dataset "
                        "để biết episode nào 'đã xong sẵn')")
    args = p.parse_args()

    out = Path(args.out)
    if args.summarize_only:
        results = [json.loads(line) for line in open(out / "results.jsonl") if line.strip()]
        fill_legacy(results, args.dataset)
        summary = summarize(results)
        (out / "summary_regrouped.txt").write_text(summary + "\n")
        print(summary)
        print(f"\nĐã ghi {out / 'summary_regrouped.txt'} (summary.txt giữ nguyên)")
        return
    if args.host is None or args.port is None:
        p.error("cần --host và --port (hoặc --summarize-only)")
    if args.rerun_rolling and not (args.resume and args.dataset):
        p.error("--rerun-rolling cần --resume và --dataset")
    out.mkdir(parents=True, exist_ok=True)
    shared = {"host": args.host, "port": args.port, "hz": args.hz, "timeout_ms": args.timeout_ms,
              "expect_embodiment": args.expect_embodiment, "lower_body": args.lower_body, "replan_steps": args.replan_steps,
              "stop_on_success": args.stop_on_success, "swap_can_colors": args.swap_can_colors,
              "seconds": args.seconds, "save_video": args.save_video, "out": str(out)}
    jobs = [(t, s, cfg, shared) for t, s, cfg in load_jobs(args)]
    results = []
    if args.resume and (out / "results.jsonl").exists():
        results = [json.loads(line) for line in open(out / "results.jsonl") if line.strip()]
        fill_legacy(results, args.dataset)
        if args.rerun_rolling:
            cfgs = {(t, s): cfg for t, s, cfg in load_jobs(args)}

            def rolling_suspect(r):
                if "other_touch_steps" in r or (r["task"], r["seed"]) not in cfgs:
                    return False
                if r.get("done"):
                    return True
                other = cd._task_can(r["task"])[1]
                other_lying = cd.sample_randomization(r["task"], r["seed"], cfgs[(r["task"], r["seed"])])["cans"][other]["lying"]
                return not r["ok"] and other_lying and r["other_moved_cm"] >= 3.0

            rerun = [r for r in results if rolling_suspect(r)]
            if rerun:
                (out / "results.before_rerun.jsonl").write_text((out / "results.jsonl").read_text())
                results = [r for r in results if not rolling_suspect(r)]
                (out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
            print(f"--rerun-rolling: chạy lại {len(rerun)} episode: "
                  + ", ".join(f"{r['task']} {r['seed']} ({r['reason']})" for r in rerun))
        now = condition_of({"can_colors_swapped": args.swap_can_colors})
        old = {condition_of(r) for r in results} - {now}
        if old:
            p.error(f"{out}/results.jsonl chạy với điều kiện khác ({', '.join(sorted(old))}), lượt này: {now}; "
                    f"dùng --out khác")
        done = {(r["task"], r["seed"]) for r in results}
        jobs =[j for j in jobs if (j[0], j[1]) not in done]
        print(f"--resume: đã có {len(results)} episode, còn {len(jobs)}")
    n_total = len(results) + len(jobs)
    print(f"Đánh giá {len(jobs)} episode trên tcp://{args.host}:{args.port} -> {out}")

    n_before, t0 = len(results), time.time()
    # Mỗi tiến trình một episode (tạo pipeline lặp lại trong cùng tiến trình bị rò rỉ bộ nhớ).
    # Ghi summary cả khi bị dừng giữa chừng (Ctrl+C, server không phản hồi); chạy tiếp bằng --resume.
    try:
        with Pool(args.workers, maxtasksperchild=1) as pool, open(out / "results.jsonl", "a" if args.resume else "w") as f:
            for r in pool.imap_unordered(run_episode, jobs):
                results.append(r)
                if len(results) == n_before + 1:
                    m = r["server_model"]
                    print(f"Server: {m['model_path']} (embodiment {m['embodiment_tag']})" if m else
                          "⚠️ Server bản cũ không báo đang chạy checkpoint nào; không kiểm tra được --expect-embodiment.")
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{time.time() - t0:5.0f}s] {len(results):3d}/{n_total} {r['task']:5s} seed {r['seed']:6d} "
                      f"{r['group']:15s} "
                      f"{'OK  ' if r['ok'] else 'FAIL'} | {r['reason']} | "
                      f"k/c khép {r['dist_at_close_cm']} cm | lỗi server {r['server_errors']}", flush=True)
    finally:
        if results:
            summary = summarize(results)
            if len(results) < n_total:
                summary = (f"⚠️ CHƯA ĐỦ: {len(results)}/{n_total} episode (bị dừng giữa chừng; "
                           f"chạy lại cùng lệnh với --resume để làm tiếp)\n") + summary
            (out / "summary.txt").write_text(summary + "\n")
            print(summary)


if __name__ == "__main__":
    main()
