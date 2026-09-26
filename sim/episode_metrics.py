"""
Chỉ số chất lượng của một episode gắp-đặt, đo theo từng bước sim (50 Hz).

Dùng chung cho tools/demo_quality.py (phát lại dataset có sẵn) và collect_demos.py (ghi lúc thu, xem
docs/PLAN_BALANCE_DATASET.md bước 1). Các chỉ số:
  - Khoảng hở nhỏ nhất tới lon còn lại, theo pha: trước khi kẹp / đang kẹp / sau khi nhả. Tính từ geom va chạm của
    cánh tay gắp (vai tới ngón) và lon đang cầm; thêm một số đo từ mọi geom của robot.
  - Lon còn lại có bị chạm (contact) không, bị xô xa nhất bao nhiêu.
  - Thân robot: góc ngả pelvis so với phương thẳng đứng, pitch, độ trượt xy và độ xoay yaw so với lúc đầu (max và lúc
    cuối); thân dưới (pelvis, chân, eo, torso) có chạm bàn không.
  - Bàn chân trượt trên sàn: độ xoay yaw và độ dời xy lớn nhất của hai bàn chân so với lúc đầu. Với chân balance, pelvis
    xoay ngược chiều eo qua khớp hông (yaw pelvis tới ~9° khi gắp chéo người) rồi tự trở lại, bàn chân gần như đứng yên;
    chỉ số bàn chân tách được việc đó khỏi trượt chân thật.
  - Lúc tay khép lần đầu: khoảng cách điểm gắp - tâm lon và vector lệch trong khung pelvis (x trước, y trái, z lên).
"""

from typing import Dict, Optional

import mujoco
import numpy as np

PHASES = ("pre", "hold", "post")  # trước khi kẹp, đang kẹp, sau khi nhả

# Ngưỡng loại demo lúc thu (collect_demos.py, docs/PLAN_BALANCE_DATASET.md bước 7):
#   - khoảng hở tới lon còn lại trước khi kẹp / đang kẹp: model lệch 2–5 cm nên demo sát hơn mức này dạy model xô lon;
#     planner đã ưu tiên khoảng hở lớn (APPROACH_MARGINS, CARRY_MARGINS) nhưng tay chạy lệch kế hoạch 2–3 cm;
#   - thân (chỉ với chân balance; chân fixed luôn ngả 5–14°): chạm bàn, ngả, bàn chân xoay trên sàn.
MIN_OTHER_CLEAR = 0.01
MAX_TILT_DEG = 5.0
# Bàn chân xoay: với balance, lúc không lọc lớn nhất 4.2° (61 episode), p90 3.0° (300 lần thử lon nằm); trượt chân thật ở
# chân fixed tới 10.8°. Ngưỡng 3° loại ~10% episode lon nằm bình thường (MEMO mục 5.2p)
MAX_FOOT_YAW_DEG = 5.0


def _subtree(model, root_name: str) -> np.ndarray:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    inside = np.zeros(model.nbody, dtype=bool)
    for b in range(model.nbody):
        a = b
        while a > 0 and a != root:
            a = model.body_parentid[a]
        inside[b] = a == root
    return inside


def _yaw(quat) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _mat_yaw(xmat) -> float:
    return float(np.arctan2(xmat[3], xmat[0]))  # xmat phẳng 9 phần tử, hàng chính: R[1,0], R[0,0]


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _pitch_deg(quat) -> float:
    w, x, y, z = quat
    return float(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))


class EpisodeMetrics:
    """Gọi step(ctrl) sau mỗi bước sim; result() trả dict chỉ số (đơn vị cm, độ)."""

    MAX_DIST = 0.30  # khoảng cách xa hơn mức này không cần tính chính xác

    def __init__(self, sim, can: str, other: str):
        m = self.model = sim.model
        self.sim, self.can, self.other = sim, can, other
        self.grasp = sim.grasps[can]
        side = "right" if self.grasp["grasp_point"][1] > 0 else "left"
        collide = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        robot = _subtree(m, "pelvis")
        arms = _subtree(m, "left_shoulder_pitch_link") | _subtree(m, "right_shoulder_pitch_link")
        body_of = m.geom_bodyid

        def geoms(mask):
            return np.flatnonzero(collide & mask[body_of])

        bodies = lambda name: np.arange(m.nbody) == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        self.arm_geoms = geoms(_subtree(m, f"{side}_shoulder_pitch_link"))
        self.robot_geoms = geoms(robot)
        self.can_geoms = geoms(bodies(f"{can}_can"))
        self.other_geoms = geoms(bodies(f"{other}_can"))
        self.table_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "work_table")
        self.lower_body = robot & ~arms  # pelvis, chân, eo, torso
        self.other_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{other}_can")
        self.can_bid = self.grasp["can_bid"]
        self.robot_body = robot
        self.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")]

        d = sim.data
        self.q0 = d.qpos[:7].copy()
        self.yaw0 = _yaw(self.q0[3:7])
        self.feet0 = [(d.xpos[b][:2].copy(), _mat_yaw(d.xmat[b])) for b in self.feet]
        self.other0 = d.xpos[self.other_bid].copy()
        self.can_adr = {"coke": sim.coke_adr, "pepsi": sim.pepsi_adr}[can]

        self.steps = 0
        self.ever_grasped = False
        self.min_arm = {p: np.inf for p in PHASES}
        self.min_robot = {p: np.inf for p in PHASES}
        self.touch_steps = {p: 0 for p in PHASES}
        self.table_steps = 0
        self.t_table: Optional[float] = None
        self.max_tilt = self.max_pitch = self.max_slide = self.max_yaw = 0.0
        self.max_foot_yaw = self.max_foot_slide = self.final_yaw = 0.0
        self.other_moved = 0.0
        self.close: Optional[Dict] = None
        self.t_grasp: Optional[float] = None

    def _min_dist(self, geoms_a, geoms_b) -> float:
        best = self.MAX_DIST
        m, d = self.model, self.sim.data
        for a in geoms_a:
            for b in geoms_b:
                best = min(best, mujoco.mj_geomDistance(m, d, int(a), int(b), best, None))
        return best

    def step(self, ctrl: np.ndarray, dt: float = 0.02):
        sim, d, m = self.sim, self.sim.data, self.model
        t = self.steps * dt
        self.steps += 1
        holding = sim.is_grasped(self.can)
        if holding and not self.ever_grasped:
            self.t_grasp = t
        self.ever_grasped |= holding
        phase = "hold" if holding else ("post" if self.ever_grasped else "pre")

        arm = self._min_dist(self.arm_geoms, self.other_geoms)
        if holding:
            arm = min(arm, self._min_dist(self.can_geoms, self.other_geoms))
        self.min_arm[phase] = min(self.min_arm[phase], arm)
        self.min_robot[phase] = min(self.min_robot[phase], arm, self._min_dist(self.robot_geoms, self.other_geoms))

        touched = table = False
        for con in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            for a, b in ((b1, b2), (b2, b1)):
                if a == self.other_bid and (self.robot_body[b] or (holding and b == self.can_bid)):
                    touched = True
                if a == self.table_bid and self.lower_body[b]:
                    table = True
        self.touch_steps[phase] += touched
        if table:
            self.table_steps += 1
            self.t_table = t if self.t_table is None else self.t_table

        q = d.qpos[:7]
        up = d.xmat[sim.pelvis_bid].reshape(3, 3)[:, 2]
        self.max_tilt = max(self.max_tilt, float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))))
        self.max_pitch = max(self.max_pitch, abs(_pitch_deg(q[3:7])))
        self.max_slide = max(self.max_slide, float(np.linalg.norm(q[:2] - self.q0[:2])))
        self.final_yaw = float(np.degrees(_wrap(_yaw(q[3:7]) - self.yaw0)))
        self.max_yaw = max(self.max_yaw, abs(self.final_yaw))
        for b, (xy0, yaw0) in zip(self.feet, self.feet0):
            self.max_foot_slide = max(self.max_foot_slide, float(np.linalg.norm(d.xpos[b][:2] - xy0)))
            self.max_foot_yaw = max(self.max_foot_yaw, abs(float(np.degrees(_wrap(_mat_yaw(d.xmat[b]) - yaw0)))))
        self.other_moved = max(self.other_moved, float(np.linalg.norm(d.xpos[self.other_bid] - self.other0)))

        closure = float(np.mean(np.abs(ctrl[self.grasp["hand_ctrl"]][1:])))
        if self.close is None and closure > sim.GRIP_CLOSED_RAD:
            w_rot = d.xmat[self.grasp["wrist_bid"]].reshape(3, 3)
            point = d.xpos[self.grasp["wrist_bid"]] + w_rot @ self.grasp["grasp_point"]
            offset = point - d.qpos[self.can_adr:self.can_adr + 3]
            p_rot = d.xmat[sim.pelvis_bid].reshape(3, 3)
            self.close = {"t": t, "dist_cm": float(np.linalg.norm(offset)) * 100,
                          "offset_pelvis_cm": (p_rot.T @ offset * 100).round(2).tolist()}

    def violation(self, body: bool = True) -> Optional[str]:
        """Lý do loại demo theo các ngưỡng ở đầu module (None nếu đạt). body=False: bỏ qua ngưỡng về thân."""
        for p, name in (("pre", "trước khi kẹp"), ("hold", "đang kẹp")):
            if self.touch_steps[p]:
                return f"chạm lon còn lại {name}"
            if self.min_arm[p] < MIN_OTHER_CLEAR:
                return f"sát lon còn lại {name}"
        if body:
            if self.table_steps:
                return "thân chạm bàn"
            if self.max_tilt > MAX_TILT_DEG:
                return f"thân ngả > {MAX_TILT_DEG:g}°"
            if self.max_foot_yaw > MAX_FOOT_YAW_DEG:
                return f"bàn chân xoay > {MAX_FOOT_YAW_DEG:g}°"
        return None

    def result(self) -> dict:
        d = self.sim.data
        up = d.xmat[self.sim.pelvis_bid].reshape(3, 3)[:, 2]
        cm = lambda v: None if not np.isfinite(v) else round(float(v) * 100, 2)
        return {
            "clear_arm_cm": {p: cm(self.min_arm[p]) for p in PHASES},
            "clear_robot_cm": {p: cm(self.min_robot[p]) for p in PHASES},
            "other_touch_steps": dict(self.touch_steps),
            "other_moved_cm": cm(self.other_moved),
            "body_table_steps": self.table_steps,
            "t_body_table": self.t_table,
            "max_tilt_deg": round(self.max_tilt, 2),
            "max_pitch_deg": round(self.max_pitch, 2),
            "final_tilt_deg": round(float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))), 2),
            "max_slide_cm": cm(self.max_slide),
            "max_yaw_deg": round(self.max_yaw, 2),
            "final_yaw_deg": round(self.final_yaw, 2),
            "max_foot_yaw_deg": round(self.max_foot_yaw, 2),
            "max_foot_slide_cm": cm(self.max_foot_slide),
            "t_grasp": self.t_grasp,
            "close": self.close,
        }
