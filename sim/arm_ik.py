"""
Module sim.arm_ik: Động học ngược (IK) cho một cánh tay Unitree G1 (7 khớp) cộng khớp waist_pitch trong MuJoCo.

Giải bằng damped least squares có trọng số, chạy trên một MjData riêng (không đụng tới trạng thái mô phỏng):
pelvis, chân và các khớp khác giữ nguyên như qpos tham chiếu, chỉ các khớp được chọn thay đổi và bị kẹp trong giới
hạn khớp.

Mục tiêu gồm vị trí một điểm trên bàn tay (mặc định là điểm gắp, trong khung wrist_yaw_link) và các ràng buộc trục:
trục `local` của khung cổ tay phải trùng hướng `world`. Ràng buộc trục mềm (trọng số nhỏ) chỉ gợi ý hướng mà không
bắt buộc, ví dụ lòng bàn tay úp xuống khi gắp lon nằm. Một số hạng kéo nhẹ về tư thế tham chiếu giữ nghiệm liền
mạch giữa các lần giải và tránh tư thế lạ ở phần bậc tự do dư.

Bộ giải là cục bộ nên có thể kẹt ở cực tiểu cục bộ; khi cần nghiệm tốt thì gọi solve với nhiều q_init khác nhau.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import mujoco
import numpy as np

ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


@dataclass
class IKTarget:
    """Pose mong muốn của bàn tay: vị trí điểm trên bàn tay và các ràng buộc trục (trục local, hướng world, trọng số)."""

    pos: np.ndarray
    axes: List[Tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)


@dataclass
class IKResult:
    q: np.ndarray        # góc các khớp được chọn (thứ tự như ArmIK.joint_names)
    pos_err: float       # sai số vị trí (m)
    axis_err: float      # sai lệch góc lớn nhất của các ràng buộc trục cứng (rad)
    posture_cost: float  # độ lệch so với tư thế tham chiếu (rad)

    def ok(self, pos_tol: float = 0.003, axis_tol: float = 0.05) -> bool:
        return self.pos_err < pos_tol and self.axis_err < axis_tol


def axis_target(pos: Sequence[float], axes: Sequence[Tuple[Sequence[float], Sequence[float], float]]) -> IKTarget:
    return IKTarget(
        pos=np.asarray(pos, np.float64),
        axes=[(np.asarray(a, np.float64), np.asarray(w, np.float64), float(wt)) for a, w, wt in axes],
    )


def rotation_target(pos: Sequence[float], rot: np.ndarray, weight: float = 1.0) -> IKTarget:
    """Mục tiêu pose đầy đủ: hai trục x, z của khung cổ tay trùng hai cột tương ứng của ma trận quay rot."""
    return axis_target(pos, [((1, 0, 0), rot[:, 0], weight), ((0, 0, 1), rot[:, 2], weight)])


class ArmIK:
    """IK cho tay `side` ('left' / 'right') cộng waist_pitch."""

    # Ràng buộc trục có trọng số >= ngưỡng này được coi là cứng khi báo sai số
    HARD_AXIS_WEIGHT = 0.5
    # 1 cm sai số vị trí có trọng số ngang 0.1 rad sai lệch trục
    POS_WEIGHT = 10.0
    # Trọng số tư thế của waist_pitch so với khớp tay
    WAIST_POSTURE_SCALE = 5.0

    def __init__(self, model: mujoco.MjModel, side: str, point: np.ndarray, use_waist_pitch: bool = True,
                 limit_margin: float = 0.05, waist_pitch_range: Tuple[float, float] = (-0.05, 0.45),
                 min_shoulder_abduction: float = 0.05, max_wrist_pitch: float = 1.45):
        self.model = model
        self.side = side
        self.point = np.asarray(point, dtype=np.float64)
        self.wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
        self.joint_names = [f"{side}_{j}_joint" for j in ARM_JOINTS]
        if use_waist_pitch:
            self.joint_names.append("waist_pitch_joint")
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.qadr = model.jnt_qposadr[jids].copy()
        self.dadr = model.jnt_dofadr[jids].copy()
        self.lo = model.jnt_range[jids, 0] + limit_margin
        self.hi = model.jnt_range[jids, 1] - limit_margin
        if use_waist_pitch:
            self.lo[-1], self.hi[-1] = waist_pitch_range
        # Vai khép vào trong (shoulder_roll hướng vào thân) quá ~0.1 rad thì bắp tay gần như luôn lồng vào thân:
        # chỉ cho vai giạng ra ngoài ít nhất min_shoulder_abduction (roll dương ở tay trái, âm ở tay phải)
        if side == "left":
            self.lo[1] = max(self.lo[1], min_shoulder_abduction)
        else:
            self.hi[1] = min(self.hi[1], -min_shoulder_abduction)
        # |wrist_pitch| >= ~1.5 rad làm khâu wrist_roll chạm khâu wrist_yaw
        self.lo[5], self.hi[5] = max(self.lo[5], -max_wrist_pitch), min(self.hi[5], max_wrist_pitch)
        self.scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        # Body của cánh tay (từ vai ra tới ngón) và của toàn robot, để phát hiện tay va vào thân / chính nó
        def subtree(root_name):
            root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
            inside = np.zeros(model.nbody, dtype=bool)
            for b in range(model.nbody):
                a = b
                while a > 0 and a != root:
                    a = model.body_parentid[a]
                inside[b] = a == root
            return inside

        self._arm_body = subtree(f"{side}_shoulder_pitch_link")
        self._robot_body = subtree("pelvis")
        # wrist_roll_link và wrist_yaw_link (cách nhau một khớp) chạm nhau sát khớp ở góc cổ tay lớn; lồng nhau chưa
        # tới 1 mm, không cản chuyển động nên không tính là va chạm
        self._ignored_pair = tuple(sorted(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_{n}_link") for n in ("roll", "yaw")
        ))

    def forward(self, qpos: np.ndarray, q: Optional[np.ndarray] = None, point: Optional[np.ndarray] = None):
        """Động học thuận: (vị trí điểm trên bàn tay, ma trận quay cổ tay) với các khớp được chọn đặt bằng q."""
        d = self.scratch
        d.qpos[:] = qpos
        if q is not None:
            d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        rot = d.xmat[self.wrist_bid].reshape(3, 3).copy()
        return d.xpos[self.wrist_bid] + rot @ (self.point if point is None else point), rot

    def body_mask(self, names=()) -> np.ndarray:
        """Mảng bool theo body: True với các body có tên trong names."""
        mask = np.zeros(self.model.nbody, dtype=bool)
        for n in names:
            mask[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)] = True
        return mask

    @property
    def robot_mask(self) -> np.ndarray:
        return self._robot_body.copy()

    def collides(self, qpos: np.ndarray, q: np.ndarray, obstacles: Optional[np.ndarray] = None) -> bool:
        """Cánh tay có lồng vào vật cản nào không: các khâu của chính nó, hoặc body có obstacles[body] = True
        (mặc định: phần còn lại của robot)."""
        m, d = self.model, self.scratch
        obstacles = self._robot_body if obstacles is None else obstacles
        d.qpos[:] = qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        b1 = m.geom_bodyid[d.contact.geom1[: d.ncon]]
        b2 = m.geom_bodyid[d.contact.geom2[: d.ncon]]
        arm1, arm2 = self._arm_body[b1], self._arm_body[b2]
        ignored = (np.minimum(b1, b2) == self._ignored_pair[0]) & (np.maximum(b1, b2) == self._ignored_pair[1])
        return bool(np.any(((arm1 & arm2) | (arm1 & obstacles[b2]) | (arm2 & obstacles[b1])) & ~ignored))

    def self_collision(self, qpos: np.ndarray, q: np.ndarray) -> bool:
        """Cánh tay có va chạm với phần khác của robot hay giữa các khâu của chính nó không."""
        return self.collides(qpos, q)

    def solve(
        self,
        ref_qpos: np.ndarray,
        target: IKTarget,
        q_init: Optional[np.ndarray] = None,
        q_nominal: Optional[np.ndarray] = None,
        point: Optional[np.ndarray] = None,
        posture_weight: float = 0.02,
        max_iters: int = 150,
        damping: float = 1e-3,
        fixed: Optional[np.ndarray] = None,
    ) -> IKResult:
        """Giải IK; ref_qpos cho tư thế pelvis/chân và giá trị của các khớp không được chọn.

        Args:
            q_init: nghiệm khởi đầu (mặc định lấy từ ref_qpos).
            q_nominal: tư thế tham chiếu mà số hạng posture kéo về (mặc định bằng q_init).
            point: điểm trên bàn tay (khung cổ tay) cần đưa tới target.pos, mặc định self.point.
            fixed: mảng bool theo khớp; khớp True giữ nguyên giá trị của q_init.
        """
        point = self.point if point is None else np.asarray(point, np.float64)
        q = np.clip(ref_qpos[self.qadr] if q_init is None else np.asarray(q_init, np.float64), self.lo, self.hi)
        q_nom = q.copy() if q_nominal is None else np.asarray(q_nominal, np.float64)
        # Số hạng tư thế: eo nặng hơn khớp tay, vì cúi eo dời trọng tâm và làm cả người ngả theo
        w_post = posture_weight * np.ones(len(q))
        if self.joint_names[-1] == "waist_pitch_joint":
            w_post[-1] *= self.WAIST_POSTURE_SCALE
        eye = np.eye(len(q))
        movable = np.ones(len(q), dtype=bool) if fixed is None else ~np.asarray(fixed, dtype=bool)
        for _ in range(max_iters):
            J, r = self._residual(ref_qpos, q, target, point)
            # Khớp đã chạm giới hạn mà bước giải còn đẩy ra ngoài thì loại khỏi bước này (active set); nếu chỉ kẹp
            # góc thì các khớp còn lại không bù phần chuyển động bị cắt và bộ giải dừng ở nghiệm sai
            free = movable.copy()
            for _ in range(len(q)):
                Jf = J * free
                A = Jf.T @ Jf + damping * eye + np.diag(w_post ** 2)
                dq = np.linalg.solve(A, Jf.T @ r + w_post ** 2 * (q_nom - q)) * free
                blocked = free & (((q <= self.lo) & (dq < 0)) | ((q >= self.hi) & (dq > 0)))
                if not blocked.any():
                    break
                free &= ~blocked
            step = np.max(np.abs(dq))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, self.lo, self.hi)
            if step < 1e-5:
                break
        pos, rot = self.forward(ref_qpos, q, point)
        hard = [np.arccos(np.clip((rot @ a) @ (w / np.linalg.norm(w)), -1.0, 1.0))
                for a, w, wt in target.axes if wt >= self.HARD_AXIS_WEIGHT]
        return IKResult(
            q=q, pos_err=float(np.linalg.norm(target.pos - pos)), axis_err=float(max(hard, default=0.0)),
            posture_cost=float(np.linalg.norm(q - q_nom)),
        )

    def _residual(self, ref_qpos, q, target: IKTarget, point):
        m, d = self.model, self.scratch
        pos, rot = self.forward(ref_qpos, q, point)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jac(m, d, self._jacp, self._jacr, pos, self.wrist_bid)
        Jr = self._jacr[:, self.dadr]
        rows_J = [self.POS_WEIGHT * self._jacp[:, self.dadr]]
        rows_r = [self.POS_WEIGHT * (target.pos - pos)]
        for local, world, wt in target.axes:
            u = rot @ local
            v = world / np.linalg.norm(world)
            # Vận tốc góc xoay u về v: u x v (độ lớn sin góc lệch); lệch quá 90° thì dùng độ lớn 1 cho khỏi chậm lại
            err = np.cross(u, v)
            if u @ v < 0:
                norm = np.linalg.norm(err)
                err = err / norm if norm > 1e-9 else np.cross(u, np.eye(3)[np.argmin(np.abs(u))])
            rows_J.append(wt * Jr)
            rows_r.append(wt * err)
        return np.vstack(rows_J), np.concatenate(rows_r)
