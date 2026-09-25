"""
Module sim.pick_place_planner: Quỹ đạo gắp-đặt lon bằng IK online, dùng để sinh demo (collect_demos.py).

Khác quỹ đạo mẫu cũ (waypoint góc khớp cố định, chỉ chịu được lon lệch ~2 cm), mọi điểm mốc ở đây được giải IK
(sim.arm_ik) từ trạng thái mô phỏng: vị trí/hướng lon đo được, pose pelvis thực tế, vị trí khay. Nhờ vậy lon có thể
nằm ở vùng rộng, đứng hoặc nằm theo nhiều hướng, và action trong demo phụ thuộc thật sự vào vị trí lon mà model phải
nhìn ảnh mới suy ra được.

Cách gắp:
  - Lon đứng: gắp ngang như quỹ đạo mẫu cũ, trục z cổ tay thẳng đứng (ngón tay cuộn quanh thân lon), ngón tay
    hướng về phía trước theo hướng từ vai tới lon; tiếp cận từ phía ngoài, ngược pháp tuyến lòng bàn tay.
  - Lon nằm: trục z cổ tay trùng trục lon, bàn tay úp từ trên xuống nhưng xoay quanh trục lon LYING_DOWN_ANGLE về
    phía ngón tay (ngón trỏ/giữa chúc xuống phía xa, ngón cái ngửa lên phía gần): ngón cái lúc mở chĩa theo pháp
    tuyến lòng bàn tay, dài tới 11 cm, úp thẳng thì đâm xuống mặt bàn ~3 cm. Cho lệch khỏi hướng chuẩn này tối đa
    MAX_PALM_TILT khi không với tới; tiếp cận từ phía trên theo hướng chuẩn đó.
Khi đặt, bàn tay chỉ được xoay quanh trục đứng so với lúc gắp, nên lon đứng vẫn đứng và lon nằm vẫn nằm.

Mỗi lon luôn do tay cùng phía với khay của nó gắp (Coca tay phải, Pepsi tay trái), kể cả khi lon nằm ở nửa bàn bên
kia: khi đó (và chỉ khi đó) IK dùng thêm waist_yaw để xoay thân về phía lon (gắp chéo người). Lon ở phía mình thì eo
không xoay: bật waist_yaw cả lúc này làm gắp lon nằm trượt nhiều hơn (đo 24/09: 67% so với 76% thành công).

Các đoạn: vươn tay (ready -> điểm cao phía trên điểm chờ) -> hạ xuống điểm chờ -> tiến vào lon -> khép ngón ->
nhấc lên -> mang tới trên khay -> hạ xuống -> mở ngón -> rút tay lên cao -> về ready -> đứng yên. Đoạn hạ xuống và
tiến vào lon được lập ngược từ tư thế gắp (luôn kết thúc đúng nhánh nghiệm của tư thế gắp) rồi chạy bằng servo: mỗi
bước giải lại IK theo pose pelvis đo được, vì robot ngả người vài cm khi eo cúi và tay vươn xa. Mọi tư thế và đường
đi được kiểm tra va chạm của cánh tay với thân robot và với vật xung quanh (vật được phép chạm tuỳ đoạn).

Tay còn lại (tay rảnh) giữ cổ tay đứng yên trong không gian bằng cách bù theo góc waist_pitch.

Planner là generator: mỗi lần next() trả về 43 mục tiêu actuator cho một bước điều khiển 50 Hz; người gọi bước mô
phỏng rồi mới gọi next() tiếp, nên các đoạn sau được lập kế hoạch từ trạng thái mới nhất. Không lập được kế hoạch
(IK không tới, va chạm, gắp trượt) thì generator ném PlanningError.
"""

import copy
from typing import Dict, Iterator, List, Optional, Tuple

import mujoco
import numpy as np

from bridge.sonic_wbc_controller import SonicWBCController
from sim.arm_ik import ArmIK, IKResult, axis_target, rotation_target

UP = np.array([0.0, 0.0, 1.0])

# Kích thước lon và khay (scene_pepsi_coke_sorting.xml)
CAN_RADIUS = 0.033
CAN_HALF_HEIGHT = 0.058
TRAY_BASE_HALF_THICKNESS = 0.005

# Số bước điều khiển 50 Hz của từng đoạn
STEPS = {
    "reach": 45,     # từ tư thế ready tới điểm cao phía trên điểm chờ
    "descend": 20,   # hạ thẳng xuống điểm chờ trước lon
    "approach": 35,  # tiến thẳng vào lon
    "close": 20,     # khép ngón
    "lift": 30,      # nhấc thẳng lên
    "carry": 65,     # mang tới trên khay (chậm vừa phải: vung tay nhanh sang ngang làm bàn chân trượt xoay)
    "lower": 25,     # hạ thẳng xuống khay
    "release": 15,   # mở ngón
    "retreat": 30,   # rút tay ra và lên cao
    "return": 35,    # về điểm cao ngay phía trên tư thế ready
    "settle": 20,    # hạ thẳng xuống tư thế ready
    "hold": 35,      # đứng yên để model học dừng lại (mặc định; collect_demos.py --hold-seconds đổi qua hold_steps)
}
TOTAL_STEPS = sum(STEPS.values())

# Hình học của các điểm mốc (m)
UPRIGHT_GRASP_HEIGHT = 0.012   # điểm gắp cao hơn tâm lon đứng (tránh ngón chạm bàn)
LYING_GRASP_HEIGHT = 0.004     # điểm gắp cao hơn tâm lon nằm
PRE_BACKOFF = 0.07             # điểm chờ lùi ra ngược hướng tiếp cận
PRE_BACK_ALONG_FINGERS = 0.03  # lon đứng: lùi thêm dọc ngón tay
PRE_RISE = 0.02
VIA_RISE = 0.08                # điểm cao phía trên điểm chờ, nơi đoạn vươn tay kết thúc
LIFT_HEIGHT = 0.10
PLACE_CLEARANCE = 0.015        # đáy/thân lon cách mặt khay lúc thả
PLACE_TOWARD_ROBOT = 0.02      # đặt lon lệch từ tâm khay về phía robot, đỡ phải vươn xa (robot ngả người)
ABOVE_TRAY = 0.08
RETREAT_BACKOFF = 0.05
RETREAT_RISE = 0.12            # đủ cao để ngón tay qua được nắp lon đứng khi về tư thế ready
READY_RISE = 0.08              # điểm cao phía trên tư thế ready: về thẳng thì mép bàn tay quét lon vừa đặt trong khay
IDLE_RISE_PER_RAD = 0.15       # tay rảnh: nâng lên và lùi về (m) trên mỗi rad eo cúi thêm so với ready
FREE_PATH_BUMPS = (0.05, 0.12)  # cung nhô lên giữa đường đi Descartes trong khoảng trống (thử lần lượt)
# Lon nằm: hướng "xuống" trong khung cổ tay lệch khỏi pháp tuyến lòng bàn tay về phía ngón tay một góc này (xem
# docstring module); 35° giữ đầu ngón cái (x 0.065, |y| 0.114) cao hơn mặt bàn khi tâm lon ở điểm gắp
LYING_DOWN_ANGLE = np.radians(35.0)
MAX_PALM_TILT = np.radians(40.0)
# Lon nằm: trục cuộn ngón tay (z cổ tay) được lệch khỏi trục lon các góc này quanh trục đứng (ngón ôm chéo thân lon),
# phạt LYING_AXIS_COST mỗi rad lệch. Bắt trùng khít thì lon nằm chéo 20-60° so với hướng vươn tay buộc cổ tay vừa gập
# vừa xoắn gần giới hạn (dataset g1_can_sort_ik: 15/76 episode, bàn tay trông dị dạng)
LYING_AXIS_OFFSETS = np.radians([0.0, 15.0, -15.0, 30.0, -30.0])
LYING_AXIS_COST = 1.5
# Phạt tư thế cổ tay vừa gập vừa xoắn khi gắp lon nằm: WRIST_TWIST_COST * |wrist_pitch * wrist_yaw|
WRIST_TWIST_COST = 2.0
LIMIT_COST_RANGE = 0.15        # khớp cách giới hạn dưới khoảng này (rad) bị phạt
# Góc xoay quanh trục đứng được thử khi đặt lon (so với hướng bàn tay lúc gắp)
PLACE_YAWS = np.radians([0, 20, -20, 40, -40, 60, -60, 90, -90, 120, -120, 150, -150, 180])
# Sai số IK chấp nhận được trên đường đi Descartes: chặt khi ở sát lon/khay, lỏng khi đi trong khoảng trống
PATH_POS_TOL = 0.006
FREE_PATH_POS_TOL = 0.03
# Khoảng hở tối thiểu (m) từ cánh tay / lon đang cầm tới lon còn lại khi kiểm tra va chạm: lúc chạy thật robot ngả và
# xoay người nên tay lệch khỏi kế hoạch; gắp chéo người thì tay đi sát lon còn lại và từng quét đổ nó (đo 24/09 trên
# 120 episode đổi phía: 1 cm giữ nguyên số episode thu được, số lần xô lon còn lại giảm từ 12 xuống 5; 2 cm mất 1/4 số ca
# lon đứng)
OTHER_CAN_MARGIN = 0.01
# Khoảng hở lớn hơn được ưu tiên ở các đoạn không bắt buộc phải sát lon còn lại, thử lần lượt tới OTHER_CAN_MARGIN
# (không làm mất episode). Demo đi sát lon còn lại thì model, vốn lệch 2–5 cm, xô trúng nó (MEMO mục 5.2k: ca đổi phía
# model làm hỏng có demo cách lon còn lại trung vị 2.6 cm, ca làm được 5.4 cm):
#   - vươn tay, hạ xuống, tiến vào lon (tư thế gắp thì vẫn do khoảng cách hai lon quyết định);
#   - mang lon tới trên khay (sau khi nhấc lon lên cao hơn đỉnh lon còn lại CARRY_CLEARANCE).
# Mức đầu lớn hơn khoảng hở mong muốn vì lúc chạy tay lệch khỏi kế hoạch 1.5–2 cm (khớp tay bám chậm theo PD, thân
# chuyển động): lập với 3 cm thì chạy thật chỉ còn trung vị 1.5 cm (đo 25/09, 32 episode đổi phía, fixed và balance như nhau)
APPROACH_MARGINS = (0.05, 0.04, 0.03, 0.02, OTHER_CAN_MARGIN)
CARRY_MARGINS = (0.06, 0.04, 0.02, OTHER_CAN_MARGIN)
# Lon còn lại nằm gần đường mang lon (cách đoạn thẳng lon -> khay theo phương ngang dưới khoảng này): nhấc lon lên cho
# đáy lon đang cầm cao hơn đỉnh lon còn lại CARRY_CLEARANCE, và điểm trên khay cũng ở độ cao đó. Nhấc LIFT_HEIGHT như
# cũ thì đáy lon đứng đang cầm (~10 cm trên mặt bàn) thấp hơn đỉnh lon đứng còn lại (11.8 cm), tay đi ngang qua nó
CARRY_CORRIDOR = 0.15
CARRY_CLEARANCE = 0.04
# Lon đứng: điểm chờ lùi (dọc pháp tuyến lòng bàn tay, dọc ngón tay, lên cao) so với điểm gắp, thử lần lượt. Mặc định
# lùi 7 cm theo pháp tuyến lòng bàn tay; khi gắp chéo người hướng đó chĩa về phía lon còn lại (cổ tay cách nó ~1 cm)
# Lon nằm: kiểu gắp theo quy tắc cố định thay vì chi phí nhỏ nhất. Trục lon đối xứng nên có hai kiểu (trục cuộn ngón theo
# chiều này hay chiều kia của trục lon, cổ tay khác nhau ~2 rad); chọn theo chi phí thì quanh chỗ hai chi phí bằng nhau
# (Coca nằm cùng phía, yaw 90–120°) hai cảnh gần giống nhau cho hai kiểu, model lưỡng lự và đổi kiểu giữa chừng (MEMO
# 5.0e, 5.2g). Ưu tiên kiểu có trục cuộn ngón hướng về phía góc tham chiếu (độ, khung world, từng tay); chỉ dùng kiểu còn
# lại khi kiểu ưu tiên không lập được. Góc chọn từ 300 cảnh lon nằm (MEMO mục 5.2p): chi phí tăng TB 0.06–0.33, 1–2/75 cảnh
# phải dùng kiểu còn lại. None: chọn theo chi phí như trước 25/09.
LYING_MODE_REF_DEG = {"right": 125.0, "left": 215.0}
# Lon nằm: số pose gắp đầu tiên (chi phí thấp nhất) được xét ở các mức khoảng hở > OTHER_CAN_MARGIN; pose xếp sau hay trượt
LYING_CLEAR_GRASPS = 4
# Ba điểm chờ cuối không lùi ngang theo lòng bàn tay (hạ thẳng từ trên, hoặc tiến dọc ngón tay): gắp chéo người thì bàn tay
# chen giữa hai lon, hai lon cách nhau 5–10 cm thì mọi điểm chờ lùi ngang đều sát lon còn lại (thu được 0/50, MEMO 5.2q)
UPRIGHT_PRE_OFFSETS = ((PRE_BACKOFF, PRE_BACK_ALONG_FINGERS, PRE_RISE), (0.04, 0.07, 0.03), (0.035, 0.03, 0.07),
                       (0.0, 0.0, 0.09), (0.0, 0.08, 0.02), (0.015, 0.05, 0.05))


class PlanningError(RuntimeError):
    pass


def smoothstep(n: int) -> np.ndarray:
    """n trọng số nội suy smoothstep trong (0, 1], không gồm điểm đầu."""
    t = np.arange(1, n + 1) / n
    return t * t * (3.0 - 2.0 * t)


def slerp_rot(r0: np.ndarray, r1: np.ndarray, w: float) -> np.ndarray:
    """Nội suy cầu giữa hai ma trận quay (w = 0 -> r0, w = 1 -> r1)."""
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, r0.reshape(9))
    mujoco.mju_mat2Quat(q1, r1.reshape(9))
    dot = q0 @ q1
    if dot < 0:
        q1, dot = -q1, -dot
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    if theta < 1e-6:
        q = q0
    else:
        q = (np.sin((1.0 - w) * theta) * q0 + np.sin(w * theta) * q1) / np.sin(theta)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q / np.linalg.norm(q))
    return mat.reshape(3, 3)


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def is_lying(can_rot: np.ndarray) -> bool:
    return abs(can_rot[2, 2]) < 0.5


class PickPlacePlanner:
    """Lập quỹ đạo gắp lon `can` ('coke' tay phải, 'pepsi' tay trái, lon có thể ở bất kỳ nửa bàn nào) đặt vào khay
    cùng màu."""

    def __init__(self, sim, can: str, waist_yaw: Optional[bool] = None, hold_steps: Optional[int] = None):
        """waist_yaw: cho IK dùng waist_yaw; mặc định (None) chỉ bật khi lon đang ở nửa bàn bên kia (gắp chéo người).
        hold_steps: số bước 50 Hz đứng yên ở tư thế ready cuối episode (mặc định STEPS["hold"])."""
        m = sim.model
        self.hold_steps = STEPS["hold"] if hold_steps is None else int(hold_steps)
        self.sim = sim
        self.can = can
        self.grasp = sim.grasps[can]
        side = "right" if can == "coke" else "left"
        self.side_y = -1.0 if side == "right" else 1.0  # phía của tay gắp theo trục y
        self.pelvis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if waist_yaw is None:
            waist_yaw = self._is_cross(sim.data.xpos[self.grasp["can_bid"]])
        # IK dùng bản sao model có geom_margin quanh lon còn lại (chỉ ảnh hưởng kiểm tra va chạm, không đổi vật lý sim)
        ik_model = copy.deepcopy(m)
        other_bid = sim.grasps["pepsi" if can == "coke" else "coke"]["can_bid"]
        self.other_bid = other_bid
        ik_model.geom_margin[ik_model.geom_bodyid == other_bid] = OTHER_CAN_MARGIN
        # Khoảng hở lớn hơn OTHER_CAN_MARGIN (APPROACH_MARGINS, CARRY_MARGINS) đo bằng mj_geomDistance: geom_margin với
        # mesh (bàn tay) chỉ đúng một phần (lập với margin 5 cm, khoảng cách thật trên kế hoạch còn 3.2 cm)
        collide = (ik_model.geom_contype != 0) | (ik_model.geom_conaffinity != 0)
        self._other_geom_ids = np.flatnonzero(collide & (ik_model.geom_bodyid == other_bid))
        shoulder = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_pitch_link")
        arm_body = np.zeros(m.nbody, dtype=bool)
        for b in range(m.nbody):
            a = b
            while a > 0 and a != shoulder:
                a = m.body_parentid[a]
            arm_body[b] = a == shoulder
        self._arm_geom_ids = np.flatnonzero(collide & arm_body[ik_model.geom_bodyid])
        self._held_geom_ids = np.flatnonzero(collide & (ik_model.geom_bodyid == self.grasp["can_bid"]))
        self._clearance = OTHER_CAN_MARGIN
        self.ik = ArmIK(ik_model, side, self.grasp["grasp_point"], use_waist_yaw=waist_yaw)
        # Pháp tuyến lòng bàn tay trong khung cổ tay: hướng từ lòng bàn tay vào tâm nắm (+y tay phải, -y tay trái)
        palm_sign = np.sign(self.grasp["grasp_point"][1])
        self.palm_local = np.array([0.0, palm_sign, 0.0])
        # Hướng trong khung cổ tay cần chĩa xuống khi gắp lon nằm (hướng tiếp cận từ trên xuống)
        self.lying_down_local = np.array([np.sin(LYING_DOWN_ANGLE), palm_sign * np.cos(LYING_DOWN_ANGLE), 0.0])
        self.hand_closed = (SonicWBCController.RIGHT_HAND_CLOSED if side == "right"
                            else SonicWBCController.LEFT_HAND_CLOSED)
        self.tray_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{can}_tray")
        self.shoulder_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_pitch_link")

        act_of_joint = {int(m.actuator_trnid[i, 0]): i for i in range(m.nu)}
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.ik.joint_names]
        self.ctrl_idx = np.array([act_of_joint[j] for j in jids])
        self.default_ctrl = sim.default_ctrl.astype(np.float64)
        self.q_ready = self.default_ctrl[self.ctrl_idx]
        # Tư thế tham chiếu khi vươn tay: như ready nhưng eo cúi thêm
        self.q_nominal = self.q_ready.copy()
        self.q_nominal[-1] = 0.2
        self._seeds: Optional[List[np.ndarray]] = None
        self._init_idle_arm(sim, "left" if side == "right" else "right", act_of_joint)

        # Vật cản của cánh tay (ngoài thân robot, luôn là vật cản): đi trong khoảng trống thì không được chạm gì;
        # sát lon được chạm lon cần gắp, mặt bàn (ngón tay lướt qua) và khay đích (lon đứng sát khay thì mu bàn tay
        # sượt thành khay); khi cầm lon được chạm lon và khay đích
        everything = np.ones(m.nbody, dtype=bool)
        self.free_obstacles = everything
        self.grasp_obstacles = everything & ~self.ik.body_mask([f"{can}_can", "work_table", f"{can}_tray"])
        self.carry_obstacles = everything & ~self.ik.body_mask([f"{can}_can", f"{can}_tray"])

        self.phase = "plan"  # đoạn quỹ đạo đang chạy (để đo theo đoạn)
        ref = LYING_MODE_REF_DEG and LYING_MODE_REF_DEG[side]
        self.lying_mode_ref = None if ref is None else np.array([np.cos(np.radians(ref)), np.sin(np.radians(ref)), 0.0])
        self.info: Dict = {}  # thông tin kế hoạch (kiểu gắp, độ nghiêng bàn tay) để ghi log
        # Lon đang cầm (tham số attached của ArmIK.collides), để kiểm tra va chạm của cả lon khi mang/hạ lon: gắp chéo
        # người thì đường mang lon đi ngang trên lon còn lại
        self.attached = None
        self.can_qadr = int(m.jnt_qposadr[m.body_jntadr[self.grasp["can_bid"]]])

    def _init_idle_arm(self, sim, idle_side: str, act_of_joint: Dict[int, int]):
        """Bảng góc khớp tay rảnh theo waist_pitch để giữ cổ tay tay rảnh gần chỗ cũ trong không gian như ở tư thế
        ready, nâng dần lên khi eo cúi thêm (IDLE_RISE_PER_RAD) vì lúc đó cả người còn ngả về trước vài độ.

        Không bù thì khi eo cúi để vươn xa, bàn tay rảnh hạ thấp ~10 cm và đè lên khay / lon bên kia.
        """
        m = sim.model
        other_can = "pepsi" if self.can == "coke" else "coke"
        idle = ArmIK(m, idle_side, sim.grasps[other_can]["grasp_point"], use_waist_pitch=False)
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in idle.joint_names]
        self.idle_ctrl_idx = np.array([act_of_joint[j] for j in jids])
        waist_adr = self.ik.qadr[-1]
        qpos = sim.data.qpos.copy()
        qpos[sim.act_qposadr] = self.default_ctrl
        q_ready = self.default_ctrl[self.idle_ctrl_idx]
        pos, rot = idle.forward(qpos, q_ready)
        self.idle_waist = np.arange(self.ik.lo[-1], self.ik.hi[-1] + 0.005, 0.01)
        # Quét từ góc eo của tư thế ready ra hai phía, mỗi bước khởi đầu từ nghiệm bên cạnh và kéo về tư thế ready
        ready_waist = qpos[waist_adr]
        start = int(np.argmin(np.abs(self.idle_waist - ready_waist)))
        table = np.zeros((len(self.idle_waist), len(q_ready)))
        for order in (range(start, len(self.idle_waist)), range(start, -1, -1)):
            q = q_ready
            for i in order:
                qpos[waist_adr] = self.idle_waist[i]
                lift = IDLE_RISE_PER_RAD * max(self.idle_waist[i] - ready_waist, 0.0)
                q = idle.solve(qpos, rotation_target(pos + [-lift, 0.0, lift], rot), q_init=q, q_nominal=q_ready,
                               max_iters=60).q
                table[i] = q
        self.idle_table = table

    def _is_cross(self, can_pos: np.ndarray) -> bool:
        """Lon ở nửa bàn bên kia so với tay gắp (tính theo pelvis)."""
        return bool((can_pos[1] - self.sim.data.xpos[self.pelvis_bid][1]) * self.side_y < 0)

    def _set_margin(self, margin: float):
        """Khoảng hở tối thiểu tới lon còn lại cho các kiểm tra va chạm từ giờ."""
        self._clearance = margin

    def _too_close(self) -> bool:
        """Cánh tay (và lon đang cầm) trên self.ik.scratch (vừa được ArmIK.collides đặt tư thế) cách lon còn lại dưới
        self._clearance."""
        if self._clearance <= OTHER_CAN_MARGIN:
            return False
        m, d = self.ik.model, self.ik.scratch
        other = d.xpos[self.other_bid]
        if np.linalg.norm(d.xpos[self.grasp["wrist_bid"]] - other) > 0.45 + self._clearance:
            return False  # cả cánh tay (vai tới đầu ngón ~0.45 m tính từ cổ tay về vai) ở xa
        geoms = self._arm_geom_ids if self.attached is None else np.r_[self._arm_geom_ids, self._held_geom_ids]
        for a in geoms:
            for b in self._other_geom_ids:
                if mujoco.mj_geomDistance(m, d, int(a), int(b), self._clearance, None) < self._clearance:
                    return True
        return False

    def _carry_height(self, can_pos, tray_pos, held_half) -> Optional[float]:
        """Độ cao tâm lon đang cầm để đáy lon cao hơn đỉnh lon còn lại CARRY_CLEARANCE, nếu lon còn lại nằm gần đường
        mang lon (CARRY_CORRIDOR); None nếu không cần."""
        d = self.sim.data
        other = d.xpos[self.other_bid]
        a, b = np.asarray(can_pos[:2]), np.asarray(tray_pos[:2])
        t = np.clip((other[:2] - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0.0, 1.0)
        if np.linalg.norm(other[:2] - (a + t * (b - a))) > CARRY_CORRIDOR:
            return None
        other_half = CAN_RADIUS if is_lying(d.xmat[self.other_bid].reshape(3, 3)) else CAN_HALF_HEIGHT
        return float(other[2] + other_half + CARRY_CLEARANCE + held_half)

    def _idle_q(self, waist_pitch: float) -> np.ndarray:
        return np.array([np.interp(waist_pitch, self.idle_waist, col) for col in self.idle_table.T])

    # ------------------------------------------------------------------ IK nhiều điểm khởi đầu

    def _seed_postures(self, qpos: np.ndarray) -> List[np.ndarray]:
        """Các tư thế khởi đầu cho IK: ready, gắp lon đứng, gắp lon nằm ở vài hướng (lon ở phía tay mình, và ở nửa bàn
        bên kia với eo xoay về phía lon), cùng vài nhiễu cố định."""
        if self._seeds is not None:
            return self._seeds
        side_y = -1.0 if self.ik.side == "right" else 1.0
        seeds = [self.q_nominal]
        places = [(0.10 * side_y, self.q_nominal)]
        if "waist_yaw_joint" in self.ik.joint_names:
            # Tư thế tham chiếu khi gắp chéo người: eo xoay về nửa bàn bên kia
            q_cross = self.q_nominal.copy()
            q_cross[self.ik.joint_names.index("waist_yaw_joint")] = -0.4 * side_y
            places += [(-0.08 * side_y, q_cross), (-0.14 * side_y, q_cross)]
        for y, q_nom in places:
            c = np.array([0.32, y, 0.0])
            r = self.ik.solve(qpos, axis_target(c + [0, 0, 0.757], [((0, 0, 1), UP, 1.0), ((1, 0, 0), (1, 0, 0), 0.2)]),
                              q_init=q_nom, q_nominal=q_nom)
            seeds.append(r.q)
            for yaw in (30, 60, 90, 120, 150):
                yaw = np.radians(yaw if side_y < 0 else 180 - yaw)
                a = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                target = axis_target(c + [0, 0, 0.717], [((0, 0, 1), a, 1.0), (self.lying_down_local, -UP, 1.0)])
                seeds.append(self.ik.solve(qpos, target, q_init=q_nom, q_nominal=q_nom).q)
        rng = np.random.default_rng(12345)
        for _ in range(4):
            seeds.append(np.clip(self.q_nominal + rng.normal(0.0, 0.5, len(self.q_nominal)), self.ik.lo, self.ik.hi))
        self._seeds = seeds
        return seeds

    def _limit_cost(self, q: np.ndarray) -> float:
        """Phạt khớp nằm sát giới hạn: nghiệm như vậy dễ mất khi pelvis dịch nhẹ và không còn chỗ để tiến/lùi."""
        gap = np.minimum(q - self.ik.lo, self.ik.hi - q)
        return float(np.sum(np.clip((LIMIT_COST_RANGE - gap) / LIMIT_COST_RANGE, 0.0, 1.0)))

    def _best(self, qpos, targets, inits, obstacles, point=None, max_tilt=None) -> Tuple[Optional[IKResult], float]:
        """Nghiệm hợp lệ, không va chạm, có chi phí nhỏ nhất trên mọi (mục tiêu, điểm khởi đầu): độ lệch tư thế, khớp
        sát giới hạn và (nếu có max_tilt) độ nghiêng bàn tay khi gắp lon nằm; nghiêng quá max_tilt thì loại.

        Returns:
            (nghiệm hoặc None, chi phí của nghiệm).
        """
        best, best_cost = None, np.inf
        for target in targets:
            for q0 in inits:
                r = self.ik.solve(qpos, target, q_init=q0, q_nominal=self.q_nominal, point=point)
                if not r.ok() or self._collides(qpos, r.q, obstacles):
                    continue
                cost = r.posture_cost + self._limit_cost(r.q)
                if max_tilt is not None:
                    tilt = self._palm_tilt(qpos, r.q)
                    if tilt > max_tilt:
                        continue
                    cost += 2.0 * tilt + WRIST_TWIST_COST * abs(r.q[5] * r.q[6])
                if cost < best_cost:
                    best, best_cost = r, cost
        return best, best_cost

    def _palm_tilt(self, qpos, q) -> float:
        """Góc giữa hướng gắp lon nằm chuẩn (lying_down_local) và phương thẳng đứng hướng xuống."""
        _, rot = self.ik.forward(qpos, q)
        return float(np.arccos(np.clip(-(rot @ self.lying_down_local)[2], -1.0, 1.0)))

    def plan_grasps(self, qpos: np.ndarray, can_pos: np.ndarray, can_rot: np.ndarray) -> List[Dict]:
        """Các pose gắp cho lon ở can_pos/can_rot, mỗi kiểu tiếp cận một pose, xếp theo chi phí tăng dần; danh sách
        rỗng nếu không với tới. Mỗi pose là dict (q, rot, pos, lying, tilt)."""
        seeds = self._seed_postures(qpos)
        lying = is_lying(can_rot)
        if lying:
            axis = can_rot[:, 2].copy()
            axis[2] = 0.0
            axis /= np.linalg.norm(axis)
            pos = can_pos + [0.0, 0.0, LYING_GRASP_HEIGHT]
            # Hướng gắp chuẩn là ràng buộc cứng, hoặc mềm (cho nghiêng, có phạt); trục lon đối xứng nên thử hai chiều;
            # trục cuộn ngón được lệch khỏi trục lon LYING_AXIS_OFFSETS (có phạt)
            groups, group_cost, group_dir = [], [], []
            for s in (1, -1):
                for off in LYING_AXIS_OFFSETS:
                    a = rot_z(off) @ (s * axis)
                    groups.append([axis_target(pos, [((0, 0, 1), a, 1.0), (self.lying_down_local, -UP, w)])
                                   for w in (1.0, 0.3)])
                    group_cost.append(LYING_AXIS_COST * abs(off))
                    group_dir.append(s * axis)
        else:
            # Ngón tay hướng theo hướng vai -> lon. Lon ở nửa bàn bên kia (gắp chéo người) thì hướng từ vai gần như
            # nằm ngang, cổ tay chắn giữa hai lon và dễ quét trúng lon còn lại; thử thêm hướng từ tâm thân (ngón chĩa
            # về trước, nhưng lon sát người thì cổ tay lại vướng thân) và hướng ở giữa
            shoulder = self.sim.data.xpos[self.shoulder_bid][:2]
            origins = [shoulder]
            if self._is_cross(can_pos):
                pelvis = self.sim.data.xpos[self.pelvis_bid][:2]
                origins += [0.5 * (shoulder + pelvis), pelvis]
            pos = can_pos + [0.0, 0.0, UPRIGHT_GRASP_HEIGHT]
            groups = []
            group_cost = [0.0] * len(origins)
            group_dir = [None] * len(origins)
            for origin in origins:
                reach = np.r_[can_pos[:2] - origin, 0.0]
                reach /= np.linalg.norm(reach)
                groups.append([axis_target(pos, [((0, 0, 1), UP, 1.0), ((1, 0, 0), reach, 0.2)])])
        grasps = []
        for targets, extra, finger_axis in zip(groups, group_cost, group_dir):
            r, cost = self._best(qpos, targets, seeds, self.grasp_obstacles,
                                 max_tilt=MAX_PALM_TILT if lying else None)
            if r is None:
                continue
            cost += extra
            _, rot = self.ik.forward(qpos, r.q)
            # finger_axis (lon nằm): chiều trục lon mà trục cuộn ngón (z cổ tay) hướng theo, tức kiểu gắp; hai chiều
            # ngược nhau cho hai tư thế cổ tay khác nhau ~2 rad (MEMO mục 5.0e)
            grasps.append((cost, {"q": r.q, "rot": rot, "pos": pos, "lying": lying, "cost": cost,
                                  "finger_axis": finger_axis,
                                  "tilt": self._palm_tilt(qpos, r.q) if lying else 0.0}))
        return [g for _, g in sorted(grasps, key=lambda c: c[0])]

    # ------------------------------------------------------------------ các đoạn quỹ đạo

    def _ctrl(self, q: np.ndarray, grip: float) -> np.ndarray:
        ctrl = self.default_ctrl.copy()
        ctrl[self.ctrl_idx] = q
        ctrl[self.idle_ctrl_idx] = self._idle_q(q[-1])
        ctrl[self.grasp["hand_ctrl"]] = grip * self.hand_closed
        return ctrl.astype(np.float32)

    def _collides(self, qpos, q, obstacles) -> bool:
        return self.ik.collides(qpos, q, obstacles, self.attached) or self._too_close()

    def _check(self, qpos, q, obstacles, what):
        if self._collides(qpos, q, obstacles):
            raise PlanningError(f"tay va chạm khi {what}")

    def _joint_path(self, qpos, q_from, q_to, n, obstacles, what="") -> List[np.ndarray]:
        """Nội suy smoothstep trong không gian khớp, có kiểm tra va chạm."""
        qs = [(1.0 - w) * q_from + w * q_to for w in smoothstep(n)]
        for q in qs[::2]:
            self._check(qpos, q, obstacles, what)
        return qs

    def _pose_path(self, qpos, q_from, p_to, rot_to, n, obstacles, point=None, bump=0.0, tol=PATH_POS_TOL,
                   what="") -> List[np.ndarray]:
        """Đường Descartes từ pose hiện tại (q_from) tới (p_to, rot_to): vị trí theo đoạn thẳng cộng một cung nhô
        lên cao `bump` ở giữa, hướng cổ tay nội suy slerp; IK từng bước, khởi đầu từ nghiệm của bước trước."""
        p_from, rot_from = self.ik.forward(qpos, q_from, point)
        qs, q = [], q_from
        for w in smoothstep(n):
            pos = (1.0 - w) * p_from + w * p_to + 4.0 * w * (1.0 - w) * bump * UP
            r = self.ik.solve(qpos, rotation_target(pos, slerp_rot(rot_from, rot_to, w)), q_init=q, q_nominal=q,
                              point=point, max_iters=60)
            if r.pos_err > tol or r.axis_err > (0.1 if tol <= PATH_POS_TOL else 0.3):
                raise PlanningError(f"IK không bám được đường đi khi {what} (lệch {r.pos_err * 1000:.0f} mm)")
            self._check(qpos, r.q, obstacles, what)
            q = r.q
            qs.append(q)
        return qs

    def _free_path(self, qpos, q_from, q_to, n, obstacles, point=None, what="") -> List[np.ndarray]:
        """Đường đi trong khoảng trống giữa hai tư thế: nội suy khớp; va chạm thì thử đường Descartes có cung nhô
        lên (thấp rồi cao), vì bàn tay úp khi gắp lon nằm có ngón cái chĩa xuống ~11 cm dễ quét trúng lon."""
        try:
            return self._joint_path(qpos, q_from, q_to, n, obstacles, what)
        except PlanningError as e:
            error = e
        p_to, rot_to = self.ik.forward(qpos, q_to, point)
        for bump in FREE_PATH_BUMPS:
            try:
                path = self._pose_path(qpos, q_from, p_to, rot_to, n, obstacles, point, bump, FREE_PATH_POS_TOL, what)
            except PlanningError as e:
                error = e
                continue
            # Nối về đúng tư thế đích (đường Descartes có thể kết thúc ở nghiệm khác cùng pose)
            k = max(1, n // 4)
            for i, w in enumerate(smoothstep(k)):
                path[n - k + i] = (1.0 - w) * path[n - k + i] + w * q_to
            return path
        raise error

    def _return_path(self, qpos, q_from) -> List[np.ndarray]:
        """Từ q_from (tay đã rút lên khỏi khay) về điểm cao phía trên tư thế ready rồi hạ thẳng xuống.

        Chỉ kiểm tra va chạm với thân robot: lúc này robot còn ngả về trước vì tay đang giơ xa (tới ~8°), nên tính với
        pose pelvis hiện tại sẽ thấy bàn tay thấp và xa hơn thực tế, trúng cả lon trong khay."""
        qpos_upright = qpos.copy()
        qpos_upright[:7] = self.sim.model.qpos0[:7]
        p_ready, rot_ready = self.ik.forward(qpos_upright, self.q_ready)
        r = self.ik.solve(qpos_upright, rotation_target(p_ready + READY_RISE * UP, rot_ready), q_init=self.q_ready,
                          q_nominal=self.q_ready)
        q_high = r.q if r.pos_err < FREE_PATH_POS_TOL else self.q_ready
        path = self._free_path(qpos, q_from, q_high, STEPS["return"], self.ik.robot_mask, what="về tư thế ready")
        return path + [(1.0 - w) * q_high + w * self.q_ready for w in smoothstep(STEPS["settle"])]

    def return_to_ready(self, ctrl_now: np.ndarray) -> List[np.ndarray]:
        """Mục tiêu actuator 50 Hz đưa tay từ mục tiêu điều khiển hiện tại `ctrl_now` (đã nhả lon) nhấc thẳng lên
        RETREAT_RISE rồi về tư thế ready như cuối targets(). Dùng khi ngừng gọi model sau khi đặt xong lon
        (launch_vla_simulation.py --stop-on-success). Raises PlanningError nếu IK/va chạm không cho phép."""
        qpos = self.sim.data.qpos.copy()
        q_now = np.asarray(ctrl_now, dtype=np.float64)[self.ctrl_idx]
        p0, rot_now = self.ik.forward(qpos, q_now)
        path = self._pose_path(qpos, q_now, p0 + RETREAT_RISE * UP, rot_now, STEPS["retreat"], self.carry_obstacles,
                               tol=FREE_PATH_POS_TOL, what="nhấc tay khỏi khay")
        path += self._return_path(qpos, path[-1])
        return [self._ctrl(q, 0.0) for q in path]

    def _servo(self, qpos_plan, path, grip, obstacles, point=None) -> Iterator[np.ndarray]:
        """Chạy lại đường đã lập nhưng bám pose trong khung world: mỗi bước giải lại IK với pose pelvis đo được lúc
        đó. Khi robot ngả người (eo cúi, tay vươn xa) pelvis dịch vài cm, nên cùng góc khớp thì bàn tay lệch khỏi lon;
        bước nào IK không tới hoặc va chạm thì dùng tư thế đã lập. Eo giữ đúng giá trị đã lập, chỉ khớp tay được
        chỉnh: nếu eo bù theo độ ngả thì chính chuyển động eo lại làm thân lắc thêm (vòng phản hồi dao động)."""
        d = self.sim.data
        fixed = self.ik.waist_mask
        names = self.ik.joint_names
        yaw_free = None
        if "waist_yaw_joint" in names:
            yaw_free = fixed.copy()
            yaw_free[names.index("waist_yaw_joint")] = False
        q = path[0]
        for q_plan in path:
            pos, rot = self.ik.forward(qpos_plan, q_plan, point)
            qpos = d.qpos.copy()
            q_init = q.copy()
            q_init[fixed] = q_plan[fixed]
            r = self.ik.solve(qpos, rotation_target(pos, rot), q_init=q_init, q_nominal=q_plan, point=point,
                              max_iters=30, fixed=fixed)
            if r.pos_err >= PATH_POS_TOL and yaw_free is not None:
                # Gắp chéo người với chân balance: pelvis xoay ngược chiều eo ~9°, tay vươn xa gần giới hạn không bù nổi
                # (quay về tư thế đã lập thì bàn tay lệch ~5 cm, gắp trượt: MEMO mục 5.2q). Cho eo yaw bù thêm; eo pitch
                # vẫn giữ nguyên
                r2 = self.ik.solve(qpos, rotation_target(pos, rot), q_init=q_init, q_nominal=q_plan, point=point,
                                   max_iters=30, fixed=yaw_free)
                r = r2 if r2.pos_err < r.pos_err else r
            q = r.q if r.pos_err < PATH_POS_TOL and not self._collides(qpos, r.q, obstacles) else q_plan
            yield self._ctrl(q, grip)
        self._servo_last = q

    def _descent(self, qpos, g, pre_offset=(PRE_BACKOFF, PRE_BACK_ALONG_FINGERS, PRE_RISE)):
        """Đường từ điểm cao xuống điểm chờ rồi tiến vào lon, lập ngược từ tư thế gắp. pre_offset: điểm chờ lùi
        (dọc hướng tiếp cận, dọc ngón tay (chỉ lon đứng), lên cao) so với điểm gắp.

        Returns:
            (tư thế ở điểm cao, STEPS['descend'] + STEPS['approach'] tư thế tiếp theo, kết thúc ở tư thế gắp).
        """
        rot = g["rot"]
        back, along_fingers, rise = pre_offset
        pre = g["pos"] - back * (rot @ self._approach_local) + rise * UP
        if not g["lying"]:
            pre -= along_fingers * rot[:, 0]
        up1 = self._pose_path(qpos, g["q"], pre, rot, STEPS["approach"], self.grasp_obstacles, what="tiến vào lon")
        up2 = self._pose_path(qpos, up1[-1], pre + VIA_RISE * UP, rot, STEPS["descend"], self.grasp_obstacles,
                              tol=FREE_PATH_POS_TOL, what="hạ xuống điểm chờ")
        # Đảo chiều: điểm cao nhất là điểm bắt đầu (đoạn vươn tay đã tới đó), thêm tư thế gắp ở cuối
        down = (up1 + up2)[::-1]
        return down[0], down[1:] + [g["q"]]

    def targets(self) -> Iterator[np.ndarray]:
        """Generator mục tiêu actuator 50 Hz cho một lần gắp-đặt (xem docstring module)."""
        sim, d = self.sim, self.sim.data
        can_bid = self.grasp["can_bid"]

        # 1. Gắp: lập kế hoạch từ vị trí/hướng lon đo được lúc bắt đầu. Thử lần lượt các pose gắp, lấy pose đầu tiên
        #    mà đường vươn tay, đường hạ xuống/tiến vào lon và đường nhấc lon (kiểm tra trước) đều hợp lệ
        qpos = d.qpos.copy()
        grasps = self.plan_grasps(qpos, d.xpos[can_bid].copy(), d.xmat[can_bid].reshape(3, 3).copy())
        if not grasps:
            raise PlanningError("không có tư thế gắp với tới được")
        #    Lon nằm: kiểu gắp ưu tiên (LYING_MODE_REF_DEG) trước, rồi mới tới kiểu còn lại. Trong mỗi kiểu, khoảng hở tới
        #    lon còn lại thử APPROACH_MARGINS từ lớn tới nhỏ, với mỗi mức thử mọi pose gắp và mọi điểm chờ
        chosen, lift_ok = None, {}
        mode_groups = [list(range(len(grasps)))]
        if self.lying_mode_ref is not None and grasps[0]["lying"]:
            pref = [i for i, g in enumerate(grasps) if g["finger_axis"] @ self.lying_mode_ref >= 0]
            mode_groups = [pref, [i for i in range(len(grasps)) if i not in pref]]
        for mode_rank, group in enumerate(mode_groups):
          for margin in APPROACH_MARGINS:
            for rank, gi in enumerate(group):
                g = grasps[gi]
                # Lon nằm: pose gắp xếp sau (chi phí cao) hay trượt khi khép tay; chỉ đổi lấy khoảng hở với vài pose đầu
                if g["lying"] and rank >= LYING_CLEAR_GRASPS and margin > OTHER_CAN_MARGIN:
                    continue
                self._approach_local = self.lying_down_local if g["lying"] else self.palm_local
                for pre_offset in ([None] if g["lying"] else UPRIGHT_PRE_OFFSETS):
                    try:
                        if gi not in lift_ok:  # đường nhấc lon xuất phát sát lon còn lại: chỉ cần OTHER_CAN_MARGIN
                            self._set_margin(OTHER_CAN_MARGIN)
                            lift_ok[gi] = None
                            self._pose_path(qpos, g["q"], g["pos"] + LIFT_HEIGHT * UP, g["rot"], STEPS["lift"],
                                            self.grasp_obstacles, tol=FREE_PATH_POS_TOL, what="nhấc lon")
                            lift_ok[gi] = True
                        if not lift_ok[gi]:
                            raise PlanningError("tay va chạm khi nhấc lon")
                        self._set_margin(margin)
                        q_via, descent = (self._descent(qpos, g) if pre_offset is None
                                          else self._descent(qpos, g, pre_offset))
                        reach = self._free_path(qpos, self.q_ready, q_via, STEPS["reach"], self.free_obstacles,
                                                what="vươn tay")
                        chosen = (g, margin, pre_offset)
                        break
                    except PlanningError as e:
                        error = e
                if chosen:
                    break
            if chosen:
                break
          if chosen:
            break
        # Lúc chạy (servo) và các đoạn sát lon còn lại dùng khoảng hở tối thiểu
        self._set_margin(OTHER_CAN_MARGIN)
        if not chosen:
            raise error
        g, margin, pre_offset = chosen
        self.info.update(lying=g["lying"], palm_tilt_deg=float(np.degrees(g["tilt"])), grasp_q=g["q"],
                         approach_margin_cm=round(margin * 100, 1),
                         pre_offset=None if pre_offset is None else UPRIGHT_PRE_OFFSETS.index(pre_offset),
                         wrist_pitch=round(float(g["q"][self.ik.joint_names.index(f"{self.ik.side}_wrist_pitch_joint")]), 3))
        if g["lying"]:
            a = g["finger_axis"]
            self.info.update(finger_axis_deg=round(float(np.degrees(np.arctan2(a[1], a[0]))), 1),
                             mode_fallback=bool(len(mode_groups) > 1 and mode_rank > 0))
        self.phase = "reach"
        for q in reach:
            yield self._ctrl(q, 0.0)

        # 2. Hạ xuống và tiến vào lon, bám pose trong khung world theo pose pelvis hiện tại
        self.phase = "approach"
        yield from self._servo(qpos, descent, 0.0, self.grasp_obstacles)
        q_now = self._servo_last
        self.phase = "close"
        for w in smoothstep(STEPS["close"]):
            yield self._ctrl(q_now, w)
        if not sim.is_grasped(self.can):
            raise PlanningError("gắp trượt (weld không bật)")

        # 3. Nhấc lên: điểm điều khiển từ đây là tâm lon trong khung cổ tay (đúng như lúc weld giữ)
        qpos = d.qpos.copy()
        w_pos, w_rot = d.xpos[self.grasp["wrist_bid"]], d.xmat[self.grasp["wrist_bid"]].reshape(3, 3)
        can_in_hand = w_rot.T @ (d.xpos[can_bid] - w_pos)
        self.attached = (can_bid, self.can_qadr, can_in_hand, w_rot.T @ d.xmat[can_bid].reshape(3, 3))
        p0, rot = self.ik.forward(qpos, q_now, can_in_hand)
        # Lon còn lại gần đường mang lon: nhấc cao hơn đỉnh nó CARRY_CLEARANCE (tốc độ như nhấc LIFT_HEIGHT); không
        # được thì nhấc LIFT_HEIGHT như cũ
        half = CAN_RADIUS if g["lying"] else CAN_HALF_HEIGHT
        high_z = self._carry_height(d.xpos[can_bid], d.xpos[self.tray_bid], half)
        lifts = [LIFT_HEIGHT]
        if high_z is not None and high_z - p0[2] > LIFT_HEIGHT:
            lifts.insert(0, high_z - p0[2])
        for lift in lifts:
            try:
                path = self._pose_path(qpos, q_now, p0 + lift * UP, rot, round(STEPS["lift"] * lift / LIFT_HEIGHT),
                                       self.grasp_obstacles, point=can_in_hand, tol=FREE_PATH_POS_TOL, what="nhấc lon")
                break
            except PlanningError as e:
                error = e
        else:
            raise error
        self.info["lift_cm"] = round(lift * 100, 1)
        self.phase = "lift"
        for q in path:
            yield self._ctrl(q, 1.0)
        q_now = path[-1]

        # 4. Mang tới trên khay: thử các góc xoay quanh trục đứng theo thứ tự chi phí tăng dần, lấy góc đầu tiên mà
        #    tư thế trên khay, tư thế đặt, đường mang lon và đường hạ lon đều hợp lệ
        #    Lon còn lại gần đường mang lon: điểm trên khay ở độ cao đã nhấc, đường mang lon kiểm tra với CARRY_MARGINS từ
        #    lớn tới nhỏ; cuối cùng thử điểm trên khay ABOVE_TRAY với OTHER_CAN_MARGIN như cũ
        qpos = d.qpos.copy()
        tray = d.xpos[self.tray_bid]
        place = np.array([tray[0] - PLACE_TOWARD_ROBOT, tray[1],
                          tray[2] + TRAY_BASE_HALF_THICKNESS + half + PLACE_CLEARANCE])
        options = []
        if high_z is not None and high_z > place[2] + ABOVE_TRAY:
            options += [(high_z - place[2], m) for m in CARRY_MARGINS[:-1]]
        options.append((ABOVE_TRAY, OTHER_CAN_MARGIN))
        plan = None
        for above_h, margin in options:
            self._set_margin(margin)
            above = place + above_h * UP
            candidates = []
            for yaw in PLACE_YAWS:
                rot_p = rot_z(yaw) @ rot
                r_above, _ = self._best(qpos, [rotation_target(above, rot_p)], [q_now] + self._seed_postures(qpos),
                                        self.carry_obstacles, can_in_hand)
                if r_above is not None:
                    candidates.append((r_above.posture_cost + 0.3 * abs(yaw), rot_p, r_above.q))
            for _, rot_p, q_above in sorted(candidates, key=lambda c: c[0]):
                try:
                    self._set_margin(margin)
                    carry = self._free_path(qpos, q_now, q_above, STEPS["carry"], self.carry_obstacles, can_in_hand,
                                            what="mang lon")
                    # Hạ xuống khay thì lon còn lại có thể nằm sát khay: chỉ cần OTHER_CAN_MARGIN
                    self._set_margin(OTHER_CAN_MARGIN)
                    lower = self._pose_path(qpos, q_above, place, rot_p,
                                            max(STEPS["lower"], round(STEPS["lower"] * above_h / ABOVE_TRAY)),
                                            self.carry_obstacles, point=can_in_hand, what="hạ lon")
                    plan = (carry, lower, above_h, margin)
                    break
                except PlanningError:
                    continue
            if plan:
                break
        self._set_margin(OTHER_CAN_MARGIN)
        if plan is None:
            raise PlanningError("không đặt được lon vào khay")
        carry, lower, above_h, margin = plan
        self.info.update(carry_margin_cm=round(margin * 100, 1), above_tray_cm=round(above_h * 100, 1))
        self.phase = "carry"
        for q in carry:
            yield self._ctrl(q, 1.0)

        # 5. Hạ xuống (servo theo pose pelvis hiện tại), thả lon
        self.phase = "lower"
        yield from self._servo(qpos, lower, 1.0, self.carry_obstacles, point=can_in_hand)
        q_now = self._servo_last
        for w in smoothstep(STEPS["release"]):
            yield self._ctrl(q_now, 1.0 - w)
        self.attached = None
        self.phase = "retreat"

        # 6. Rút tay ra ngược hướng tiếp cận và lên cao, rồi về tư thế ready (lập từ trạng thái sau khi rút tay, lúc
        #    lon đã nằm yên trong khay)
        qpos = d.qpos.copy()
        p0, rot_now = self.ik.forward(qpos, q_now)
        p1 = p0 - RETREAT_BACKOFF * (rot_now @ self._approach_local) + RETREAT_RISE * UP
        path = self._pose_path(qpos, q_now, p1, rot_now, STEPS["retreat"], self.carry_obstacles,
                               tol=FREE_PATH_POS_TOL, what="rút tay")
        for q in path:
            yield self._ctrl(q, 0.0)
        for q in self._return_path(d.qpos.copy(), path[-1]):
            yield self._ctrl(q, 0.0)
        for _ in range(self.hold_steps):
            yield self._ctrl(self.q_ready, 0.0)
