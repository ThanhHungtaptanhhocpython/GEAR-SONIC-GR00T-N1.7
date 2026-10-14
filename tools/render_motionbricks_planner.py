"""Trực quan hoá đầu ra thật của Kinematic Planner SONIC (huấn luyện bằng MotionBricks).

Chạy checkpoints/GEAR-SONIC/planner_sonic.onnx trên CPU: từ cùng một tư thế đứng, gọi planner với 4 kiểu chuyển động,
mỗi kiểu ghép 3 lần gọi liên tiếp (lập kế hoạch lại sau mỗi 1 s, ngữ cảnh lấy từ đoạn đang phát như chương trình
triển khai của NVIDIA). Tư thế chỉ được đặt vào MuJoCo để hiển thị, không mô phỏng vật lý.

    report/figures/motionbricks/planner_onion.png        ảnh chồng khung, 4 hàng, cho báo cáo
    report/slides/assets/motionbricks/planner_demo.mp4   video 2x2 cho slide
    report/slides/assets/motionbricks/planner_demo_still.png   ảnh tĩnh của video (bản PDF của slide)

Cần: onnxruntime, mujoco, pillow, av (có sẵn trong .venv). Máy không có màn hình thì dùng MUJOCO_GL=egl.
Dùng:  .venv/bin/python tools/render_motionbricks_planner.py
"""
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")
import av  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
ONNX = REPO / "checkpoints/GEAR-SONIC/planner_sonic.onnx"
XML = REPO / "third_party/GR00T-WholeBodyControl/motionbricks/assets/skeletons/g1/scene_29dof.xml"
FIG_DIR = REPO / "report/figures/motionbricks"
SLIDE_DIR = REPO / "report/slides/assets/motionbricks"
FPS = 30  # planner xuất 30 khung mỗi giây
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# (nhãn, chỉ số kiểu chuyển động của planner V2, độ cao root mong muốn hoặc -1)
MODES = [("Đi", 2, -1.0), ("Chạy", 3, -1.0), ("Ngồi xổm", 4, 0.45), ("Bò bằng tay", 8, -1.0)]

sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])


def plan(context, mode, direction=(1.0, 0.0, 0.0), height=-1.0, seed=1234):
    inp = {
        "context_mujoco_qpos": context[None].astype(np.float32),
        "target_vel": np.array([-1.0], np.float32),  # tốc độ mặc định của kiểu chuyển động
        "mode": np.array([mode], np.int64),
        "movement_direction": np.array([direction], np.float32),
        "facing_direction": np.array([[1.0, 0.0, 0.0]], np.float32),
        "random_seed": np.array([seed], np.int64),
        "has_specific_target": np.array([[0]], np.int64),
        "specific_target_positions": np.zeros([1, 4, 3], np.float32),
        "specific_target_headings": np.zeros([1, 4], np.float32),
        "allowed_pred_num_tokens": np.ones([1, 11], np.int64),
        "height": np.array([height], np.float32),
    }
    qpos, n = sess.run(None, inp)
    return qpos[0, : int(np.asarray(n).reshape(-1)[0])]


def standing_pose():
    """Tư thế đứng: dựng thô rồi cho planner chạy kiểu đứng yên vài lần để ra tư thế tự nhiên."""
    q0 = np.zeros(36)
    q0[2], q0[3] = 0.79, 1.0
    for side in (0, 6):  # hông gập, gối gập, cổ chân gập của hai chân
        q0[7 + side + 0], q0[7 + side + 3], q0[7 + side + 4] = -0.1, 0.3, -0.2
    ctx = np.repeat(q0[None], 4, axis=0)
    for _ in range(3):
        ctx = plan(ctx, mode=0, direction=(0.0, 0.0, 0.0))[-4:]
    return ctx


def rollout(start, mode, height, n_calls=3, replan_at=FPS):
    pieces, ctx = [], start
    for i in range(n_calls):
        seq = plan(ctx, mode=mode, height=height, seed=1234 + i)
        pieces.append(seq if i == n_calls - 1 else seq[:replan_at])
        ctx = seq[replan_at - 4:replan_at]
    return np.concatenate(pieces, axis=0)


def onion(model, data, seqs, path, seconds=3.0, n_pose=7, W=1300, H=430, view_h=1.9):
    """Mỗi hàng một kiểu chuyển động; camera chiếu song song nhìn ngang, cùng tỉ lệ cho mọi hàng."""
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    rend = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.orthographic = 1
    cam.azimuth, cam.elevation, cam.distance = 90, -6, 6.0
    fovy0, ortho0 = model.vis.global_.fovy, model.vis.global_.orthographic
    model.vis.global_.orthographic = 1
    model.vis.global_.fovy = view_h  # chiếu song song: fovy là chiều cao nhìn thấy, đơn vị mét
    scale = H / view_h  # điểm ảnh mỗi mét
    f_lab, f_t = ImageFont.truetype(FONT_B, 30), ImageFont.truetype(FONT, 24)
    LW = 230
    sheet = Image.new("RGB", (LW + W, H * len(seqs)), "white")
    draw = ImageDraw.Draw(sheet)
    c_old, c_new = np.array([40, 95, 215.0]), np.array([235, 115, 30.0])
    for r, ((label, _, _), seq) in enumerate(zip(MODES, seqs)):
        seq = seq[: int(seconds * FPS) + 1]
        lx = seq[0, 0] + W / 2 / scale - 0.6  # mọi hàng cùng khung nhìn; tư thế đầu cách mép trái 0,6 m
        cam.lookat[:] = [lx, 0.0, 0.85]
        canvas = np.full((H, W, 3), 247.0)  # nền trơn: sàn caro nhìn ngang tạo vân sọc khi in
        idx = np.linspace(0, len(seq) - 1, n_pose).round().astype(int)
        for j, k in enumerate(idx):
            u = j / (n_pose - 1)
            data.qpos[:] = seq[k]
            mujoco.mj_forward(model, data)
            rend.update_scene(data, camera=cam)
            rgb = rend.render().astype(np.float32)
            rend.enable_segmentation_rendering()
            rend.update_scene(data, camera=cam)
            seg = rend.render()
            rend.disable_segmentation_rendering()
            mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0) & (seg[..., 0] != floor)
            col = 0.45 * rgb + 0.55 * ((1 - u) * c_old + u * c_new)
            a = 0.35 + 0.65 * u  # tư thế cũ mờ, tư thế mới rõ
            canvas[mask] = (1 - a) * canvas[mask] + a * col[mask]
        tile = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
        d = ImageDraw.Draw(tile)
        gy = H / 2 + 0.85 * scale  # mặt sàn (z = 0) trên ảnh
        d.line([(0, gy), (W, gy)], fill=(150, 150, 150), width=2)
        for k, txt in ((idx[0], "0 s"), (idx[-1], f"{seconds:.0f} s")):
            px = W / 2 + (seq[k, 0] - lx) * scale
            d.text((px - 20, 10), txt, fill=(60, 60, 60), font=f_t)
        d.line([(0, H - 1), (W, H - 1)], fill=(210, 210, 210), width=2)
        sheet.paste(tile, (LW, r * H))
        draw.text((16, r * H + H // 2 - 34), label, fill="black", font=f_lab)
        draw.text((16, r * H + H // 2 + 8), f"di chuyển {seq[-1, 0] - seq[0, 0]:.1f} m".replace(".", ","), fill="dimgray", font=f_t)
    sheet.save(path)
    model.vis.global_.fovy, model.vis.global_.orthographic = fovy0, ortho0


def video(model, data, seqs, mp4, still, W=640, H=520, still_t=2.6):
    rend = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.0, 120, -12
    f_big, f_small = ImageFont.truetype(FONT_B, 28), ImageFont.truetype(FONT, 22)
    T = max(len(s) for s in seqs)
    frames = []
    for t in range(T):
        sheet = Image.new("RGB", (2 * W, 2 * H), "white")
        for i, ((label, _, _), seq) in enumerate(zip(MODES, seqs)):
            k = min(t, len(seq) - 1)
            data.qpos[:] = seq[k]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [seq[k, 0], seq[k, 1], 0.55]
            rend.update_scene(data, camera=cam)
            tile = Image.fromarray(rend.render())
            d = ImageDraw.Draw(tile)
            d.rectangle([0, 0, W, 42], fill=(255, 255, 255))
            d.text((14, 6), label, fill="black", font=f_big)
            d.text((W - 130, 10), f"t = {k / FPS:.1f} s", fill="dimgray", font=f_small)
            sheet.paste(tile, ((i % 2) * W, (i // 2) * H))
        frames.append(sheet)
    with av.open(str(mp4), "w") as c:
        s = c.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = 2 * W, 2 * H, "yuv420p"
        s.options = {"crf": "23"}
        for im in frames + [frames[-1]] * FPS:  # giữ khung cuối 1 s
            for p in s.encode(av.VideoFrame.from_image(im)):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    frames[min(int(still_t * FPS), T - 1)].save(still)
    return T


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SLIDE_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(XML))
    model.vis.global_.offwidth, model.vis.global_.offheight = 1600, 1200
    data = mujoco.MjData(model)
    start = standing_pose()
    seqs = [rollout(start, m, h) for _, m, h in MODES]
    for (label, _, _), s in zip(MODES, seqs):
        print(f"{label}: {len(s)} khung = {len(s) / FPS:.1f} s, root dời {np.round(s[-1, :3] - s[0, :3], 2)} m")
    T = video(model, data, seqs, SLIDE_DIR / "planner_demo.mp4", SLIDE_DIR / "planner_demo_still.png")
    onion(model, data, seqs, FIG_DIR / "planner_onion.png")
    print(f"Đã ghi {FIG_DIR / 'planner_onion.png'}; video {T} khung vào {SLIDE_DIR}")


if __name__ == "__main__":
    main()
