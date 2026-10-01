"""
Sinh hai notebook Colab từ mã nguồn trong repo, để các cell %%writefile luôn khớp với file gốc:
  - colab_gr00t_server.ipynb:   chạy PolicyServer GR00T (checkpoint base hoặc checkpoint fine-tune)
  - colab_gr00t_finetune.ipynb: fine-tune GR00T N1.7 trên dataset do collect_demos.py tạo ra

Chạy lại sau khi sửa colab_gr00t_server.py hoặc finetune/g1_can_sort_config.py:
    .venv/bin/python tools/build_colab_notebooks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Commit Isaac-GR00T đã đối chiếu API Gr00tPolicy, finetune.sh, stats.py, open_loop_eval.py
ISAAC_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"


def lines(text):
    text = text.strip("\n")
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


def writefile(dest: str, src: Path):
    return code(f"%%writefile {dest}\n" + src.read_text())


GPU_CELL = code("""
# [BƯỚC 1] Kiểm tra GPU
!nvidia-smi
""")

INSTALL_CELL = code(f"""
%%bash
# [BƯỚC 2] Cài Isaac-GR00T vào môi trường riêng (.venv, Python 3.12) bằng uv (~5-10 phút).
# set -e: dừng ngay ở lệnh lỗi đầu tiên để thấy rõ nguyên nhân (lệnh "!" của Colab không tự dừng khi lỗi).
set -euxo pipefail
cd /content
[ -d Isaac-GR00T/.git ] || GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
git checkout {ISAAC_COMMIT}
# Clone bỏ qua file Git LFS (video, dataset demo) cho nhẹ, nhưng uv cần wheel LFS trong
# scripts/deployment/dgpu/wheels để giải lock (kể cả wheel aarch64 không dùng trên Colab): tải riêng thư mục này.
command -v git-lfs || (apt-get -qq update && apt-get -qq install -y git-lfs)
git lfs install --local
git lfs pull --include="scripts/deployment/dgpu/wheels/*"
# Trình cài uv chọn /usr/local/bin hoặc ~/.local/bin tùy môi trường: gọi uv qua PATH, không dùng đường dẫn cứng
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
# Venv luôn nằm ở /content/Isaac-GR00T/.venv (bỏ qua biến môi trường nếu có)
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
uv sync --python 3.12
.venv/bin/python -c "import gr00t; print('GR00T installed successfully')"
""")

HF_TOKEN_CELL = code("""
# [BƯỚC 3] Nạp HuggingFace token từ Colab Secrets (không dán token trực tiếp vào notebook)
# và kiểm tra ngay quyền tải backbone gated nvidia/Cosmos-Reason2-2B.
# Chạy lại cell này mỗi khi runtime khởi động lại.
import os
from huggingface_hub import hf_hub_download, whoami
from huggingface_hub.utils import GatedRepoError

try:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
except Exception as e:
    print(f"Không đọc được secret HF_TOKEN ({type(e).__name__}); nhập token thủ công.")
    from getpass import getpass
    token = getpass("HuggingFace token: ")
token = (token or "").strip()
assert token, "HF_TOKEN rỗng: thêm secret tên HF_TOKEN (biểu tượng 🔑) và bật Notebook access."
os.environ["HF_TOKEN"] = token

try:
    user = whoami(token=token)
except Exception as e:
    raise RuntimeError(
        "Token không hợp lệ hoặc đã bị thu hồi. Tạo token mới tại https://huggingface.co/settings/tokens "
        "rồi cập nhật secret HF_TOKEN."
    ) from e
print(f"Token hợp lệ, tài khoản: {user['name']}")

try:
    hf_hub_download("nvidia/Cosmos-Reason2-2B", "config.json", token=token)
except GatedRepoError as e:
    raise RuntimeError(
        f"Tài khoản {user['name']} chưa tải được nvidia/Cosmos-Reason2-2B. Kiểm tra:\\n"
        "  1. Đã bấm xin quyền và được duyệt tại https://huggingface.co/nvidia/Cosmos-Reason2-2B "
        "(đăng nhập đúng tài khoản trên).\\n"
        "  2. Nếu là token fine-grained: bật quyền 'Read access to contents of all public gated repos "
        "you can access', hoặc dùng token loại Read."
    ) from e
print("✅ Có quyền tải nvidia/Cosmos-Reason2-2B.")
""")

VENV_CHECK = '''assert os.path.exists("/content/Isaac-GR00T/.venv/bin/python"), (
    "Chưa có môi trường Isaac-GR00T: Bước 2 chưa chạy xong. Chạy lại Bước 2 và xem dòng lỗi đầu tiên."
)
assert os.environ.get("HF_TOKEN"), "Chưa có HF_TOKEN trong phiên này: chạy lại Bước 3 (sau mỗi lần runtime khởi động lại)."'''


def server_notebook():
    return [
        md(f"""
# 🚀 NVIDIA GR00T-N1.7 VLA PolicyServer trên Google Colab

Notebook chạy model **GR00T N1.7 thật** trên GPU Colab và nối với mô phỏng MuJoCo trên máy local qua đường hầm TCP Pinggy. Dùng được cho checkpoint base (`nvidia/GR00T-N1.7-3B`, embodiment `REAL_G1`) và checkpoint fine-tune từ `colab_gr00t_finetune.ipynb` (embodiment `new_embodiment`).

### Yêu cầu
* **GPU:** nên dùng **L4** hoặc **A100**. T4 vẫn chạy được (server tự tắt flash-attn), nhưng bfloat16 trên T4 là giả lập nên suy luận chậm; khi đó ở máy local tăng `--server-timeout-ms`.
* **HuggingFace:** backbone `nvidia/Cosmos-Reason2-2B` là repo **gated**. Tài khoản của bạn phải được cấp quyền trên trang model, rồi thêm token vào **Colab Secrets** (biểu tượng 🔑 bên trái) với tên `HF_TOKEN` và bật *Notebook access*.

### Các bước
1. **Runtime → Change runtime type → L4 GPU** (hoặc A100).
2. Chạy lần lượt từ Bước 1 đến Bước 6. Cell `%%writefile` (Bước 4) ghi file server. Ở Bước 5 chọn checkpoint; server **chạy nền**.
3. Copy lệnh in ra ở Bước 6 sang terminal máy local.
4. Tunnel Pinggy miễn phí hết hạn sau khoảng 60 phút: chạy lại Bước 6 để lấy host/port mới, không cần nạp lại model.
   Bước 7 xem trạng thái và log server; Bước 8 dừng server (phải đặt `STOP_SERVER = True`).

Với checkpoint base, model chưa từng thấy cảnh sim này nên robot sẽ cử động nhưng gần như chắc chắn không gắp được lon.

Isaac-GR00T được ghim ở commit `{ISAAC_COMMIT[:7]}`.
"""),
        GPU_CELL,
        INSTALL_CELL,
        HF_TOKEN_CELL,
        writefile("/content/colab_gr00t_server.py", ROOT / "colab_gr00t_server.py"),
        code(r"""
# [BƯỚC 5] Khởi chạy PolicyServer chạy nền (port 5550), log ghi ra /content/server.log.
# Cell chờ tới khi server sẵn sàng. Bấm ■ chỉ dừng việc chờ, server vẫn chạy; xem trạng thái ở Bước 7, dừng ở Bước 8.
# Checkpoint fine-tune: đường dẫn checkpoint-<bước> trên Drive (Bước 10 của colab_gr00t_finetune.ipynb),
#   thư mục con = DATASET_NAME lúc fine-tune (checkpoint pilot cũ: .../g1_can_sort_pilot/checkpoint-5000).
# Checkpoint base (chỉ để thử, không gắp được lon): lần đầu tải GR00T-N1.7-3B (~7 GB) và backbone Cosmos-Reason2-2B
#   MODEL_PATH = "nvidia/GR00T-N1.7-3B"
#   EMBODIMENT_TAG = "real_g1_relative_eef_relative_joints"
#   (tools/eval_closed_loop.py mặc định dừng nếu server không chạy checkpoint new_embodiment)
# Checkpoint trên Drive được chép về /content/ckpt trước: nạp thẳng từ Drive rất chậm (~13 phút cho 3 shard),
# nạp từ đĩa local nhanh hơn nhiều. Lần chạy sau trên cùng runtime bỏ qua các file đã chép đủ.
# Lần chép đầu vẫn bị giới hạn bởi tốc độ Drive (~15 MB/s, có thể vài phút mỗi shard); tiến độ in mỗi 30 s.
# Nếu Drive rớt (Errno 107) thì tự mount lại và chép tiếp từ chỗ dừng; chạy lại cell cũng chép tiếp từ file .part.
# POLICY = "scripted" để phục vụ quỹ đạo mẫu thay cho model.
MODEL_PATH = "/content/drive/MyDrive/gr00t/checkpoints/g1_can_sort_ik/checkpoint-10000"
EMBODIMENT_TAG = "new_embodiment"
POLICY = "gr00t"

import os, re, socket, subprocess, time
""" + VENV_CHECK + r"""


def port_open(port=5550):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


assert not port_open(), "Port 5550 đang có server chạy. Dừng bằng Bước 8 rồi chạy lại cell này."

if MODEL_PATH.startswith("/content/drive"):
    import errno
    from google.colab import drive
    drive.mount("/content/drive")
    assert os.path.isdir(MODEL_PATH), f"Không thấy checkpoint {MODEL_PATH}"
    local = "/content/ckpt/" + "_".join(MODEL_PATH.rstrip("/").split("/")[-2:])
    jobs = [(os.path.join(root, name), os.path.getsize(os.path.join(root, name)))
            for root, _, files in os.walk(MODEL_PATH) for name in sorted(files)]
    total = sum(size for _, size in jobs)
    print(f"Checkpoint trên Drive: {len(jobs)} file, {total / 1e9:.2f} GB")

    def copy_resumable(src, dst, size, chunk=64 << 20):
        # Chép theo khối và nối tiếp vào .part nếu lần trước bị ngắt; in tiến độ mỗi 30 s
        part = dst + ".part"
        done = os.path.getsize(part) if os.path.exists(part) else 0
        t0, t_print, start = time.time(), time.time(), done
        with open(src, "rb") as fi, open(part, "ab") as fo:
            fi.seek(done)
            while done < size:
                buf = fi.read(chunk)
                if not buf:
                    raise OSError(errno.EIO, f"Đọc thiếu {src}: {done}/{size} byte")
                fo.write(buf)
                done += len(buf)
                if time.time() - t_print > 30:
                    rate = (done - start) / 1e6 / (time.time() - t0)
                    print(f"    {os.path.basename(src)}: {done / 1e9:.2f}/{size / 1e9:.2f} GB, {rate:.0f} MB/s, "
                          f"còn khoảng {(size - done) / 1e6 / max(rate, 1e-3) / 60:.1f} phút")
                    t_print = time.time()
        os.replace(part, dst)
        return (done - start) / 1e6 / max(time.time() - t0, 1e-3)

    for src, size in jobs:
        dst = os.path.join(local, os.path.relpath(src, MODEL_PATH))
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if size > 100e6:
            print(f"  chép {os.path.relpath(src, MODEL_PATH)} ({size / 1e9:.2f} GB)...")
        for attempt in range(1, 6):
            try:
                rate = copy_resumable(src, dst, size)
                break
            except OSError as e:
                # Errno 107 (Transport endpoint is not connected): FUSE của Drive bị rớt khi đọc file lớn.
                # Mount lại rồi chép tiếp từ chỗ dừng.
                if e.errno not in (errno.ENOTCONN, errno.EIO) or attempt == 5:
                    raise
                print(f"  ⚠️ {e}; mount lại Drive và chép tiếp (lần {attempt + 1}/5)")
                time.sleep(5)
                drive.mount("/content/drive", force_remount=True)
        if size > 100e6:
            print(f"  xong {os.path.relpath(src, MODEL_PATH)}, {rate:.0f} MB/s")
    print(f"Checkpoint local: {local}")
    MODEL_PATH = local

SERVER_LOG = "/content/server.log"
server_proc = subprocess.Popen(
    ["/content/Isaac-GR00T/.venv/bin/python", "/content/colab_gr00t_server.py", "--port", "5550",
     "--policy", POLICY, "--model-path", MODEL_PATH, "--embodiment-tag", EMBODIMENT_TAG],
    cwd="/content/Isaac-GR00T", stdout=open(SERVER_LOG, "w"), stderr=subprocess.STDOUT,
    env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
print(f"Model: {MODEL_PATH} | embodiment {EMBODIMENT_TAG} | policy {POLICY}")
print(f"Đã khởi động server (PID {server_proc.pid}). Log: {SERVER_LOG}. Đang chờ server sẵn sàng...")

t0, last = time.time(), ""
while True:
    log = open(SERVER_LOG, errors="replace").read()
    if "ĐANG LẮNG NGHE" in log:
        warmup = re.search(r"Warm-up xong trong [\d.]+s", log)
        print(f"✅ Server sẵn sàng sau {time.time() - t0:.0f} s" + (f" ({warmup.group(0)})." if warmup else "."))
        print("Mở tunnel ở Bước 6 (nếu tunnel cũ còn hạn thì dùng tiếp host/port cũ).")
        break
    if server_proc.poll() is not None:
        print(log[-4000:])
        raise RuntimeError(f"Server đã thoát (mã {server_proc.returncode}); xem log ở trên.")
    tail = re.split(r"[\r\n]+", log.strip())[-1][:160] if log.strip() else ""
    if tail and tail != last:
        print(f"[{time.time() - t0:4.0f} s] {tail}")
        last = tail
    time.sleep(15)
"""),
        code(r"""
# [BƯỚC 6] Mở đường hầm TCP Pinggy qua cổng 443 (không cần tài khoản).
# Tunnel miễn phí hết hạn sau khoảng 60 phút: chạy lại cell này để lấy host/port mới, server không cần khởi động lại.
import socket, subprocess, time

if "tunnel_proc" in globals() and tunnel_proc.poll() is None:
    tunnel_proc.terminate()
    print("Đã đóng tunnel cũ.")
with socket.socket() as s:
    s.settimeout(1)
    if s.connect_ex(("127.0.0.1", 5550)) != 0:
        print("⚠️ Chưa có server ở port 5550 (Bước 5 chưa chạy xong). Tunnel vẫn mở được, nhưng local sẽ chưa kết nối được.")

cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30", "-p", "443", "-R0:localhost:5550", "tcp@free.pinggy.io"]
tunnel_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

print("Đang mở đường hầm TCP qua Pinggy cổng 443...")
time.sleep(3)
pub_url = None
for _ in range(30):
    line = tunnel_proc.stdout.readline()
    if not line:
        time.sleep(0.2)
        continue
    if "tcp://" in line:
        pub_url = line.strip().split("tcp://")[1].split()[0].replace("/", "")
        break

if pub_url and ":" in pub_url:
    host, port = pub_url.split(":")
    print("=" * 75)
    print(f"🎉 Đường hầm TCP: tcp://{pub_url} (mở lúc {time.strftime('%H:%M')}, hết hạn khoảng {time.strftime('%H:%M', time.localtime(time.time() + 3600))})")
    print("=" * 75)
    print("👉 ĐÁNH GIÁ CLOSED-LOOP (terminal máy local):")
    print(f".venv/bin/python tools/eval_closed_loop.py --host {host} --port {port} --dataset datasets/g1_can_sort_bal_test --out eval/<tên> --seconds 20 --stop-on-success")
    print("👉 XEM TRỰC QUAN TRONG GUI:")
    print(f".venv/bin/python launch_vla_simulation.py --policy-mode remote --server-host {host} --server-port {port} --server-timeout-ms 10000")
    print("   Với checkpoint fine-tune từ dataset collect_demos.py (ghi ở 25 Hz), thêm: --vla-action-hz 25")
    print("=" * 75)
else:
    print("Chưa đọc được địa chỉ tunnel; tunnel vẫn chạy nền. Chạy lại cell này nếu cần.")
"""),
        code(r"""
# [BƯỚC 7] Trạng thái server: chạy lại bất cứ lúc nào để xem số request, thời gian suy luận và cuối log.
import os, re, socket
log = open("/content/server.log", errors="replace").read() if os.path.exists("/content/server.log") else ""
with socket.socket() as s:
    s.settimeout(1)
    listening = s.connect_ex(("127.0.0.1", 5550)) == 0
print("Server:", "đang lắng nghe ở port 5550" if listening else "KHÔNG chạy")
loaded = re.search(r"Đang nạp GR00T từ (.*?) \(embodiment (\S+)\)", log)
if loaded:
    print(f"Model: {loaded.group(1)} | embodiment {loaded.group(2)}")
reqs = re.findall(r"^\[(\d+)\] .*suy luận (\d+) ms", log, re.M)
if reqs:
    ms = [int(m) for _, m in reqs[-50:]]
    print(f"Đã phục vụ {reqs[-1][0]} request; suy luận TB {sum(ms) / len(ms):.0f} ms (50 request gần nhất).")
print(f"Lỗi (Traceback) trong log: {log.count('Traceback')}")
print("--- cuối log ---")
print("\n".join(re.split(r"[\r\n]+", log.strip())[-15:]))
"""),
        code(r"""
# [BƯỚC 8] Dừng server (ví dụ để đổi checkpoint ở Bước 5). Tunnel giữ nguyên.
# Phải đổi STOP_SERVER = True mới dừng, để "Chạy tất cả" không tắt server vừa khởi động ở Bước 5.
STOP_SERVER = False

import subprocess, time
if not STOP_SERVER:
    print("Không dừng server. Muốn dừng thì đặt STOP_SERVER = True rồi chạy lại cell này.")
else:
    subprocess.run(["pkill", "-f", "colab_gr00t_server.py"])
    time.sleep(3)
    print("Đã dừng server." if subprocess.run(["pgrep", "-f", "colab_gr00t_server.py"], capture_output=True).returncode
          else "⚠️ Server vẫn còn chạy; chạy lại cell này.")
"""),
    ]


def finetune_notebook():
    return [
        md(f"""
# 🏋️ Fine-tune GR00T N1.7 cho nhiệm vụ phân loại lon trên Google Colab

Fine-tune `nvidia/GR00T-N1.7-3B` trên dataset demo sim do `collect_demos.py` tạo ra (định dạng GR00T LeRobot v2, layout state/action giống REAL_G1, 25 Hz). Checkpoint đầu ra chạy bằng `colab_gr00t_server.ipynb` với `--embodiment-tag new_embodiment`.

### Yêu cầu
* **GPU A100.** Fine-tune mặc định (projector + action head, không tune backbone) cần khoảng 35 GB VRAM. Nếu hết bộ nhớ, giảm `GLOBAL_BATCH_SIZE` ở Bước 4.
* **HuggingFace:** quyền truy cập repo gated `nvidia/Cosmos-Reason2-2B`, token trong **Colab Secrets** tên `HF_TOKEN`.
* **Google Drive:** file dataset nén `.zip` trong `MyDrive/gr00t/` (mặc định `g1_can_sort_bal.zip`, đổi bằng `DATASET_NAME` ở Bước 4). Bước 8a chép mọi checkpoint từ bước `SYNC_FROM_STEP` về `MyDrive/gr00t/checkpoints/<DATASET_NAME>/`, mỗi checkpoint khoảng 10–15 GB: cần đủ chỗ cho tất cả (ví dụ 5000 → 10700 với `SAVE_STEPS` 1000 là 7 checkpoint).

### Tạo file zip ở máy local
```bash
.venv/bin/python -c "import shutil; shutil.make_archive('datasets/g1_can_sort_ik', 'zip', 'datasets', 'g1_can_sort_ik')"
```
rồi tải `datasets/g1_can_sort_ik.zip` lên thư mục `gr00t` trong Google Drive.
Để đánh giá tổng quát hoá (Bước 11), tải thêm tập test `datasets/g1_can_sort_ik_test.zip` (tạo bằng `collect_demos.py --out datasets/g1_can_sort_ik_test --episodes-per-task 20 --seed 100000`, tên đặt ở `TEST_DATASET_NAME`) vào cùng thư mục.

### Lưu ý
* Checkpoint trong lúc train lưu ở ổ đĩa tạm `/content/outputs`, và `finetune.sh` chỉ giữ 5 checkpoint gần nhất (`--save_total_limit 5`). Bước 8a chép từng checkpoint về Drive ngay khi ghi xong, nên không mất khi runtime bị ngắt hoặc khi checkpoint cũ bị xoá. Bước 11 đánh giá mọi checkpoint và chỉ ra checkpoint tốt nhất.
* Train tiếp từ một checkpoint (ví dụ runtime bị ngắt giữa chừng): đặt `INIT_FROM` ở Bước 4 và `MAX_STEPS` = số bước còn thiếu. Chỉ nạp trọng số (không phải resume), checkpoint mới lưu vào thư mục riêng `<DATASET_NAME>_from<N>`.
* Isaac-GR00T được ghim ở commit `{ISAAC_COMMIT[:7]}`.
"""),
        GPU_CELL,
        INSTALL_CELL,
        HF_TOKEN_CELL,
        code("""
# [BƯỚC 4] Cấu hình. Các biến môi trường ở đây được các cell %%bash phía sau dùng lại.
import os
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive/gr00t"
# Tên dataset = tên file zip trên Drive (không có .zip) = tên thư mục bên trong zip.
# Checkpoint lưu theo tên này: MyDrive/gr00t/checkpoints/<DATASET_NAME>/checkpoint-*
DATASET_NAME = "g1_can_sort_bal"            # checkpoint C; A: "g1_can_sort_ik", B: "g1_can_sort_swap_hold"
TEST_DATASET_NAME = "g1_can_sort_bal_test"  # tập test cho Bước 11
# Trọng số khởi tạo: "" = base nvidia/GR00T-N1.7-3B. Đặt đường dẫn một checkpoint để train tiếp từ trọng số của nó, ví dụ
#   INIT_FROM = f"{DRIVE_DIR}/checkpoints/g1_can_sort_bal/checkpoint-4000"
# Đây KHÔNG phải resume: checkpoint lưu --save-only-model nên không có trạng thái optimizer/scheduler. Số bước đếm lại từ 0,
# LR warmup lại lên 1e-4 rồi chạy cosine trong MAX_STEPS bước mới, nên đặt MAX_STEPS = số bước còn thiếu. Checkpoint mới
# lưu vào thư mục riêng <DATASET_NAME>_from<N> (không trùng tên checkpoint của lượt trước), chép và đánh giá từ bước 0.
# Nạp checkpoint từ Drive chậm (~10 phút ở "Loading checkpoint shards").
INIT_FROM = ""
os.environ.update({
    "DATASET_ZIP": f"{DRIVE_DIR}/{DATASET_NAME}.zip",
    "DATASET_DIR": f"/content/datasets/{DATASET_NAME}",
    "TEST_ZIP": f"{DRIVE_DIR}/{TEST_DATASET_NAME}.zip",
    "TEST_DIR": f"/content/datasets/{TEST_DATASET_NAME}",
    "MODALITY_CONFIG": "/content/g1_can_sort_config.py",
    "OUTPUT_DIR": "/content/outputs",
    "EXPERIMENT": DATASET_NAME,
    "DRIVE_CKPT_DIR": f"{DRIVE_DIR}/checkpoints",
    # Tham số train (đọc bởi examples/finetune.sh)
    "NUM_GPUS": "1",
    # C: batch 80 và ~8.6 lần đi qua 99 343 khung như B (80 × 10 000 trên 93 200 khung); GPU 40 GB thì batch 32
    "MAX_STEPS": "10700",
    "SAVE_STEPS": "1000",
    "GLOBAL_BATCH_SIZE": "80",
    # Bước 8a chép checkpoint từ bước này về Drive; Bước 11 đánh giá checkpoint từ bước này
    "SYNC_FROM_STEP": "5000",
    "EVAL_FROM_STEP": "5000",
    "DATALOADER_NUM_WORKERS": "4",
    "USE_WANDB": "0",
    "BASE_MODEL_PATH": "nvidia/GR00T-N1.7-3B",
})
assert os.path.exists(os.environ["DATASET_ZIP"]), f"Không thấy {os.environ['DATASET_ZIP']} trên Drive."
if INIT_FROM:
    import glob, re
    INIT_FROM = INIT_FROM.rstrip("/")
    assert os.path.exists(f"{INIT_FROM}/config.json"), f"Không thấy checkpoint {INIT_FROM} (thiếu config.json)."
    m = re.search(r"checkpoint-(\\d+)$", INIT_FROM)
    os.environ.update({
        "BASE_MODEL_PATH": INIT_FROM,
        "EXPERIMENT": f"{DATASET_NAME}_from{m.group(1) if m else os.path.basename(INIT_FROM)}",
        "SYNC_FROM_STEP": "0",
        "EVAL_FROM_STEP": "0",
    })
    existing = glob.glob(f"{os.environ['DRIVE_CKPT_DIR']}/{os.environ['EXPERIMENT']}/checkpoint-*")
    if existing:
        print(f"⚠️ Drive đã có {len(existing)} checkpoint trong {os.environ['EXPERIMENT']}: Bước 8a sẽ bỏ qua checkpoint "
              "trùng tên. Xoá hoặc đổi tên thư mục đó nếu là của lượt khác.")
print(f"Khởi tạo từ: {os.environ['BASE_MODEL_PATH']}")
print(f"Checkpoint lưu vào: {os.environ['OUTPUT_DIR']}/{os.environ['EXPERIMENT']} -> "
      f"{os.environ['DRIVE_CKPT_DIR']}/{os.environ['EXPERIMENT']}")
print(f"Batch {os.environ['GLOBAL_BATCH_SIZE']} × {os.environ['MAX_STEPS']} bước, chép/đánh giá từ bước "
      f"{os.environ['SYNC_FROM_STEP']}/{os.environ['EVAL_FROM_STEP']}")
print("✅ Đã mount Drive và đặt cấu hình.")
"""),
        code("""
%%bash
# [BƯỚC 5] Giải nén dataset từ Drive
set -euo pipefail
mkdir -p "$(dirname "$DATASET_DIR")"
[ -d "$DATASET_DIR" ] || unzip -q "$DATASET_ZIP" -d "$(dirname "$DATASET_DIR")"
ls "$DATASET_DIR" "$DATASET_DIR/meta"
"""),
        writefile("/content/g1_can_sort_config.py", ROOT / "finetune" / "g1_can_sort_config.py"),
        code("""
%%bash
# [BƯỚC 7] Tính thống kê chuẩn hoá (meta/stats.json, meta/relative_stats.json) và nạp thử một episode
# bằng data loader của Isaac-GR00T, để phát hiện lỗi dataset trước khi nạp model 3B.
set -euo pipefail
cd /content/Isaac-GR00T
source .venv/bin/activate
# Như gr00t/data/stats.py, nhưng thống kê action relative tính song song theo episode: bản gốc lặp tuần tự từng khung
# bằng Python, riêng 2 khoá EEF (tạo pose, đổi rot6d) mất ~0.9 s/episode, tức ~7 phút với 400 episode.
# Cùng hàm load_relative_actions, cùng thứ tự episode nên kết quả như bản gốc. Chạy lại trên cùng runtime thì bỏ qua
# (stats.py tự dùng lại meta/*.json nếu fingerprint khớp).
python - "$DATASET_DIR" "$MODALITY_CONFIG" <<'EOF'
import os, sys, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # mỗi tiến trình con một luồng BLAS
from multiprocessing import Pool
import numpy as np
import gr00t.data.stats as S
from gr00t.data.types import EmbodimentTag

_loader = None


def _init(dataset_path, embodiment_tag, group_key):
    global _loader
    _loader = S.RelativeActionLoader(dataset_path, embodiment_tag, group_key)


def _load(episode_id):
    return _loader.load_relative_actions(episode_id)


def calculate_stats_for_key(dataset_path, embodiment_tag, group_key, max_episodes=-1):
    n = len(S.RelativeActionLoader(dataset_path, embodiment_tag, group_key))
    n = n if max_episodes == -1 else min(n, max_episodes)
    t0 = time.time()
    with Pool(os.cpu_count(), initializer=_init, initargs=(dataset_path, embodiment_tag, group_key)) as pool:
        trajectories = [t for ep in pool.imap(_load, range(n), chunksize=4) for t in ep]
    print(f"  {group_key}: {n} episode, {time.time() - t0:.0f} s ({os.cpu_count()} tiến trình)", flush=True)
    return {
        "max": np.max(trajectories, axis=0),
        "min": np.min(trajectories, axis=0),
        "q01": np.quantile(trajectories, 0.01, axis=0),
        "q99": np.quantile(trajectories, 0.99, axis=0),
        "mean": np.mean(trajectories, axis=0),
        "std": np.std(trajectories, axis=0),
    }


S.calculate_stats_for_key = calculate_stats_for_key
t0 = time.time()
S.main(sys.argv[1], EmbodimentTag.NEW_EMBODIMENT, sys.argv[2])
print(f"Thống kê xong trong {time.time() - t0:.0f} s")
EOF
python - <<'EOF'
import os, sys
sys.path.append(os.path.dirname(os.environ["MODALITY_CONFIG"]))
import g1_can_sort_config  # noqa: F401  (đăng ký modality config cho NEW_EMBODIMENT)
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

loader = LeRobotEpisodeLoader(os.environ["DATASET_DIR"], MODALITY_CONFIGS["new_embodiment"])
episode = loader[0]
print(f"{len(loader)} episode; episode 0: {episode.shape[0]} bước, các cột: {list(episode.columns)[:12]}")
print("✅ Dataset nạp được bằng data loader của Isaac-GR00T.")
EOF
"""),
        code("""
# [BƯỚC 8] Fine-tune chạy nền, log ghi ra $OUTPUT_DIR/train.log. Cell này kết thúc ngay sau khi khởi động;
# xem tiến độ và ETA bằng Bước 8b. Trình duyệt mất kết nối không làm dừng train (miễn runtime còn sống).
# Chỉ lưu trọng số model (--save-only-model) để checkpoint gọn; không tiếp tục train được từ checkpoint này.
import os, subprocess
os.makedirs(os.environ["OUTPUT_DIR"], exist_ok=True)
TRAIN_LOG = f"{os.environ['OUTPUT_DIR']}/train.log"
assert "train_proc" not in globals() or train_proc.poll() is not None, "Train đang chạy; xem Bước 8b."
train_cmd = f\"\"\"
cd /content/Isaac-GR00T && source .venv/bin/activate && bash examples/finetune.sh \\\\
    --base-model-path "$BASE_MODEL_PATH" \\\\
    --dataset-path "$DATASET_DIR" \\\\
    --embodiment-tag NEW_EMBODIMENT \\\\
    --modality-config-path "$MODALITY_CONFIG" \\\\
    --output-dir "$OUTPUT_DIR" \\\\
    --experiment-name "$EXPERIMENT" \\\\
    --save-only-model > "{TRAIN_LOG}" 2>&1
\"\"\"
train_proc = subprocess.Popen(["bash", "-c", train_cmd], env={**os.environ, "PYTHONUNBUFFERED": "1"},
                              start_new_session=True)
print(f"Đã khởi động train (PID {train_proc.pid}). Log: {TRAIN_LOG}. Chạy ngay Bước 8a (chép checkpoint về Drive), "
      "rồi Bước 8b để xem tiến độ.")
"""),
        code("""
# [BƯỚC 8a] Chép từng checkpoint về Drive ngay khi ghi xong (luồng chạy nền, kiểm tra mỗi phút). Chạy ngay sau Bước 8,
# trước Bước 8c (Bước 8c chờ tới khi train xong).
# finetune.sh chỉ giữ 5 checkpoint gần nhất và ổ /content mất khi runtime bị ngắt, nên không chép thì checkpoint giữa
# chừng (để chọn checkpoint tốt nhất ở Bước 11) sẽ mất. Chép vào thư mục tạm "_partial_checkpoint-N" rồi đổi tên, nên
# thư mục "checkpoint-N" trên Drive luôn đầy đủ. Luồng tự dừng sau lần chép cuối khi train xong; runtime khởi động lại
# thì chạy lại cell này để chép nốt.
import glob, os, re, shutil, threading, time

SYNC_FROM_STEP = int(os.environ["SYNC_FROM_STEP"])
SRC = f"{os.environ['OUTPUT_DIR']}/{os.environ['EXPERIMENT']}"
DST = f"{os.environ['DRIVE_CKPT_DIR']}/{os.environ['EXPERIMENT']}"
step_of = lambda p: int(re.search(r"checkpoint-(\\d+)$", p).group(1))

def dir_size(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)

def ready(path):
    # Trainer ghi trainer_state.json sau cùng; đợi thêm 60 s không có file nào thay đổi
    files = [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs]
    return (os.path.exists(f"{path}/trainer_state.json")
            and time.time() - max(os.path.getmtime(f) for f in files) > 60)

def sync_once(log):
    os.makedirs(DST, exist_ok=True)
    for src in sorted(glob.glob(f"{SRC}/checkpoint-*"), key=step_of):
        step = step_of(src)
        dest = f"{DST}/checkpoint-{step}"
        if step < SYNC_FROM_STEP or os.path.isdir(dest) or not ready(src):
            continue
        size = dir_size(src)
        free = shutil.disk_usage(DST).free
        if free < size * 1.1:
            log(f"⚠️ Drive còn {free / 1e9:.1f} GB, không đủ chép checkpoint-{step} ({size / 1e9:.1f} GB)")
            continue
        tmp = f"{DST}/_partial_checkpoint-{step}"
        shutil.rmtree(tmp, ignore_errors=True)
        t0 = time.time()
        shutil.copytree(src, tmp)
        os.rename(tmp, dest)
        log(f"✅ checkpoint-{step} -> Drive ({size / 1e9:.1f} GB, {time.time() - t0:.0f} s)")

sync_log = []
def sync_loop():
    log = lambda m: sync_log.append(f"[{time.strftime('%H:%M:%S')}] {m}")
    while True:
        training = "train_proc" in globals() and train_proc.poll() is None
        if not training:
            time.sleep(65)  # checkpoint cuối vừa ghi: đợi qua ngưỡng 60 s của ready() rồi mới chép lượt cuối
        try:
            sync_once(log)
        except Exception as e:  # Drive tạm lỗi: thử lại ở vòng sau
            log(f"❌ {type(e).__name__}: {e}")
            if not training:
                continue
        if not training:
            log("Train đã dừng, chép xong lượt cuối.")
            return
        time.sleep(60)

if "sync_thread" in globals() and sync_thread.is_alive():
    print("Luồng chép đang chạy.")
else:
    sync_thread = threading.Thread(target=sync_loop, daemon=True)
    sync_thread.start()
    print(f"Đã bật luồng chép checkpoint từ bước {SYNC_FROM_STEP}: {SRC} -> {DST}")
print("Nhật ký (chạy lại cell này để xem):", *sync_log[-10:], sep="\\n  ")
print("Trên Drive:", sorted((os.path.basename(p) for p in glob.glob(f"{DST}/checkpoint-*")), key=lambda n: int(n.split("-")[1])))
"""),
        code("""
# [BƯỚC 8b] Theo dõi train: chạy lại cell này bất cứ lúc nào để xem bước hiện tại, ETA, loss và VRAM.
import os, re, subprocess
log = open(f"{os.environ['OUTPUT_DIR']}/train.log", errors="replace").read()
# Thanh tiến độ của HuggingFace Trainer dạng "150/5000 [05:10<2:47:30,  2.07s/it]"
bars = re.findall(r"(\\d+)/(\\d+) \\[([\\d:]+)<([\\d:?]+),\\s*([\\d.]+)(s/it|it/s)", log)
bars = [b for b in bars if b[1] == os.environ["MAX_STEPS"]]
if bars:
    done, total, elapsed, eta, rate, unit = bars[-1]
    print(f"Bước {done}/{total} ({int(done) / int(total):.0%}) | đã chạy {elapsed} | còn lại ~{eta} | {rate} {unit}")
else:
    print("Chưa vào vòng train (đang tải model / dựng dữ liệu). 20 dòng log cuối:")
    print("\\n".join(log.replace("\\r", "\\n").splitlines()[-20:]))
losses = re.findall(r"'loss': ([\\d.e-]+)", log)
if losses:
    print(f"Loss gần nhất: {losses[-1]} (đầu tiên: {losses[0]}, {len(losses)} lần ghi, mỗi 10 bước)")
if "train_proc" not in globals():
    status = None
    print("Trạng thái: không rõ (kernel đã khởi động lại nên mất tiến trình train; xem log ở trên)")
else:
    status = train_proc.poll()
    print("Trạng thái:", "đang chạy" if status is None else f"đã kết thúc, mã thoát {status}")
if status not in (None, 0):
    print("Train lỗi. 30 dòng log cuối:")
    print("\\n".join(log.replace("\\r", "\\n").splitlines()[-30:]))
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv"],
                     capture_output=True, text=True).stdout)
"""),
        code("""
# [BƯỚC 8c] Chờ tới khi train xong: in tiến độ mỗi phút, xong thì phát tiếng bíp.
# Dừng cell này (■) lúc nào cũng được, train vẫn chạy tiếp.
import os, re, time
import numpy as np
from IPython.display import Audio, clear_output, display

while train_proc.poll() is None:
    log = open(f"{os.environ['OUTPUT_DIR']}/train.log", errors="replace").read()
    bars = [b for b in re.findall(r"(\\d+)/(\\d+) \\[([\\d:]+)<([\\d:?]+)", log) if b[1] == os.environ["MAX_STEPS"]]
    losses = re.findall(r"'loss': ([\\d.e-]+)", log)
    clear_output(wait=True)
    print(time.strftime("%H:%M:%S"), "|",
          f"bước {bars[-1][0]}/{bars[-1][1]}, còn lại ~{bars[-1][3]}" if bars else "đang khởi động",
          f"| loss {losses[-1]}" if losses else "")
    time.sleep(60)

code = train_proc.returncode
print("✅ Train xong." if code == 0 else f"❌ Train dừng với mã thoát {code}: chạy Bước 8b để xem log lỗi.")
t = np.linspace(0, 1.5, 24000)
display(Audio(np.sin(2 * np.pi * 880 * t) * (t % 0.5 < 0.25), rate=16000, autoplay=True))
"""),
        code("""
# [BƯỚC 8d] Biểu đồ loss, grad norm và learning rate theo bước train. Chạy được trong lúc train (đọc train.log)
# hoặc sau khi runtime khởi động lại (đọc trainer_state.json trong checkpoint trên ổ tạm hoặc Drive).
# Ảnh lưu ở $OUTPUT_DIR/loss_curve.png và cạnh checkpoint trên Drive.
import ast, glob, json, os, re, shutil
import numpy as np
import matplotlib.pyplot as plt

def from_train_log(path):
    # HF Trainer in mỗi lần ghi log dạng {'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...};
    # bước của lần ghi lấy từ thanh tiến độ "N/MAX_STEPS [" gần nhất phía trước nó
    if not os.path.exists(path):
        return []
    log = open(path, errors="replace").read()
    bars = [(m.start(), int(m.group(1))) for m in re.finditer(r"(\\d+)/" + os.environ["MAX_STEPS"] + r" \\[", log)]
    rows = []
    for m in re.finditer(r"\\{[^{}]*'loss'[^{}]*\\}", log):
        try:
            d = {k: float(v) for k, v in ast.literal_eval(m.group(0)).items()}
        except (ValueError, SyntaxError, TypeError):
            continue
        before = [s for pos, s in bars if pos < m.start()]
        d["step"] = before[-1] if before else 10 * (len(rows) + 1)
        rows.append(d)
    return rows

def from_trainer_state():
    dirs = [f"{os.environ['OUTPUT_DIR']}/{os.environ['EXPERIMENT']}", f"{os.environ['DRIVE_CKPT_DIR']}/{os.environ['EXPERIMENT']}"]
    for d in dirs:
        states = sorted(glob.glob(f"{d}/checkpoint-*/trainer_state.json"),
                        key=lambda p: int(re.search(r"checkpoint-(\\d+)", p).group(1)))
        if states:
            print("Đọc", states[-1])
            return [e for e in json.load(open(states[-1]))["log_history"] if "loss" in e]
    return []

rows = from_train_log(f"{os.environ['OUTPUT_DIR']}/train.log") or from_trainer_state()
assert rows, "Chưa có loss nào: train chưa tới lần ghi log đầu tiên (10 bước), hoặc không thấy train.log / checkpoint."

step = np.array([r["step"] for r in rows], dtype=float)
series = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in ("loss", "grad_norm", "learning_rate")}
loss = series["loss"]
win = max(5, len(loss) // 20)  # trung bình trượt khoảng 5% số lần ghi
smooth = np.array([loss[max(0, i - win + 1):i + 1].mean() for i in range(len(loss))])

# Màu: cùng một hue xanh (nhạt = giá trị gốc, đậm = trung bình trượt); chữ và lưới dùng màu trung tính
BLUE, BLUE_LIGHT, INK, INK_2, GRID = "#2a78d6", "#86b6ef", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                     "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9})
fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
for ax in axes:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fcfcfb")

ax = axes[0]
ax.plot(step, loss, color=BLUE_LIGHT, linewidth=1, label="loss (mỗi lần ghi log)")
ax.plot(step, smooth, color=BLUE, linewidth=2, label=f"trung bình trượt {win} lần ghi")
ax.set_yscale("log")
ax.set_title(f"Loss flow matching – {os.environ['EXPERIMENT']}", loc="left")
ax.set_ylabel("loss (thang log)")
ax.legend(frameon=False, loc="upper right", labelcolor=INK_2)
ax.annotate(f"{smooth[-1]:.4f}", (step[-1], smooth[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", color=INK, fontweight="bold")

for ax, key, title in ((axes[1], "grad_norm", "Grad norm"), (axes[2], "learning_rate", "Learning rate")):
    vals = series[key]
    if np.isfinite(vals).any():
        ax.plot(step, vals, color=BLUE, linewidth=1.5)
    else:
        ax.text(0.5, 0.5, "không có trong log", transform=ax.transAxes, ha="center", color=INK_2)
    ax.set_title(title, loc="left")
axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes[2].set_xlabel("bước train")
fig.tight_layout()

png = f"{os.environ['OUTPUT_DIR']}/loss_curve.png"
fig.savefig(png, dpi=130, facecolor="white")
plt.show()
drive_png = f"{os.environ['DRIVE_CKPT_DIR']}/{os.environ['EXPERIMENT']}/loss_curve.png"
os.makedirs(os.path.dirname(drive_png), exist_ok=True)
shutil.copy(png, drive_png)

i_min = int(np.nanargmin(smooth))
print(f"{len(loss)} lần ghi, bước {int(step[0])}–{int(step[-1])} | loss đầu {loss[0]:.4f} -> cuối (trung bình trượt) "
      f"{smooth[-1]:.4f}, giảm {loss[0] / smooth[-1]:.0f} lần | thấp nhất {smooth[i_min]:.4f} ở bước {int(step[i_min])}")
print("Đã lưu:", png, "và", drive_png)
"""),
        code("""
# [BƯỚC 9] Đánh giá open-loop: so action model dự đoán với action trong demo trên vài episode
# (episode 0 là Coca, episode 1 là Pepsi). Log in MSE/MAE của action; biểu đồ từng episode lưu ở
# /tmp/open_loop_eval/traj_<id>.jpeg (mở bằng thanh Files bên trái).
import glob, os, re
assert "train_proc" not in globals() or train_proc.poll() is not None, "Train chưa xong; xem tiến độ ở Bước 8b."
ckpts = sorted(glob.glob(f"{os.environ['OUTPUT_DIR']}/{os.environ['EXPERIMENT']}/checkpoint-*"),
               key=lambda p: int(re.search(r"checkpoint-(\\d+)", p).group(1)))
assert ckpts, "Chưa có checkpoint: Bước 8 chưa chạy xong."
ckpt, dataset_dir = ckpts[-1], os.environ["DATASET_DIR"]
os.environ["CKPT"] = ckpt  # Bước 10 dùng lại
print("Checkpoint:", ckpt)
# MPLBACKEND=Agg: Colab đặt backend matplotlib_inline, gói này không có trong môi trường Isaac-GR00T
!cd /content/Isaac-GR00T && MPLBACKEND=Agg .venv/bin/python gr00t/eval/open_loop_eval.py --model-path "{ckpt}" --dataset-path "{dataset_dir}" --embodiment-tag NEW_EMBODIMENT --traj-ids 0 1 --steps 400 --execution-horizon 16
"""),
        code("""
# [BƯỚC 10] Chép checkpoint về Google Drive để dùng với colab_gr00t_server.ipynb. Bước 8a đã chép mọi checkpoint từ
# SYNC_FROM_STEP; cell này để chép checkpoint trước mốc đó hoặc khi không bật Bước 8a.
# COPY_STEPS: "latest" (mới nhất), "all" (mọi checkpoint còn trên ổ tạm), hoặc danh sách bước, ví dụ [4000].
import glob, os, re, shutil
COPY_STEPS = "latest"
if "train_proc" in globals() and train_proc.poll() is None:
    print("⚠️ Train vẫn đang chạy: checkpoint mới nhất chưa phải checkpoint cuối.")
step_of = lambda p: int(re.search(r"checkpoint-(\\d+)$", p).group(1))
ckpts = sorted(glob.glob(f"{os.environ['OUTPUT_DIR']}/{os.environ['EXPERIMENT']}/checkpoint-*"), key=step_of)
assert ckpts, f"Không thấy checkpoint trong {os.environ['OUTPUT_DIR']}/{os.environ['EXPERIMENT']}."
chosen = (ckpts[-1:] if COPY_STEPS == "latest" else ckpts if COPY_STEPS == "all"
          else [c for c in ckpts if step_of(c) in COPY_STEPS])
assert chosen, f"Không có checkpoint {COPY_STEPS} trên ổ tạm (còn: {[step_of(c) for c in ckpts]})."
for ckpt in chosen:
    dest = f"{os.environ['DRIVE_CKPT_DIR']}/{os.environ['EXPERIMENT']}/{os.path.basename(ckpt)}"
    if os.path.isdir(dest):
        print(f"Đã có trên Drive: {dest}")
        continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Đang chép {ckpt} -> {dest} ...")
    tmp = f"{os.path.dirname(dest)}/_partial_{os.path.basename(ckpt)}"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(ckpt, tmp)
    os.rename(tmp, dest)
    size_gb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dest) for f in fs) / 1e9
    print(f"✅ Đã chép ({size_gb:.1f} GB).")
print(f'Chạy server (Bước 5 của colab_gr00t_server.ipynb): MODEL_PATH = "{dest}", EMBODIMENT_TAG = "new_embodiment"')
print("Ở máy local thêm --vla-action-hz 25.")
"""),
        code("""
# [BƯỚC 11] Đánh giá open-loop trên tập test độc lập (collect_demos.py --seed 100000, seed không trùng train) với mọi
# checkpoint từ EVAL_FROM_STEP (ổ tạm, hoặc Drive nếu ổ tạm không còn), chọn checkpoint có MAE test thấp nhất, rồi đánh
# giá thêm checkpoint đó trên N_EPISODES episode train để xem khoảng cách train-test. Cần file zip tập test
# (TEST_DATASET_NAME ở Bước 4) trên Drive, cùng thư mục với dataset train. Episode chẵn là Coca, lẻ là Pepsi.
import glob, json, os, re, subprocess
import numpy as np

TEST_ZIP, TEST_DIR = os.environ["TEST_ZIP"], os.environ["TEST_DIR"]
N_EPISODES = 40
assert os.path.exists(TEST_ZIP), f"Không thấy {TEST_ZIP}: tải file zip tập test lên Drive trước."
if not os.path.isdir(TEST_DIR):
    subprocess.run(["unzip", "-q", TEST_ZIP, "-d", os.path.dirname(TEST_DIR)], check=True)

# Mọi checkpoint từ EVAL_FROM_STEP: ưu tiên bản trên ổ tạm (nạp nhanh), không còn thì lấy bản trên Drive (Bước 8a/10;
# nạp từ Drive chậm, vài phút mỗi checkpoint)
step = lambda p: int(re.search(r"checkpoint-(\\d+)$", p).group(1))
found = {}
for root in (os.environ["DRIVE_CKPT_DIR"], os.environ["OUTPUT_DIR"]):  # ổ tạm ghi đè Drive
    for p in glob.glob(f"{root}/{os.environ['EXPERIMENT']}/checkpoint-*"):
        if step(p) >= int(os.environ["EVAL_FROM_STEP"]):
            found[step(p)] = p
assert found, "Không thấy checkpoint từ EVAL_FROM_STEP trên ổ tạm lẫn trên Drive."
ckpts = [found[k] for k in sorted(found)]
isaac_python = "/content/Isaac-GR00T/.venv/bin/python"
env = {**os.environ, "MPLBACKEND": "Agg"}  # Colab đặt backend matplotlib_inline, môi trường Isaac-GR00T không có

# Data loader cần meta/stats.json (runtime mới giải nén lại thì chưa có); lúc đánh giá model chuẩn hoá bằng thống kê
# trong checkpoint nên chỉ cần tạo file
for ds in (os.environ["DATASET_DIR"], TEST_DIR):
    if not os.path.exists(f"{ds}/meta/stats.json"):
        subprocess.run([isaac_python, "-c", f"from gr00t.data.stats import generate_stats; generate_stats('{ds}')"],
                       cwd="/content/Isaac-GR00T", env=env, check=True)

def evaluate(dataset_dir, n_episodes):
    out = subprocess.run(
        [isaac_python, "gr00t/eval/open_loop_eval.py", "--model-path", ckpt, "--dataset-path", dataset_dir,
         "--embodiment-tag", "NEW_EMBODIMENT", "--traj-ids", *map(str, range(n_episodes)), "--steps", "400",
         "--execution-horizon", "16"],
        cwd="/content/Isaac-GR00T", env=env, capture_output=True, text=True)
    rows = re.findall(r"MSE for trajectory (\\d+): ([\\d.e-]+), MAE: ([\\d.e-]+)", out.stdout + out.stderr)
    if not rows:
        print((out.stdout + out.stderr)[-3000:])
        raise RuntimeError(f"Đánh giá {dataset_dir} lỗi, xem log ở trên.")
    return {int(i): (float(mse), float(mae)) for i, mse, mae in rows}

with open(f"{TEST_DIR}/meta/episodes.jsonl") as f:
    n_test = sum(1 for line in f if line.strip())
per_ckpt = {}
for c in ckpts:
    print(f"Đang đánh giá {os.path.basename(c)} trên {n_test} episode test...")
    ckpt = c
    per_ckpt[c] = evaluate(TEST_DIR, n_test)
print(f"\\n{'checkpoint':<18}{'MSE test TB':>13}{'MAE test TB':>13}{'MAE lớn nhất':>14}")
for c, res in per_ckpt.items():
    v = np.array(list(res.values()))
    print(f"{os.path.basename(c):<18}{v[:, 0].mean():>13.6f}{v[:, 1].mean():>13.5f}{v[:, 1].max():>14.5f}")
ckpt = min(per_ckpt, key=lambda c: np.mean([mae for _, mae in per_ckpt[c].values()]))
print(f"\\n⭐ Checkpoint tốt nhất (MAE test TB thấp nhất): {ckpt}")
print(f"   Đánh giá thêm {N_EPISODES} episode train; bảng chi tiết dưới đây là của checkpoint này.")
results = {"train": evaluate(os.environ["DATASET_DIR"], N_EPISODES), "test": per_ckpt[ckpt]}

print(f"\\n{'tập':<6}{'lon':<7}{'số ep':>6}{'MSE TB':>11}{'MAE TB':>10}{'MAE lớn nhất':>14}")
for name, res in results.items():
    for label, parity in (("Coca", 0), ("Pepsi", 1), ("cả hai", None)):
        vals = np.array([v for i, v in res.items() if parity is None or i % 2 == parity])
        print(f"{name:<6}{label:<7}{len(vals):>6}{vals[:, 0].mean():>11.6f}{vals[:, 1].mean():>10.5f}{vals[:, 1].max():>14.5f}")

# Theo lon đứng / nằm (collect_demos.py ghi "lying" vào meta/collection.jsonl; dataset cũ không có thì bỏ qua)
def lying_of(ds):
    with open(f"{ds}/meta/collection.jsonl") as f:
        eps = [json.loads(line) for line in f if line.strip()]
    return {e["episode_index"]: e["lying"] for e in eps if "lying" in e}

lying = {name: lying_of(ds) for name, ds in (("train", os.environ["DATASET_DIR"]), ("test", TEST_DIR))}
if all(lying.values()):
    print(f"\\n{'tập':<6}{'lon':<12}{'số ep':>6}{'MSE TB':>11}{'MAE TB':>10}{'MAE lớn nhất':>14}")
    for name, res in results.items():
        for label, want in (("đứng", False), ("nằm", True)):
            vals = np.array([v for i, v in res.items() if lying[name].get(i) == want])
            if len(vals):
                print(f"{name:<6}{label:<12}{len(vals):>6}{vals[:, 0].mean():>11.6f}{vals[:, 1].mean():>10.5f}{vals[:, 1].max():>14.5f}")

# Theo lon cùng phía / đổi phía với khay (collect_demos.py --swap-prob ghi "swapped"; dataset cũ không có thì bỏ qua)
def swapped_of(ds):
    with open(f"{ds}/meta/collection.jsonl") as f:
        eps = [json.loads(line) for line in f if line.strip()]
    return {e["episode_index"]: e["swapped"] for e in eps if "swapped" in e}

swapped = {name: swapped_of(ds) for name, ds in (("train", os.environ["DATASET_DIR"]), ("test", TEST_DIR))}
if all(swapped.values()) and any(any(v.values()) for v in swapped.values()):
    print(f"\\n{'tập':<6}{'lon':<12}{'số ep':>6}{'MSE TB':>11}{'MAE TB':>10}{'MAE lớn nhất':>14}")
    for name, res in results.items():
        for label, want in (("cùng phía", False), ("đổi phía", True)):
            vals = np.array([v for i, v in res.items() if swapped[name].get(i) == want])
            if len(vals):
                print(f"{name:<6}{label:<12}{len(vals):>6}{vals[:, 0].mean():>11.6f}{vals[:, 1].mean():>10.5f}{vals[:, 1].max():>14.5f}")

worst = sorted(results["test"].items(), key=lambda kv: -kv[1][1])[:5]
kind = lambda i: ("" if i not in lying["test"] else (", nằm" if lying["test"][i] else ", đứng")) + \\
    (", đổi phía" if swapped["test"].get(i) else "")
print("\\nEpisode test có MAE lớn nhất:", ", ".join(f"{i} ({'Coca' if i % 2 == 0 else 'Pepsi'}{kind(i)}, {mae:.4f})" for i, (_, mae) in worst))
"""),
    ]


def write_notebook(path: Path, cells, gpu_type: str):
    metadata = {
        "accelerator": "GPU",
        "colab": {"gpuType": gpu_type, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 0}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(f"Đã ghi {path.relative_to(ROOT)} ({len(cells)} cell)")


if __name__ == "__main__":
    write_notebook(ROOT / "colab_gr00t_server.ipynb", server_notebook(), "L4")
    write_notebook(ROOT / "colab_gr00t_finetune.ipynb", finetune_notebook(), "A100")
