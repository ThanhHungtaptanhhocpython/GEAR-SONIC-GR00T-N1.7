"""Sinh các slide theo bộ khung mới (thanh gradient trái, dải ruy băng góc phải, footer mảnh + ô số trang)."""
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "deck/slides"
LOGO_H = "/_blob/baafa340f95ff6609ebd3c315364b382"
LOGO_BIG = "/_blob/f1ac4e10cfa6c9c1fb5c0e9c0bfcf6b5"
FONT = "font-family:'Be Vietnam Pro', Arial, sans-serif"
INK, BODY, MUTED, GREEN, DEEP = "#123F2B", "#3F4F47", "#6B7A72", "#1FA35C", "#0E6B3F"

RIBBON = """<svg aria-hidden="true" width="1320" height="200" viewBox="0 0 1320 200" style="position:absolute; left:600px; top:0px; width:1320px; height:200px">
<defs><linearGradient id="rb" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#E6F5EC" stop-opacity="0"/><stop offset="1" stop-color="#BFE6CF"/></linearGradient>
<linearGradient id="rc" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#EAF7EF"/><stop offset="1" stop-color="#C4E9D3"/></linearGradient></defs>
<polygon points="0,116 {w},116 {w2},190 0,190" fill="url(#rb)"/>
<polygon points="1000,190 1160,0 1320,0 1320,190" fill="url(#rc)"/>
<polygon points="1150,190 1310,0 1320,0 1320,190" fill="#B3E2C6" opacity="0.8"/>
<line x1="1080" y1="190" x2="1240" y2="0" stroke="#FFFFFF" stroke-width="3"/>
</svg>"""

CORNER_BR = """<svg aria-hidden="true" width="440" height="300" viewBox="0 0 440 300" style="position:absolute; left:1480px; top:780px; width:440px; height:300px">
<defs><linearGradient id="cb" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#E4F5EA"/><stop offset="1" stop-color="#9ED9B6"/></linearGradient>
<linearGradient id="cd" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#2DB36A"/><stop offset="1" stop-color="#0E6B3F"/></linearGradient></defs>
<polygon points="440,20 440,300 120,300" fill="url(#cb)"/>
<polygon points="440,140 440,300 270,300" fill="url(#cd)"/>
<line x1="90" y1="300" x2="440" y2="0" stroke="#FFFFFF" stroke-width="4"/>
</svg>"""


def ribbon(w=1000):
    return RIBBON.format(w=w, w2=w - 64)


def frame(page, *, bar=True, ribbon_w=1000, corner=False, footer_w=1620):
    parts = []
    if bar:
        parts.append('<div style="position:absolute; left:0px; top:0px; width:30px; height:1080px; '
                     'background:linear-gradient(180deg, #2DB36A 0%, #0E6B3F 45%, #0A3F28 100%)"></div>')
    if ribbon_w:
        parts.append(ribbon(ribbon_w))
    if corner:
        parts.append(CORNER_BR)
    parts.append(f'<img src="{LOGO_H}" alt="VinDynamics" style="position:absolute; left:100px; top:52px; width:460px; height:97px; object-fit:contain">')
    parts.append(f'<div style="position:absolute; left:112px; top:990px; width:{footer_w}px; height:2px; '
                 'background:linear-gradient(90deg, #8FCDA9 0%, #D9EEE2 100%)"></div>')
    parts.append(f'<p style="position:absolute; left:112px; top:1012px; width:600px; font-size:24px; color:{MUTED}">VinDynamics  |  Confidential</p>')
    parts.append(f'<p style="position:absolute; left:1744px; top:970px; width:120px; font-size:28px; font-weight:700; '
                 f'line-height:1.4; text-align:center; color:{INK}; background:#E6F4EC; padding:10px 0px; border-radius:16px">{page:02d}</p>')
    return "\n".join(parts)


def header(title, sub, size=72):
    return f"""<div style="display:flex; flex-direction:column; gap:14px">
<h2 style="font-size:{size}px; font-weight:800; line-height:1.12; letter-spacing:-1px; color:{INK}; text-transform:uppercase">{title}</h2>
<p style="font-size:34px; line-height:1.35; color:{MUTED}">{sub}</p>
<div style="width:140px; height:8px; border-radius:4px; background:linear-gradient(90deg, #1FA35C 0%, #BFE6CF 100%)"></div>
</div>"""


def card_head(icon, label):
    return f"""<div style="display:flex; flex-direction:row; gap:24px; align-items:center">
<div style="width:80px; height:80px; border-radius:50%; background:#D5EFE0; display:flex; align-items:center; justify-content:center">
<x-icon name="{icon}" style="color:{DEEP}; width:48px; height:48px"></x-icon>
</div>
<h3 style="font-size:34px; font-weight:800; line-height:1.2; color:{INK}; text-transform:uppercase">{label}</h3>
</div>
<hr style="border-top:2px solid #D6E8DD">"""


def bullet(title, desc):
    return f"""<div style="display:flex; flex-direction:row; gap:22px; align-items:start">
<div style="width:16px; height:16px; border-radius:50%; background:{GREEN}"></div>
<div style="flex:1; display:flex; flex-direction:column; gap:4px">
<p style="font-size:30px; font-weight:700; line-height:1.3; color:{INK}">{title}</p>
<p style="font-size:25px; line-height:1.4; color:{BODY}">{desc}</p>
</div>
</div>"""


def section(sid, style, body, notes):
    return (f'<section id="{sid}" data-transition="fade" style="background:#FFFFFF; color:{BODY}; {FONT}; {style}">\n'
            f"{body}\n<aside>{notes}</aside>\n</section>\n")


CARD_BG = "background:linear-gradient(180deg, #EEF8F2 0%, #F7FBF9 100%); border-radius:24px"

# ---------------------------------------------------------------- 03 GR00T
gr00t = section("gr00t", "padding:210px 128px 116px 120px; display:flex; flex-direction:column; gap:40px", f"""
{frame(3)}
{header("Kiến trúc GR00T N1.7", "System 2 hiểu cảnh và câu lệnh, System 1 sinh chuỗi hành động")}
<div style="display:flex; flex-direction:row; gap:36px; flex:1">
<div style="width:760px; display:flex; flex-direction:column; gap:24px; {CARD_BG}; padding:32px 40px">
{card_head("Book", "Ba khối chính")}
{bullet("Encoder ảnh và văn bản", "Cắt ảnh thành ô 16×16 thành token; tách câu lệnh thành token chữ")}
{bullet("VLM · Cosmos-Reason2-2B", "Trộn token ảnh và chữ; giữ 16/28 lớp, đóng băng")}
{bullet("Flow-matching DiT · 32 lớp", "Sinh chuỗi hành động 40 bước/chunk")}
</div>
<div style="flex:1; display:flex; flex-direction:column; background:#F3F7F5; border-radius:24px; padding:22px">
<div style="flex:1; display:flex; flex-direction:column; gap:16px; border:2px dashed #C6DCCF; border-radius:18px; padding:18px 22px; align-items:center; justify-content:center">
<img src="/_blob/68d57725bccee1d31dd9ea25c8f4a5f9" alt="Sơ đồ GR00T: ảnh, câu lệnh, trạng thái robot đi qua VLM và Diffusion Transformer ra hành động" style="width:780px; height:404px; object-fit:contain; background:#FFFFFF; border-radius:12px">
<div style="display:flex; flex-direction:row; gap:48px; justify-content:center">
<p style="font-size:26px; color:{INK}"><b>3,1 tỉ</b> tham số</p>
<p style="font-size:26px; color:{INK}"><b>40</b> bước / chunk</p>
<p style="font-size:26px; color:{INK}"><b>27,9 ms</b> trên H100</p>
</div>
</div>
</div>
</div>""", "Ba khối: encoder ảnh và văn bản (ảnh cắt ô 16×16, gộp 4 ô thành một token; câu lệnh tách thành token chữ); VLM Cosmos-Reason2-2B đóng băng, chỉ giữ 16 trên 28 lớp, nối sang action head qua 4 lớp self-attention; DiT 32 lớp khử nhiễu 40 token hành động trong 4 bước bằng flow matching, xen kẽ cross-attention tới văn bản và tới ảnh. Mỗi robot có ngăn trọng số riêng cho phần mã hoá trạng thái và giải mã hành động. 27,9 ms là thời gian suy luận khi chạy TensorRT.")


# ---------------------------------------------------------------- 04 DỮ LIỆU HUẤN LUYỆN GR00T
VERS = [
    ("N1", "03/2025", "#0E6B3F", [
        ("88 giờ", "teleop GR-1 thật"),
        ("780 nghìn", "quỹ đạo mô phỏng DexMimicGen"),
        ("827 giờ", "neural trajectory sinh từ video"),
        ("", "Open X-Embodiment, AgiBot-Alpha, video người Ego4D, EPIC-KITCHENS"),
    ]),
    ("N1.5", "06/2025", "#1C8A4B", [
        ("+", "neural trajectory DreamGen"),
        ("+", "AgiBot-Beta"),
        ("", "250 nghìn bước trên 1.000 GPU H100"),
    ]),
    ("N1.6", "12/2025", "#26A262", [
        ("+", "vài nghìn giờ teleop mới"),
        ("+", "tay đôi YAM, AgiBot Genie-1"),
        ("+", "Unitree G1 vừa đi vừa thao tác"),
        ("+", "Galaxea R1 Pro mô phỏng"),
    ]),
    ("N1.7", "2026", "#2DB36A", [
        ("21,6 triệu", "mẫu từ 13 dataset"),
        ("20 nghìn giờ", "video người EgoScale"),
        ("", "Relative action EEF chung cho người và robot"),
    ]),
]


def ver_card(name, date, color, rows):
    items = []
    for big, text in rows:
        if big == "+":
            items.append(f'<p style="font-size:24px; line-height:1.35; color:{BODY}"><span style="color:{GREEN}"><b>+</b></span> {text}</p>')
        elif big:
            items.append(f'<p style="font-size:24px; line-height:1.35; color:{BODY}"><b><span style="color:{INK}">{big}</span></b> {text}</p>')
        else:
            items.append(f'<p style="font-size:24px; line-height:1.35; color:{MUTED}">{text}</p>')
    return f"""<div style="flex:1; display:flex; flex-direction:column; background:#F3F8F5; border-radius:20px; overflow:hidden">
<div style="display:flex; flex-direction:row; align-items:center; justify-content:space-between; background:{color}; padding:16px 24px">
<p style="font-size:36px; font-weight:800; color:#FFFFFF">{name}</p>
<p style="font-size:24px; color:#E3F4EA">{date}</p>
</div>
<div style="flex:1; display:flex; flex-direction:column; gap:12px; padding:20px 24px">
{"".join(items)}
</div>
</div>"""


def ds_pill(name, desc):
    return f"""<div style="flex:1; display:flex; flex-direction:column; gap:2px; background:#1F5540; border-radius:14px; padding:12px 18px">
<p style="font-size:24px; font-weight:700; color:#F2F7F4">{name}</p>
<p style="font-size:24px; color:#BFD8CB">{desc}</p>
</div>"""


gr00tdata = section("gr00tdata", "padding:210px 128px 124px 120px; display:flex; flex-direction:column; gap:26px", f"""
{frame(4, ribbon_w=760)}
{header("Dữ liệu huấn luyện GR00T", "Nền tảng là kim tự tháp dữ liệu của N1; mỗi phiên bản bổ sung nguồn mới", size=64)}
<div style="display:flex; flex-direction:row; gap:20px; flex:1">
{"".join(ver_card(*v) for v in VERS)}
</div>
<div style="display:flex; flex-direction:row; gap:16px; align-items:center; background:{DEEP}; border-radius:18px; padding:14px 18px">
<p style="width:170px; font-size:24px; font-weight:700; line-height:1.3; color:#FFFFFF; text-transform:uppercase">Công khai trên HF</p>
{ds_pill("GR00T-X-Embodiment-Sim", "quỹ đạo mô phỏng nhiều robot")}
{ds_pill("GR00T-Teleop-Sim", "1.000 quỹ đạo GR-1 mô phỏng")}
{ds_pill("GR00T-Teleop-G1", "1.000 quỹ đạo G1 thật")}
</div>""", "N1 (bài báo arXiv 2503.14734): kim tự tháp dữ liệu. Đáy là video web và video người (Ego4D, Ego-Exo4D, EPIC-KITCHENS...); giữa là dữ liệu tổng hợp: 780 nghìn quỹ đạo mô phỏng DexMimicGen, tương đương 6.500 giờ, và 827 giờ neural trajectory sinh bằng mô hình video từ 88 giờ dữ liệu thật; đỉnh là dữ liệu robot thật: 88 giờ teleop GR-1, Open X-Embodiment, 140 nghìn quỹ đạo AgiBot-Alpha. Tiền huấn luyện khoảng 50.000 giờ GPU H100. N1.5: thêm neural trajectory DreamGen và AgiBot-Beta, 250 nghìn bước trên 1.000 GPU H100. N1.6: thêm vài nghìn giờ teleop từ tay đôi YAM, AgiBot Genie-1, Galaxea R1 Pro mô phỏng trên BEHAVIOR và Unitree G1 vừa đi vừa thao tác (loco-manipulation); 300 nghìn bước. N1.7: 21,6 triệu mẫu từ 13 dataset, thêm 20 nghìn giờ video người EgoScale nhờ ghi hành động relative ở EEF. Các dataset công khai nằm trong bộ PhysicalAI-Robotics của NVIDIA trên Hugging Face.")

# ---------------------------------------------------------------- 04 COMPARE
td_head = 'style="color:#F2F7F4"'
compare = section("compare", "padding:204px 128px 110px 120px; display:flex; flex-direction:column; gap:30px", f"""
{frame(5, ribbon_w=900)}
{header("So sánh với các mô hình VLA khác", "Cùng kiến trúc VLM kết hợp action head, khác biệt ở mục tiêu và dữ liệu", size=64)}
<table style="font-size:25px; color:{BODY}">
<tr style="background:#0E6B3F"><th style="width:22%; color:#F2F7F4">Tiêu chí</th><th style="width:28%; color:#F2F7F4">GR00T N1.7</th><th style="width:27%; color:#F2F7F4">π0.5</th><th style="width:23%; color:#F2F7F4">OpenVLA</th></tr>
<tr><td style="color:{INK}">Robot hình người</td><td style="color:{INK}">Hỗ trợ, kết hợp SONIC</td><td>Chưa hỗ trợ WBC</td><td>Chưa hỗ trợ WBC</td></tr>
<tr style="background:#F3F8F5"><td style="color:{INK}">Sinh hành động</td><td style="color:{INK}">Flow matching, 40 bước</td><td>Flow matching</td><td>Token rời rạc, kém mượt</td></tr>
<tr><td style="color:{INK}">Suy luận nhiều bước</td><td style="color:{INK}">Không</td><td>Có, tự sinh bước con</td><td>Không</td></tr>
<tr style="background:#F3F8F5"><td style="color:{INK}">Backbone</td><td style="color:{INK}">Cosmos-Reason2, đóng băng</td><td>PaliGemma 3B, huấn luyện cùng</td><td>Llama 2 7B</td></tr>
</table>
<div style="display:flex; flex-direction:row; gap:32px">
<div style="flex:1; display:flex; flex-direction:column; gap:12px; {CARD_BG}; padding:26px 34px">
<div style="display:flex; flex-direction:row; gap:18px; align-items:center">
<div style="width:60px; height:60px; border-radius:50%; background:#D5EFE0; display:flex; align-items:center; justify-content:center"><x-icon name="ThumbsUp" style="color:{DEEP}; width:36px; height:36px"></x-icon></div>
<h3 style="font-size:30px; font-weight:800; color:{DEEP}; text-transform:uppercase">Ưu điểm</h3>
</div>
<p style="font-size:25px; line-height:1.35; color:{INK}">Hỗ trợ trọn vẹn robot hình người</p>
<p style="font-size:25px; line-height:1.35; color:{INK}">Chi phí fine-tune thấp, chỉ huấn luyện action head</p>
<p style="font-size:25px; line-height:1.35; color:{INK}">Sẵn công cụ triển khai: ONNX, TensorRT, Jetson</p>
</div>
<div style="flex:1; display:flex; flex-direction:column; gap:12px; background:#F4F5F4; border-radius:24px; padding:26px 34px">
<div style="display:flex; flex-direction:row; gap:18px; align-items:center">
<div style="width:60px; height:60px; border-radius:50%; background:#E3E7E5; display:flex; align-items:center; justify-content:center"><x-icon name="Warning" style="color:{INK}; width:36px; height:36px"></x-icon></div>
<h3 style="font-size:30px; font-weight:800; color:{INK}; text-transform:uppercase">Nhược điểm</h3>
</div>
<p style="font-size:25px; line-height:1.35; color:{INK}">Chưa tự phân rã nhiệm vụ dài như π0.5</p>
<p style="font-size:25px; line-height:1.35; color:{INK}">Ít kết quả kiểm chứng khái quát ngoài thực tế</p>
<p style="font-size:25px; line-height:1.35; color:{INK}">Robot mới vẫn cần thu thập demo và fine-tune</p>
</div>
</div>""", "π0.5 là mô hình gần nhất: cùng khung VLM khoảng 3 tỉ tham số và action expert dùng flow matching. Ưu điểm của GR00T: hỗ trợ robot hình người (có SONIC), fine-tune rẻ vì backbone đóng băng, công cụ triển khai đầy đủ. Nhược điểm: không có suy luận nhiều bước và ít bằng chứng khái quát ngoài thực tế. OpenVLA rời rạc hoá hành động nên chậm và kém mượt.")

# ---------------------------------------------------------------- 05 SONIC
def layer(hz, title, desc):
    return f"""<div style="display:flex; flex-direction:row; gap:22px; align-items:start">
<p style="width:170px; font-size:25px; font-weight:800; line-height:1.3; color:{DEEP}; background:#D5EFE0; padding:6px 0px; border-radius:10px; text-align:center">{hz}</p>
<div style="flex:1; display:flex; flex-direction:column; gap:2px">
<p style="font-size:28px; font-weight:700; line-height:1.3; color:{INK}">{title}</p>
<p style="font-size:24px; line-height:1.35; color:{BODY}">{desc}</p>
</div>
</div>"""

sonic = section("sonic", "padding:210px 128px 116px 120px; display:flex; flex-direction:row; gap:40px", f"""
{frame(6, ribbon_w=560)}
<div style="flex:1; display:flex; flex-direction:column; gap:32px">
{header("Kiến trúc<br>GEAR-SONIC", "Bộ điều khiển toàn thân 42 triệu tham số", size=64)}
<div style="flex:1; display:flex; flex-direction:column; gap:22px; {CARD_BG}; padding:30px 36px">
{card_head("Settings", "Ba lớp điều khiển")}
{layer("2,5–30 Hz", "Bộ sinh chuyển động", "Kinematic Planner, VR, GEM hoặc GR00T")}
{layer("50 Hz", "Universal Token 64D", "FSQ quy mọi nguồn lệnh về token chung")}
{layer("50 → 500 Hz", "Giải mã và PD", "Bộ giải mã giữ thăng bằng, PD tạo mô-men")}
</div>
</div>
<div style="width:860px; display:flex; flex-direction:column; gap:18px; background:#F3F7F5; border-radius:24px; padding:22px">
<div style="flex:1; display:flex; flex-direction:column; border:2px dashed #C6DCCF; border-radius:18px; padding:16px; align-items:center; justify-content:center">
<img src="/_blob/0092ac61791bf793be24ab6022adbf52" alt="Ba lớp của SONIC: bộ sinh chuyển động, không gian Universal Token, giải mã và bộ điều khiển PD" style="width:780px; height:500px; object-fit:contain; background:#FFFFFF; border-radius:12px">
</div>
<div style="display:flex; flex-direction:row; gap:18px">
<div style="flex:1; display:flex; flex-direction:row; gap:16px; align-items:center; background:{DEEP}; border-radius:16px; padding:16px 22px">
<p style="font-size:36px; font-weight:800; color:#F2F7F4; white-space:nowrap">611 giờ</p><p style="font-size:24px; color:#CDE7D8">dữ liệu mocap</p>
</div>
<div style="flex:1; display:flex; flex-direction:row; gap:16px; align-items:center; background:{DEEP}; border-radius:16px; padding:16px 22px">
<p style="font-size:36px; font-weight:800; color:#F2F7F4; white-space:nowrap">4.096</p><p style="font-size:24px; color:#CDE7D8">robot ảo, PPO</p>
</div>
</div>
</div>""", "SONIC chia điều khiển thành ba lớp theo tần số. Lớp 1 là các nguồn lệnh. Lớp 2 dùng FSQ để lượng tử hoá mọi nguồn về cùng một Universal Token 64 chiều. Lớp 3: bộ giải mã điều khiển (Actor của PPO) nhận token và cảm biến, xuất góc cho 29 khớp ở 50 Hz, rồi PD 500 Hz tạo mô-men. Huấn luyện trong Isaac Lab trên khoảng 130 nghìn đoạn chuyển động (611 giờ), 4.096 robot song song, có adaptive sampling và domain randomization.")

# ---------------------------------------------------------------- 10 ROADMAP
STEPS = [
    ("Database", "Bước 1 · Hoàn tất", "Checkpoint C", "Dữ liệu 440 episode thu với chân tự giữ thăng bằng, cân bằng 4 nhóm cảnh. Mục tiêu: không kém B trên cả hai tập, lon đứng đổi phía vượt 0/4."),
    ("Wrench", "Bước 2", "Gắp vật lý thật", "Bỏ cơ chế dính lon vào tay trong mô phỏng, gắp bằng lực ma sát; ngẫu nhiên hoá ánh sáng, nền và camera."),
    ("Verified", "Bước 3", "Kiểm chứng SONIC", "Gán motion token bằng bộ mã hoá SONIC. Đạt thì fine-tune GR00T với tag UNITREE_G1_SONIC; không đạt thì giữ bộ thăng bằng GR00T-WBC."),
    ("Globe", "Bước 4", "Thử trên G1 thật", "Bắt đầu với nhiệm vụ phân loại lon, bổ sung demo thật qua teleop PICO; sau đó mở rộng nhiệm vụ mới."),
]
GAP, LEFT = 20, 120
COL_W = (1672 - 3 * GAP) // 4
ROW_TOP = 210 + 170 + 20          # padding + khối tiêu đề + gap
CY = ROW_TOP + 82                 # tâm vòng tròn
centers = [LEFT + i * (COL_W + GAP) + COL_W // 2 for i in range(4)]
line = (f'<div style="position:absolute; left:{centers[0]}px; top:{CY - 1}px; width:{centers[-1] - centers[0]}px; height:3px; background:{GREEN}"></div>'
        + "".join(f'<div style="position:absolute; left:{(a + b) // 2 - 10}px; top:{CY - 10}px; width:20px; height:20px; border-radius:50%; background:{GREEN}"></div>'
                  for a, b in zip(centers, centers[1:])))
cols = []
for icon, step, title, desc in STEPS:
    cols.append(f"""<div style="position:relative; width:{COL_W}px; display:flex; flex-direction:column; padding:104px 0px 0px 0px">
<div style="flex:1; display:flex; flex-direction:column; gap:8px; background:linear-gradient(180deg, #EAF6EF 0%, #F6FBF8 100%); border-radius:22px; padding:74px 24px 22px 24px">
<p style="font-size:26px; font-weight:700; color:{GREEN}">{step}</p>
<h3 style="font-size:34px; font-weight:800; line-height:1.2; color:{INK}; white-space:nowrap">{title}</h3>
<hr style="border-top:2px solid #D6E8DD">
<p style="font-size:25px; line-height:1.45; color:{BODY}">{desc}</p>
</div>
<div style="position:absolute; left:{COL_W // 2 - 82}px; top:0px; width:164px; height:164px; border-radius:50%; background:#F2FAF5; border:3px solid #D3ECDD"></div>
<div style="position:absolute; left:{COL_W // 2 - 64}px; top:18px; width:128px; height:128px; border-radius:50%; background:#D2EEDD"></div>
<x-icon name="{icon}" style="position:absolute; left:{COL_W // 2 - 36}px; top:46px; width:72px; height:72px; color:{INK}"></x-icon>
</div>""")
roadmap = section("roadmap", "padding:210px 128px 126px 120px; display:flex; flex-direction:column; gap:20px", f"""
{line}
{frame(11, bar=False, ribbon_w=0, corner=True, footer_w=1460)}
{header("Định hướng nghiên cứu", "Lộ trình trong thời gian tới: từ mô phỏng đến robot Unitree G1 thật")}
<div style="display:flex; flex-direction:row; gap:{GAP}px; flex:1">
{"".join(cols)}
</div>
<p style="font-size:24px; font-style:italic; color:{MUTED}">LoRA cho bộ mã hoá ảnh chỉ thử nếu lỗi đến từ nhận diện màu lon.</p>""", "Bước 1 là việc đang làm: checkpoint C train trên dữ liệu 440 episode thu với chân tự giữ thăng bằng. Bước 2 bỏ ràng buộc weld: trong mô phỏng lon dính vào cổ tay khi tay khép trong bán kính 5 cm, robot thật không có cơ chế này. Bước 3 có điểm dừng: cách ghép đầu vào bộ mã hoá SONIC chưa kiểm chứng và NVIDIA chưa công bố checkpoint hay dữ liệu cho tag UNITREE_G1_SONIC; nếu không gán được token thì giữ bộ thăng bằng GR00T-WBC. Bước 4 bắt đầu với chính nhiệm vụ phân loại lon trên robot thật.")

# ---------------------------------------------------------------- 11 THANKS
TR = """<svg aria-hidden="true" width="620" height="520" viewBox="0 0 620 520" style="position:absolute; left:1300px; top:0px; width:620px; height:520px">
<defs><linearGradient id="ta" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FFFFFF" stop-opacity="0"/><stop offset="1" stop-color="#9ED9B6"/></linearGradient>
<linearGradient id="tb" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#D8F1E2"/><stop offset="1" stop-color="#4FBF80"/></linearGradient></defs>
<polygon points="80,0 620,0 620,520" fill="url(#ta)"/>
<polygon points="330,0 620,0 620,380" fill="url(#tb)"/>
<line x1="240" y1="0" x2="620" y2="500" stroke="#FFFFFF" stroke-width="4"/>
</svg>"""
BL = """<svg aria-hidden="true" width="560" height="440" viewBox="0 0 560 440" style="position:absolute; left:0px; top:640px; width:560px; height:440px">
<defs><linearGradient id="ba" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#9ED9B6"/><stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></linearGradient>
<linearGradient id="bb" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1FA35C"/><stop offset="1" stop-color="#0E6B3F"/></linearGradient></defs>
<polygon points="0,0 400,440 0,440" fill="url(#ba)"/>
<polygon points="0,290 130,440 0,440" fill="url(#bb)"/>
<line x1="0" y1="90" x2="320" y2="440" stroke="#FFFFFF" stroke-width="4"/>
</svg>"""
def pill(icon, text):
    return f"""<div style="display:flex; flex-direction:row; gap:18px; align-items:center">
<div style="width:64px; height:64px; border-radius:50%; background:#D5EFE0; display:flex; align-items:center; justify-content:center"><x-icon name="{icon}" style="color:{DEEP}; width:38px; height:38px"></x-icon></div>
<p style="font-size:27px; color:{INK}; white-space:nowrap">{text}</p>
</div>"""
thanks = section("thanks", "padding:160px 128px 150px 128px; display:flex; flex-direction:column; gap:30px; align-items:center; justify-content:center", f"""
{TR}
{BL}
<div style="position:absolute; left:112px; top:990px; width:1620px; height:2px; background:linear-gradient(90deg, #D9EEE2 0%, #8FCDA9 100%)"></div>
<p style="position:absolute; left:112px; top:1012px; width:600px; font-size:24px; color:{MUTED}">VinDynamics  |  Confidential</p>
<p style="position:absolute; left:1744px; top:970px; width:120px; font-size:28px; font-weight:700; line-height:1.4; text-align:center; color:{INK}; background:#E6F4EC; padding:10px 0px; border-radius:16px">12</p>
<img src="{LOGO_BIG}" alt="VinDynamics" style="width:460px; height:274px; object-fit:contain">
<h1 style="font-size:150px; font-weight:800; line-height:1.05; letter-spacing:-2px; color:{DEEP}; text-align:center">XIN CẢM ƠN</h1>
<p style="font-size:38px; color:{MUTED}; text-align:center">Rất mong nhận được ý kiến đóng góp và trao đổi thêm.</p>
<div style="width:160px; height:8px; border-radius:4px; background:linear-gradient(90deg, #1FA35C 0%, #CFEBDB 100%)"></div>
<div style="display:flex; flex-direction:row; gap:40px; align-items:center; background:#F1F8F4; border-radius:40px; padding:18px 44px">
{pill("Users", "Nguyễn Duy Hoàng · Phạm Thanh Hưng")}
<div style="width:2px; height:48px; background:#CFE3D7"></div>
{pill("GraduationCap", "Mentor: Trần Ngọc Sơn")}
</div>""", "Cảm ơn và mời đặt câu hỏi.")

for name, html in [("gr00t", gr00t), ("gr00tdata", gr00tdata), ("compare", compare), ("sonic", sonic), ("roadmap", roadmap), ("thanks", thanks)]:
    (OUT / f"{name}.html").write_text(html, encoding="utf-8")
print("ok")
