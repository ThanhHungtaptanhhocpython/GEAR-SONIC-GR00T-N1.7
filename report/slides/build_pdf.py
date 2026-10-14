"""Xuất bộ slide ra PDF.

Ghép deck/slides/*.html theo thứ tự trong deck/deck.json thành một trang HTML in được,
rồi in bằng Edge hoặc Chrome chạy headless.

Dùng:
    python report/slides/build_pdf.py              -> report/Slides_GR00T_N1.7_SONIC.pdf
    python report/slides/build_pdf.py --html-only  -> chỉ tạo build/deck_print.html để xem trên trình duyệt
    python report/slides/build_pdf.py --deck deck_tikz --out Slides_Hinh_TikZ.pdf
                                                   -> in bộ slide khác (thư mục trong slides/, PDF đặt trong report/)
"""
import json
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DECK = HERE / "deck"
ASSETS = HERE / "assets"
FIG = HERE.parent / "figures"
BUILD = HERE / "build"
PDF_OUT = HERE.parent / "Slides_GR00T_N1.7_SONIC.pdf"


def _arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


if _arg("--deck"):
    DECK = HERE / _arg("--deck")
if _arg("--out"):
    PDF_OUT = HERE.parent / _arg("--out")

# Slide trên web tham chiếu ảnh bằng id trong kho asset (/_blob/<id>); khi in cục bộ thay bằng file tương ứng.
BLOBS = {
    "199565b59f8ae0c9c6658b583fbd6750": ASSETS / "logo.png",
    "baafa340f95ff6609ebd3c315364b382": ASSETS / "logo_h.png",
    "f1ac4e10cfa6c9c1fb5c0e9c0bfcf6b5": ASSETS / "logo_big.png",
    "556bcd70d48db8418bdfc56fe6ea8214": ASSETS / "can_positions.png",
    "d730541930fbe81fb1f892b073b01231": ASSETS / "can_positions_C.png",
    "094397df87a7c87e25a4bffbd8d6c683": ASSETS / "coke_stand_same_0.png",
    "cc9f5ca42a19c8d5468a36e5f25c9832": ASSETS / "pepsi_lying_swap_1.png",
    "68d57725bccee1d31dd9ea25c8f4a5f9": FIG / "arch.png",
    "0092ac61791bf793be24ab6022adbf52": FIG / "sonic_3layer_control_architecture.jpg",
    "d8a9dacc0f64424f30f501a67cd14cb7": FIG / "vla_gr00t_sonic_flow.jpg",
    "70a4340cdd36daf82960b7336fa94eef": FIG / "front.png",
    "130ba67d447c403ff86e17f6dc8435b7": ASSETS / "cover_video_still.png",  # ảnh tĩnh của video bìa
    "ab2f94b32e672f942b159bb6b59077e6": ASSETS / "cover_video.mp4",  # video bìa (PDF chỉ hiện ảnh tĩnh)
}

# Bản vẽ nét thay cho <x-icon>: trình xem slide web có bộ icon riêng, trình duyệt thì không.
ICONS = {
    "Database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
    "Wrench": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.8-3.8a6 6 0 0 1-7.9 7.9l-6.9 6.9a2.1 2.1 0 0 1-3-3l6.9-6.9a6 6 0 0 1 7.9-7.9z"/>',
    "Verified": '<path d="M9 12l2 2 4-4"/><path d="M12 2l2.4 1.8 3-.2.9 2.9 2.5 1.7-1 2.8 1 2.8-2.5 1.7-.9 2.9-3-.2L12 22l-2.4-1.8-3 .2-.9-2.9L3.2 15.8l1-2.8-1-2.8 2.5-1.7.9-2.9 3 .2z"/>',
    "Book": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8 13h8M8 17h8M8 9h2"/>',
    "ThumbsUp": '<path d="M7 10v12"/><path d="M15 5.9 14 10h5.8a2 2 0 0 1 2 2.3l-1.4 8A2 2 0 0 1 18.4 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.8a2 2 0 0 0 1.8-1.1L12 2a3.1 3.1 0 0 1 3 3.9z"/>',
    "Warning": '<path d="m21.7 18-8-14a2 2 0 0 0-3.5 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3z"/><path d="M12 9v4M12 17h.01"/>',
    "Settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    "Code": '<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>',
    "Chart": '<path d="M3 3v18h18"/><path d="M8 17v-5M13 17V8M18 17v-9"/>',
    "Globe": '<circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15 15 0 0 1 0 20 15 15 0 0 1 0-20z"/>',
    "Users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/>',
    "GraduationCap": '<path d="M22 10 12 5 2 10l10 5 10-5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>',
}

# CSS mô phỏng cách trình xem slide dựng trang: div mặc định là cột flex, khung 1920x1080 mỗi trang.
CSS = """@page{size:1920px 1080px;margin:0}
*{margin:0;padding:0;box-sizing:border-box}
html,body{background:#fff}
section{position:relative;width:1920px;height:1080px;overflow:hidden;page-break-after:always;break-after:page}
section:not([style*="display"]){display:flex;flex-direction:column}
div{display:flex;flex-direction:column}
div[style*="display:flex"]{display:flex;flex-direction:row}
img{display:block}
h1{font-size:96px}h2{font-size:64px}h3{font-size:44px}
table{width:100%;border-collapse:collapse}
th,td{padding:0.35em 0.6em;text-align:left;border-bottom:1px solid #E2E9E5}
th{font-weight:600}
hr{border:none}
"""

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "microsoft-edge", "google-chrome", "chromium", "chromium-browser",
]


def _px(style, prop):
    return int(re.search(rf"{prop}:(\d+)px", style).group(1))


def _arrow(m):
    st = m.group(1)
    w, h = _px(st, "width"), _px(st, "height")
    c = re.search(r"background:(#\w+)", st).group(1)
    pts = f"0,{h*0.3} {w*0.6},{h*0.3} {w*0.6},0 {w},{h/2} {w*0.6},{h} {w*0.6},{h*0.7} 0,{h*0.7}"
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="flex:none"><polygon points="{pts}" fill="{c}"/></svg>'


def _icon(m):
    name, st = m.group(1), m.group(2)
    w = _px(st, "width")
    c = re.search(r"color:(#\w+)", st).group(1)
    pos = ";".join(x for x in st.split(";") if x.strip().split(":")[0] in ("position", "left", "top"))
    # glyph chiếm ~62% ô, giống icon trên web; icon ghim tuyệt đối thì dời vào giữa ô
    shift = f"transform:translate({int(w*0.19)}px,{int(w*0.19)}px)" if "absolute" in pos else ""
    s = int(w * 0.62)
    return (f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" style="flex:none;{pos};{shift}">{ICONS[name]}</svg>')


def build_html():
    deck = json.loads((DECK / "deck.json").read_text(encoding="utf-8"))
    parts = []
    for sid in deck["order"]:
        h = (DECK / f"slides/{sid}.html").read_text(encoding="utf-8")
        h = re.sub(r"<aside>.*?</aside>", "", h, flags=re.S)  # bỏ ghi chú thuyết trình
        h = re.sub(r"/_blob/([0-9a-f]{32})", lambda m: BLOBS[m.group(1)].as_uri(), h)
        h = re.sub(r"/_asset/([\w./-]+)", lambda m: (ASSETS / m.group(1)).as_uri(), h)  # file trong assets/
        h = re.sub(r"/_fig/([\w./-]+)", lambda m: (FIG / m.group(1)).as_uri(), h)  # file trong report/figures/
        h = re.sub(r'<x-shape kind="arrow-right" style="([^"]*)"></x-shape>', _arrow, h)
        h = re.sub(r'<x-icon name="(\w+)" style="([^"]*)"></x-icon>', _icon, h)
        parts.append(h)
    font = deck["faces"]["be-vietnam-pro"]["href"]
    html = (f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>{deck["title"]}</title>\n'
            f'<link rel="stylesheet" href="{font}"><style>{CSS}</style></head><body>\n{"".join(parts)}</body></html>')
    BUILD.mkdir(exist_ok=True)
    path = BUILD / f"{DECK.name}_print.html"
    path.write_text(html, encoding="utf-8")
    return path, len(deck["order"])


def find_browser():
    for b in BROWSERS:
        p = shutil.which(b) or (b if pathlib.Path(b).exists() else None)
        if p:
            return p
    sys.exit("Không tìm thấy Edge hoặc Chrome để in PDF.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    html, n = build_html()
    print(f"HTML: {html} ({n} slide)")
    if "--html-only" not in sys.argv:
        before = PDF_OUT.stat().st_mtime if PDF_OUT.exists() else None
        # virtual-time-budget: chờ font Google tải xong rồi mới in
        subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=15000", f"--print-to-pdf={PDF_OUT}", html.as_uri()], check=True)
        # Trình duyệt vẫn báo thành công khi file PDF đang mở ở chương trình khác (bị khoá), nên tự kiểm tra
        if not PDF_OUT.exists() or PDF_OUT.stat().st_mtime == before:
            sys.exit(f"LỖI: {PDF_OUT.name} không được ghi đè. Đóng file PDF đang mở (VS Code, trình xem PDF) rồi chạy lại.")
        print(f"PDF:  {PDF_OUT}")
