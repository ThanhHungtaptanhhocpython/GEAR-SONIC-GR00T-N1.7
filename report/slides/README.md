# Bộ slide "GR00T N1.7 × GEAR-SONIC"

Mã nguồn của bộ slide thuyết trình. Bản PDF xuất ra: `report/Slides_GR00T_N1.7_SONIC.pdf`.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `deck/deck.json` | Tên deck, thứ tự slide, font |
| `deck/slides/*.html` | Mỗi file là một slide (khung 1920×1080), ghi chú thuyết trình nằm trong `<aside>` |
| `gen.py` | Sinh 6 slide dùng khung mới: `gr00t`, `gr00tdata`, `compare`, `sonic`, `roadmap`, `thanks` |
| `build_pdf.py` | Ghép các slide và in ra PDF bằng Edge/Chrome |
| `assets/` | Logo và hình dữ liệu dùng trong slide |
| `tools/make_figures.py` | Vẽ lại hình dữ liệu trong `assets/` từ dataset |

Hình kiến trúc (GR00T, SONIC, luồng ghép, cảnh mô phỏng) lấy từ `report/figures/`.

## Sửa slide

- 6 slide trong `gen.py`: sửa nội dung trong `gen.py` rồi chạy `python report/slides/gen.py`. Không sửa tay các file HTML này vì lần chạy sau sẽ ghi đè.
- 6 slide còn lại (`cover`, `overall`, `combine`, `data`, `results`, `lessons`): sửa trực tiếp file HTML.
- Thêm hoặc đổi thứ tự slide: sửa `order` trong `deck/deck.json`. Số trang ở góc slide ghi cứng trong từng slide, nhớ cập nhật.

Slide chỉ dùng một tập CSS con: flex/grid, kích thước bằng px, không có `margin`. Ảnh được tham chiếu dạng `/_blob/<id>` (id trong kho ảnh của bản slide trên web); `build_pdf.py` ánh xạ các id này sang file cục bộ trong `BLOBS`. Khi thêm ảnh mới cần thêm cả vào `BLOBS`.

## Xuất PDF

```bash
python report/slides/build_pdf.py              # ghi report/Slides_GR00T_N1.7_SONIC.pdf
python report/slides/build_pdf.py --html-only  # chỉ tạo build/deck_print.html để xem nhanh
```

Cần Edge hoặc Chrome và kết nối mạng để tải font Be Vietnam Pro. Icon trong PDF là bản vẽ thay thế nên hơi khác icon trên bản web.

## Vẽ lại hình dữ liệu

```bash
pip install mujoco imageio imageio-ffmpeg pandas matplotlib pillow
python report/slides/tools/make_figures.py
```

## Bộ slide hình TikZ

Bộ slide riêng gồm 14 hình TikZ của báo cáo (Hình 1.1–1.3, 3.1–3.3, 4.1–4.4, 5.1–5.4) và phần thống kê Bones-SEED
(slide tổng quan số liệu + 7 biểu đồ Hình 2.1–2.3, A.1–A.4, đặt ngay sau các hình Chương 1), cùng mẫu trình bày với bộ chính.
Thư mục `deck_tikz/` do `gen_tikz.py` sinh ra, không sửa tay; đổi tiêu đề, phụ đề hoặc phần giải thích thì sửa `FIGS`, `EXPLAIN`, `STATS`, `OVERVIEW` trong script. Mọi slide hình TikZ để giải thích ở cột trái, hình bên phải.
Biểu đồ thống kê lấy thẳng từ `report/figures/bones_seed/*.png` (đường dẫn `/_fig/...`), nên vẽ lại biểu đồ thì chỉ cần chạy lại hai lệnh cuối.

```bash
report/slides/tools/extract_tikz.sh   # tách hình từ Nvidia_groot.tex ra assets/tikz/*.svg (build report trước)
python report/slides/gen_tikz.py      # ghi deck_tikz/deck.json và deck_tikz/slides/*.html
python report/slides/build_pdf.py --deck deck_tikz --out Slides_Hinh_TikZ_GR00T_SONIC.pdf
```

Sửa hình trong báo cáo thì chạy lại cả ba lệnh. Ảnh trong slide trỏ `/_asset/<đường dẫn trong assets/>`, `build_pdf.py` tự đổi sang file cục bộ.
