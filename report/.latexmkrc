# Cấu hình latexmk cho Nvidia_groot.tex
$pdf_mode = 5;                 # 5 = XeLaTeX -> PDF
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
$out_dir = 'build';            # file trung gian (.aux, .log, .toc ...) để trong build/
$pdf_previewer = 'start ""';   # Windows: mở PDF bằng trình xem mặc định
$clean_ext = 'synctex.gz run.xml';
@default_files = ('Nvidia_groot.tex');
