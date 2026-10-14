#!/usr/bin/env bash
# Tách các hình TikZ của Nvidia_groot.tex thành SVG trong slides/assets/tikz/ (dùng cho deck_tikz).
# Cần build report trước (./compile.sh) để có build/Nvidia_groot_build.aux, nhờ đó "Hình 4.1", "Chương 5"
# trong hình không thành "??".
set -euo pipefail
REPORT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$REPORT/build/tikz_extract"
OUT="$REPORT/slides/assets/tikz"
mkdir -p "$WORK" "$OUT"
# Gói preview: mỗi tikzpicture thành một trang PDF vừa khít hình
sed 's/^\\usepackage{tikz}/\\usepackage{tikz}\n\\usepackage[active,tightpage]{preview}\\PreviewEnvironment{tikzpicture}\\setlength\\PreviewBorder{8pt}/' \
    "$REPORT/Nvidia_groot.tex" > "$WORK/figs.tex"
cp "$REPORT/build/Nvidia_groot_build.aux" "$WORK/figs.aux"
(cd "$WORK" && TEXINPUTS="$REPORT//:" xelatex -interaction=nonstopmode figs.tex >/dev/null || true)
NAMES=(gr00t_arch sonic_3layer gr00t_sonic_arch motionbricks mb_arch rp01_pipeline seed_prep ft_flow sonic_train enc_window fsq prop_window
       reward_curve reward_body dr_timeline ppo_iter ppo_clip combine runtime_detail timeline label_flow)
PAGES=$(pdfinfo "$WORK/figs.pdf" | awk '/^Pages/{print $2}')
[ "$PAGES" -eq "${#NAMES[@]}" ] || { echo "LỖI: $PAGES trang, cần ${#NAMES[@]}; sửa danh sách NAMES"; exit 1; }
for i in "${!NAMES[@]}"; do
    pdftocairo -svg -f $((i+1)) -l $((i+1)) "$WORK/figs.pdf" "$OUT/${NAMES[$i]}.svg"
done
cp "$WORK/figs.pdf" "$OUT/tikz_figures.pdf"
echo "Đã ghi ${#NAMES[@]} SVG vào $OUT"
