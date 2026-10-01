#!/usr/bin/env bash
# Biên dịch Nvidia_groot.tex trên Linux (tương đương compile.cmd)
# Dùng: ./compile.sh          -> biên dịch một lần
#       ./compile.sh watch    -> tự biên dịch lại khi lưu file
cd "$(dirname "$0")" || exit 1
if [ "$1" = "watch" ]; then
  exec latexmk -xelatex -pvc -view=none -jobname=Nvidia_groot_build Nvidia_groot.tex
fi
if ! latexmk -xelatex -jobname=Nvidia_groot_build Nvidia_groot.tex; then
  echo; echo "LOI BIEN DICH. Xem build/Nvidia_groot_build.log"; exit 1
fi
cp -f build/Nvidia_groot_build.pdf Nvidia_groot.pdf && echo && echo "Da xuat: $PWD/Nvidia_groot.pdf"
