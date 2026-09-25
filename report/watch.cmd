@echo off
cd /d "%~dp0"
echo Dang theo doi Nvidia_groot.tex ... (Ctrl+C de dung)
echo PDF cap nhat lien tuc tai build\Nvidia_groot_build.pdf - mo bang SumatraPDF hoac VS Code
latexmk -xelatex -pvc -view=none -jobname=Nvidia_groot_build Nvidia_groot.tex
