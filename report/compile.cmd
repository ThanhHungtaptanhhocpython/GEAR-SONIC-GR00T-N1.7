@echo off
cd /d "%~dp0"
latexmk -xelatex -jobname=Nvidia_groot_build Nvidia_groot.tex
if errorlevel 1 (
  echo.
  echo LOI BIEN DICH. Xem build\Nvidia_groot_build.log
  pause
  exit /b 1
)
copy /y build\Nvidia_groot_build.pdf Nvidia_groot.pdf >nul 2>&1
if errorlevel 1 (
  echo.
  echo Khong ghi de duoc Nvidia_groot.pdf vi dang mo trong trinh xem PDF.
  echo Ban moi nam o: %~dp0build\Nvidia_groot_build.pdf
) else (
  echo.
  echo Da xuat: %~dp0Nvidia_groot.pdf
)
pause
