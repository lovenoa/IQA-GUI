@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
set "IQA_GPU_CHECK=1"
echo 将先显示实际 CUDA 状态，再启动工作台；CUDA 不可用时程序使用 CPU。
call "%~dp0run_gui.bat"
exit /b %errorlevel%
