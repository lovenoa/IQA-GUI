@echo off
chcp 65001 >nul
cd /d "%~dp0"
title IQA-GUI 启动器
echo ========================================================
echo   正在启动 IQA-PyTorch 图像质量评估桌面工作台...
echo ========================================================
python -m iqa_gui.main
if %errorlevel% neq 0 (
    echo.
    echo [提示] 默认 Python 启动异常，正在尝试使用 py -3.12 启动...
    py -3.12 -m iqa_gui.main
)
if %errorlevel% neq 0 (
    echo.
    echo 程序启动遇到错误，请查看上方提示信息。按任意键退出...
    pause >nul
)
