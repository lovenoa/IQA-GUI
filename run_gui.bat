@echo off
chcp 65001 >nul
cd /d "%~dp0"
title IQA-GUI 启动器
echo ========================================================
echo   正在启动 IQA-PyTorch 图像质量评估桌面工作台...
echo ========================================================
python -m iqa_gui.main
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 启动发生错误，按任意键退出...
    pause > nul
)
