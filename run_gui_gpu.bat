@echo off
chcp 65001 >nul
cd /d "%~dp0"
title IQA-GUI (GPU 显卡加速版)
echo ========================================================
echo   正在使用 Python 3.12 (GPU 加速环境) 启动工作台...
echo ========================================================
"C:\Users\loven\AppData\Local\Programs\Python\Python312\python.exe" -m iqa_gui.main
if %errorlevel% neq 0 (
    echo.
    echo [提示] 绝对路径启动异常，正在尝试使用 py 启动器重试...
    py -3.12 -m iqa_gui.main
)
if %errorlevel% neq 0 (
    echo.
    echo 程序启动遇到错误，请查看上方提示信息。按任意键退出...
    pause >nul
)
