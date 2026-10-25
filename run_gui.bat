@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
title IQA-GUI 启动器
if exist "%~dp0.venv\Scripts\python.exe" (
    set "IQA_PYTHON=%~dp0.venv\Scripts\python.exe"
    goto launch
)
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "IQA_PYTHON=%LocalAppData%\Programs\Python\Python312\python.exe"
    goto launch
)
where python >nul 2>nul
if errorlevel 1 (
    echo 未找到 Python。请按 README.md 准备 Python 3.12 和运行依赖。
    pause
    exit /b 1
)
set "IQA_PYTHON=python"
:launch
echo 使用解释器: "%IQA_PYTHON%"
if defined IQA_GPU_CHECK (
    "%IQA_PYTHON%" -c "import torch; print('CUDA: ' + str(torch.cuda.is_available()) + ' / PyTorch: ' + torch.__version__)"
    if errorlevel 1 goto failed
)
"%IQA_PYTHON%" -m iqa_gui.main
set "IQA_EXIT=%errorlevel%"
if not "%IQA_EXIT%"=="0" goto failed
exit /b 0
:failed
echo 启动或运行失败。请查看上方错误及 README.md；启动器不会安装或替换任何依赖。
pause
exit /b 1
