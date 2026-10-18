"""Hardware detection and diagnostic module for GPU and CUDA acceleration."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HardwareInfo:
    """Hardware configuration and CUDA acceleration capability."""
    has_nvidia_gpu: bool
    gpu_name: Optional[str]
    vram_mb: Optional[int]
    driver_version: Optional[str]
    torch_cuda_available: bool
    torch_version: str
    effective_device: str
    status_summary: str
    recommendation_zh: str


def detect_hardware() -> HardwareInfo:
    """Detect local NVIDIA GPU and PyTorch CUDA capability."""
    import torch

    has_nvidia_gpu = False
    gpu_name = None
    vram_mb = None
    driver_version = None

    # 1. Detect physical NVIDIA GPU using nvidia-smi
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                has_nvidia_gpu = True
                gpu_name = parts[0]
                try:
                    vram_mb = int(parts[1])
                except ValueError:
                    vram_mb = None
                driver_version = parts[2]
    except Exception:
        pass

    # 2. Check PyTorch CUDA availability
    torch_cuda = torch.cuda.is_available()
    torch_ver = getattr(torch, "__version__", "unknown")

    if torch_cuda:
        effective_device = "cuda"
        cuda_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else (gpu_name or "CUDA GPU")
        status_summary = f"CUDA GPU 加速已就绪: {cuda_name}"
        recommendation_zh = "GPU 硬件加速已就绪，画质评估算法将优先在 GPU 上运算。"
    elif has_nvidia_gpu:
        effective_device = "cpu"
        status_summary = f"检测到独显: {gpu_name} (当前为 CPU 运行模式)"
        recommendation_zh = (
            f"检测到独立显卡【{gpu_name} ({vram_mb or 8192}MB 显存)】，"
            f"但当前运行的 PyTorch 版本为【{torch_ver} (CPU构建)】，未包含 CUDA 支持。\n"
            "建议：如需使用 GPU 加速，请在安装了 CUDA 支持的 PyTorch 环境中运行。"
        )
    else:
        effective_device = "cpu"
        status_summary = "CPU 运算模式"
        recommendation_zh = "未检测到可用的 NVIDIA 独立显卡，当前使用 CPU 运行。"

    return HardwareInfo(
        has_nvidia_gpu=has_nvidia_gpu,
        gpu_name=gpu_name,
        vram_mb=vram_mb,
        driver_version=driver_version,
        torch_cuda_available=torch_cuda,
        torch_version=torch_ver,
        effective_device=effective_device,
        status_summary=status_summary,
        recommendation_zh=recommendation_zh,
    )
