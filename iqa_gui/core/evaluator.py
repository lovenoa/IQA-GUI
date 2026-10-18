"""Core evaluation engine for single image / image-pair quality assessment."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

# Safe mock for missing optional dependencies (e.g. icecream in pyiqa q_align)
try:
    import icecream
except ImportError:
    import sys
    import types
    mod = types.ModuleType("icecream")
    mod.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
    sys.modules["icecream"] = mod

# Safe fallback shim for pkg_resources (used by clip)
try:
    import pkg_resources
except ImportError:
    try:
        import sys
        import types
        import packaging
        import packaging.version
        mod = types.ModuleType("pkg_resources")
        mod.packaging = packaging
        sys.modules["pkg_resources"] = mod
    except Exception:
        pass

try:
    import pyiqa
    import torch
except ImportError:
    pyiqa = None
    torch = None

from iqa_gui.core.model_manager import ModelManager


def patch_mplug_owl_compatibility() -> bool:
    """Patch MplugOwlVisualAbstractorModel for transformers >= 4.38 compatibility.

    In newer transformers releases, PreTrainedModel/ModuleUtilsMixin removed
    invert_attention_mask, leading to an AttributeError when running Q-Align or
    mPLUG-Owl2 based visual abstractors.
    """
    try:
        import pyiqa.archs.q_align.visual_encoder as ve
        if hasattr(ve, "MplugOwlVisualAbstractorModel"):
            cls = ve.MplugOwlVisualAbstractorModel
            if not hasattr(cls, "invert_attention_mask"):
                def invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
                    if encoder_attention_mask.dim() == 3:
                        extended = encoder_attention_mask[:, None, :, :]
                    elif encoder_attention_mask.dim() == 2:
                        extended = encoder_attention_mask[:, None, None, :]
                    else:
                        raise ValueError(
                            f"Wrong shape for encoder_attention_mask (shape {encoder_attention_mask.shape})"
                        )
                    extended = extended.to(dtype=self.dtype)
                    return (1.0 - extended) * -10000.0

                cls.invert_attention_mask = invert_attention_mask
                return True
    except Exception:
        pass
    return False


# Apply patch immediately upon module import
patch_mplug_owl_compatibility()


@dataclass
class SingleEvalResult:
    """Result of evaluating a single image or image pair."""
    metric_name: str
    score: float
    time_seconds: float
    lower_better: bool
    score_range: str
    metric_mode: str
    target_path: str
    ref_path: Optional[str] = None
    prompt: Optional[str] = None

    @property
    def formatted_score(self) -> str:
        """Format score to 4 decimal places."""
        return f"{self.score:.4f}"

    @property
    def direction_hint(self) -> str:
        """Friendly direction indicator."""
        return "▼ 越低越好" if self.lower_better else "▲ 越高越好"


class EvaluatorEngine:
    """Evaluation engine managing cached metric instances and running inference."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_manager = ModelManager()
        self._loaded_models: Dict[str, Any] = {}
        patch_mplug_owl_compatibility()

    def set_device(self, device: str) -> None:
        """Switch active device and clear model cache if changed."""
        if device != self.device:
            self.device = device
            self.release_vram(force_unload=True)

    def get_or_create_metric(self, metric_name: str) -> Any:
        """Get preloaded metric or instantiate new one."""
        if pyiqa is None:
            raise RuntimeError("pyiqa 未安装或未成功加载。")

        key = f"{metric_name.lower()}_{self.device}"
        if key not in self._loaded_models:
            metric = pyiqa.create_metric(metric_name, device=self.device)
            self._loaded_models[key] = metric
        return self._loaded_models[key]

    def get_vram_usage(self) -> Dict[str, float]:
        """Get GPU VRAM statistics in Megabytes (MB)."""
        if torch is not None and torch.cuda.is_available() and "cuda" in self.device:
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                alloc_b = torch.cuda.memory_allocated()
                res_b = torch.cuda.memory_reserved()
                return {
                    "allocated_mb": alloc_b / (1024 ** 2),
                    "reserved_mb": res_b / (1024 ** 2),
                    "free_mb": free_b / (1024 ** 2),
                    "total_mb": total_b / (1024 ** 2),
                    "available": True,
                }
            except Exception:
                pass
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "free_mb": 0.0,
            "total_mb": 0.0,
            "available": False,
        }

    def unload_metric(self, metric_name: str) -> bool:
        """Unload a specific metric model from memory and clear GPU cache."""
        target_keys = [k for k in list(self._loaded_models.keys()) if k.lower().startswith(metric_name.lower())]
        if not target_keys:
            return False
        for k in target_keys:
            del self._loaded_models[k]
        self.release_vram(force_unload=False)
        return True

    def release_vram(self, force_unload: bool = False, unload_heavy_only: bool = False) -> Dict[str, float]:
        """Free GPU memory cache and optionally unload models from memory.

        Args:
            force_unload: If True, completely deletes all loaded model instances from memory.
            unload_heavy_only: If True, unloads multi-GB large models (such as Q-Align).
        """
        import gc
        if force_unload:
            self._loaded_models.clear()
        elif unload_heavy_only:
            heavy_keywords = ["qalign", "q_align", "mplug", "llm", "vlm"]
            to_remove = [k for k in list(self._loaded_models.keys()) if any(h in k.lower() for h in heavy_keywords)]
            for k in to_remove:
                del self._loaded_models[k]

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        return self.get_vram_usage()

    def evaluate_single(
        self,
        metric_name: str,
        target_path: str,
        ref_path: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> SingleEvalResult:
        """Evaluate a target image against optional reference or prompt."""
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"目标图像文件不存在: {target_path}")

        distribution_metrics = {"fid", "fid_dinov2", "sfid", "inception_score"}
        if metric_name.lower() in distribution_metrics:
            raise ValueError(
                f"算法 '{metric_name}' 属于生成模型集合分布度量算法 (FID / Inception Score)，"
                f"需要对整个图片文件夹与基准数据集进行宏观特征统计，不支持单张图片评估。\n"
                f"如需使用，请前往【📁 文件夹多图批量评测】或【📈 数据集学术基准测试】模块中使用。"
            )

        metric = self.get_or_create_metric(metric_name)
        info = self.model_manager.get_model(metric_name)
        mode = metric.metric_mode if hasattr(metric, "metric_mode") else (info.mode if info else "NR")
        lower_better = metric.lower_better if hasattr(metric, "lower_better") else (info.lower_better if info else False)
        score_range = getattr(metric, "score_range", info.score_range if info else "~0, ~1")

        start_time = time.perf_counter()

        if metric_name.lower() == "clipscore":
            clean_prompt = str(prompt).strip() if prompt else ""
            if not clean_prompt:
                raise ValueError("评估 CLIPScore 需要提供文字描述 (提示词 Prompt)。请在左侧【生图提示词】输入框中输入描述。")
            score_tensor = metric(target_path, caption_list=[clean_prompt])
        elif mode == "FR":
            if not ref_path or not os.path.exists(ref_path):
                raise ValueError(f"指标 '{metric_name}' 属于全参考 (FR) 算法，必须提供有效的参考原图。")
            score_tensor = metric(target_path, ref_path)
        else:
            # NR mode
            score_tensor = metric(target_path)

        # Convert score to float
        if hasattr(score_tensor, "cpu"):
            score = score_tensor.cpu().item()
        elif isinstance(score_tensor, (int, float)):
            score = float(score_tensor)
        else:
            score = float(np.mean(score_tensor))

        elapsed = time.perf_counter() - start_time

        return SingleEvalResult(
            metric_name=metric_name,
            score=score,
            time_seconds=elapsed,
            lower_better=lower_better,
            score_range=str(score_range),
            metric_mode=mode,
            target_path=target_path,
            ref_path=ref_path if mode == "FR" else None,
            prompt=prompt,
        )


def generate_difference_heatmap(target_path: str, ref_path: str, amplify: float = 3.0) -> Image.Image:
    """Generate difference heatmap between target and reference image."""
    img_tar = Image.open(target_path).convert("RGB")
    img_ref = Image.open(ref_path).convert("RGB")

    # Resize reference to match target if dimensions differ
    if img_ref.size != img_tar.size:
        img_ref = img_ref.resize(img_tar.size, Image.Resampling.BILINEAR)

    arr_tar = np.asarray(img_tar, dtype=np.float32)
    arr_ref = np.asarray(img_ref, dtype=np.float32)

    # Absolute difference per channel, averaged
    diff = np.abs(arr_tar - arr_ref)
    diff_mag = np.mean(diff, axis=-1)  # (H, W) in [0, 255]

    # Amplify difference for subtle distortion visualization
    amplified = np.clip(diff_mag * amplify, 0, 255).astype(np.uint8)

    # Apply Jet-like colormap manually without heavy cv2 dependency
    # 0 -> Blue, 64 -> Cyan, 128 -> Yellow, 255 -> Red
    h, w = amplified.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    val = amplified / 255.0
    r = np.clip(1.5 - np.abs(val * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(val * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(val * 4 - 1), 0, 1)

    heatmap[..., 0] = (r * 255).astype(np.uint8)
    heatmap[..., 1] = (g * 255).astype(np.uint8)
    heatmap[..., 2] = (b * 255).astype(np.uint8)

    return Image.fromarray(heatmap, mode="RGB")
