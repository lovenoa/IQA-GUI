"""Global configuration and persistence for IQA-GUI.

Handles mirror endpoints, default device selection, cache directories, and user preferences.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path.home() / ".pyiqa_gui"
CONFIG_FILE = CONFIG_DIR / "settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "hf_mirror_enabled": True,
    "hf_mirror_endpoint": "https://hf-mirror.com",
    "official_hf_endpoint": "https://huggingface.co",
    "preferred_device": "auto",  # 'auto', 'cuda', 'cpu'
    "cache_dir": str(Path.home() / ".cache" / "torch" / "hub" / "pyiqa"),
    "last_target_dir": "",
    "last_ref_dir": "",
    "default_metrics_fr": ["psnr", "ssim", "lpips"],
    "default_metrics_nr": ["brisque", "niqe", "musiq"],
    "theme": "light",
    "show_tips": True,
    "auto_release_vram": True,
    "metric_presets": {
        "综合感知与美学": ["laion_aes", "clipiqa", "musiq", "clipscore", "maniqa", "qalign"],
        "经典基础通用": ["psnr", "ssim", "lpips", "brisque", "niqe"],
    },
}


def load_settings() -> Dict[str, Any]:
    """Load settings from JSON file or return defaults."""
    settings = DEFAULT_SETTINGS.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                settings.update(saved)
                # Ensure built-in presets are always present
                presets = DEFAULT_SETTINGS["metric_presets"].copy()
                if "metric_presets" in saved and isinstance(saved["metric_presets"], dict):
                    presets.update(saved["metric_presets"])
                settings["metric_presets"] = presets
        except Exception:
            pass
    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    """Save user settings to JSON file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save settings: {e}")


def apply_hf_mirror(enabled: bool = True, endpoint: str = "https://hf-mirror.com") -> None:
    """Apply or remove HuggingFace mirror environment variable."""
    if enabled and endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]


def get_effective_device(preferred: str = "auto") -> str:
    """Determine effective PyTorch device ('cuda' or 'cpu')."""
    import torch

    if preferred == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif preferred == "cpu":
        return "cpu"
    else:  # 'auto'
        return "cuda" if torch.cuda.is_available() else "cpu"


# Initialize mirror settings upon module import
_current_settings = load_settings()
if _current_settings.get("hf_mirror_enabled", True):
    apply_hf_mirror(True, _current_settings.get("hf_mirror_endpoint", "https://hf-mirror.com"))
