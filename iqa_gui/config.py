"""Global configuration and persistence for IQA-GUI.

Handles mirror endpoints, default device selection, cache directories, and user preferences.
"""

from __future__ import annotations

import json
import os
import copy
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit, urlunsplit

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


_settings_lock = threading.RLock()


class SettingsSaveError(OSError):
    """The previous settings remain intact when saving fails."""


def load_settings() -> Dict[str, Any]:
    """Load settings from JSON file or return defaults."""
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if not isinstance(saved, dict):
                    return settings
                settings.update(saved)
                # Ensure built-in presets are always present
                presets = copy.deepcopy(DEFAULT_SETTINGS["metric_presets"])
                if "metric_presets" in saved and isinstance(saved["metric_presets"], dict):
                    presets.update(saved["metric_presets"])
                settings["metric_presets"] = presets
    except (OSError, ValueError, TypeError):
        pass
    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    """Atomically save settings, raising SettingsSaveError on any failure."""
    temporary = None
    with _settings_lock:
        try:
            payload = json.dumps(settings, indent=2, ensure_ascii=False)
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=CONFIG_FILE.parent,
                                             prefix=".settings-", suffix=".tmp", delete=False) as f:
                temporary = Path(f.name)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, CONFIG_FILE)
        except (OSError, ValueError, TypeError) as exc:
            raise SettingsSaveError(f"无法保存配置 settings ({CONFIG_FILE}): {exc}") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def update_settings(changes: Dict[str, Any]) -> Dict[str, Any]:
    """Merge changed fields into the latest saved settings within this process."""
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")
    with _settings_lock:
        current = load_settings()
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(current.get(key), dict):
                current[key].update(copy.deepcopy(value))
            else:
                current[key] = copy.deepcopy(value)
        save_settings(current)
        return current


def install_pyiqa_mirror_adapter() -> None:
    """Adapt the installed pyiqa direct URL downloader in memory only.

    Call after pyiqa import as well as when toggling. Existing Hugging Face
    clients retain their endpoint and require application restart.
    """
    module = sys.modules.get("pyiqa.utils.download_util")
    if module is None:
        return
    original = module.download_url_to_file
    if getattr(original, "_iqa_mirror_adapter", False):
        return

    def download(url, *args, **kwargs):
        endpoint = os.environ.get("HF_ENDPOINT", "")
        parsed = urlsplit(url)
        mirror = urlsplit(endpoint)
        if parsed.hostname == "huggingface.co" and mirror.scheme in ("http", "https") and mirror.netloc:
            url = urlunsplit((mirror.scheme, mirror.netloc, mirror.path.rstrip("/") + parsed.path,
                             parsed.query, parsed.fragment))
        return original(url, *args, **kwargs)

    download._iqa_mirror_adapter = True
    module.download_url_to_file = download


def apply_hf_mirror(enabled: bool = True, endpoint: str = "https://hf-mirror.com") -> None:
    """Update direct pyiqa HF downloads; restart for imported HF clients."""
    if enabled and endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
    install_pyiqa_mirror_adapter()


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
apply_hf_mirror(_current_settings.get("hf_mirror_enabled", True),
                _current_settings.get("hf_mirror_endpoint", "https://hf-mirror.com"))
