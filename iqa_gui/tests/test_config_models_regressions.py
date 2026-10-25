"""Offline regressions: every writable resource belongs to tmp_path."""

import copy
import json
from pathlib import Path

import pytest

from iqa_gui import config
from iqa_gui.core.model_manager import ModelManager


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "settings.json")
    return config.CONFIG_FILE


def test_loaded_defaults_do_not_share_nested_lists(isolated_settings):
    original = copy.deepcopy(config.DEFAULT_SETTINGS)
    try:
        settings = config.load_settings()
        settings["default_metrics_fr"].append("custom")
        settings["metric_presets"]["经典基础通用"].clear()
        assert config.load_settings()["default_metrics_fr"] == ["psnr", "ssim", "lpips"]
        assert config.load_settings()["metric_presets"]["经典基础通用"]
    finally:
        config.DEFAULT_SETTINGS.clear()
        config.DEFAULT_SETTINGS.update(original)


def test_permission_denied_read_returns_defaults(isolated_settings, monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: (_ for _ in ()).throw(PermissionError("denied")))
    assert config.load_settings()["preferred_device"] == "auto"


def test_failed_serialization_preserves_previous_file(isolated_settings):
    config.save_settings({"custom": "keep me"})
    before = isolated_settings.read_bytes()
    with pytest.raises(OSError, match="settings|配置"):
        config.save_settings({"bad": object()})
    assert isolated_settings.read_bytes() == before


def test_failed_atomic_replace_preserves_previous_file(isolated_settings, monkeypatch):
    config.save_settings({"custom": "keep me"})
    before = isolated_settings.read_bytes()
    monkeypatch.setattr(config.os, "replace", lambda *args: (_ for _ in ()).throw(PermissionError("denied")))
    with pytest.raises(OSError):
        config.save_settings({"custom": "lost"})
    assert isolated_settings.read_bytes() == before
    assert sorted(p.name for p in isolated_settings.parent.iterdir()) == ["settings.json"]


def test_update_merges_latest_disk_values(isolated_settings):
    stale = config.load_settings()
    saved = config.load_settings()
    saved["metric_presets"]["我的组合"] = ["psnr"]
    config.save_settings(saved)
    updated = config.update_settings({"show_tips": False})
    assert updated["metric_presets"]["我的组合"] == ["psnr"]
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["show_tips"] is False
    assert "我的组合" not in stale["metric_presets"]


@pytest.fixture
def manager(tmp_path, monkeypatch):
    mgr = ModelManager(cache_dir=str(tmp_path))
    monkeypatch.setattr(mgr, "get_all_cache_roots", lambda: [tmp_path, tmp_path / "nested"])
    return mgr


@pytest.mark.parametrize("name, filename", [("niqe", "niqe_modelparameters.mat"), ("ilniqe", "ILNIQE_templateModel.mat"),
                                          ("brisque", "brisque_svm_weights.pth"), ("musiq", "musiq_koniq_ckpt-e95806b9.pth")])
def test_statistical_models_require_parameter_files(manager, name, filename):
    assert manager.get_model(name).requires_weights
    assert manager.is_cached(name) is False
    (manager.cache_dir / filename).write_bytes(b"parameters")
    assert manager.is_cached(name) is True


def test_dists_needs_head_and_vgg_backbone(manager):
    assert manager.get_model("dists").default_weight_file == "DISTS_weights-f5e65c96.pth"
    (manager.cache_dir / "DISTS_weights-f5e65c96.pth").write_bytes(b"head")
    assert manager.is_cached("dists") is False
    (manager.cache_dir / "vgg16-397923af.pth").write_bytes(b"backbone")
    assert manager.is_cached("dists") is True


def test_lpips_head_alone_does_not_imply_complete_model(manager):
    (manager.cache_dir / "v0.1_alex.pth").write_bytes(b"head")
    assert manager.is_cached("lpips") is None


def test_cache_inventory_deduplicates_paths_not_basenames(manager):
    (manager.cache_dir / "same.pth").write_bytes(b"12")
    nested = manager.cache_dir / "nested"
    nested.mkdir()
    (nested / "same.pth").write_bytes(b"345")
    (nested / "ignore.txt").write_bytes(b"irrelevant")
    assert len(manager.get_cached_weight_files()) == 2
    assert manager.get_cache_summary() == (2, 5)


def test_workflow_filter_rejects_collections_and_promptless_metrics(manager):
    for workflow in ("single", "batch", "benchmark", "gmad"):
        assert not manager.supports_workflow("fid", workflow, has_prompt=True)
        assert not manager.supports_workflow("clipscore", workflow, has_prompt=False)
    assert manager.supports_workflow("clipscore", "single", has_prompt=True)
    assert not manager.supports_workflow("clipscore", "benchmark", has_prompt=True)
    assert manager.supports_workflow("psnr", "benchmark")


def test_dependency_command_uses_constraint_installer(manager, monkeypatch):
    monkeypatch.setattr(manager, "check_package_installed", lambda name: False)
    command = manager.check_dependencies("musiq").safe_install_command
    assert "-m iqa_gui.core.dependency_installer timm" in command


def test_mirror_rewrites_only_pyiqa_huggingface_downloads(tmp_path, monkeypatch):
    from pyiqa.utils import download_util

    urls = []
    monkeypatch.setattr(download_util, "download_url_to_file", lambda url, *args, **kwargs: urls.append(url))
    monkeypatch.setenv("HF_ENDPOINT", "https://prior.example")
    config.apply_hf_mirror(True, "https://mirror.example")
    download_util.load_file_from_url("https://huggingface.co/owner/repo/resolve/main/a.pth", str(tmp_path))
    download_util.load_file_from_url("https://download.pytorch.org/a.pth", str(tmp_path))
    config.apply_hf_mirror(False)
    download_util.load_file_from_url("https://huggingface.co/owner/repo/resolve/main/b.pth", str(tmp_path))
    assert urls == ["https://mirror.example/owner/repo/resolve/main/a.pth", "https://download.pytorch.org/a.pth", "https://huggingface.co/owner/repo/resolve/main/b.pth"]


def test_disabled_mirror_setting_clears_inherited_endpoint_on_startup(tmp_path, monkeypatch):
    import importlib.util

    folder = tmp_path / ".pyiqa_gui"
    folder.mkdir()
    (folder / "settings.json").write_text('{"hf_mirror_enabled": false}', encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HF_ENDPOINT", "https://inherited-mirror.example")
    spec = importlib.util.spec_from_file_location("isolated_config", config.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert "HF_ENDPOINT" not in config.os.environ


@pytest.fixture
def zoo_view(isolated_settings, manager, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from iqa_gui.ui.views import model_zoo_view

    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(model_zoo_view, "ModelManager", lambda: manager)
    view = model_zoo_view.ModelZooView(object())
    yield view, model_zoo_view
    view.close()
    application.processEvents()


def test_preload_result_keeps_buttons_busy_until_native_finished(zoo_view, monkeypatch):
    from PySide6.QtCore import QObject, Signal

    view, module = zoo_view

    class Worker(QObject):
        progress = Signal(str, int, int)
        result_ready = Signal(list, list)
        finished = Signal()
        error = Signal(str)
        cancelled = Signal()

        def __init__(self, **kwargs):
            super().__init__()

        def start(self):
            pass

    worker = Worker()
    monkeypatch.setattr(module, "PreloadWeightsWorker", lambda **kwargs: worker)
    monkeypatch.setattr(module, "launch_worker", lambda worker, owner: True, raising=False)
    messages = []
    monkeypatch.setattr(module.QMessageBox, "information", lambda *args: messages.append(args))
    view._start_preload(["psnr"])
    worker.result_ready.emit(["psnr"], [])
    assert messages
    assert not view.btn_preload_popular.isEnabled()
    worker.finished.emit()
    assert view.btn_preload_popular.isEnabled()


def test_preload_cleanup_failure_is_visible(zoo_view, monkeypatch):
    from PySide6.QtCore import QObject, Signal

    view, module = zoo_view

    class Worker(QObject):
        progress = Signal(str, int, int)
        result_ready = Signal(list, list)
        finished = Signal()
        error = Signal(str)
        cancelled = Signal()

    worker = Worker()
    monkeypatch.setattr(module, "PreloadWeightsWorker", lambda **kwargs: worker)
    monkeypatch.setattr(module, "launch_worker", lambda worker, owner: True)
    messages = []
    monkeypatch.setattr(module.QMessageBox, "warning", lambda *args: messages.append(args))
    view._start_preload(["psnr"])
    worker.error.emit("cleanup failed")
    assert messages and "cleanup failed" in messages[0][-1]
    assert not view.btn_preload_popular.isEnabled()
    worker.finished.emit()
    assert view.btn_preload_popular.isEnabled()


def test_cache_label_uses_count_and_size_from_same_inventory(zoo_view, manager):
    view, _ = zoo_view
    (manager.cache_dir / "first.pth").write_bytes(b"a" * 1024 * 1024)
    nested = manager.cache_dir / "nested"
    nested.mkdir()
    (nested / "second.mat").write_bytes(b"b" * 2 * 1024 * 1024)
    view._update_cache_summary()
    assert "2 个文件 (3.0 MB)" in view.lbl_cache_summary.text()
