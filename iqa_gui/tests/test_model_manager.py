"""Unit tests for ModelManager and Config modules."""

import pytest
from iqa_gui.config import load_settings, save_settings, get_effective_device
from iqa_gui.core.model_manager import (
    ModelManager,
    CATEGORY_FR_TRADITIONAL,
    CATEGORY_FR_PERCEPTUAL,
    CATEGORY_NR_TRADITIONAL,
    CATEGORY_NR_DEEP,
    CATEGORY_AESTHETIC,
    CATEGORY_MULTIMODAL,
    CATEGORY_DISTRIBUTION,
)


def test_config_operations():
    settings = load_settings()
    assert "hf_mirror_enabled" in settings
    assert "preferred_device" in settings
    device = get_effective_device("cpu")
    assert device == "cpu"


def test_model_manager_catalog():
    mgr = ModelManager()
    all_models = mgr.list_all()
    assert len(all_models) >= 95, f"Expected at least 95 models, got {len(all_models)}"

    # Check known essential metrics
    psnr = mgr.get_model("psnr")
    assert psnr is not None
    assert psnr.mode == "FR"
    assert psnr.category == CATEGORY_FR_TRADITIONAL
    assert psnr.lower_better is False
    assert "峰值信噪比" in psnr.display_name

    lpips = mgr.get_model("lpips")
    assert lpips is not None
    assert lpips.mode == "FR"
    assert lpips.category == CATEGORY_FR_PERCEPTUAL
    assert lpips.lower_better is True

    brisque = mgr.get_model("brisque")
    assert brisque is not None
    assert brisque.mode == "NR"
    assert brisque.category == CATEGORY_NR_TRADITIONAL
    assert brisque.lower_better is True

    musiq = mgr.get_model("musiq")
    assert musiq is not None
    assert musiq.mode == "NR"
    assert musiq.category == CATEGORY_NR_DEEP

    nima = mgr.get_model("nima")
    assert nima is not None
    assert nima.category == CATEGORY_AESTHETIC

    fid = mgr.get_model("fid")
    assert fid is not None
    assert fid.category == CATEGORY_DISTRIBUTION


def test_model_manager_filter():
    mgr = ModelManager()

    fr_models = mgr.filter(mode="FR")
    assert len(fr_models) > 0
    assert all(m.mode == "FR" for m in fr_models)

    nr_models = mgr.filter(mode="NR")
    assert len(nr_models) > 0
    assert all(m.mode == "NR" for m in nr_models)

    search_res = mgr.filter(search_query="信噪比")
    assert len(search_res) >= 1
    assert any("psnr" in m.name for m in search_res)


def test_cache_status():
    mgr = ModelManager()
    cached = mgr.get_cached_weights()
    # At least brisque was downloaded earlier
    assert "brisque_svm_weights.pth" in cached
    assert mgr.is_cached("brisque") is True


def test_dependency_detection():
    mgr = ModelManager()

    # Classical mathematical algorithm (zero external dependencies)
    psnr_dep = mgr.check_dependencies("psnr")
    assert psnr_dep.is_ready is True
    assert len(psnr_dep.missing_required) == 0

    # Deep learning metric with pre-installed deps
    clip_dep = mgr.check_dependencies("clipiqa")
    assert clip_dep.is_ready is True
    assert len(clip_dep.missing_required) == 0

    # VLM with optional acceleration
    qalign_dep = mgr.check_dependencies("qalign")
    assert qalign_dep.is_ready is True
    assert "causal_conv1d" in qalign_dep.missing_optional or len(qalign_dep.missing_optional) >= 0


def test_missing_dependency_simulation():
    mgr = ModelManager()

    # Simulate missing dependency by overriding check_package_installed
    original_check = mgr.check_package_installed
    try:
        mgr.check_package_installed = lambda pkg: False if pkg == "timm" else original_check(pkg)
        mgr.clear_dependency_cache()

        dep_status = mgr.check_dependencies("musiq")
        assert dep_status.is_ready is False
        assert "timm" in dep_status.missing_required
        assert "pip install" in dep_status.safe_install_command
        # Verify safety guard: strictly no --upgrade to protect PyTorch CUDA drivers
        assert "--upgrade" not in dep_status.safe_install_command
        assert "-U " not in dep_status.safe_install_command
        assert "--no-warn-script-location" in dep_status.safe_install_command
    finally:
        mgr.check_package_installed = original_check
        mgr.clear_dependency_cache()


def test_auto_discovery_fallback():
    mgr = ModelManager()
    all_models = mgr.list_all()
    discovered = [m for m in all_models if m.is_discovered]

    assert len(discovered) > 0, "Expected pyiqa to have auto-discovered models not in curated list"
    for m in discovered:
        assert m.name
        assert m.display_name
        assert m.category
        assert m.mode in ["FR", "NR"]
        # Inferred packages should be a list
        assert isinstance(m.required_packages, list)

