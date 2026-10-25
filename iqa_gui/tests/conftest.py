"""Keep regression tests offline and away from personal settings and logs."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", default=False,
                     help="Run tests using already cached BRISQUE/NIQE weights (no downloads).")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: uses real locally cached model weights")
    config.addinivalue_line("markers", "gpu: requires a working local CUDA device")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip = pytest.mark.skip(reason="Use --run-integration for locally cached model tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def sample_image_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("iqa-samples")
    dist, ref = root / "dist", root / "ref"
    dist.mkdir()
    ref.mkdir()
    rng = np.random.default_rng(314)
    for i, name in enumerate(("I03.bmp", "I04.bmp", "I06.bmp")):
        clean = rng.integers(24, 232, (256, 256, 3), dtype=np.uint8)
        distorted = np.clip(clean.astype(float) + rng.normal(0, (i + 1) * 5, clean.shape), 0, 255).astype(np.uint8)
        Image.fromarray(clean).save(ref / name)
        Image.fromarray(distorted).save(dist / name)
    return dist, ref


@pytest.fixture(autouse=True)
def isolated_user_state(tmp_path, monkeypatch, request, sample_image_dirs):
    from iqa_gui import config
    from iqa_gui.core import log_manager

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "settings")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "settings" / "settings.json")
    monkeypatch.setattr(log_manager, "LOG_FILE_PATH", str(tmp_path / "runtime.log"))
    # Existing test module constants are now generated, tracked-code fixtures.
    dist, ref = sample_image_dirs
    replacements = {
        "DIST_DIR": str(dist), "REF_DIR": str(ref),
        "SAMPLE_DIST": str(dist / "I03.bmp"), "SAMPLE_REF": str(ref / "I03.bmp"),
        "SAMPLE_IMAGES": [str(dist / n) for n in ("I03.bmp", "I04.bmp", "I06.bmp")],
    }
    for name, value in replacements.items():
        if hasattr(request.module, name):
            monkeypatch.setattr(request.module, name, value)
    # New tests may mock metric construction, but no test may fetch a weight.
    import pyiqa.utils.download_util as download_util
    def offline_download(*args, **kwargs):
        pytest.fail("Tests may only use cached weights; network download was requested")
    monkeypatch.setattr(download_util, "download_url_to_file", offline_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    if "integration" in request.keywords:
        from iqa_gui.core.model_manager import ModelManager
        manager = ModelManager()
        required = [] if "gpu" in request.keywords else (["brisque", "niqe"] if request.node.name == "test_gmad_execution" else ["brisque"])
        if any(manager.is_cached(name) is not True for name in required):
            pytest.skip("Integration requires existing local BRISQUE/NIQE weights")
