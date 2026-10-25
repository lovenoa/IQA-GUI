"""Regressions for audit F03/F06/F07/F08 and protected installs."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from iqa_gui.core.evaluator import EvaluatorEngine, SingleEvalResult
from iqa_gui.core.worker_threads import SingleEvalWorker


def result(metric="psnr", score=30, target="image", ref=None):
    return SingleEvalResult(metric, score, .01, False, "0, 60", "FR", target, ref)


def test_clipscore_is_similarity_not_an_always_failing_quality_grade():
    r = SingleEvalResult("clipscore", 2.5, .01, False, "0,2.5", "NR", "image")
    assert r.normalized_quality == 1.0
    assert r.grade == "—"
    assert "图文" in r.grade_desc


def test_invalid_prompt_is_rejected_before_loading_weights(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    path.touch()
    engine = EvaluatorEngine("cpu")
    monkeypatch.setattr(engine, "get_or_create_metric", lambda *a: pytest.fail("loaded before validation"))
    with pytest.raises(ValueError, match="文字描述"):
        engine.evaluate_single("clipscore", str(path))


def test_concurrent_load_creates_one_instance(monkeypatch):
    engine = EvaluatorEngine("cpu")
    import iqa_gui.core.evaluator as module
    calls = []
    def create(*args, **kwargs):
        calls.append(1)
        time.sleep(.03)
        return object()
    monkeypatch.setattr(module.pyiqa, "create_metric", create)
    with ThreadPoolExecutor(2) as pool:
        models = list(pool.map(lambda _: engine.get_or_create_metric("psnr"), range(2)))
    assert len(calls) == 1
    assert models[0] is models[1]


class FakeEngine:
    def __init__(self):
        self.model_manager = SimpleNamespace(get_model=lambda m: SimpleNamespace(mode="FR"))
        self.cleaned = False
    def evaluate_single(self, **kwargs):
        return result(target=kwargs["target_path"])
    def unload_metric(self, name):
        pass
    def release_vram(self, **kwargs):
        if kwargs.get("force_unload"):
            self.cleaned = True


def test_business_result_is_emitted_after_cleanup():
    app = QApplication.instance() or QApplication([])
    engine = FakeEngine()
    worker = SingleEvalWorker(engine, "psnr", "image", ref_path="ref")
    observed = []
    worker.result_ready.connect(lambda value: observed.append(engine.cleaned))
    worker.run()
    assert observed == [True]


def test_cancelled_worker_has_distinct_terminal_signal():
    app = QApplication.instance() or QApplication([])
    worker = SingleEvalWorker(FakeEngine(), "psnr", "image")
    terminal, results = [], []
    worker.cancelled.connect(lambda: terminal.append("cancelled"))
    worker.result_ready.connect(results.append)
    worker.cancel()
    worker.run()
    assert terminal == ["cancelled"]
    assert results == []


def test_whole_task_guard_is_released_on_exception():
    engine = EvaluatorEngine("cpu")
    with pytest.raises(RuntimeError):
        with engine.task_guard():
            raise RuntimeError("test")
    with engine.task_guard():
        pass


def test_package_constraints_pin_existing_cuda_build(tmp_path, monkeypatch):
    from iqa_gui.core.dependency_installer import write_constraints
    path = write_constraints(tmp_path / "constraints.txt", {"torch": "2.14.1+cu126", "numpy": "2.2.0"})
    assert "torch==2.14.1+cu126" in path.read_text(encoding="utf-8")
    assert "numpy==2.2.0" in path.read_text(encoding="utf-8")


def test_installer_uses_constraints_and_correct_clip_distribution(monkeypatch):
    from iqa_gui.core import dependency_installer as module
    observed = []
    class Process:
        stdout = []
        def __init__(self, cmd, **kwargs):
            observed.append((cmd, __import__('pathlib').Path(cmd[cmd.index('--constraint')+1]).read_text(encoding='utf-8')))
        def wait(self):
            return 1  # A resolution conflict remains a failed install.
    monkeypatch.setattr(module.subprocess, 'Popen', Process)
    assert module.install_packages(['clip'], on_progress=lambda t: None) == 1
    command, constraints = observed[0]
    assert command[-1] == 'openai-clip'
    assert '--upgrade' not in command
    assert 'torch==' in constraints


def test_json_rejects_invalid_scores_without_overwriting_file(tmp_path):
    from iqa_gui.core.exporter import ResultExporter
    from iqa_gui.core.batch_evaluator import BatchItem
    path = tmp_path / 'report.json'
    path.write_text('keep original', encoding='utf-8')
    item = BatchItem(1, 'image', 'image', scores={'psnr': float('nan')})
    with pytest.raises(ValueError):
        ResultExporter.export_to_json(str(path), [item], ['psnr'])
    assert path.read_text(encoding='utf-8') == 'keep original'


@pytest.mark.integration
@pytest.mark.gpu
def test_real_cuda_psnr_uses_the_selected_gpu(sample_image_dirs):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('No CUDA device is available')
    engine = EvaluatorEngine('cuda')
    dist, ref = sample_image_dirs
    try:
        r = engine.evaluate_single('psnr', str(dist/'I03.bmp'), str(ref/'I03.bmp'))
        assert np.isfinite(r.score) and r.score > 10
        metric = engine.get_or_create_metric('psnr')
        assert str(metric.dummy_param.device).startswith('cuda')
    finally:
        engine.release_vram(force_unload=True)


def test_batch_rejects_collection_metrics_before_preloading():
    from iqa_gui.core.batch_evaluator import BatchEvaluator, BatchItem
    class NeverLoad(FakeEngine):
        def get_or_create_metric(self, name):
            pytest.fail('Unsupported collection metric was preloaded')
    batch = BatchEvaluator(NeverLoad())
    with pytest.raises(ValueError, match='逐张|不支持'):
        batch.run_batch([BatchItem(1, 'image', 'image')], ['fid'])
