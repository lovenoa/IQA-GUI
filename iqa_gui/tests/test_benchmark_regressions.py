"""Offline regressions for benchmark statistics, CSV contracts and UI lifecycle."""

import csv
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from iqa_gui.core.benchmark_engine import BenchmarkEngine, compute_correlation_metrics


class ScoreEvaluator:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def evaluate_single(self, metric_name, target_path, ref_path=None):
        self.calls.append((metric_name, target_path, ref_path))
        idx = int(Path(target_path).stem)
        return SimpleNamespace(score=self.values[metric_name][idx])


def dataset(tmp_path, labels):
    for i in range(len(labels)):
        (tmp_path / f"{i}.png").touch()
    path = tmp_path / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "mos"])
        writer.writerows((f"{i}.png", label) for i, label in enumerate(labels))
    return str(path)


def test_rmse_calibrates_units_on_same_samples_without_hiding_inverse_order():
    good = compute_correlation_metrics([.1, .2, .3, .4], [10, 20, 30, 40])
    assert good[:3] == pytest.approx([1, 1, 1])
    assert good[3] == pytest.approx(0, abs=1e-10)
    inverse = compute_correlation_metrics([4, 3, 2, 1], [10, 20, 30, 40])
    assert inverse[:3] == pytest.approx([-1, -1, -1])
    # A negative fit slope would disguise the reversed quality ranking.
    assert inverse[3] == pytest.approx(125 ** .5)


@pytest.mark.parametrize("preds,targets", [
    ([1], [2]), ([1, 2], [2]), ([1, np.nan], [1, 2]),
    ([1, 2], [1, np.inf]), ([1, 1], [1, 2]), ([1, 2], [1, 1]),
])
def test_undefined_statistics_raise_instead_of_succeeding_with_nan(preds, targets):
    with pytest.raises(ValueError):
        compute_correlation_metrics(preds, targets)


@pytest.mark.parametrize("lower,labels", [(False, [40, 30, 20, 10]), (True, [10, 20, 30, 40])])
def test_metric_and_label_directions_are_both_applied(tmp_path, lower, labels):
    evaluator = ScoreEvaluator({"niqe": [1, 2, 3, 4]})
    result = BenchmarkEngine(evaluator).run_benchmark_from_csv(
        "direction", str(tmp_path), dataset(tmp_path, labels), ["niqe"],
        label_lower_better=lower,
    )
    score = result.metrics_scores["niqe"]
    assert [score.plcc, score.srcc, score.krcc] == pytest.approx([1, 1, 1])
    assert score.rmse == pytest.approx(0, abs=1e-10)
    assert score.raw_rmse > 0
    assert result.detailed_predictions[0]["niqe"] == 1


@pytest.mark.parametrize("labels", [[1], [1, 1], [1, "nan"], [1, "inf"], [1, "invalid"]])
def test_invalid_csv_labels_fail_before_model_execution(tmp_path, labels):
    evaluator = ScoreEvaluator({"niqe": [1, 2]})
    with pytest.raises(ValueError):
        BenchmarkEngine(evaluator).run_benchmark_from_csv(
            "invalid", str(tmp_path), dataset(tmp_path, labels), ["niqe"])
    assert evaluator.calls == []


def test_bad_prediction_is_recorded_per_algorithm_without_nan_scorecard(tmp_path):
    evaluator = ScoreEvaluator({"niqe": [1, np.nan, 3], "brisque": [3, 2, 1]})
    result = BenchmarkEngine(evaluator).run_benchmark_from_csv(
        "partial", str(tmp_path), dataset(tmp_path, [10, 20, 30]), ["niqe", "brisque"])
    assert set(result.metrics_scores) == {"brisque"}
    assert "niqe" in result.metric_errors


def test_reference_paths_use_independent_root_and_mapped_columns(tmp_path):
    target = tmp_path / "target"
    reference = tmp_path / "reference"
    target.mkdir()
    reference.mkdir()
    for i in range(2):
        (target / f"{i}.png").touch()
        (reference / f"ref{i}.png").touch()
    path = tmp_path / "fr.csv"
    path.write_text("dist,label,original\n0.png,10,ref0.png\n1.png,20,ref1.png\n", encoding="utf-8")
    evaluator = ScoreEvaluator({"psnr": [10, 20]})
    result = BenchmarkEngine(evaluator).run_benchmark_from_csv(
        "FR", str(target), str(path), ["psnr"], image_col="dist", score_col="label",
        ref_col="original", ref_root=str(reference))
    assert result.metrics_scores["psnr"].plcc == pytest.approx(1)
    assert evaluator.calls[0][2] == str(reference / "ref0.png")


@pytest.mark.parametrize("metric", ["clipscore", "fid"])
def test_unavailable_workflow_metric_never_runs(tmp_path, metric):
    evaluator = ScoreEvaluator({metric: [1, 2]})
    result = BenchmarkEngine(evaluator).run_benchmark_from_csv(
        "unsupported", str(tmp_path), dataset(tmp_path, [10, 20]), [metric])
    assert result.metrics_scores == {}
    assert metric in result.metric_errors
    assert evaluator.calls == []


@pytest.fixture
def benchmark_view(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from iqa_gui.ui.views import benchmark_view as module

    application = QApplication.instance() or QApplication([])
    view = module.BenchmarkView(ScoreEvaluator({}))
    yield view, module
    view.close()
    application.processEvents()


def test_benchmark_gui_defaults_exclude_prompt_and_collection_algorithms(benchmark_view):
    view, _ = benchmark_view
    assert view.metric_selector.get_selected()
    assert set(view.metric_selector.get_selected()).isdisjoint({"clipscore", "fid", "sfid", "inception_score"})


def test_gui_passes_mapping_settings_and_waits_for_native_thread_end(benchmark_view, monkeypatch, tmp_path):
    from PySide6.QtCore import QObject, Signal
    from iqa_gui.core.benchmark_engine import BenchmarkResult

    view, module = benchmark_view
    class Worker(QObject):
        progress = Signal(str, int, int)
        result_ready = Signal(object)
        finished = Signal()
        error = Signal(str)
        cancelled = Signal()
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.running = True
            self.cancel_requested = False
        def isRunning(self):
            return self.running
        def cancel(self):
            self.cancel_requested = True

    monkeypatch.setattr(module, "BenchmarkWorker", Worker)
    monkeypatch.setattr(module, "launch_worker", lambda worker, owner: True, raising=False)
    monkeypatch.setattr(module, "load_settings", lambda: {"auto_release_vram": False}, raising=False)
    csv_path = dataset(tmp_path, [10, 20])
    view.edit_root.setText(str(tmp_path))
    view.edit_csv.setText(csv_path)
    view.edit_image_col.setText("dist")
    view.edit_score_col.setText("label")
    view.edit_ref_col.setText("original")
    view.edit_ref_root.setText(str(tmp_path))
    view.combo_label_direction.setCurrentIndex(1)
    view.run_benchmark()
    worker = view.active_worker
    assert worker.kwargs["image_col"] == "dist"
    assert worker.kwargs["score_col"] == "label"
    assert worker.kwargs["ref_col"] == "original"
    assert worker.kwargs["ref_root"] == str(tmp_path)
    assert worker.kwargs["label_lower_better"] is True
    assert worker.kwargs["auto_release_vram"] is False
    worker.result_ready.emit(BenchmarkResult("test", 2))
    assert not view.btn_run.isEnabled()
    view.btn_cancel.click()
    assert worker.cancel_requested
    worker.running = False
    worker.finished.emit()
    assert view.btn_run.isEnabled()
    assert not view.btn_cancel.isEnabled()


def test_dataset_preset_sets_real_direction_and_reference_root(benchmark_view, monkeypatch):
    view, module = benchmark_view
    monkeypatch.setattr(module.BenchmarkEngine, "get_supported_standard_datasets", staticmethod(lambda: {
        "csiq": {"dataroot_target": "dist", "dataroot_ref": "ref", "meta_info_file": "labels.csv", "lower_better": True}}))
    view.combo_dataset.setCurrentIndex(view.combo_dataset.findData("csiq"))
    assert view.edit_root.text() == "dist"
    assert view.edit_ref_root.text() == "ref"
    assert view.combo_label_direction.currentData() is True
    assert "列" in view.lbl_preset_notice.text()


def test_external_cancellation_survives_engine_reset(tmp_path):
    evaluator = ScoreEvaluator({"niqe": [1, 2]})
    result = BenchmarkEngine(evaluator).run_benchmark_from_csv(
        "cancel", str(tmp_path), dataset(tmp_path, [10, 20]), ["niqe"], cancel_check=lambda: True)
    assert result.cancelled
    assert not result.metrics_scores
    assert evaluator.calls == []
