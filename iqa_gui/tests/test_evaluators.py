"""Tests for single evaluator, difference heatmap, batch evaluator, and exporter."""

import os
import tempfile
import pytest
from iqa_gui.core.evaluator import EvaluatorEngine, generate_difference_heatmap
from iqa_gui.core.batch_evaluator import BatchEvaluator
from iqa_gui.core.exporter import ResultExporter

DIST_DIR = os.path.abspath("IQA-PyTorch/ResultsCalibra/dist_dir")
REF_DIR = os.path.abspath("IQA-PyTorch/ResultsCalibra/ref_dir")
SAMPLE_DIST = os.path.join(DIST_DIR, "I03.bmp")
SAMPLE_REF = os.path.join(REF_DIR, "I03.bmp")


def test_single_evaluator():
    engine = EvaluatorEngine(device="cpu")

    # FR metric: PSNR
    res_psnr = engine.evaluate_single("psnr", SAMPLE_DIST, SAMPLE_REF)
    assert res_psnr.metric_name == "psnr"
    assert res_psnr.metric_mode == "FR"
    assert res_psnr.score > 0
    assert res_psnr.lower_better is False
    assert res_psnr.time_seconds > 0

    # NR metric: BRISQUE
    res_brisque = engine.evaluate_single("brisque", SAMPLE_DIST)
    assert res_brisque.metric_name == "brisque"
    assert res_brisque.metric_mode == "NR"
    assert res_brisque.score > 0
    assert res_brisque.lower_better is True


def test_missing_ref_error():
    engine = EvaluatorEngine(device="cpu")
    # Evaluating FR metric without reference should raise ValueError
    with pytest.raises(ValueError, match="必须提供有效的参考原图"):
        engine.evaluate_single("psnr", SAMPLE_DIST, ref_path=None)


def test_difference_heatmap():
    heatmap = generate_difference_heatmap(SAMPLE_DIST, SAMPLE_REF)
    assert heatmap is not None
    assert heatmap.size[0] > 0
    assert heatmap.size[1] > 0
    assert heatmap.mode == "RGB"


def test_batch_evaluator_and_exporter():
    batch = BatchEvaluator(device="cpu")
    items = batch.prepare_batch(DIST_DIR, REF_DIR, pair_mode="exact_name")
    assert len(items) > 0
    assert items[0].ref_path is not None

    # Limit to first 2 items for rapid test
    sub_items = items[:2]
    progress_calls = []

    def on_prog(cur, tot, item, speed):
        progress_calls.append((cur, tot))

    summary = batch.run_batch(sub_items, metrics=["psnr"], on_progress=on_prog)
    assert summary.total_images == 2
    assert summary.completed_images == 2
    assert "psnr" in summary.stats
    assert len(progress_calls) == 2

    # Test exporters
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "report.csv")
        json_path = os.path.join(tmpdir, "report.json")

        ResultExporter.export_to_csv(csv_path, sub_items, ["psnr"], summary)
        assert os.path.exists(csv_path)
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            content = f.read()
            assert "PSNR" in content.upper()
            assert "平均值" in content

        ResultExporter.export_to_json(json_path, sub_items, ["psnr"], summary)
        assert os.path.exists(json_path)


def test_vram_release_and_management():
    engine = EvaluatorEngine(device="cpu")
    # Simulate loading a dummy model in cache
    engine._loaded_models["dummy_model_cpu"] = object()
    assert "dummy_model_cpu" in engine._loaded_models

    usage_before = engine.get_vram_usage()
    assert isinstance(usage_before, dict)
    assert "allocated_mb" in usage_before

    # Test unload specific metric
    unloaded = engine.unload_metric("dummy_model")
    assert unloaded is True
    assert "dummy_model_cpu" not in engine._loaded_models

    # Test force unload all
    engine._loaded_models["test_metric_1"] = object()
    engine._loaded_models["test_metric_2"] = object()
    res = engine.release_vram(force_unload=True)
    assert len(engine._loaded_models) == 0
    assert isinstance(res, dict)


def test_clipscore_prompt_handling():
    engine = EvaluatorEngine(device="cpu")

    # When prompt is empty or None, it must raise ValueError with friendly guidance
    with pytest.raises(ValueError, match="评估 CLIPScore 需要提供文字描述"):
        engine.evaluate_single("clipscore", SAMPLE_DIST, prompt="")

    with pytest.raises(ValueError, match="评估 CLIPScore 需要提供文字描述"):
        engine.evaluate_single("clipscore", SAMPLE_DIST, prompt=None)


def test_mplug_owl_patch():
    from iqa_gui.core.evaluator import patch_mplug_owl_compatibility
    import torch

    patched = patch_mplug_owl_compatibility()
    try:
        import pyiqa.archs.q_align.visual_encoder as ve
        if hasattr(ve, "MplugOwlVisualAbstractorModel"):
            cls = ve.MplugOwlVisualAbstractorModel
            assert hasattr(cls, "invert_attention_mask")

            # Test that invert_attention_mask behaves properly
            dummy_self = type("Dummy", (), {"dtype": torch.float32})()
            mask_2d = torch.ones((1, 10))
            inverted = cls.invert_attention_mask(dummy_self, mask_2d)
            assert inverted.shape == (1, 1, 1, 10)
            assert (inverted == 0.0).all()
    except ImportError:
        pass


def test_distribution_metrics_in_single_eval():
    engine = EvaluatorEngine(device="cpu")
    for m in ["fid", "fid_dinov2", "sfid", "inception_score"]:
        with pytest.raises(ValueError, match="生成模型集合分布度量算法"):
            engine.evaluate_single(m, SAMPLE_DIST)


def test_step_by_step_vram_release():
    from iqa_gui.core.worker_threads import SingleEvalWorker
    engine = EvaluatorEngine(device="cpu")

    # Run SingleEvalWorker with auto_release_vram=True across two metrics
    worker = SingleEvalWorker(
        evaluator=engine,
        metrics=["psnr", "brisque"],
        target_path=SAMPLE_DIST,
        ref_path=SAMPLE_REF,
        auto_release_vram=True,
    )
    worker.run()
    # When auto_release_vram=True, all metrics must be unloaded step-by-step and at the end
    assert len(engine._loaded_models) == 0


def test_score_range_and_normalized_quality():
    from iqa_gui.core.evaluator import parse_score_range, compute_normalized_quality, calculate_grade, SingleEvalResult

    # 1. Test range parsing
    assert parse_score_range("~0, ~1") == (0.0, 1.0)
    assert parse_score_range("0, 100") == (0.0, 100.0)
    assert parse_score_range("[-1, 1]") == (-1.0, 1.0)
    assert parse_score_range("~0, ~40") == (0.0, 40.0)
    assert parse_score_range("invalid") is None
    assert parse_score_range("~0, ~inf") is None
    assert parse_score_range(None) is None

    # 2. Test normalized quality computation
    q_high = compute_normalized_quality(score=38.0, score_range_str="None", lower_better=False, metric_name="psnr")
    assert 0.0 <= q_high <= 1.0
    assert q_high > 0.6  # High quality PSNR

    q_low_better = compute_normalized_quality(score=0.08, score_range_str="[0, 1]", lower_better=True, metric_name="lpips")
    assert q_low_better > 0.8  # Low LPIPS means high quality

    # 3. Test grade calculation
    grade_s, desc_s = calculate_grade(score=7.5, normalized_quality=0.9, metric_name="laion_aes")
    assert grade_s == "S"
    assert "美学" in desc_s

    grade_c, _ = calculate_grade(score=4.0, normalized_quality=0.3, metric_name="laion_aes")
    assert grade_c == "C"

    # 4. Test SingleEvalResult post-init automatic evaluation
    res = SingleEvalResult(
        metric_name="maniqa",
        metric_mode="NR",
        score=0.72,
        lower_better=False,
        score_range="[0, 1]",
        time_seconds=0.05,
        target_path=SAMPLE_DIST,
    )
    assert res.grade == "S"
    assert res.normalized_quality > 0.5
    assert len(res.grade_desc) > 0


def test_multi_target_single_eval_worker():
    from iqa_gui.core.worker_threads import SingleEvalWorker
    engine = EvaluatorEngine(device="cpu")

    multi_results = {}

    def on_multi_finished(eval_map):
        multi_results.update(eval_map)

    worker = SingleEvalWorker(
        evaluator=engine,
        metrics=["brisque"],
        target_path=[SAMPLE_DIST, SAMPLE_DIST],
        auto_release_vram=True,
    )
    worker.multi_finished.connect(on_multi_finished)
    worker.run()

    assert len(multi_results) == 1  # Deduplicated path key
    assert SAMPLE_DIST in multi_results
    assert len(multi_results[SAMPLE_DIST]) == 1
    assert multi_results[SAMPLE_DIST][0].metric_name == "brisque"


def test_variable_metrics_diagnostic_warning():
    from iqa_gui.core.evaluator import is_variable_metric, SingleEvalResult

    # Verify classification of stochastic vs deterministic algorithms
    assert is_variable_metric("qalign") is True
    assert is_variable_metric("qalign_8bit") is True
    assert is_variable_metric("mplug_owl") is True
    assert is_variable_metric("psnr") is False
    assert is_variable_metric("ssim") is False

    # Verify that SingleEvalResult diagnostic description includes the warning
    res_qalign = SingleEvalResult(
        metric_name="qalign_8bit",
        metric_mode="NR",
        score=4.2,
        lower_better=False,
        score_range="[1, 5]",
        time_seconds=0.1,
        target_path=SAMPLE_DIST,
    )
    assert "易浮动" in res_qalign.grade_desc


def test_multi_round_single_eval_worker():
    from iqa_gui.core.worker_threads import SingleEvalWorker
    engine = EvaluatorEngine(device="cpu")

    rounds_results = {}

    def on_rounds_finished(rounds_map):
        rounds_results.update(rounds_map)

    worker = SingleEvalWorker(
        evaluator=engine,
        metrics=["brisque"],
        target_path=SAMPLE_DIST,
        rounds=3,
        auto_release_vram=True,
    )
    worker.rounds_finished.connect(on_rounds_finished)
    worker.run()

    assert len(rounds_results) == 3
    assert 1 in rounds_results
    assert 2 in rounds_results
    assert 3 in rounds_results
    assert SAMPLE_DIST in rounds_results[1]
    assert len(rounds_results[1][SAMPLE_DIST]) == 1



