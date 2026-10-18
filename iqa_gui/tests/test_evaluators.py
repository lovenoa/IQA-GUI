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

