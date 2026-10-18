"""Tests for gMAD and Benchmark engines."""

import os
import tempfile
import csv
import pytest
from iqa_gui.core.gmad_engine import GMADEngine
from iqa_gui.core.benchmark_engine import BenchmarkEngine, compute_correlation_metrics

DIST_DIR = os.path.abspath("IQA-PyTorch/ResultsCalibra/dist_dir")
SAMPLE_IMAGES = [
    os.path.join(DIST_DIR, f)
    for f in ["I03.bmp", "I04.bmp", "I06.bmp"]
    if os.path.exists(os.path.join(DIST_DIR, f))
]


def test_standardize_scores():
    scores = [10.0, 20.0, 30.0]
    norm_higher = GMADEngine.standardize_scores(scores, lower_better=False)
    assert len(norm_higher) == 3
    assert norm_higher[0] < norm_higher[1] < norm_higher[2]

    # Lower better should invert order
    norm_lower = GMADEngine.standardize_scores(scores, lower_better=True)
    assert norm_lower[0] > norm_lower[1] > norm_lower[2]


def test_find_best_pair():
    tie_scores = [0.0, 0.05, 1.0, 1.05]
    target_scores = [1.0, 5.0, 1.1, 1.2]
    # tolerance 0.1 -> index 0 and 1 are tied in tie_scores, target gap is |1.0 - 5.0| = 4.0
    best = GMADEngine.find_best_pair(tie_scores, target_scores, tolerance=0.1)
    assert best is not None
    assert best["target_gap"] == 4.0
    assert (best["first_index"], best["second_index"]) in [(0, 1), (1, 0)]


def test_gmad_execution():
    assert len(SAMPLE_IMAGES) >= 2
    engine = GMADEngine(device="cpu")

    # Run gmad with brisque and niqe
    res = engine.run_gmad(
        image_paths=SAMPLE_IMAGES,
        metric_a="brisque",
        metric_b="niqe",
        tolerance=2.0,  # generous tolerance for 3 images
    )
    assert res.total_images == len(SAMPLE_IMAGES)
    assert len(res.all_scores) == len(SAMPLE_IMAGES)
    assert res.pair_a_tie is not None or res.pair_b_tie is not None


def test_compute_correlation_metrics():
    preds = [1.0, 2.0, 3.0, 4.0, 5.0]
    targets = [1.1, 1.9, 3.1, 3.9, 5.2]
    plcc, srcc, krcc, rmse = compute_correlation_metrics(preds, targets)
    assert plcc > 0.95
    assert srcc > 0.95
    assert krcc > 0.90
    assert rmse < 0.3


def test_benchmark_from_csv():
    engine = BenchmarkEngine(device="cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "test_annotations.csv")
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mos"])
            for img in SAMPLE_IMAGES:
                # Use relative filename from DIST_DIR
                writer.writerow([os.path.basename(img), "65.5"])

        res = engine.run_benchmark_from_csv(
            dataset_name="MiniTest",
            image_root=DIST_DIR,
            csv_path=csv_file,
            metrics=["brisque"],
        )
        assert res.num_samples == len(SAMPLE_IMAGES)
        assert "brisque" in res.metrics_scores
        score = res.metrics_scores["brisque"]
        assert score.num_samples == len(SAMPLE_IMAGES)
