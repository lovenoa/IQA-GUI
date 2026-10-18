"""Benchmark dataset evaluation engine for calculating PLCC, SRCC, KRCC, and RMSE."""

from __future__ import annotations

import csv
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import pyiqa
    from pyiqa.metrics import calculate_plcc, calculate_srcc, calculate_krcc
except ImportError:
    torch = None
    pyiqa = None
    calculate_plcc = None
    calculate_srcc = None
    calculate_krcc = None

from iqa_gui.core.evaluator import EvaluatorEngine
from iqa_gui.core.model_manager import ModelManager


@dataclass
class MetricBenchmarkScore:
    """Benchmark performance metrics for a single algorithm on a dataset."""
    metric_name: str
    plcc: float
    srcc: float
    krcc: float
    rmse: float
    num_samples: int
    eval_time_seconds: float


@dataclass
class BenchmarkResult:
    """Result of running benchmark on a dataset with multiple metrics."""
    dataset_name: str
    num_samples: int
    metrics_scores: Dict[str, MetricBenchmarkScore] = field(default_factory=dict)
    detailed_predictions: List[Dict[str, Any]] = field(default_factory=list)


def compute_correlation_metrics(
    preds: Union[List[float], np.ndarray, torch.Tensor],
    targets: Union[List[float], np.ndarray, torch.Tensor],
) -> Tuple[float, float, float, float]:
    """Compute PLCC, SRCC, KRCC, and RMSE between predictions and ground-truth targets."""
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, dtype=torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    if calculate_plcc and calculate_srcc and calculate_krcc:
        plcc = float(calculate_plcc(preds, targets))
        srcc = float(calculate_srcc(preds, targets))
        krcc = float(calculate_krcc(preds, targets))
    else:
        # Fallback numpy implementations if pyiqa metrics not available
        p = preds.numpy()
        t = targets.numpy()
        plcc = float(np.corrcoef(p, t)[0, 1])
        from scipy.stats import spearmanr, kendalltau
        srcc = float(spearmanr(p, t).correlation)
        krcc = float(kendalltau(p, t).correlation)

    rmse = float(torch.sqrt(torch.mean((preds - targets) ** 2)).item())
    return plcc, srcc, krcc, rmse


class BenchmarkEngine:
    """Manages dataset benchmarking across pyiqa algorithms."""

    def __init__(self, evaluator: Optional[EvaluatorEngine] = None, device: str = "cpu"):
        self.evaluator = evaluator or EvaluatorEngine(device=device)
        self.model_manager = ModelManager()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def reset_cancellation(self) -> None:
        self._cancel_event.clear()

    @staticmethod
    def get_supported_standard_datasets() -> Dict[str, Dict]:
        """Return supported standard datasets from pyiqa."""
        if pyiqa:
            try:
                return pyiqa.get_dataset_info()
            except Exception:
                pass
        return {}

    def run_benchmark_from_csv(
        self,
        dataset_name: str,
        image_root: str,
        csv_path: str,
        metrics: List[str],
        image_col: str = "image",
        score_col: str = "mos",
        ref_col: Optional[str] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> BenchmarkResult:
        """Run benchmark on a custom dataset annotated with a CSV file."""
        self.reset_cancellation()
        csv_file = Path(csv_path)
        root = Path(image_root)
        if not csv_file.is_file():
            raise FileNotFoundError(f"数据集标注 CSV 文件不存在: {csv_path}")

        records = []
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_rel = row.get(image_col)
                gt_score = row.get(score_col)
                if img_rel and gt_score:
                    tar_path = (root / img_rel).resolve()
                    ref_path = (root / row[ref_col]).resolve() if (ref_col and row.get(ref_col)) else None
                    if tar_path.exists():
                        records.append({
                            "target_path": str(tar_path),
                            "ref_path": str(ref_path) if ref_path else None,
                            "gt_score": float(gt_score),
                        })

        if not records:
            raise ValueError(f"未在 {csv_path} 中找到有效的图像与分数记录。")

        total_samples = len(records)
        targets_gt = [r["gt_score"] for r in records]
        result = BenchmarkResult(dataset_name=dataset_name, num_samples=total_samples)

        for m_idx, metric_name in enumerate(metrics):
            if self._cancel_event.is_set():
                break

            import time
            start_t = time.perf_counter()
            preds = []

            for idx, rec in enumerate(records):
                if self._cancel_event.is_set():
                    break
                res = self.evaluator.evaluate_single(
                    metric_name=metric_name,
                    target_path=rec["target_path"],
                    ref_path=rec["ref_path"],
                )
                preds.append(res.score)
                rec[metric_name] = res.score

                if on_progress and (idx % 5 == 0 or idx == total_samples - 1):
                    msg = f"评估 {metric_name} ({m_idx+1}/{len(metrics)}): 样本 {idx+1}/{total_samples}"
                    on_progress(msg, idx + 1, total_samples)

            if len(preds) == total_samples:
                plcc, srcc, krcc, rmse = compute_correlation_metrics(preds, targets_gt)
                eval_time = time.perf_counter() - start_t
                result.metrics_scores[metric_name] = MetricBenchmarkScore(
                    metric_name=metric_name,
                    plcc=plcc,
                    srcc=srcc,
                    krcc=krcc,
                    rmse=rmse,
                    num_samples=total_samples,
                    eval_time_seconds=eval_time,
                )

        result.detailed_predictions = records
        return result
