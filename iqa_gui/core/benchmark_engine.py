"""Benchmark dataset evaluation engine for calculating PLCC, SRCC, KRCC, and RMSE."""

from __future__ import annotations

import csv
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import pyiqa
except ImportError:
    torch = None
    pyiqa = None

from iqa_gui.core.evaluator import EvaluatorEngine
from iqa_gui.core.model_manager import CATEGORY_DISTRIBUTION, ModelManager


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
    raw_rmse: Optional[float] = None
    metric_lower_better: bool = False


@dataclass
class BenchmarkResult:
    """Result of running benchmark on a dataset with multiple metrics."""
    dataset_name: str
    num_samples: int
    metrics_scores: Dict[str, MetricBenchmarkScore] = field(default_factory=dict)
    detailed_predictions: List[Dict[str, Any]] = field(default_factory=list)
    metric_errors: Dict[str, str] = field(default_factory=dict)
    label_lower_better: bool = False
    cancelled: bool = False


def compute_correlation_metrics(
    preds: Union[List[float], np.ndarray, torch.Tensor],
    targets: Union[List[float], np.ndarray, torch.Tensor],
) -> Tuple[float, float, float, float]:
    """Signed correlations and in-sample, nonnegative-slope affine calibrated RMSE.

    Callers must first orient both arrays so higher means better quality. The
    affine fit uses these same samples and does not estimate generalization.
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau

    def as_array(values):
        if torch is not None and isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=np.float64).reshape(-1)

    p, t = as_array(preds), as_array(targets)
    if len(p) != len(t) or len(p) < 2:
        raise ValueError("相关性统计需要至少两个且数量相同的样本。")
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("预测分数和主观分数必须都是有限数值。")
    if np.ptp(p) == 0 or np.ptp(t) == 0:
        raise ValueError("预测分数或主观分数为常量，无法计算相关性。")
    plcc = float(pearsonr(p, t).statistic)
    srcc = float(spearmanr(p, t).statistic)
    krcc = float(kendalltau(p, t).statistic)
    pc, tc = p - p.mean(), t - t.mean()
    slope = max(0.0, float(np.dot(pc, tc) / np.dot(pc, pc)))
    calibrated = slope * pc + t.mean()
    rmse = float(np.sqrt(np.mean((calibrated - t) ** 2)))
    if not all(math.isfinite(v) for v in (plcc, srcc, krcc, rmse)):
        raise ValueError("统计结果不是有限数值，请检查分数范围和样本变化。")
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
        label_lower_better: bool = False,
        ref_root: Optional[str] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> BenchmarkResult:
        """Run benchmark on a custom dataset annotated with a CSV file."""
        self.reset_cancellation()
        def is_cancelled() -> bool:
            return self._cancel_event.is_set() or bool(cancel_check and cancel_check())
        csv_file = Path(csv_path)
        root = Path(image_root)
        reference_root = Path(ref_root) if ref_root else root
        if not csv_file.is_file():
            raise FileNotFoundError(f"数据集标注 CSV 文件不存在: {csv_path}")

        records = []
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = [image_col, score_col] + ([ref_col] if ref_col else [])
            if not reader.fieldnames or any(c not in reader.fieldnames for c in required):
                raise ValueError(f"CSV 缺少映射列: {', '.join(required)}；请检查列名和 CSV 格式。")
            for line, row in enumerate(reader, 2):
                if is_cancelled():
                    return BenchmarkResult(dataset_name, len(records), detailed_predictions=records,
                                           label_lower_better=label_lower_better, cancelled=True)
                img_rel = (row.get(image_col) or "").strip()
                try:
                    gt_score = float(row.get(score_col) or "")
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"CSV 第 {line} 行主观分数无效。") from exc
                if not math.isfinite(gt_score):
                    raise ValueError(f"CSV 第 {line} 行主观分数必须为有限数值。")
                tar_path = (root / img_rel).resolve()
                if not img_rel or not tar_path.is_file():
                    raise ValueError(f"CSV 第 {line} 行目标图像不存在: {img_rel}")
                ref_rel = (row.get(ref_col) or "").strip() if ref_col else ""
                ref_path = (reference_root / ref_rel).resolve() if ref_rel else None
                if ref_col and (ref_path is None or not ref_path.is_file()):
                    raise ValueError(f"CSV 第 {line} 行参考图像不存在: {ref_rel}")
                records.append({"target_path": str(tar_path), "ref_path": str(ref_path) if ref_path else None,
                                "gt_score": gt_score})

        if len(records) < 2:
            raise ValueError("基准评测需要至少两个有效图像与分数记录。")

        total_samples = len(records)
        targets_gt = [r["gt_score"] for r in records]
        if np.ptp(targets_gt) == 0:
            raise ValueError("主观分数为常量，无法计算相关性。")
        quality_gt = np.asarray(targets_gt) * (-1 if label_lower_better else 1)
        result = BenchmarkResult(dataset_name=dataset_name, num_samples=total_samples,
                                 label_lower_better=label_lower_better)

        for m_idx, metric_name in enumerate(metrics):
            if is_cancelled():
                break

            import time
            start_t = time.perf_counter()
            preds = []
            try:
                info = self.model_manager.get_model(metric_name)
                if info is None:
                    raise ValueError("无法确认算法方向，请选择已登记的算法。")
                if info.category == CATEGORY_DISTRIBUTION:
                    raise ValueError("集合分布算法不支持此逐张基准流程。")
                if metric_name.lower() == "clipscore":
                    raise ValueError("此基准流程没有提示词输入，不支持 CLIPScore。")
                if info.mode == "FR" and not ref_col:
                    raise ValueError("全参考算法需要映射参考图列，并设置参考图根目录。")
                for idx, rec in enumerate(records):
                    if is_cancelled():
                        break
                    res = self.evaluator.evaluate_single(
                        metric_name=metric_name, target_path=rec["target_path"], ref_path=rec["ref_path"])
                    score = float(res.score)
                    if not math.isfinite(score):
                        raise ValueError(f"样本 {idx + 1} 的预测分数不是有限数值。")
                    preds.append(score)
                    rec[metric_name] = score
                    if on_progress and (idx % 5 == 0 or idx == total_samples - 1):
                        msg = f"评估 {metric_name} ({m_idx+1}/{len(metrics)}): 样本 {idx+1}/{total_samples}"
                        on_progress(msg, idx + 1, total_samples)
                if len(preds) != total_samples:
                    break
                quality_preds = np.asarray(preds) * (-1 if info.lower_better else 1)
                plcc, srcc, krcc, rmse = compute_correlation_metrics(quality_preds, quality_gt)
                eval_time = time.perf_counter() - start_t
                result.metrics_scores[metric_name] = MetricBenchmarkScore(
                    metric_name=metric_name,
                    plcc=plcc,
                    srcc=srcc,
                    krcc=krcc,
                    rmse=rmse,
                    num_samples=total_samples,
                    eval_time_seconds=eval_time,
                    raw_rmse=float(np.sqrt(np.mean((np.asarray(preds) - targets_gt) ** 2))),
                    metric_lower_better=info.lower_better,
                )
            except Exception as exc:
                result.metric_errors[metric_name] = str(exc)

        result.detailed_predictions = records
        result.cancelled = is_cancelled()
        return result
