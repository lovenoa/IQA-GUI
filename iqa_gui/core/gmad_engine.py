"""gMAD (generalized Minimum/Maximum Added Difference) candidate-pool competition engine."""

from __future__ import annotations

import math
import os
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import pyiqa
    from pyiqa.models.inference_model import DEFAULT_CONFIGS
except ImportError:
    torch = None
    pyiqa = None
    DEFAULT_CONFIGS = {}

from iqa_gui.core.evaluator import EvaluatorEngine
from iqa_gui.core.model_manager import ModelManager


@dataclass
class GMADImageScore:
    """Score info for one image in the candidate pool."""
    image_path: str
    filename: str
    raw_score_a: float
    raw_score_b: float
    norm_score_a: float
    norm_score_b: float


@dataclass
class GMADPairResult:
    """Extreme differentiating image pair found by gMAD."""
    image_1: GMADImageScore
    image_2: GMADImageScore
    tie_metric: str
    target_metric: str
    tie_gap: float
    target_gap: float


@dataclass
class GMADResult:
    """Full outcome of gMAD competition between two metrics."""
    metric_a: str
    metric_b: str
    tolerance: float
    total_images: int
    # Pair where Metric A ties but Metric B differs most
    pair_a_tie: Optional[GMADPairResult] = None
    # Pair where Metric B ties but Metric A differs most
    pair_b_tie: Optional[GMADPairResult] = None
    all_scores: List[GMADImageScore] = None


class GMADEngine:
    """Engine executing gMAD pairwise competition on candidate pools."""

    def __init__(self, evaluator: Optional[EvaluatorEngine] = None, device: str = "cpu"):
        self.evaluator = evaluator or EvaluatorEngine(device=device)
        self.model_manager = ModelManager()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def reset_cancellation(self) -> None:
        self._cancel_event.clear()

    @staticmethod
    def standardize_scores(scores: List[float], lower_better: bool) -> List[float]:
        """Orient higher quality as better and compute z-scores."""
        oriented = [-s if lower_better else s for s in scores]
        mean = float(np.mean(oriented))
        std = float(np.std(oriented))
        if std <= 1e-12:
            return [0.0] * len(oriented)
        return [(s - mean) / std for s in oriented]

    @staticmethod
    def find_best_pair(
        tie_scores: List[float],
        target_scores: List[float],
        tolerance: float,
    ) -> Optional[Dict[str, Any]]:
        """Find pair with maximum target_score gap within tie_tolerance."""
        n = len(tie_scores)
        if n < 2:
            return None

        ordered = sorted(range(n), key=tie_scores.__getitem__)
        minimum = deque()
        maximum = deque()
        right = 1
        best = None

        for left in range(n - 1):
            while minimum and minimum[0][0] <= left:
                minimum.popleft()
            while maximum and maximum[0][0] <= left:
                maximum.popleft()

            right = max(right, left + 1)
            while right < n and (tie_scores[ordered[right]] - tie_scores[ordered[left]] <= tolerance):
                val = target_scores[ordered[right]]
                while minimum and minimum[-1][1] >= val:
                    minimum.pop()
                minimum.append((right, val))
                while maximum and maximum[-1][1] <= val:
                    maximum.pop()
                maximum.append((right, val))
                right += 1

            if not minimum:
                continue

            first_idx = ordered[left]
            for extreme_pos, extreme_val in (minimum[0], maximum[0]):
                gap = abs(target_scores[first_idx] - extreme_val)
                if best is None or gap > best["target_gap"]:
                    second_idx = ordered[extreme_pos]
                    tie_gap = abs(tie_scores[first_idx] - tie_scores[second_idx])
                    best = {
                        "first_index": first_idx,
                        "second_index": second_idx,
                        "tie_gap": tie_gap,
                        "target_gap": gap,
                    }

        return best

    def run_gmad(
        self,
        image_paths: List[str],
        metric_a: str,
        metric_b: str,
        tolerance: float = 0.1,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> GMADResult:
        """Run complete gMAD analysis over candidate images."""
        self.reset_cancellation()
        def cancelled():
            return self._cancel_event.is_set() or bool(cancel_check and cancel_check())
        total = len(image_paths)
        if total < 2:
            raise ValueError("gMAD 对抗分析至少需要 2 张候选图片。")

        info_a = self.model_manager.get_model(metric_a)
        info_b = self.model_manager.get_model(metric_b)
        lower_a = info_a.lower_better if info_a else False
        lower_b = info_b.lower_better if info_b else False

        raw_a: List[float] = []
        raw_b: List[float] = []

        # 1. Score with Metric A
        if on_progress:
            on_progress(f"正在计算指标 A: {metric_a}...", 0, total)
        for idx, img in enumerate(image_paths):
            if cancelled():
                raise InterruptedError("gMAD 运算已取消。")
            res_a = self.evaluator.evaluate_single(metric_a, img)
            raw_a.append(res_a.score)
            if on_progress and (idx % 5 == 0 or idx == total - 1):
                on_progress(f"计算指标 A ({metric_a}): {idx+1}/{total}", idx + 1, total)

        # 2. Score with Metric B
        if on_progress:
            on_progress(f"正在计算指标 B: {metric_b}...", 0, total)
        for idx, img in enumerate(image_paths):
            if cancelled():
                raise InterruptedError("gMAD 运算已取消。")
            res_b = self.evaluator.evaluate_single(metric_b, img)
            raw_b.append(res_b.score)
            if on_progress and (idx % 5 == 0 or idx == total - 1):
                on_progress(f"计算指标 B ({metric_b}): {idx+1}/{total}", idx + 1, total)

        # 3. Standardize
        norm_a = self.standardize_scores(raw_a, lower_a)
        norm_b = self.standardize_scores(raw_b, lower_b)

        image_scores = [
            GMADImageScore(
                image_path=img,
                filename=os.path.basename(img),
                raw_score_a=raw_a[i],
                raw_score_b=raw_b[i],
                norm_score_a=norm_a[i],
                norm_score_b=norm_b[i],
            )
            for i, img in enumerate(image_paths)
        ]

        # 4. Find pairs
        # Case 1: Metric A ties, Metric B differentiates
        best_a_tie = self.find_best_pair(norm_a, norm_b, tolerance)
        pair_a: Optional[GMADPairResult] = None
        if best_a_tie:
            pair_a = GMADPairResult(
                image_1=image_scores[best_a_tie["first_index"]],
                image_2=image_scores[best_a_tie["second_index"]],
                tie_metric=metric_a,
                target_metric=metric_b,
                tie_gap=best_a_tie["tie_gap"],
                target_gap=best_a_tie["target_gap"],
            )

        # Case 2: Metric B ties, Metric A differentiates
        best_b_tie = self.find_best_pair(norm_b, norm_a, tolerance)
        pair_b: Optional[GMADPairResult] = None
        if best_b_tie:
            pair_b = GMADPairResult(
                image_1=image_scores[best_b_tie["first_index"]],
                image_2=image_scores[best_b_tie["second_index"]],
                tie_metric=metric_b,
                target_metric=metric_a,
                tie_gap=best_b_tie["tie_gap"],
                target_gap=best_b_tie["target_gap"],
            )

        return GMADResult(
            metric_a=metric_a,
            metric_b=metric_b,
            tolerance=tolerance,
            total_images=total,
            pair_a_tie=pair_a,
            pair_b_tie=pair_b,
            all_scores=image_scores,
        )
