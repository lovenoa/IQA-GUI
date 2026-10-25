"""Batch evaluation engine for processing image directories with progress reporting."""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np

from iqa_gui.core.evaluator import EvaluatorEngine
from iqa_gui.core.model_manager import ModelManager

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class BatchItem:
    """Represents a single evaluation item in the batch."""
    index: int
    target_path: str
    target_filename: str
    ref_path: Optional[str] = None
    ref_filename: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)
    time_taken: float = 0.0
    status: str = "pending"  # 'pending', 'processing', 'completed', 'failed'
    error_msg: Optional[str] = None


@dataclass
class BatchSummary:
    """Statistical summary of batch evaluation."""
    total_images: int
    completed_images: int
    failed_images: int
    total_time_seconds: float
    # Metric stats: metric_name -> {'mean': float, 'std': float, 'min': float, 'max': float, 'lower_better': bool}
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


class BatchEvaluator:
    """Executes multi-metric evaluations over folders of images."""

    def __init__(self, evaluator_engine: Optional[EvaluatorEngine] = None, device: str = "cpu"):
        self.engine = evaluator_engine or EvaluatorEngine(device=device)
        self.model_manager = ModelManager()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Signal batch execution to stop immediately."""
        self._cancel_event.set()

    def reset_cancellation(self) -> None:
        """Reset the cancellation flag."""
        self._cancel_event.clear()

    @staticmethod
    def scan_directory(dir_path: str) -> List[str]:
        """Scan directory recursively for supported images, returning sorted absolute paths."""
        p = Path(dir_path)
        if not p.is_dir():
            return []
        files = [
            str(f) for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        files.sort()
        return files

    def prepare_batch(
        self,
        target_dir: str,
        ref_dir: Optional[str] = None,
        pair_mode: str = "exact_name",  # 'exact_name' or 'alphabetic'
    ) -> List[BatchItem]:
        """Scan and pair target and reference images."""
        targets = self.scan_directory(target_dir)
        if not targets:
            return []

        refs = self.scan_directory(ref_dir) if ref_dir else []
        items: List[BatchItem] = []

        if not refs:
            for idx, t in enumerate(targets):
                items.append(BatchItem(
                    index=idx + 1,
                    target_path=t,
                    target_filename=os.path.basename(t),
                ))
            return items

        # Pairing logic
        if pair_mode == "exact_name":
            ref_dict = {os.path.basename(r): r for r in refs}
            for idx, t in enumerate(targets):
                t_name = os.path.basename(t)
                matched_ref = ref_dict.get(t_name)
                items.append(BatchItem(
                    index=idx + 1,
                    target_path=t,
                    target_filename=t_name,
                    ref_path=matched_ref,
                    ref_filename=os.path.basename(matched_ref) if matched_ref else None,
                ))
        else:  # alphabetic
            for idx, t in enumerate(targets):
                r = refs[idx] if idx < len(refs) else None
                items.append(BatchItem(
                    index=idx + 1,
                    target_path=t,
                    target_filename=os.path.basename(t),
                    ref_path=r,
                    ref_filename=os.path.basename(r) if r else None,
                ))

        return items

    def run_batch(
        self,
        items: List[BatchItem],
        metrics: List[str],
        on_progress: Optional[Callable[[int, int, BatchItem, float], None]] = None,
        on_item_finished: Optional[Callable[[BatchItem], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> BatchSummary:
        """Run batch evaluation across items and metrics."""
        if not metrics or any(not self.model_manager.supports_workflow(m, "batch") for m in metrics):
            raise ValueError("当前逐张批量流程不支持所选集合/提示词算法，请选择适用的图像指标。")
        self.reset_cancellation()
        def cancelled():
            return self._cancel_event.is_set() or bool(cancel_check and cancel_check())
        total_items = len(items)
        completed = 0
        failed = 0
        batch_start = time.perf_counter()

        # Load lazily after input validation; the evaluator caches valid models.
        # Invalid FR pairs must not trigger eager downloads of large weights.

        for idx, item in enumerate(items):
            if cancelled():
                break

            item_start = time.perf_counter()
            item.status = "processing"
            all_succeeded = True

            for metric_name in metrics:
                if cancelled():
                    break
                try:
                    res = self.engine.evaluate_single(
                        metric_name=metric_name,
                        target_path=item.target_path,
                        ref_path=item.ref_path,
                    )
                    item.scores[metric_name] = res.score
                except Exception as e:
                    all_succeeded = False
                    item.error_msg = str(e)
                    break

            item.time_taken = time.perf_counter() - item_start

            if cancelled():
                item.status = "cancelled"
                break

            if all_succeeded:
                item.status = "completed"
                completed += 1
            else:
                item.status = "failed"
                failed += 1

            # Flush intermediate PyTorch GPU memory cache after each batch item
            if hasattr(self.engine, "release_vram"):
                self.engine.release_vram(force_unload=False)

            now = time.perf_counter()
            elapsed = now - batch_start
            speed = (idx + 1) / elapsed if elapsed > 0 else 0.0

            if on_item_finished:
                on_item_finished(item)
            if on_progress:
                on_progress(idx + 1, total_items, item, speed)

        total_time = time.perf_counter() - batch_start

        # Compute summary statistics
        stats: Dict[str, Dict[str, float]] = {}
        for m in metrics:
            scores = [item.scores[m] for item in items if m in item.scores and item.status == "completed"]
            if scores:
                arr = np.array(scores)
                info = self.model_manager.get_model(m)
                stats[m] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "median": float(np.median(arr)),
                    "lower_better": info.lower_better if info else False,
                }

        return BatchSummary(
            total_images=total_items,
            completed_images=completed,
            failed_images=failed,
            total_time_seconds=total_time,
            stats=stats,
        )
