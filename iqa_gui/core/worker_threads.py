"""Qt workers separating business results from native thread completion."""
from __future__ import annotations
from contextlib import nullcontext
from typing import List, Optional, Union
from PySide6.QtCore import QObject, QThread, Signal
from iqa_gui.core.evaluator import EvaluatorEngine
from iqa_gui.core.batch_evaluator import BatchEvaluator, BatchItem
from iqa_gui.core.gmad_engine import GMADEngine
from iqa_gui.core.benchmark_engine import BenchmarkEngine
from iqa_gui.core.log_manager import LogManager


def launch_worker(worker, owner) -> bool:
    window = owner.window()
    if hasattr(window, "begin_task"):
        return window.begin_task(worker, owner)
    if worker.isRunning():
        return False
    if worker.parent() is None:
        worker.setParent(owner)
    worker.start()
    return True


class EvaluationWorker(QThread):
    """Hold the engine guard through cleanup. finished remains QThread.finished()."""
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, evaluator, auto_release_vram=True, parent=None):
        super().__init__(parent)
        self.evaluator = evaluator
        self.auto_release_vram = auto_release_vram
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        self._cancel_engine()

    def _cancel_engine(self):
        pass

    def run(self):
        value, failure = None, None
        guard = (self.evaluator.task_guard(lambda: self._is_cancelled)
                 if hasattr(self.evaluator, "task_guard") else nullcontext())
        try:
            with guard:
                try:
                    if self._is_cancelled:
                        raise InterruptedError("任务已取消。")
                    value = self._execute()
                except Exception as exc:
                    failure = exc
                finally:
                    try:
                        self.evaluator.release_vram(force_unload=bool(failure) or self.auto_release_vram)
                    except Exception as exc:
                        failure = failure or exc
        except Exception as exc:
            failure = exc
        if self._is_cancelled or isinstance(failure, InterruptedError):
            self.cancelled.emit()
        elif failure is not None:
            self.error.emit(str(failure))
        else:
            self._publish_result(value)


class SingleEvalWorker(EvaluationWorker):
    progress = Signal(int, int, str)
    result_ready = Signal(object)
    multi_finished = Signal(object)
    rounds_finished = Signal(object)

    def __init__(self, evaluator: EvaluatorEngine, metrics: Union[str, List[str]],
                 target_path: Union[str, List[str]], ref_path: Optional[str] = None,
                 prompt: Optional[str] = None, rounds: int = 1,
                 auto_release_vram: bool = True, parent: Optional[QObject] = None):
        super().__init__(evaluator, auto_release_vram, parent)
        self.metric_names = [metrics] if isinstance(metrics, str) else list(metrics)
        self.target_paths = list(target_path) if isinstance(target_path, list) else [target_path]
        self.target_path = self.target_paths[0] if self.target_paths else ""
        self.ref_path, self.prompt = ref_path, prompt
        self.rounds = max(1, int(rounds))

    def _execute(self):
        multi_round_map = {}
        total = self.rounds * len(self.target_paths) * len(self.metric_names)
        step = 0
        for r_idx in range(1, self.rounds + 1):
            round_map = {}
            for img_idx, target in enumerate(self.target_paths):
                results = []
                for name in self.metric_names:
                    if self._is_cancelled:
                        raise InterruptedError("评估已取消。")
                    step += 1
                    self.progress.emit(step, total, f"[轮次 {r_idx}/{self.rounds}] [图 {img_idx+1}/{len(self.target_paths)}] {name}")
                    info = self.evaluator.model_manager.get_model(name)
                    ref = self.ref_path if info and info.mode == "FR" else None
                    results.append(self.evaluator.evaluate_single(
                        metric_name=name, target_path=target, ref_path=ref, prompt=self.prompt))
                    if self.auto_release_vram:
                        self.evaluator.unload_metric(name)
                round_map[target] = results
            multi_round_map[r_idx] = round_map
        return multi_round_map

    def _publish_result(self, value):
        self.rounds_finished.emit(value)
        first = value.get(1, {})
        self.multi_finished.emit(first)
        self.result_ready.emit(first.get(self.target_path, []))


class BatchEvalWorker(EvaluationWorker):
    progress = Signal(int, int, object, float)
    item_finished = Signal(object)
    result_ready = Signal(object)

    def __init__(self, batch_evaluator: BatchEvaluator, items: List[BatchItem],
                 metrics: List[str], auto_release_vram: bool = True, parent=None):
        super().__init__(batch_evaluator.engine, auto_release_vram, parent)
        self.batch_evaluator, self.items, self.metrics = batch_evaluator, items, metrics

    def _cancel_engine(self):
        self.batch_evaluator.cancel()

    def _execute(self):
        return self.batch_evaluator.run_batch(
            items=self.items, metrics=self.metrics,
            on_progress=lambda cur, total, item, speed: self.progress.emit(cur, total, item, speed),
            on_item_finished=self.item_finished.emit, cancel_check=lambda: self._is_cancelled)

    def _publish_result(self, value):
        self.result_ready.emit(value)


class GMADWorker(EvaluationWorker):
    progress = Signal(str, int, int)
    result_ready = Signal(object)

    def __init__(self, gmad_engine: GMADEngine, image_paths: List[str],
                 metric_a: str, metric_b: str, tolerance: float = .1,
                 auto_release_vram: bool = True, parent=None):
        super().__init__(gmad_engine.evaluator, auto_release_vram, parent)
        self.gmad_engine, self.image_paths = gmad_engine, image_paths
        self.metric_a, self.metric_b, self.tolerance = metric_a, metric_b, tolerance

    def _cancel_engine(self):
        self.gmad_engine.cancel()

    def _execute(self):
        return self.gmad_engine.run_gmad(
            image_paths=self.image_paths, metric_a=self.metric_a, metric_b=self.metric_b,
            tolerance=self.tolerance,
            on_progress=lambda msg, cur, total: self.progress.emit(msg, cur, total),
            cancel_check=lambda: self._is_cancelled)

    def _publish_result(self, value):
        self.result_ready.emit(value)


class BenchmarkWorker(EvaluationWorker):
    progress = Signal(str, int, int)
    result_ready = Signal(object)

    def __init__(self, benchmark_engine: BenchmarkEngine, dataset_name: str,
                 image_root: str, csv_path: str, metrics: List[str],
                 auto_release_vram: bool = True, parent=None, *,
                 image_col="image", score_col="mos", ref_col=None,
                 ref_root=None, label_lower_better=False):
        super().__init__(benchmark_engine.evaluator, auto_release_vram, parent)
        self.benchmark_engine, self.dataset_name = benchmark_engine, dataset_name
        self.image_root, self.csv_path, self.metrics = image_root, csv_path, metrics
        self.image_col, self.score_col, self.ref_col = image_col, score_col, ref_col
        self.ref_root, self.label_lower_better = ref_root, label_lower_better

    def _cancel_engine(self):
        self.benchmark_engine.cancel()

    def _execute(self):
        return self.benchmark_engine.run_benchmark_from_csv(
            dataset_name=self.dataset_name, image_root=self.image_root,
            csv_path=self.csv_path, metrics=self.metrics, image_col=self.image_col,
            score_col=self.score_col, ref_col=self.ref_col, ref_root=self.ref_root,
            label_lower_better=self.label_lower_better,
            on_progress=lambda msg, cur, total: self.progress.emit(msg, cur, total),
            cancel_check=lambda: self._is_cancelled)

    def _publish_result(self, value):
        self.result_ready.emit(value)


class PreloadWeightsWorker(EvaluationWorker):
    progress = Signal(str, int, int)
    result_ready = Signal(list, list)

    def __init__(self, evaluator, metrics, parent=None):
        super().__init__(evaluator, False, parent)
        self.metrics = list(metrics)

    def _execute(self):
        succeeded, failed = [], []
        for idx, name in enumerate(self.metrics):
            if self._is_cancelled:
                raise InterruptedError("预载已取消。")
            self.progress.emit(f"正在下载/载入模型: {name}", idx + 1, len(self.metrics))
            try:
                self.evaluator.get_or_create_metric(name)
                succeeded.append(name)
            except Exception as exc:
                failed.append((name, str(exc)))
        return succeeded, failed

    def _publish_result(self, value):
        self.result_ready.emit(*value)


class SafePackageInstallerWorker(QThread):
    """Constrain installed versions; do not kill pip while it is writing."""
    progress = Signal(str)
    result_ready = Signal(bool, str)

    def __init__(self, packages, parent=None):
        super().__init__(parent)
        self.packages = list(packages)
        self._is_cancelled = self._install_started = False

    def cancel(self):
        if self._install_started:
            self.progress.emit("组件安装正在写入，请等待完成后安全退出。")
        else:
            self._is_cancelled = True

    def run(self):
        from iqa_gui.core.dependency_installer import install_packages
        if self._is_cancelled:
            self.result_ready.emit(False, "组件安装已取消。")
            return
        def progress(text):
            self.progress.emit(text)
            LogManager.get_instance()._append_to_buffer(f"[组件安装] {text}\n")
        try:
            self._install_started = True
            code = install_packages(self.packages, on_progress=progress)
            if code == 0:
                self.result_ready.emit(True, "组件安装完成。请重新启动工作台，在新进程中加载并检查组件。")
            else:
                self.result_ready.emit(False, "安装未完成，已有环境版本受约束。请查看日志中的冲突或网络错误。")
        except Exception as exc:
            self.result_ready.emit(False, f"组件安装未完成: {exc}")
        finally:
            self._install_started = False

