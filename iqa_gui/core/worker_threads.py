"""PySide6 QThread worker threads for non-blocking asynchronous computations."""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional
from PySide6.QtCore import QObject, QThread, Signal

from iqa_gui.core.evaluator import EvaluatorEngine, SingleEvalResult
from iqa_gui.core.batch_evaluator import BatchEvaluator, BatchItem, BatchSummary
from iqa_gui.core.gmad_engine import GMADEngine, GMADResult
from iqa_gui.core.benchmark_engine import BenchmarkEngine, BenchmarkResult
from iqa_gui.core.log_manager import LogManager


class SingleEvalWorker(QThread):
    """Background worker for evaluating single image or image pair across one or multiple metrics."""
    progress = Signal(int, int, str)   # Emits (current_index, total_count, metric_name)
    finished = Signal(object)          # Emits List[SingleEvalResult]
    error = Signal(str)                # Emits error message

    def __init__(
        self,
        evaluator: EvaluatorEngine,
        metrics: Union[str, List[str]],
        target_path: str,
        ref_path: Optional[str] = None,
        prompt: Optional[str] = None,
        auto_release_vram: bool = True,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.evaluator = evaluator
        if isinstance(metrics, str):
            self.metric_names = [metrics]
        else:
            self.metric_names = list(metrics)
        self.target_path = target_path
        self.ref_path = ref_path
        self.prompt = prompt
        self.auto_release_vram = auto_release_vram
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        heavy_metrics = {
            "qalign", "qalign_8bit", "qalign_4bit", "mplug_owl",
            "qrealign", "qrealign-mini", "qrealign-lite", "qrealign-pro",
        }
        error_occurred = False
        try:
            results: List[SingleEvalResult] = []
            total = len(self.metric_names)
            for idx, m_name in enumerate(self.metric_names, start=1):
                if self._is_cancelled:
                    break
                self.progress.emit(idx, total, m_name)

                # Determine if reference image is needed for this specific metric
                model_info = self.evaluator.model_manager.get_model(m_name)
                need_ref = (model_info.mode == "FR") if model_info else False
                ref = self.ref_path if need_ref else None

                res = self.evaluator.evaluate_single(
                    metric_name=m_name,
                    target_path=self.target_path,
                    ref_path=ref,
                    prompt=self.prompt,
                )
                results.append(res)

                # --- STEP-BY-STEP VRAM RELEASE (每一步完成即时释放显存) ---
                if self.auto_release_vram:
                    # 开启自动释放时：每一步算完立即卸载该模型并回收显存，确保多算法绝不累加显存
                    self.evaluator.unload_metric(m_name)
                    self.evaluator.release_vram(force_unload=False)
                elif m_name.lower() in heavy_metrics:
                    # 未开启自动释放时：超大多模态大模型也必须步步卸载，防止8GB显卡瞬时爆满
                    self.evaluator.unload_metric(m_name)
                    self.evaluator.release_vram(force_unload=False)
                else:
                    # 普通模型保留在缓存以备重用，但及时清空本步计算产生的激活中间张量
                    self.evaluator.release_vram(force_unload=False)

            if not self._is_cancelled:
                self.finished.emit(results)
        except Exception as e:
            error_occurred = True
            self.error.emit(str(e))
        finally:
            if error_occurred or self.auto_release_vram:
                self.evaluator.release_vram(force_unload=True)
            else:
                self.evaluator.release_vram(force_unload=False, unload_heavy_only=True)


class BatchEvalWorker(QThread):
    """Background worker for running multi-metric batch evaluations."""
    progress = Signal(int, int, object, float)  # current, total, BatchItem, speed
    item_finished = Signal(object)             # BatchItem
    finished = Signal(object)                  # BatchSummary
    error = Signal(str)

    def __init__(
        self,
        batch_evaluator: BatchEvaluator,
        items: List[BatchItem],
        metrics: List[str],
        auto_release_vram: bool = True,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.batch_evaluator = batch_evaluator
        self.items = items
        self.metrics = metrics
        self.auto_release_vram = auto_release_vram

    def cancel(self) -> None:
        self.batch_evaluator.cancel()

    def run(self) -> None:
        error_occurred = False
        try:
            summary = self.batch_evaluator.run_batch(
                items=self.items,
                metrics=self.metrics,
                on_progress=lambda cur, tot, item, speed: self.progress.emit(cur, tot, item, speed),
                on_item_finished=lambda item: self.item_finished.emit(item),
            )
            self.finished.emit(summary)
        except Exception as e:
            error_occurred = True
            self.error.emit(str(e))
        finally:
            if error_occurred or self.auto_release_vram:
                self.batch_evaluator.engine.release_vram(force_unload=True)
            else:
                self.batch_evaluator.engine.release_vram(force_unload=False)


class GMADWorker(QThread):
    """Background worker for gMAD candidate-pool competition."""
    progress = Signal(str, int, int)           # message, current, total
    finished = Signal(object)                  # GMADResult
    error = Signal(str)

    def __init__(
        self,
        gmad_engine: GMADEngine,
        image_paths: List[str],
        metric_a: str,
        metric_b: str,
        tolerance: float = 0.1,
        auto_release_vram: bool = True,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.gmad_engine = gmad_engine
        self.image_paths = image_paths
        self.metric_a = metric_a
        self.metric_b = metric_b
        self.tolerance = tolerance
        self.auto_release_vram = auto_release_vram

    def cancel(self) -> None:
        self.gmad_engine.cancel()

    def run(self) -> None:
        error_occurred = False
        try:
            result = self.gmad_engine.run_gmad(
                image_paths=self.image_paths,
                metric_a=self.metric_a,
                metric_b=self.metric_b,
                tolerance=self.tolerance,
                on_progress=lambda msg, cur, tot: self.progress.emit(msg, cur, tot),
            )
            self.finished.emit(result)
        except Exception as e:
            error_occurred = True
            self.error.emit(str(e))
        finally:
            if error_occurred or self.auto_release_vram:
                self.gmad_engine.evaluator.release_vram(force_unload=True)
            else:
                self.gmad_engine.evaluator.release_vram(force_unload=False)


class BenchmarkWorker(QThread):
    """Background worker for dataset benchmark evaluation."""
    progress = Signal(str, int, int)           # message, current, total
    finished = Signal(object)                  # BenchmarkResult
    error = Signal(str)

    def __init__(
        self,
        benchmark_engine: BenchmarkEngine,
        dataset_name: str,
        image_root: str,
        csv_path: str,
        metrics: List[str],
        auto_release_vram: bool = True,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.benchmark_engine = benchmark_engine
        self.dataset_name = dataset_name
        self.image_root = image_root
        self.csv_path = csv_path
        self.metrics = metrics
        self.auto_release_vram = auto_release_vram

    def cancel(self) -> None:
        self.benchmark_engine.cancel()

    def run(self) -> None:
        error_occurred = False
        try:
            result = self.benchmark_engine.run_benchmark_from_csv(
                dataset_name=self.dataset_name,
                image_root=self.image_root,
                csv_path=self.csv_path,
                metrics=self.metrics,
                on_progress=lambda msg, cur, tot: self.progress.emit(msg, cur, tot),
            )
            self.finished.emit(result)
        except Exception as e:
            error_occurred = True
            self.error.emit(str(e))
        finally:
            if error_occurred or self.auto_release_vram:
                self.benchmark_engine.evaluator.release_vram(force_unload=True)
            else:
                self.benchmark_engine.evaluator.release_vram(force_unload=False)


class PreloadWeightsWorker(QThread):
    """Background worker to download/preload model weights."""
    progress = Signal(str, int, int)           # current model, index, total
    finished = Signal(list, list)              # succeeded_list, failed_list

    def __init__(
        self,
        evaluator: EvaluatorEngine,
        metrics: List[str],
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.evaluator = evaluator
        self.metrics = metrics
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        succeeded = []
        failed = []
        total = len(self.metrics)
        for idx, m in enumerate(self.metrics):
            if self._is_cancelled:
                break
            self.progress.emit(f"正在下载/载入模型: {m}", idx + 1, total)
            try:
                self.evaluator.get_or_create_metric(m)
                succeeded.append(m)
            except Exception as e:
                failed.append((m, str(e)))
        self.finished.emit(succeeded, failed)


class SafePackageInstallerWorker(QThread):
    """Safely install missing packages in background, preserving PyTorch and CUDA drivers."""
    progress = Signal(str)          # Current progress/stdout line
    finished = Signal(bool, str)    # (success: bool, summary_message: str)

    def __init__(self, packages: List[str], parent: Optional[QObject] = None):
        super().__init__(parent)
        self.packages = list(packages)
        self._is_cancelled = False
        self._process: Optional[subprocess.Popen] = None

    def cancel(self) -> None:
        self._is_cancelled = True
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
            except Exception:
                pass

    def run(self) -> None:
        if not self.packages:
            self.finished.emit(True, "无需安装任何组件。")
            return

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            *self.packages,
            "--no-warn-script-location",
        ]

        logger = LogManager.get_instance()
        logger._append_to_buffer(f"\n[安全组件安装] 执行受控安装: {' '.join(cmd)}\n")
        logger._append_to_buffer("[安全组件安装] 严格保护核心环境，禁止覆盖升级现有 PyTorch/CUDA 驱动...\n")
        self.progress.emit(f"开始安全安装组件: {', '.join(self.packages)}")

        try:
            # Check cuda availability before
            try:
                import torch
                cuda_before = torch.cuda.is_available()
            except Exception:
                cuda_before = False

            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags,
            )

            if self._process.stdout:
                for line in self._process.stdout:
                    if self._is_cancelled:
                        break
                    stripped = line.strip()
                    if stripped:
                        self.progress.emit(stripped)
                        logger._append_to_buffer(f"[pip] {stripped}\n")

            self._process.wait()

            if self._is_cancelled:
                self.finished.emit(False, "安装已被用户取消。")
                return

            if self._process.returncode == 0:
                # Re-verify torch cuda
                try:
                    import torch
                    cuda_after = torch.cuda.is_available()
                    if cuda_before and not cuda_after:
                        logger._append_to_buffer("[警告] 检测到 CUDA 环境状态变动，请在后台控制台中复查。\n")
                except Exception:
                    pass

                self.finished.emit(True, f"组件 {', '.join(self.packages)} 安装成功！")
            else:
                self.finished.emit(False, f"组件安装失败，返回码: {self._process.returncode}。请查看后台日志获取详细信息。")
        except Exception as e:
            self.finished.emit(False, f"启动安装进程异常: {str(e)}")

