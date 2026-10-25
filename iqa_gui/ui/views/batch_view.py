"""Tab 2: Batch folder evaluation view."""

from __future__ import annotations

import os
from typing import List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.core.batch_evaluator import BatchEvaluator, BatchItem, BatchSummary
from iqa_gui.core.evaluator import EvaluatorEngine
from iqa_gui.config import load_settings, update_settings, SettingsSaveError
from iqa_gui.core.worker_threads import BatchEvalWorker, launch_worker
from iqa_gui.ui.components.metric_selector import MetricSelectorWidget
from iqa_gui.ui.components.progress_bar import TaskProgressWidget
from iqa_gui.ui.components.results_table import ResultsTableWidget


class BatchEvalView(QWidget):
    """View for batch folder evaluation and tabular reporting."""

    def __init__(self, evaluator_engine: EvaluatorEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.evaluator = evaluator_engine
        self.batch_engine = BatchEvaluator(evaluator_engine=self.evaluator)
        self.active_worker: Optional[BatchEvalWorker] = None

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Upper splitter: Folder config & Metric selection
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Folder setup
        folder_card = QFrame()
        folder_card.setProperty("class", "CardFrame")
        folder_layout = QVBoxLayout(folder_card)
        folder_layout.setSpacing(8)

        lbl_folder_title = QLabel("<b>1. 设置图像文件夹路径</b>")
        folder_layout.addWidget(lbl_folder_title)

        grid = QGridLayout()
        grid.setSpacing(6)

        # Target Dir
        lbl_target = QLabel("待评估目标目录:")
        self.edit_target_dir = QLineEdit()
        self.edit_target_dir.setPlaceholderText("选择包含待评估失真图片的文件夹...")
        self.btn_browse_tar = QPushButton("浏览...")
        self.btn_browse_tar.clicked.connect(self._browse_target_dir)

        grid.addWidget(lbl_target, 0, 0)
        grid.addWidget(self.edit_target_dir, 0, 1)
        grid.addWidget(self.btn_browse_tar, 0, 2)

        # Ref Dir
        lbl_ref = QLabel("参考原图目录(可选):")
        self.edit_ref_dir = QLineEdit()
        self.edit_ref_dir.setPlaceholderText("全参考算法(FR)必选；若只测无参考NR可留空")
        self.btn_browse_ref = QPushButton("浏览...")
        self.btn_browse_ref.clicked.connect(self._browse_ref_dir)

        grid.addWidget(lbl_ref, 1, 0)
        grid.addWidget(self.edit_ref_dir, 1, 1)
        grid.addWidget(self.btn_browse_ref, 1, 2)

        # Pairing mode
        lbl_mode = QLabel("配对匹配规则:")
        self.combo_pair_mode = QComboBox()
        self.combo_pair_mode.addItem("文件名相同精准匹配 (如 I01.bmp 对应 I01.bmp)", "exact_name")
        self.combo_pair_mode.addItem("按文件排序索引顺序配对 (第1张对应第1张)", "alphabetic")

        grid.addWidget(lbl_mode, 2, 0)
        grid.addWidget(self.combo_pair_mode, 2, 1, 1, 2)

        folder_layout.addLayout(grid)

        # Run buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("开始批量评估")
        self.btn_start.setProperty("class", "PrimaryButton")
        self.btn_start.setMinimumHeight(36)
        self.btn_start.clicked.connect(self.start_batch)

        btn_layout.addWidget(self.btn_start)
        folder_layout.addLayout(btn_layout)
        folder_layout.addStretch()

        top_splitter.addWidget(folder_card)

        # Right: Metric selector (multi-select)
        self.metric_selector = MetricSelectorWidget(
            multi_select=True, parent=self,
            allowed_metrics=self.batch_engine.model_manager.workflow_metrics("batch"))
        top_splitter.addWidget(self.metric_selector)

        top_splitter.setStretchFactor(0, 5)
        top_splitter.setStretchFactor(1, 5)
        main_layout.addWidget(top_splitter, 3)

        # Middle: Progress bar widget
        self.progress_widget = TaskProgressWidget(self)
        self.progress_widget.cancel_requested.connect(self.cancel_batch)
        main_layout.addWidget(self.progress_widget)

        # Bottom: Results table
        self.results_table = ResultsTableWidget(self)
        main_layout.addWidget(self.results_table, 5)

        # Select anime presets by default
        self.metric_selector.set_selected(["brisque", "niqe"])
        settings = load_settings()
        self.edit_target_dir.setText(settings.get("last_target_dir", ""))
        self.edit_ref_dir.setText(settings.get("last_ref_dir", ""))

    def _browse_target_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择待评估图像文件夹")
        if path:
            self.edit_target_dir.setText(path)

    def _browse_ref_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择参考基准图像文件夹")
        if path:
            self.edit_ref_dir.setText(path)

    def start_batch(self) -> None:
        """Scan folders, initialize table, and launch worker."""
        if getattr(self.window(), "_active_tasks", None):
            self.progress_widget.lbl_status.setText("请等待当前任务及清理结束。")
            return
        tar_dir = self.edit_target_dir.text().strip()
        ref_dir = self.edit_ref_dir.text().strip() or None
        pair_mode = self.combo_pair_mode.currentData()
        metrics = self.metric_selector.get_selected()

        if not tar_dir or not os.path.isdir(tar_dir):
            QMessageBox.warning(self, "路径错误", "请先选择有效的待评估图像目标文件夹！")
            return

        if not metrics:
            QMessageBox.warning(self, "未勾选算法", "请在右侧至少勾选一个质量评估算法。")
            return

        # Prepare items
        items = self.batch_engine.prepare_batch(tar_dir, ref_dir, pair_mode=pair_mode)
        if not items:
            QMessageBox.warning(self, "未发现图像", f"在目录 {tar_dir} 下未找到受支持的图像文件 (PNG/JPG/BMP/WEBP)。")
            return

        # Check if FR metric requires ref
        model_mgr = self.metric_selector.model_manager
        has_fr = any(model_mgr.get_model(m) and model_mgr.get_model(m).mode == "FR" for m in metrics)
        if has_fr and not ref_dir:
            reply = QMessageBox.question(
                self,
                "缺少参考目录",
                "您勾选了全参考(FR)算法，但未提供参考图目录。\n这些算法在没有原图时将无法计算。是否仍继续执行？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Initialize table and progress
        self.results_table.set_batch(items, metrics)
        self.progress_widget.set_running(True, f"准备开始批量评估，共 {len(items)} 张图片...")
        self.btn_start.setEnabled(False)
        try:
            update_settings({"last_target_dir": tar_dir, "last_ref_dir": ref_dir or ""})
        except SettingsSaveError as exc:
            self.progress_widget.lbl_status.setText(f"路径记忆未保存，本次评估仍可继续: {exc}")

        # Launch worker
        self.active_worker = BatchEvalWorker(
            batch_evaluator=self.batch_engine,
            items=items,
            metrics=metrics,
            auto_release_vram=getattr(self.window(), "auto_release_vram", load_settings().get("auto_release_vram", True)),
        )
        self.active_worker.progress.connect(self._on_batch_progress)
        self.active_worker.item_finished.connect(self.results_table.update_item)
        self.active_worker.result_ready.connect(self._on_batch_finished)
        self.active_worker.finished.connect(self._on_thread_finished)
        self.active_worker.cancelled.connect(self._on_batch_cancelled)
        self.active_worker.error.connect(self._on_batch_error)
        if not launch_worker(self.active_worker, self):
            self.active_worker = None
            self.progress_widget.set_running(False, "已有任务，请等待结束后再开始。")
            self._on_thread_finished()

    def _on_thread_finished(self):
        self.btn_start.setEnabled(True)
        self.progress_widget.btn_cancel.setEnabled(False)

    def _on_batch_cancelled(self):
        self.progress_widget.set_running(False, "批量评估已停止，保留已完成的结果行。")
        self.results_table.lbl_stats.setText("已停止；未处理行不计入完成结果。")

    def cancel_batch(self) -> None:
        """Cancel batch execution."""
        if self.active_worker:
            self.progress_widget.lbl_status.setText("正在停止中...")
            self.active_worker.cancel()

    def _on_batch_progress(self, current: int, total: int, item: BatchItem, speed: float) -> None:
        msg = f"正在评估: {item.target_filename} ({current}/{total})"
        self.progress_widget.update_progress(current, total, status_msg=msg, speed=speed)

    def _on_batch_finished(self, summary: BatchSummary) -> None:
        self.progress_widget.set_running(False, f"批量评估完成。共处理 {summary.completed_images} 项，耗时 {summary.total_time_seconds:.1f} 秒")
        self.results_table.set_summary(summary)

    def _on_batch_error(self, err_msg: str) -> None:
        self.progress_widget.set_running(False, f"评估异常终止: {err_msg}")
        if not getattr(self.window(), "_closing", False):
            QMessageBox.critical(self, "批量评估出错", f"批处理过程中遇到错误:\n{err_msg}")
