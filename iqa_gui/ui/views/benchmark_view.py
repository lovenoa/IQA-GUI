"""Tab 4: Dataset benchmark evaluation and academic correlation metrics (PLCC/SRCC/KRCC/RMSE)."""

from __future__ import annotations

import csv
import os
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.core.benchmark_engine import BenchmarkEngine, BenchmarkResult, MetricBenchmarkScore
from iqa_gui.core.evaluator import EvaluatorEngine
from iqa_gui.config import load_settings
from iqa_gui.core.worker_threads import BenchmarkWorker, launch_worker
from iqa_gui.ui.components.metric_selector import MetricSelectorWidget


class BenchmarkView(QWidget):
    """View for evaluating algorithms on academic benchmark datasets."""

    def __init__(self, evaluator_engine: EvaluatorEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.evaluator = evaluator_engine
        self.benchmark_engine = BenchmarkEngine(evaluator=self.evaluator)
        self.active_worker: Optional[BenchmarkWorker] = None
        self.last_result: Optional[BenchmarkResult] = None

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Upper splitter: Config & Metric Selector
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Dataset Configuration Card
        cfg_card = QFrame()
        cfg_card.setProperty("class", "CardFrame")
        cfg_layout = QVBoxLayout(cfg_card)
        cfg_layout.setSpacing(8)

        lbl_title = QLabel("<b>1. 数据集基准评测配置 (Benchmark Setup)</b>")
        cfg_layout.addWidget(lbl_title)

        lbl_desc = QLabel(
            "统一算法与 MOS/DMOS 的好坏方向后，相关系数越接近 +1 越一致；负值表示方向相反。<br>"
            "RMSE 为<b>同一样本内非负斜率线性拟合</b>后的误差，单位与标签相同；"
            "它<b>不代表独立测试的泛化性能</b>。集合算法和需要提示词的 CLIPScore 不适用此流程。"
        )
        lbl_desc.setProperty("class", "InfoBanner")
        lbl_desc.setWordWrap(True)
        cfg_layout.addWidget(lbl_desc)

        grid = QGridLayout()
        grid.setSpacing(6)

        # Preset standard dataset combo
        lbl_preset = QLabel("常用标准数据集预设:")
        self.combo_dataset = QComboBox()
        self.combo_dataset.addItem("自定义本地数据集 (带 CSV 标注文件)", "custom")
        self.combo_dataset.addItem("KonIQ-10k (经典大规模真实自然无参考)", "koniq10k")
        self.combo_dataset.addItem("CSIQ (经典合成失真全参考图集)", "csiq")
        self.combo_dataset.addItem("LIVE (经典基准测试集)", "live")
        self.combo_dataset.addItem("KADID-10k (合成失真大规模集)", "kadid10k")
        self.combo_dataset.addItem("AVA (摄影美学质量评分基准集)", "ava")
        self.combo_dataset.currentIndexChanged.connect(self._on_dataset_preset_changed)

        grid.addWidget(lbl_preset, 0, 0)
        grid.addWidget(self.combo_dataset, 0, 1, 1, 2)

        # Image Root
        lbl_root = QLabel("图像所在根目录:")
        self.edit_root = QLineEdit()
        self.edit_root.setPlaceholderText("选择包含测试图像的目录...")
        self.btn_browse_root = QPushButton("浏览...")
        self.btn_browse_root.clicked.connect(self._browse_root)

        grid.addWidget(lbl_root, 1, 0)
        grid.addWidget(self.edit_root, 1, 1)
        grid.addWidget(self.btn_browse_root, 1, 2)

        # CSV Annotations file
        lbl_csv = QLabel("标注 CSV 文件路径:")
        self.edit_csv = QLineEdit()
        self.edit_csv.setPlaceholderText("带表头的 CSV；请在下方映射实际列名")
        self.btn_browse_csv = QPushButton("选择...")
        self.btn_browse_csv.clicked.connect(self._browse_csv)

        grid.addWidget(lbl_csv, 2, 0)
        grid.addWidget(self.edit_csv, 2, 1)
        grid.addWidget(self.btn_browse_csv, 2, 2)

        self.edit_ref_root = QLineEdit()
        self.edit_ref_root.setPlaceholderText("参考图根目录；留空使用目标图根目录")
        self.btn_browse_ref_root = QPushButton("浏览...")
        self.btn_browse_ref_root.clicked.connect(self._browse_ref_root)
        grid.addWidget(QLabel("参考图根目录:"), 3, 0)
        grid.addWidget(self.edit_ref_root, 3, 1)
        grid.addWidget(self.btn_browse_ref_root, 3, 2)

        self.edit_image_col = QLineEdit("image")
        self.edit_score_col = QLineEdit("mos")
        self.edit_ref_col = QLineEdit()
        self.edit_ref_col.setPlaceholderText("全参考算法必填")
        column_fields = QHBoxLayout()
        for title, edit in [("目标列", self.edit_image_col), ("分数列", self.edit_score_col), ("参考列", self.edit_ref_col)]:
            column_fields.addWidget(QLabel(title))
            column_fields.addWidget(edit)
        grid.addLayout(column_fields, 4, 0, 1, 3)

        self.combo_label_direction = QComboBox()
        self.combo_label_direction.addItem("MOS：分数越高质量越好", False)
        self.combo_label_direction.addItem("DMOS：分数越低质量越好", True)
        grid.addWidget(QLabel("标签方向:"), 5, 0)
        grid.addWidget(self.combo_label_direction, 5, 1, 1, 2)
        self.lbl_preset_notice = QLabel("预设只填路径与标签方向；官方 CSV 格式各异，须核对列名，必要时转换为带表头的逐图 CSV。")
        self.lbl_preset_notice.setWordWrap(True)
        grid.addWidget(self.lbl_preset_notice, 6, 0, 1, 3)

        self.btn_run = QPushButton("开始基准评测并计算相关系数")
        self.btn_run.setProperty("class", "PrimaryButton")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.clicked.connect(self.run_benchmark)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_benchmark)
        grid.addWidget(self.btn_run, 7, 0, 1, 2)
        grid.addWidget(self.btn_cancel, 7, 2)

        cfg_layout.addLayout(grid)
        cfg_layout.addStretch()

        cfg_scroll = QScrollArea()
        cfg_scroll.setWidgetResizable(True)
        cfg_scroll.setToolTip("小窗口时可向下滚动，填写列名和参考目录，并开始评测。")
        cfg_scroll.setFrameShape(QFrame.Shape.NoFrame)
        cfg_scroll.setWidget(cfg_card)
        splitter.addWidget(cfg_scroll)

        # Right: Metric Selector
        self.metric_selector = MetricSelectorWidget(
            multi_select=True, parent=self,
            allowed_metrics=self.benchmark_engine.model_manager.workflow_metrics("benchmark"))
        splitter.addWidget(self.metric_selector)

        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 5)
        main_layout.addWidget(splitter, 3)

        # Status Label
        self.lbl_status = QLabel("就绪: 请配置数据集并勾选需测试的算法。")
        self.lbl_status.setProperty("class", "MutedText")
        self.lbl_status.setStyleSheet("font-weight: 600;")
        main_layout.addWidget(self.lbl_status)

        # Benchmark Scorecard Table
        table_frame = QFrame()
        table_frame.setProperty("class", "CardFrame")
        table_layout = QVBoxLayout(table_frame)
        table_layout.setSpacing(6)

        tbl_header = QHBoxLayout()
        lbl_tbl_title = QLabel("<b>学术评测排行榜 (Benchmark Scorecard)</b>")
        self.btn_export = QPushButton("导出排行榜 CSV")
        self.btn_export.clicked.connect(self._export_scorecard)
        self.btn_export.setEnabled(False)

        tbl_header.addWidget(lbl_tbl_title)
        tbl_header.addStretch()
        tbl_header.addWidget(self.btn_export)
        table_layout.addLayout(tbl_header)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "算法", "PLCC ↑", "SRCC ↑", "KRCC ↑", "拟合 RMSE ↓", "样本数", "耗时 (秒)",
        ])
        self.table.horizontalHeaderItem(4).setToolTip("同一样本内非负斜率线性校准后的误差，不代表独立测试集的泛化性能。")
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        table_layout.addWidget(self.table)

        main_layout.addWidget(table_frame, 4)

        # Select recommended metrics
        self.metric_selector.set_selected(["brisque", "niqe", "musiq"])

    def _on_dataset_preset_changed(self, idx: int) -> None:
        key = self.combo_dataset.currentData()
        if key != "custom":
            info = BenchmarkEngine.get_supported_standard_datasets().get(key, {})
            if info:
                self.edit_root.setText(info.get("dataroot_target", "./datasets"))
                self.edit_csv.setText(info.get("meta_info_file", "./datasets/meta_info"))
                self.edit_ref_root.setText(info.get("dataroot_ref", info.get("dataroot_target", "")))
                self.combo_label_direction.setCurrentIndex(1 if info.get("lower_better", False) else 0)

    def _browse_ref_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择参考图根目录")
        if path:
            self.edit_ref_root.setText(path)

    def _browse_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集图像根目录")
        if path:
            self.edit_root.setText(path)

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择标注 CSV 文件", "", "CSV 文件 (*.csv);;所有文件 (*.*)")
        if path:
            self.edit_csv.setText(path)

    def run_benchmark(self) -> None:
        """Start benchmark evaluation."""
        if getattr(self.window(), "_active_tasks", None):
            self.lbl_status.setText("请等待当前任务及清理结束。")
            return
        if self.active_worker is not None and self.active_worker.isRunning():
            return
        image_root = self.edit_root.text().strip()
        csv_path = self.edit_csv.text().strip()
        metrics = self.metric_selector.get_selected()
        image_col = self.edit_image_col.text().strip()
        score_col = self.edit_score_col.text().strip()
        ref_col = self.edit_ref_col.text().strip() or None
        ref_root = self.edit_ref_root.text().strip() or None

        if not os.path.isdir(image_root):
            QMessageBox.warning(self, "路径错误", "图像根目录不存在，请先正确指定或下载数据集。")
            return

        if not os.path.isfile(csv_path):
            QMessageBox.warning(self, "文件缺失", "未找到指定的标注 CSV 文件，请检查路径。")
            return

        if not metrics:
            QMessageBox.warning(self, "未勾选算法", "请在右侧至少勾选一个需测试的算法。")
            return

        if not image_col or not score_col:
            QMessageBox.warning(self, "列名缺失", "请填写目标图像列和主观分数列的实际表头名称。")
            return
        if ref_root and not os.path.isdir(ref_root):
            QMessageBox.warning(self, "路径错误", "参考图根目录不存在。")
            return
        if any(self.benchmark_engine.model_manager.get_model(name).mode == "FR" for name in metrics) and not ref_col:
            QMessageBox.warning(self, "参考列缺失", "已选全参考算法，请填写参考图像列名。")
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("正在进行基准评测...")
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setText("正在逐图评分，计算方向统一后的相关性与样本内拟合误差...")
        self.btn_export.setEnabled(False)

        self.active_worker = BenchmarkWorker(
            benchmark_engine=self.benchmark_engine,
            dataset_name=self.combo_dataset.currentText(),
            image_root=image_root,
            csv_path=csv_path,
            metrics=metrics,
            image_col=image_col,
            score_col=score_col,
            ref_col=ref_col,
            ref_root=ref_root,
            label_lower_better=bool(self.combo_label_direction.currentData()),
            auto_release_vram=bool(getattr(self.window(), "auto_release_vram", load_settings().get("auto_release_vram", True))),
        )
        self.active_worker.progress.connect(lambda msg, cur, tot: self.lbl_status.setText(f"{msg} ({cur}/{tot})"))
        self.active_worker.result_ready.connect(self._on_benchmark_finished)
        self.active_worker.finished.connect(self._on_worker_finished)
        self.active_worker.error.connect(self._on_benchmark_error)
        self.active_worker.cancelled.connect(lambda: self.lbl_status.setText("基准评测已取消。"))
        if not launch_worker(self.active_worker, self):
            self.active_worker = None
            self._on_worker_finished()
            self.lbl_status.setText("其他任务正在运行，请等待完成后再开始。")

    def _cancel_benchmark(self) -> None:
        if self.active_worker is not None and self.active_worker.isRunning():
            self.active_worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("正在取消，等待当前计算与资源清理结束...")

    def _on_worker_finished(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_run.setText("开始基准评测并计算相关系数")
        self.btn_cancel.setEnabled(False)

    def _on_benchmark_finished(self, res: BenchmarkResult) -> None:
        self.last_result = res
        self.btn_export.setEnabled(bool(res.metrics_scores))
        status = "已取消" if res.cancelled else "已完成"
        summary = f"基准评测{status}：{len(res.metrics_scores)} 个算法成功，{len(res.metric_errors)} 个失败；共 {res.num_samples} 张图像。"
        if res.metric_errors:
            summary += " " + "；".join(f"{name}: {reason}" for name, reason in res.metric_errors.items())
        self.lbl_status.setText(summary)
        self.lbl_status.setWordWrap(True)

        # Fill table
        self.table.setRowCount(len(res.metrics_scores))
        for row, (name, s) in enumerate(res.metrics_scores.items()):
            # Metric Name
            self.table.setItem(row, 0, QTableWidgetItem(name.upper()))
            # PLCC
            plcc_item = QTableWidgetItem(f"{s.plcc:.4f}")
            plcc_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, plcc_item)
            # SRCC
            srcc_item = QTableWidgetItem(f"{s.srcc:.4f}")
            srcc_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, srcc_item)
            # KRCC
            krcc_item = QTableWidgetItem(f"{s.krcc:.4f}")
            krcc_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, krcc_item)
            # RMSE
            rmse_item = QTableWidgetItem(f"{s.rmse:.4f}")
            rmse_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, rmse_item)
            # Samples
            self.table.setItem(row, 5, QTableWidgetItem(str(s.num_samples)))
            # Time
            time_item = QTableWidgetItem(f"{s.eval_time_seconds:.2f}")
            time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 6, time_item)

    def _on_benchmark_error(self, err_msg: str) -> None:
        self.lbl_status.setText(f"评测失败: {err_msg}")
        if not getattr(self.window(), "_closing", False):
            QMessageBox.critical(self, "基准评测异常", f"测评过程发生错误:\n{err_msg}")

    def _export_scorecard(self) -> None:
        if not self.last_result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出排行榜 CSV", "benchmark_scorecard.csv", "CSV 文件 (*.csv)")
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["算法", "PLCC(方向统一)", "SRCC(方向统一)", "KRCC(方向统一)",
                                     "RMSE(同样本内非负斜率线性拟合，非泛化性能)", "样本数", "耗时(秒)",
                                     "原始RMSE(未校准跨量纲，不用于算法比较)", "算法越低越好", "标签越低越好"])
                    for name, s in self.last_result.metrics_scores.items():
                        writer.writerow([name, f"{s.plcc:.4f}", f"{s.srcc:.4f}", f"{s.krcc:.4f}", f"{s.rmse:.4f}", s.num_samples,
                                         f"{s.eval_time_seconds:.2f}", "" if s.raw_rmse is None else f"{s.raw_rmse:.4f}",
                                         s.metric_lower_better, self.last_result.label_lower_better])
                QMessageBox.information(self, "导出成功", f"排行榜已保存至:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"无法导出文件:\n{e}")
