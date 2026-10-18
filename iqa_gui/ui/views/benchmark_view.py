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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.core.benchmark_engine import BenchmarkEngine, BenchmarkResult, MetricBenchmarkScore
from iqa_gui.core.evaluator import EvaluatorEngine
from iqa_gui.core.worker_threads import BenchmarkWorker
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
            "<b>学术评测标准说明 (通俗解释)：</b><br>"
            "在标准图库上对比算法打分与人眼真实主观打分 (MOS)。<br>"
            "• <b>PLCC / SRCC / KRCC</b>：相关性指标，数值<b>越接近 1.0</b>，说明算法对画质好坏的排序越符合真实人眼！<br>"
            "• <b>RMSE</b>：预测误差，数值<b>越小</b>越精确。"
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
        self.edit_csv.setPlaceholderText("包含 image 列与 mos 真实得分列的 CSV 表格...")
        self.btn_browse_csv = QPushButton("选择...")
        self.btn_browse_csv.clicked.connect(self._browse_csv)

        grid.addWidget(lbl_csv, 2, 0)
        grid.addWidget(self.edit_csv, 2, 1)
        grid.addWidget(self.btn_browse_csv, 2, 2)

        # Run Button
        self.btn_run = QPushButton("📊 开始基准评测并计算相关系数")
        self.btn_run.setProperty("class", "PrimaryButton")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.clicked.connect(self.run_benchmark)

        grid.addWidget(self.btn_run, 3, 0, 1, 3)

        cfg_layout.addLayout(grid)
        cfg_layout.addStretch()

        splitter.addWidget(cfg_card)

        # Right: Metric Selector
        self.metric_selector = MetricSelectorWidget(multi_select=True, parent=self)
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
        self.btn_export = QPushButton("📋 导出排行榜 CSV")
        self.btn_export.clicked.connect(self._export_scorecard)
        self.btn_export.setEnabled(False)

        tbl_header.addWidget(lbl_tbl_title)
        tbl_header.addStretch()
        tbl_header.addWidget(self.btn_export)
        table_layout.addLayout(tbl_header)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "算法名称 (Metric)",
            "PLCC 线性相关 (↑越接近1越好)",
            "SRCC 秩相关 (↑越接近1越好)",
            "KRCC 肯德尔相关 (↑越接近1越好)",
            "RMSE 预测误差 (↓越小越好)",
            "测试样本量",
            "评测耗时 (秒)",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        table_layout.addWidget(self.table)

        main_layout.addWidget(table_frame, 4)

        # Select recommended metrics
        self.metric_selector.select_recommended()

    def _on_dataset_preset_changed(self, idx: int) -> None:
        key = self.combo_dataset.currentData()
        if key != "custom":
            info = BenchmarkEngine.get_supported_standard_datasets().get(key, {})
            if info:
                self.edit_root.setText(info.get("dataroot_target", "./datasets"))
                self.edit_csv.setText(info.get("meta_info_file", "./datasets/meta_info"))

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
        image_root = self.edit_root.text().strip()
        csv_path = self.edit_csv.text().strip()
        metrics = self.metric_selector.get_selected()

        if not os.path.isdir(image_root):
            QMessageBox.warning(self, "路径错误", "图像根目录不存在，请先正确指定或下载数据集。")
            return

        if not os.path.isfile(csv_path):
            QMessageBox.warning(self, "文件缺失", "未找到指定的标注 CSV 文件，请检查路径。")
            return

        if not metrics:
            QMessageBox.warning(self, "未勾选算法", "请在右侧至少勾选一个需测试的算法。")
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ 正在进行基准测评...")
        self.lbl_status.setText("正在逐项提取图像特征并进行主观拟合相关性计算...")

        self.active_worker = BenchmarkWorker(
            benchmark_engine=self.benchmark_engine,
            dataset_name=self.combo_dataset.currentText(),
            image_root=image_root,
            csv_path=csv_path,
            metrics=metrics,
        )
        self.active_worker.progress.connect(lambda msg, cur, tot: self.lbl_status.setText(f"{msg} ({cur}/{tot})"))
        self.active_worker.finished.connect(self._on_benchmark_finished)
        self.active_worker.error.connect(self._on_benchmark_error)
        self.active_worker.start()

    def _on_benchmark_finished(self, res: BenchmarkResult) -> None:
        self.last_result = res
        self.btn_run.setEnabled(True)
        self.btn_run.setText("📊 开始基准评测并计算相关系数")
        self.btn_export.setEnabled(True)
        self.lbl_status.setText(f"✅ 基准评测已完成！共测试 {len(res.metrics_scores)} 个算法，测试集包含 {res.num_samples} 张图像。")

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
        self.btn_run.setEnabled(True)
        self.btn_run.setText("📊 开始基准评测并计算相关系数")
        self.lbl_status.setText(f"❌ 评测失败: {err_msg}")
        QMessageBox.critical(self, "基准评测异常", f"测评过程发生错误:\n{err_msg}")

    def _export_scorecard(self) -> None:
        if not self.last_result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出排行榜 CSV", "benchmark_scorecard.csv", "CSV 文件 (*.csv)")
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["算法", "PLCC", "SRCC", "KRCC", "RMSE", "样本数", "耗时(秒)"])
                    for name, s in self.last_result.metrics_scores.items():
                        writer.writerow([name, f"{s.plcc:.4f}", f"{s.srcc:.4f}", f"{s.krcc:.4f}", f"{s.rmse:.4f}", s.num_samples, f"{s.eval_time_seconds:.2f}"])
                QMessageBox.information(self, "导出成功", f"排行榜已保存至:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"无法导出文件:\n{e}")
