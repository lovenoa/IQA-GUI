"""Results table widget with live updating, sorting, statistical summary, and export."""

from __future__ import annotations

import os
from typing import Dict, List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.core.batch_evaluator import BatchItem, BatchSummary
from iqa_gui.core.exporter import ResultExporter


class _NumericItem(QTableWidgetItem):
    """Keep formatted display text while sorting on the underlying number."""

    def __init__(self, text: str, value: Optional[float]):
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole + 1, value)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(Qt.ItemDataRole.UserRole + 1)
        right = other.data(Qt.ItemDataRole.UserRole + 1)
        if left is None or right is None:
            return left is None and right is not None
        return left < right


class ResultsTableWidget(QFrame):
    """Table widget for displaying and exporting batch evaluation results."""
    row_selected = Signal(object)  # Emits selected BatchItem

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("class", "CardFrame")
        self.current_metrics: List[str] = []
        self.current_items: List[BatchItem] = []
        self.current_summary: Optional[BatchSummary] = None

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Header bar with export buttons and summary count
        header_layout = QHBoxLayout()
        self.lbl_title = QLabel("<b>评估结果清单 (Results List)</b>")
        self.lbl_count = QLabel("共 0 项")
        self.lbl_count.setProperty("class", "MutedText")

        self.btn_export_csv = QPushButton("导出为 CSV 报表")
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_export_csv.setEnabled(False)

        self.btn_export_json = QPushButton("导出为 JSON")
        self.btn_export_json.clicked.connect(self._export_json)
        self.btn_export_json.setEnabled(False)

        header_layout.addWidget(self.lbl_title)
        header_layout.addWidget(self.lbl_count)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_export_csv)
        header_layout.addWidget(self.btn_export_json)
        layout.addLayout(header_layout)

        # Table
        self.table = QTableWidget(self)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, 1)

        # Summary statistics card
        self.stats_frame = QFrame()
        self.stats_frame.setProperty("class", "StatsFrame")
        self.stats_layout = QHBoxLayout(self.stats_frame)
        self.stats_layout.setContentsMargins(8, 4, 8, 4)
        self.lbl_stats = QLabel("统计摘要: 暂无完成数据")
        self.lbl_stats.setProperty("class", "MutedText")
        self.stats_layout.addWidget(self.lbl_stats)
        layout.addWidget(self.stats_frame)

    def set_batch(self, items: List[BatchItem], metrics: List[str]) -> None:
        """Initialize table with empty/pending batch items."""
        self.current_items = items
        self.current_metrics = metrics
        self.current_summary = None

        self.table.setSortingEnabled(False)
        self.table.clear()

        # Build column headers
        headers = ["序号", "目标图像", "参考图像", "状态", "耗时(秒)"] + [f"{m.upper()}" for m in metrics] + ["异常提示"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(items))

        for row, item in enumerate(items):
            self._set_row_data(row, item)

        self.table.setSortingEnabled(True)
        self.lbl_count.setText(f"共 {len(items)} 项")
        self.btn_export_csv.setEnabled(len(items) > 0)
        self.btn_export_json.setEnabled(len(items) > 0)
        self.lbl_stats.setText("统计摘要: 正在计算中...")

    def update_item(self, item: BatchItem) -> None:
        """Update a specific item in the table as it finishes."""
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) != item.index:
                continue
            for idx, current in enumerate(self.current_items):
                if current.index == item.index:
                    self.current_items[idx] = item
                    break
            sorting = self.table.isSortingEnabled()
            self.table.setSortingEnabled(False)
            self._set_row_data(row, item)
            self.table.setSortingEnabled(sorting)
            break

    def _set_row_data(self, row: int, item: BatchItem) -> None:
        # Index
        item_idx = _NumericItem(str(item.index), item.index)
        item_idx.setData(Qt.ItemDataRole.UserRole, item.index)
        item_idx.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 0, item_idx)

        # Target file
        item_tar = QTableWidgetItem(item.target_filename)
        item_tar.setToolTip(item.target_path)
        self.table.setItem(row, 1, item_tar)

        # Ref file
        item_ref = QTableWidgetItem(item.ref_filename or "-")
        if item.ref_path:
            item_ref.setToolTip(item.ref_path)
        self.table.setItem(row, 2, item_ref)

        # Status
        status_text = "待处理"
        if item.status == "processing":
            status_text = "计算中..."
        elif item.status == "completed":
            status_text = "已完成"
        elif item.status == "failed":
            status_text = "失败"
        item_status = QTableWidgetItem(status_text)
        item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if item.status == "completed":
            item_status.setForeground(Qt.GlobalColor.darkGreen)
        elif item.status == "failed":
            item_status.setForeground(Qt.GlobalColor.red)
        self.table.setItem(row, 3, item_status)

        # Time taken
        time_str = f"{item.time_taken:.2f}" if item.time_taken > 0 else "-"
        item_time = _NumericItem(time_str, item.time_taken if item.time_taken > 0 else None)
        item_time.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 4, item_time)

        # Scores
        col_offset = 5
        for idx, m in enumerate(self.current_metrics):
            val = item.scores.get(m)
            score_str = f"{val:.4f}" if val is not None else "-"
            item_score = _NumericItem(score_str, val)
            item_score.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col_offset + idx, item_score)

        # Error
        err_item = QTableWidgetItem(item.error_msg or "")
        self.table.setItem(row, col_offset + len(self.current_metrics), err_item)

    def set_summary(self, summary: BatchSummary) -> None:
        """Update statistical summary card."""
        self.current_summary = summary
        if not summary.stats:
            self.lbl_stats.setText(f"评估结束: 成功 {summary.completed_images} 项，失败 {summary.failed_images} 项")
            return

        parts = [f"完成: {summary.completed_images}/{summary.total_images} (总耗时: {summary.total_time_seconds:.1f}s)"]
        for m, s in summary.stats.items():
            parts.append(f"<b>{m.upper()}</b> [均值: {s['mean']:.3f} ± {s['std']:.3f}, 范围: {s['min']:.3f}~{s['max']:.3f}]")

        self.lbl_stats.setText(" | ".join(parts))

    def _on_selection_changed(self) -> None:
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            cell = self.table.item(row, 0)
            if cell is not None:
                item_id = cell.data(Qt.ItemDataRole.UserRole)
                item = next((item for item in self.current_items if item.index == item_id), None)
                if item is not None:
                    self.row_selected.emit(item)

    def _export_csv(self) -> None:
        if not self.current_items:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存 CSV 报表", "iqa_results.csv", "CSV 文件 (*.csv)")
        if path:
            try:
                ResultExporter.export_to_csv(path, self.current_items, self.current_metrics, self.current_summary)
                QMessageBox.information(self, "导出成功", f"报表已成功保存至:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出 CSV 出现错误:\n{e}")

    def _export_json(self) -> None:
        if not self.current_items:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存 JSON 数据", "iqa_results.json", "JSON 文件 (*.json)")
        if path:
            try:
                ResultExporter.export_to_json(path, self.current_items, self.current_metrics, self.current_summary)
                QMessageBox.information(self, "导出成功", f"数据已成功保存至:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出 JSON 出现错误:\n{e}")
