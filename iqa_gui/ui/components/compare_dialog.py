"""Multi-image quality comparison dialog (PK Benchmark for 2 to 3 images)."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
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

from iqa_gui.core.evaluator import SingleEvalResult


class MultiImageCompareDialog(QDialog):
    """Dialog displaying side-by-side metric comparison and PK winner highlights across 2-3 images."""

    def __init__(
        self,
        image_paths: List[str],
        eval_results_map: Dict[str, List[SingleEvalResult]],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.image_paths = [p for p in image_paths if p in eval_results_map]
        self.eval_results_map = eval_results_map

        self.setWindowTitle("多图画质横向比对 PK 报表")
        self.resize(1050, 680)
        self.setMinimumSize(850, 500)

        self._init_ui()
        self._populate_comparison_table()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Card
        header_card = QFrame()
        header_card.setProperty("class", "CardFrame")
        h_layout = QVBoxLayout(header_card)
        h_layout.setContentsMargins(10, 10, 10, 10)
        h_layout.setSpacing(6)

        title_row = QHBoxLayout()
        self.lbl_title = QLabel("<b>多图画质横向比对看板</b>")
        self.lbl_title.setStyleSheet("font-size: 15px;")
        title_row.addWidget(self.lbl_title)
        title_row.addStretch()

        self.lbl_count = QLabel(f"已对比 <b>{len(self.image_paths)}</b> 张待测图片")
        self.lbl_count.setProperty("class", "MutedText")
        title_row.addWidget(self.lbl_count)
        h_layout.addLayout(title_row)

        # Image legend strip
        legend_row = QHBoxLayout()
        legend_row.setSpacing(12)
        for idx, path in enumerate(self.image_paths):
            fname = os.path.basename(path)
            tag = QLabel(f"<b>图 {idx+1}:</b> {fname}")
            tag.setStyleSheet(
                "background: rgba(37, 99, 235, 0.08); border: 1px solid rgba(37, 99, 235, 0.25); "
                "border-radius: 4px; padding: 3px 8px; font-size: 12px;"
            )
            tag.setToolTip(path)
            legend_row.addWidget(tag)
        legend_row.addStretch()
        h_layout.addLayout(legend_row)

        # Winner summary banner
        self.lbl_summary_bar = QLabel("")
        self.lbl_summary_bar.setProperty("class", "InfoBanner")
        self.lbl_summary_bar.setStyleSheet("font-size: 12px; font-weight: bold;")
        h_layout.addWidget(self.lbl_summary_bar)

        layout.addWidget(header_card)

        # Comparison Table
        num_imgs = len(self.image_paths)
        # Columns: Metric, Mode, Direction, Image 1, Image 2, [Image 3], Winner, Applicability
        base_cols = ["指标名称", "模式", "倾向"]
        img_cols = [f"图 {i+1}: {os.path.basename(p)}" for i, p in enumerate(self.image_paths)]
        end_cols = ["胜出图片", "比对说明"]
        self.col_headers = base_cols + img_cols + end_cols

        self.table_compare = QTableWidget(self)
        self.table_compare.setColumnCount(len(self.col_headers))
        self.table_compare.setHorizontalHeaderLabels(self.col_headers)
        self.table_compare.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_compare.horizontalHeader().setStretchLastSection(True)
        self.table_compare.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table_compare, 1)

        # Footer Actions
        footer_layout = QHBoxLayout()
        self.btn_copy = QPushButton("复制对比数据")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)

        self.btn_export = QPushButton("导出为 CSV 报表...")
        self.btn_export.clicked.connect(self._export_to_csv)

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.accept)

        footer_layout.addWidget(self.btn_copy)
        footer_layout.addWidget(self.btn_export)
        footer_layout.addStretch()
        footer_layout.addWidget(self.btn_close)
        layout.addLayout(footer_layout)

    def _populate_comparison_table(self) -> None:
        """Align metrics across all evaluated images and populate table with winner highlights."""
        if not self.image_paths:
            return

        # Collect ordered metrics from the first image
        first_path = self.image_paths[0]
        first_results = self.eval_results_map.get(first_path, [])
        metric_names = [r.metric_name for r in first_results]

        self.table_compare.setRowCount(len(metric_names))

        num_imgs = len(self.image_paths)
        win_counts = [0] * num_imgs
        lose_counts = [0] * num_imgs

        for row_idx, m_name in enumerate(metric_names):
            # Gather results for m_name across images
            results_for_m: List[Optional[SingleEvalResult]] = []
            for p in self.image_paths:
                m_list = self.eval_results_map.get(p, [])
                match = next((r for r in m_list if r.metric_name == m_name), None)
                results_for_m.append(match)

            first_valid = next((r for r in results_for_m if r is not None), None)
            if not first_valid:
                continue

            lower_better = first_valid.lower_better
            mode = first_valid.metric_mode

            # Col 0: Metric Name
            item_name = QTableWidgetItem(m_name)
            item_name.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.table_compare.setItem(row_idx, 0, item_name)

            # Col 1: Mode
            item_mode = QTableWidgetItem(mode)
            item_mode.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_compare.setItem(row_idx, 1, item_mode)

            # Col 2: Direction
            dir_text = "▼ 越低越好" if lower_better else "▲ 越高越好"
            item_dir = QTableWidgetItem(dir_text)
            item_dir.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_compare.setItem(row_idx, 2, item_dir)

            # Determine best score (winner) and worst score (loser)
            valid_scores = [(idx, r.score) for idx, r in enumerate(results_for_m) if r is not None]
            best_idx: Optional[int] = None
            worst_idx: Optional[int] = None
            is_tie = False
            if valid_scores:
                if lower_better:
                    best_score = min(s for _, s in valid_scores)
                    worst_score = max(s for _, s in valid_scores)
                else:
                    best_score = max(s for _, s in valid_scores)
                    worst_score = min(s for _, s in valid_scores)

                tied_indices = [idx for idx, s in valid_scores if abs(s - best_score) < 1e-5]
                if len(tied_indices) == 1:
                    best_idx = tied_indices[0]
                    win_counts[best_idx] += 1
                else:
                    is_tie = True

                # When comparing >= 3 images, also determine the single worst performer
                if num_imgs >= 3 and abs(best_score - worst_score) > 1e-5:
                    worst_tied = [idx for idx, s in valid_scores if abs(s - worst_score) < 1e-5]
                    if len(worst_tied) == 1:
                        worst_idx = worst_tied[0]
                        lose_counts[worst_idx] += 1

            # Col 3 .. 3 + num_imgs - 1: Image Scores
            for img_idx, r in enumerate(results_for_m):
                col_idx = 3 + img_idx
                if r is not None:
                    txt = f"{r.formatted_score} ({r.grade}级)"
                    item_score = QTableWidgetItem(txt)
                    item_score.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                    # Highlight winner in green
                    if best_idx is not None and img_idx == best_idx:
                        item_score.setBackground(QColor(39, 174, 96, 40))
                        item_score.setForeground(QColor(39, 174, 96))
                        font = item_score.font()
                        font.setBold(True)
                        item_score.setFont(font)
                        item_score.setToolTip("该图片此项指标得分最优 (胜出)")
                    # Highlight worst performer in soft coral red when >= 3 images
                    elif num_imgs >= 3 and worst_idx is not None and img_idx == worst_idx:
                        item_score.setBackground(QColor(225, 29, 72, 35))
                        item_score.setForeground(QColor(225, 29, 72))
                        font = item_score.font()
                        font.setBold(True)
                        item_score.setFont(font)
                        item_score.setToolTip("该图片此项指标得分垫底 (最差)")

                    self.table_compare.setItem(row_idx, col_idx, item_score)
                else:
                    item_na = QTableWidgetItem("N/A")
                    item_na.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table_compare.setItem(row_idx, col_idx, item_na)

            # Col: Winner (and Loser if >= 3 images)
            win_col_idx = 3 + num_imgs
            if is_tie:
                item_win = QTableWidgetItem("平局")
            elif best_idx is not None:
                win_text = f"图 {best_idx + 1} 优胜"
                if num_imgs >= 3 and worst_idx is not None:
                    win_text += f" | 图 {worst_idx + 1} 垫底"
                item_win = QTableWidgetItem(win_text)
                item_win.setForeground(QColor(39, 174, 96))
                font = item_win.font()
                font.setBold(True)
                item_win.setFont(font)
            else:
                item_win = QTableWidgetItem("-")
            item_win.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_compare.setItem(row_idx, win_col_idx, item_win)

            # Col: Applicability Advice / Caution
            advice_col_idx = 4 + num_imgs
            advice = self._get_metric_comparison_advice(m_name, mode)
            item_advice = QTableWidgetItem(advice)
            item_advice.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            if "仅衡量" in advice or "不适用" in advice:
                item_advice.setForeground(QColor(225, 29, 72))
            elif "全参考" in advice:
                item_advice.setForeground(QColor(217, 119, 6))
            else:
                item_advice.setForeground(QColor(16, 185, 129))
            self.table_compare.setItem(row_idx, advice_col_idx, item_advice)

        # Resize columns to content
        for c in range(self.table_compare.columnCount() - 1):
            self.table_compare.resizeColumnToContents(c)

        # Update summary banner
        summary_parts = []
        for i, w_cnt in enumerate(win_counts):
            if num_imgs >= 3:
                l_cnt = lose_counts[i]
                summary_parts.append(f"图 {i+1} 胜出 <b>{w_cnt}</b> 项 (垫底 <b>{l_cnt}</b> 项)")
            else:
                summary_parts.append(f"图 {i+1} 胜出 <b>{w_cnt}</b> 项")
        self.lbl_summary_bar.setText(f"比对统计: {' | '.join(summary_parts)}")

    def _get_metric_comparison_advice(self, metric_name: str, mode: str) -> str:
        """Provide suitability analysis and precautions for multi-image comparison."""
        m = metric_name.lower()
        if m == "clipscore":
            return "提示词契合度指标: 仅衡量图像与 Prompt 的语义契合度。若待测图片主题内容不一致，横向对比数值无参考意义。"
        if mode == "FR":
            return "全参考指标: 仅当待测图均来源于同一张原图 (如不同压缩/超分版本) 时，横向比对有效。"
        if m in {"fid", "sfid", "fid_dinov2", "inception_score"}:
            return "分布度量指标: 不适用于单张或少样本对比。"
        return "通用无参考画质指标: 数值可直接横向比对。"

    def _copy_to_clipboard(self) -> str:
        """Copy table data to clipboard in TSV and Markdown format."""
        lines = []
        # Header
        lines.append("\t".join(self.col_headers))
        for row in range(self.table_compare.rowCount()):
            row_vals = []
            for col in range(self.table_compare.columnCount()):
                item = self.table_compare.item(row, col)
                row_vals.append(item.text().replace("\t", " ") if item else "")
            lines.append("\t".join(row_vals))

        text = "\n".join(lines)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            cb = QApplication.clipboard()
            if cb is not None:
                cb.setText(text)
            QMessageBox.information(self, "复制成功", "已将多图横向比对数据复制到剪贴板。")
        return text

    def _export_to_csv(self) -> None:
        """Export comparison table to CSV."""
        save_path, _ = QFileDialog.getSaveFileName(self, "导出对比报表", "IQA_Multi_Compare_Report.csv", "CSV 文件 (*.csv)")
        if not save_path:
            return
        try:
            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(self.col_headers)
                for row in range(self.table_compare.rowCount()):
                    row_vals = []
                    for col in range(self.table_compare.columnCount()):
                        item = self.table_compare.item(row, col)
                        row_vals.append(item.text() if item else "")
                    writer.writerow(row_vals)
            QMessageBox.information(self, "导出成功", f"对比报表已成功导出至:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出 CSV 报表时出错: {e}")
