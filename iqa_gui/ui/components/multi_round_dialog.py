"""Multi-round evaluation summary dialog showing stability metrics (Range, Std Dev, Mean) and PK comparisons."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional
import numpy as np
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
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.core.evaluator import SingleEvalResult


class MultiRoundSummaryDialog(QDialog):
    """Dialog displaying aggregated multi-round evaluation statistics (Range, Std Dev, Mean, PK)."""

    def __init__(
        self,
        image_paths: List[str],
        rounds_eval_map: Dict[int, Dict[str, List[SingleEvalResult]]],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.image_paths = list(image_paths)
        self.rounds_eval_map = rounds_eval_map
        self.total_rounds = len(rounds_eval_map)

        self.setWindowTitle("📈 多轮测试统计聚合与稳定性分析报表")
        self.resize(1150, 700)
        self.setMinimumSize(900, 520)

        self.active_table: Optional[QTableWidget] = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Frame
        header_card = QFrame()
        header_card.setProperty("class", "CardFrame")
        h_layout = QVBoxLayout(header_card)
        h_layout.setContentsMargins(10, 10, 10, 10)
        h_layout.setSpacing(4)

        top_row = QHBoxLayout()
        self.lbl_title = QLabel("<b>多轮重复测试聚合统计与算法稳定性诊断 (Multi-Round Stability Benchmark)</b>")
        self.lbl_title.setStyleSheet("font-size: 15px;")
        top_row.addWidget(self.lbl_title)
        top_row.addStretch()

        self.lbl_meta = QLabel(f"测试轮数: <b>{self.total_rounds} 轮</b> | 待测图像: <b>{len(self.image_paths)} 张</b>")
        self.lbl_meta.setProperty("class", "MutedText")
        top_row.addWidget(self.lbl_meta)
        h_layout.addLayout(top_row)

        desc_text = (
            "💡 说明: 对深度学习算法进行多轮次独立推断，可检验算法是否受量化误差、随机采样或数值截断影响。"
            "统计值包含：变动极差 (Max - Min)、样本标准差 (σ) 以及平均值 (Mean μ)。"
        )
        self.lbl_sub = QLabel(desc_text)
        self.lbl_sub.setProperty("class", "InfoBanner")
        self.lbl_sub.setStyleSheet("font-size: 11px;")
        h_layout.addWidget(self.lbl_sub)

        layout.addWidget(header_card)

        # Content Area: Single Image vs Multi-Image PK
        if len(self.image_paths) <= 1:
            # Single image mode: one direct table
            self.table_single = QTableWidget(self)
            self._setup_single_image_table()
            layout.addWidget(self.table_single, 1)
            self.active_table = self.table_single
        else:
            # Multi-image mode: Tabs with Mean Comparison PK & Detailed breakdown
            self.tab_widget = QTabWidget(self)

            tab_summary = QWidget()
            tab_sum_layout = QVBoxLayout(tab_summary)
            tab_sum_layout.setContentsMargins(4, 4, 4, 4)
            self.table_multi_summary = QTableWidget(tab_summary)
            self._setup_multi_image_summary_table()
            tab_sum_layout.addWidget(self.table_multi_summary)
            self.tab_widget.addTab(tab_summary, "📊 多图平均值 PK 与稳定性比较")

            tab_details = QWidget()
            tab_det_layout = QVBoxLayout(tab_details)
            tab_det_layout.setContentsMargins(4, 4, 4, 4)
            self.table_multi_details = QTableWidget(tab_details)
            self._setup_multi_image_details_table()
            tab_det_layout.addWidget(self.table_multi_details)
            self.tab_widget.addTab(tab_details, "📋 逐轮全部详细明细表 (全部排出)")

            layout.addWidget(self.tab_widget, 1)
            self.tab_widget.currentChanged.connect(self._on_tab_changed)
            self.active_table = self.table_multi_summary

        # Footer Actions
        footer_layout = QHBoxLayout()
        self.btn_copy = QPushButton("📋 复制当前表格至剪贴板")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)

        self.btn_export = QPushButton("💾 导出为 CSV 报表...")
        self.btn_export.clicked.connect(self._export_to_csv)

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.accept)

        footer_layout.addWidget(self.btn_copy)
        footer_layout.addWidget(self.btn_export)
        footer_layout.addStretch()
        footer_layout.addWidget(self.btn_close)
        layout.addLayout(footer_layout)

    def _on_tab_changed(self, index: int) -> None:
        self.active_table = self.table_multi_summary if index == 0 else self.table_multi_details

    def _setup_single_image_table(self) -> None:
        """Populate table for single image evaluated across multiple rounds."""
        target_path = self.image_paths[0] if self.image_paths else ""
        first_round = self.rounds_eval_map.get(1, {})
        first_results = first_round.get(target_path, [])
        metric_names = [r.metric_name for r in first_results]

        cols = ["算法名称", "模式", "倾向"]
        for r_num in range(1, self.total_rounds + 1):
            cols.append(f"第 {r_num} 轮得分")
        cols.extend(["均值 (Mean μ)", "变动极差 (Range)", "标准差 (Std σ)", "稳定性诊断"])

        self.table_single.setColumnCount(len(cols))
        self.table_single.setHorizontalHeaderLabels(cols)
        self.table_single.setRowCount(len(metric_names))
        self.table_single.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_single.horizontalHeader().setStretchLastSection(True)

        for row_idx, m_name in enumerate(metric_names):
            # Gather scores across rounds
            scores: List[float] = []
            sample_res: Optional[SingleEvalResult] = None
            for r_num in range(1, self.total_rounds + 1):
                m_list = self.rounds_eval_map.get(r_num, {}).get(target_path, [])
                match = next((r for r in m_list if r.metric_name == m_name), None)
                if match:
                    scores.append(match.score)
                    if not sample_res:
                        sample_res = match

            if not sample_res:
                continue

            # Col 0: Name
            self.table_single.setItem(row_idx, 0, QTableWidgetItem(m_name))
            # Col 1: Mode
            self.table_single.setItem(row_idx, 1, QTableWidgetItem(sample_res.metric_mode))
            # Col 2: Direction
            dir_str = "▼ 越低越好" if sample_res.lower_better else "▲ 越高越好"
            self.table_single.setItem(row_idx, 2, QTableWidgetItem(dir_str))

            # Rounds scores
            for i, sc in enumerate(scores):
                col_i = 3 + i
                item_sc = QTableWidgetItem(f"{sc:.4f}")
                item_sc.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_single.setItem(row_idx, col_i, item_sc)

            # Math stats
            mean_val = float(np.mean(scores)) if scores else 0.0
            range_val = float(np.max(scores) - np.min(scores)) if scores else 0.0
            std_val = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

            col_mean = 3 + len(scores)
            col_rng = col_mean + 1
            col_std = col_mean + 2
            col_diag = col_mean + 3

            item_mean = QTableWidgetItem(f"{mean_val:.4f}")
            item_mean.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_mean.setFont(QFont("", -1, QFont.Weight.Bold))
            self.table_single.setItem(row_idx, col_mean, item_mean)

            item_rng = QTableWidgetItem(f"{range_val:.4f}")
            item_rng.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_single.setItem(row_idx, col_rng, item_rng)

            item_std = QTableWidgetItem(f"{std_val:.4f}")
            item_std.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_single.setItem(row_idx, col_std, item_std)

            # Diagnosis text
            if range_val < 1e-5:
                diag = "✅ 绝对确定性 (完全一致，无波动)"
                color = QColor(16, 185, 129)
            elif range_val < 0.02:
                diag = f"👌 微幅数值扰动 (极差={range_val:.4f})"
                color = QColor(59, 130, 246)
            else:
                diag = f"⚠️ 存在浮动变化 (极差={range_val:.4f}, σ={std_val:.4f})"
                color = QColor(225, 29, 72)

            item_diag = QTableWidgetItem(diag)
            item_diag.setForeground(color)
            self.table_single.setItem(row_idx, col_diag, item_diag)

        for c in range(self.table_single.columnCount() - 1):
            self.table_single.resizeColumnToContents(c)

    def _setup_multi_image_summary_table(self) -> None:
        """Populate summary PK table comparing mean values, range, and std across images."""
        first_round = self.rounds_eval_map.get(1, {})
        first_path = self.image_paths[0] if self.image_paths else ""
        first_results = first_round.get(first_path, [])
        metric_names = [r.metric_name for r in first_results]

        cols = ["指标名称", "模式", "倾向"]
        for idx, p in enumerate(self.image_paths):
            fname = os.path.basename(p)
            short_name = (fname[:10] + "..") if len(fname) > 12 else fname
            cols.append(f"图 {idx+1} 均值 (μ)")
            cols.append(f"图 {idx+1} 极差")
            cols.append(f"图 {idx+1} 标准差 (σ)")

        cols.extend(["🏆 平均值 PK 胜出者", "各图稳定性综合评价"])

        self.table_multi_summary.setColumnCount(len(cols))
        self.table_multi_summary.setHorizontalHeaderLabels(cols)
        self.table_multi_summary.setRowCount(len(metric_names))
        self.table_multi_summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_multi_summary.horizontalHeader().setStretchLastSection(True)

        num_imgs = len(self.image_paths)

        for row_idx, m_name in enumerate(metric_names):
            # Gather results per image across rounds
            img_scores_list: List[List[float]] = []
            sample_res: Optional[SingleEvalResult] = None

            for p in self.image_paths:
                scores_for_p: List[float] = []
                for r_num in range(1, self.total_rounds + 1):
                    m_list = self.rounds_eval_map.get(r_num, {}).get(p, [])
                    match = next((r for r in m_list if r.metric_name == m_name), None)
                    if match:
                        scores_for_p.append(match.score)
                        if not sample_res:
                            sample_res = match
                img_scores_list.append(scores_for_p)

            if not sample_res:
                continue

            lower_better = sample_res.lower_better

            # Basic info
            self.table_multi_summary.setItem(row_idx, 0, QTableWidgetItem(m_name))
            self.table_multi_summary.setItem(row_idx, 1, QTableWidgetItem(sample_res.metric_mode))
            dir_str = "▼ 越低越好" if lower_better else "▲ 越高越好"
            self.table_multi_summary.setItem(row_idx, 2, QTableWidgetItem(dir_str))

            # Compute stats per image
            means: List[float] = []
            ranges: List[float] = []
            stds: List[float] = []

            for img_idx, sc_list in enumerate(img_scores_list):
                m_val = float(np.mean(sc_list)) if sc_list else 0.0
                r_val = float(np.max(sc_list) - np.min(sc_list)) if sc_list else 0.0
                s_val = float(np.std(sc_list, ddof=1)) if len(sc_list) > 1 else 0.0
                means.append(m_val)
                ranges.append(r_val)
                stds.append(s_val)

                base_col = 3 + img_idx * 3
                item_m = QTableWidgetItem(f"{m_val:.4f}")
                item_m.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_m.setFont(QFont("", -1, QFont.Weight.Bold))
                self.table_multi_summary.setItem(row_idx, base_col, item_m)

                item_r = QTableWidgetItem(f"{r_val:.4f}")
                item_r.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_multi_summary.setItem(row_idx, base_col + 1, item_r)

                item_s = QTableWidgetItem(f"{s_val:.4f}")
                item_s.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_multi_summary.setItem(row_idx, base_col + 2, item_s)

            # Determine best mean and worst mean
            best_idx: Optional[int] = None
            worst_idx: Optional[int] = None
            if means:
                if lower_better:
                    best_m = min(means)
                    worst_m = max(means)
                else:
                    best_m = max(means)
                    worst_m = min(means)

                best_tied = [i for i, m in enumerate(means) if abs(m - best_m) < 1e-5]
                if len(best_tied) == 1:
                    best_idx = best_tied[0]

                if num_imgs >= 3 and abs(best_m - worst_m) > 1e-5:
                    worst_tied = [i for i, m in enumerate(means) if abs(m - worst_m) < 1e-5]
                    if len(worst_tied) == 1:
                        worst_idx = worst_tied[0]

            # Highlight best mean cell in green, worst in red
            for img_idx in range(num_imgs):
                base_col = 3 + img_idx * 3
                item_m = self.table_multi_summary.item(row_idx, base_col)
                if item_m:
                    if best_idx is not None and img_idx == best_idx:
                        item_m.setBackground(QColor(39, 174, 96, 40))
                        item_m.setForeground(QColor(39, 174, 96))
                    elif num_imgs >= 3 and worst_idx is not None and img_idx == worst_idx:
                        item_m.setBackground(QColor(225, 29, 72, 35))
                        item_m.setForeground(QColor(225, 29, 72))

            # Col: Mean PK Winner
            col_winner = 3 + num_imgs * 3
            if best_idx is not None:
                w_text = f"🏆 图 {best_idx + 1} (均值: {means[best_idx]:.4f})"
                if num_imgs >= 3 and worst_idx is not None:
                    w_text += f" | 📉 图 {worst_idx + 1} 垫底"
                item_w = QTableWidgetItem(w_text)
                item_w.setForeground(QColor(39, 174, 96))
                item_w.setFont(QFont("", -1, QFont.Weight.Bold))
            else:
                item_w = QTableWidgetItem("🤝 均值平局")
            item_w.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_multi_summary.setItem(row_idx, col_winner, item_w)

            # Col: Stability Evaluation
            col_eval = col_winner + 1
            max_r = max(ranges) if ranges else 0.0
            if max_r < 1e-5:
                eval_text = "✅ 各图结果绝对一致"
                eval_color = QColor(16, 185, 129)
            elif max_r < 0.02:
                eval_text = f"👌 极小幅微动 (最大极差={max_r:.4f})"
                eval_color = QColor(59, 130, 246)
            else:
                eval_text = f"⚠️ 各图均存在小幅波动 (最大极差={max_r:.4f})"
                eval_color = QColor(225, 29, 72)
            item_eval = QTableWidgetItem(eval_text)
            item_eval.setForeground(eval_color)
            self.table_multi_summary.setItem(row_idx, col_eval, item_eval)

        for c in range(self.table_multi_summary.columnCount() - 1):
            self.table_multi_summary.resizeColumnToContents(c)

    def _setup_multi_image_details_table(self) -> None:
        """Populate detailed table listing every round's score for every image and metric."""
        first_round = self.rounds_eval_map.get(1, {})
        first_path = self.image_paths[0] if self.image_paths else ""
        first_results = first_round.get(first_path, [])
        metric_names = [r.metric_name for r in first_results]

        cols = ["测试轮次", "指标名称", "模式", "倾向"]
        for idx, p in enumerate(self.image_paths):
            cols.append(f"图 {idx+1} 得分")
        cols.append("本轮最优图像")

        total_rows = self.total_rounds * len(metric_names)
        self.table_multi_details.setColumnCount(len(cols))
        self.table_multi_details.setHorizontalHeaderLabels(cols)
        self.table_multi_details.setRowCount(total_rows)
        self.table_multi_details.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_multi_details.horizontalHeader().setStretchLastSection(True)

        current_row = 0
        for r_num in range(1, self.total_rounds + 1):
            round_dict = self.rounds_eval_map.get(r_num, {})
            for m_name in metric_names:
                # Gather scores this round across images
                scores_this_round: List[Optional[float]] = []
                sample_res: Optional[SingleEvalResult] = None
                for p in self.image_paths:
                    m_list = round_dict.get(p, [])
                    match = next((r for r in m_list if r.metric_name == m_name), None)
                    if match:
                        scores_this_round.append(match.score)
                        if not sample_res:
                            sample_res = match
                    else:
                        scores_this_round.append(None)

                if not sample_res:
                    continue

                lower_better = sample_res.lower_better

                self.table_multi_details.setItem(current_row, 0, QTableWidgetItem(f"第 {r_num} 轮"))
                self.table_multi_details.setItem(current_row, 1, QTableWidgetItem(m_name))
                self.table_multi_details.setItem(current_row, 2, QTableWidgetItem(sample_res.metric_mode))
                dir_str = "▼ 越低越好" if lower_better else "▲ 越高越好"
                self.table_multi_details.setItem(current_row, 3, QTableWidgetItem(dir_str))

                valid_scores = [(idx, s) for idx, s in enumerate(scores_this_round) if s is not None]
                best_idx: Optional[int] = None
                if valid_scores:
                    if lower_better:
                        b_s = min(s for _, s in valid_scores)
                    else:
                        b_s = max(s for _, s in valid_scores)
                    tied = [idx for idx, s in valid_scores if abs(s - b_s) < 1e-5]
                    if len(tied) == 1:
                        best_idx = tied[0]

                for img_idx, s in enumerate(scores_this_round):
                    col_idx = 4 + img_idx
                    if s is not None:
                        item_s = QTableWidgetItem(f"{s:.4f}")
                        item_s.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        if best_idx is not None and img_idx == best_idx:
                            item_s.setBackground(QColor(39, 174, 96, 35))
                            item_s.setForeground(QColor(39, 174, 96))
                            item_s.setFont(QFont("", -1, QFont.Weight.Bold))
                        self.table_multi_details.setItem(current_row, col_idx, item_s)
                    else:
                        self.table_multi_details.setItem(current_row, col_idx, QTableWidgetItem("N/A"))

                winner_col = 4 + len(self.image_paths)
                if best_idx is not None:
                    w_item = QTableWidgetItem(f"🏆 图 {best_idx + 1}")
                    w_item.setForeground(QColor(39, 174, 96))
                    w_item.setFont(QFont("", -1, QFont.Weight.Bold))
                else:
                    w_item = QTableWidgetItem("🤝 平局")
                w_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_multi_details.setItem(current_row, winner_col, w_item)

                current_row += 1

        for c in range(self.table_multi_details.columnCount() - 1):
            self.table_multi_details.resizeColumnToContents(c)

    def _copy_to_clipboard(self) -> str:
        """Copy active table content to clipboard."""
        table = self.active_table
        if not table:
            return ""

        headers = []
        for c in range(table.columnCount()):
            h_item = table.horizontalHeaderItem(c)
            headers.append(h_item.text() if h_item else f"Col{c}")

        lines = ["\t".join(headers)]
        for r in range(table.rowCount()):
            row_vals = []
            for c in range(table.columnCount()):
                item = table.item(r, c)
                row_vals.append(item.text().replace("\t", " ") if item else "")
            lines.append("\t".join(row_vals))

        text = "\n".join(lines)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            cb = QApplication.clipboard()
            if cb is not None:
                cb.setText(text)
            QMessageBox.information(self, "复制成功", "已将多轮测试统计聚合数据复制到剪贴板！可直接粘贴至 Excel 中。")
        return text

    def _export_to_csv(self) -> None:
        """Export active table to CSV file."""
        table = self.active_table
        if not table:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出多轮统计报表",
            "IQA_Multi_Round_Summary.csv",
            "CSV 文件 (*.csv)",
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                headers = []
                for c in range(table.columnCount()):
                    h_item = table.horizontalHeaderItem(c)
                    headers.append(h_item.text() if h_item else f"Col{c}")
                writer.writerow(headers)

                for r in range(table.rowCount()):
                    row_vals = []
                    for c in range(table.columnCount()):
                        item = table.item(r, c)
                        row_vals.append(item.text() if item else "")
                    writer.writerow(row_vals)
            QMessageBox.information(self, "导出成功", f"统计报表已成功导出至:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出 CSV 报表出错: {e}")
