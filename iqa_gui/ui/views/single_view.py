"""Tab 1: Single image and image-pair quality assessment view with Anime & AIGC quality scorecard."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.config import load_settings
from iqa_gui.core.evaluator import (
    EvaluatorEngine,
    SingleEvalResult,
    calculate_grade,
    compute_normalized_quality,
    generate_difference_heatmap,
)
from iqa_gui.core.model_manager import ModelManager
from iqa_gui.core.worker_threads import SingleEvalWorker
from iqa_gui.ui.components.compare_dialog import MultiImageCompareDialog
from iqa_gui.ui.components.multi_round_dialog import MultiRoundSummaryDialog
from iqa_gui.ui.components.image_viewer import ImageViewerCard
from iqa_gui.ui.components.metric_selector import MetricSelectorWidget


def interpret_perceptual_score(
    metric_name: str,
    score: float,
    score_range: Optional[str] = None,
    lower_better: bool = False,
) -> Tuple[str, str]:
    """Provide domain quality rating (Tier, Explanation) for perceptual and aesthetic quality assessment."""
    q = compute_normalized_quality(score, score_range, lower_better, metric_name)
    return calculate_grade(score, q, metric_name)


interpret_anime_aigc_score = interpret_perceptual_score  # 兼容别名


class SingleEvalView(QWidget):
    """View for evaluating single images, image pairs, or multiple comparison targets across multiple rounds."""

    def __init__(self, evaluator_engine: EvaluatorEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.evaluator = evaluator_engine
        self.model_manager = ModelManager()
        self.active_worker: Optional[SingleEvalWorker] = None

        # State management
        self.eval_results_map: Dict[str, List[SingleEvalResult]] = {}
        self.rounds_eval_map: Dict[int, Dict[str, List[SingleEvalResult]]] = {}
        self.current_round: int = 1
        self.default_order_results: List[SingleEvalResult] = []
        self.latest_results: List[SingleEvalResult] = []
        self.sort_state: Optional[Tuple[int, bool]] = None  # (column_index, ascending)
        self.base_headers = ["算法", "模式", "得分", "倾向", "评级", "指标综合诊断"]

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---------------- Left Control Panel ----------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # 1. Metric Selector (Multi-select enabled with presets and selected chips)
        self.metric_selector = MetricSelectorWidget(multi_select=True, parent=self)
        self.metric_selector.selection_changed.connect(self._on_metric_changed)
        left_layout.addWidget(self.metric_selector, 3)

        # 2. Prompt input for CLIPScore / Text-Image metrics
        self.prompt_frame = QFrame()
        self.prompt_frame.setProperty("class", "CardFrame")
        prompt_layout = QVBoxLayout(self.prompt_frame)
        prompt_layout.setContentsMargins(6, 6, 6, 6)
        prompt_layout.setSpacing(4)
        self.lbl_prompt = QLabel("<b>生图提示词 (Prompt - 供 CLIPScore 校验契合度):</b>")
        self.edit_prompt = QLineEdit()
        self.edit_prompt.setPlaceholderText("例如: 1girl, masterpiece, anime aesthetic, highly detailed")
        prompt_layout.addWidget(self.lbl_prompt)
        prompt_layout.addWidget(self.edit_prompt)
        self.prompt_frame.hide()
        left_layout.addWidget(self.prompt_frame)

        # 3. Action & Multi-Round Row
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("开始评估")
        self.btn_run.setProperty("class", "PrimaryButton")
        self.btn_run.setMinimumHeight(32)
        self.btn_run.clicked.connect(self.run_evaluation)
        run_row.addWidget(self.btn_run, 1)

        self.chk_multi_round = QCheckBox("启用多轮")
        self.chk_multi_round.setToolTip("重复执行指定轮数，统计极差、标准差与均值")
        self.spin_rounds = QSpinBox()
        self.spin_rounds.setRange(2, 20)
        self.spin_rounds.setValue(3)
        self.spin_rounds.setSuffix(" 次")
        self.spin_rounds.setFixedWidth(68)
        self.spin_rounds.setEnabled(False)
        self.chk_multi_round.toggled.connect(self.spin_rounds.setEnabled)
        run_row.addWidget(self.chk_multi_round)
        run_row.addWidget(self.spin_rounds)
        left_layout.addLayout(run_row)

        self.btn_open_compare = QPushButton("打开多图对比报表")
        self.btn_open_compare.setProperty("class", "SecondaryButton")
        self.btn_open_compare.setMinimumHeight(30)
        self.btn_open_compare.clicked.connect(self.open_comparison_dialog)
        self.btn_open_compare.hide()
        left_layout.addWidget(self.btn_open_compare)

        self.btn_open_multi_round = QPushButton("查看多轮统计报表")
        self.btn_open_multi_round.setProperty("class", "SecondaryButton")
        self.btn_open_multi_round.setMinimumHeight(30)
        self.btn_open_multi_round.clicked.connect(self.open_multi_round_dialog)
        self.btn_open_multi_round.hide()
        left_layout.addWidget(self.btn_open_multi_round)

        # 4. Result Scorecard Panel
        self.result_card = QFrame()
        self.result_card.setProperty("class", "CardFrame")
        card_layout = QVBoxLayout(self.result_card)
        card_layout.setContentsMargins(6, 6, 6, 6)
        card_layout.setSpacing(4)

        header_res = QHBoxLayout()
        self.lbl_result_title = QLabel("<b>评估结果看板</b>")
        self.lbl_eval_status = QLabel("就绪")
        self.lbl_eval_status.setProperty("class", "MutedText")
        header_res.addWidget(self.lbl_result_title)
        header_res.addStretch()

        # Multi-round selector dropdown in scorecard header
        self.round_switch_widget = QWidget()
        round_switch_layout = QHBoxLayout(self.round_switch_widget)
        round_switch_layout.setContentsMargins(0, 0, 0, 0)
        round_switch_layout.setSpacing(4)
        self.lbl_round_select = QLabel("<b>查看轮次:</b>")
        self.combo_round_select = QComboBox()
        self.combo_round_select.currentIndexChanged.connect(self._on_round_selection_changed)
        round_switch_layout.addWidget(self.lbl_round_select)
        round_switch_layout.addWidget(self.combo_round_select)
        self.round_switch_widget.hide()
        header_res.addWidget(self.round_switch_widget)

        header_res.addWidget(self.lbl_eval_status)

        # Button to restore preview when hidden
        self.btn_show_preview = QPushButton("展开预览")
        self.btn_show_preview.setToolTip("重新展开右侧图像预览画布")
        self.btn_show_preview.clicked.connect(lambda: self.set_preview_visible(True))
        self.btn_show_preview.hide()
        header_res.addWidget(self.btn_show_preview)

        card_layout.addLayout(header_res)

        # Multi-metric results table
        self.table_scorecard = QTableWidget(self)
        self.table_scorecard.setColumnCount(len(self.base_headers))
        self.table_scorecard.setHorizontalHeaderLabels(self.base_headers)
        self.table_scorecard.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_scorecard.horizontalHeader().setStretchLastSection(True)
        self.table_scorecard.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_scorecard.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table_scorecard.horizontalHeader().sectionClicked.connect(self._on_header_section_clicked)
        self.table_scorecard.itemSelectionChanged.connect(self._on_table_selection_changed)
        card_layout.addWidget(self.table_scorecard, 3)

        # Explanation / Guidance banner
        self.lbl_explanation = QLabel("提示: 评估完成后点击上方表格中的算法行，可在此查看该算法的技术原理与公式依据。点击表头可正序/倒序/默认排序。")
        self.lbl_explanation.setProperty("class", "InfoBanner")
        self.lbl_explanation.setWordWrap(True)
        card_layout.addWidget(self.lbl_explanation)

        left_layout.addWidget(self.result_card, 3)
        splitter.addWidget(left_widget)

        # ---------------- Right Image Comparison Views ----------------
        self.tabs_view = QTabWidget()

        # Tab 1: Side-by-side comparison
        tab_side = QWidget()
        side_layout = QHBoxLayout(tab_side)
        side_layout.setContentsMargins(4, 4, 4, 4)
        side_layout.setSpacing(6)

        # Target viewer supports multi-image comparison (unlimited targets)
        self.viewer_target = ImageViewerCard(
            title="待评测目标图像 (Target)",
            placeholder_text="拖入待评估图像或点击选择 (支持多图横向对比PK)",
            multi_image_mode=True,
        )
        self.viewer_target.image_changed.connect(self._on_target_image_changed)
        self.viewer_target.images_updated.connect(self._on_target_images_updated)

        self.viewer_ref = ImageViewerCard(
            title="基准参考图像 (Reference)",
            placeholder_text="全参考(FR)模式需提供无失真原图 (无参考模式自动隐藏)",
            multi_image_mode=False,
        )

        # Synchronize pan and zoom between Target and Reference
        self.viewer_target.sync_requested.connect(self.viewer_ref.sync_from)
        self.viewer_ref.sync_requested.connect(self.viewer_target.sync_from)

        side_layout.addWidget(self.viewer_target, 1)
        side_layout.addWidget(self.viewer_ref, 1)
        self.tabs_view.addTab(tab_side, "图像预览 (Preview)")

        # Tab 2: Difference heatmap
        tab_diff = QWidget()
        diff_layout = QVBoxLayout(tab_diff)
        diff_layout.setContentsMargins(4, 4, 4, 4)
        self.viewer_diff = ImageViewerCard(
            title="差异分布 (Difference Heatmap)",
            placeholder_text="生成全参考评估后自动渲染差异分布",
            multi_image_mode=False,
        )
        diff_layout.addWidget(self.viewer_diff)
        self.tabs_view.addTab(tab_diff, "差异分布 (Difference)")

        # Close preview button on the top-right corner of preview tabs
        self.btn_close_preview = QPushButton("关闭预览")
        self.btn_close_preview.setToolTip("隐藏图像预览画布，最大化查看结果看板")
        self.btn_close_preview.clicked.connect(lambda: self.set_preview_visible(False))
        self.tabs_view.setCornerWidget(self.btn_close_preview, Qt.Corner.TopRightCorner)

        splitter.addWidget(self.tabs_view)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        main_layout.addWidget(splitter)

        # Default: Select anime & AIGC preset algorithms
        self.metric_selector.select_anime_presets()

    def set_preview_visible(self, visible: bool) -> None:
        """Toggle preview panel visibility to maximize results."""
        self.tabs_view.setVisible(visible)
        self.btn_show_preview.setVisible(not visible)

    def _on_metric_changed(self, selected: list) -> None:
        """Handle selection change in metric selector."""
        has_clipscore = any(m.lower() == "clipscore" for m in selected)
        if has_clipscore:
            self.prompt_frame.show()
        else:
            self.prompt_frame.hide()

        has_fr = False
        for m_name in selected:
            info = self.model_manager.get_model(m_name)
            if info and info.mode == "FR":
                has_fr = True
                break

        if has_fr:
            self.viewer_ref.lbl_title.setText("基准参考图像 (Reference - 包含全参考算法，需提供原图)")
            self.viewer_ref.setEnabled(True)
            self.viewer_ref.show()
            self.tabs_view.setTabText(0, "双图比对")
            self.tabs_view.setTabEnabled(1, True)
        else:
            self.viewer_ref.lbl_title.setText("基准参考图像 (已选算法均为无参考NR，无需原图)")
            self.viewer_ref.setEnabled(False)
            self.viewer_ref.hide()
            self.tabs_view.setTabText(0, "图像预览")
            self.tabs_view.setTabEnabled(1, False)

    def _on_target_image_changed(self, target_path: str) -> None:
        """Called when user selects a different image tab in target viewer."""
        current_map = self.rounds_eval_map.get(self.current_round, self.eval_results_map)

        if not target_path or target_path not in current_map:
            self.default_order_results = []
            self.latest_results = []
            self.sort_state = None
            self._apply_sorting_and_render()
            if target_path:
                fname = os.path.basename(target_path)
                self.lbl_eval_status.setText(f"已选中 [{fname}] (尚未评估，点击开始评估)")
            else:
                self.lbl_eval_status.setText("就绪: 拖入图片后点击评估")
            return

        self.default_order_results = list(current_map[target_path])
        self.sort_state = None
        self._apply_sorting_and_render()
        fname = os.path.basename(target_path)
        round_hint = f" [第 {self.current_round} 轮]" if len(self.rounds_eval_map) > 1 else ""
        self.lbl_eval_status.setText(f"展示中: [{fname}]{round_hint} (共 {len(self.default_order_results)} 项算法)")

    def _on_target_images_updated(self, paths: List[str]) -> None:
        """Called when images are added or removed from target canvas."""
        existing_keys = list(self.eval_results_map.keys())
        for k in existing_keys:
            if k not in paths:
                del self.eval_results_map[k]

        for r_num in list(self.rounds_eval_map.keys()):
            for k in list(self.rounds_eval_map[r_num].keys()):
                if k not in paths:
                    del self.rounds_eval_map[r_num][k]

        evaluated_count = len(self.eval_results_map)
        if evaluated_count >= 2:
            self.btn_open_compare.show()
            self.btn_open_compare.setEnabled(True)
            self.btn_open_compare.setText(f"打开多图比对报表 ({evaluated_count} 张图片)")
        else:
            self.btn_open_compare.hide()
            self.btn_open_compare.setEnabled(False)

        if len(self.rounds_eval_map) <= 1:
            self.btn_open_multi_round.hide()

    def open_comparison_dialog(self) -> None:
        """Open the multi-image comparison PK report dialog."""
        if len(self.eval_results_map) < 2:
            QMessageBox.information(self, "对比提示", "请至少评测 2 张待测图片后再查看横向比对报表。")
            return
        paths = [p for p in self.viewer_target.get_image_paths() if p in self.eval_results_map]
        dlg = MultiImageCompareDialog(paths, self.eval_results_map, self)
        dlg.exec()

    def open_multi_round_dialog(self) -> None:
        """Open the multi-round summary dialog."""
        paths = [p for p in self.viewer_target.get_image_paths() if any(p in self.rounds_eval_map.get(r, {}) for r in self.rounds_eval_map)]
        if not paths or len(self.rounds_eval_map) <= 1:
            QMessageBox.information(self, "提示", "请先执行多轮测试后再查看多轮统计聚合报表。")
            return
        dlg = MultiRoundSummaryDialog(paths, self.rounds_eval_map, self)
        dlg.exec()

    def _on_round_selection_changed(self, idx: int) -> None:
        """Handle user selecting a different round in the scorecard header combo."""
        if idx < 0:
            return
        round_num = idx + 1
        self.current_round = round_num

        cur_target = self.viewer_target.current_file_path
        if cur_target and self.rounds_eval_map.get(round_num, {}).get(cur_target):
            self.default_order_results = list(self.rounds_eval_map[round_num][cur_target])
            self._apply_sorting_and_render()
            fname = os.path.basename(cur_target)
            self.lbl_eval_status.setText(f"第 {round_num} 轮结果: [{fname}] (共 {len(self.default_order_results)} 项算法)")

    def _on_header_section_clicked(self, logical_index: int) -> None:
        """Handle header click for 3-state cycling: Descending -> Ascending -> Default order."""
        if not self.default_order_results:
            return

        if self.sort_state is None or self.sort_state[0] != logical_index:
            self.sort_state = (logical_index, False)  # Descending
        elif self.sort_state == (logical_index, False):
            self.sort_state = (logical_index, True)   # Ascending
        else:
            self.sort_state = None                   # Reset to default order

        self._apply_sorting_and_render()

    def _apply_sorting_and_render(self) -> None:
        """Sort and render scorecard table based on current sort_state."""
        if not self.default_order_results:
            self.latest_results = []
            self.table_scorecard.setRowCount(0)
            self._update_header_labels()
            return

        if self.sort_state is None:
            sorted_results = list(self.default_order_results)
        else:
            col_idx, ascending = self.sort_state
            if col_idx == 0:
                # 算法: Alphabetical order
                sorted_results = sorted(
                    self.default_order_results,
                    key=lambda r: r.metric_name.lower(),
                    reverse=(not ascending),
                )
            elif col_idx == 1:
                # 模式: FR vs NR
                sorted_results = sorted(
                    self.default_order_results,
                    key=lambda r: r.metric_mode,
                    reverse=(not ascending),
                )
            elif col_idx == 2:
                # 得分: Normalized quality index Q in [0, 1]
                sorted_results = sorted(
                    self.default_order_results,
                    key=lambda r: r.normalized_quality,
                    reverse=(not ascending),
                )
            elif col_idx == 3:
                # 倾向: 越高越好 vs 越低越好
                sorted_results = sorted(
                    self.default_order_results,
                    key=lambda r: 1 if r.lower_better else 0,
                    reverse=(not ascending),
                )
            elif col_idx == 4:
                # 评级: Grade rank S(4) > A(3) > B(2) > C(1)
                tier_order = {"S": 4, "A": 3, "B": 2, "C": 1}
                sorted_results = sorted(
                    self.default_order_results,
                    key=lambda r: tier_order.get(r.grade, 0),
                    reverse=(not ascending),
                )
            elif col_idx == 5:
                # 指标综合诊断
                sorted_results = sorted(
                    self.default_order_results,
                    key=lambda r: r.grade_desc,
                    reverse=(not ascending),
                )
            else:
                sorted_results = list(self.default_order_results)

        self.latest_results = sorted_results
        self._update_header_labels()
        self._render_scorecard_rows()

    def _update_header_labels(self) -> None:
        """Update column headers with sorting indicator arrows."""
        headers = []
        for i, title in enumerate(self.base_headers):
            if self.sort_state is not None and self.sort_state[0] == i:
                arrow = " ▲" if self.sort_state[1] else " ▼"
                headers.append(f"{title}{arrow}")
            else:
                headers.append(title)
        self.table_scorecard.setHorizontalHeaderLabels(headers)

    def _render_scorecard_rows(self) -> None:
        """Render rows into scorecard table."""
        total = len(self.latest_results)
        self.table_scorecard.setRowCount(total)

        s_count = 0
        a_count = 0
        fr_result_for_heatmap: Optional[SingleEvalResult] = None

        for row_idx, res in enumerate(self.latest_results):
            # Col 0: Metric Name
            item_name = QTableWidgetItem(res.metric_name)
            item_name.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.table_scorecard.setItem(row_idx, 0, item_name)

            # Col 1: Mode
            item_mode = QTableWidgetItem(res.metric_mode)
            item_mode.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_scorecard.setItem(row_idx, 1, item_mode)

            # Col 2: Score
            item_score = QTableWidgetItem(res.formatted_score)
            item_score.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_scorecard.setItem(row_idx, 2, item_score)

            # Col 3: Direction
            dir_str = "▼ 越低越好" if res.lower_better else "▲ 越高越好"
            item_dir = QTableWidgetItem(dir_str)
            item_dir.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_scorecard.setItem(row_idx, 3, item_dir)

            # Col 4: Tier badge
            tier = res.grade
            insight = res.grade_desc
            lbl_tier = QLabel(f" {tier}级 ")
            lbl_tier.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if tier == "S":
                lbl_tier.setProperty("class", "TierS")
                s_count += 1
            elif tier == "A":
                lbl_tier.setProperty("class", "TierA")
                a_count += 1
            elif tier == "B":
                lbl_tier.setProperty("class", "TierB")
            else:
                lbl_tier.setProperty("class", "TierC")
            self.table_scorecard.setCellWidget(row_idx, 4, lbl_tier)

            # Col 5: Insight
            item_insight = QTableWidgetItem(insight)
            item_insight.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.table_scorecard.setItem(row_idx, 5, item_insight)

            if res.metric_mode == "FR" and res.ref_path:
                fr_result_for_heatmap = res

        for c in range(5):
            self.table_scorecard.resizeColumnToContents(c)

        if total > 0:
            self.table_scorecard.selectRow(0)
            overall_text = "综合评级：S 级 (细节与画质表现良好)" if s_count >= 2 else (
                "综合评级：A 级 (水准良好，无明显瑕疵)" if (s_count + a_count) >= 2 else
                "综合评级：B 级 (整体合格，建议关注局部微小噪点或模糊)"
            )
            first_res = self.latest_results[0]
            info = self.model_manager.get_model(first_res.metric_name)
            desc = info.description_zh if info else ""
            self.lbl_explanation.setText(f"{overall_text}\n\n【{first_res.metric_name.upper()} 原理】{desc}")

        if fr_result_for_heatmap and fr_result_for_heatmap.ref_path:
            try:
                diff_pil = generate_difference_heatmap(
                    fr_result_for_heatmap.target_path,
                    fr_result_for_heatmap.ref_path,
                )
                data = diff_pil.tobytes("raw", "RGB")
                qimg = QImage(data, diff_pil.width, diff_pil.height, diff_pil.width * 3, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.viewer_diff.display_pixmap(pixmap)
            except Exception as e:
                print(f"Error rendering difference heatmap: {e}")

    def run_evaluation(self) -> None:
        """Trigger multi-metric evaluation across all loaded target images."""
        selected = self.metric_selector.get_selected()
        if not selected:
            QMessageBox.warning(self, "请选择算法", "请先在左侧选择至少一种图像质量评估算法。")
            return

        target_paths = self.viewer_target.get_image_paths()
        ref_path = self.viewer_ref.current_file_path
        prompt = self.edit_prompt.text().strip()

        if not target_paths:
            QMessageBox.warning(self, "缺少图像", "请先选择或拖入待评测目标图像！")
            return

        for p in target_paths:
            if not os.path.exists(p):
                QMessageBox.warning(self, "文件丢失", f"待测图像路径不存在:\n{p}")
                return

        # Check if any FR metric is selected without reference image
        for m_name in selected:
            info = self.model_manager.get_model(m_name)
            if info and info.mode == "FR" and (not ref_path or not os.path.exists(ref_path)):
                QMessageBox.warning(
                    self,
                    "缺少参考原图",
                    f"算法 '{m_name}' 属于全参考算法，必须提供对应的参考原图！\n请在右侧拖入参考图，或取消勾选该全参考算法。",
                )
                return

        # Check if clipscore is selected without prompt
        has_clipscore = any(m.lower() == "clipscore" for m in selected)
        if has_clipscore and not prompt:
            self.prompt_frame.show()
            self.edit_prompt.setFocus()
            QMessageBox.warning(
                self,
                "需要提供提示词 (Prompt)",
                "已选算法包含 'CLIPScore' (图文匹配度算法)。\n\n该算法需根据文字描述来衡量图像契合度，请在左侧【生图提示词 (Prompt)】输入框中填入文字描述，或取消勾选该算法。",
            )
            return

        # Prepare UI
        rounds = self.spin_rounds.value() if self.chk_multi_round.isChecked() else 1
        self.btn_run.setEnabled(False)
        self.btn_run.setText("正在计算...")
        total_evals = rounds * len(target_paths) * len(selected)
        round_hint = f"共 {rounds} 轮, " if rounds > 1 else ""
        self.lbl_eval_status.setText(f"正在计算 ({round_hint}1/{total_evals}) ...")
        self.lbl_explanation.setText("正在执行深度学习算法评估，请稍候...")
        self.table_scorecard.setRowCount(0)

        settings = load_settings()
        auto_release = settings.get("auto_release_vram", True)

        # Start worker thread
        self.active_worker = SingleEvalWorker(
            evaluator=self.evaluator,
            metrics=selected,
            target_path=target_paths,
            ref_path=ref_path,
            prompt=prompt,
            rounds=rounds,
            auto_release_vram=auto_release,
        )
        self.active_worker.progress.connect(self._on_worker_progress)
        self.active_worker.rounds_finished.connect(self._on_rounds_eval_finished)
        self.active_worker.multi_finished.connect(self._on_multi_eval_finished)
        self.active_worker.finished.connect(self._on_eval_finished)
        self.active_worker.error.connect(self._on_eval_error)
        self.active_worker.start()

    def _on_worker_progress(self, current: int, total: int, metric_name: str) -> None:
        self.lbl_eval_status.setText(f"正在计算 ({current}/{total}): {metric_name} ...")

    def _on_rounds_eval_finished(self, multi_round_map: Dict[int, Dict[str, List[SingleEvalResult]]]) -> None:
        """Handle completion of multi-round evaluation."""
        self.rounds_eval_map = multi_round_map
        total_rounds = len(multi_round_map)

        if total_rounds > 1:
            self.combo_round_select.blockSignals(True)
            self.combo_round_select.clear()
            for r in range(1, total_rounds + 1):
                self.combo_round_select.addItem(f"第 {r} 轮")
            self.combo_round_select.setCurrentIndex(0)
            self.combo_round_select.blockSignals(False)
            self.current_round = 1
            self.round_switch_widget.show()
            self.btn_open_multi_round.show()
        else:
            self.round_switch_widget.hide()
            self.btn_open_multi_round.hide()

    def _on_multi_eval_finished(self, eval_map: Dict[str, List[SingleEvalResult]]) -> None:
        """Handle evaluation completion for all targets."""
        self.eval_results_map.update(eval_map)
        evaluated_count = len(self.eval_results_map)
        if evaluated_count >= 2:
            self.btn_open_compare.show()
            self.btn_open_compare.setEnabled(True)
            self.btn_open_compare.setText(f"打开多图比对报表 ({evaluated_count} 张图片)")
        else:
            self.btn_open_compare.hide()
            self.btn_open_compare.setEnabled(False)

    def _on_eval_finished(self, results: List[SingleEvalResult]) -> None:
        """Handle evaluation completion."""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("开始评估")

        cur_target = self.viewer_target.current_file_path
        current_map = self.rounds_eval_map.get(self.current_round, self.eval_results_map)

        if cur_target and cur_target in current_map:
            self.default_order_results = list(current_map[cur_target])
        else:
            self.default_order_results = list(results)

        self.sort_state = None
        self._apply_sorting_and_render()

        fname = os.path.basename(cur_target) if cur_target else "目标图像"
        total = len(self.default_order_results)
        round_hint = f" [第 {self.current_round} 轮]" if len(self.rounds_eval_map) > 1 else ""
        self.lbl_eval_status.setText(f"完成: [{fname}]{round_hint} ({total} 项)")

        # Refresh VRAM status on main window
        main_win = self.window()
        if hasattr(main_win, "update_vram_display"):
            main_win.update_vram_display()

    def _on_table_selection_changed(self) -> None:
        """When user clicks a row in the scorecard table, show its technical details."""
        selected_rows = self.table_scorecard.selectionModel().selectedRows()
        if not selected_rows or not self.latest_results:
            return
        row = selected_rows[0].row()
        if 0 <= row < len(self.latest_results):
            res = self.latest_results[row]
            info = self.model_manager.get_model(res.metric_name)
            desc = info.description_zh if info else ""
            self.lbl_explanation.setText(
                f"【{res.metric_name} 结果: {res.formatted_score} ({res.direction_hint}) | {res.grade}级】\n"
                f"{res.grade_desc}\n\n"
                f"【标准化质量】{res.normalized_quality * 100:.1f} / 100\n\n"
                f"【算法原理】{desc}"
            )

    def set_tips_visible(self, visible: bool) -> None:
        """Toggle visibility of operational hints."""
        self.lbl_explanation.setVisible(visible)
        self.metric_selector.set_tips_visible(visible)

    def _on_eval_error(self, err_msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_run.setText("开始评估")
        self.lbl_eval_status.setText("评估出错")
        self.lbl_explanation.setText(f"评估出错: {err_msg}")

        # Refresh VRAM status on main window
        main_win = self.window()
        if hasattr(main_win, "update_vram_display"):
            main_win.update_vram_display()

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("评估失败")
        msg_box.setText(f"计算图像质量时发生错误:\n{err_msg}\n\n您可以点击下方按钮打开后台日志监视器查看具体工作与排查信息。")
        btn_open_log = msg_box.addButton("查看日志", QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton("确定", QMessageBox.ButtonRole.AcceptRole)
        msg_box.exec()

        if msg_box.clickedButton() == btn_open_log:
            main_win = self.window()
            if hasattr(main_win, "open_log_viewer"):
                main_win.open_log_viewer()
