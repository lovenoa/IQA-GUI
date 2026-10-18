"""Tab 1: Single image and image-pair quality assessment view with Anime & AIGC quality scorecard."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.config import load_settings
from iqa_gui.core.evaluator import EvaluatorEngine, SingleEvalResult, generate_difference_heatmap
from iqa_gui.core.model_manager import ModelManager
from iqa_gui.core.worker_threads import SingleEvalWorker
from iqa_gui.ui.components.image_viewer import ImageViewerCard
from iqa_gui.ui.components.metric_selector import MetricSelectorWidget


def interpret_perceptual_score(metric_name: str, score: float) -> Tuple[str, str]:
    """Provide domain quality rating (Tier, Explanation) for perceptual and aesthetic quality assessment."""
    m = metric_name.lower()
    if m == "laion_aes":
        if score >= 7.0:
            return "S", "🌟 【S级 大师级美学】艺术质感极佳，色彩光影绝美，美学水准出众。"
        elif score >= 6.0:
            return "A", "✨ 【A级 优质画面】画面干净精致，构图与色调和谐，质感良好。"
        elif score >= 5.0:
            return "B", "👌 【B级 达标画质】整体合格，可能存在轻微色彩平淡或局部细节较普通。"
        else:
            return "C", "⚠️ 【C级 偏低/瑕疵】美学评分偏低，画面可能存在杂色、暗沉或构图失衡。"
    elif "clipiqa" in m:
        if score >= 0.70:
            return "S", "🌟 【S级 高清精修】纹理细腻，线条清晰锐利，符合高品质视觉先验。"
        elif score >= 0.55:
            return "A", "✨ 【A级 良好画质】质感良好，画面层次清晰完整。"
        elif score >= 0.40:
            return "B", "👌 【B级 画质一般】细节中规中矩，无明显亮点。"
        else:
            return "C", "⚠️ 【C级 线条模糊/噪点】可能存在线条发虚、涂抹感重或局部细节缺失。"
    elif "musiq" in m:
        if score >= 70.0:
            return "S", "🌟 【S级 全局超清】多尺度细节极为丰富，无拉伸或失真损耗。"
        elif score >= 58.0:
            return "A", "✨ 【A级 清晰优秀】不同观看尺度下清晰度均维持在较高水准。"
        elif score >= 45.0:
            return "B", "👌 【B级 清晰达标】基本满足常规分辨率观看要求。"
        else:
            return "C", "⚠️ 【C级 清晰度不足】在全图视角下细节略显软绵或存在压缩失真。"
    elif m == "maniqa":
        if score >= 0.65:
            return "S", "🌟 【S级 结构高度精准】主体边缘与纹理细节感知极为稳定。"
        elif score >= 0.50:
            return "A", "✨ 【A级 结构正常】未检测到明显的纹理崩解或严重形变。"
        elif score >= 0.35:
            return "B", "👌 【B级 结构可接受】存在轻微不自然线条或伪影。"
        else:
            return "C", "⚠️ 【C级 局部可能畸变】建议放大检查主体边缘线条或复杂纹理细节。"
    elif m == "clipscore":
        if score >= 28.0:
            return "S", "🎯 【S级 提示词高契合度】画面内容与文字提示词高度对齐！"
        elif score >= 22.0:
            return "A", "👌 【A级 契合度良好】主体内容与提示词相符，修饰细节部分体现。"
        else:
            return "B", "⚠️ 【B/C级 契合度较低】画面内容与提示词存在偏差，存在偏题可能。"
    elif m in ["psnr", "ssim", "lpips", "dists", "fsim"]:
        if m == "psnr":
            tier = "S" if score >= 35 else ("A" if score >= 28 else "B")
            return tier, f"峰值信噪比: {score:.2f} dB (全参考)"
        elif m == "ssim":
            tier = "S" if score >= 0.95 else ("A" if score >= 0.85 else "B")
            return tier, f"结构相似度: {score:.4f} (全参考)"
        elif m == "lpips":
            tier = "S" if score <= 0.10 else ("A" if score <= 0.25 else "B")
            return tier, f"深度感知差异: {score:.4f} (全参考)"
    return "-", "已完成基础画质测算"

interpret_anime_aigc_score = interpret_perceptual_score  # 兼容别名


class SingleEvalView(QWidget):
    """View for evaluating a single image (NR) or an image pair (FR) across multiple metrics."""

    def __init__(self, evaluator_engine: EvaluatorEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.evaluator = evaluator_engine
        self.model_manager = ModelManager()
        self.active_worker: Optional[SingleEvalWorker] = None
        self.latest_results: List[SingleEvalResult] = []

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

        # 1. Metric Selector (Multi-select enabled with anime presets and selected chips)
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

        # 3. Action Button & Progress Status
        self.btn_run = QPushButton("🚀 立即开始质量评估")
        self.btn_run.setProperty("class", "PrimaryButton")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self.run_evaluation)
        left_layout.addWidget(self.btn_run)

        # 4. Result Scorecard Panel (High density table + explanation)
        self.result_card = QFrame()
        self.result_card.setProperty("class", "CardFrame")
        card_layout = QVBoxLayout(self.result_card)
        card_layout.setContentsMargins(6, 6, 6, 6)
        card_layout.setSpacing(4)

        header_res = QHBoxLayout()
        self.lbl_result_title = QLabel("<b>综合画质体检看板 (Quality Scorecard)</b>")
        self.lbl_eval_status = QLabel("就绪: 拖入图片后点击评估")
        self.lbl_eval_status.setProperty("class", "MutedText")
        header_res.addWidget(self.lbl_result_title)
        header_res.addStretch()
        header_res.addWidget(self.lbl_eval_status)
        card_layout.addLayout(header_res)

        # Multi-metric results table
        self.table_scorecard = QTableWidget(self)
        self.table_scorecard.setColumnCount(6)
        self.table_scorecard.setHorizontalHeaderLabels(["算法", "模式", "得分", "倾向", "评级", "指标综合诊断"])
        self.table_scorecard.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_scorecard.horizontalHeader().setStretchLastSection(True)
        self.table_scorecard.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_scorecard.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table_scorecard.itemSelectionChanged.connect(self._on_table_selection_changed)
        card_layout.addWidget(self.table_scorecard, 3)

        # Explanation / Guidance banner
        self.lbl_explanation = QLabel("提示: 评估完成后点击上方表格中的算法行，可在此查看该算法的技术原理与公式依据。")
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

        self.viewer_target = ImageViewerCard(title="待评测目标图像 (Target)", placeholder_text="拖入待评估图像或点击右上角选择文件")
        self.viewer_ref = ImageViewerCard(title="基准参考图像 (Reference)", placeholder_text="全参考(FR)模式需提供无失真原图 (无参考模式自动隐藏)")

        # Synchronize pan and zoom between Target and Reference
        self.viewer_target.sync_requested.connect(self.viewer_ref.sync_from)
        self.viewer_ref.sync_requested.connect(self.viewer_target.sync_from)

        side_layout.addWidget(self.viewer_target, 1)
        side_layout.addWidget(self.viewer_ref, 1)
        self.tabs_view.addTab(tab_side, "🔍 双图联动同步比对 (Side-by-Side)")

        # Tab 2: Difference heatmap
        tab_diff = QWidget()
        diff_layout = QVBoxLayout(tab_diff)
        diff_layout.setContentsMargins(4, 4, 4, 4)
        self.viewer_diff = ImageViewerCard(title="像素级差异热力图 (Difference Heatmap)", placeholder_text="生成全参考评估后自动渲染差异分布")
        diff_layout.addWidget(self.viewer_diff)
        self.tabs_view.addTab(tab_diff, "🔥 差异分布热力图 (Difference)")

        splitter.addWidget(self.tabs_view)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        main_layout.addWidget(splitter)

        # Default: Select anime & AIGC preset algorithms
        self.metric_selector.select_anime_presets()

    def _on_metric_changed(self, selected: list) -> None:
        """Handle selection change in metric selector."""
        # Toggle prompt input if clipscore is in selected metrics
        has_clipscore = any(m.lower() == "clipscore" for m in selected)
        if has_clipscore:
            self.prompt_frame.show()
        else:
            self.prompt_frame.hide()

        # Check if any selected metric is Full Reference (FR)
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
            self.tabs_view.setTabText(0, "🔍 双图联动同步比对 (Side-by-Side)")
            self.tabs_view.setTabEnabled(1, True)
        else:
            self.viewer_ref.lbl_title.setText("基准参考图像 (已选算法均为无参考NR，无需原图)")
            self.viewer_ref.setEnabled(False)
            self.viewer_ref.hide()
            self.tabs_view.setTabText(0, "🖼️ 单图无参考深度画布 (Target)")
            self.tabs_view.setTabEnabled(1, False)

    def run_evaluation(self) -> None:
        """Trigger multi-metric evaluation."""
        selected = self.metric_selector.get_selected()
        if not selected:
            QMessageBox.warning(self, "请选择算法", "请先在左侧选择至少一种图像质量评估算法。")
            return

        tar_path = self.viewer_target.current_file_path
        ref_path = self.viewer_ref.current_file_path
        prompt = self.edit_prompt.text().strip()

        if not tar_path or not os.path.exists(tar_path):
            QMessageBox.warning(self, "缺少图像", "请先选择或拖入待评测目标图像！")
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
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ 正在计算中...")
        self.lbl_eval_status.setText(f"正在计算 1/{len(selected)}: {selected[0]} ...")
        self.lbl_explanation.setText("正在执行深度学习算法评估，请稍候...")
        self.table_scorecard.setRowCount(0)

        settings = load_settings()
        auto_release = settings.get("auto_release_vram", True)

        # Start worker thread
        self.active_worker = SingleEvalWorker(
            evaluator=self.evaluator,
            metrics=selected,
            target_path=tar_path,
            ref_path=ref_path,
            prompt=prompt,
            auto_release_vram=auto_release,
        )
        self.active_worker.progress.connect(self._on_worker_progress)
        self.active_worker.finished.connect(self._on_eval_finished)
        self.active_worker.error.connect(self._on_eval_error)
        self.active_worker.start()

    def _on_worker_progress(self, current: int, total: int, metric_name: str) -> None:
        self.lbl_eval_status.setText(f"正在计算 ({current}/{total}): {metric_name} ...")

    def _on_eval_finished(self, results: List[SingleEvalResult]) -> None:
        self.btn_run.setEnabled(True)
        self.btn_run.setText("🚀 立即开始质量评估")
        self.latest_results = results

        total = len(results)
        self.lbl_eval_status.setText(f"评估完成: 共 {total} 项算法")
        self.table_scorecard.setRowCount(total)

        s_count = 0
        a_count = 0
        fr_result_for_heatmap: Optional[SingleEvalResult] = None

        for row_idx, res in enumerate(results):
            # Column 0: Metric Name
            item_name = QTableWidgetItem(res.metric_name)
            item_name.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.table_scorecard.setItem(row_idx, 0, item_name)

            # Column 1: Mode
            item_mode = QTableWidgetItem(res.metric_mode)
            item_mode.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_scorecard.setItem(row_idx, 1, item_mode)

            # Column 2: Score
            item_score = QTableWidgetItem(res.formatted_score)
            item_score.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_scorecard.setItem(row_idx, 2, item_score)

            # Column 3: Direction
            dir_str = "▼ 越低越好" if res.lower_better else "▲ 越高越好"
            item_dir = QTableWidgetItem(dir_str)
            item_dir.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_scorecard.setItem(row_idx, 3, item_dir)

            # Column 4: Tier badge
            tier, insight = interpret_anime_aigc_score(res.metric_name, res.score)
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

            # Column 5: Anime/AIGC Insight
            item_insight = QTableWidgetItem(insight)
            item_insight.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.table_scorecard.setItem(row_idx, 5, item_insight)

            if res.metric_mode == "FR" and res.ref_path:
                fr_result_for_heatmap = res

        # Resize columns cleanly
        self.table_scorecard.resizeColumnToContents(0)
        self.table_scorecard.resizeColumnToContents(1)
        self.table_scorecard.resizeColumnToContents(2)
        self.table_scorecard.resizeColumnToContents(3)
        self.table_scorecard.resizeColumnToContents(4)

        # Select first row by default to trigger explanation display
        if total > 0:
            self.table_scorecard.selectRow(0)
            overall_text = "🌟 综合评级：S 级 (卓越画质，光影美学与细节表现极佳)" if s_count >= 2 else (
                "✨ 综合评级：A 级 (优质水准，细节良好，无明显结构瑕疵)" if (s_count + a_count) >= 2 else
                "👌 综合评级：B 级 (整体合格，建议关注局部微小噪点或模糊)"
            )
            first_res = results[0]
            info = self.model_manager.get_model(first_res.metric_name)
            desc = info.description_zh if info else ""
            self.lbl_explanation.setText(f"{overall_text}\n\n【{first_res.metric_name.upper()} 原理】{desc}")

        # Render difference heatmap if any FR metric was evaluated with reference
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
            _, insight = interpret_perceptual_score(res.metric_name, res.score)
            self.lbl_explanation.setText(
                f"【{res.metric_name.upper()} 评估结果: {res.formatted_score} ({res.direction_hint})】\n"
                f"{insight}\n\n"
                f"【算法原理】{desc}"
            )

    def set_tips_visible(self, visible: bool) -> None:
        """Toggle visibility of operational hints."""
        self.lbl_explanation.setVisible(visible)
        self.metric_selector.set_tips_visible(visible)

    def _on_eval_error(self, err_msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_run.setText("🚀 立即开始质量评估")
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
        btn_open_log = msg_box.addButton("📋 打开后台运行日志", QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton("确定", QMessageBox.ButtonRole.AcceptRole)
        msg_box.exec()

        if msg_box.clickedButton() == btn_open_log:
            main_win = self.window()
            if hasattr(main_win, "open_log_viewer"):
                main_win.open_log_viewer()
