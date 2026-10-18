"""Tab 3: gMAD (generalized Minimum/Maximum Added Difference) competition analysis view."""

from __future__ import annotations

import os
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.core.evaluator import EvaluatorEngine
from iqa_gui.core.gmad_engine import GMADEngine, GMADPairResult, GMADResult
from iqa_gui.core.model_manager import ModelManager
from iqa_gui.core.worker_threads import GMADWorker
from iqa_gui.ui.components.image_viewer import ImageViewerCard


class GMADView(QWidget):
    """View for candidate-pool gMAD metric competition and failure-mode discovery."""

    def __init__(self, evaluator_engine: EvaluatorEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.evaluator = evaluator_engine
        self.gmad_engine = GMADEngine(evaluator=self.evaluator)
        self.model_manager = ModelManager()
        self.active_worker: Optional[GMADWorker] = None

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Config Card
        config_card = QFrame()
        config_card.setProperty("class", "CardFrame")
        cfg_layout = QVBoxLayout(config_card)
        cfg_layout.setSpacing(8)

        # Introduction banner
        lbl_intro = QLabel(
            "<b>gMAD 算法对抗分析说明 (通俗解释)：</b><br>"
            "让两种无参考画质算法打擂台！系统会自动在候选图片池中找出两种算法发生<b>最大分歧</b>的极端图对："
            "（1）算法 A 认为画质几乎相同，但算法 B 判定差距悬殊；（2）算法 B 认为画质相同，但算法 A 判定差距悬殊。"
            "可一针见血地探测出某个质量评估模型的评价盲区与偏见。"
        )
        lbl_intro.setProperty("class", "InfoBanner")
        lbl_intro.setWordWrap(True)
        cfg_layout.addWidget(lbl_intro)

        grid = QGridLayout()
        grid.setSpacing(8)

        # Candidate Directory
        lbl_dir = QLabel("候选图片池目录:")
        self.edit_dir = QLineEdit()
        self.edit_dir.setPlaceholderText("选择包含至少 2 张测试图片的文件夹...")
        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self._browse_dir)

        grid.addWidget(lbl_dir, 0, 0)
        grid.addWidget(self.edit_dir, 0, 1, 1, 3)
        grid.addWidget(self.btn_browse, 0, 4)

        # Metric A & Metric B
        lbl_a = QLabel("对抗算法 A (无参考 NR):")
        self.combo_a = QComboBox()

        lbl_b = QLabel("对抗算法 B (无参考 NR):")
        self.combo_b = QComboBox()

        # Populate NR metrics
        nr_models = self.model_manager.filter(mode="NR")
        for m in nr_models:
            self.combo_a.addItem(f"{m.name} ({m.display_name})", m.name)
            self.combo_b.addItem(f"{m.name} ({m.display_name})", m.name)

        # Default select musiq and brisque
        idx_musiq = self.combo_a.findData("musiq")
        if idx_musiq >= 0:
            self.combo_a.setCurrentIndex(idx_musiq)
        idx_brisque = self.combo_b.findData("brisque")
        if idx_brisque >= 0:
            self.combo_b.setCurrentIndex(idx_brisque)

        grid.addWidget(lbl_a, 1, 0)
        grid.addWidget(self.combo_a, 1, 1)
        grid.addWidget(lbl_b, 1, 2)
        grid.addWidget(self.combo_b, 1, 3)

        # Tie Tolerance
        lbl_tol = QLabel("判定相同容差 (Tolerance):")
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(0.01, 5.0)
        self.spin_tol.setSingleStep(0.05)
        self.spin_tol.setValue(0.10)
        self.spin_tol.setToolTip("标准化分差在此阈值内即视为两图质量平手/相等")

        self.btn_run = QPushButton("⚡ 开始 gMAD 对抗分析")
        self.btn_run.setProperty("class", "PrimaryButton")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self.start_gmad)

        grid.addWidget(lbl_tol, 2, 0)
        grid.addWidget(self.spin_tol, 2, 1)
        grid.addWidget(self.btn_run, 2, 2, 1, 2)

        cfg_layout.addLayout(grid)
        main_layout.addWidget(config_card)

        # Status label
        self.lbl_status = QLabel("状态: 待启动")
        self.lbl_status.setProperty("class", "MutedText")
        self.lbl_status.setStyleSheet("font-weight: 600;")
        main_layout.addWidget(self.lbl_status)

        # Scroll area for comparison pairs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        # Result Section 1: Metric A ties, Metric B differentiates
        self.card_pair_a = self._create_pair_card("对决结果 1：算法 A 判定画质一致，但算法 B 判定差距最大")
        scroll_layout.addWidget(self.card_pair_a["frame"])

        # Result Section 2: Metric B ties, Metric A differentiates
        self.card_pair_b = self._create_pair_card("对决结果 2：算法 B 判定画质一致，但算法 A 判定差距最大")
        scroll_layout.addWidget(self.card_pair_b["frame"])

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll, 1)

    def _create_pair_card(self, title: str) -> dict:
        frame = QFrame()
        frame.setProperty("class", "CardFrame")
        layout = QVBoxLayout(frame)
        layout.setSpacing(8)

        lbl_title = QLabel(f"<b>{title}</b>")
        lbl_info = QLabel("尚未运算")
        lbl_info.setProperty("class", "MutedText")

        img_layout = QHBoxLayout()
        v1 = ImageViewerCard(title="对抗图片 1", placeholder_text="极值图片 1")
        v2 = ImageViewerCard(title="对抗图片 2", placeholder_text="极值图片 2")
        v1.setMaximumHeight(280)
        v2.setMaximumHeight(280)
        img_layout.addWidget(v1, 1)
        img_layout.addWidget(v2, 1)

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_info)
        layout.addLayout(img_layout)

        return {
            "frame": frame,
            "title": lbl_title,
            "info": lbl_info,
            "v1": v1,
            "v2": v2,
        }

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 gMAD 候选图片文件夹")
        if path:
            self.edit_dir.setText(path)

    def start_gmad(self) -> None:
        """Collect images and launch gMAD analysis."""
        folder = self.edit_dir.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "路径错误", "请先选择有效的候选图片文件夹！")
            return

        from iqa_gui.core.batch_evaluator import BatchEvaluator
        images = BatchEvaluator.scan_directory(folder)
        if len(images) < 2:
            QMessageBox.warning(self, "图片数量不足", "gMAD 分析至少需要文件夹中包含 2 张有效图片。")
            return

        metric_a = self.combo_a.currentData()
        metric_b = self.combo_b.currentData()
        if metric_a == metric_b:
            QMessageBox.warning(self, "算法重复", "请选择两个不同的无参考算法进行对抗比对。")
            return

        tol = self.spin_tol.value()

        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ 正在打擂比对中...")
        self.lbl_status.setText("正在执行候选池全量打分与标准化...")

        self.active_worker = GMADWorker(
            gmad_engine=self.gmad_engine,
            image_paths=images,
            metric_a=metric_a,
            metric_b=metric_b,
            tolerance=tol,
        )
        self.active_worker.progress.connect(lambda msg, cur, tot: self.lbl_status.setText(f"{msg} ({cur}/{tot})"))
        self.active_worker.finished.connect(self._on_gmad_finished)
        self.active_worker.error.connect(self._on_gmad_error)
        self.active_worker.start()

    def _on_gmad_finished(self, res: GMADResult) -> None:
        self.btn_run.setEnabled(True)
        self.btn_run.setText("⚡ 开始 gMAD 对抗分析")
        self.lbl_status.setText(f"✅ 对抗分析完成！共检索 {res.total_images} 张候选图像。")

        # Update Pair A
        if res.pair_a_tie:
            p = res.pair_a_tie
            self.card_pair_a["info"].setText(
                f"【{res.metric_a.upper()} 认为相差微小】分差仅: {p.tie_gap:.4f} (容差≤{res.tolerance})  |  "
                f"【{res.metric_b.upper()} 发生最大分歧】分差高达: {p.target_gap:.4f}\n"
                f"图1 ({p.image_1.filename}): A得分={p.image_1.raw_score_a:.2f}, B得分={p.image_1.raw_score_b:.2f}  |  "
                f"图2 ({p.image_2.filename}): A得分={p.image_2.raw_score_a:.2f}, B得分={p.image_2.raw_score_b:.2f}"
            )
            self.card_pair_a["v1"].load_image(p.image_1.image_path)
            self.card_pair_a["v2"].load_image(p.image_2.image_path)
        else:
            self.card_pair_a["info"].setText(f"在容差 {res.tolerance} 下未找到符合算法 A 相等条件的候选图对。")

        # Update Pair B
        if res.pair_b_tie:
            p = res.pair_b_tie
            self.card_pair_b["info"].setText(
                f"【{res.metric_b.upper()} 认为相差微小】分差仅: {p.tie_gap:.4f} (容差≤{res.tolerance})  |  "
                f"【{res.metric_a.upper()} 发生最大分歧】分差高达: {p.target_gap:.4f}\n"
                f"图1 ({p.image_1.filename}): A得分={p.image_1.raw_score_a:.2f}, B得分={p.image_1.raw_score_b:.2f}  |  "
                f"图2 ({p.image_2.filename}): A得分={p.image_2.raw_score_a:.2f}, B得分={p.image_2.raw_score_b:.2f}"
            )
            self.card_pair_b["v1"].load_image(p.image_1.image_path)
            self.card_pair_b["v2"].load_image(p.image_2.image_path)
        else:
            self.card_pair_b["info"].setText(f"在容差 {res.tolerance} 下未找到符合算法 B 相等条件的候选图对。")

    def _on_gmad_error(self, err_msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_run.setText("⚡ 开始 gMAD 对抗分析")
        self.lbl_status.setText(f"❌ 运行失败: {err_msg}")
        QMessageBox.critical(self, "gMAD 运算失败", f"分析失败:\n{err_msg}")
