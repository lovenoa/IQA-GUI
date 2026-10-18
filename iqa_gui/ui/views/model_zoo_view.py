"""Tab 5: Model Zoo, metadata inspector, and weights cache management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.config import apply_hf_mirror, load_settings, save_settings
from iqa_gui.core.evaluator import EvaluatorEngine
from iqa_gui.core.model_manager import (
    CATEGORY_NAMES,
    MetricInfo,
    ModelManager,
)
from iqa_gui.core.worker_threads import PreloadWeightsWorker, SafePackageInstallerWorker


class ModelZooView(QWidget):
    """View for browsing the entire catalog of 106+ algorithms and managing local weights."""

    def __init__(self, evaluator_engine: EvaluatorEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.evaluator = evaluator_engine
        self.model_manager = ModelManager()
        self.preload_worker: Optional[PreloadWeightsWorker] = None
        self._current_filtered_models: List[MetricInfo] = []

        self._init_ui()
        self.refresh_catalog()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Upper settings banner: Hugging Face mirror & cache status
        settings_card = QFrame()
        settings_card.setProperty("class", "CardFrame")
        set_layout = QHBoxLayout(settings_card)
        set_layout.setContentsMargins(12, 8, 12, 8)

        self.chk_mirror = QCheckBox("启用国内 Hugging Face 镜像加速 (https://hf-mirror.com)")
        settings = load_settings()
        self.chk_mirror.setChecked(settings.get("hf_mirror_enabled", True))
        self.chk_mirror.toggled.connect(self._on_mirror_toggled)

        self.lbl_cache_summary = QLabel("本地权重缓存: 检测中...")
        self.lbl_cache_summary.setProperty("class", "MutedText")
        self.lbl_cache_summary.setStyleSheet("font-weight: 600;")

        self.btn_preload_popular = QPushButton("预载常用算法权重 (LPIPS/BRISQUE/MUSIQ)")
        self.btn_preload_popular.clicked.connect(self._preload_popular)

        set_layout.addWidget(self.chk_mirror)
        set_layout.addStretch()
        set_layout.addWidget(self.lbl_cache_summary)
        set_layout.addWidget(self.btn_preload_popular)
        main_layout.addWidget(settings_card)

        # Splitter: Left Table & Right Details
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Model list and filter
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Filter row
        filter_row = QHBoxLayout()
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("搜索算法名称、缩写或中文描述...")
        self.edit_search.textChanged.connect(self.refresh_catalog)
        self.edit_search.setClearButtonEnabled(True)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("全部模式", "")
        self.combo_mode.addItem("全参考 (FR)", "FR")
        self.combo_mode.addItem("无参考 (NR)", "NR")
        self.combo_mode.currentIndexChanged.connect(self.refresh_catalog)

        self.combo_cat = QComboBox()
        self.combo_cat.addItem("全部分类", "")
        for cid, cname in CATEGORY_NAMES.items():
            self.combo_cat.addItem(cname, cid)
        self.combo_cat.currentIndexChanged.connect(self.refresh_catalog)

        filter_row.addWidget(self.edit_search, 2)
        filter_row.addWidget(self.combo_mode, 1)
        filter_row.addWidget(self.combo_cat, 2)
        left_layout.addLayout(filter_row)

        # Table of models
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["算法名", "中文全称", "模式", "方向", "典型分值", "权重状态"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        left_layout.addWidget(self.table, 1)

        splitter.addWidget(left_widget)

        # Right: Detail Inspector Card
        self.detail_card = QFrame()
        self.detail_card.setProperty("class", "CardFrame")
        detail_layout = QVBoxLayout(self.detail_card)
        detail_layout.setSpacing(8)

        self.lbl_detail_title = QLabel("<b>算法详情检视器</b>")
        self.lbl_detail_name = QLabel("请在左侧选择算法")
        self.lbl_detail_name.setProperty("class", "DetailTitle")

        self.txt_desc = QTextEdit()
        self.txt_desc.setReadOnly(True)

        self.btn_download_single = QPushButton("下载/加载此模型权重")
        self.btn_download_single.clicked.connect(self._download_selected)
        self.btn_download_single.setEnabled(False)

        self.btn_install_dep = QPushButton("一键安装所需组件")
        self.btn_install_dep.clicked.connect(self._install_selected_deps)
        self.btn_install_dep.setVisible(False)

        detail_layout.addWidget(self.lbl_detail_title)
        detail_layout.addWidget(self.lbl_detail_name)
        detail_layout.addWidget(self.txt_desc, 1)
        detail_layout.addWidget(self.btn_install_dep)
        detail_layout.addWidget(self.btn_download_single)

        splitter.addWidget(self.detail_card)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter, 1)

    def _on_mirror_toggled(self, checked: bool) -> None:
        settings = load_settings()
        settings["hf_mirror_enabled"] = checked
        save_settings(settings)
        apply_hf_mirror(checked)
        status = "已启用国内镜像加速 (https://hf-mirror.com)" if checked else "已切换回 Hugging Face 官方源"
        QMessageBox.information(self, "网络镜像设置更新", status)

    def refresh_catalog(self) -> None:
        """Filter and update model table."""
        search = self.edit_search.text().strip()
        mode = self.combo_mode.currentData() or None
        cat = self.combo_cat.currentData() or None

        self._current_filtered_models = self.model_manager.filter(mode=mode, category=cat, search_query=search)
        self.table.setRowCount(len(self._current_filtered_models))

        for row, info in enumerate(self._current_filtered_models):
            # Name
            self.table.setItem(row, 0, QTableWidgetItem(info.name))
            # Display Name
            self.table.setItem(row, 1, QTableWidgetItem(info.display_name))
            # Mode
            mode_item = QTableWidgetItem(info.mode)
            mode_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, mode_item)
            # Direction
            dir_item = QTableWidgetItem("▼ 越低越好" if info.lower_better else "▲ 越高越好")
            dir_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, dir_item)
            # Range
            range_item = QTableWidgetItem(info.score_range)
            range_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, range_item)
            # Dependency & Weight Status
            dep_status = self.model_manager.check_dependencies(info.name)
            if not dep_status.is_ready:
                status_item = QTableWidgetItem(f"缺组件: {', '.join(dep_status.missing_required)}")
                status_item.setForeground(Qt.GlobalColor.darkRed)
            elif not info.requires_weights:
                status_item = QTableWidgetItem("免权重")
                status_item.setForeground(Qt.GlobalColor.darkGray)
            else:
                cached_status = self.model_manager.is_cached(info.name)
                if cached_status is True:
                    status_item = QTableWidgetItem("已下载")
                    status_item.setForeground(Qt.GlobalColor.darkGreen)
                elif cached_status is False:
                    status_item = QTableWidgetItem("待下载")
                    status_item.setForeground(Qt.GlobalColor.darkYellow)
                else:
                    # 无法确定状态的算法不显示下载状态，显示横杠 "-"
                    status_item = QTableWidgetItem("-")
                    status_item.setForeground(Qt.GlobalColor.gray)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 5, status_item)

        self._update_cache_summary()

    def _update_cache_summary(self) -> None:
        cached_files = self.model_manager.get_cached_weights()
        cache_dir = self.model_manager.cache_dir
        total_size_mb = 0.0
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                if f.is_file():
                    total_size_mb += f.stat().st_size / (1024 * 1024)
        self.lbl_cache_summary.setText(f"本地权重缓存: {len(cached_files)} 个文件 ({total_size_mb:.1f} MB)")

    def _on_table_selection(self) -> None:
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            self.btn_download_single.setEnabled(False)
            self.btn_install_dep.setVisible(False)
            return

        row = selected_rows[0].row()
        if 0 <= row < len(self._current_filtered_models):
            info = self._current_filtered_models[row]
            self.lbl_detail_name.setText(info.display_name)
            dep_status = self.model_manager.check_dependencies(info.name)

            if dep_status.is_ready:
                dep_html = "<span style='color: #22c55e;'>依赖包已就绪</span>"
            else:
                dep_html = f"<span style='color: #ef4444;'>缺少必要组件: <b>{', '.join(dep_status.missing_required)}</b></span>"

            if dep_status.missing_optional:
                dep_html += f"<br><span style='color: #888888;'>可选组件: 未安装 {', '.join(dep_status.missing_optional)}</span>"

            details_html = (
                f"<b>内部标识:</b> <code>{info.name}</code><br>"
                f"<b>评估模式:</b> {info.mode} ({'全参考' if info.mode == 'FR' else '无参考'})<br>"
                f"<b>所属分类:</b> {CATEGORY_NAMES.get(info.category, info.category)}<br>"
                f"<b>得分方向:</b> {info.direction_str}<br>"
                f"<b>分值范围:</b> {info.score_range}<br>"
                f"<b>运行依赖:</b> {dep_html}<br>"
                f"<b>预训练权重:</b> {'需要下载或本地提供' if info.requires_weights else '免权重 (数学/统计计算)'}<br>"
                f"<b>标签:</b> {', '.join(info.tags)}<br><br>"
                f"<b>【功能与原理说明】</b><br>{info.description_zh}"
            )
            self.txt_desc.setHtml(details_html)
            self.btn_download_single.setEnabled(info.requires_weights and dep_status.is_ready)
            self.btn_install_dep.setVisible(not dep_status.is_ready)

    def _install_selected_deps(self) -> None:
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        info = self._current_filtered_models[selected_rows[0].row()]
        dep_status = self.model_manager.check_dependencies(info.name)
        if dep_status.is_ready:
            return

        packages = dep_status.missing_required
        progress_dialog = QProgressDialog(f"正在安全安装组件: {', '.join(packages)}...", "后台运行", 0, 0, self)
        progress_dialog.setWindowTitle("安全安装扩展组件")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.show()

        worker = SafePackageInstallerWorker(packages, parent=self)
        self._installer_worker = worker
        worker.progress.connect(lambda txt: progress_dialog.setLabelText(f"正在安装: {txt[:40]}..."))

        def on_finished(success: bool, msg: str):
            progress_dialog.close()
            self.model_manager.clear_dependency_cache()
            if success:
                QMessageBox.information(self, "组件安装完成", msg)
            else:
                QMessageBox.warning(self, "组件安装提示", msg)
            self.refresh_catalog()
            self._on_table_selection()

        worker.finished.connect(on_finished)
        worker.start()

    def _download_selected(self) -> None:
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        info = self._current_filtered_models[selected_rows[0].row()]
        self._start_preload([info.name])

    def _preload_popular(self) -> None:
        popular = ["lpips", "brisque", "musiq", "dists"]
        self._start_preload(popular)

    def _start_preload(self, metrics: List[str]) -> None:
        self.btn_preload_popular.setEnabled(False)
        self.btn_download_single.setEnabled(False)

        self.preload_worker = PreloadWeightsWorker(evaluator=self.evaluator, metrics=metrics)
        self.preload_worker.progress.connect(lambda msg, cur, tot: self.lbl_cache_summary.setText(f"{msg} ({cur}/{tot})"))
        self.preload_worker.finished.connect(self._on_preload_finished)
        self.preload_worker.start()

    def _on_preload_finished(self, succeeded: list, failed: list) -> None:
        self.btn_preload_popular.setEnabled(True)
        self.btn_download_single.setEnabled(True)
        self.refresh_catalog()

        msg = f"权重载入完成。成功: {len(succeeded)} 个"
        if failed:
            msg += f"，失败: {len(failed)} 个 ({', '.join(f[0] for f in failed)})"
        QMessageBox.information(self, "模型权重更新", msg)
