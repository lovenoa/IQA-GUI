"""High-density, professional metric selector widget using linear list view with Anime/AIGC presets and selected chips."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.config import SettingsSaveError, load_settings, save_settings
from iqa_gui.core.model_manager import (
    CATEGORY_NAMES,
    MetricInfo,
    ModelManager,
)
from iqa_gui.core.worker_threads import SafePackageInstallerWorker, launch_worker

# 常用预设算法组合
RECOMMENDED_AESTHETIC_PRESETS = [
    "laion_aes",    # 视觉美学评分 (1~10分)
    "clipiqa",      # 基于 CLIP 的零样本多模态画质打分
    "musiq",        # 多尺度 Transformer，自适应各类图像宽高比例
    "clipscore",    # 提示词(Prompt)与图像语义契合度
    "maniqa",       # NTIRE 冠军算法，精细感知局部与整体结构
    "qalign",       # 大视觉语言模型多模态质量评分
]
ANIME_AIGC_PRESET_METRICS = RECOMMENDED_AESTHETIC_PRESETS  # 兼容别名

GENERAL_CLASSIC_PRESETS = ["psnr", "ssim", "lpips", "brisque", "niqe"]


class MetricSelectorWidget(QFrame):
    """High-density, non-fragmenting linear list selector with presets, selected chips, and instant search."""
    selection_changed = Signal(list)  # Emits list of selected metric names

    def __init__(
        self,
        multi_select: bool = True,
        mode_filter: Optional[str] = None,  # None, 'FR', 'NR'
        parent: Optional[QWidget] = None,
        allowed_metrics: Optional[Iterable[str]] = None,
    ):
        super().__init__(parent)
        self.setProperty("class", "CardFrame")
        self.multi_select = multi_select
        self.mode_filter = mode_filter
        self.allowed_metrics = None if allowed_metrics is None else set(allowed_metrics)
        self._clicked_check_indicator = False
        self._installer_worker = None
        self.model_manager = ModelManager()

        self._selected_metrics: Set[str] = set()
        self._all_models: List[MetricInfo] = self.model_manager.list_all()

        self._init_ui()
        self.apply_filter()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Preset Selector & Management Bar
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)

        self.combo_presets = QComboBox()
        self.combo_presets.setToolTip("选择常用快捷算法组合或您保存的自定义组合")
        self._populate_presets_combo()
        self.combo_presets.currentIndexChanged.connect(self._on_preset_combo_changed)

        self.btn_save_preset = QPushButton("另存组合")
        self.btn_save_preset.setToolTip("将当前勾选的算法组合保存为新的快捷预设")
        self.btn_save_preset.clicked.connect(self._save_current_as_preset)

        self.btn_manage_preset = QPushButton("管理")
        self.btn_manage_preset.setToolTip("覆盖更新当前组合或删除自定义组合")
        self.btn_manage_preset.clicked.connect(self._show_preset_manage_menu)

        preset_layout.addWidget(self.combo_presets, 3)
        preset_layout.addWidget(self.btn_save_preset, 2)
        preset_layout.addWidget(self.btn_manage_preset, 2)

        if self.multi_select:
            self.btn_select_all = QPushButton("全选")
            self.btn_select_all.clicked.connect(self.select_all_visible)

            self.btn_clear = QPushButton("清空")
            self.btn_clear.clicked.connect(self.clear_selection)

            preset_layout.addWidget(self.btn_select_all)
            preset_layout.addWidget(self.btn_clear)
        else:
            preset_layout.addStretch()

        layout.addLayout(preset_layout)

        # 2. Search & Category Filter
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(4)

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("快速过滤 (如 aes, clip, musiq, ssim)...")
        self.edit_search.textChanged.connect(self.apply_filter)
        self.edit_search.setClearButtonEnabled(True)

        self.combo_category = QComboBox()
        self.combo_category.addItem("全部分类 (106+)", "")
        for cat_id, cat_name in CATEGORY_NAMES.items():
            short_name = cat_name.split("(")[0].strip()
            self.combo_category.addItem(short_name, cat_id)
        self.combo_category.currentIndexChanged.connect(self.apply_filter)

        filter_layout.addWidget(self.edit_search, 3)
        filter_layout.addWidget(self.combo_category, 2)
        layout.addLayout(filter_layout)

        # 3. Compact Sequential List (No empty holes, linear flow)
        self.list_widget = QListWidget(self)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.viewport().installEventFilter(self)
        layout.addWidget(self.list_widget, 1)

        # 4. Bottom Selected Chips & Summary Bar
        bottom_box = QVBoxLayout()
        bottom_box.setSpacing(3)

        summary_row = QHBoxLayout()
        self.lbl_count = QLabel("已选: 0 个算法")
        self.lbl_count.setStyleSheet("font-weight: 600;")
        self.lbl_hint = QLabel("提示: 悬停查看算法原理与打分建议")
        self.lbl_hint.setProperty("class", "MutedText")
        summary_row.addWidget(self.lbl_count)
        summary_row.addStretch()
        summary_row.addWidget(self.lbl_hint)
        bottom_box.addLayout(summary_row)

        # Chips scroll area (compact height, completely unblocked)
        self.chips_scroll = QScrollArea()
        self.chips_scroll.setFixedHeight(32)
        self.chips_scroll.setWidgetResizable(True)
        self.chips_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chips_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chips_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.chips_container = QWidget()
        self.chips_container.setStyleSheet("background: transparent;")
        self.chips_layout = QHBoxLayout(self.chips_container)
        self.chips_layout.setContentsMargins(2, 0, 2, 0)
        self.chips_layout.setSpacing(4)
        self.chips_scroll.setWidget(self.chips_container)
        bottom_box.addWidget(self.chips_scroll)

        layout.addLayout(bottom_box)

    def apply_filter(self) -> None:
        """Filter and repopulate the list linearly without any gaps."""
        search = self.edit_search.text().strip().lower()
        selected_category = self.combo_category.currentData()

        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        visible_count = 0
        for info in self._all_models:
            if self.allowed_metrics is not None and info.name not in self.allowed_metrics:
                continue
            # Mode filter
            if self.mode_filter and info.mode != self.mode_filter:
                continue

            # Category filter
            if selected_category and info.category != selected_category:
                continue

            # Search text
            if search:
                match = (
                    search in info.name.lower()
                    or search in info.display_name.lower()
                    or search in info.description_zh.lower()
                    or any(search in t.lower() for t in info.tags)
                )
                if not match:
                    continue

            # Build high-density item text and dependency badge
            dep_status = self.model_manager.check_dependencies(info.name)
            badge_dir = "▲" if not info.lower_better else "▼"
            display_title = info.display_name.split("(")[0].strip()

            if not dep_status.is_ready:
                missing_str = ",".join(dep_status.missing_required)
                item_text = f"[缺依赖] {info.name:<14} [{badge_dir} {info.mode}]  {display_title} (缺{missing_str})"
            else:
                item_text = f"{info.name:<18} [{badge_dir} {info.mode}]  {display_title}"

            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, info.name)

            tooltip = (
                f"【{info.display_name}】\n"
                f"算法标识: {info.name}\n"
                f"评估模式: {info.mode} ({'全参考 FR' if info.mode == 'FR' else '无参考 NR'})\n"
                f"评分倾向: {info.direction_str}\n"
                f"分值范围: {info.score_range}\n"
                f"权重需求: {'需下载权重' if info.requires_weights else '无需权重 (数学/统计)'}\n\n"
                f"{info.description_zh}"
            )
            if not dep_status.is_ready:
                tooltip += f"\n\n运行依赖未就绪: 缺少必要包 {', '.join(dep_status.missing_required)}。点击可查看安装引导。"
            else:
                tooltip += f"\n\n运行依赖: 依赖库已全部就绪。"

            item.setToolTip(tooltip)

            if self.multi_select:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                check_state = Qt.CheckState.Checked if info.name in self._selected_metrics else Qt.CheckState.Unchecked
                item.setCheckState(check_state)
            else:
                if info.name in self._selected_metrics:
                    item.setSelected(True)

            self.list_widget.addItem(item)
            visible_count += 1

        self.list_widget.blockSignals(False)
        self._update_summary_label()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if not self.multi_select:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            dep_status = self.model_manager.check_dependencies(name)
            if not dep_status.is_ready:
                self._selected_metrics.discard(name)
                self.list_widget.blockSignals(True)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.list_widget.blockSignals(False)
                self._prompt_missing_dependencies(name, dep_status, item)
                return
            self._selected_metrics.add(name)
        else:
            self._selected_metrics.discard(name)
        self._update_summary_label()
        self.selection_changed.emit(self.get_selected())

    def eventFilter(self, watched, event):
        if watched is self.list_widget.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            item = self.list_widget.itemAt(event.position().toPoint())
            self._clicked_check_indicator = False
            if item is not None and self.multi_select:
                option = QStyleOptionViewItem()
                option.initFrom(self.list_widget)
                option.rect = self.list_widget.visualItemRect(item)
                option.features |= QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
                rect = self.list_widget.style().subElementRect(
                    QStyle.SubElement.SE_ItemViewItemCheckIndicator, option, self.list_widget)
                self._clicked_check_indicator = rect.contains(event.position().toPoint())
        return super().eventFilter(watched, event)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        name = item.data(Qt.ItemDataRole.UserRole)
        if not self.multi_select:
            dep_status = self.model_manager.check_dependencies(name)
            if not dep_status.is_ready:
                self._prompt_missing_dependencies(name, dep_status, item)
                return
            self._selected_metrics = {name}
            self._update_summary_label()
            self.selection_changed.emit([name])
        elif not self._clicked_check_indicator:
            # Toggle check on click for effortless zero-code mouse operation
            new_state = Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
            item.setCheckState(new_state)

    def _prompt_missing_dependencies(self, name: str, dep_status: Any, item: QListWidgetItem) -> None:
        info = self.model_manager.get_model(name)
        disp_name = info.display_name if info else name
        missing_str = ", ".join(dep_status.missing_required)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("运行环境扩展组件提醒")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setText(
            f"<h3>算法【{disp_name}】需要安装支持组件</h3>"
            f"<p>检测到当前环境尚未安装：<b>{missing_str}</b></p>"
            f"<p><b>【环境说明】</b><br>"
            f"• 安装会约束当前已安装依赖的版本；发现版本冲突时停止，不自动替换现有核心包。<br>"
            f"• 这不能保证所有扩展与当前环境兼容；安装后请重启并验证运行环境。<br>"
            f"• 进度将在后台控制台输出。</p>"
            f"<p>是否立即安装？</p>"
        )
        btn_install = msg_box.addButton("一键安装", QMessageBox.ButtonRole.AcceptRole)
        btn_copy = msg_box.addButton("复制安装命令", QMessageBox.ButtonRole.ActionRole)
        btn_cancel = msg_box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        msg_box.setDefaultButton(btn_install)
        msg_box.exec()

        if msg_box.clickedButton() == btn_install:
            self._start_safe_install(dep_status.missing_required, name)
        else:
            if msg_box.clickedButton() == btn_copy:
                import os
                if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                    cb = QApplication.clipboard()
                    if cb is not None:
                        cb.setText(dep_status.safe_install_command)
                QMessageBox.information(self, "复制成功", "安装命令已复制到剪贴板，您可在终端中手动执行。")
            if self.multi_select:
                self.list_widget.blockSignals(True)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.list_widget.blockSignals(False)

    def _start_safe_install(self, packages: List[str], target_metric_name: str) -> None:
        if self._installer_worker is not None and self._installer_worker.isRunning():
            return
        progress_dialog = QProgressDialog(f"正在按当前依赖版本约束安装组件: {', '.join(packages)}...", "后台运行", 0, 0, self)
        progress_dialog.setWindowTitle("受约束安装扩展组件")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.show()

        worker = SafePackageInstallerWorker(packages, parent=self)
        self._installer_worker = worker
        controls = [self.list_widget, self.combo_presets, self.btn_save_preset, self.btn_manage_preset]
        if self.multi_select:
            controls.extend([self.btn_select_all, self.btn_clear])
        for control in controls:
            control.setEnabled(False)
        worker.progress.connect(lambda txt: progress_dialog.setLabelText(f"正在安装: {txt[:40]}..."))

        result = []
        worker.result_ready.connect(lambda success, msg: result.append((success, msg)))

        def on_finished():
            progress_dialog.close()
            for control in controls:
                control.setEnabled(True)
            self._installer_worker = None
            self.model_manager.clear_dependency_cache()
            success, msg = result[-1] if result else (False, "安装线程已结束，但未返回结果。")
            if success:
                QMessageBox.information(self, "组件安装完成", msg)
                names = self.get_selected() + [target_metric_name] if self.multi_select else [target_metric_name]
                self.set_selected(names)
            else:
                QMessageBox.warning(self, "组件安装提示", msg)
                self.apply_filter()

        worker.finished.connect(on_finished)
        if not launch_worker(worker, self):
            progress_dialog.close()
            for control in controls:
                control.setEnabled(True)
            self._installer_worker = None
            worker.deleteLater()


    def _update_summary_label(self) -> None:
        count = len(self._selected_metrics)
        self.lbl_count.setText(f"已选 ({count} 项):")

        # Clear existing chips
        while self.chips_layout.count() > 0:
            child = self.chips_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not self._selected_metrics:
            empty_lbl = QLabel("未勾选任何算法 (点击列表或上方预设)")
            empty_lbl.setProperty("class", "MutedText")
            self.chips_layout.addWidget(empty_lbl)
            self.chips_layout.addStretch()
            return

        for name in sorted(self._selected_metrics):
            chip = QPushButton(f"{name} ✕")
            chip.setProperty("class", "MetricChip")
            chip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setToolTip(f"点击取消勾选 '{name}'")
            chip.clicked.connect(lambda _, n=name: self._remove_selected(n))
            self.chips_layout.addWidget(chip)

        self.chips_layout.addStretch()

    def _remove_selected(self, metric_name: str) -> None:
        """Remove a single metric when chip ✕ is clicked."""
        self._selected_metrics.discard(metric_name)
        # Uncheck item in list widget if visible
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == metric_name:
                item.setCheckState(Qt.CheckState.Unchecked)
        self._update_summary_label()
        self.selection_changed.emit(self.get_selected())

    def get_selected(self) -> List[str]:
        return sorted(self._selected_metrics)

    def set_selected(self, metric_names: List[str]) -> None:
        allowed = {info.name for info in self._all_models
                   if (self.allowed_metrics is None or info.name in self.allowed_metrics)
                   and (not self.mode_filter or info.mode == self.mode_filter)}
        self._selected_metrics = {name for name in metric_names if name in allowed
                                  and self.model_manager.check_dependencies(name).is_ready}
        self.apply_filter()
        self.selection_changed.emit(self.get_selected())

    def set_tips_visible(self, visible: bool) -> None:
        """Toggle visibility of operational hints."""
        self.lbl_hint.setVisible(visible)

    def _populate_presets_combo(self, select_name: Optional[str] = None) -> None:
        """Populate preset combo from user settings."""
        self.combo_presets.blockSignals(True)
        self.combo_presets.clear()
        self.combo_presets.addItem("快捷算法组合 (点击切换)...", "")

        settings = load_settings()
        presets = settings.get("metric_presets", {})

        target_idx = 0
        for idx, (name, metrics) in enumerate(presets.items(), start=1):
            self.combo_presets.addItem(f"{name} ({len(metrics)}项)", name)
            if select_name and name == select_name:
                target_idx = idx

        if target_idx > 0:
            self.combo_presets.setCurrentIndex(target_idx)
        else:
            self.combo_presets.setCurrentIndex(0)

        self.combo_presets.blockSignals(False)

    def _on_preset_combo_changed(self, index: int) -> None:
        name = self.combo_presets.currentData()
        if not name:
            return
        settings = load_settings()
        presets = settings.get("metric_presets", {})
        if name in presets:
            metrics = presets[name]
            self.set_selected(metrics)

    def _save_current_as_preset(self) -> None:
        if not self._selected_metrics:
            QMessageBox.warning(self, "无法保存", "请先在列表中勾选要包含在该组合中的算法。")
            return

        name, ok = QInputDialog.getText(
            self,
            "新建快捷算法组合",
            f"将当前已选的 {len(self._selected_metrics)} 项算法保存为新组合。\n请输入组合名称:",
        )
        if ok and name.strip():
            preset_name = name.strip()
            settings = load_settings()
            if "metric_presets" not in settings:
                settings["metric_presets"] = {}
            settings["metric_presets"][preset_name] = sorted(list(self._selected_metrics))
            try:
                save_settings(settings)
            except SettingsSaveError as exc:
                QMessageBox.warning(self, "保存失败", str(exc))
                return
            self._populate_presets_combo(select_name=preset_name)
            QMessageBox.information(self, "保存成功", f"算法组合【{preset_name}】已成功保存。")

    def _show_preset_manage_menu(self) -> None:
        current_preset = self.combo_presets.currentData()
        menu = QMenu(self)

        action_update = menu.addAction("覆盖更新当前组合 (以当前勾选更新)")
        action_update.setEnabled(bool(current_preset))

        is_custom = bool(current_preset and current_preset not in ["综合感知与美学", "经典基础通用"])
        action_delete = menu.addAction(f"删除自定义组合【{current_preset or ''}】")
        action_delete.setEnabled(is_custom)

        action_new = menu.addAction("新建其他组合...")

        pos = self.btn_manage_preset.mapToGlobal(self.btn_manage_preset.rect().bottomLeft())
        selected = menu.exec(pos)

        if selected == action_update and current_preset:
            if not self._selected_metrics:
                QMessageBox.warning(self, "无法更新", "当前未勾选任何算法，无法更新组合。")
                return
            reply = QMessageBox.question(
                self,
                "确认覆盖更新",
                f"是否使用当前勾选的 {len(self._selected_metrics)} 项算法覆盖更新组合【{current_preset}】？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                settings = load_settings()
                settings.setdefault("metric_presets", {})[current_preset] = sorted(list(self._selected_metrics))
                try:
                    save_settings(settings)
                except SettingsSaveError as exc:
                    QMessageBox.warning(self, "保存失败", str(exc))
                    return
                self._populate_presets_combo(select_name=current_preset)
                QMessageBox.information(self, "更新成功", f"组合【{current_preset}】已更新为当前选择。")

        elif selected == action_delete and current_preset:
            reply = QMessageBox.question(
                self,
                "确认删除",
                f"确定要删除自定义组合【{current_preset}】吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                settings = load_settings()
                if "metric_presets" in settings and current_preset in settings["metric_presets"]:
                    del settings["metric_presets"][current_preset]
                    try:
                        save_settings(settings)
                    except SettingsSaveError as exc:
                        QMessageBox.warning(self, "保存失败", str(exc))
                        return
                    self._populate_presets_combo()
                    QMessageBox.information(self, "删除成功", f"组合【{current_preset}】已删除。")

        elif selected == action_new:
            self._save_current_as_preset()

    def select_anime_presets(self) -> None:
        """Select default aesthetic & perceptual preset."""
        settings = load_settings()
        metrics = settings.get("metric_presets", {}).get("综合感知与美学", RECOMMENDED_AESTHETIC_PRESETS)
        self.set_selected(metrics)
        self._populate_presets_combo(select_name="综合感知与美学")

    def select_classic_presets(self) -> None:
        """Select general classical IQA metrics."""
        settings = load_settings()
        metrics = settings.get("metric_presets", {}).get("经典基础通用", GENERAL_CLASSIC_PRESETS)
        self.set_selected(metrics)
        self._populate_presets_combo(select_name="经典基础通用")

    def select_recommended(self) -> None:
        """Alias for selecting recommended metrics."""
        self.select_anime_presets()

    def select_all_visible(self) -> None:
        """Select all visible items."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            if self.model_manager.check_dependencies(name).is_ready:
                self._selected_metrics.add(name)
        self.set_selected(self.get_selected())

    def clear_selection(self) -> None:
        """Clear all selections."""
        self._selected_metrics.clear()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)
        self._update_summary_label()
        self.selection_changed.emit([])
