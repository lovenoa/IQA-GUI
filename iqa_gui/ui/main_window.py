"""Main application window for IQA-GUI with hardware diagnostics and dual-theme switching."""

from __future__ import annotations

import sys
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.config import get_effective_device, load_settings, save_settings
from iqa_gui.core.evaluator import EvaluatorEngine
from iqa_gui.core.hardware import detect_hardware
from iqa_gui.ui.theme import THEME_MODE_AUTO, THEME_MODE_DARK, THEME_MODE_LIGHT, apply_theme_to_app
from iqa_gui.ui.views.batch_view import BatchEvalView
from iqa_gui.ui.views.benchmark_view import BenchmarkView
from iqa_gui.ui.views.gmad_view import GMADView
from iqa_gui.ui.views.model_zoo_view import ModelZooView
from iqa_gui.ui.views.single_view import SingleEvalView


class MainWindow(QMainWindow):
    """Main application window uniting all 5 primary functional views."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("IQA-GUI - 图像质量评估工具")
        self.resize(1260, 820)
        self.setMinimumSize(980, 620)

        # Detect Hardware
        self.hw_info = detect_hardware()

        # Settings
        self.settings = load_settings()
        preferred_device = self.settings.get("preferred_device", "auto")
        self.effective_device = get_effective_device(preferred_device)
        self.evaluator_engine = EvaluatorEngine(device=self.effective_device)

        self._init_ui()

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # ---------------- Top Header Bar (High density pro toolbar) ----------------
        header_frame = QFrame()
        header_frame.setObjectName("HeaderBar")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(8)

        # Title
        title_layout = QVBoxLayout()
        title_layout.setSpacing(1)
        lbl_title = QLabel("IQA-PyTorch 图像质量评估工具")
        lbl_title.setObjectName("AppTitle")
        lbl_subtitle = QLabel("支持 106+ 项图像质量评估算法")
        lbl_subtitle.setObjectName("AppSubtitle")
        title_layout.addWidget(lbl_title)
        title_layout.addWidget(lbl_subtitle)
        header_layout.addLayout(title_layout)

        header_layout.addStretch()

        # Tips / Guidance Toggle Button
        self.tips_enabled = self.settings.get("show_tips", True)
        self.btn_toggle_tips = QPushButton("操作提示: 开" if self.tips_enabled else "操作提示: 关")
        self.btn_toggle_tips.setCheckable(True)
        self.btn_toggle_tips.setChecked(self.tips_enabled)
        self.btn_toggle_tips.setToolTip("开启或关闭界面中的操作指引与说明")
        self.btn_toggle_tips.toggled.connect(self._on_tips_toggled)
        header_layout.addWidget(self.btn_toggle_tips)

        # Auto Release VRAM Toggle Button
        self.auto_release_vram = self.settings.get("auto_release_vram", True)
        self.btn_toggle_vram = QPushButton("自动释放显存: 开" if self.auto_release_vram else "自动释放显存: 关")
        self.btn_toggle_vram.setCheckable(True)
        self.btn_toggle_vram.setChecked(self.auto_release_vram)
        self.btn_toggle_vram.setToolTip("开启后，每次运算完成时会自动释放显存")
        self.btn_toggle_vram.toggled.connect(self._on_auto_release_vram_toggled)
        header_layout.addWidget(self.btn_toggle_vram)

        # Hardware Badge Button
        if self.hw_info.has_nvidia_gpu:
            gpu_short = self.hw_info.gpu_name.replace("NVIDIA GeForce ", "")
            if self.hw_info.torch_cuda_available:
                btn_hw_text = f"独显加速: {gpu_short} (CUDA)"
                self.btn_hardware = QPushButton(btn_hw_text)
                self.btn_hardware.setProperty("class", "PrimaryButton")
            else:
                btn_hw_text = f"独显: {gpu_short} (CPU运行中)"
                self.btn_hardware = QPushButton(btn_hw_text)
        else:
            btn_hw_text = "CPU 运算"
            self.btn_hardware = QPushButton(btn_hw_text)

        self.btn_hardware.setToolTip("查看显卡状态与运行环境")
        self.btn_hardware.clicked.connect(self._show_hardware_dialog)
        header_layout.addWidget(self.btn_hardware)

        # Background Console Log Button
        self.btn_logs = QPushButton("运行日志")
        self.btn_logs.setToolTip("查看运行日志与输出详情")
        self.btn_logs.clicked.connect(self.open_log_viewer)
        header_layout.addWidget(self.btn_logs)

        # Theme Selector
        self.combo_theme = QComboBox()
        self.combo_theme.addItem("跟随系统主题", THEME_MODE_AUTO)
        self.combo_theme.addItem("深色模式", THEME_MODE_DARK)
        self.combo_theme.addItem("浅色模式", THEME_MODE_LIGHT)

        saved_theme = self.settings.get("theme", THEME_MODE_AUTO)
        idx_theme = self.combo_theme.findData(saved_theme)
        if idx_theme >= 0:
            self.combo_theme.setCurrentIndex(idx_theme)
        self.combo_theme.currentIndexChanged.connect(self._on_theme_changed)
        header_layout.addWidget(self.combo_theme)

        main_layout.addWidget(header_frame)

        # ---------------- Navigation Tabs ----------------
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # View 1: Single & Pair Assessment
        self.view_single = SingleEvalView(self.evaluator_engine, self)
        self.tabs.addTab(self.view_single, "单图与多图评估")

        # View 2: Batch Evaluation
        self.view_batch = BatchEvalView(self.evaluator_engine, self)
        self.tabs.addTab(self.view_batch, "文件夹批量评测")

        # View 3: gMAD Analysis
        self.view_gmad = GMADView(self.evaluator_engine, self)
        self.tabs.addTab(self.view_gmad, "gMAD 差异对生成")

        # View 4: Dataset Benchmark
        self.view_bench = BenchmarkView(self.evaluator_engine, self)
        self.tabs.addTab(self.view_bench, "基准性能评测")

        # View 5: Model Zoo
        self.view_zoo = ModelZooView(self.evaluator_engine, self)
        self.tabs.addTab(self.view_zoo, "算法库管理")

        main_layout.addWidget(self.tabs, 1)

        # ---------------- Status Bar ----------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        dev_label = "CUDA 模式" if self.effective_device == "cuda" else "CPU 模式"
        self.lbl_status_main = QLabel(f"系统就绪 | 运算核心: {dev_label} | 算法库规模: 106+ 种模型")
        self.status_bar.addWidget(self.lbl_status_main, 1)

        # VRAM live monitor in status bar
        self.lbl_vram = QLabel("显存: -")
        self.lbl_vram.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.lbl_vram.setToolTip("当前独立显卡显存占用情况 (PyTorch 已分配容量 / 显存总容量)")
        self.status_bar.addPermanentWidget(self.lbl_vram)

        # Force Purge VRAM Button
        self.btn_purge_vram = QPushButton("释放显存")
        self.btn_purge_vram.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        self.btn_purge_vram.setToolTip("立即卸载所有已加载模型权重并彻底清空 GPU 显存缓存")
        self.btn_purge_vram.clicked.connect(self._on_purge_vram_clicked)
        self.status_bar.addPermanentWidget(self.btn_purge_vram)

        self.btn_status_logs = QPushButton("后台日志")
        self.btn_status_logs.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        self.btn_status_logs.clicked.connect(self.open_log_viewer)
        self.status_bar.addPermanentWidget(self.btn_status_logs)

        # Apply initial tips visibility
        self.apply_tips_visibility(self.tips_enabled)

        # Periodic timer for VRAM updates
        from PySide6.QtCore import QTimer
        self.vram_timer = QTimer(self)
        self.vram_timer.setInterval(2500)
        self.vram_timer.timeout.connect(self.update_vram_display)
        self.vram_timer.start()
        self.update_vram_display()

    def _on_tips_toggled(self, checked: bool) -> None:
        self.tips_enabled = checked
        self.btn_toggle_tips.setText("操作提示: 开" if checked else "操作提示: 关")
        self.settings["show_tips"] = checked
        save_settings(self.settings)
        self.apply_tips_visibility(checked)

    def apply_tips_visibility(self, visible: bool) -> None:
        if hasattr(self, "view_single") and self.view_single:
            self.view_single.set_tips_visible(visible)
        if hasattr(self, "view_batch") and hasattr(self.view_batch, "metric_selector"):
            self.view_batch.metric_selector.set_tips_visible(visible)

    def open_log_viewer(self) -> None:
        """Open or focus the live background log console dialog."""
        if not hasattr(self, "_log_dialog") or self._log_dialog is None:
            from iqa_gui.ui.components.log_dialog import LogViewerDialog
            self._log_dialog = LogViewerDialog(self)
        self._log_dialog.show()
        self._log_dialog.raise_()
        self._log_dialog.activateWindow()

    def _on_theme_changed(self) -> None:
        mode = self.combo_theme.currentData()
        self.settings["theme"] = mode
        save_settings(self.settings)
        apply_theme_to_app(QApplication.instance(), mode)

    def _show_hardware_dialog(self) -> None:
        """Show diagnostic dialog for NVIDIA GPU and CUDA support."""
        dlg = QDialog(self)
        dlg.setWindowTitle("独立显卡与计算硬件诊断报告")
        dlg.setFixedSize(540, 360)
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(16, 16, 16, 16)
        dlg_layout.setSpacing(10)

        lbl_head = QLabel("<b>硬件与 PyTorch 环境诊断信息：</b>")
        dlg_layout.addWidget(lbl_head)

        txt_info = (
            f"• <b>检测到的物理显卡:</b> {self.hw_info.gpu_name or '无'}<br>"
            f"• <b>显卡显存容量:</b> {self.hw_info.vram_mb or '-'} MB<br>"
            f"• <b>NVIDIA 驱动版本:</b> {self.hw_info.driver_version or '-'}<br>"
            f"• <b>当前 Python 环境:</b> Python {sys.version.split()[0]}<br>"
            f"• <b>PyTorch 版本:</b> {self.hw_info.torch_version}<br>"
            f"• <b>PyTorch CUDA 支持:</b> {'是 (已启用显卡加速)' if self.hw_info.torch_cuda_available else '否 (当前为 CPU 构建版本)'}<br>"
            f"• <b>当前生效运算设备:</b> {self.effective_device.upper()}<br><br>"
            f"<b>【诊断与优化建议】</b><br>"
            f"<div style='background: #334155; color: #F1F5F9; padding: 10px; border-radius: 4px;'>"
            f"{self.hw_info.recommendation_zh.replace(chr(10), '<br>')}"
            f"</div>"
        )

        lbl_body = QLabel(txt_info)
        lbl_body.setWordWrap(True)
        lbl_body.setTextFormat(Qt.TextFormat.RichText)
        dlg_layout.addWidget(lbl_body, 1)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dlg.accept)
        dlg_layout.addWidget(btn_close, 0, Qt.AlignmentFlag.AlignRight)

        dlg.exec()

    def update_vram_display(self) -> None:
        """Update VRAM usage indicator in the status bar."""
        info = self.evaluator_engine.get_vram_usage()
        if info["available"]:
            alloc = info["allocated_mb"]
            total = info["total_mb"]
            if alloc > 6000:
                color = "#EF4444"
            elif alloc > 2000:
                color = "#F59E0B"
            else:
                color = "#10B981"
            self.lbl_vram.setText(f"显存: <span style='color:{color}; font-weight:bold;'>{alloc:.0f} MB</span> / {total:.0f} MB")
            self.lbl_vram.setTextFormat(Qt.TextFormat.RichText)
        else:
            self.lbl_vram.setText("CPU 模式")

    def _on_purge_vram_clicked(self) -> None:
        """Manually purge all loaded models and free VRAM."""
        self.evaluator_engine.release_vram(force_unload=True)
        self.update_vram_display()
        self.status_bar.showMessage("显存已强制释放，所有已加载模型均已卸载。", 4000)

    def _on_auto_release_vram_toggled(self, checked: bool) -> None:
        self.auto_release_vram = checked
        self.btn_toggle_vram.setText("自动释放显存: 开" if checked else "自动释放显存: 关")
        self.settings["auto_release_vram"] = checked
        save_settings(self.settings)
        if checked:
            self.evaluator_engine.release_vram(force_unload=True)
            self.update_vram_display()

