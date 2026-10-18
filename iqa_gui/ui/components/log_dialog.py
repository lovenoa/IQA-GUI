"""High-density background log viewer and real-time task console dialog."""

from __future__ import annotations

import os
import subprocess
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from iqa_gui.core.log_manager import LOG_FILE_PATH, LogManager


class LogViewerDialog(QDialog):
    """Floating dialog monitoring real-time background computations, downloads, and errors."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("后台运行日志")
        self.resize(880, 560)
        self.log_mgr = LogManager.get_instance()

        self._init_ui()
        self._load_existing_logs()

        # Connect live log stream
        self.log_mgr.emitter.log_received.connect(self._on_log_received)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Header description
        header_layout = QHBoxLayout()
        lbl_title = QLabel("<b>后台运行日志</b>")
        lbl_subtitle = QLabel("实时监测算法推断、权重下载、标准输出与报错详情")
        lbl_subtitle.setProperty("class", "MutedText")
        header_layout.addWidget(lbl_title)
        header_layout.addWidget(lbl_subtitle)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Controls row
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(6)

        self.edit_filter = QLineEdit()
        self.edit_filter.setPlaceholderText("过滤关键字 (如 error, download, cuda, clip)...")
        self.edit_filter.textChanged.connect(self._apply_filter)
        self.edit_filter.setClearButtonEnabled(True)

        self.chk_autoscroll = QCheckBox("自动滚屏")
        self.chk_autoscroll.setChecked(True)

        self.btn_copy = QPushButton("复制全部")
        self.btn_copy.clicked.connect(self._copy_all)

        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self._clear_logs)

        self.btn_open_file = QPushButton("打开日志文件")
        self.btn_open_file.clicked.connect(self._open_log_file)

        ctrl_layout.addWidget(self.edit_filter, 3)
        ctrl_layout.addWidget(self.chk_autoscroll)
        ctrl_layout.addWidget(self.btn_copy)
        ctrl_layout.addWidget(self.btn_clear)
        ctrl_layout.addWidget(self.btn_open_file)
        layout.addLayout(ctrl_layout)

        # Console text area
        self.txt_console = QPlainTextEdit(self)
        self.txt_console.setReadOnly(True)
        self.txt_console.setMaximumBlockCount(4000)

        # Monospace terminal font
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.txt_console.setFont(font)
        layout.addWidget(self.txt_console, 1)

        # Footer info
        footer_layout = QHBoxLayout()
        self.lbl_status = QLabel(f"日志文件路径: {LOG_FILE_PATH}")
        self.lbl_status.setProperty("class", "MutedText")
        footer_layout.addWidget(self.lbl_status)
        footer_layout.addStretch()

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        footer_layout.addWidget(btn_close)
        layout.addLayout(footer_layout)

    def _load_existing_logs(self) -> None:
        """Load currently buffered logs."""
        all_logs = self.log_mgr.get_all_logs()
        self.txt_console.setPlainText(all_logs)
        if self.chk_autoscroll.isChecked():
            self.txt_console.moveCursor(QTextCursor.MoveOperation.End)

    def _on_log_received(self, text: str) -> None:
        """Append real-time incoming log line."""
        filter_text = self.edit_filter.text().strip().lower()
        if filter_text and filter_text not in text.lower():
            return

        cursor = self.txt_console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)

        if self.chk_autoscroll.isChecked():
            self.txt_console.moveCursor(QTextCursor.MoveOperation.End)

    def _apply_filter(self, filter_text: str) -> None:
        """Filter viewable logs based on user search term."""
        filter_text = filter_text.strip().lower()
        all_logs = self.log_mgr.get_all_logs()
        if not filter_text:
            self.txt_console.setPlainText(all_logs)
        else:
            lines = all_logs.splitlines(keepends=True)
            matched = [line for line in lines if filter_text in line.lower()]
            self.txt_console.setPlainText("".join(matched))

        if self.chk_autoscroll.isChecked():
            self.txt_console.moveCursor(QTextCursor.MoveOperation.End)

    def _copy_all(self) -> None:
        """Copy current console text to clipboard."""
        self.txt_console.selectAll()
        self.txt_console.copy()
        cursor = self.txt_console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.txt_console.setTextCursor(cursor)
        QMessageBox.information(self, "复制成功", "后台运行日志已复制到剪贴板。")

    def _clear_logs(self) -> None:
        """Clear memory and file logs."""
        self.log_mgr.clear_logs()
        self.txt_console.clear()

    def _open_log_file(self) -> None:
        """Open log file in default system editor or folder."""
        if os.path.exists(LOG_FILE_PATH):
            try:
                os.startfile(LOG_FILE_PATH)
            except Exception as e:
                QMessageBox.warning(self, "打开失败", f"无法打开日志文件: {e}")
        else:
            QMessageBox.information(self, "文件尚未生成", "尚未产生日志文件。")
