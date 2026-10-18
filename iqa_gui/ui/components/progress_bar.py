"""Progress bar component with speed, ETA, status text, and cancel button."""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class TaskProgressWidget(QFrame):
    """Real-time task progress bar with status, throughput, ETA, and cancellation."""
    cancel_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("class", "CardFrame")
        self._init_ui()
        self.reset()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # Upper row: message, speed/ETA, cancel button
        row_top = QHBoxLayout()
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("font-weight: 600;")

        self.lbl_eta = QLabel("")
        self.lbl_eta.setProperty("class", "MutedText")

        self.btn_cancel = QPushButton("⏹ 停止计算")
        self.btn_cancel.setProperty("class", "DangerButton")
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        self.btn_cancel.setEnabled(False)

        row_top.addWidget(self.lbl_status)
        row_top.addStretch()
        row_top.addWidget(self.lbl_eta)
        row_top.addWidget(self.btn_cancel)
        layout.addLayout(row_top)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

    def set_running(self, running: bool, status_text: str = "") -> None:
        """Toggle running status."""
        self.btn_cancel.setEnabled(running)
        if status_text:
            self.lbl_status.setText(status_text)
        if not running:
            self.lbl_eta.setText("")

    def update_progress(self, current: int, total: int, status_msg: str = "", speed: float = 0.0) -> None:
        """Update progress bar, status message, speed, and ETA."""
        if total <= 0:
            self.progress_bar.setValue(0)
            return

        pct = int((current / total) * 100)
        self.progress_bar.setValue(pct)

        if status_msg:
            self.lbl_status.setText(status_msg)
        else:
            self.lbl_status.setText(f"处理中: {current} / {total} ({pct}%)")

        # ETA calculation
        if speed > 0 and current < total:
            remaining_secs = (total - current) / speed
            mins, secs = divmod(int(remaining_secs), 60)
            eta_str = f"{mins}分{secs}秒" if mins > 0 else f"{secs}秒"
            self.lbl_eta.setText(f"速度: {speed:.1f} 图/秒 | 预估剩余: {eta_str}")
        elif current >= total:
            self.lbl_eta.setText("✅ 已完成")

    def reset(self) -> None:
        """Reset progress bar to initial state."""
        self.progress_bar.setValue(0)
        self.lbl_status.setText("就绪")
        self.lbl_eta.setText("")
        self.btn_cancel.setEnabled(False)
