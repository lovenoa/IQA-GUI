"""Interactive image viewer supporting drag-and-drop, zoom, pan, and view synchronization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif")


def _has_image_url(mime_data) -> bool:
    if not mime_data.hasUrls():
        return False
    urls = mime_data.urls()
    if not urls:
        return False
    local_path = urls[0].toLocalFile()
    return local_path.lower().endswith(SUPPORTED_IMAGE_EXTS)


class SyncedGraphicsView(QGraphicsView):
    """Graphics view that emits pan/zoom signals for synchronized multi-view comparison."""
    view_transformed = Signal(float, float, float)  # center_x, center_y, scale_factor

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAcceptDrops(True)
        if self.viewport():
            self.viewport().setAcceptDrops(True)
        self.current_zoom = 1.0
        self._is_syncing = False

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if _has_image_url(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if _has_image_url(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if _has_image_url(event.mimeData()):
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                parent_card = self.parent()
                if parent_card and hasattr(parent_card, "load_image"):
                    parent_card.load_image(file_path)
            event.acceptProposedAction()
        else:
            event.ignore()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel zooming."""
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        new_zoom = self.current_zoom * zoom_factor
        if 0.1 <= new_zoom <= 20.0:
            self.scale(zoom_factor, zoom_factor)
            self.current_zoom = new_zoom
            self._notify_transform()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._notify_transform()

    def _notify_transform(self) -> None:
        if self._is_syncing:
            return
        center = self.mapToScene(self.viewport().rect().center())
        self.view_transformed.emit(center.x(), center.y(), self.current_zoom)

    def apply_sync(self, center_x: float, center_y: float, zoom: float) -> None:
        """Apply zoom and center coordinates from another synced view."""
        self._is_syncing = True
        try:
            factor = zoom / self.current_zoom if self.current_zoom > 0 else 1.0
            self.scale(factor, factor)
            self.current_zoom = zoom
            self.centerOn(center_x, center_y)
        finally:
            self._is_syncing = False


class ImageViewerCard(QFrame):
    """Comprehensive image card with drag & drop, file selection, zoom controls, and metadata."""
    image_changed = Signal(str)  # Emits file path
    sync_requested = Signal(float, float, float)

    def __init__(self, title: str = "图像视图", placeholder_text: str = "拖入图片或点击选择", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("class", "CardFrame")
        self.title_text = title
        self.current_file_path: Optional[str] = None

        self.setAcceptDrops(True)
        self._init_ui(placeholder_text)

    def _init_ui(self, placeholder_text: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Header bar
        header_layout = QHBoxLayout()
        self.lbl_title = QLabel(f"<b>{self.title_text}</b>")
        self.lbl_info = QLabel("")
        self.lbl_info.setProperty("class", "MutedText")

        self.btn_browse = QPushButton("选择文件...")
        self.btn_browse.clicked.connect(self._on_browse_clicked)

        self.btn_fit = QPushButton("适应窗口")
        self.btn_fit.clicked.connect(self.fit_to_view)

        self.btn_reset = QPushButton("1:1 原图")
        self.btn_reset.clicked.connect(self.reset_zoom)

        header_layout.addWidget(self.lbl_title)
        header_layout.addWidget(self.lbl_info)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_fit)
        header_layout.addWidget(self.btn_reset)
        header_layout.addWidget(self.btn_browse)
        layout.addLayout(header_layout)

        # Scene and View
        self.scene = QGraphicsScene(self)
        self.view = SyncedGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.view_transformed.connect(self.sync_requested.emit)
        layout.addWidget(self.view, 1)

        # Placeholder item
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.placeholder_label = QLabel(placeholder_text, self.view)
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setProperty("class", "MutedText")
        self.placeholder_label.setStyleSheet("font-size: 13px; background: transparent;")
        self.placeholder_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.placeholder_label.setGeometry(self.view.rect())

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if _has_image_url(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if _has_image_url(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if _has_image_url(event.mimeData()):
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if os.path.isfile(file_path):
                    self.load_image(file_path)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _on_browse_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择{self.title_text}",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;所有文件 (*.*)",
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path: str) -> bool:
        """Load and display an image from file path."""
        norm_path = os.path.normpath(file_path)
        if not os.path.exists(norm_path):
            return False

        pixmap = QPixmap(norm_path)
        if pixmap.isNull():
            return False

        self.current_file_path = norm_path
        self.display_pixmap(pixmap)

        # Update info label
        filename = os.path.basename(norm_path)
        filesize = os.path.getsize(norm_path) / 1024
        self.lbl_info.setText(f"{filename} ({pixmap.width()}×{pixmap.height()}, {filesize:.1f} KB)")
        self.image_changed.emit(norm_path)
        return True

    def display_pixmap(self, pixmap: QPixmap) -> None:
        """Display an existing QPixmap in the scene."""
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.placeholder_label.hide()
        self.fit_to_view()

    def fit_to_view(self) -> None:
        """Fit image inside viewport."""
        if self.pixmap_item:
            self.view.resetTransform()
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            # Update zoom factor
            transform = self.view.transform()
            self.view.current_zoom = transform.m11()
            self.view._notify_transform()

    def reset_zoom(self) -> None:
        """Reset to 1:1 pixel scale."""
        if self.pixmap_item:
            self.view.resetTransform()
            self.view.current_zoom = 1.0
            self.view.centerOn(self.pixmap_item)
            self.view._notify_transform()

    def sync_from(self, center_x: float, center_y: float, zoom: float) -> None:
        """Synchronize this viewer with coordinates from peer viewer."""
        self.view.apply_sync(center_x, center_y, zoom)
