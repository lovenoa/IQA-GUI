"""Interactive image viewer supporting drag-and-drop, zoom, pan, and view synchronization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QCursor, QDragEnterEvent, QDropEvent, QImage, QPainter, QPixmap, QWheelEvent
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
    return any(u.toLocalFile().lower().endswith(SUPPORTED_IMAGE_EXTS) for u in urls)


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
                parent_card = self.parent()
                if parent_card and hasattr(parent_card, "handle_dropped_urls"):
                    parent_card.handle_dropped_urls(urls)
                elif parent_card and hasattr(parent_card, "load_image"):
                    file_path = urls[0].toLocalFile()
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
    image_changed = Signal(str)  # Emits active file path
    images_updated = Signal(list)  # Emits list of all loaded file paths (List[str])
    active_index_changed = Signal(int)  # Emits active index in image list
    sync_requested = Signal(float, float, float)

    def __init__(
        self,
        title: str = "图像视图",
        placeholder_text: str = "拖入图片或点击选择",
        multi_image_mode: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setProperty("class", "CardFrame")
        self.title_text = title
        self.multi_image_mode = multi_image_mode
        self.current_file_path: Optional[str] = None
        self.image_paths: List[str] = []
        self.active_index: int = 0

        self.setAcceptDrops(True)
        self._init_ui(placeholder_text)

    def _init_ui(self, placeholder_text: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
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

        # Multi-image pills strip (for target comparison)
        self.pill_bar = QWidget()
        self.pill_bar_layout = QHBoxLayout(self.pill_bar)
        self.pill_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.pill_bar_layout.setSpacing(6)
        self.pills_container = QWidget()
        self.pills_layout = QHBoxLayout(self.pills_container)
        self.pills_layout.setContentsMargins(0, 0, 0, 0)
        self.pills_layout.setSpacing(6)
        self.pill_bar_layout.addWidget(self.pills_container)
        self.pill_bar_layout.addStretch()

        self.btn_clear_all = QPushButton("🗑️ 清空待测列表")
        self.btn_clear_all.setToolTip("清空当前所有待测图片")
        self.btn_clear_all.clicked.connect(self.clear_all_images)
        self.pill_bar_layout.addWidget(self.btn_clear_all)

        layout.addWidget(self.pill_bar)
        if not self.multi_image_mode:
            self.pill_bar.hide()
        else:
            self._update_pills_ui()

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
                self.handle_dropped_urls(urls)
            event.acceptProposedAction()
        else:
            event.ignore()

    def handle_dropped_urls(self, urls: list) -> None:
        """Handle dropped URLs (supports multiple images in multi-image mode)."""
        paths = [
            u.toLocalFile()
            for u in urls
            if u.toLocalFile().lower().endswith(SUPPORTED_IMAGE_EXTS) and os.path.isfile(u.toLocalFile())
        ]
        if not paths:
            return
        if self.multi_image_mode:
            self.add_images(paths)
        else:
            self.load_image(paths[0])

    def _on_browse_clicked(self) -> None:
        if self.multi_image_mode:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                f"选择待评测图像 (最多3张)",
                "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;所有文件 (*.*)",
            )
            if file_paths:
                self.add_images(file_paths)
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                f"选择{self.title_text}",
                "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;所有文件 (*.*)",
            )
            if file_path:
                self.load_image(file_path)

    def get_image_paths(self) -> List[str]:
        """Return list of loaded image paths."""
        if self.multi_image_mode:
            return list(self.image_paths)
        return [self.current_file_path] if self.current_file_path else []

    def add_images(self, paths: List[str]) -> None:
        """Add up to 3 images to the multi-image list."""
        added = False
        for p in paths:
            norm = os.path.normpath(p)
            if os.path.exists(norm) and norm not in self.image_paths:
                if len(self.image_paths) < 3:
                    self.image_paths.append(norm)
                    added = True
        if self.image_paths:
            if added:
                self.active_index = len(self.image_paths) - 1
            else:
                self.active_index = min(self.active_index, len(self.image_paths) - 1)
            self._update_pills_ui()
            self.select_image_at(self.active_index)
            self.images_updated.emit(list(self.image_paths))

    def load_image(self, file_path: str) -> bool:
        """Load and display an image from file path."""
        norm_path = os.path.normpath(file_path)
        if not os.path.exists(norm_path):
            return False

        if self.multi_image_mode:
            if norm_path not in self.image_paths:
                if len(self.image_paths) >= 3:
                    self.image_paths[self.active_index] = norm_path
                else:
                    self.image_paths.append(norm_path)
                    self.active_index = len(self.image_paths) - 1
            else:
                self.active_index = self.image_paths.index(norm_path)
            self._update_pills_ui()
            self.select_image_at(self.active_index)
            self.images_updated.emit(list(self.image_paths))
            return True

        pixmap = QPixmap(norm_path)
        if pixmap.isNull():
            return False

        self.current_file_path = norm_path
        self.display_pixmap(pixmap)

        filename = os.path.basename(norm_path)
        filesize = os.path.getsize(norm_path) / 1024
        self.lbl_info.setText(f"{filename} ({pixmap.width()}×{pixmap.height()}, {filesize:.1f} KB)")
        self.image_changed.emit(norm_path)
        return True

    def select_image_at(self, idx: int) -> None:
        """Select and display image at index in image_paths."""
        if not (0 <= idx < len(self.image_paths)):
            return
        self.active_index = idx
        target_path = self.image_paths[idx]
        self.current_file_path = target_path

        pixmap = QPixmap(target_path)
        if not pixmap.isNull():
            self.display_pixmap(pixmap)
            filename = os.path.basename(target_path)
            filesize = os.path.getsize(target_path) / 1024
            self.lbl_info.setText(f"[图 {idx+1}/{len(self.image_paths)}] {filename} ({pixmap.width()}×{pixmap.height()}, {filesize:.1f} KB)")
            self._update_pills_ui()
            self.image_changed.emit(target_path)
            self.active_index_changed.emit(idx)

    def remove_image_at(self, idx: int) -> None:
        """Remove image at specified index."""
        if 0 <= idx < len(self.image_paths):
            self.image_paths.pop(idx)
            if not self.image_paths:
                self.clear_all_images()
            else:
                self.active_index = min(self.active_index, len(self.image_paths) - 1)
                self.select_image_at(self.active_index)
                self.images_updated.emit(list(self.image_paths))

    def clear_all_images(self) -> None:
        """Clear all loaded images and reset canvas."""
        self.image_paths.clear()
        self.active_index = 0
        self.current_file_path = None
        self.scene.clear()
        self.pixmap_item = None
        self.lbl_info.setText("")
        self.placeholder_label.show()
        if self.multi_image_mode:
            self._update_pills_ui()
        self.image_changed.emit("")
        self.images_updated.emit([])
        self.active_index_changed.emit(-1)

    def _update_pills_ui(self) -> None:
        """Refresh multi-image pills strip."""
        if not self.multi_image_mode:
            return

        # Clear existing pill widgets
        while self.pills_layout.count():
            item = self.pills_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not self.image_paths:
            self.btn_clear_all.hide()
            hint = QLabel("💡 待测区支持拖入最多 3 张图像进行横向对比")
            hint.setProperty("class", "MutedText")
            hint.setStyleSheet("font-size: 11px;")
            self.pills_layout.addWidget(hint)
            return

        self.btn_clear_all.show()

        for idx, path in enumerate(self.image_paths):
            is_active = (idx == self.active_index)
            fname = os.path.basename(path)
            short_name = (fname[:14] + "..") if len(fname) > 16 else fname

            pill_frame = QFrame()
            pill_layout = QHBoxLayout(pill_frame)
            pill_layout.setContentsMargins(6, 2, 6, 2)
            pill_layout.setSpacing(4)

            btn_sel = QPushButton(f"图 {idx+1}: {short_name}")
            btn_sel.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn_sel.setToolTip(f"点击切换显示该图并在左侧看板展示结果: {fname}\n完整路径: {path}")
            if is_active:
                btn_sel.setStyleSheet("font-weight: bold; color: #2563eb; background: transparent; border: none;")
                pill_frame.setStyleSheet(
                    "QFrame { background-color: rgba(37, 99, 235, 0.15); border: 1.5px solid #2563eb; border-radius: 6px; }"
                )
            else:
                btn_sel.setStyleSheet("color: inherit; background: transparent; border: none;")
                pill_frame.setStyleSheet(
                    "QFrame { background-color: rgba(128, 128, 128, 0.1); border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 6px; }"
                )

            # Bind click
            btn_sel.clicked.connect(lambda checked=False, i=idx: self.select_image_at(i))
            pill_layout.addWidget(btn_sel)

            # Close button
            btn_close = QPushButton("✕")
            btn_close.setFixedSize(16, 16)
            btn_close.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn_close.setToolTip(f"移除图 {idx+1}")
            btn_close.setStyleSheet(
                "QPushButton { border: none; font-size: 11px; color: #888; background: transparent; padding: 0px; } "
                "QPushButton:hover { color: #e11d48; font-weight: bold; }"
            )
            btn_close.clicked.connect(lambda checked=False, i=idx: self.remove_image_at(i))
            pill_layout.addWidget(btn_close)

            self.pills_layout.addWidget(pill_frame)

        # Add image button if < 3
        if len(self.image_paths) < 3:
            btn_add = QPushButton(f"➕ 添加第 {len(self.image_paths)+1} 张图")
            btn_add.setToolTip("选择或直接拖入新图片至待测框")
            btn_add.setStyleSheet("font-size: 11px; padding: 2px 8px;")
            btn_add.clicked.connect(self._on_browse_clicked)
            self.pills_layout.addWidget(btn_add)

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
