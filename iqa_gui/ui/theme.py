"""Dual-theme system supporting Dark Mode, Light Mode, and System Sync with strict visual consistency."""

from __future__ import annotations

import winreg
from typing import Tuple
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QWidget

THEME_MODE_AUTO = "auto"
THEME_MODE_DARK = "dark"
THEME_MODE_LIGHT = "light"


def detect_system_is_dark() -> bool:
    """Detect whether Windows system is currently in Dark Mode."""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return val == 0
    except Exception:
        return True  # Default to dark on modern dev machines


# ==============================================================================
# 1. DARK THEME (专业深色工作台 - 严禁在 QWidget/QLabel 上画背景色块，全局统一色标)
# ==============================================================================
DARK_STYLE_SHEET = """
/* Base Windows */
QMainWindow, QDialog {
    background-color: #0F172A;
    font-size: 12px;
    color: #F1F5F9;
}

/* Global label transparency: Prevents ugly background patches behind text */
QLabel {
    background-color: transparent;
    color: #F1F5F9;
    font-size: 12px;
}

/* Header & Compact Toolbar */
#HeaderBar {
    background-color: #1E293B;
    border-bottom: 1px solid #334155;
    padding: 6px 14px;
}

#AppTitle {
    font-size: 15px;
    font-weight: 700;
    color: #F8FAFC;
    background: transparent;
}

#AppSubtitle {
    font-size: 11px;
    color: #94A3B8;
    background: transparent;
}

/* Navigation Tabs */
QTabWidget::pane {
    border: 1px solid #334155;
    background: #1E293B;
    border-radius: 4px;
    margin-top: -1px;
}

QTabBar::tab {
    background: #0F172A;
    color: #94A3B8;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 12px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    border: 1px solid #334155;
    border-bottom: none;
}

QTabBar::tab:selected {
    background: #1E293B;
    color: #38BDF8;
    border: 1px solid #334155;
    border-bottom: 2px solid #38BDF8;
}

QTabBar::tab:hover:!selected {
    background: #1E293B;
    color: #E2E8F0;
}

/* Compact Frames & Cards */
QFrame.CardFrame {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 6px;
}

/* Semantic Info Banners (Replaces all hardcoded white boxes) */
QFrame.InfoBanner, QLabel.InfoBanner {
    background-color: #1E293B;
    border: 1px solid #0284C7;
    border-left: 3px solid #38BDF8;
    border-radius: 4px;
    padding: 6px 8px;
    color: #E2E8F0;
    font-size: 11px;
}

QGroupBox {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: 600;
    font-size: 12px;
    color: #F8FAFC;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    background-color: #1E293B;
    color: #38BDF8;
}

/* Buttons */
QPushButton {
    background-color: #334155;
    color: #F8FAFC;
    border: 1px solid #475569;
    border-radius: 4px;
    padding: 4px 10px;
    font-weight: 500;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #475569;
    border-color: #64748B;
}

QPushButton:pressed {
    background-color: #1E293B;
}

QPushButton:disabled {
    background-color: #1E293B;
    color: #64748B;
    border-color: #334155;
}

QPushButton.PrimaryButton {
    background-color: #0284C7;
    color: #FFFFFF;
    border: 1px solid #38BDF8;
    font-weight: 600;
}

QPushButton.PrimaryButton:hover {
    background-color: #0369A1;
}

QPushButton.DangerButton {
    background-color: #DC2626;
    color: #FFFFFF;
    border: 1px solid #EF4444;
    font-weight: 600;
}

QPushButton.DangerButton:hover {
    background-color: #B91C1C;
}

QPushButton.TabCornerButton {
    background-color: #1E293B;
    color: #CBD5E1;
    border: 1px solid #475569;
    border-radius: 4px;
    padding: 1px 10px;
    font-size: 11px;
    font-weight: 500;
    min-height: 20px;
    max-height: 22px;
}

QPushButton.TabCornerButton:hover {
    background-color: #334155;
    border-color: #64748B;
    color: #F8FAFC;
}

QPushButton.TabCornerButton:pressed {
    background-color: #0F172A;
}

/* Inputs, TextEdits & Combos */
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #0F172A;
    border: 1px solid #475569;
    border-radius: 4px;
    padding: 4px 6px;
    color: #F8FAFC;
    min-height: 20px;
    font-size: 12px;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1.5px solid #38BDF8;
}

QComboBox QAbstractItemView {
    background-color: #1E293B;
    color: #F8FAFC;
    selection-background-color: #0284C7;
    border: 1px solid #475569;
}

/* Lists and Tables */
QListWidget, QTableWidget, QTableView {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 4px;
    gridline-color: #1E293B;
    color: #F8FAFC;
    selection-background-color: #0C4A6E;
    selection-color: #38BDF8;
    font-size: 12px;
}

QListWidget::item {
    padding: 3px 6px;
    border-bottom: 1px solid #1E293B;
}

QListWidget::item:hover {
    background-color: #1E293B;
}

QListWidget::item:selected {
    background-color: #0C4A6E;
    color: #38BDF8;
}

QHeaderView::section {
    background-color: #1E293B;
    color: #94A3B8;
    padding: 4px 6px;
    font-weight: 600;
    font-size: 11px;
    border: none;
    border-bottom: 1px solid #334155;
    border-right: 1px solid #1E293B;
}

/* Scroll Area */
QScrollArea {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 4px;
}

QScrollArea > QWidget > QWidget {
    background-color: #0F172A;
}

/* Graphics View / Image Canvas */
QGraphicsView {
    background-color: #0A0F1D;
    border: 1px solid #334155;
    border-radius: 4px;
}

/* CheckBox & RadioButton */
QCheckBox, QRadioButton {
    background-color: transparent;
    color: #F1F5F9;
    font-size: 12px;
    spacing: 5px;
}

QCheckBox:hover, QRadioButton:hover {
    color: #38BDF8;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #64748B;
    background-color: #0F172A;
}

QCheckBox::indicator {
    border-radius: 3px;
}

QRadioButton::indicator {
    border-radius: 7px;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #38BDF8;
    border-color: #38BDF8;
}

/* ScrollBars */
QScrollBar:vertical {
    background: #0F172A;
    width: 7px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #475569;
    min-height: 20px;
    border-radius: 3px;
}

QScrollBar::handle:vertical:hover {
    background: #64748B;
}

QScrollBar:horizontal {
    background: transparent;
    height: 3px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #475569;
    min-width: 20px;
    border-radius: 1.5px;
}

QScrollBar::handle:horizontal:hover {
    background: #64748B;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
    height: 0px;
}

/* Status Badges */
QLabel.BadgeInfo {
    background-color: #0C4A6E;
    color: #7DD3FC;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.BadgeSuccess {
    background-color: #064E3B;
    color: #6EE7B7;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.BadgeWarning {
    background-color: #78350F;
    color: #FCD34D;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 11px;
    font-weight: 600;
}

QProgressBar {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 3px;
    text-align: center;
    color: #F8FAFC;
    height: 14px;
    font-size: 11px;
}

QProgressBar::chunk {
    background-color: #38BDF8;
}

/* Common Frames & Utilities */
QFrame.StatsFrame {
    background-color: #1E293B;
    border: 1px dashed #475569;
    border-radius: 4px;
    padding: 6px;
}

QLabel.MutedText {
    color: #94A3B8;
    background: transparent;
    font-size: 11px;
}

QLabel.DetailTitle {
    font-size: 15px;
    font-weight: 700;
    color: #38BDF8;
    background: transparent;
}

QLabel.ScoreHero {
    font-size: 26px;
    font-weight: 700;
    color: #38BDF8;
    background: transparent;
}

QPushButton.MetricChip, QLabel.MetricChip {
    background-color: #0C4A6E;
    color: #7DD3FC;
    border: 1px solid #0284C7;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 11px;
    min-height: 20px;
    max-height: 22px;
}

QPushButton.MetricChip:hover {
    background-color: #0369A1;
    color: #FFFFFF;
}

QLabel.TierS {
    background-color: #064E3B;
    color: #6EE7B7;
    border: 1px solid #059669;
    border-radius: 3px;
    padding: 2px 6px;
    font-weight: 700;
    font-size: 11px;
}

QLabel.TierA {
    background-color: #0C4A6E;
    color: #7DD3FC;
    border: 1px solid #0284C7;
    border-radius: 3px;
    padding: 2px 6px;
    font-weight: 700;
    font-size: 11px;
}

QLabel.TierB {
    background-color: #78350F;
    color: #FCD34D;
    border: 1px solid #D97706;
    border-radius: 3px;
    padding: 2px 6px;
    font-weight: 700;
    font-size: 11px;
}

QLabel.TierC {
    background-color: #7F1D1D;
    color: #FCA5A5;
    border: 1px solid #DC2626;
    border-radius: 3px;
    padding: 2px 6px;
    font-weight: 700;
    font-size: 11px;
}
"""

# ==============================================================================
# 2. LIGHT THEME (专业浅色工作台 - 严禁在 QWidget/QLabel 上画背景色块，全局统一色标)
# ==============================================================================
LIGHT_STYLE_SHEET = """
/* Base Windows */
QMainWindow, QDialog {
    background-color: #F8FAFC;
    font-size: 12px;
    color: #0F172A;
}

/* Global label transparency */
QLabel {
    background-color: transparent;
    color: #0F172A;
    font-size: 12px;
}

/* Header & Compact Toolbar */
#HeaderBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    padding: 6px 14px;
}

#AppTitle {
    font-size: 15px;
    font-weight: 700;
    color: #0F172A;
    background: transparent;
}

#AppSubtitle {
    font-size: 11px;
    color: #64748B;
    background: transparent;
}

/* Navigation Tabs */
QTabWidget::pane {
    border: 1px solid #CBD5E1;
    background: #FFFFFF;
    border-radius: 4px;
    margin-top: -1px;
}

QTabBar::tab {
    background: #F1F5F9;
    color: #475569;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 12px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    border: 1px solid #CBD5E1;
    border-bottom: none;
}

QTabBar::tab:selected {
    background: #FFFFFF;
    color: #2563EB;
    border: 1px solid #CBD5E1;
    border-bottom: 2px solid #2563EB;
}

QTabBar::tab:hover:!selected {
    background: #E2E8F0;
    color: #0F172A;
}

/* Compact Frames & Cards */
QFrame.CardFrame {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    padding: 6px;
}

/* Semantic Info Banners */
QFrame.InfoBanner, QLabel.InfoBanner {
    background-color: #EFF6FF;
    border: 1px solid #93C5FD;
    border-left: 3px solid #2563EB;
    border-radius: 4px;
    padding: 6px 8px;
    color: #1E3A8A;
    font-size: 11px;
}

QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: 600;
    font-size: 12px;
    color: #0F172A;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    background-color: #FFFFFF;
    color: #2563EB;
}

/* Buttons */
QPushButton {
    background-color: #F1F5F9;
    color: #0F172A;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    padding: 4px 10px;
    font-weight: 500;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #E2E8F0;
    border-color: #94A3B8;
}

QPushButton:pressed {
    background-color: #CBD5E1;
}

QPushButton:disabled {
    background-color: #F8FAFC;
    color: #94A3B8;
    border-color: #E2E8F0;
}

QPushButton.PrimaryButton {
    background-color: #2563EB;
    color: #FFFFFF;
    border: 1px solid #1D4ED8;
    font-weight: 600;
}

QPushButton.PrimaryButton:hover {
    background-color: #1D4ED8;
}

QPushButton.DangerButton {
    background-color: #DC2626;
    color: #FFFFFF;
    border: 1px solid #B91C1C;
    font-weight: 600;
}

QPushButton.DangerButton:hover {
    background-color: #B91C1C;
}

QPushButton.TabCornerButton {
    background-color: #F8FAFC;
    color: #475569;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    padding: 1px 10px;
    font-size: 11px;
    font-weight: 500;
    min-height: 20px;
    max-height: 22px;
}

QPushButton.TabCornerButton:hover {
    background-color: #E2E8F0;
    border-color: #94A3B8;
    color: #0F172A;
}

QPushButton.TabCornerButton:pressed {
    background-color: #CBD5E1;
}

/* Inputs, TextEdits & Combos */
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    padding: 4px 6px;
    color: #0F172A;
    min-height: 20px;
    font-size: 12px;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1.5px solid #2563EB;
}

QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    color: #0F172A;
    selection-background-color: #DBEAFE;
    selection-color: #1E40AF;
    border: 1px solid #CBD5E1;
}

/* Lists and Tables */
QListWidget, QTableWidget, QTableView {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    gridline-color: #F1F5F9;
    color: #0F172A;
    selection-background-color: #DBEAFE;
    selection-color: #1E40AF;
    font-size: 12px;
}

QListWidget::item {
    padding: 3px 6px;
    border-bottom: 1px solid #F1F5F9;
}

QListWidget::item:hover {
    background-color: #F8FAFC;
}

QListWidget::item:selected {
    background-color: #DBEAFE;
    color: #1E40AF;
}

QHeaderView::section {
    background-color: #F8FAFC;
    color: #475569;
    padding: 4px 6px;
    font-weight: 600;
    font-size: 11px;
    border: none;
    border-bottom: 1px solid #CBD5E1;
    border-right: 1px solid #F1F5F9;
}

/* Scroll Area */
QScrollArea {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
}

QScrollArea > QWidget > QWidget {
    background-color: #FFFFFF;
}

/* Graphics View / Image Canvas */
QGraphicsView {
    background-color: #F1F5F9;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
}

/* CheckBox & RadioButton */
QCheckBox, QRadioButton {
    background-color: transparent;
    color: #0F172A;
    font-size: 12px;
    spacing: 5px;
}

QCheckBox:hover, QRadioButton:hover {
    color: #2563EB;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #94A3B8;
    background-color: #FFFFFF;
}

QCheckBox::indicator {
    border-radius: 3px;
}

QRadioButton::indicator {
    border-radius: 7px;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #2563EB;
    border-color: #2563EB;
}

/* ScrollBars */
QScrollBar:vertical {
    background: #F8FAFC;
    width: 7px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #CBD5E1;
    min-height: 20px;
    border-radius: 3px;
}

QScrollBar::handle:vertical:hover {
    background: #94A3B8;
}

QScrollBar:horizontal {
    background: transparent;
    height: 3px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #CBD5E1;
    min-width: 20px;
    border-radius: 1.5px;
}

QScrollBar::handle:horizontal:hover {
    background: #94A3B8;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
    height: 0px;
}

/* Status Badges */
QLabel.BadgeInfo {
    background-color: #DBEAFE;
    color: #1E40AF;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.BadgeSuccess {
    background-color: #DCFCE7;
    color: #166534;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.BadgeWarning {
    background-color: #FEF3C7;
    color: #92400E;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 11px;
    font-weight: 600;
}

QProgressBar {
    background-color: #E2E8F0;
    border: 1px solid #CBD5E1;
    border-radius: 3px;
    text-align: center;
    color: #0F172A;
    height: 14px;
    font-size: 11px;
}

QProgressBar::chunk {
    background-color: #2563EB;
}

/* Common Frames & Utilities */
QFrame.StatsFrame {
    background-color: #F8FAFC;
    border: 1px dashed #CBD5E1;
    border-radius: 4px;
    padding: 6px;
}

QLabel.MutedText {
    color: #64748B;
    background: transparent;
    font-size: 11px;
}

QLabel.DetailTitle {
    font-size: 15px;
    font-weight: 700;
    color: #2563EB;
    background: transparent;
}

QLabel.ScoreHero {
    font-size: 26px;
    font-weight: 700;
    color: #2563EB;
    background: transparent;
}

QPushButton.MetricChip, QLabel.MetricChip {
    background-color: #EFF6FF;
    color: #1E40AF;
    border: 1px solid #93C5FD;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 11px;
    min-height: 20px;
    max-height: 22px;
}

QPushButton.MetricChip:hover {
    background-color: #DBEAFE;
    color: #1D4ED8;
}

QLabel.TierS {
    background-color: #DCFCE7;
    color: #166534;
    border: 1px solid #86EFAC;
    border-radius: 3px;
    padding: 2px 6px;
    font-weight: 700;
    font-size: 11px;
}

QLabel.TierA {
    background-color: #EFF6FF;
    color: #1E40AF;
    border: 1px solid #93C5FD;
    border-radius: 3px;
    padding: 2px 6px;
    font-weight: 700;
    font-size: 11px;
}

QLabel.TierB {
    background-color: #FEF9C3;
    color: #854D0E;
    border: 1px solid #FDE047;
    border-radius: 3px;
    padding: 2px 6px;
    font-weight: 700;
    font-size: 11px;
}

QLabel.TierC {
    background-color: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FCA5A5;
    border-radius: 3px;
    padding: 2px 6px;
    font-weight: 700;
    font-size: 11px;
}
"""


def apply_theme_to_app(app: QApplication, mode: str = THEME_MODE_AUTO) -> str:
    """Apply the chosen theme and configure QPalette cleanly."""
    is_dark = detect_system_is_dark() if mode == THEME_MODE_AUTO else (mode == THEME_MODE_DARK)

    # Set default high-readability font
    app_font = QFont("Microsoft YaHei", 9)
    app.setFont(app_font)

    palette = QPalette()
    if is_dark:
        palette.setColor(QPalette.ColorRole.Window, QColor("#0F172A"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#F1F5F9"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#1E293B"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#0F172A"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1E293B"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#F8FAFC"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#F8FAFC"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#1E293B"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#F8FAFC"))
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#EF4444"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#0284C7"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
        app.setPalette(palette)
        app.setStyleSheet(DARK_STYLE_SHEET)
        return THEME_MODE_DARK
    else:
        palette.setColor(QPalette.ColorRole.Window, QColor("#F8FAFC"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#0F172A"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#F1F5F9"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#0F172A"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#0F172A"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#0F172A"))
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#EF4444"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#2563EB"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
        app.setPalette(palette)
        app.setStyleSheet(LIGHT_STYLE_SHEET)
        return THEME_MODE_LIGHT
