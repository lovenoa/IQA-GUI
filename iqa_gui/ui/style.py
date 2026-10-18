"""Modern styling and QSS theme definitions for IQA-GUI."""

MAIN_STYLE_SHEET = """
QMainWindow, QDialog {
    background-color: #F8FAFC;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
    color: #0F172A;
}

QWidget {
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
    color: #0F172A;
}

/* Header & Navigation */
#HeaderBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    padding: 10px 20px;
}

#AppTitle {
    font-size: 18px;
    font-weight: 700;
    color: #1E293B;
}

#AppSubtitle {
    font-size: 12px;
    color: #64748B;
}

/* Tab Widget & Navigation */
QTabWidget::pane {
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
    border-radius: 8px;
    margin-top: -1px;
}

QTabBar::tab {
    background: #F1F5F9;
    color: #475569;
    padding: 10px 22px;
    font-weight: 600;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 4px;
    border: 1px solid transparent;
}

QTabBar::tab:selected {
    background: #FFFFFF;
    color: #2563EB;
    border: 1px solid #E2E8F0;
    border-bottom: 2px solid #2563EB;
}

QTabBar::tab:hover:!selected {
    background: #E2E8F0;
    color: #1E293B;
}

/* Cards & Panels */
QFrame.CardFrame {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px;
}

QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 14px;
    font-weight: 600;
    color: #1E293B;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    background-color: #FFFFFF;
}

/* Buttons */
QPushButton {
    background-color: #F1F5F9;
    color: #1E293B;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: 500;
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

QPushButton.PrimaryButton:pressed {
    background-color: #1E40AF;
}

QPushButton.DangerButton {
    background-color: #EF4444;
    color: #FFFFFF;
    border: 1px solid #DC2626;
    font-weight: 600;
}

QPushButton.DangerButton:hover {
    background-color: #DC2626;
}

QPushButton.SuccessButton {
    background-color: #10B981;
    color: #FFFFFF;
    border: 1px solid #059669;
    font-weight: 600;
}

QPushButton.SuccessButton:hover {
    background-color: #059669;
}

/* Inputs & Form Controls */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 6px 10px;
    color: #0F172A;
    min-height: 22px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1.5px solid #2563EB;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

/* Tables */
QTableWidget, QTableView {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    gridline-color: #F1F5F9;
    selection-background-color: #EFF6FF;
    selection-color: #1E40AF;
}

QHeaderView::section {
    background-color: #F8FAFC;
    color: #475569;
    padding: 8px 10px;
    font-weight: 600;
    border: none;
    border-bottom: 1px solid #E2E8F0;
    border-right: 1px solid #F1F5F9;
}

/* Progress Bar */
QProgressBar {
    background-color: #E2E8F0;
    border-radius: 6px;
    text-align: center;
    font-weight: 600;
    color: #0F172A;
    height: 18px;
}

QProgressBar::chunk {
    background-color: #2563EB;
    border-radius: 6px;
}

/* Status Badges & Labels */
QLabel.BadgeSuccess {
    background-color: #DCFCE7;
    color: #166534;
    border-radius: 4px;
    padding: 2px 8px;
    font-weight: 600;
    font-size: 11px;
}

QLabel.BadgeWarning {
    background-color: #FEF3C7;
    color: #92400E;
    border-radius: 4px;
    padding: 2px 8px;
    font-weight: 600;
    font-size: 11px;
}

QLabel.BadgeInfo {
    background-color: #DBEAFE;
    color: #1E40AF;
    border-radius: 4px;
    padding: 2px 8px;
    font-weight: 600;
    font-size: 11px;
}

/* Scroll Areas & Viewports */
QScrollArea {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
}

QScrollArea > QWidget > QWidget {
    background-color: #FFFFFF;
}

QScrollArea QWidget {
    background-color: #FFFFFF;
}

/* CheckBox & RadioButton */
QCheckBox, QRadioButton {
    background-color: transparent;
    color: #1E293B;
    font-size: 13px;
    spacing: 6px;
}

QCheckBox:hover, QRadioButton:hover {
    color: #2563EB;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #94A3B8;
    background-color: #FFFFFF;
}

QCheckBox::indicator {
    border-radius: 4px;
}

QRadioButton::indicator {
    border-radius: 8px;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #2563EB;
    border-color: #2563EB;
}

/* ScrollBars */
QScrollBar:vertical {
    background: #F1F5F9;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #CBD5E1;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #94A3B8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""
