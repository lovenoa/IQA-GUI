"""Headless integration tests for PySide6 GUI views."""

import os
import sys
import pytest

# Configure offscreen Qt platform before importing Qt widgets
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

# Ensure single QApplication instance across test session
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from iqa_gui.ui.main_window import MainWindow
from iqa_gui.core.evaluator import EvaluatorEngine

DIST_DIR = os.path.abspath("IQA-PyTorch/ResultsCalibra/dist_dir")
REF_DIR = os.path.abspath("IQA-PyTorch/ResultsCalibra/ref_dir")
SAMPLE_DIST = os.path.join(DIST_DIR, "I03.bmp")
SAMPLE_REF = os.path.join(REF_DIR, "I03.bmp")


def test_main_window_headless():
    window = MainWindow()
    assert window is not None
    assert window.tabs.count() == 5
    assert "单图" in window.tabs.tabText(0)
    assert "批量" in window.tabs.tabText(1)
    assert "gMAD" in window.tabs.tabText(2)
    assert "基准" in window.tabs.tabText(3)
    assert "算法" in window.tabs.tabText(4)


def test_single_view_headless():
    window = MainWindow()
    view = window.view_single

    # Test loading image into viewer
    loaded = view.viewer_target.load_image(SAMPLE_DIST)
    assert loaded is True
    assert view.viewer_target.current_file_path == SAMPLE_DIST

    loaded_ref = view.viewer_ref.load_image(SAMPLE_REF)
    assert loaded_ref is True

    # Check metric selection and chips
    view.metric_selector.select_anime_presets()
    selected = view.metric_selector.get_selected()
    assert len(selected) >= 5
    assert "laion_aes" in selected

    # Test removing one chip
    view.metric_selector._remove_selected("laion_aes")
    assert "laion_aes" not in view.metric_selector.get_selected()

    # Test scorecard table rendering with mock results
    from iqa_gui.core.evaluator import SingleEvalResult
    mock_res1 = SingleEvalResult(
        metric_name="laion_aes",
        metric_mode="NR",
        score=7.45,
        lower_better=False,
        score_range="[1, 10]",
        time_seconds=0.12,
        target_path=SAMPLE_DIST,
    )
    mock_res2 = SingleEvalResult(
        metric_name="clipiqa",
        metric_mode="NR",
        score=0.76,
        lower_better=False,
        score_range="[0, 1]",
        time_seconds=0.15,
        target_path=SAMPLE_DIST,
    )
    view._on_eval_finished([mock_res1, mock_res2])
    # Verify reference viewer is hidden when all selected are NR
    view.metric_selector.set_selected(["musiq", "clipiqa"])
    assert view.viewer_ref.isHidden() is True

    # Verify reference viewer is visible when an FR metric is selected
    view.metric_selector.set_selected(["psnr"])
    assert view.viewer_ref.isHidden() is False


def test_log_viewer_headless():
    window = MainWindow()
    window.open_log_viewer()
    assert window._log_dialog is not None
    window._log_dialog.close()


def test_batch_view_headless():
    window = MainWindow()
    view = window.view_batch

    view.edit_target_dir.setText(DIST_DIR)
    view.edit_ref_dir.setText(REF_DIR)
    view.metric_selector.set_selected(["psnr"])

    items = view.batch_engine.prepare_batch(DIST_DIR, REF_DIR, pair_mode="exact_name")
    assert len(items) > 0
    view.results_table.set_batch(items, ["psnr"])
    assert view.results_table.table.rowCount() == len(items)


def test_gmad_view_headless():
    window = MainWindow()
    view = window.view_gmad

    view.edit_dir.setText(DIST_DIR)
    assert view.combo_a.count() > 0
    assert view.combo_b.count() > 0
    assert view.spin_tol.value() == 0.10


def test_benchmark_view_headless():
    window = MainWindow()
    view = window.view_bench

    assert view.combo_dataset.count() >= 5
    view.combo_dataset.setCurrentIndex(1)  # koniq10k
    assert len(view.edit_root.text()) > 0


def test_model_zoo_view_headless():
    window = MainWindow()
    view = window.view_zoo

    # Check table populated
    assert view.table.rowCount() >= 95

    # Check search filter
    view.edit_search.setText("SSIM")
    view.refresh_catalog()
    assert view.table.rowCount() > 0
    assert view.table.rowCount() < 95

    # Check dependency status column and detail inspection
    view.table.selectRow(0)
    assert "运行依赖环境" in view.txt_desc.toHtml()
    assert view.btn_install_dep is not None


def test_drag_and_drop_on_viewer():
    from PySide6.QtCore import QMimeData, QUrl, QPointF, Qt
    from PySide6.QtGui import QDropEvent

    window = MainWindow()
    viewer = window.view_single.viewer_target

    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(SAMPLE_DIST)])
    drop_event = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    # Test drop directly on SyncedGraphicsView
    viewer.view.dropEvent(drop_event)
    assert os.path.normpath(viewer.current_file_path) == os.path.normpath(SAMPLE_DIST)


def test_preset_combinations_and_tips_toggle():
    window = MainWindow()
    selector = window.view_single.metric_selector

    # Verify built-in presets loaded in combo
    combo = selector.combo_presets
    assert combo.count() >= 3  # placeholder + 2 built-ins

    # Switch to 经典基础通用
    idx_classic = -1
    for i in range(combo.count()):
        if "经典基础通用" in combo.itemText(i):
            idx_classic = i
            break
    assert idx_classic != -1
    combo.setCurrentIndex(idx_classic)
    selected = selector.get_selected()
    assert "psnr" in selected
    assert "ssim" in selected

    # Test tips toggle in MainWindow
    window.btn_toggle_tips.setChecked(False)
    assert window.view_single.lbl_explanation.isHidden() is True
    assert selector.lbl_hint.isHidden() is True

    window.btn_toggle_tips.setChecked(True)
    assert window.view_single.lbl_explanation.isHidden() is False
    assert selector.lbl_hint.isHidden() is False


def test_vram_and_auto_release_gui():
    window = MainWindow()
    assert hasattr(window, "lbl_vram")
    assert hasattr(window, "btn_purge_vram")
    assert hasattr(window, "btn_toggle_vram")

    # Click purge vram
    window.btn_purge_vram.click()
    assert "显存已强制释放" in window.status_bar.currentMessage()

    # Toggle auto release
    window.btn_toggle_vram.setChecked(False)
    assert window.auto_release_vram is False
    window.btn_toggle_vram.setChecked(True)
    assert window.auto_release_vram is True



