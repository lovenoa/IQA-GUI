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
    assert "运行依赖" in view.txt_desc.toHtml()
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


def test_scorecard_table_sorting():
    from iqa_gui.core.evaluator import SingleEvalResult
    window = MainWindow()
    view = window.view_single

    r_laion = SingleEvalResult(
        metric_name="laion_aes",
        metric_mode="NR",
        score=6.2,
        lower_better=False,
        score_range="[1, 10]",
        time_seconds=0.1,
        target_path=SAMPLE_DIST,
    )
    r_psnr = SingleEvalResult(
        metric_name="psnr",
        metric_mode="FR",
        score=37.5,
        lower_better=False,
        score_range="[0, 50]",
        time_seconds=0.1,
        target_path=SAMPLE_DIST,
        ref_path=SAMPLE_REF,
    )
    r_brisque = SingleEvalResult(
        metric_name="brisque",
        metric_mode="NR",
        score=75.0,
        lower_better=True,
        score_range="[0, 100]",
        time_seconds=0.1,
        target_path=SAMPLE_DIST,
    )

    view._on_eval_finished([r_laion, r_psnr, r_brisque])
    assert view.table_scorecard.rowCount() == 3

    # Test Column 0 ("算法") tri-state sorting: Descending -> Ascending -> Default
    view._on_header_section_clicked(0)
    assert view.sort_state == (0, False)  # Descending
    names_desc = [view.latest_results[i].metric_name for i in range(3)]
    assert names_desc == ["psnr", "laion_aes", "brisque"]

    view._on_header_section_clicked(0)
    assert view.sort_state == (0, True)   # Ascending
    names_asc = [view.latest_results[i].metric_name for i in range(3)]
    assert names_asc == ["brisque", "laion_aes", "psnr"]

    view._on_header_section_clicked(0)
    assert view.sort_state is None        # Default order
    names_def = [view.latest_results[i].metric_name for i in range(3)]
    assert names_def == ["laion_aes", "psnr", "brisque"]

    # Test Column 4 ("评级") sorting: Descending should put S first
    view._on_header_section_clicked(4)
    assert view.latest_results[0].grade == "S"  # psnr (37.5 dB) has grade S


def test_image_viewer_multi_image_mode():
    from iqa_gui.ui.components.image_viewer import ImageViewerCard
    card = ImageViewerCard(title="多图待测", multi_image_mode=True)
    assert card.multi_image_mode is True

    # 1. Add images
    card.add_images([SAMPLE_DIST, SAMPLE_REF])
    assert len(card.get_image_paths()) == 2
    assert card.active_index == 1

    # 2. Add 3rd image and attempt exceeding max 3
    card.add_images([SAMPLE_DIST])  # duplicate path should be ignored
    assert len(card.get_image_paths()) == 2

    # 3. Select image at 0
    card.select_image_at(0)
    assert card.active_index == 0
    assert card.current_file_path == SAMPLE_DIST

    # 4. Remove image
    card.remove_image_at(0)
    assert len(card.get_image_paths()) == 1
    assert card.current_file_path == SAMPLE_REF

    # 5. Clear all images
    card.clear_all_images()
    assert len(card.get_image_paths()) == 0
    assert card.current_file_path is None


def test_multi_image_compare_dialog():
    import tempfile
    from iqa_gui.core.evaluator import SingleEvalResult
    from iqa_gui.ui.components.compare_dialog import MultiImageCompareDialog

    r1 = SingleEvalResult(
        metric_name="psnr",
        metric_mode="FR",
        score=32.0,
        lower_better=False,
        score_range="[0, 50]",
        time_seconds=0.1,
        target_path=SAMPLE_DIST,
    )
    r2 = SingleEvalResult(
        metric_name="psnr",
        metric_mode="FR",
        score=38.0,
        lower_better=False,
        score_range="[0, 50]",
        time_seconds=0.1,
        target_path=SAMPLE_REF,
    )
    eval_map = {
        SAMPLE_DIST: [r1],
        SAMPLE_REF: [r2],
    }

    dlg = MultiImageCompareDialog([SAMPLE_DIST, SAMPLE_REF], eval_map)
    assert dlg.table_compare.rowCount() == 1
    # Check winner column has trophy and points to Image 2 (higher PSNR)
    winner_item = dlg.table_compare.item(0, 5)
    assert "🏆 图 2 优胜" in winner_item.text()

    # Test clipboard copy method
    copy_text = dlg._copy_to_clipboard()
    assert "psnr" in copy_text.lower()
    assert "🏆 图 2 优胜" in copy_text

    # Test CSV export
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        csv_path = tf.name
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(dlg.col_headers)
            row_vals = [dlg.table_compare.item(0, c).text() for c in range(dlg.table_compare.columnCount())]
            writer.writerow(row_vals)
        assert os.path.getsize(csv_path) > 0
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


def test_multi_image_compare_dialog_three_images():
    from iqa_gui.core.evaluator import SingleEvalResult
    from iqa_gui.ui.components.compare_dialog import MultiImageCompareDialog

    # 3 images: img 1 has 32.0, img 2 has 38.0 (best), img 3 has 25.0 (worst)
    p3 = SAMPLE_DIST + "_3"
    r1 = SingleEvalResult("psnr", 32.0, 0.1, False, "[0, 50]", "FR", SAMPLE_DIST)
    r2 = SingleEvalResult("psnr", 38.0, 0.1, False, "[0, 50]", "FR", SAMPLE_REF)
    r3 = SingleEvalResult("psnr", 25.0, 0.1, False, "[0, 50]", "FR", p3)

    eval_map = {
        SAMPLE_DIST: [r1],
        SAMPLE_REF: [r2],
        p3: [r3],
    }

    dlg = MultiImageCompareDialog([SAMPLE_DIST, SAMPLE_REF, p3], eval_map)
    assert dlg.table_compare.rowCount() == 1
    # Check winner column has both winner (Image 2) and loser (Image 3)
    col_winner_idx = 3 + 3
    winner_item = dlg.table_compare.item(0, col_winner_idx)
    assert "🏆 图 2 优胜" in winner_item.text()
    assert "📉 图 3 垫底" in winner_item.text()


def test_multi_round_summary_dialog_single_and_multi():
    from iqa_gui.core.evaluator import SingleEvalResult
    from iqa_gui.ui.components.multi_round_dialog import MultiRoundSummaryDialog

    # Mode 1: Single image, 3 rounds
    r1 = SingleEvalResult("qalign_8bit", 4.10, 0.1, False, "[1, 5]", "NR", SAMPLE_DIST)
    r2 = SingleEvalResult("qalign_8bit", 4.15, 0.1, False, "[1, 5]", "NR", SAMPLE_DIST)
    r3 = SingleEvalResult("qalign_8bit", 4.05, 0.1, False, "[1, 5]", "NR", SAMPLE_DIST)
    rounds_map_single = {
        1: {SAMPLE_DIST: [r1]},
        2: {SAMPLE_DIST: [r2]},
        3: {SAMPLE_DIST: [r3]},
    }
    dlg_single = MultiRoundSummaryDialog([SAMPLE_DIST], rounds_map_single)
    assert dlg_single.table_single.rowCount() == 1
    # Check mean, range, std dev
    # mean = (4.10+4.15+4.05)/3 = 4.1000, range = 4.15-4.05 = 0.1000
    mean_item = dlg_single.table_single.item(0, 6)
    rng_item = dlg_single.table_single.item(0, 7)
    assert "4.1000" in mean_item.text()
    assert "0.1000" in rng_item.text()

    copy_text = dlg_single._copy_to_clipboard()
    assert "4.1000" in copy_text

    # Mode 2: Multi image (2 images, 2 rounds)
    r1_b = SingleEvalResult("qalign_8bit", 3.80, 0.1, False, "[1, 5]", "NR", SAMPLE_REF)
    r2_b = SingleEvalResult("qalign_8bit", 3.82, 0.1, False, "[1, 5]", "NR", SAMPLE_REF)
    rounds_map_multi = {
        1: {SAMPLE_DIST: [r1], SAMPLE_REF: [r1_b]},
        2: {SAMPLE_DIST: [r2], SAMPLE_REF: [r2_b]},
    }
    dlg_multi = MultiRoundSummaryDialog([SAMPLE_DIST, SAMPLE_REF], rounds_map_multi)
    assert dlg_multi.tab_widget.count() == 2
    assert dlg_multi.table_multi_summary.rowCount() == 1
    # Check winner column in summary
    assert "🏆 图 1" in dlg_multi.table_multi_summary.item(0, 9).text()


def test_single_view_multi_round_gui():
    window = MainWindow()
    view = window.view_single

    # Multi-round controls
    assert hasattr(view, "chk_multi_round")
    assert hasattr(view, "spin_rounds")
    assert hasattr(view, "combo_round_select")
    assert hasattr(view, "btn_open_multi_round")

    # Toggle checkbox
    view.chk_multi_round.setChecked(True)
    assert view.spin_rounds.isEnabled() is True
    view.spin_rounds.setValue(3)

    # Simulate multi-round finish
    from iqa_gui.core.evaluator import SingleEvalResult
    r1 = SingleEvalResult("psnr", 35.0, 0.1, False, "[0, 50]", "FR", SAMPLE_DIST)
    r2 = SingleEvalResult("psnr", 35.5, 0.1, False, "[0, 50]", "FR", SAMPLE_DIST)
    rounds_map = {
        1: {SAMPLE_DIST: [r1]},
        2: {SAMPLE_DIST: [r2]},
    }
    view.viewer_target.load_image(SAMPLE_DIST)
    view._on_rounds_eval_finished(rounds_map)
    view._on_eval_finished(rounds_map[1][SAMPLE_DIST])

    assert view.round_switch_widget.isHidden() is False
    assert view.btn_open_multi_round.isHidden() is False
    assert view.combo_round_select.count() == 2

    # Switch to round 2 in GUI
    view.combo_round_select.setCurrentIndex(1)
    assert view.current_round == 2
    assert "第 2 轮" in view.lbl_eval_status.text()

    # Test preview toggle (close / restore)
    assert hasattr(view, "btn_close_preview")
    assert hasattr(view, "btn_show_preview")
    assert view.tabs_view.isHidden() is False
    assert view.btn_show_preview.isHidden() is True

    view.set_preview_visible(False)
    assert view.tabs_view.isHidden() is True
    assert view.btn_show_preview.isHidden() is False

    view.set_preview_visible(True)
    assert view.tabs_view.isHidden() is False
    assert view.btn_show_preview.isHidden() is True





