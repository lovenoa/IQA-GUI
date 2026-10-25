"""Real Qt lifecycle checks and view-level audit regressions."""
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QThread, QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from iqa_gui import config
from iqa_gui.core.evaluator import SingleEvalResult
from iqa_gui.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def clean_test_windows():
    yield
    app.processEvents()
    for widget in QApplication.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


class ControlledWorker(QThread):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.was_cancelled = False
    def run(self):
        self.entered.set()
        self.release.wait(3)
    def cancel(self):
        self.was_cancelled = True


def finish(worker):
    worker.release.set()
    assert worker.wait(3000)
    app.processEvents()


def test_window_rejects_concurrent_tasks_without_starting_second():
    window = MainWindow()
    first, second = ControlledWorker(), ControlledWorker()
    try:
        assert window.begin_task(first, window.view_single)
        assert first.entered.wait(2)
        assert not window.begin_task(second, window.view_batch)
        assert not second.isRunning()
        assert not window.view_batch.btn_start.isEnabled()
        assert not window.btn_purge_vram.isEnabled()
    finally:
        finish(first)
        window.close()


def test_window_close_cancels_and_waits_before_destroying_running_thread():
    window = MainWindow()
    worker = ControlledWorker()
    window.show()
    try:
        assert window.begin_task(worker, window.view_single)
        assert worker.entered.wait(2)
        window.close()
        assert worker.was_cancelled
        assert worker.isRunning()
        assert not window.isHidden()
    finally:
        finish(worker)
        QTest.qWait(80)
        window.close()
    assert window.isHidden()


def test_saved_preset_survives_main_window_setting_toggle():
    config.save_settings(config.load_settings())
    window = MainWindow()
    settings = config.load_settings()
    settings['metric_presets']['new-preset'] = ['psnr']
    config.save_settings(settings)
    window._on_tips_toggled(not window.tips_enabled)
    assert config.load_settings()['metric_presets']['new-preset'] == ['psnr']
    window.close()


def test_poor_individual_grades_do_not_become_overall_qualified():
    window = MainWindow()
    view = window.view_single
    r = SingleEvalResult('ssim', .1, .01, False, '0,1', 'FR', 'missing')
    view._on_eval_finished([r])
    assert '整体合格' not in view.lbl_explanation.text()
    window.close()


def test_sorting_reuses_heatmap_without_gui_recomputation(sample_image_dirs, monkeypatch):
    window = MainWindow()
    view = window.view_single
    target = str(sample_image_dirs[0] / 'I03.bmp')
    ref = str(sample_image_dirs[1] / 'I03.bmp')
    import iqa_gui.ui.views.single_view as module
    original = module.generate_difference_heatmap
    calls = []
    def heatmap(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'generate_difference_heatmap', heatmap)
    r = SingleEvalResult('psnr', 30, .01, False, '0,60', 'FR', target, ref)
    view._on_eval_finished([r])
    view._on_header_section_clicked(0)
    view._on_header_section_clicked(0)
    assert len(calls) == 1
    window.close()


def test_unsaved_runtime_vram_toggle_is_honored_by_all_views(sample_image_dirs, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    window = MainWindow()
    import iqa_gui.ui.main_window as main_module
    def denied(*args, **kwargs):
        raise config.SettingsSaveError('read-only settings')
    monkeypatch.setattr(main_module, 'update_settings', denied)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: None)
    window._on_auto_release_vram_toggled(False)
    observed = []
    monkeypatch.setattr(window, 'begin_task', lambda worker, owner: observed.append(worker.auto_release_vram) or True)
    dist, ref = sample_image_dirs
    single = window.view_single
    single.metric_selector.set_selected(['niqe'])
    single.viewer_target.load_image(str(dist / 'I03.bmp'))
    single.run_evaluation()
    batch = window.view_batch
    batch.metric_selector.set_selected(['niqe'])
    batch.edit_target_dir.setText(str(dist))
    batch.start_batch()
    window.view_gmad.edit_dir.setText(str(dist))
    window.view_gmad.start_gmad()
    csv_file = tmp_path / 'scores.csv'
    csv_file.write_text('image,mos\nI03.bmp,10\nI04.bmp,20\n', encoding='utf-8')
    bench = window.view_bench
    bench.metric_selector.set_selected(['niqe'])
    bench.edit_root.setText(str(dist))
    bench.edit_csv.setText(str(csv_file))
    bench.run_benchmark()
    assert observed == [False, False, False, False]
    window.close()
