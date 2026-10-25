"""Offline regressions for sortable results, metric selection, and sparse reports."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from iqa_gui.core.batch_evaluator import BatchItem
from iqa_gui.core.evaluator import SingleEvalResult
from iqa_gui.core.model_manager import DependencyStatus, MetricInfo
from iqa_gui.ui.components import metric_selector as selector_module
from iqa_gui.ui.components.compare_dialog import MultiImageCompareDialog
from iqa_gui.ui.components.multi_round_dialog import MultiRoundSummaryDialog
from iqa_gui.ui.components.results_table import ResultsTableWidget


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def cleanup_widgets(qt_app):
    yield
    for widget in QApplication.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt_app.processEvents()


@pytest.fixture
def selector_factory(monkeypatch):
    models = [MetricInfo(n, n, mode, "test", False, "[0, 100]", "test")
              for n, mode in [("psnr", "FR"), ("ssim", "FR"), ("missing", "NR"), ("fid", "NR")]]
    monkeypatch.setattr(selector_module.ModelManager, "list_all", lambda self: models)
    monkeypatch.setattr(selector_module.ModelManager, "check_dependencies", lambda self, n:
                        DependencyStatus(n != "missing", ["absent-package"] if n == "missing" else []))
    monkeypatch.setattr(selector_module, "load_settings", lambda: {
        "metric_presets": {"经典基础通用": ["fid", "ssim", "psnr", "missing"]}})
    widgets = []
    def create(**kwargs):
        widget = selector_module.MetricSelectorWidget(**kwargs)
        widget.resize(600, 450)
        widget.show()
        QApplication.processEvents()
        widgets.append(widget)
        return widget
    yield create
    for widget in widgets:
        widget.close()


def result(name, score, path="a"):
    return SingleEvalResult(name, score, .1, False, "[0, 100]", "NR", path)


def test_sorted_batch_update_and_selection_keep_item_identity():
    widget = ResultsTableWidget()
    z = BatchItem(1, "z.bmp", "z.bmp")
    a = BatchItem(2, "a.bmp", "a.bmp")
    widget.set_batch([z, a], ["psnr"])
    widget.table.sortItems(1, Qt.SortOrder.AscendingOrder)
    z.status, z.scores = "completed", {"psnr": 31.0}
    widget.update_item(z)
    assert [widget.table.item(i, 1).text() for i in range(2)] == ["a.bmp", "z.bmp"]
    seen = []
    widget.row_selected.connect(seen.append)
    widget.table.selectRow(0)
    assert seen[-1] is a


@pytest.mark.parametrize("column", [0, 4, 5])
def test_batch_numeric_columns_sort_numerically(column):
    widget = ResultsTableWidget()
    widget.set_batch([BatchItem(10, "x", "x", scores={"psnr": 10}, time_taken=10),
                      BatchItem(2, "y", "y", scores={"psnr": 2}, time_taken=2)], ["psnr"])
    widget.table.sortItems(column, Qt.SortOrder.AscendingOrder)
    assert widget.table.item(0, 1).text() == "y"


def test_checkbox_click_toggles_once(selector_factory):
    widget = selector_factory()
    item = widget.list_widget.item(0)
    rect = widget.list_widget.visualItemRect(item)
    QTest.mouseClick(widget.list_widget.viewport(), Qt.MouseButton.LeftButton,
                     pos=QPoint(rect.left() + 8, rect.center().y()))
    assert widget.get_selected() == ["psnr"]
    QTest.mouseClick(widget.list_widget.viewport(), Qt.MouseButton.LeftButton,
                     pos=QPoint(rect.left() + 8, rect.center().y()))
    assert widget.get_selected() == []


def test_select_all_does_not_select_missing_dependencies(selector_factory, monkeypatch):
    widget = selector_factory()
    # A declined prompt must not leave hidden selection state behind.
    def decline(name, status, item):
        widget.list_widget.blockSignals(True)
        item.setCheckState(Qt.CheckState.Unchecked)
        widget.list_widget.blockSignals(False)
    monkeypatch.setattr(widget, "_prompt_missing_dependencies", decline)
    widget.select_all_visible()
    assert widget.get_selected() == ["fid", "psnr", "ssim"]


def test_allowed_metrics_constrain_filters_presets_and_programmatic_selection(selector_factory):
    widget = selector_factory(allowed_metrics=["ssim", "psnr"])
    assert widget.list_widget.count() == 2
    widget.set_selected(["fid", "ssim", "missing", "psnr", "unknown"])
    assert widget.get_selected() == ["psnr", "ssim"]
    widget.clear_selection()
    widget.select_classic_presets()
    assert widget.get_selected() == ["psnr", "ssim"]
    widget.clear_selection()
    widget.select_all_visible()
    assert widget.get_selected() == ["psnr", "ssim"]


def test_mode_filter_also_constrains_programmatic_selection(selector_factory):
    widget = selector_factory(mode_filter="FR")
    widget.set_selected(["fid", "psnr"])
    assert widget.get_selected() == ["psnr"]


def test_preset_save_failure_is_reported_without_success(selector_factory, monkeypatch):
    from iqa_gui.config import SettingsSaveError
    widget = selector_factory()
    widget.set_selected(["psnr"])
    monkeypatch.setattr(selector_module.QInputDialog, "getText", lambda *a: ("custom", True))
    def fail_save(settings):
        raise SettingsSaveError("read-only settings")
    monkeypatch.setattr(selector_module, "save_settings", fail_save)
    messages = []
    monkeypatch.setattr(selector_module.QMessageBox, "warning", lambda *a: messages.append(a[2]))
    monkeypatch.setattr(selector_module.QMessageBox, "information", lambda *a: pytest.fail("false save success"))
    widget._save_current_as_preset()
    assert messages and "read-only settings" in messages[0]
    assert widget.get_selected() == ["psnr"]


def test_installer_controls_wait_for_native_finished(selector_factory, monkeypatch):
    class Installer(QObject):
        progress = Signal(str)
        result_ready = Signal(bool, str)
        finished = Signal()
        def __init__(self, packages, parent=None):
            super().__init__(parent)
        def start(self):
            pass
        def isRunning(self):
            return True
    monkeypatch.setattr(selector_module, "SafePackageInstallerWorker", Installer)
    monkeypatch.setattr(selector_module, "launch_worker", lambda worker, owner: True)
    monkeypatch.setattr(selector_module.QMessageBox, "information", lambda *a: None)
    widget = selector_factory()
    widget._start_safe_install(["absent-package"], "psnr")
    assert not widget.btn_select_all.isEnabled()
    worker = widget._installer_worker
    worker.result_ready.emit(True, "done")
    assert not widget.btn_select_all.isEnabled()
    worker.finished.emit()
    assert widget.btn_select_all.isEnabled()
    assert widget.get_selected() == ["psnr"]


def test_rejected_installer_launch_restores_controls(selector_factory, monkeypatch):
    widget = selector_factory()
    monkeypatch.setattr(selector_module, "launch_worker", lambda worker, owner: False)
    widget._start_safe_install(["absent-package"], "psnr")
    assert widget.btn_select_all.isEnabled()
    assert widget._installer_worker is None


def test_compare_clipscore_does_not_add_quality_grade():
    dialog = MultiImageCompareDialog(["a", "b"], {
        "a": [result("clipscore", 1.5)], "b": [result("clipscore", 1.9, "b")]})
    assert "级" not in dialog.table_compare.item(0, 3).text()


def test_compare_uses_metric_union_and_requires_two_valid_scores():
    dialog = MultiImageCompareDialog(["a", "b"], {"a": [result("psnr", 10)], "b": [result("ssim", 20, "b")]})
    assert dialog.table_compare.rowCount() == 2
    assert dialog.table_compare.item(0, 4).text() == "N/A"
    assert "不足" in dialog.table_compare.item(0, 5).text()
    assert "不足" in dialog.table_compare.item(1, 5).text()
    assert "胜出 <b>0</b>" in dialog.lbl_summary_bar.text()


def test_single_round_report_keeps_missing_round_and_statistics_columns():
    dialog = MultiRoundSummaryDialog(["a"], {1: {"a": [result("psnr", 10)]}, 2: {},
                                           3: {"a": [result("psnr", 14), result("ssim", 30)]}})
    table = dialog.table_single
    assert table.rowCount() == 2
    assert [table.item(0, col).text() for col in range(3, 9)] == [
        "10.0000", "N/A", "14.0000", "12.0000", "4.0000", "2.8284"]
    assert table.item(1, 3).text() == "N/A"
    assert table.item(1, 4).text() == "N/A"
    assert table.item(1, 5).text() == "30.0000"
    assert "不足" in table.item(1, 9).text()


def test_multi_round_union_missing_images_are_na_and_not_winners():
    dialog = MultiRoundSummaryDialog(["a", "b"], {1: {"a": [result("psnr", 10)]},
                                                    3: {"b": [result("ssim", 20, "b")]}})
    table = dialog.table_multi_summary
    assert table.rowCount() == 2
    assert [table.item(0, col).text() for col in (6, 7, 8)] == ["N/A"] * 3
    assert "不足" in table.item(0, 9).text()
    assert "不足" in table.item(1, 9).text()
    details = dialog.table_multi_details
    assert details.rowCount() == 4
    assert details.item(2, 0).text() == "第 3 轮"
    assert details.item(1, 4).text() == "N/A"
    assert details.item(1, 5).text() == "N/A"
    assert "不足" in details.item(0, 6).text()
