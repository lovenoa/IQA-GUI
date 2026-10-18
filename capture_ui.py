import os
import sys

# Use default Windows platform for full DirectWrite / Microsoft YaHei font support

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication
from iqa_gui.ui.main_window import MainWindow
from iqa_gui.ui.theme import apply_theme_to_app, THEME_MODE_DARK, THEME_MODE_LIGHT

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

output_dir = r"C:\Users\loven\.gemini\antigravity\brain\5253b8da-04c6-4819-b118-8974b9e67c1b"
sample_dist = os.path.abspath("IQA-PyTorch/ResultsCalibra/dist_dir/I03.bmp")
sample_ref = os.path.abspath("IQA-PyTorch/ResultsCalibra/ref_dir/I03.bmp")

def capture_mode(theme_mode: str, suffix: str):
    apply_theme_to_app(app, theme_mode)
    win = MainWindow()
    win.resize(1200, 800)
    win.show()

    # Tab 0: Single View
    win.tabs.setCurrentIndex(0)
    win.view_single.viewer_target.load_image(sample_dist)
    win.view_single.viewer_ref.load_image(sample_ref)
    win.view_single.metric_selector.select_anime_presets()
    
    # Simulate single eval results for scorecard view
    from iqa_gui.core.evaluator import SingleEvalResult
    mock_results = [
        SingleEvalResult(
            metric_name="laion_aes",
            score=7.42,
            time_seconds=0.14,
            lower_better=False,
            score_range="[1, 10]",
            metric_mode="NR",
            target_path=sample_dist,
        ),
        SingleEvalResult(
            metric_name="clipiqa",
            score=0.73,
            time_seconds=0.18,
            lower_better=False,
            score_range="[0, 1]",
            metric_mode="NR",
            target_path=sample_dist,
        ),
        SingleEvalResult(
            metric_name="musiq",
            score=71.80,
            time_seconds=0.22,
            lower_better=False,
            score_range="[0, 100]",
            metric_mode="NR",
            target_path=sample_dist,
        ),
        SingleEvalResult(
            metric_name="maniqa",
            score=0.64,
            time_seconds=0.20,
            lower_better=False,
            score_range="[0, 1]",
            metric_mode="NR",
            target_path=sample_dist,
        ),
    ]
    win.view_single._on_eval_finished(mock_results)
    
    app.processEvents()
    pix = win.grab()
    pix.save(os.path.join(output_dir, f"ui_tab0_single_{suffix}.png"))

    # Tab 2: gMAD View
    win.tabs.setCurrentIndex(2)
    app.processEvents()
    pix2 = win.grab()
    pix2.save(os.path.join(output_dir, f"ui_tab2_gmad_{suffix}.png"))

    # Tab 3: Benchmark View
    win.tabs.setCurrentIndex(3)
    app.processEvents()
    pix3 = win.grab()
    pix3.save(os.path.join(output_dir, f"ui_tab3_bench_{suffix}.png"))

    # Tab 4: Model Zoo View
    win.tabs.setCurrentIndex(4)
    if win.view_zoo.table.rowCount() > 0:
        win.view_zoo.table.selectRow(0)
    app.processEvents()
    pix4 = win.grab()
    pix4.save(os.path.join(output_dir, f"ui_tab4_zoo_{suffix}.png"))

    win.close()

if __name__ == "__main__":
    capture_mode(THEME_MODE_DARK, "dark")
    capture_mode(THEME_MODE_LIGHT, "light")
    print("Screenshots captured successfully.")
