"""Application launcher and entry point for IQA-GUI."""

from __future__ import annotations

import os
import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from iqa_gui.config import load_settings
from iqa_gui.ui.main_window import MainWindow
from iqa_gui.ui.theme import apply_theme_to_app


def global_exception_hook(exctype, value, traceback):
    """Global exception hook to prevent app crash and display friendly dialog."""
    import traceback as tb
    err_str = "".join(tb.format_exception(exctype, value, traceback))
    print("Unhandled Exception:\n", err_str, file=sys.stderr)
    try:
        QMessageBox.critical(
            None,
            "程序运行异常",
            f"遇到未处理的错误，详情如下:\n\n{value}\n\n详细追踪信息已记录至终端控制台。",
        )
    except Exception:
        pass


def main() -> int:
    """Main entry point."""
    # Ensure high DPI scaling looks crisp on modern displays
    if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QApplication(sys.argv)
    app.setApplicationName("IQA-GUI")
    app.setOrganizationName("IQA-PyTorch")

    # Safe mock for missing optional packages like icecream required by q_align
    try:
        import icecream
    except ImportError:
        import types
        mod = types.ModuleType("icecream")
        mod.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
        sys.modules["icecream"] = mod

    # Safe fallback shim for pkg_resources (used by clip)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            import pkg_resources
        except ImportError:
            try:
                import types
                import packaging
                import packaging.version
                mod = types.ModuleType("pkg_resources")
                mod.packaging = packaging
                sys.modules["pkg_resources"] = mod
            except Exception:
                pass

    from iqa_gui.core.evaluator import patch_mplug_owl_compatibility
    patch_mplug_owl_compatibility()

    from iqa_gui.core.log_manager import LogManager
    LogManager.get_instance().install()

    # Apply saved theme or auto-detect Windows dark/light mode
    settings = load_settings()
    theme_mode = settings.get("theme", "auto")
    apply_theme_to_app(app, theme_mode)

    sys.excepthook = global_exception_hook

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
