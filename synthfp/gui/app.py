"""GUI 启动入口。

职责：
1. 建立 ``QApplication`` 并应用主题；
2. 安装全局异常钩子（把未捕获异常写成 ``crash.log`` 并弹窗，而不是静默退出）；
3. 显示主窗口。

启动方式::

    synthfp gui
    python -m synthfp gui
    synthfp-gui
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

__all__ = ["run_gui", "install_excepthook"]

_CRASH_LOG = Path("crash.log")


def install_excepthook(log_path: Path = _CRASH_LOG) -> None:
    """把未捕获异常写入 crash.log 并提示用户，避免 GUI 静默崩溃。"""

    def hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write("\n" + "=" * 60 + "\n" + text)
        except OSError:
            pass
        sys.stderr.write(text)
        try:
            from .qt import QApplication, QMessageBox

            if QApplication.instance() is not None:
                QMessageBox.critical(
                    None,
                    "出现未处理的错误",
                    f"{exc_type.__name__}: {exc_value}\n\n"
                    f"完整堆栈已写入 {log_path.resolve()}\n\n"
                    "如果问题可复现，请把该文件连同操作步骤一起反馈。",
                )
        except Exception:  # pragma: no cover
            pass

    sys.excepthook = hook


def run_gui(config=None) -> int:
    """启动图形界面，返回进程退出码。"""
    try:
        from .qt import QApplication
    except ImportError as exc:  # pragma: no cover
        print(
            "无法启动 GUI：缺少 PySide6。\n"
            f"  详情: {exc}\n"
            "  安装: pip install PySide6\n"
            "  提示: 不想装 GUI 也可以用 CLI：\n"
            "        synthfp generate --preset quick_test\n"
            "        synthfp stats --dataset outputs/synth_quick_test",
            file=sys.stderr,
        )
        return 1

    from .. import __version__
    from ..config import load_default_config
    from .main_window import MainWindow
    from .theme import apply_theme

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("SLIM-FRNet")
    # 刻意**不**设置 applicationDisplayName：Qt 会把它追加到每个窗口标题后面
    # （实测 Windows 上标题变成 "SLIM-FRNet … — demo - SLIM-FRNet …"）。
    # 品牌名已经写在 MainWindow 的标题里，这里留空即可。
    app.setApplicationDisplayName("")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("SLIM-FRNet")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    install_excepthook()
    apply_theme(app, "dark")

    if config is None:
        config = load_default_config()

    window = MainWindow(config)
    window.show()
    window.statusBar().showMessage(
        f"synthfp v{__version__} 就绪 — 左侧 8 个页面：配置 → 后端 → TTA → 质量 → 监控 → 预览 → 统计 → 导出",
        12000,
    )
    return int(app.exec())


def main() -> int:  # pragma: no cover - 供 synthfp-gui 脚本使用
    return run_gui()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
