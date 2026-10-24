"""Qt 绑定兼容层 —— 同时支持 PySide6 与 PyQt6。

**为什么需要它**：设计文档 3.5 选定 PySide6（官方绑定、LGPL），
但部署环境里往往已经装好了 PyQt6（很多科研机器都带着）。
让 GUI 同时接受两者，可以省掉一次 80 MB 的安装，也让打包更灵活。

**用法**（GUI 内所有模块统一这样写）::

    from synthfp.gui.qt import QtWidgets, Qt, Signal, QApplication, QMessageBox

**API 差异处理**

* PySide6 用 ``Signal`` / ``Slot`` / ``Property``，PyQt6 用
  ``pyqtSignal`` / ``pyqtSlot`` / ``pyqtProperty`` → 本模块统一暴露前者。
* 枚举访问方式两边一致（都用作用域枚举，如 ``Qt.AlignmentFlag.AlignCenter``），
  因此 ``Qt`` 可以直接转发。
* 其余差异极少，未显式导出的名字可通过 ``__getattr__`` 回退到当前绑定的
  ``QtCore`` / ``QtGui`` / ``QtWidgets``，所以永远不会因为漏写一个名字而崩。
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "QT_BINDING",
    "QT_VERSION",
    "QtBindingError",
    "is_available",
    "require_qt",
    "QtCore",
    "QtGui",
    "QtWidgets",
]


class QtBindingError(ImportError):
    """两个 Qt 绑定都不可用时抛出（信息里给出安装指引）。"""


def _load() -> tuple:
    """载入可用的 Qt 绑定，返回 ``(binding_name, QtCore, QtGui, QtWidgets)``。"""
    errors: list = []

    try:  # 首选 PySide6（设计文档 3.5 的选型）
        return (
            "PySide6",
            importlib.import_module("PySide6.QtCore"),
            importlib.import_module("PySide6.QtGui"),
            importlib.import_module("PySide6.QtWidgets"),
        )
    except ImportError as exc:  # pragma: no cover - 取决于环境
        errors.append(f"PySide6: {exc}")

    try:  # 备选 PyQt6
        return (
            "PyQt6",
            importlib.import_module("PyQt6.QtCore"),
            importlib.import_module("PyQt6.QtGui"),
            importlib.import_module("PyQt6.QtWidgets"),
        )
    except ImportError as exc:  # pragma: no cover - 取决于环境
        errors.append(f"PyQt6: {exc}")

    raise QtBindingError(
        "未找到可用的 Qt 绑定，图形界面无法启动。\n"
        "  " + "\n  ".join(errors) + "\n"
        "安装其一即可（推荐前者）：\n"
        "  pip install PySide6\n"
        "  pip install PyQt6\n"
        "提示：不装 GUI 也能用命令行：synthfp generate / stats / validate"
    )


QtCore: Any = None
QtGui: Any = None
QtWidgets: Any = None
QT_BINDING: str = ""
QT_VERSION: str = ""

_LOAD_ERROR: Exception | None = None

try:
    QT_BINDING, QtCore, QtGui, QtWidgets = _load()
    QT_VERSION = str(
        getattr(QtCore, "__version__", "") or getattr(QtCore, "PYQT_VERSION_STR", "") or ""
    )
except QtBindingError as exc:  # pragma: no cover - 无 Qt 环境
    _LOAD_ERROR = exc


def is_available() -> bool:
    """当前环境是否有可用的 Qt 绑定。"""
    return QtCore is not None


def require_qt() -> None:
    """没有 Qt 时抛出带安装指引的异常。"""
    if _LOAD_ERROR is not None:
        raise QtBindingError(str(_LOAD_ERROR))


# ---------------------------------------------------------------------------
# 信号 / 槽 / 属性：统一命名
# ---------------------------------------------------------------------------
if QtCore is not None:
    Signal = getattr(QtCore, "Signal", None) or QtCore.pyqtSignal
    Slot = getattr(QtCore, "Slot", None) or QtCore.pyqtSlot
    Property = getattr(QtCore, "Property", None) or getattr(QtCore, "pyqtProperty", property)
else:  # pragma: no cover - 无 Qt 时给出可导入的占位符
    Signal = Slot = Property = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 名字转发：常用 Qt 名字直接挂在本模块命名空间上
# ---------------------------------------------------------------------------
_QT_NAMES = (
    # QtCore
    "Qt", "QObject", "QTimer", "QThread", "QUrl", "QSize", "QPoint", "QPointF", "QRect",
    "QRectF", "QEvent", "QEasingCurve", "QAbstractItemModel", "QModelIndex", "QSettings",
    # QtGui
    "QAction", "QColor", "QCloseEvent", "QDesktopServices", "QFont", "QIcon", "QImage",
    "QKeySequence", "QPainter", "QPalette", "QPen", "QBrush", "QPixmap", "QTransform",
    "QWheelEvent", "QMouseEvent", "QStandardItemModel",
    # QtWidgets
    "QAbstractItemView", "QApplication", "QCheckBox", "QComboBox", "QDialog", "QDoubleSpinBox",
    "QFileDialog", "QFormLayout", "QFrame", "QGraphicsPixmapItem", "QGraphicsScene",
    "QGraphicsView", "QGridLayout", "QGroupBox", "QHBoxLayout", "QHeaderView", "QLabel",
    "QLineEdit", "QListWidget", "QListWidgetItem", "QMainWindow", "QMenu", "QMenuBar",
    "QMessageBox", "QPlainTextEdit", "QProgressBar", "QPushButton", "QRadioButton",
    "QScrollArea", "QSizePolicy", "QSlider", "QSpinBox", "QSplitter", "QStackedWidget",
    "QStatusBar", "QTabBar", "QTabWidget", "QTableWidget", "QTableWidgetItem", "QTextEdit",
    "QToolBar", "QToolButton", "QTreeWidget", "QTreeWidgetItem", "QVBoxLayout", "QWidget",
    "QStyle", "QStyleFactory", "QInputDialog",
)


def _install_names() -> list:
    """把常用名字挂到本模块；返回成功安装的名字列表。"""
    installed = []
    for name in _QT_NAMES:
        for module in (QtWidgets, QtGui, QtCore):
            if module is None:
                continue
            value = getattr(module, name, None)
            if value is not None:
                globals()[name] = value
                installed.append(name)
                break
    return installed


_NAMES: list = _install_names()
__all__.extend(_NAMES)


def __getattr__(name: str) -> Any:  # pragma: no cover - 兜底路径
    """未显式导出的 Qt 名字回退到当前绑定，避免漏写导致 ImportError。"""
    for module in (QtWidgets, QtGui, QtCore):
        if module is None:
            continue
        value = getattr(module, name, None)
        if value is not None:
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
