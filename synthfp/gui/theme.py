"""界面主题与通用样式（暗色 / 亮色）。

刻意不引入 qdarktheme / pyqtgraph 等额外依赖（设计文档 0.2「最小依赖」），
只用一个 QSS 字符串 + QPalette 实现，任何 Qt 版本都能跑。
"""

from __future__ import annotations

from .qt import QApplication, QColor, QPalette

__all__ = ["THEMES", "apply_theme", "theme_names"]

THEMES = ("dark", "light")

# 调色板（暗色为默认：科研长时间盯屏更舒适，截图对比也更清晰）
_DARK = {
    "bg": "#1e1e23",
    "bg_alt": "#26262c",
    "base": "#2d2d34",
    "border": "#3d3d46",
    "text": "#dcdce2",
    "text_dim": "#9a9aa6",
    "accent": "#0a84ff",
    "accent_hover": "#3d9bff",
    "ok": "#34c759",
    "warn": "#ffb020",
    "err": "#ff453a",
    "sel": "#0a84ff",
}

_LIGHT = {
    "bg": "#f6f6f9",
    "bg_alt": "#ececf1",
    "base": "#ffffff",
    "border": "#d0d0d8",
    "text": "#1e1e22",
    "text_dim": "#6a6a75",
    "accent": "#0a6fd8",
    "accent_hover": "#1a83ea",
    "ok": "#1a8a3a",
    "warn": "#b07000",
    "err": "#d02b20",
    "sel": "#0a6fd8",
}

_QSS = """
QWidget {{
    background-color: {bg};
    color: {text};
    font-size: 13px;
    font-family: "Microsoft YaHei UI", "Segoe UI", "PingFang SC", sans-serif;
}}
QMainWindow, QDialog {{ background-color: {bg}; }}

/* --- 导航侧栏 --- */
QListWidget#NavList {{
    background-color: {bg_alt};
    border: none;
    border-right: 1px solid {border};
    outline: none;
    padding: 8px 0;
    font-size: 14px;
}}
QListWidget#NavList::item {{
    padding: 11px 16px;
    border-left: 3px solid transparent;
    color: {text_dim};
}}
QListWidget#NavList::item:hover {{ background-color: {base}; color: {text}; }}
QListWidget#NavList::item:selected {{
    background-color: {base};
    border-left: 3px solid {accent};
    color: {text};
    font-weight: 600;
}}

/* --- 分组框 --- */
QGroupBox {{
    border: 1px solid {border};
    border-radius: 6px;
    margin-top: 14px;
    padding: 12px 10px 8px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {accent};
}}

/* --- 按钮 --- */
QPushButton {{
    background-color: {base};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 6px 14px;
    min-height: 22px;
}}
QPushButton:hover {{ border-color: {accent}; background-color: {bg_alt}; }}
QPushButton:pressed {{ background-color: {border}; }}
QPushButton:disabled {{ color: {text_dim}; border-color: {border}; background-color: {bg_alt}; }}
QPushButton#Primary {{
    background-color: {accent};
    color: #ffffff;
    border: none;
    font-weight: 700;
    padding: 8px 22px;
    font-size: 14px;
}}
QPushButton#Primary:hover {{ background-color: {accent_hover}; }}
QPushButton#Primary:disabled {{ background-color: {border}; color: {text_dim}; }}
QPushButton#Danger {{
    background-color: {err}; color: #ffffff; border: none; font-weight: 600;
}}
QPushButton#Danger:hover {{ background-color: {err}; }}
QPushButton#Danger:disabled {{
    background-color: {bg_alt};
    color: {text_dim};
    border: 1px solid {border};
}}
QPushButton#Ghost {{ background-color: transparent; border: 1px dashed {border}; }}

/* --- 输入控件 --- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background-color: {base};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: {sel};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {accent}; }}
QLineEdit[invalid="true"], QSpinBox[invalid="true"], QDoubleSpinBox[invalid="true"] {{
    border-color: {err};
    background-color: rgba(255, 69, 58, 0.08);
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {base};
    border: 1px solid {border};
    selection-background-color: {sel};
    color: {text};
}}
/* 刻意不给 QSpinBox/QDoubleSpinBox 的上下按钮写样式：
   一旦用 QSS 改写了 ::up-button / ::down-button 却不提供箭头图片，
   Qt 就只画出一堆小方点，反而比原生箭头更难用。这里保持系统绘制。 */

/* --- 表格 / 列表 / 树 --- */
QTableWidget, QTableView, QTreeWidget, QListWidget {{
    background-color: {base};
    alternate-background-color: {bg_alt};
    border: 1px solid {border};
    border-radius: 4px;
    gridline-color: {border};
    selection-background-color: {sel};
    selection-color: #ffffff;
}}
QHeaderView::section {{
    background-color: {bg_alt};
    color: {text_dim};
    border: none;
    border-right: 1px solid {border};
    border-bottom: 1px solid {border};
    padding: 5px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background-color: {bg_alt}; border: none; }}

/* --- 进度条 --- */
QProgressBar {{
    background-color: {bg_alt};
    border: 1px solid {border};
    border-radius: 5px;
    text-align: center;
    color: {text};
    min-height: 18px;
}}
QProgressBar::chunk {{ background-color: {accent}; border-radius: 4px; }}

/* --- 标签页 --- */
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 4px; background-color: {bg}; }}
QTabBar::tab {{
    background-color: {bg_alt};
    color: {text_dim};
    padding: 7px 16px;
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 3px;
}}
QTabBar::tab:selected {{ background-color: {bg}; color: {text}; font-weight: 600; }}
QTabBar::tab:hover {{ color: {text}; }}

/* --- 滚动条 --- */
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {border}; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {text_dim}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {border}; border-radius: 5px; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* --- 其他 --- */
QToolBar {{
    background-color: {bg_alt};
    border-bottom: 1px solid {border};
    spacing: 4px;
    padding: 4px 6px;
}}
QToolBar QToolButton {{ padding: 5px 10px; border-radius: 4px; }}
QToolBar QToolButton:hover {{ background-color: {base}; }}
QStatusBar {{ background-color: {bg_alt}; border-top: 1px solid {border}; color: {text_dim}; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background-color: {border}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}
QCheckBox, QRadioButton {{ spacing: 7px; }}
/* 显式给勾选框画底框：只写 width/height 而不给 image 时，未选中状态
   什么都不会画（看起来像纯文字），用户无法判断哪些项没勾。
   这里用"边框 + 选中填充强调色"代替原生图片，跨平台一致。 */
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {border};
    border-radius: 3px;
    background-color: {base};
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: {bg_alt};
    border-color: {border};
}}
QRadioButton::indicator {{ border-radius: 7px; }}
QRadioButton::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}
QLabel#Hint {{ color: {text_dim}; }}
QLabel#Accent {{ color: {accent}; font-weight: 600; }}
QLabel#Ok {{ color: {ok}; font-weight: 600; }}
QLabel#Warn {{ color: {warn}; font-weight: 600; }}
QLabel#Err {{ color: {err}; font-weight: 600; }}
QLabel#StatValue {{ font-size: 20px; font-weight: 700; color: {accent}; }}
QLabel#Card {{
    background-color: {base};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 10px 14px;
}}
QFrame#Card {{
    background-color: {base};
    border: 1px solid {border};
    border-radius: 6px;
}}
QFrame#HLine {{ background-color: {border}; max-height: 1px; border: none; }}
QGraphicsView {{ background-color: {bg_alt}; border: 1px solid {border}; border-radius: 4px; }}
QSlider::groove:horizontal {{ height: 4px; background: {border}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {accent}; width: 13px; margin: -5px 0; border-radius: 6px;
}}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QMenu {{ background-color: {base}; border: 1px solid {border}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 3px; }}
QMenu::item:selected {{ background-color: {sel}; color: #ffffff; }}
QToolTip {{
    background-color: {base};
    color: {text};
    border: 1px solid {accent};
    padding: 4px 6px;
}}
QMessageBox {{ background-color: {bg}; }}
"""


def theme_names() -> tuple:
    return THEMES


def apply_theme(app: QApplication, name: str = "dark") -> None:
    """把主题应用到整个应用（可随时切换）。"""
    colors = _DARK if name == "dark" else _LIGHT

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(colors["bg"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(colors["base"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["bg_alt"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(colors["base"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["sel"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(colors["base"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(colors["text_dim"]))
    app.setPalette(palette)
    app.setStyleSheet(_QSS.format(**colors))
