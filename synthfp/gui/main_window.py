"""主窗口 —— 三区布局 + 侧边导航（设计文档 3.1 / 3.2）。

::

    ┌────────────────────────────────────────────────────────────┐
    │ 菜单栏: 文件 | 配置 | 生成 | 视图 | 帮助                       │
    ├────────────────────────────────────────────────────────────┤
    │ 工具栏: 新建 | 打开 | 保存 | 预设 | 开始 | 暂停 | 停止        │
    ├──────────────┬─────────────────────────────────────────────┤
    │  侧边导航     │            页面（QStackedWidget）             │
    │  8 个页面     │                                             │
    ├──────────────┴─────────────────────────────────────────────┤
    │ 状态栏: 数据集 | 规模 | 进度 | 吞吐 | 校验状态                 │
    └────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .. import __version__
from ..config import PRESETS, AppConfig, ConfigError, build_config, preset_names
from .config_store import ConfigStore
from .pages import PAGE_CLASSES, MonitorPage
from .qt import (
    QAction,
    QApplication,
    QCloseEvent,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QKeySequence,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    Qt,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from .theme import apply_theme

logger = logging.getLogger(__name__)

__all__ = ["MainWindow"]

#: 窗口标题前缀（品牌名，集中一处便于统一修改）
APP_TITLE = "SLIM-FRNet 合成指纹数据集平台"
APP_SUBTITLE = "SLIM-FRNet"


def _sep() -> QLabel:
    """状态栏里的细分隔符。"""
    label = QLabel("│")
    label.setObjectName("Hint")
    return label


class MainWindow(QMainWindow):
    """应用主窗口。"""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__()
        self.store = ConfigStore(config or build_config("default"), self)
        self.setWindowTitle(f"{APP_TITLE} v{__version__}")
        self.resize(1440, 900)
        self.setMinimumSize(1080, 680)

        self.pages: List[QWidget] = []
        self._page_index: Dict[str, int] = {}
        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()

        # 各页面在构造时做了首屏「配置 → 控件」回填；那只读不写，
        # 这里再显式清一次"有未保存修改"标记，避免刚打开就被问是否保存。
        self.store.mark_clean()

        self.store.changed.connect(self._on_config_changed)
        self.store.error.connect(self._on_config_error)
        self.store.loaded.connect(
            lambda path: self.statusBar().showMessage(f"已加载配置：{path}", 6000)
        )
        self._on_config_changed()

    # ==================================================================
    # 界面
    # ==================================================================
    def _build_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 侧边导航
        self.nav = QListWidget()
        self.nav.setObjectName("NavList")
        self.nav.setFixedWidth(186)
        for index, page_cls in enumerate(PAGE_CLASSES):
            item = QListWidgetItem(f"{page_cls.icon}  {page_cls.title}")
            item.setToolTip(f"第 {index + 1} 页：{page_cls.title}")
            self.nav.addItem(item)
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        layout.addWidget(self.nav)

        # 页面容器
        from .qt import QStackedWidget

        self.stack = QStackedWidget()
        for page_cls in PAGE_CLASSES:
            page = page_cls(self.store)
            self.stack.addWidget(page)
            self.pages.append(page)
            self._page_index[page_cls.__name__] = len(self.pages) - 1
        layout.addWidget(self.stack, 1)

        splitter_host = QWidget()
        splitter_host.setLayout(layout)
        self.setCentralWidget(splitter_host)
        self.nav.setCurrentRow(0)

        # 监控页 → 状态栏
        monitor = self._page("MonitorPage")
        if monitor is not None:
            monitor.status_message.connect(self._on_run_status)

    def _build_menu(self) -> None:
        bar = self.menuBar()

        # 文件
        file_menu = bar.addMenu("文件(&F)")
        self._action(file_menu, "新建（默认预设）", QKeySequence.StandardKey.New, self._new_config)
        self._action(file_menu, "打开配置…", QKeySequence.StandardKey.Open, self._open_config)
        self._action(file_menu, "保存配置", QKeySequence.StandardKey.Save, self._save_config)
        self._action(file_menu, "另存配置为…", QKeySequence.StandardKey.SaveAs, self._save_config_as)
        file_menu.addSeparator()
        self._action(file_menu, "打开数据集目录", None, self._open_dataset_dir)
        file_menu.addSeparator()
        self._action(file_menu, "退出", QKeySequence.StandardKey.Quit, self.close)

        # 配置
        config_menu = bar.addMenu("配置(&C)")
        preset_menu = config_menu.addMenu("应用预设")
        for name in preset_names():
            preset = PRESETS[name]
            action = QAction(f"{preset.label}（{name}）", self)
            action.setToolTip(preset.description)
            action.triggered.connect(lambda _c=False, n=name: self._apply_preset(n))
            preset_menu.addAction(action)
        config_menu.addSeparator()
        self._action(config_menu, "校验当前配置", "Ctrl+Shift+V", self._validate_config)
        self._action(config_menu, "复制配置为 YAML", None, self._copy_yaml)
        self._action(config_menu, "查看完整配置…", None, self._show_config)

        # 生成
        run_menu = bar.addMenu("生成(&R)")
        self._action(run_menu, "开始生成", "F5", self._start_run)
        self._action(run_menu, "暂停 / 恢复", "F6", self._toggle_pause)
        self._action(run_menu, "停止", "F7", self._stop_run)
        run_menu.addSeparator()
        self._action(run_menu, "预览生成计划", "F8", self._preview_plan)

        # 视图
        view_menu = bar.addMenu("视图(&V)")
        theme_menu = view_menu.addMenu("主题")
        for name, label in (("dark", "暗色"), ("light", "亮色")):
            action = QAction(label, self)
            action.triggered.connect(lambda _c=False, n=name: self._set_theme(n))
            theme_menu.addAction(action)
        view_menu.addSeparator()
        for index, page_cls in enumerate(PAGE_CLASSES):
            action = QAction(f"{index + 1}. {page_cls.title}", self)
            action.setShortcut(f"Ctrl+{index + 1}")
            action.triggered.connect(lambda _c=False, i=index: self.nav.setCurrentRow(i))
            view_menu.addAction(action)

        # 帮助
        help_menu = bar.addMenu("帮助(&H)")
        self._action(help_menu, "自检（渲染 + 各阶段）", None, self._self_check)
        self._action(help_menu, "关于 synthfp", None, self._about)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(toolbar)

        self._tool(toolbar, "📄 新建", self._new_config, "新建默认配置")
        self._tool(toolbar, "📂 打开", self._open_config, "打开 YAML 配置")
        self._tool(toolbar, "💾 保存", self._save_config, "保存当前配置")

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("应用预设…", "")
        for name in preset_names():
            self.preset_combo.addItem(f"{PRESETS[name].label}", name)
        self.preset_combo.setToolTip("一键套用内置预设")
        self.preset_combo.currentIndexChanged.connect(self._on_preset_combo)
        toolbar.addWidget(self.preset_combo)

        toolbar.addSeparator()
        self._tool(toolbar, "▶ 开始生成", self._start_run, "开始批量生成（F5）")
        self._tool(toolbar, "⏸ 暂停", self._toggle_pause, "暂停 / 恢复（F6）")
        self._tool(toolbar, "■ 停止", self._stop_run, "停止（F7）")

        toolbar.addSeparator()
        self._tool(toolbar, "📊 统计", lambda: self._goto("StatsPage"), "跳到统计报告")
        self._tool(toolbar, "📦 导出", lambda: self._goto("ExportPage"), "跳到导出")
        self._tool(toolbar, "🔍 预览", lambda: self._goto("PreviewPage"), "跳到结果预览")

        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy(), spacer.sizePolicy().verticalPolicy())
        toolbar.addWidget(spacer)

    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.status_dataset = QLabel("—")
        self.status_scale = QLabel("—")
        self.status_run = QLabel("就绪")
        self.status_theme = QLabel("暗色")
        bar.addWidget(self.status_dataset)
        bar.addWidget(_sep())
        bar.addWidget(self.status_scale)
        bar.addPermanentWidget(self.status_run)
        bar.addPermanentWidget(_sep())
        bar.addPermanentWidget(self.status_theme)

    # ==================================================================
    # 辅助
    # ==================================================================
    def _action(self, menu, text: str, shortcut, slot) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _tool(self, toolbar: QToolBar, text: str, slot, tip: str = "") -> QAction:
        action = QAction(text, self)
        action.setToolTip(tip or text)
        action.triggered.connect(slot)
        toolbar.addAction(action)
        return action

    def _page(self, class_name: str):
        index = self._page_index.get(class_name)
        return self.pages[index] if index is not None else None

    def _goto(self, class_name: str) -> None:
        index = self._page_index.get(class_name)
        if index is not None:
            self.nav.setCurrentRow(index)

    def _on_nav_changed(self, row: int) -> None:
        self.stack.setCurrentIndex(max(row, 0))
        page = self.pages[row] if 0 <= row < len(self.pages) else None
        if page is not None:
            self.statusBar().showMessage(
                f"第 {row + 1} 页：{getattr(page, 'title', '')}", 3000
            )

    # ==================================================================
    # 配置操作
    # ==================================================================
    def _on_config_changed(self) -> None:
        cfg = self.store.config
        estimate = cfg.estimate()
        w, h = cfg.resolved_image_size()
        self.status_dataset.setText(f"数据集 {cfg.dataset.name}")
        self.status_scale.setText(
            f"{estimate['masters']} 主指纹 / {estimate['samples']} 样本 / {w}×{h}"
        )
        title = f"{APP_TITLE} — {cfg.dataset.name}"
        if self.store.dirty:
            title += " *"
        if self.store.path:
            title += f"  [{self.store.path.name}]"
        self.setWindowTitle(title)

    def _on_config_error(self, message: str) -> None:
        text = message if len(message) < 600 else message[:600] + "…"
        self.statusBar().showMessage("配置校验失败", 6000)
        QMessageBox.warning(self, "配置校验失败", text)

    def _on_run_status(self, message: str) -> None:
        self.status_run.setText(message)

    def _new_config(self) -> None:
        if not self._confirm_discard():
            return
        self.store.replace(build_config("default"), None)
        self.statusBar().showMessage("已新建默认配置", 4000)

    def _open_config(self) -> None:
        if not self._confirm_discard():
            return
        start = str(self.store.path.parent if self.store.path else Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "打开配置", start, "YAML (*.yaml *.yml)")
        if path:
            self.store.load(Path(path))

    def _save_config(self) -> None:
        if self.store.path is None:
            self._save_config_as()
            return
        if self.store.save():
            self.statusBar().showMessage(f"已保存 {self.store.path}", 5000)
        self._on_config_changed()

    def _save_config_as(self) -> None:
        default = str(self.store.path or Path(f"{self.store.config.dataset.name}.yaml"))
        path, _ = QFileDialog.getSaveFileName(self, "保存配置", default, "YAML (*.yaml *.yml)")
        if not path:
            return
        if self.store.save(Path(path)):
            self.statusBar().showMessage(f"已保存 {path}", 5000)
        self._on_config_changed()

    def _apply_preset(self, name: str) -> None:
        if not self._confirm_discard():
            return
        if self.store.load_preset(name):
            self.statusBar().showMessage(f"已应用预设：{PRESETS[name].label}", 5000)

    def _on_preset_combo(self, index: int) -> None:
        name = self.preset_combo.itemData(index)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        if name:
            self._apply_preset(name)

    def _validate_config(self) -> None:
        cfg = self.store.config
        lines: List[str] = ["✓ 配置校验通过", ""]
        lines.extend(cfg.summary_lines())
        lines.append("")
        lines.append(f"规模估算：{cfg.estimate()}")
        QMessageBox.information(self, "配置校验", "\n".join(lines))

    def _copy_yaml(self) -> None:
        QApplication.clipboard().setText(self.store.to_yaml())
        self.statusBar().showMessage("完整配置 YAML 已复制到剪贴板", 5000)

    def _show_config(self) -> None:
        from .qt import QDialog, QPlainTextEdit, QPushButton

        dialog = QDialog(self)
        dialog.setWindowTitle(f"完整配置 — {self.store.config.dataset.name}")
        dialog.resize(760, 640)
        layout = QVBoxLayout(dialog)
        editor = QPlainTextEdit(self.store.to_yaml())
        font = editor.font()
        font.setFamilies(["Cascadia Mono", "Consolas", "monospace"])
        editor.setFont(font)
        layout.addWidget(editor)

        buttons = QHBoxLayout()
        apply_btn = QPushButton("应用修改")
        apply_btn.setObjectName("Primary")

        def apply_text() -> None:
            if self.store.load_yaml_text(editor.toPlainText()):
                dialog.accept()
                self.statusBar().showMessage("已应用 YAML 修改", 5000)
            else:
                QMessageBox.warning(dialog, "应用失败", "YAML 校验失败，请看下方提示重试。")

        apply_btn.clicked.connect(apply_text)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.reject)
        buttons.addStretch(1)
        buttons.addWidget(apply_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        dialog.exec()

    def _confirm_discard(self) -> bool:
        if not self.store.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "未保存的修改",
            "当前配置有未保存的修改，继续操作会丢弃它们。\n\n确定继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    # ==================================================================
    # 生成操作
    # ==================================================================
    def _monitor(self) -> Optional[MonitorPage]:
        return self._page("MonitorPage")  # type: ignore[return-value]

    def _start_run(self) -> None:
        monitor = self._monitor()
        if monitor is None:
            return
        self._goto("MonitorPage")
        monitor._on_start()

    def _toggle_pause(self) -> None:
        monitor = self._monitor()
        if monitor is not None:
            monitor._on_pause()

    def _stop_run(self) -> None:
        monitor = self._monitor()
        if monitor is not None:
            monitor._on_stop()

    def _preview_plan(self) -> None:
        from ..app import GenerationService

        try:
            summary = GenerationService(self.store.config, export_after_run=False).dry_run()
        except ConfigError as exc:
            QMessageBox.warning(self, "计划不可用", str(exc))
            return
        classes = "\n".join(f"  {k}: {v}" for k, v in sorted(summary["classes"].items()))
        splits = "\n".join(f"  {k}: {v}" for k, v in sorted(summary["splits"].items()))
        QMessageBox.information(
            self,
            "生成计划预览",
            f"数据集：{summary['dataset']}　seed={summary['seed']}\n\n"
            f"主指纹 {summary['masters']}　捺印 {summary['impressions']}　"
            f"样本 {summary['samples']}\n\n"
            f"类别分布：\n{classes}\n\n划分：\n{splits or '  （未启用）'}\n\n"
            f"配置指纹：{summary['config_fingerprint']}",
        )

    def _open_dataset_dir(self) -> None:
        from .qt import QDesktopServices, QUrl

        path = Path(self.store.config.dataset_dir)
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    # ==================================================================
    def _set_theme(self, name: str) -> None:
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, name)
        self.status_theme.setText("暗色" if name == "dark" else "亮色")
        self.statusBar().showMessage(f"已切换到{'暗色' if name == 'dark' else '亮色'}主题", 3000)

    def _self_check(self) -> None:
        from types import SimpleNamespace

        from ..cli import _cmd_selfcheck

        self.statusBar().showMessage("自检中…", 2000)
        QApplication.processEvents()
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                code = _cmd_selfcheck(SimpleNamespace())
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "自检失败", f"{type(exc).__name__}: {exc}")
            return
        text = buffer.getvalue()
        box = QMessageBox(self)
        box.setWindowTitle("自检结果")
        box.setIcon(QMessageBox.Icon.Information if code == 0 else QMessageBox.Icon.Warning)
        box.setText("全部通过" if code == 0 else "存在未通过项")
        box.setDetailedText(text)
        box.setInformativeText(text.splitlines()[-1] if text else "")
        box.exec()

    def _about(self) -> None:
        QMessageBox.about(
            self,
            f"关于 {APP_SUBTITLE}",
            f"<h3>{APP_TITLE} v{__version__}</h3>"
            "<p>轻量级 SFinGe 合成指纹数据集合成软件（GUI 优先）。</p>"
            "<p><b>生成链</b>：手指轮廓 → 奇异点 → 零极点方向场 → Perlin 频率图 "
            "→ Gabor 迭代脊线 → 捺印（退化链 / 按压变形 / 传感器）→ TTA "
            "→ 质量评估 → 细节点提取 → 标准化存储。</p>"
            "<p><b>核心依赖只有 4 个</b>：numpy、opencv-python、pydantic、PyYAML"
            "（GUI 另需 PySide6 或 PyQt6）。</p>"
            "<p style='color:#ffb020'><b>合成数据声明</b>：本软件生成的图像"
            "完全由算法合成，不含任何真实人类指纹数据，不涉及个人生物特征信息。"
            "可用于算法研究、模型预训练与教学；不得用于冒充真实身份或规避生物识别系统。</p>"
            "<p>算法出处：Cappelli et al., <i>Synthetic Fingerprint-Image Generation</i>, "
            "ICPR 2000。</p>",
        )

    # ==================================================================
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt 约定
        monitor = self._monitor()
        if monitor is not None and monitor.controller.is_running:
            answer = QMessageBox.question(
                self,
                "生成进行中",
                "生成任务仍在运行。\n\n关闭窗口会请求停止并等待当前样本收尾。\n"
                "已完成的样本会完整保留，可用同样配置断点续传。\n\n确定关闭吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            monitor.controller.stop()
            monitor.controller.wait(8000)
        if not self._confirm_discard():
            event.ignore()
            return
        event.accept()
