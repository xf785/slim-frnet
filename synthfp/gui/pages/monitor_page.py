"""页面 5：生成监控（设计文档 F5 / F6 / 3.2）。

进度、实时统计、任务表（含失败重试）、彩色日志，全部由一个
:class:`~synthfp.gui.qt_bridge.GenerationController` 驱动。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

from ...core.types import RunStats, SampleResult
from ..config_store import ConfigStore
from ..qt import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    Signal,
)
from ..qt_bridge import GenerationController
from ..widgets import Card, LogPanel, StatCard
from .base import BasePage

__all__ = ["MonitorPage"]

_MAX_ROWS = 600


class MonitorPage(BasePage):
    """实时进度 + 任务表 + 日志。"""

    title = "生成监控"
    icon = "🚀"
    #: 内部有任务表与日志区，自己管理空间，不需要外层滚动区
    scrollable = False

    run_finished = Signal(object)  # RunStats
    status_message = Signal(str)  # 状态栏文本

    def __init__(self, store: ConfigStore, parent: Optional[QWidget] = None) -> None:
        # 控制器必须在 build() 之前就绪（build 会连接它的信号）
        self.controller = GenerationController(store.config)
        super().__init__(store, parent)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """生成中不刷新（配置在运行期不可变），结束后再同步一次。"""
        if self.controller.is_running:
            return
        self.controller.set_config(self.store.config)
        self.on_refresh()

    # ------------------------------------------------------------------
    def build(self) -> None:
        self._rows: Dict[str, int] = {}
        self._t0 = 0.0

        # ---------------- 控制条 ----------------
        card = Card("运行控制")
        controls = QHBoxLayout()
        self.start_btn = QPushButton("▶  开始生成")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self._on_start)
        self.pause_btn = QPushButton("⏸  暂停")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause)
        self.stop_btn = QPushButton("■  停止")
        self.stop_btn.setObjectName("Danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)

        self.export_check = QCheckBox("完成后自动导出")
        self.export_check.setChecked(True)
        self.export_check.setToolTip("生成结束后自动写 README / 统计 / 划分 / 校验和")

        self.open_btn = QPushButton("打开输出目录")
        self.open_btn.clicked.connect(self._open_output)

        for widget in (self.start_btn, self.pause_btn, self.stop_btn):
            controls.addWidget(widget)
        controls.addSpacing(12)
        controls.addWidget(self.export_check)
        controls.addStretch(1)
        controls.addWidget(self.open_btn)
        card.add_layout(controls)

        self.plan_label = QLabel("尚未生成")
        self.plan_label.setObjectName("Hint")
        self.plan_label.setWordWrap(True)
        card.add(self.plan_label)
        self.add_widget(card)

        # ---------------- 进度 ----------------
        progress_card = Card("进度")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%  (%v/%m)")
        progress_card.add(self.progress)

        stats_row = QHBoxLayout()
        self.card_total = StatCard("总样本", "0")
        self.card_ok = StatCard("成功", "0")
        self.card_rejected = StatCard("超阈值丢弃", "0")
        self.card_failed = StatCard("失败", "0")
        self.card_skipped = StatCard("断点跳过", "0")
        self.card_rate = StatCard("吞吐", "0.00", "样本/秒")
        self.card_eta = StatCard("预计剩余", "—")
        for card_widget in (self.card_total, self.card_ok, self.card_rejected,
                            self.card_failed, self.card_skipped, self.card_rate, self.card_eta):
            stats_row.addWidget(card_widget)
        progress_card.add_layout(stats_row)
        self.add_widget(progress_card)

        # ---------------- 任务表 / 日志 ----------------
        tabs = QTabWidget()

        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        table_layout.setContentsMargins(0, 6, 0, 0)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["样本 ID", "身份", "手指", "类别", "捺印", "TTA 算子", "状态", "质量分", "耗时"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 9):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        table_layout.addWidget(self.table)

        row = QHBoxLayout()
        self.failed_only = QCheckBox("只看失败")
        self.failed_only.toggled.connect(self._apply_filter)
        row.addWidget(self.failed_only)
        row.addStretch(1)
        clear = QPushButton("清空表格")
        clear.clicked.connect(self._clear_table)
        row.addWidget(clear)
        table_layout.addLayout(row)
        tabs.addTab(table_page, "任务列表")

        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        log_layout.setContentsMargins(0, 6, 0, 0)
        self.log_panel = LogPanel()
        log_layout.addWidget(self.log_panel)
        log_row = QHBoxLayout()
        self.autoscroll = QCheckBox("自动滚动")
        self.autoscroll.setChecked(True)
        log_row.addWidget(self.autoscroll)
        log_row.addStretch(1)
        save_log = QPushButton("导出日志")
        save_log.clicked.connect(self._save_log)
        clear_log = QPushButton("清空日志")
        clear_log.clicked.connect(self.log_panel.clear)
        log_row.addWidget(save_log)
        log_row.addWidget(clear_log)
        log_layout.addLayout(log_row)
        tabs.addTab(log_page, "运行日志")

        self.add_widget(tabs)

        # ---------------- 控制器信号 ----------------
        self.controller.log.connect(self._on_log)
        self.controller.phase.connect(lambda name: self._on_log("INFO", f"── 阶段：{name} ──"))
        self.controller.planned.connect(self._on_planned)
        self.controller.progress.connect(self._on_progress)
        self.controller.sample_done.connect(self._on_sample)
        self.controller.finished.connect(self._on_finished)
        self.controller.running_changed.connect(self._on_running_changed)
        self.controller.paused_changed.connect(self._on_paused_changed)

        self.store.changed.connect(self._sync_config)

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------
    def _sync_config(self) -> None:
        if not self.controller.is_running:
            self.controller.set_config(self.store.config)

    def _on_start(self) -> None:
        self._clear_table()
        self.log_panel.clear()
        self._rows.clear()
        self._t0 = time.perf_counter()
        self._sync_config()
        try:
            started = self.controller.start(export_after_run=self.export_check.isChecked())
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "无法启动", f"{type(exc).__name__}: {exc}")
            return
        if not started:
            return
        cfg = self.store.config
        self._on_log("INFO", f"数据集：{cfg.dataset.name}　{cfg.dataset.summary()}")
        self._on_log("INFO", f"输出目录：{cfg.dataset_dir}")
        self._on_log("INFO", f"配置指纹：{self.controller.service.fingerprint if self.controller.service else '?'}")

    def _on_pause(self) -> None:
        self.controller.toggle_pause()

    def _on_stop(self) -> None:
        answer = QMessageBox.question(
            self,
            "确认停止",
            "停止后当前正在处理的样本会收尾，已完成的样本完整保留，"
            "下次用同样配置可断点续传。\n\n确定停止吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.controller.stop()

    def _on_running_changed(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.pause_btn.setEnabled(running)
        if not running:
            self.pause_btn.setText("⏸  暂停")

    def _on_paused_changed(self, paused: bool) -> None:
        self.pause_btn.setText("▶  恢复" if paused else "⏸  暂停")

    def _on_planned(self, summary: dict) -> None:
        if not summary:
            self.plan_label.setText("尚未生成")
            return
        classes = summary.get("classes", {})
        class_text = "　".join(f"{k}={v}" for k, v in sorted(classes.items()))
        splits = summary.get("splits", {})
        split_text = "　".join(f"{k}={v}" for k, v in sorted(splits.items()))
        self.plan_label.setText(
            f"数据集 <b>{summary.get('dataset', '?')}</b>　"
            f"主指纹 {summary.get('masters', 0)}　捺印 {summary.get('impressions', 0)}　"
            f"样本 <b>{summary.get('samples', 0)}</b><br>"
            f"类别：{class_text}<br>划分：{split_text or '—'}"
        )
        self.progress.setMaximum(max(int(summary.get("samples", 1)), 1))

    def _on_log(self, level: str, message: str) -> None:
        self.log_panel.append(level, message)

    def _on_progress(self, event) -> None:
        stats: RunStats = event.stats
        total = max(stats.total, 1)
        self.progress.setMaximum(total)
        self.progress.setValue(stats.processed)
        self.progress.setFormat(f"%p%  ({stats.processed}/{stats.total})")
        self.card_total.set_value(str(stats.total))
        self.card_ok.set_value(str(stats.ok))
        self.card_rejected.set_value(str(stats.rejected))
        self.card_failed.set_value(str(stats.failed))
        self.card_skipped.set_value(str(stats.skipped))
        self.card_rate.set_value(f"{stats.rate:.2f}")
        eta = stats.eta_s
        self.card_eta.set_value(_format_duration(eta) if eta > 0 else "—")
        self.status_message.emit(
            f"已生成 {stats.processed}/{stats.total} | 失败 {stats.failed} | "
            f"耗时 {_format_duration(stats.elapsed_s)} | 吞吐 {stats.rate:.2f}/s"
        )

    def _on_sample(self, result: SampleResult) -> None:
        spec = result.spec
        row = self._rows.get(spec.sample_id)
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows[spec.sample_id] = row
            if row >= _MAX_ROWS:
                self._trim_table()
                row = self._rows[spec.sample_id]
            for col, text in enumerate(
                (
                    spec.sample_id,
                    spec.identity_id,
                    spec.finger,
                    spec.fp_class,
                    f"#{spec.impression_index}",
                    "、".join(result.tta_operators) or "（原始）",
                    "",
                    "",
                    "",
                )
            ):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(row, col, item)
        else:
            self._set(row, 5, "、".join(result.tta_operators) or "（原始）")

        status_text = {"ok": "✓ 成功", "rejected": "⊘ 丢弃", "failed": "✗ 失败"}.get(
            result.status, result.status
        )
        self._set(row, 6, status_text)
        self._set(row, 7, f"{result.quality.score:.1f}" if result.quality else "—")
        self._set(row, 8, f"{result.duration_s:.2f}s")

        color = {"ok": "#34c759", "rejected": "#ffb020", "failed": "#ff453a"}.get(
            result.status, "#dcdce2"
        )
        item = self.table.item(row, 6)
        if item is not None:
            from ..qt import QColor

            item.setForeground(QColor(color))
        if result.status == "failed" and result.error:
            item = self.table.item(row, 6)
            if item is not None:
                item.setToolTip(result.error)

        if self.autoscroll.isChecked():
            self.table.scrollToBottom()

    def _on_finished(self, stats: RunStats) -> None:
        wall = time.perf_counter() - self._t0
        level = "ERROR" if stats.failed else ("WARNING" if stats.status == "stopped" else "OK")
        self._on_log(
            level,
            f"运行结束（{stats.status}）：成功 {stats.ok}，丢弃 {stats.rejected}，"
            f"失败 {stats.failed}，跳过 {stats.skipped}，墙钟 {wall:.1f}s",
        )
        self.run_finished.emit(stats)
        if stats.ok and self.export_check.isChecked():
            self._on_log("INFO", f"数据集已导出到：{stats.output_dir}")

    # ------------------------------------------------------------------
    def _set(self, row: int, col: int, text: str) -> None:
        item = self.table.item(row, col)
        if item is None:
            item = QTableWidgetItem(text)
            self.table.setItem(row, col, item)
        else:
            item.setText(text)

    def _trim_table(self) -> None:
        """超过上限时删掉最早的一批行，保持 ``_rows`` 映射同步。"""
        self.table.removeRow(0)
        self._rows = {sid: idx - 1 for sid, idx in self._rows.items() if idx > 0}

    def _clear_table(self) -> None:
        self.table.setRowCount(0)
        self._rows.clear()

    def _apply_filter(self, failed_only: bool) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 6)
            text = item.text() if item else ""
            self.table.setRowHidden(row, failed_only and "失败" not in text)

    def _open_output(self) -> None:
        from ..qt import QDesktopServices, QUrl

        path = Path(self.store.config.dataset_dir)
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _save_log(self) -> None:
        default = str(Path(self.store.config.dataset_dir) / "logs" / "gui_session.log")
        target, _ = QFileDialog.getSaveFileName(self, "导出日志", default, "日志 (*.log *.txt)")
        if not target:
            return
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(self.log_panel.to_text(), encoding="utf-8")
        self._on_log("INFO", f"日志已导出：{target}")

    # ------------------------------------------------------------------
    def on_refresh(self) -> None:
        if not self.controller.is_running:
            self.controller.set_config(self.store.config)


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
