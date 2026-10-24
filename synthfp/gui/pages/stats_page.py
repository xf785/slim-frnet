"""页面 7：统计报告（设计文档 F10 / 3.2）。

从 ``dataset_manifest.jsonl`` 直接计算，几万样本也是秒级；
所有表格都能导出 CSV，报告本身复用 CLI 的 ``stats.json``。
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...app.stats import compute_dataset_stats, format_stats_text
from ..qt import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from ..widgets import Card, StatCard
from .base import BasePage

__all__ = ["StatsPage"]


class StatsPage(BasePage):
    """类别 / 质量 / 细节点 / TTA / 划分 / 吞吐 多视角统计。"""

    title = "统计报告"
    icon = "📊"

    def build(self) -> None:
        self._stats: Optional[Dict[str, Any]] = None

        bar = QHBoxLayout()
        self.dir_edit = QLineEdit(str(self.store.config.dataset_dir))
        self.dir_edit.setPlaceholderText("数据集目录（含 dataset_manifest.jsonl）")
        bar.addWidget(QLabel("数据集"))
        bar.addWidget(self.dir_edit, 1)
        for text, slot, tip in (
            ("当前输出", lambda: (self.dir_edit.setText(str(self.store.config.dataset_dir)),
                                  self.reload()), "统计配置里的输出目录"),
            ("浏览…", self._browse, "选择其它数据集目录"),
            ("重新统计", self.reload, "重新读取 manifest 并计算"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        self.add_widget(_wrap(bar))

        # 概要卡片
        cards = QHBoxLayout()
        self.card_samples = StatCard("样本总数", "0")
        self.card_identities = StatCard("身份数", "0")
        self.card_masters = StatCard("主指纹", "0")
        self.card_quality = StatCard("平均质量分", "—", "0-100")
        self.card_minutiae = StatCard("平均细节点", "—")
        self.card_size = StatCard("数据集体积", "—")
        self.card_rejected = StatCard("丢弃 / 失败", "0")
        for card in (self.card_samples, self.card_identities, self.card_masters,
                     self.card_quality, self.card_minutiae, self.card_size, self.card_rejected):
            cards.addWidget(card)
        self.add_widget(_wrap(cards))

        self.summary_label = QLabel("尚未统计")
        self.summary_label.setObjectName("Hint")
        self.summary_label.setWordWrap(True)
        self.add_widget(self.summary_label)

        # 明细标签页
        self.tabs = QTabWidget()
        self.tables: Dict[str, QTableWidget] = {}
        for key, title, headers in (
            ("classes", "类别分布", ["类别", "样本数", "占比"]),
            ("quality", "质量分布", ["区间 / 等级", "样本数", "占比"]),
            ("minutiae", "细节点分布", ["区间", "样本数", "占比"]),
            ("tta", "TTA 算子", ["算子", "使用次数", "占比"]),
            ("splits", "划分", ["集合", "样本数", "占比"]),
            ("fingers", "手指分布", ["手指", "样本数", "占比"]),
        ):
            table = QTableWidget(0, len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for col in range(1, len(headers)):
                table.horizontalHeader().setSectionResizeMode(
                    col, QHeaderView.ResizeMode.ResizeToContents
                )
            self.tables[key] = table
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(0, 6, 0, 0)
            layout.addWidget(table)
            row = QHBoxLayout()
            export = QPushButton("导出本表 CSV")
            export.clicked.connect(lambda _c=False, k=key: self._export_table(k))
            row.addStretch(1)
            row.addWidget(export)
            layout.addLayout(row)
            self.tabs.addTab(page, title)
        self.add_widget(self.tabs)

        # 文本报告
        report_card = Card("完整统计报告（与 stats.json 一致）")
        self.report_view = QLabel("—")
        self.report_view.setObjectName("Hint")
        self.report_view.setWordWrap(True)
        self.report_view.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        report_card.add(self.report_view)
        row = QHBoxLayout()
        row.addStretch(1)
        copy_btn = QPushButton("复制到剪贴板")
        copy_btn.clicked.connect(self._copy_report)
        save_btn = QPushButton("另存为 stats.json")
        save_btn.clicked.connect(self._save_stats)
        row.addWidget(copy_btn)
        row.addWidget(save_btn)
        report_card.add_layout(row)
        self.add_widget(report_card)

    # ------------------------------------------------------------------
    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择数据集目录", self.dir_edit.text())
        if directory:
            self.dir_edit.setText(directory)
            self.reload()

    def reload(self) -> None:
        from ...core.storage import DatasetLayout

        root = Path(self.dir_edit.text().strip())
        layout = DatasetLayout(root)
        if not layout.manifest.exists():
            for parent in [root, *root.parents]:
                if (parent / "dataset_manifest.jsonl").exists():
                    root = parent
                    layout = DatasetLayout(root)
                    self.dir_edit.setText(str(root))
                    break
        if not layout.manifest.exists():
            self.summary_label.setText("✗ 未找到 dataset_manifest.jsonl，请先生成数据集")
            self.summary_label.setObjectName("Err")
            self._stats = None
            return

        cfg = self._resolve_config(layout)
        self._stats = compute_dataset_stats(cfg, layout)
        self._render(self._stats)

    def _resolve_config(self, layout) -> Any:
        """优先用数据集自带的 config.resolved.yaml（保证按当年的配置解释）。"""
        from ...config import load_config
        from ...config.loader import ConfigError

        if layout.config_resolved.exists():
            try:
                return load_config(layout.config_resolved)
            except ConfigError:
                pass
        return self.store.config

    def _render(self, stats: Dict[str, Any]) -> None:
        totals = stats.get("totals", {})
        quality = stats.get("quality", {})
        minutiae = stats.get("minutiae", {})
        storage = stats.get("storage", {})

        self.card_samples.set_value(str(totals.get("ok", 0)))
        self.card_identities.set_value(str(totals.get("identities", 0)))
        self.card_masters.set_value(str(totals.get("masters", 0)))
        self.card_quality.set_value(
            f"{quality.get('mean', 0):.1f}" if quality.get("count") else "—"
        )
        self.card_minutiae.set_value(
            f"{minutiae.get('mean', 0):.0f}" if minutiae.get("count") else "—"
        )
        self.card_size.set_value(f"{storage.get('dataset_total_mb', 0):.1f} MB")
        self.card_rejected.set_value(
            f"{totals.get('rejected', 0)} / {totals.get('failed', 0)}"
        )

        self.summary_label.setObjectName("Hint")
        self.summary_label.setText(
            f"数据集 <b>{stats.get('dataset', '?')}</b>　"
            f"配置指纹 {stats.get('config_fingerprint') or '—'}　"
            f"seed {stats.get('seed', '?')}　"
            f"划分策略见下方「划分」页"
        )

        total = max(int(totals.get("ok", 0)), 1)
        self._fill("classes",
                   [(_label_of_class(k, v), v.get("count", 0)) for k, v in
                    stats.get("class_distribution", {}).items()], total)
        self._fill("quality",
                   [(k, v) for k, v in quality.get("histogram", {}).items()], total)
        self._fill("minutiae",
                   [(k, v) for k, v in minutiae.get("histogram", {}).items()], total)
        self._fill("tta",
                   sorted(stats.get("tta_operators", {}).items(), key=lambda kv: -kv[1]), total)
        self._fill("splits",
                   sorted(stats.get("split_distribution", {}).items()), total)
        self._fill("fingers",
                   sorted(stats.get("finger_distribution", {}).items(), key=lambda kv: -kv[1]),
                   total)

        levels = quality.get("levels", {})
        level_text = "　".join(f"L{k}={v}" for k, v in sorted(levels.items()))
        self.report_view.setText(
            format_stats_text(stats).replace("\n", "<br>")
            + (f"<br>质量等级：{level_text}" if level_text else "")
        )

    def _fill(self, key: str, rows: List[tuple], total: int) -> None:
        table = self.tables[key]
        table.setRowCount(0)
        for name, count in rows:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(str(name)))
            table.setItem(row, 1, QTableWidgetItem(str(count)))
            table.setItem(row, 2, QTableWidgetItem(f"{count / total:.2%}"))

    # ------------------------------------------------------------------
    def _export_table(self, key: str) -> None:
        table = self.tables[key]
        if table.rowCount() == 0:
            return
        default = str(Path(self.dir_edit.text().strip()) / f"stats_{key}.csv")
        target, _ = QFileDialog.getSaveFileName(self, "导出 CSV", default, "CSV (*.csv)")
        if not target:
            return
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([table.horizontalHeaderItem(c).text() for c in range(table.columnCount())])
        for row in range(table.rowCount()):
            writer.writerow([
                table.item(row, col).text() if table.item(row, col) else ""
                for col in range(table.columnCount())
            ])
        Path(target).write_text(buffer.getvalue(), encoding="utf-8-sig")
        self.summary_label.setText(f"✓ 已导出 {target}")

    def _copy_report(self) -> None:
        from ..qt import QApplication

        if self._stats:
            QApplication.clipboard().setText(format_stats_text(self._stats))
            self.summary_label.setText("✓ 统计报告已复制到剪贴板")

    def _save_stats(self) -> None:
        if not self._stats:
            return
        default = str(Path(self.dir_edit.text().strip()) / "stats.json")
        target, _ = QFileDialog.getSaveFileName(self, "另存统计报告", default, "JSON (*.json)")
        if not target:
            return
        Path(target).write_text(
            json.dumps(self._stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.summary_label.setText(f"✓ 已保存 {target}")

    # ------------------------------------------------------------------
    def on_refresh(self) -> None:
        if self._stats is None:
            self.dir_edit.setText(str(self.store.config.dataset_dir))


def _label_of_class(key: str, item: Dict[str, Any]) -> str:
    label = item.get("label") or key
    return f"{label}（{key}）" if label != key else key


def _wrap(layout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget
