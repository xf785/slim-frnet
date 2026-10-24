"""页面 1：数据集配置（设计文档 F1 / F2 / F12）。"""

from __future__ import annotations

from typing import Dict

from ...config.schema import CLASS_LABELS_ZH, FINGERPRINT_CLASSES
from ..qt import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)
from ..widgets import SectionTitle
from .base import BasePage

__all__ = ["DatasetPage"]


class DatasetPage(BasePage):
    """身份数 / 手指数 / 捺印数 / TTA 数 + 类别比例 + 划分比例。"""

    title = "数据集配置"
    icon = "📋"

    def build(self) -> None:
        # ---------------- 基本 ----------------
        box, form = self.group("数据集")
        self.name_edit = QLineEdit()
        self.bind_line("dataset.name", self.name_edit)
        form.addRow("名称", self.name_edit)

        root_row = QWidget()
        root_layout = QHBoxLayout(root_row)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(6)
        self.root_edit = QLineEdit()
        self.bind_line("dataset.output_root", self.root_edit)
        browse = QPushButton("浏览…")
        browse.setToolTip("选择输出根目录")
        browse.clicked.connect(self._browse_root)
        root_layout.addWidget(self.root_edit, 1)
        root_layout.addWidget(browse)
        form.addRow("输出根目录", root_row)

        self.identities = self.bind_spin(
            "dataset.identities", _spin(), on_change=self._on_scale, minimum=1, maximum=100000
        )
        form.addRow("身份数", self.identities)

        self.fingers = self.bind_spin(
            "dataset.fingers_per_identity", _spin(), on_change=self._on_scale, minimum=1, maximum=10
        )
        form.addRow("每身份手指数", self.fingers)

        self.impressions = self.bind_spin(
            "dataset.impressions_per_finger", _spin(), on_change=self._on_scale,
            minimum=1, maximum=50,
        )
        form.addRow("每手指捺印数", self.impressions)

        self.tta_count = self.bind_spin(
            "dataset.tta_per_impression", _spin(), on_change=self._on_scale,
            minimum=0, maximum=500,
        )
        form.addRow("每捺印 TTA 数", self.tta_count)

        hint = QLabel(
            "说明：<b>tta_index = 0</b> 的样本始终保留<b>未变换的原始捺印</b>作为干净对照，"
            "其余序号依次施加随机 TTA 算子链。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        form.addRow("", hint)

        self.add_estimate_card()

        # ---------------- 类别比例 ----------------
        section = SectionTitle("指纹类别比例（Henry 分类）")
        self.add_widget(section)
        self.class_table = QTableWidget(0, 3)
        self.class_table.setHorizontalHeaderLabels(["类别", "比例", "预计样本数"])
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.verticalHeader().setDefaultSectionSize(34)
        self.class_table.setMinimumHeight(200)
        # 5 行固定内容，不需要滚动条；滚动条会压住右列的输入框箭头
        self.class_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.class_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self.class_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        # 数值列用固定宽度：ResizeToContents 算不准带箭头的 QDoubleSpinBox，
        # 列宽会小于控件本身，箭头会压到相邻单元格上。
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.class_table.setColumnWidth(1, 130)
        self.class_table.setColumnWidth(2, 120)
        self.add_widget(self.class_table)

        row = QHBoxLayout()
        self.sum_label = QLabel()
        self.sum_label.setObjectName("Hint")
        row.addWidget(self.sum_label, 1)
        for text, slot, tip in (
            ("归一化", self._normalize_classes, "按当前比例等比缩放，使总和 = 100%"),
            ("自然分布", lambda: self._apply_class_preset("natural"), "Henry 自然分布 5/35/35/20/5"),
            ("均衡分布", lambda: self._apply_class_preset("balanced"), "五类各 20%"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)
        self.add_widget(_wrap(row))

        # ---------------- 划分 ----------------
        box2, form2 = self.group("数据集划分（按身份，防止身份泄漏）")
        self.split_enabled = self.bind_check("split.enabled", _check(), on_change=self._on_scale)
        form2.addRow("启用划分", self.split_enabled)
        self.split_combo = self.bind_combo(
            "split.strategy",
            QComboBox(),
            values=["by_identity", "by_finger", "random"],
            labels=["按身份（推荐，无身份泄漏）", "按手指", "随机"],
            on_change=self._on_scale,
        )
        form2.addRow("划分策略", self.split_combo)
        self.train_spin = self._split_row(form2, "train", "训练集")
        self.val_spin = self._split_row(form2, "val", "验证集")
        self.test_spin = self._split_row(form2, "test", "测试集")
        self.split_sum = QLabel()
        self.split_sum.setObjectName("Hint")
        form2.addRow("", self.split_sum)

    # -- 内部 -------------------------------------------------------------
    def _split_row(self, form, key: str, label: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        self.bind_double(f"split.ratios.{key}", spin, on_change=self._on_split)
        form.addRow(label, spin)
        return spin

    def _browse_root(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出根目录", self.root_edit.text())
        if directory:
            self.root_edit.setText(directory)
            self._push("dataset.output_root", directory)

    def _on_scale(self, *_args) -> None:
        self.update_estimate()
        self._refresh_class_counts()

    def _restyle(self, label: QLabel, object_name: str, text: str) -> None:
        """改 ``objectName`` 后必须重新 polish，QSS 才会生效。"""
        label.setObjectName(object_name)
        label.setText(text)
        label.style().unpolish(label)
        label.style().polish(label)

    def _on_split(self, *_args) -> None:
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        if abs(total - 1.0) > 1e-6:
            self._restyle(
                self.split_sum,
                "Warn",
                f"⚠ 三集合比例之和为 {total:.2f}，将自动归一化后使用（建议调到 1.00）",
            )
        else:
            self._restyle(self.split_sum, "Ok", f"✓ 比例之和 = {total:.2f}")

    def _normalize_classes(self) -> None:
        weights = {key: item[1].value() for key, item in self._class_rows().items()}
        total = sum(weights.values())
        if total <= 0:
            QMessageBox.warning(self, "无法归一化", "所有类别比例都是 0，请先填入比例。")
            return
        new = {key: round(value / total, 4) for key, value in weights.items() if value > 0}
        diff = 1.0 - sum(new.values())
        if new:
            first = max(new, key=lambda k: new[k])
            new[first] = round(new[first] + diff, 4)
        self._apply(new)

    def _apply_class_preset(self, kind: str) -> None:
        if kind == "natural":
            self._apply({"arch": 0.05, "left_loop": 0.35, "right_loop": 0.35,
                         "whorl": 0.20, "twin_loop": 0.05})
        else:
            self._apply({name: 0.2 for name in FINGERPRINT_CLASSES})

    def _apply(self, distribution: Dict[str, float]) -> None:
        if not self.store.update_flat({"classes.distribution": distribution}):
            QMessageBox.warning(
                self,
                "比例无效",
                "类别比例校验失败：\n\n"
                f"{sorted(distribution.items())}\n\n"
                "要求：所有类别比例之和为 1.0，且不含未知类别。",
            )

    def _class_rows(self) -> Dict[str, tuple]:
        rows: Dict[str, tuple] = {}
        for row in range(self.class_table.rowCount()):
            item = self.class_table.item(row, 0)
            spin = self.class_table.cellWidget(row, 1)
            count = self.class_table.item(row, 2)
            if item is not None and spin is not None and count is not None:
                rows[item.text()] = (item, spin, count)
        return rows

    def _refresh_class_counts(self) -> None:
        est = self.store.config.estimate()
        total_samples = max(est["samples"], 1)
        for _key, (_item, spin, count) in self._class_rows().items():
            count.setText(f"{int(round(spin.value() * total_samples))}")

    def _refresh_class_sum(self) -> None:
        weights = [spin.value() for _i, spin, _c in self._class_rows().values()]
        total = sum(weights)
        if abs(total - 1.0) <= 1e-6:
            self._restyle(self.sum_label, "Ok", f"✓ 比例之和 = {total:.4f}")
        else:
            self._restyle(
                self.sum_label, "Warn",
                f"⚠ 比例之和 = {total:.4f}（必须为 1.0000 才能保存）",
            )

    def _rebuild_class_table(self) -> None:
        distribution = self.store.config.classes.distribution
        self.class_table.setRowCount(0)
        for name in FINGERPRINT_CLASSES:
            row = self.class_table.rowCount()
            self.class_table.insertRow(row)

            item = QTableWidgetItem(CLASS_LABELS_ZH.get(name, name))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.class_table.setItem(row, 0, item)

            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(4)
            spin.setValue(float(distribution.get(name, 0.0)))
            spin.valueChanged.connect(
                lambda _v, n=name: self._on_class_changed(n)
            )
            self.class_table.setCellWidget(row, 1, spin)

            count = QTableWidgetItem("0")
            count.setFlags(count.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.class_table.setItem(row, 2, count)

    def _on_class_changed(self, _name: str) -> None:
        if self._updating:
            return
        self._refresh_class_sum()
        self._refresh_class_counts()

    # -- 刷新 -------------------------------------------------------------
    def on_refresh(self) -> None:
        self._rebuild_class_table()
        self._refresh_class_sum()
        self.update_estimate()
        self._on_split()
        self._refresh_class_counts()


def _spin():
    from ..qt import QSpinBox

    return QSpinBox()


def _check():
    from ..qt import QCheckBox

    return QCheckBox()


def _wrap(layout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget
