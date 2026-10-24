"""页面 8：导出（设计文档 F10 / 4.8）。

导出的每一项都可以单独执行（幂等，不会动图像）：

* README.md（合成声明 + 字段释义 + 复现方法）
* stats.json（统计报告）
* splits/{train,val,test}.csv（划分清单）
* checksums/sha256sums.txt（全量校验和）
* environment.json 补齐
* 可选：打包成 zip

此外提供「导出清单」表格，让用户看到数据集里到底有哪些文件、多大体积。
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict

from ...app.exporter import export_dataset
from ...core.manifest import read_jsonl
from ...core.storage import DatasetLayout
from ..qt import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)
from ..widgets import Card, SectionTitle
from .base import BasePage

__all__ = ["ExportPage"]


class ExportPage(BasePage):
    """导出与打包。"""

    title = "导出"
    icon = "📦"

    def build(self) -> None:
        # ---------------- 目录 ----------------
        bar = QHBoxLayout()
        self.dir_edit = QLineEdit(str(self.store.config.dataset_dir))
        bar.addWidget(QLabel("数据集"))
        bar.addWidget(self.dir_edit, 1)
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        current = QPushButton("当前输出")
        current.clicked.connect(
            lambda: (self.dir_edit.setText(str(self.store.config.dataset_dir)), self.refresh_view())
        )
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh_view)
        bar.addWidget(current)
        bar.addWidget(browse)
        bar.addWidget(refresh)
        self.add_widget(_wrap(bar))

        self.status_label = QLabel("尚未检查数据集")
        self.status_label.setObjectName("Hint")
        self.status_label.setWordWrap(True)
        self.add_widget(self.status_label)

        # ---------------- 导出项 ----------------
        options = Card("导出项（全部幂等，不修改任何图像）")
        rows = [
            ("write_readme", "生成 README.md（合成数据声明 + 字段释义 + 复现方法）"),
            ("write_stats", "生成 stats.json（综合统计报告）"),
            ("write_checksums", "生成 checksums/sha256sums.txt（全量文件校验和）"),
            ("write_splits", "生成 splits/{train,val,test}.csv（划分清单）"),
            ("write_environment", "补齐 environment.json（环境 + 导出口径）"),
        ]
        self.option_boxes: Dict[str, QCheckBox] = {}
        for key, label in rows:
            check = QCheckBox(label)
            check.setChecked(True)
            options.add(check)
            self.option_boxes[key] = check

        pack_row = QHBoxLayout()
        self.pack_check = QCheckBox("额外打包为 ZIP")
        self.pack_check.setToolTip("使用标准库 zipfile，不引入额外依赖")
        pack_row.addWidget(self.pack_check)
        self.pack_path = QLineEdit()
        self.pack_path.setPlaceholderText("ZIP 输出路径（留空则放在数据集同级目录）")
        pack_row.addWidget(self.pack_path, 1)
        options.add_layout(pack_row)

        action_row = QHBoxLayout()
        self.export_btn = QPushButton("执行导出")
        self.export_btn.setObjectName("Primary")
        self.export_btn.clicked.connect(self._on_export)
        action_row.addWidget(self.export_btn)
        self.open_btn = QPushButton("打开数据集目录")
        self.open_btn.clicked.connect(self._open_dir)
        action_row.addWidget(self.open_btn)
        action_row.addStretch(1)
        options.add_layout(action_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        options.add(self.progress)
        self.add_widget(options)

        # ---------------- 数据集结构 ----------------
        self.add_widget(SectionTitle("数据集结构"))
        self.tree_table = QTableWidget(0, 3)
        self.tree_table.setHorizontalHeaderLabels(["条目", "文件数", "体积"])
        self.tree_table.verticalHeader().setVisible(False)
        self.tree_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.tree_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.tree_table.setMinimumHeight(220)
        self.add_widget(self.tree_table)

        # ---------------- 复现信息 ----------------
        reproduce = Card("复现信息")
        self.reproduce_label = QLabel("—")
        self.reproduce_label.setObjectName("Hint")
        self.reproduce_label.setWordWrap(True)
        # 4 行中文 + 折行，不给最小高度会被卡片边框裁掉最后一行
        self.reproduce_label.setMinimumHeight(92)
        self.reproduce_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.reproduce_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        reproduce.add(self.reproduce_label)
        copy_btn = QPushButton("复制复现命令")
        copy_btn.clicked.connect(self._copy_command)
        reproduce.add(copy_btn)
        self.add_widget(reproduce)

    # ------------------------------------------------------------------
    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择数据集目录", self.dir_edit.text())
        if directory:
            self.dir_edit.setText(directory)
            self.refresh_view()

    def _dataset_layout(self) -> DatasetLayout:
        """解析输入框里的目录，返回数据集布局。

        注意方法名不能叫 ``_layout``：``BasePage`` 用 ``self._layout`` 保存页面
        的 QVBoxLayout，同名方法会被那个属性覆盖，调用时直接
        ``TypeError: 'QVBoxLayout' object is not callable``。
        """
        root = Path(self.dir_edit.text().strip())
        if not (root / "dataset_manifest.jsonl").exists():
            for parent in [root, *root.parents]:
                if (parent / "dataset_manifest.jsonl").exists():
                    self.dir_edit.setText(str(parent))
                    return DatasetLayout(parent)
        return DatasetLayout(root)

    def refresh_view(self) -> None:
        layout = self._dataset_layout()
        if not layout.manifest.exists():
            self.status_label.setText(
                "✗ 该目录不是 synthfp 数据集（缺少 dataset_manifest.jsonl）。"
                "请先在「生成监控」页完成一次生成。"
            )
            self.status_label.setObjectName("Err")
            self.tree_table.setRowCount(0)
            return

        rows = [r for r in read_jsonl(layout.manifest) if r.get("status") in (None, "ok")]
        total_bytes = sum(int(r.get("bytes", 0) or 0) for r in rows)
        failed = _count_lines(layout.failed)
        self.status_label.setObjectName("Hint")
        self.status_label.setText(
            f"✓ <b>{layout.root.name}</b>　样本 {len(rows)}　图像合计 {total_bytes / 1048576:.1f} MB　"
            f"失败记录 {failed} 条<br>"
            f"清单 {layout.manifest.name}　"
            f"配置 {'已写入' if layout.config_resolved.exists() else '缺失'}　"
            f"校验和 {'已生成' if layout.checksums.exists() else '未生成'}"
        )

        self.tree_table.setRowCount(0)
        for name, count, size in layout.summary_tree():
            row = self.tree_table.rowCount()
            self.tree_table.insertRow(row)
            self.tree_table.setItem(row, 0, QTableWidgetItem(name))
            self.tree_table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.tree_table.setItem(row, 2, QTableWidgetItem(_human(size)))

        command = f"synthfp generate --config {layout.root.name}.yaml"
        fingerprint = ""
        if layout.environment.exists():
            try:
                fingerprint = json.loads(layout.environment.read_text(encoding="utf-8")).get(
                    "config_fingerprint", ""
                )
            except Exception:  # pragma: no cover
                pass
        self.reproduce_label.setText(
            f"复现四要素：<b>配置</b>（{layout.config_resolved.name}）　"
            f"<b>种子</b>（manifest 每行的 seed 字段）　"
            f"<b>版本</b>（environment.json）　<b>校验和</b>（checksums/sha256sums.txt）<br>"
            f"配置指纹：<code>{fingerprint or '未记录'}</code><br>"
            f"复现命令：<code>{command}</code><br>"
            "把 config.resolved.yaml 复制为同名文件即可用上述命令重建同源数据集。"
        )
        self._command = command

    # ------------------------------------------------------------------
    def _on_export(self) -> None:
        layout = self._dataset_layout()
        if not layout.manifest.exists():
            QMessageBox.warning(self, "无法导出", "该目录不是 synthfp 数据集。")
            return

        cfg = self.store.config.model_copy(deep=True)
        cfg.export.write_readme = self.option_boxes["write_readme"].isChecked()
        cfg.export.write_stats = self.option_boxes["write_stats"].isChecked()
        cfg.export.write_sha256 = self.option_boxes["write_checksums"].isChecked()
        cfg.split.write_csv = self.option_boxes["write_splits"].isChecked()
        cfg.export.write_environment = self.option_boxes["write_environment"].isChecked()

        self.export_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(10)
        try:
            summary = export_dataset(cfg, layout)
            self.progress.setValue(70)
            message = [
                f"样本 {summary.get('samples', 0)}",
                f"产物 {summary.get('artifacts', 0)}",
            ]
            if "checksums" in summary:
                message.append(f"校验和 {summary['checksums']} 条")
            if "splits" in summary:
                message.append(
                    "划分 " + "/".join(f"{k}={v}" for k, v in sorted(summary["splits"].items()))
                )
            if self.pack_check.isChecked():
                zip_path = self._make_zip(layout)
                message.append(f"ZIP → {zip_path}")
                self.progress.setValue(95)
            self.progress.setValue(100)
            self.status_label.setText("✓ 导出完成：" + "　".join(message))
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"{type(exc).__name__}: {exc}")
            self.status_label.setText(f"✗ 导出失败：{exc}")
        finally:
            self.export_btn.setEnabled(True)
            self.refresh_view()

    def _make_zip(self, layout: DatasetLayout) -> Path:
        target = self.pack_path.text().strip()
        zip_path = Path(target) if target else layout.root.parent / f"{layout.root.name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(layout.root.rglob("*")):
                if path.is_file() and path.suffix not in (".tmp",):
                    archive.write(path, path.relative_to(layout.root.parent))
        return zip_path

    # ------------------------------------------------------------------
    def _open_dir(self) -> None:
        from ..qt import QDesktopServices, QUrl

        layout = self._dataset_layout()
        layout.root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(layout.root.resolve())))

    def _copy_command(self) -> None:
        from ..qt import QApplication

        QApplication.clipboard().setText(getattr(self, "_command", "synthfp generate"))
        self.status_label.setText("✓ 复现命令已复制到剪贴板")

    # ------------------------------------------------------------------
    def on_refresh(self) -> None:
        if not self.dir_edit.text().strip():
            self.dir_edit.setText(str(self.store.config.dataset_dir))


def _wrap(layout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
