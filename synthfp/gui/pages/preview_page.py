"""页面 6：结果预览（设计文档 3.2 / 3.4）。

提供两种使用方式：

* **数据集浏览**：扫描输出目录，按「身份 → 手指 → 捺印」树状浏览所有样本，
  点击即可看大图，并可叠加掩模 / 细节点 / 方向场 / 奇异点；
* **即时渲染**：不改动数据集，实时渲染单张样本与逐算子 TTA 对比，
  用于调参阶段快速看效果。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...app import PreviewService
from ...core.types import Minutia
from ..qt import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    Qt,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from ..qt_bridge import PreviewWorker
from ..widgets import ImageView, SectionTitle, ThumbnailGrid
from .base import BasePage

__all__ = ["PreviewPage"]


class PreviewPage(BasePage):
    """数据集浏览 + 即时渲染预览。"""

    title = "结果预览"
    icon = "🔍"
    #: 三栏浏览器内部已有树/缩略图/大图各自的滚动，不能再套外层滚动区
    scrollable = False

    def build(self) -> None:
        self._workers: List[PreviewWorker] = []
        tabs = QTabWidget()
        tabs.addTab(self._build_dataset_tab(), "数据集浏览")
        tabs.addTab(self._build_live_tab(), "即时渲染")
        self.add_widget(tabs)

    # ==================================================================
    # 数据集浏览
    # ==================================================================
    def _build_dataset_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        bar = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("数据集目录（含 dataset_manifest.jsonl）")
        self.dir_edit.setText(str(self.store.config.dataset_dir))
        self.dir_edit.editingFinished.connect(self._scan_dataset)
        bar.addWidget(QLabel("目录"))
        bar.addWidget(self.dir_edit, 1)
        for text, slot, tip in (
            ("当前输出", lambda: self._use_current_output(), "跳到配置里的输出目录"),
            ("浏览…", self._browse_dataset, "选择已生成的数据集目录"),
            ("扫描", self._scan_dataset, "重新扫描目录，刷新文件树与缩略图"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        outer.addLayout(bar)

        self.dataset_info = QLabel("尚未扫描")
        self.dataset_info.setObjectName("Hint")
        self.dataset_info.setWordWrap(True)
        outer.addWidget(self.dataset_info)

        split = QSplitter(Qt.Orientation.Horizontal)

        # 左：文件树
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(SectionTitle("样本树"))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["节点", "数量"])
        self.tree.setColumnWidth(0, 190)
        self.tree.itemClicked.connect(self._on_tree_click)
        left_layout.addWidget(self.tree, 1)

        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("定位身份"))
        self.identity_combo = QComboBox()
        self.identity_combo.setMinimumWidth(120)
        self.identity_combo.currentIndexChanged.connect(self._on_identity_picked)
        pick_row.addWidget(self.identity_combo, 1)
        left_layout.addLayout(pick_row)
        split.addWidget(left)

        # 中：缩略图网格
        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(SectionTitle("缩略图"))
        self.thumb_info = QLabel("—")
        self.thumb_info.setObjectName("Hint")
        head.addWidget(self.thumb_info)
        head.addStretch(1)
        self.max_thumbs = QComboBox()
        for count in (12, 24, 48, 96):
            self.max_thumbs.addItem(f"最多 {count} 张", count)
        self.max_thumbs.setCurrentIndex(1)
        self.max_thumbs.currentIndexChanged.connect(lambda: self._refresh_thumbs())
        head.addWidget(self.max_thumbs)
        middle_layout.addLayout(head)
        thumb_scroll = QScrollArea()
        thumb_scroll.setWidgetResizable(True)
        thumb_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.thumbs = ThumbnailGrid(columns=3)
        thumb_scroll.setWidget(self.thumbs)
        middle_layout.addWidget(thumb_scroll, 1)
        split.addWidget(middle)

        # 右：大图 + 叠加控制
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(SectionTitle("大图"))

        self.view = ImageView()
        self.view.setMinimumSize(300, 280)
        self.view.clear_image("选择左侧节点或中间缩略图")
        right_layout.addWidget(self.view, 1)

        zoom_row = QHBoxLayout()
        for text, slot in (("−", self.view.zoom_out), ("适应", self.view.fit), ("+", self.view.zoom_in)):
            button = QPushButton(text)
            button.setFixedWidth(48)
            button.clicked.connect(slot)
            zoom_row.addWidget(button)
        self.pixel_label = QLabel("—")
        self.pixel_label.setObjectName("Hint")
        self.view.pixel_hovered.connect(
            lambda x, y, g: self.pixel_label.setText(f"({x}, {y})  灰度 {g}")
        )
        zoom_row.addWidget(self.pixel_label, 1)
        right_layout.addLayout(zoom_row)

        overlay_row = QHBoxLayout()
        self.show_minutiae = QCheckBox("细节点")
        self.show_minutiae.setChecked(True)
        self.show_orientation = QCheckBox("方向场")
        self.show_mask = QCheckBox("掩模")
        self.show_singular = QCheckBox("奇异点")
        self.show_singular.setChecked(True)
        for check in (self.show_minutiae, self.show_orientation, self.show_mask, self.show_singular):
            check.toggled.connect(self._apply_overlays)
        overlay_row.addWidget(self.show_minutiae)
        overlay_row.addWidget(self.show_orientation)
        overlay_row.addWidget(self.show_mask)
        overlay_row.addWidget(self.show_singular)
        overlay_row.addStretch(1)
        right_layout.addLayout(overlay_row)

        self.detail_label = QLabel("—")
        self.detail_label.setObjectName("Hint")
        self.detail_label.setWordWrap(True)
        self.detail_label.setMinimumHeight(72)
        self.detail_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        right_layout.addWidget(self.detail_label)

        split.addWidget(right)
        split.setSizes([260, 420, 560])
        # 显式最小宽度 + 拉伸因子，否则某栏内容为空时会被压成一条缝
        left.setMinimumWidth(200)
        middle.setMinimumWidth(260)
        right.setMinimumWidth(320)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 2)
        outer.addWidget(split, 1)

        self._manifest: List[Dict] = []
        self._current: Dict = {}
        return page

    # ------------------------------------------------------------------
    def _use_current_output(self) -> None:
        self.dir_edit.setText(str(self.store.config.dataset_dir))
        self._scan_dataset()

    def _browse_dataset(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择数据集目录", self.dir_edit.text())
        if directory:
            self.dir_edit.setText(directory)
            self._scan_dataset()

    def _scan_dataset(self) -> None:
        from ...core.manifest import read_jsonl

        root = Path(self.dir_edit.text().strip())
        manifest = root / "dataset_manifest.jsonl"
        if not manifest.exists():
            # 允许用户直接选中 images/ 子目录或数据集内的子目录
            for parent in [root, *root.parents]:
                if (parent / "dataset_manifest.jsonl").exists():
                    root, manifest = parent, parent / "dataset_manifest.jsonl"
                    self.dir_edit.setText(str(root))
                    break
        if not manifest.exists():
            self.dataset_info.setText(
                f"✗ 未找到 dataset_manifest.jsonl（{manifest}）。"
                "请先生成数据集，或选择正确的数据集目录。"
            )
            self.dataset_info.setObjectName("Err")
            self.tree.clear()
            self.thumbs.clear()
            self.view.clear_image("未找到数据集")
            self._manifest = []
            return

        self._manifest = [r for r in read_jsonl(manifest) if r.get("status") in (None, "ok")]
        self.dataset_info.setObjectName("Hint")

        # 树：身份 → 手指 → 捺印 → 样本
        self.tree.clear()
        self.identity_combo.blockSignals(True)
        self.identity_combo.clear()
        grouped: Dict[str, Dict[str, Dict[str, List[Dict]]]] = {}
        for row in self._manifest:
            grouped.setdefault(str(row.get("identity_id", "?")), {}) \
                   .setdefault(str(row.get("finger", "?")), {}) \
                   .setdefault(str(row.get("impression_id", "?")), []).append(row)

        for identity, fingers in sorted(grouped.items()):
            total = sum(len(v) for f in fingers.values() for v in f.values())
            node = QTreeWidgetItem([identity, str(total)])
            node.setData(0, Qt.ItemDataRole.UserRole, {"kind": "identity", "id": identity})
            for finger, impressions in sorted(fingers.items()):
                finger_count = sum(len(v) for v in impressions.values())
                fnode = QTreeWidgetItem([finger, str(finger_count)])
                fnode.setData(0, Qt.ItemDataRole.UserRole,
                              {"kind": "finger", "id": identity, "finger": finger})
                for impression, rows in sorted(impressions.items()):
                    inode = QTreeWidgetItem([impression, str(len(rows))])
                    inode.setData(
                        0, Qt.ItemDataRole.UserRole,
                        {"kind": "impression", "id": identity, "finger": finger,
                         "impression": impression},
                    )
                    for row in sorted(rows, key=lambda r: str(r.get("sample_id", ""))):
                        leaf = QTreeWidgetItem([str(row.get("sample_id", "?")), str(row.get("tta_id", ""))])
                        leaf.setData(0, Qt.ItemDataRole.UserRole, {"kind": "sample", "row": row})
                        inode.addChild(leaf)
                    fnode.addChild(inode)
                node.addChild(fnode)
            self.tree.addTopLevelItem(node)
            self.identity_combo.addItem(f"{identity} ({total})", identity)
        self.identity_combo.blockSignals(False)

        size_mb = _dir_size(root) / (1024 * 1024)
        self.dataset_info.setText(
            f"✓ <b>{root.name}</b>　样本 {len(self._manifest)}　身份 {len(grouped)}　"
            f"目录体积 {size_mb:.1f} MB"
        )
        if self.tree.topLevelItemCount():
            self.tree.topLevelItem(0).setExpanded(True)
            self._load_group(self.tree.topLevelItem(0).data(0, Qt.ItemDataRole.UserRole))

        # 默认选中一张：否则右侧大图区是空的，用户会以为没扫描成功
        if self._manifest:
            first = min(self._manifest, key=lambda r: str(r.get("sample_id", "")))
            self._show_sample(first)

    def _on_identity_picked(self, _index: int) -> None:
        identity = self.identity_combo.currentData()
        if not identity:
            return
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data.get("id") == identity:
                self.tree.setCurrentItem(item)
                item.setExpanded(True)
                self._load_group(data)
                return

    def _on_tree_click(self, item: QTreeWidgetItem, _column: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        if data.get("kind") == "sample":
            self._show_sample(data["row"])
        else:
            self._load_group(data)

    def _load_group(self, data: Dict) -> None:
        kind = data.get("kind")
        rows = self._manifest
        if kind in ("identity", "finger", "impression"):
            rows = [
                r
                for r in rows
                if str(r.get("identity_id")) == str(data.get("id"))
                and (data.get("finger") is None or str(r.get("finger")) == str(data.get("finger")))
                and (
                    data.get("impression") is None
                    or str(r.get("impression_id")) == str(data.get("impression"))
                )
            ]
        self._refresh_thumbs(rows)

    def _refresh_thumbs(self, rows: Optional[List[Dict]] = None) -> None:
        if rows is None:
            rows = [self._current] if self._current else self._manifest
        limit = int(self.max_thumbs.currentData() or 24)
        items: List[Tuple[np.ndarray, Dict]] = []
        root = Path(self.dir_edit.text().strip())
        for row in rows[:limit]:
            image = _load_image(root, row.get("image_path"))
            if image is None:
                continue
            thumb = cv2.resize(image, (110, 150), interpolation=cv2.INTER_AREA)
            items.append((thumb, row))
        self.thumbs.set_items(items)
        self.thumbs.selected.connect(lambda _i, row: self._show_sample(row))
        self.thumb_info.setText(f"{len(items)} / {len(rows)} 张")

    def _show_sample(self, row: Dict) -> None:
        root = Path(self.dir_edit.text().strip())
        image = _load_image(root, row.get("image_path"))
        if image is None:
            self.view.clear_image("图像不可读")
            return
        self._current = row

        minutiae = _load_minutiae(root, row.get("minutiae_path"))
        orientation = _load_orientation(root, row.get("orientation_path"))
        singular = _load_singular_points(root, row)

        self.view.set_image(
            image,
            minutiae=minutiae,
            orientation=orientation if self.show_orientation.isChecked() else None,
            singular_points=singular,
        )
        self._apply_overlays()
        self._refresh_detail(row, len(minutiae or []))

    def _apply_overlays(self) -> None:
        self.view.set_overlays(
            show_minutiae=self.show_minutiae.isChecked(),
            show_orientation=self.show_orientation.isChecked(),
            show_singular=self.show_singular.isChecked(),
        )
        if self.show_mask.isChecked():
            root = Path(self.dir_edit.text().strip())
            mask = _load_image(root, self._current.get("mask_path"))
            if mask is not None and self._current:
                image = _load_image(root, self._current.get("image_path"))
                if image is not None:
                    overlay = image.copy()
                    overlay[mask == 0] = 255
                    overlay[mask > 0] = (image[mask > 0] * 0.45 + 255 * 0.55).astype(np.uint8)
                    self.view.set_image(
                        overlay,
                        minutiae=_load_minutiae(root, self._current.get("minutiae_path")),
                        orientation=None,
                        singular_points=_load_singular_points(root, self._current),
                        reset_zoom=False,
                    )

    def _refresh_detail(self, row: Dict, minutiae_count: int) -> None:
        quality = row.get("quality_detail") or {}
        lines = [
            f"<b>{row.get('sample_id', '?')}</b>　类别 <b>{row.get('class_label', '?')}</b>　"
            f"划分 {row.get('split', '?')}",
            f"尺寸 {row.get('width', '?')}×{row.get('height', '?')} @ {row.get('ppi', '?')}ppi　"
            f"TTA：{'、'.join(row.get('tta_operators') or []) or '（原始捺印）'}　"
            f"失真度 {row.get('tta_distortion', 0):.2f}",
        ]
        if row.get("nfiq2") is not None:
            lines.append(
                f"质量 <b>{row['nfiq2']:.1f}</b>/100 (L{row.get('quality_level', '?')})　"
                f"清晰度 {quality.get('clarity', 0):.3f}　对比度 {quality.get('contrast', 0):.3f}　"
                f"方向一致性 {quality.get('orientation_certainty', 0):.3f}　"
                f"有效面积 {quality.get('usable_area', 0):.3f}"
            )
        lines.append(
            f"细节点 <b>{row.get('minutiae_count', minutiae_count)}</b>"
            f"（末梢 {row.get('minutiae_endings', '?')}　分叉 {row.get('minutiae_bifurcations', '?')}）　"
            f"图像 {row.get('bytes', 0) / 1024:.1f} KB"
        )
        lines.append(f"<span style='color:#8a8a96'>sha256 {str(row.get('sha256', ''))[:24]}…</span>")
        self.detail_label.setText("<br>".join(lines))

    # ==================================================================
    # 即时渲染
    # ==================================================================
    def _build_live_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        controls = QHBoxLayout()
        self.live_identity = QLineEdit("000001")
        self.live_identity.setMaximumWidth(90)
        self.live_finger = QComboBox()
        from ...config.schema import (
            CLASS_LABELS_ZH,
            FINGER_LABELS_ZH,
            FINGER_NAMES,
            FINGERPRINT_CLASSES,
        )

        for name in FINGER_NAMES:
            self.live_finger.addItem(FINGER_LABELS_ZH.get(name, name), name)
        self.live_class = QComboBox()
        self.live_class.addItem("（按配置分布随机）", "")
        for name in FINGERPRINT_CLASSES:
            self.live_class.addItem(CLASS_LABELS_ZH.get(name, name), name)
        self.live_tta = QComboBox()
        self.live_tta.addItem("原始捺印 (tta=0)", 0)
        for i in range(1, 6):
            self.live_tta.addItem(f"TTA #{i}", i)
        self.live_seed = QLineEdit("1234")
        self.live_seed.setMaximumWidth(90)

        for label, widget in (
            ("身份", self.live_identity),
            ("手指", self.live_finger),
            ("类别", self.live_class),
            ("TTA", self.live_tta),
            ("种子", self.live_seed),
        ):
            controls.addWidget(QLabel(label))
            controls.addWidget(widget)

        self.live_btn = QPushButton("渲染")
        self.live_btn.setObjectName("Primary")
        self.live_btn.clicked.connect(self._on_live_render)
        controls.addWidget(self.live_btn)
        self.live_status = QLabel("就绪")
        self.live_status.setObjectName("Hint")
        controls.addWidget(self.live_status, 1)
        outer.addLayout(controls)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.live_view = ImageView()
        self.live_view.clear_image("点击「渲染」")
        split.addWidget(self.live_view)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(SectionTitle("渲染详情"))
        self.live_detail = QLabel("—")
        self.live_detail.setObjectName("Hint")
        self.live_detail.setWordWrap(True)
        self.live_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        side_layout.addWidget(self.live_detail)
        side_layout.addWidget(SectionTitle("逐算子对比"))
        grid_scroll = QScrollArea()
        grid_scroll.setWidgetResizable(True)
        grid_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.live_grid = ThumbnailGrid(columns=2, thumb_size=(130, 195))
        grid_scroll.setWidget(self.live_grid)
        side_layout.addWidget(grid_scroll, 1)
        split.addWidget(side)
        split.setSizes([560, 420])
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        outer.addWidget(split, 1)
        return page

    def _on_live_render(self) -> None:
        if getattr(self, "_live_running", False):
            return
        self._live_running = True
        self.live_btn.setEnabled(False)
        self.live_status.setText("渲染中…")

        try:
            seed = int(self.live_seed.text() or "1234")
        except ValueError:
            seed = 1234

        fp_class = self.live_class.currentData() or None
        worker = PreviewWorker(
            PreviewService(self.store.config),
            {
                "identity_id": self.live_identity.text().strip() or "000001",
                "finger": self.live_finger.currentData(),
                "fp_class": fp_class,
                "seed": seed,
                "tta_index": int(self.live_tta.currentData() or 0),
                "with_annotation": True,
            },
            grid_operators=list(self.store.config.tta.operators)[:6],
        )
        worker.rendered.connect(self._on_live_done)
        worker.failed.connect(self._on_live_failed)
        worker.finished.connect(lambda: self._workers.remove(worker) if worker in self._workers else None)
        self._workers.append(worker)
        worker.start()

    def _on_live_done(self, result) -> None:
        self._live_running = False
        self.live_btn.setEnabled(True)
        if result.error:
            self.live_status.setText(f"✗ {result.error}")
            return
        self.live_view.set_image(
            result.image,
            minutiae=result.minutiae,
            orientation=result.orientation,
            singular_points=(result.master.singular_points if result.master else []),
        )
        master = result.master
        quality = result.quality
        lines = [
            f"类别 <b>{master.fp_class if master else '?'}</b>　"
            f"尺寸 {result.image.shape[1]}×{result.image.shape[0]}　"
            f"耗时 {result.duration_s:.2f}s",
            f"轮廓 5 参数：{master.shape_params if master else {}}",
            f"奇异点 {len(master.singular_points) if master else 0} 个",
            f"捺印效果：{'、'.join(str(e.get('stage')) for e in result.effects) or '无'}",
            f"TTA：{'、'.join(result.tta_operators) or '（原始捺印）'}",
        ]
        if result.tta_params:
            lines.append(
                "<span style='color:#9a9aa6'>"
                + json.dumps(result.tta_params, ensure_ascii=False)[:300]
                + "</span>"
            )
        if quality is not None:
            lines.append(
                f"质量 <b>{quality.score:.1f}</b>/100 (L{quality.level})　清晰度 {quality.clarity:.3f}　"
                f"对比度 {quality.contrast:.3f}　方向一致性 {quality.orientation_certainty:.3f}　"
                f"脊线周期 {quality.ridge_period:.1f}px"
            )
        lines.append(f"细节点 <b>{len(result.minutiae)}</b> 个")
        self.live_detail.setText("<br>".join(lines))

        tiles = getattr(result, "tta_grid", []) or []
        items: List[Tuple[np.ndarray, str]] = []
        for label, image in tiles:
            items.append((cv2.resize(image, (130, 195), interpolation=cv2.INTER_AREA), label))
        items.append(
            (cv2.resize(result.image, (130, 195), interpolation=cv2.INTER_AREA), "当前 TTA 结果")
        )
        self.live_grid.set_items(items)
        self.live_status.setText(f"✓ 完成（{len(items)} 张预览）")

    def _on_live_failed(self, message: str) -> None:
        self._live_running = False
        self.live_btn.setEnabled(True)
        self.live_status.setText(f"✗ {message}")

    # ==================================================================
    def on_refresh(self) -> None:
        if not self.dir_edit.text().strip():
            self.dir_edit.setText(str(self.store.config.dataset_dir))

    def wait_for_workers(self, timeout_ms: int = 8000) -> None:
        for worker in list(self._workers):
            worker.wait(timeout_ms)


# ---------------------------------------------------------------------------
# 读取助手（直接读磁盘，不经过核心层，保证预览与落盘结果完全一致）
# ---------------------------------------------------------------------------
def _load_image(root: Path, relative: Optional[str]) -> Optional[np.ndarray]:
    if not relative:
        return None
    path = root / relative
    if not path.exists():
        return None
    from ...legacy.util.image_utils import imread_unicode

    return imread_unicode(str(path), cv2.IMREAD_GRAYSCALE)


def _load_minutiae(root: Path, relative: Optional[str]) -> List[Minutia]:
    if not relative:
        return []
    path = root / relative
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    out: List[Minutia] = []
    for item in payload.get("minutiae", []):
        out.append(
            Minutia(
                x=int(item.get("x", 0)),
                y=int(item.get("y", 0)),
                angle=float(item.get("angle", 0.0)),
                kind=str(item.get("type", "ending")),
                quality=float(item.get("quality", 1.0)),
            )
        )
    return out


def _load_orientation(root: Path, relative: Optional[str]) -> Optional[np.ndarray]:
    if not relative:
        return None
    path = root / relative
    if not path.exists():
        return None
    from ...core.quantize import decode_orientation, load_array_png

    try:
        encoded, _meta = load_array_png(path)
    except FileNotFoundError:
        return None
    return decode_orientation(encoded)


def _load_singular_points(root: Path, row: Dict) -> List[Dict]:
    """奇异点只在主指纹阶段有；这里从同捺印的细节点 JSON 里读（若已写入）。"""
    payload_path = row.get("minutiae_path")
    if not payload_path:
        return []
    path = root / payload_path
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return list(payload.get("singular_points", []) or [])


def _dir_size(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:  # pragma: no cover
                continue
    return total
