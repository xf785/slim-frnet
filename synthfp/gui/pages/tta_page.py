"""页面 3：TTA 配置（设计文档 F4 / 4.6）。

左侧勾选算子并调节参数范围，右侧**实时预览**同一张捺印在各算子下的效果，
这样「参数范围」不再是抽象数字，而是看得见的结果。
"""

from __future__ import annotations

from typing import Any, Dict, List

from ...config.schema import TTA_CATALOG, TTA_PARAM_SPECS
from ..qt import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from ..qt_bridge import PreviewWorker
from ..widgets import ImageView, SectionTitle
from .base import BasePage

__all__ = ["TTAPage"]

_CATEGORIES = ("几何", "光度", "噪声", "压缩", "遮挡", "传感器")
_CATEGORY_ICON = {
    "几何": "🔷",
    "光度": "🔆",
    "噪声": "🌫",
    "压缩": "🗜",
    "遮挡": "🚧",
    "传感器": "📷",
}
_LABELS = {name: label for _cat, name, label in TTA_CATALOG}


class TTAPage(BasePage):
    """TTA 算子勾选 + 参数范围 + 效果对比。"""

    title = "TTA 配置"
    icon = "⇄"

    def build(self) -> None:
        self._checks: Dict[str, QCheckBox] = {}
        self._param_widgets: Dict[str, Dict[str, QWidget]] = {}
        self._preview_workers: List[PreviewWorker] = []

        # ---------------- 组合策略 ----------------
        _box, form = self.group("组合策略")
        self.enabled_check = QCheckBox("启用 TTA 阶段")
        self.bind_check("tta.enabled", self.enabled_check, on_change=self._on_enabled)
        form.addRow("", self.enabled_check)

        self.mode_combo = QComboBox()
        for value, label in (
            ("mixed", "混合（随机单算子或算子链，推荐）"),
            ("single", "单算子（每个 TTA 只施加 1 个）"),
            ("chain", "算子链（每个 TTA 施加多个）"),
        ):
            self.mode_combo.addItem(label, value)
        self.bind_combo("tta.mode", self.mode_combo, values=["mixed", "single", "chain"],
                        labels=["混合（随机单算子或算子链，推荐）", "单算子（每个 TTA 只施加 1 个）",
                                "算子链（每个 TTA 施加多个）"])
        form.addRow("模式", self.mode_combo)

        chain_holder = QWidget()
        chain_layout = QHBoxLayout(chain_holder)
        chain_layout.setContentsMargins(0, 0, 0, 0)
        self.chain_low = QSpinBox()
        self.chain_high = QSpinBox()
        for spin in (self.chain_low, self.chain_high):
            spin.setRange(1, 8)
            chain_layout.addWidget(spin)
        chain_layout.insertWidget(1, QLabel("~"))
        chain_layout.addStretch(1)
        self._int_interval("tta.chain_length", self.chain_low, self.chain_high)
        form.addRow("链长范围", chain_holder)

        self.chain_prob = QDoubleSpinBox()
        self.chain_prob.setRange(0.0, 1.0)
        self.chain_prob.setSingleStep(0.05)
        self.chain_prob.setDecimals(2)
        self.bind_double("tta.chain_probability", self.chain_prob)
        self.chain_prob.setToolTip("仅「混合」模式生效：走算子链的概率")
        form.addRow("走链概率", self.chain_prob)

        self.distortion = QDoubleSpinBox()
        self.distortion.setRange(0.05, 1.0)
        self.distortion.setSingleStep(0.05)
        self.distortion.setDecimals(2)
        self.bind_double("tta.max_distortion", self.distortion)
        self.distortion.setToolTip("累计失真度上限，超出则不追加更多算子")
        form.addRow("最大累计失真", self.distortion)

        hint = QLabel(
            "约定：<b>tta_index = 0</b> 的样本不施加任何算子，保留原始捺印作为干净对照；"
            "算子链在<b>样本种子</b>上采样，因此同一样本号在任何机器上都得到相同结果。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        form.addRow("", hint)

        # ---------------- 算子选择 + 参数 ----------------
        split = QHBoxLayout()
        split.setSpacing(12)

        # 左：算子勾选
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.addWidget(SectionTitle("算子清单"))
        title_row.addStretch(1)
        for text, slot, tip in (
            ("全选", lambda: self._set_all(True), "勾选全部 21 个算子"),
            ("全不选", lambda: self._set_all(False), "取消全部勾选"),
            ("推荐", self._apply_recommended, "恢复默认的 6 个轻度算子"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            title_row.addWidget(button)
        left_layout.addLayout(title_row)

        scroll = QWidget()
        scroll.setMinimumWidth(300)
        holder = scroll
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 6, 0)
        holder_layout.setSpacing(8)
        for category in _CATEGORIES:
            names = [n for cat, n, _l in TTA_CATALOG if cat == category]
            if not names:
                continue
            box = QGroupBox(f"{_CATEGORY_ICON.get(category, '')} {category}（{len(names)}）")
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(10, 14, 10, 8)
            box_layout.setSpacing(4)
            for name in names:
                check = QCheckBox(_LABELS.get(name, name))
                check.setToolTip(self._operator_tooltip(name))
                check.toggled.connect(lambda checked, n=name: self._on_operator_toggled(n, checked))
                box_layout.addWidget(check)
                self._checks[name] = check
            holder_layout.addWidget(box)
        holder_layout.addStretch(1)
        left_layout.addWidget(scroll)
        split.addWidget(left, 2)

        # 右：参数 + 预览
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        tabs = QTabWidget()
        self.param_tabs = QTabWidget()
        for category in _CATEGORIES:
            names = [n for cat, n, _l in TTA_CATALOG if cat == category and n in TTA_PARAM_SPECS]
            if not names:
                continue
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(8, 8, 8, 8)
            page_layout.setSpacing(6)
            for name in names:
                page_layout.addWidget(self._build_param_group(name))
            page_layout.addStretch(1)
            # 不再给每个标签页套 QScrollArea：页面本身已经有滚动区，
            # 嵌套滚动区会互相争尺寸，最终把内容压成 0 高度。
            self.param_tabs.addTab(page, f"{_CATEGORY_ICON.get(category, '')} {category}")
        tabs.addTab(self.param_tabs, "参数范围")

        preview_page = QWidget()
        preview_layout = QVBoxLayout(preview_page)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(6)
        row = QHBoxLayout()
        self.preview_btn = QPushButton("生成对比预览")
        self.preview_btn.setObjectName("Primary")
        self.preview_btn.setToolTip("用当前勾选与参数渲染一张捺印，并逐算子展示效果")
        self.preview_btn.clicked.connect(self._on_preview)
        row.addWidget(self.preview_btn)
        self.preview_status = QLabel("尚未预览")
        self.preview_status.setObjectName("Hint")
        self.preview_status.setWordWrap(True)
        row.addWidget(self.preview_status, 1)
        preview_layout.addLayout(row)

        self.preview_view = ImageView()
        self.preview_view.setMinimumHeight(240)
        self.preview_view.clear_image("点击「生成对比预览」查看效果")
        preview_layout.addWidget(self.preview_view, 1)
        tabs.addTab(preview_page, "效果对比")

        right_layout.addWidget(tabs, 1)
        split.addWidget(right, 4)

        wrapper = QWidget()
        wrapper.setLayout(split)
        wrapper.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.add_widget(wrapper)

    # ------------------------------------------------------------------
    # 参数控件
    # ------------------------------------------------------------------
    def _build_param_group(self, name: str) -> QGroupBox:
        box = QGroupBox(_LABELS.get(name, name))
        form = QVBoxLayout(box)
        form.setContentsMargins(10, 14, 10, 8)
        form.setSpacing(4)
        widgets: Dict[str, QWidget] = {}
        for param, spec in TTA_PARAM_SPECS.get(name, {}).items():
            label, lo, hi, step, unit = spec
            is_int = float(step).is_integer() and float(lo).is_integer() and float(hi).is_integer()
            row = QHBoxLayout()
            text = QLabel(label)
            text.setMinimumWidth(140)
            row.addWidget(text)
            if is_int:
                widget: QWidget = QSpinBox()
                widget.setRange(int(lo), int(hi))
                widget.setSingleStep(max(int(step), 1))
                widget.setValue(int(lo))
                widget.valueChanged.connect(
                    lambda value, n=name, p=param: self._on_param_changed(n, p, int(value))
                )
            else:
                widget = QDoubleSpinBox()
                widget.setRange(float(lo), float(hi))
                widget.setSingleStep(float(step) if step else 0.01)
                widget.setDecimals(3 if float(step) < 0.01 else 2)
                widget.setValue(float(hi))
                widget.valueChanged.connect(
                    lambda value, n=name, p=param: self._on_param_changed(n, p, float(value))
                )
            if unit:
                widget.setSuffix(" " + unit)
            widget.setToolTip("用户固定的值；未固定的参数会在该范围内随机采样")
            row.addWidget(widget, 1)
            fix = QCheckBox("固定")
            fix.setToolTip("勾选后该参数不再随机采样，始终使用此值")
            fix.toggled.connect(
                lambda checked, n=name, p=param: self._on_param_fixed(n, p, checked)
            )
            row.addWidget(fix)
            widgets[param] = widget
            widgets[f"__fix_{param}"] = fix
            form.addWidget(_wrap(row))
        self._param_widgets[name] = widgets
        return box

    def _on_param_changed(self, name: str, param: str, value: Any) -> None:
        if self._updating:
            return
        params: Dict[str, Dict[str, Any]] = {
            key: dict(val) for key, val in self.store.config.tta.params.items()
        }
        entry = params.setdefault(name, {})
        entry[param] = value
        self.store.update_flat({"tta.params": params})

    def _on_param_fixed(self, name: str, param: str, checked: bool) -> None:
        if self._updating:
            return
        params: Dict[str, Dict[str, Any]] = {
            key: dict(val) for key, val in self.store.config.tta.params.items()
        }
        entry = params.setdefault(name, {})
        if checked:
            widget = self._param_widgets[name][param]
            entry[param] = (
                int(widget.value()) if isinstance(widget, QSpinBox) else float(widget.value())
            )
        else:
            # 取消固定 = 从 params 中移除该键，回到 [min, max] 随机采样
            entry.pop(param, None)
            if not entry:
                params.pop(name, None)
        self.store.update_flat({"tta.params": params})

    # ------------------------------------------------------------------
    # 算子勾选
    # ------------------------------------------------------------------
    def _on_operator_toggled(self, name: str, checked: bool) -> None:
        if self._updating:
            return
        operators = list(self.store.config.tta.operators)
        if checked and name not in operators:
            operators.append(name)
        elif not checked and name in operators:
            operators.remove(name)
        if not operators and self.store.config.tta.enabled:
            # 不允许一个都不选（Pydantic 也会拒绝），直接回滚这一个勾选
            self._updating = True
            self._checks[name].setChecked(True)
            self._updating = False
            self._notify("启用 TTA 时至少需要勾选一个算子")
            return
        if not self.store.update_flat({"tta.operators": operators}):
            self._updating = True
            self._checks[name].setChecked(not checked)
            self._updating = False

    def _set_all(self, checked: bool) -> None:
        operators = list(_LABELS) if checked else []
        if not operators:
            self._notify("至少保留一个算子；如需完全不启用 TTA，请取消勾选「启用 TTA 阶段」")
            return
        self.store.update_flat({"tta.operators": operators})

    def _apply_recommended(self) -> None:
        self.store.update_flat(
            {
                "tta.operators": ["rotate", "scale", "brightness", "contrast",
                                  "gaussian_noise", "blur"],
                "tta.mode": "mixed",
                "tta.params": {},
            }
        )

    def _on_enabled(self, enabled: bool) -> None:
        self.param_tabs.setEnabled(True)
        self.mode_combo.setEnabled(enabled)

    # ------------------------------------------------------------------
    # 预览
    # ------------------------------------------------------------------
    def _on_preview(self) -> None:
        from ...app import PreviewService

        if getattr(self, "_preview_running", False):
            return
        cfg = self.store.config
        operators: List[str] = list(cfg.tta.operators)[:6]  # 最多并排 6 张，避免界面拥挤
        if not operators:
            self._notify("请先勾选至少一个算子")
            return

        self._preview_running = True
        self.preview_btn.setEnabled(False)
        self.preview_status.setText("渲染中…（首次会构建 Gabor 核库，稍慢）")

        worker = PreviewWorker(
            PreviewService(cfg),
            {
                "identity_id": "000001",
                "finger": "left_thumb",
                "fp_class": "left_loop",
                "seed": 1234,
                "tta_index": 0,
                "with_annotation": False,
            },
            grid_operators=operators,
        )
        worker.rendered.connect(self._on_preview_done)
        worker.failed.connect(self._on_preview_failed)
        worker.finished.connect(lambda: self._preview_workers.remove(worker)
                                if worker in self._preview_workers else None)
        self._preview_workers.append(worker)
        worker.start()

    def _on_preview_done(self, result) -> None:

        self._preview_running = False
        self.preview_btn.setEnabled(True)
        if result.error:
            self.preview_status.setText(f"✗ {result.error}")
            return
        tiles = list(getattr(result, "tta_grid", []) or [])
        if not tiles:
            self.preview_status.setText("✗ 未生成对比图")
            return
        sheet = _montage([image for _label, image in tiles])
        self.preview_view.set_image(sheet)
        names = "、".join(label for label, _img in tiles)
        self.preview_status.setText(
            f"✓ 已渲染 {len(tiles)} 张（{names}）　基础图 {result.tta_grid[0][1].shape[1]}"
            f"×{result.tta_grid[0][1].shape[0]}"
        )

    def _on_preview_failed(self, message: str) -> None:
        self._preview_running = False
        self.preview_btn.setEnabled(True)
        self.preview_status.setText(f"✗ {message}")

    # ------------------------------------------------------------------
    # 刷新
    # ------------------------------------------------------------------
    def on_refresh(self) -> None:
        cfg = self.store.config
        enabled = set(cfg.tta.operators)
        for name, check in self._checks.items():
            check.setChecked(name in enabled)
        for name, widgets in self._param_widgets.items():
            fixed = cfg.tta.params.get(name, {})
            for key, widget in widgets.items():
                if key.startswith("__fix_"):
                    continue
                if widget is None:
                    continue
                if key in fixed:
                    widget.blockSignals(True)
                    if isinstance(widget, QSpinBox):
                        widget.setValue(int(fixed[key]))
                    else:
                        widget.setValue(float(fixed[key]))  # type: ignore[attr-defined]
                    widget.blockSignals(False)
                fix = widgets.get(f"__fix_{key}")
                if isinstance(fix, QCheckBox):
                    fix.blockSignals(True)
                    fix.setChecked(key in fixed)
                    fix.blockSignals(False)
        self.enabled_check.setChecked(cfg.tta.enabled)
        self.chain_low.setValue(int(cfg.tta.chain_length[0]))
        self.chain_high.setValue(int(cfg.tta.chain_length[1]))
        self.mode_combo.setEnabled(cfg.tta.enabled)

    # ------------------------------------------------------------------
    def _int_interval(self, path: str, low: QSpinBox, high: QSpinBox) -> None:
        def push(*_args: Any) -> None:
            if self._updating:
                return
            value = (int(low.value()), int(high.value()))
            if value[0] > value[1]:
                value = (value[1], value[1])
                high.setValue(value[1])
            self.store.update_flat({path: value})

        low.valueChanged.connect(push)
        high.valueChanged.connect(push)

        def setter(_w: QWidget, value: Any) -> None:
            if isinstance(value, (tuple, list)) and len(value) == 2:
                low.setValue(int(value[0]))
                high.setValue(int(value[1]))

        self._bindings.append((path, low, lambda w: int(w.value()), setter))

    @staticmethod
    def _operator_tooltip(name: str) -> str:
        spec = TTA_PARAM_SPECS.get(name, {})
        if not spec:
            return _LABELS.get(name, name)
        parts = ", ".join(f"{p}({s[0]} {s[1]}~{s[2]}{s[4]})" for p, s in spec.items())
        return f"{_LABELS.get(name, name)}：{parts}"


def _wrap(layout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget


def _montage(images: List[Any], columns: int = 3, tile: tuple = (200, 300)) -> Any:
    """把多张灰度图拼成一张对比图（等宽等高、带间隔）。"""
    import cv2
    import numpy as np

    if not images:
        return np.full((tile[1], tile[0]), 255, dtype=np.uint8)
    tiles = [
        cv2.resize(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), tile,
                   interpolation=cv2.INTER_AREA)
        for img in images
    ]
    rows = []
    for start in range(0, len(tiles), columns):
        chunk = tiles[start : start + columns]
        while len(chunk) < columns:
            chunk.append(np.full(tile[::-1], 255, dtype=np.uint8))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)
