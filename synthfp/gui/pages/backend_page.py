"""页面 2：后端配置（设计文档 F3 / 4.5）。

展示并调节生成内核（SFinGe 复刻）的全部参数：分辨率、脊线合成、
外观、手指轮廓比例、奇异点。所有控件都直接读写配置对象，
因此「GUI 参数」与「config.resolved.yaml」永远一致。
"""

from __future__ import annotations

from typing import Optional

from ...config.schema import RESOLUTION_PRESETS
from ..qt import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)
from .base import BasePage

__all__ = ["BackendPage"]

_RESOLUTION_LABELS = {
    "finger_500dpi": "单指 320×480 @500ppi（推荐）",
    "finger_1000dpi": "单指 480×720 @1000ppi",
    "square_512": "方形 512×512",
    "thumb_flat": "拇指平捺 400×640",
    "custom": "自定义尺寸",
}


def _wrap(layout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget


class BackendPage(BasePage):
    """生成内核参数页面。"""

    title = "后端配置"
    icon = "⚙️"

    def build(self) -> None:
        self._build_backend()
        self._build_resolution()
        self._build_ridge()
        self._build_appearance()
        self._build_shape()
        self._build_test()

    # ------------------------------------------------------------------
    # 生成后端
    # ------------------------------------------------------------------
    def _build_backend(self) -> None:
        _box, form = self.group("生成后端（可插拔）")
        self.backend_combo = QComboBox()
        self._populate_backends()
        self.bind_combo(
            "backend.name", self.backend_combo,
            values=self._backend_names(), labels=self._backend_labels(),
            on_change=self._on_backend,
        )
        form.addRow("后端", self.backend_combo)

        self.backend_desc = QLabel()
        self.backend_desc.setObjectName("Hint")
        self.backend_desc.setWordWrap(True)
        form.addRow("", self.backend_desc)

        refresh = QPushButton("重新扫描后端")
        refresh.setToolTip("重新导入插件层，发现新注册的后端")
        refresh.clicked.connect(self._rescan_backends)
        form.addRow("", refresh)

    def _backend_names(self):
        from ...plugins.backends import list_backends

        return list_backends() or ["sfinge"]

    def _backend_labels(self):
        from ...plugins.backends import BackendRegistry, list_backends

        out = []
        for name in list_backends():
            try:
                out.append(f"{name} — {BackendRegistry.get(name).description[:28]}")
            except Exception:  # pragma: no cover
                out.append(name)
        return out or ["sfinge"]

    def _populate_backends(self) -> None:
        self.backend_combo.clear()
        for label, name in zip(self._backend_labels(), self._backend_names(), strict=True):
            self.backend_combo.addItem(label, name)

    def _rescan_backends(self) -> None:
        current = self.backend_combo.currentData()
        self.backend_combo.blockSignals(True)
        self._populate_backends()
        index = self.backend_combo.findData(current)
        self.backend_combo.setCurrentIndex(max(index, 0))
        self.backend_combo.blockSignals(False)
        self.refresh()

    def _on_backend(self, _value: str) -> None:
        self.refresh()

    # ------------------------------------------------------------------
    # 分辨率
    # ------------------------------------------------------------------
    def _build_resolution(self) -> None:
        _box, form = self.group("图像分辨率")
        self.resolution_combo = QComboBox()
        for key in RESOLUTION_PRESETS:
            self.resolution_combo.addItem(_RESOLUTION_LABELS.get(key, key), key)
        self.bind_combo("backend.resolution", self.resolution_combo, values=list(RESOLUTION_PRESETS),
                        labels=[_RESOLUTION_LABELS.get(k, k) for k in RESOLUTION_PRESETS],
                        on_change=self._on_resolution)
        form.addRow("分辨率预设", self.resolution_combo)

        size_row = QWidget()
        size_layout = QHBoxLayout(size_row)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setSpacing(6)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 2048)
        self.width_spin.setToolTip("自定义图像宽度（像素）")
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 2048)
        self.height_spin.setToolTip("自定义图像高度（像素）")
        self.custom_check = QCheckBox("启用自定义")
        self.custom_check.setToolTip("勾选后忽略分辨率预设，使用下方宽高")
        self.custom_check.toggled.connect(self._on_custom_toggled)
        for spin in (self.width_spin, self.height_spin):
            spin.valueChanged.connect(self._on_custom_size)
        clear = QPushButton("回到预设")
        clear.clicked.connect(self._clear_custom_size)
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(self.height_spin)
        size_layout.addWidget(self.custom_check)
        size_layout.addWidget(clear)
        size_layout.addStretch(1)
        form.addRow("自定义尺寸", size_row)

        self.ppi_spin = QSpinBox()
        self.ppi_spin.setRange(100, 4000)
        self.bind_spin("backend.ppi", self.ppi_spin, suffix=" ppi")
        self.ppi_spin.setToolTip("写入 manifest 的分辨率标注（不改变像素数）")
        form.addRow("分辨率标注", self.ppi_spin)

    def _on_resolution(self, _value: Optional[str]) -> None:
        self._sync_resolution_widgets()
        self.update_estimate()

    def _on_custom_toggled(self, enabled: bool) -> None:
        if enabled:
            self.store.update_flat(
                {"backend.image_size": [self.width_spin.value(), self.height_spin.value()]}
            )
        else:
            self.store.update_flat({"backend.image_size": None})
        self._sync_resolution_widgets()
        self.update_estimate()

    def _on_custom_size(self, *_args) -> None:
        if not self.custom_check.isChecked():
            return
        self.store.update_flat(
            {"backend.image_size": [self.width_spin.value(), self.height_spin.value()]}
        )
        self.update_estimate()

    def _clear_custom_size(self) -> None:
        self.custom_check.setChecked(False)

    def _sync_resolution_widgets(self) -> None:
        custom = self.store.config.backend.image_size is not None
        self.custom_check.blockSignals(True)
        self.custom_check.setChecked(custom)
        self.custom_check.blockSignals(False)
        for spin in (self.width_spin, self.height_spin):
            spin.setEnabled(custom)
        self.resolution_combo.setEnabled(not custom)

    # ------------------------------------------------------------------
    # 脊线
    # ------------------------------------------------------------------
    def _build_ridge(self) -> None:
        _box, form = self.group("脊线合成（Gabor 迭代）")
        hint = QLabel(
            "「频率档数」与「迭代分辨率比例」是吞吐的两个主要杠杆："
            "前者决定每次迭代的卷积次数，后者决定每次卷积的像素量。"
            "默认值 5 档 / 0.7 在 320×480 下单张约 0.6~0.9 秒。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        form.addRow("", hint)

        self._int_row(form, "最窄脊宽", "backend.min_ridge_width", 2, 40,
                      "像素；对应最高空间频率 1/宽度")
        self._int_row(form, "最宽脊宽", "backend.max_ridge_width", 3, 60,
                      "像素；对应最低空间频率 1/宽度")
        self._int_row(form, "Gabor 核半径", "backend.gabor_filter_size", 4, 24,
                      "核尺寸 = 2r+1；SFinGe 取 10")
        self._int_row(form, "频率档数", "backend.gabor_n_frequencies", 2, 20,
                      "5 档对指纹已足够；调大更慢")
        self._int_row(form, "迭代上限", "backend.max_iterations", 10, 400,
                      "达到收敛阈值会提前结束")
        self._float_row(form, "迭代分辨率比例", "backend.coarse_scale", 0.3, 1.0, 0.05, 2, "×",
                        "0.7 速度约 2 倍且几乎无肉眼差异；1.0 最锐利")
        self._float_row(form, "收敛阈值", "backend.convergence_eps", 1e-5, 0.1, 1e-4, 5, "",
                        "图案像素数变化率低于此值即停止")
        self._float_row(form, "初始种子密度", "backend.seed_density", 1e-5, 0.05, 1e-3, 5, "",
                        "SFinGe 取 0.1%")
        self._float_row(form, "频率波动比例", "backend.freq_spread", 0.05, 1.0, 0.05, 2, "×",
                        "频率图相对脊宽范围的波动幅度")

    # ------------------------------------------------------------------
    # 外观
    # ------------------------------------------------------------------
    def _build_appearance(self) -> None:
        _box, form = self.group("外观")

        self.ridge_only_black = QCheckBox("仅脊线显示为黑色（纯二值图）")
        self.ridge_only_black.setToolTip(
            "打开后输出纯二值图：脊线为 0（纯黑），谷线为 255（纯白）。\n"
            "同时忽略下面的灰度区间、皮肤纹理与底噪 —— 适合分割 / 细化 / "
            "细节点算法直接使用，也方便数据集人工核对。"
        )
        self.bind_check(
            "backend.ridge_only_black",
            self.ridge_only_black,
            on_change=lambda _v: self._sync_color_enabled(),
        )
        form.addRow("", self.ridge_only_black)

        self._color_widgets = []
        for label, path, tip in (
            ("脊线灰度下界", "backend.ridge_gray.0", "脊线像素灰度最小值"),
            ("脊线灰度上界", "backend.ridge_gray.1", "脊线像素灰度最大值"),
            ("谷线灰度下界", "backend.valley_gray.0", "谷线像素灰度最小值"),
            ("谷线灰度上界", "backend.valley_gray.1", "谷线像素灰度最大值"),
        ):
            widget = self._int_row(form, label, path, 0, 255, tip)
            self._color_widgets.append(widget)
        self._float_row(form, "底噪标准差", "backend.noise_sigma", 0.0, 20.0, 0.5, 1, "",
                        "叠加在指纹上的高斯噪声强度")
        self._float_row(form, "皮肤纹理强度", "backend.texture_strength", 0.0, 60.0, 1.0, 1, "",
                        "多尺度皮肤纹理幅度")
        self._float_row(form, "边缘羽化", "backend.mask_feather", 0, 80, 1, 0, " px",
                        "手指边界过渡宽度")
        self._float_row(form, "手指占画布比例", "backend.fill_ratio", 0.4, 1.0, 0.02, 2, "×",
                        "越小留白越多；影响边缘羽化与按压平移的余量")

        self.shape_style = QComboBox()
        self.shape_style.addItem("超椭圆指腹（自然，推荐）", "round")
        self.shape_style.addItem("原版 5 参数模型（矩形 + 上下椭圆帽）", "box")
        self.bind_combo("backend.shape_style", self.shape_style, values=["round", "box"],
                        labels=["超椭圆指腹（自然，推荐）", "原版 5 参数模型（矩形 + 上下椭圆帽）"])
        form.addRow("轮廓模型", self.shape_style)
        self._sync_color_enabled()

    def _sync_color_enabled(self) -> None:
        """二值模式下灰度区间不起作用，置灰避免误解。"""
        binary = self.store.config.backend.ridge_only_black
        for widget in self._color_widgets:
            widget.setEnabled(not binary)

    def _build_shape(self) -> None:
        """手指轮廓比例（区间参数）。"""
        _box, form = self.group("手指轮廓比例")
        hint = QLabel(
            "5 参数模型：<b>sL + sR = 手指宽度</b>，<b>sT + sM + sB = 手指高度</b>。"
            "上下弧高按<b>手指宽度</b>的比例给出，因此换分辨率时形态保持不变。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        form.addRow("", hint)

        self._interval_row(form, "宽度占画布", "backend.finger_width_frac", 0.05, 1.0)
        self._interval_row(form, "高度占画布", "backend.finger_height_frac", 0.05, 1.0)
        self._interval_row(form, "左右半宽偏向", "backend.aspect_bias", 0.05, 0.95)
        self._interval_row(form, "顶部弧高/宽度", "backend.top_cap_frac", 0.01, 1.0)
        self._interval_row(form, "底部弧高/宽度", "backend.bottom_cap_frac", 0.01, 1.0)

    def _build_test(self) -> None:
        _box, form = self.group("奇异点与自检")
        self._float_row(form, "奇异点抖动比例", "backend.singular_jitter", 0.0, 0.4, 0.01, 2, "×",
                        "Core / Delta 相对画布的位置抖动幅度")

        row = QHBoxLayout()
        self.test_btn = QPushButton("渲染测试图")
        self.test_btn.setToolTip("用当前参数渲染一张主指纹（不落盘），检验参数是否合理")
        self.test_btn.clicked.connect(self._on_test)
        row.addWidget(self.test_btn)
        self.test_result = QLabel("尚未测试")
        self.test_result.setObjectName("Hint")
        self.test_result.setWordWrap(True)
        row.addWidget(self.test_result, 1)
        form.addRow("", _wrap(row))

    # ------------------------------------------------------------------
    # 控件工厂
    # ------------------------------------------------------------------
    def _int_row(self, form, label: str, path: str, lo: int, hi: int, tip: str = "") -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        if tip:
            spin.setToolTip(tip)
        self.bind_spin(path, spin)
        form.addRow(label, spin)
        return spin

    def _float_row(self, form, label: str, path: str, lo: float, hi: float, step: float,
                   decimals: int, suffix: str, tip: str = "") -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        if suffix:
            spin.setSuffix(" " + suffix)
        if tip:
            spin.setToolTip(tip)
        self.bind_double(path, spin)
        form.addRow(label, spin)
        return spin

    def _interval_row(self, form, label: str, path: str, lo: float, hi: float) -> None:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        low = QDoubleSpinBox()
        high = QDoubleSpinBox()
        for spin in (low, high):
            spin.setRange(lo, hi)
            spin.setDecimals(2)
            spin.setSingleStep(0.02)
        self.bind_interval(path, low, high, decimals=2, step=0.02)
        layout.addWidget(low)
        layout.addWidget(QLabel("~"))
        layout.addWidget(high)
        layout.addStretch(1)
        form.addRow(label, holder)

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------
    def _on_test(self) -> None:
        from ...app import PreviewService

        self.test_btn.setEnabled(False)
        self.test_result.setText("渲染中…")
        try:
            service = PreviewService(self.store.config)
            result = service.render("000001", "left_thumb", "left_loop", seed=1234,
                                    tta_index=0, with_annotation=False)
            if result.error:
                self.test_result.setText(f"✗ {result.error}")
            else:
                h, w = result.image.shape[:2]
                rate = 1.0 / max(result.duration_s, 1e-3)
                self.test_result.setText(
                    f"✓ {w}×{h}　耗时 {result.duration_s:.2f}s　"
                    f"前景 {float((result.mask > 0).mean()):.0%}　"
                    f"（单单元约 {rate:.1f} 张/秒）"
                )
        except Exception as exc:  # pragma: no cover
            self.test_result.setText(f"✗ {type(exc).__name__}: {exc}")
        finally:
            self.test_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # 刷新
    # ------------------------------------------------------------------
    def on_refresh(self) -> None:
        from ...plugins.backends import BackendRegistry

        cfg = self.store.config
        self._sync_color_enabled()
        try:
            cls = BackendRegistry.get(cfg.backend.name)
            self.backend_desc.setText(f"<b>{cls.name}</b> v{cls.version} — {cls.description}")
        except Exception:  # pragma: no cover
            self.backend_desc.setText("（后端信息不可用）")

        size = cfg.backend.image_size
        if size:
            self.width_spin.setValue(int(size[0]))
            self.height_spin.setValue(int(size[1]))
        else:
            w, h = cfg.resolved_image_size()
            self.width_spin.setValue(int(w))
            self.height_spin.setValue(int(h))
        self._sync_resolution_widgets()
        self.update_estimate()
