"""页面 4：质量与标注配置（设计文档 4.7 / F7 / F8）。"""

from __future__ import annotations

from ..qt import QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget
from .base import BasePage

__all__ = ["QualityPage"]


def _wrap(layout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget


class QualityPage(BasePage):
    """质量阈值 / 细节点 / 标注产物 / 捺印效果。"""

    title = "质量与标注"
    icon = "📐"

    def build(self) -> None:
        self._build_quality()
        self._build_minutiae()
        self._build_annotation()
        self._build_impression()

    # ------------------------------------------------------------------
    def _build_quality(self) -> None:
        _box, form = self.group("质量评估（自研 NFIQ2 风格评分，无需外部依赖）")
        hint = QLabel(
            "综合分 0-100，由五项加权：脊线清晰度 35% + 对比度 20% + 方向一致性 25% "
            "+ 有效面积 10% + 脊线周期合理性 10%。等级 L1（≥80，最好）～ L5（&lt;35，最差）。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        form.addRow("", hint)

        self.quality_enabled = QCheckBox("启用质量评估")
        self.bind_check("quality.enabled", self.quality_enabled)
        form.addRow("", self.quality_enabled)

        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0.0, 100.0)
        self.min_score.setSingleStep(5.0)
        self.min_score.setDecimals(1)
        self.bind_double("quality.min_score", self.min_score)
        form.addRow("最低质量分", self.min_score)

        self.min_area = QDoubleSpinBox()
        self.min_area.setRange(0.0, 1.0)
        self.min_area.setSingleStep(0.05)
        self.min_area.setDecimals(2)
        self.bind_double("quality.min_usable_area", self.min_area)
        form.addRow("最低有效面积占比", self.min_area)

        self.reject = QCheckBox("低于阈值时丢弃该样本")
        self.reject.setToolTip(
            "不勾选：仅把质量分写入 manifest（推荐，便于事后按分数筛选）；\n"
            "勾选：低于阈值的样本标记为 rejected 且不写入图像。"
        )
        self.bind_check("quality.reject_on_fail", self.reject)
        form.addRow("", self.reject)

    # ------------------------------------------------------------------
    def _build_minutiae(self) -> None:
        _box, form = self.group("细节点提取（骨架化 + Crossing Number + 伪点过滤）")
        self.minutiae_enabled = QCheckBox("启用细节点提取")
        self.bind_check("minutiae.enabled", self.minutiae_enabled)
        form.addRow("", self.minutiae_enabled)

        for label, path, lo, hi, step, decimals, tip in (
            ("最小间距", "minutiae.min_distance", 0.0, 64.0, 1.0, 1,
             "距离小于该值的细节点会合并，保留质量更高的一个（像素）"),
            ("边界裁剪边距", "minutiae.border_margin", 0, 64, 1, 0,
             "距图像边界该距离内的细节点一律丢弃（通常是骨架化伪点）"),
            ("单图最大数量", "minutiae.max_count", 1, 5000, 10, 0,
             "超出时按质量降序截断"),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setToolTip(tip)
            self.bind_double(path, spin)
            form.addRow(label, spin)

        self.angle_check = QCheckBox("计算细节点方向")
        self.angle_check.setToolTip("方向来自局部方向场估计，会让细节点带 angle 字段")
        self.bind_check("minutiae.compute_angle", self.angle_check)
        form.addRow("", self.angle_check)

        self.iso_check = QCheckBox("同时导出 ISO/IEC 19794-2 风格模板（.iso）")
        self.iso_check.setToolTip("每个样本额外写一个 24 + 6n 字节的二进制模板")
        self.bind_check("minutiae.export_iso", self.iso_check)
        form.addRow("", self.iso_check)

    # ------------------------------------------------------------------
    def _build_annotation(self) -> None:
        _box, form = self.group("标注产物")
        hint = QLabel(
            "方向场与频率图以 <b>uint8 PNG</b> 存储（体积约为 float32 .npy 的 1%），"
            "方向场还原公式：<code>angle = value / 255 × π</code>；"
            "频率图的量纲记录在同名 <code>_meta.json</code> 里。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        form.addRow("", hint)

        self.mask_check = QCheckBox("手指区域掩模 mask/（{0,255} PNG）")
        self.bind_check("annotation.export_mask", self.mask_check)
        form.addRow("", self.mask_check)

        self.orient_check = QCheckBox("方向场 orientation/（uint8 量化 PNG）")
        self.bind_check("annotation.export_orientation", self.orient_check)
        form.addRow("", self.orient_check)

        self.centerline_check = QCheckBox("中心线 centerline/（距离变换脊线）")
        self.bind_check("annotation.export_centerline", self.centerline_check)
        form.addRow("", self.centerline_check)

        self.qmap_check = QCheckBox("局部质量图 quality/（分块标准差）")
        self.bind_check("annotation.export_quality_map", self.qmap_check)
        form.addRow("", self.qmap_check)

    # ------------------------------------------------------------------
    def _build_impression(self) -> None:
        _box, form = self.group("捺印阶段（每个手指生成多次按压采集）")
        hint = QLabel(
            "一个主指纹可以派生多次捺印：先是随机退化链（干湿 / 污渍 / 断线），"
            "再是皮肤按压变形（旋转 + 平移 + 压力位移场），最后是传感器特性。"
            "主指纹只合成一次并被全部捺印复用，因此增加捺印数的边际成本很低。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        form.addRow("", hint)

        self.deform_check = QCheckBox("启用皮肤按压变形")
        self.bind_check("impression.deform_enabled", self.deform_check)
        form.addRow("", self.deform_check)

        self._interval(form, "按压力度范围", "impression.pressure_range", 0.0, 1.0, 0.05, 2, "",
                       "抽象力度（不是像素）；真实位移由下面的「形变强度折算」决定")
        self._float(form, "形变强度折算", "impression.influence_ratio", 0.0, 0.6, 0.02, 2, "×",
                    "把力度换算为像素位移的比例；320px 短边下 0.20 ⇒ 最大位移约 6px")
        self._interval(form, "旋转角度范围", "impression.rotation_range", -180.0, 180.0, 1.0, 1, "°",
                       "捺印时手指相对采集面的旋转角度")
        self._int(form, "平移抖动", "impression.translation_px", 0, 200, 1, " px",
                  "捺印位置的随机平移幅度")
        self._float(form, "弹性变形尺度", "impression.elastic_sigma", 1.0, 200.0, 1.0, 1, " px")
        self._float(form, "弹性变形强度", "impression.elastic_alpha", 0.0, 60.0, 0.5, 1, " px",
                    "过大将破坏脊线连续性，建议 ≤ 6")

        self.sensor_check = QCheckBox("启用传感器特性模拟")
        self.bind_check("impression.sensor_enabled", self.sensor_check)
        form.addRow("", self.sensor_check)

        from ..qt import QComboBox

        self.sensor_combo = QComboBox()
        for value, label in (
            ("optical", "光学（暗角 + 低对比 + 散斑）"),
            ("capacitive", "电容（高频噪声 + 椒盐 + 行噪声）"),
            ("thermal", "热感（低频漂移 + 条带）"),
            ("none", "不使用"),
        ):
            self.sensor_combo.addItem(label, value)
        self.bind_combo("impression.sensor_type", self.sensor_combo,
                        values=["optical", "capacitive", "thermal", "none"],
                        labels=["光学（暗角 + 低对比 + 散斑）", "电容（高频噪声 + 椒盐 + 行噪声）",
                                "热感（低频漂移 + 条带）", "不使用"])
        form.addRow("传感器类型", self.sensor_combo)

        self._float(form, "传感器强度", "impression.sensor_strength", 0.0, 3.0, 0.1, 2, "×")

        self.stress_check = QCheckBox("启用随机退化链（干湿 / 污渍 / 断线）")
        self.bind_check("impression.stress_enabled", self.stress_check)
        form.addRow("", self.stress_check)

        self._interval(form, "退化算子数范围", "impression.stress_num_operators", 1, 8, 1, 0, "",
                       "每次捺印从下方勾选的算子里随机取这么多个")

        self.stress_boxes: dict = {}
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(2)
        from ...pipeline.stages import STRESS_PRESETS

        for name in sorted(STRESS_PRESETS):
            check = QCheckBox(name)
            check.setToolTip("复用原项目 19 个退化算子中的这一个作为捺印状态扰动")
            check.toggled.connect(lambda checked, n=name: self._on_stress_toggle(n, checked))
            holder_layout.addWidget(check)
            self.stress_boxes[name] = check
        form.addRow("可用退化算子", holder)

    # ------------------------------------------------------------------
    def _float(self, form, label: str, path: str, lo: float, hi: float, step: float,
               decimals: int, suffix: str, tip: str = "") -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix((" " + suffix) if suffix != "°" else "°")
        if tip:
            spin.setToolTip(tip)
        self.bind_double(path, spin)
        form.addRow(label, spin)
        return spin

    def _int(self, form, label: str, path: str, lo: int, hi: int, step: int, suffix: str,
             tip: str = "") -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        if suffix:
            spin.setSuffix(suffix)
        if tip:
            spin.setToolTip(tip)
        self.bind_spin(path, spin)
        form.addRow(label, spin)
        return spin

    def _interval(self, form, label: str, path: str, lo: float, hi: float, step: float,
                  decimals: int, suffix: str, tip: str = "") -> None:
        """区间控件（下界 ~ 上界），绑定到配置里的 ``(下界, 上界)`` 元组。"""
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        low = QDoubleSpinBox()
        high = QDoubleSpinBox()
        for spin in (low, high):
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            if suffix:
                spin.setSuffix((" " + suffix) if suffix != "°" else "°")
            if tip:
                spin.setToolTip(tip)
        self.bind_interval(path, low, high, decimals=decimals, step=step, suffix=suffix)
        layout.addWidget(low)
        layout.addWidget(QLabel("~"))
        layout.addWidget(high)
        layout.addStretch(1)
        form.addRow(label, holder)

    def _on_stress_toggle(self, name: str, checked: bool) -> None:
        if self._updating:
            return
        operators = dict(self.store.config.impression.stress_operators)
        operators[name] = bool(checked)
        self.store.update_flat({"impression.stress_operators": operators})

    # ------------------------------------------------------------------
    def on_refresh(self) -> None:
        operators = self.store.config.impression.stress_operators
        for name, check in getattr(self, "stress_boxes", {}).items():
            check.blockSignals(True)
            check.setChecked(bool(operators.get(name, False)))
            check.blockSignals(False)
