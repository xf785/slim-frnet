"""配置系统 —— Pydantic Schema（单一事实来源）。

设计文档 4.1 节：GUI 与 CLI 共用同一 schema，支持 YAML 加载/保存与预设管理。

所有可调参数都在这里定义，GUI 控件直接读写这些字段，
因此「配置面板」与「配置文件」天然一致，不会出现两套参数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: Henry 分类体系（SFinGe 使用的 5 类）
FINGERPRINT_CLASSES: Tuple[str, ...] = (
    "arch",
    "left_loop",
    "right_loop",
    "whorl",
    "twin_loop",
)

CLASS_LABELS_ZH: Dict[str, str] = {
    "arch": "弓型 Arch",
    "left_loop": "左箕型 Left Loop",
    "right_loop": "右箕型 Right Loop",
    "whorl": "螺型 Whorl",
    "twin_loop": "双箕型 Twin Loop",
}

#: 与 500 ppi 采集设备对齐的推荐分辨率
RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "finger_500dpi": (320, 480),
    "finger_1000dpi": (480, 720),
    "square_512": (512, 512),
    "thumb_flat": (400, 640),
    "custom": (320, 480),
}

#: 10 指命名（ISO/IEC 19794-4 约定）
FINGER_NAMES: Tuple[str, ...] = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_little",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_little",
)

FINGER_LABELS_ZH: Dict[str, str] = {
    "left_thumb": "左拇指",
    "left_index": "左食指",
    "left_middle": "左中指",
    "left_ring": "左环指",
    "left_little": "左小指",
    "right_thumb": "右拇指",
    "right_index": "右食指",
    "right_middle": "右中指",
    "right_ring": "右环指",
    "right_little": "右小指",
}

_Base = ConfigDict(validate_assignment=True, extra="forbid", arbitrary_types_allowed=False)


class _Section(BaseModel):
    """所有配置段的基类：赋值即校验，禁止未知字段。"""

    model_config = _Base


def _check_interval(
    value: Tuple[float, float], name: str, lo: float = 0.0, hi: float = float("inf")
) -> Tuple[float, float]:
    """校验 ``(下界, 上界)`` 形式的区间字段（保留元素原始类型）。

    Pydantic 的 ``gt``/``le`` 只能作用于标量，作用在元组上会直接抛
    ``TypeError``，因此所有区间字段的边界都在这里手工校验。
    """
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} 必须是 (下界, 上界) 二元组，收到 {value!r}")
    a, b = value[0], value[1]
    if float(a) > float(b):
        raise ValueError(f"{name} 下界 {a} 不能大于上界 {b}")
    if float(a) < lo or float(b) > hi:
        raise ValueError(f"{name} 必须落在 [{lo}, {hi}] 内，收到 ({a}, {b})")
    return (a, b)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 1. 数据集规模
# ---------------------------------------------------------------------------
class DatasetSection(_Section):
    """身份数 / 手指数 / 捺印数 / TTA 数 —— 设计文档 F1。"""

    name: str = Field(default="synth_fp_v1", description="数据集名称（同时作为输出子目录名）")
    output_root: Path = Field(default=Path("outputs"), description="输出根目录")
    identities: int = Field(default=10, ge=1, le=100_000, description="身份数")
    fingers_per_identity: int = Field(default=2, ge=1, le=10, description="每身份手指数")
    impressions_per_finger: int = Field(default=2, ge=1, le=50, description="每手指捺印数")
    tta_per_impression: int = Field(default=3, ge=0, le=500, description="每捺印 TTA 变换数")

    @property
    def total_masters(self) -> int:
        return self.identities * self.fingers_per_identity

    @property
    def total_impressions(self) -> int:
        return self.total_masters * self.impressions_per_finger

    @property
    def total_samples(self) -> int:
        return self.total_impressions * max(self.tta_per_impression, 1)

    def summary(self) -> str:
        return (
            f"{self.identities} 身份 × {self.fingers_per_identity} 指 "
            f"× {self.impressions_per_finger} 捺印 × {self.tta_per_impression} TTA "
            f"= {self.total_samples} 样本"
        )


# ---------------------------------------------------------------------------
# 2. 类别分布 —— 设计文档 F2
# ---------------------------------------------------------------------------
class ClassDistributionSection(_Section):
    """指纹类别比例（Henry 分类）。"""

    distribution: Dict[str, float] = Field(
        default_factory=lambda: {
            "arch": 0.05,
            "left_loop": 0.35,
            "right_loop": 0.35,
            "whorl": 0.20,
            "twin_loop": 0.05,
        },
        description="类别 -> 比例，和必须为 1.0",
    )

    @field_validator("distribution")
    @classmethod
    def _check(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("类别分布不能为空")
        unknown = set(v) - set(FINGERPRINT_CLASSES)
        if unknown:
            raise ValueError(f"未知指纹类别: {sorted(unknown)}；可用: {list(FINGERPRINT_CLASSES)}")
        if any(w < 0 for w in v.values()):
            raise ValueError("类别比例不能为负")
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"类别比例之和必须为 1.0，当前为 {total:.4f}")
        return {k: float(w) for k, w in v.items() if w > 0}

    def weights(self) -> Tuple[List[str], List[float]]:
        names = list(self.distribution)
        return names, [self.distribution[n] for n in names]

    def summary(self) -> str:
        return " ".join(f"{CLASS_LABELS_ZH.get(k, k)}={v:.0%}" for k, v in self.distribution.items())


# ---------------------------------------------------------------------------
# 3. 后端 —— 设计文档 F3
# ---------------------------------------------------------------------------
class BackendSection(_Section):
    """生成后端（当前为 SFinGe 复刻内核，可插拔）。"""

    name: Literal["sfinge", "dummy"] = Field(default="sfinge", description="后端标识")
    resolution: str = Field(default="finger_500dpi", description="分辨率预设名")
    image_size: Optional[Tuple[int, int]] = Field(
        default=None,
        description="自定义图像尺寸 (宽, 高)；为 None 时使用分辨率预设",
    )
    ppi: int = Field(default=500, ge=100, le=4000, description="采集分辨率标注 (ppi)")

    # --- 手指轮廓 5 参数模型（按画布比例表达，随分辨率自适应） ---
    # 5 参数模型：sL/sR 为左右半宽，sT/sB 为上下弧高，sM 为中段高度。
    #   sL + sR = 手指宽度，sT + sM + sB = 手指高度
    # 用比例而非像素表达，保证换分辨率时形态不变。
    finger_width_frac: Tuple[float, float] = Field(
        default=(0.62, 0.72), description="手指宽度占画布宽度比例 (0-1)"
    )
    finger_height_frac: Tuple[float, float] = Field(
        default=(0.88, 0.95), description="手指高度占画布高度比例 (0-1)"
    )
    aspect_bias: Tuple[float, float] = Field(
        default=(0.46, 0.54), description="左右半宽分配比例（>0.5 表示左半更宽）"
    )
    top_cap_frac: Tuple[float, float] = Field(
        default=(0.24, 0.38), description="顶部弧高占手指宽度比例（越大顶部越圆）"
    )
    bottom_cap_frac: Tuple[float, float] = Field(
        default=(0.18, 0.30), description="底部弧高占手指宽度比例（越大底部越圆）"
    )

    # --- 脊线（Gabor 迭代） ---
    min_ridge_width: int = Field(default=5, ge=2, le=40, description="最窄脊线宽度 (px)")
    max_ridge_width: int = Field(default=15, ge=3, le=60, description="最宽脊线宽度 (px)")
    gabor_filter_size: int = Field(default=10, ge=4, le=24, description="Gabor 核半径")
    gabor_n_frequencies: int = Field(
        default=5, ge=2, le=20, description="Gabor 频率档数（吞吐第一杠杆，5 档足够）"
    )
    coarse_scale: float = Field(
        default=0.7, ge=0.3, le=1.0, description="迭代分辨率比例，1.0 = 最锐利但最慢"
    )
    max_iterations: int = Field(default=80, ge=10, le=400, description="迭代上限")
    convergence_eps: float = Field(default=1e-3, gt=0, le=0.1, description="收敛阈值")
    seed_density: float = Field(default=1e-3, gt=0, le=0.05, description="初始种子密度")
    freq_spread: float = Field(
        default=0.25, ge=0.05, le=1.0, description="频率图相对脊宽范围的波动比例"
    )

    # --- 外观 ---
    ridge_gray: Tuple[int, int] = Field(default=(15, 55), description="脊线灰度范围")
    valley_gray: Tuple[int, int] = Field(default=(225, 252), description="谷线灰度范围")
    ridge_only_black: bool = Field(
        default=False,
        description=(
            "仅脊线显示为黑色：输出纯二值图（脊线 0、谷线 255），"
            "跳过皮肤纹理、底噪与整体亮度拉伸，适合做分割/细化算法的干净输入"
        ),
    )
    mask_feather: int = Field(default=12, ge=0, le=80, description="边缘羽化宽度 (px)")
    noise_sigma: float = Field(default=1.5, ge=0, le=20, description="底噪标准差")
    texture_strength: float = Field(default=10.0, ge=0, le=60, description="皮肤纹理强度")
    fill_ratio: float = Field(
        default=0.88, ge=0.4, le=1.0, description="手指占画布比例（留白避免裁切羽化边）"
    )
    shape_style: Literal["round", "box"] = Field(
        default="round",
        description="轮廓模型：round=超椭圆指腹轮廓（自然）；box=原版 5 参数矩形+椭圆帽",
    )

    # --- 奇异点 ---
    singular_jitter: float = Field(default=0.13, ge=0, le=0.4, description="奇异点抖动比例")

    @field_validator("gabor_filter_size")
    @classmethod
    def _even_kernel_check(cls, v: int) -> int:
        # 核半径决定 (2r+1)×(2r+1) 的核，半径 >= 4 才有意义
        return int(v)

    @model_validator(mode="after")
    def _check_ranges(self) -> "BackendSection":
        """区间字段的统一校验（区间不能倒挂、比例必须落在 (0, 1]）。"""
        _check_interval(self.finger_width_frac, "finger_width_frac", lo=0.05, hi=1.0)
        _check_interval(self.finger_height_frac, "finger_height_frac", lo=0.05, hi=1.0)
        _check_interval(self.aspect_bias, "aspect_bias", lo=0.05, hi=0.95)
        _check_interval(self.top_cap_frac, "top_cap_frac", lo=0.01, hi=1.0)
        _check_interval(self.bottom_cap_frac, "bottom_cap_frac", lo=0.01, hi=1.0)
        return self

    @model_validator(mode="after")
    def _check_width(self) -> "BackendSection":
        if self.min_ridge_width >= self.max_ridge_width:
            raise ValueError(
                f"最窄脊宽 ({self.min_ridge_width}) 必须小于最宽脊宽 ({self.max_ridge_width})"
            )
        if self.ridge_gray[0] >= self.ridge_gray[1] or self.valley_gray[0] >= self.valley_gray[1]:
            raise ValueError("灰度范围必须为 (最小值, 最大值) 且最小 < 最大")
        if self.resolution not in RESOLUTION_PRESETS:
            raise ValueError(
                f"未知分辨率预设 '{self.resolution}'；可用: {list(RESOLUTION_PRESETS)}"
            )
        return self

    def resolved_image_size(self) -> Tuple[int, int]:
        if self.image_size is not None:
            w, h = self.image_size
            return int(w), int(h)
        return RESOLUTION_PRESETS[self.resolution]


# ---------------------------------------------------------------------------
# 4. 捺印阶段（皮肤变形 + 传感器） —— 生成主指纹 -> 多次捺印
# ---------------------------------------------------------------------------
class ImpressionSection(_Section):
    """每个手指生成多次捺印，模拟按压变形与传感器差异。"""

    deform_enabled: bool = Field(default=True, description="启用皮肤按压变形")
    pressure_range: Tuple[float, float] = Field(
        default=(0.30, 0.70), description="按压力度（0-1 的抽象力度，不是像素）"
    )
    influence_ratio: float = Field(
        default=0.20,
        gt=0.0,
        le=0.6,
        description="按压力度到形变幅度的折算比例；320px 短边下 0.20 ⇒ 最大位移约 2% 短边",
    )
    rotation_range: Tuple[float, float] = Field(
        default=(-8.0, 8.0), description="捺印旋转角度范围 (度)"
    )
    translation_px: int = Field(default=8, ge=0, le=200, description="平移抖动 (px)")
    elastic_sigma: float = Field(default=30.0, gt=0, le=200, description="弹性变形尺度")
    elastic_alpha: float = Field(
        default=3.0, ge=0, le=60, description="弹性变形强度 (px)；过大将破坏脊线连续性"
    )

    sensor_enabled: bool = Field(default=True, description="启用传感器特性模拟")
    sensor_type: Literal["optical", "capacitive", "thermal", "none"] = Field(default="optical")
    sensor_strength: float = Field(default=1.0, ge=0, le=3, description="传感器效果强度倍数")

    stress_enabled: bool = Field(default=True, description="启用随机退化算子链（干湿/污渍/断线）")
    stress_num_operators: Tuple[int, int] = Field(default=(1, 2), description="随机算子数量范围")
    stress_operators: Dict[str, bool] = Field(
        default_factory=lambda: {
            "ridge_breaks": True,
            "dirt_stain": True,
            "ink_smudge": True,
            "blemish_spot": True,
            "ridge_edge_roughness": True,
            "ghosting": False,
            "partial_missing": False,
        },
        description="可用的捺印退化算子开关",
    )

    @field_validator("pressure_range")
    @classmethod
    def _check_pressure(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        return _check_interval(v, "pressure_range", lo=0.0, hi=1.0)

    @field_validator("rotation_range")
    @classmethod
    def _check_rotation(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        # 旋转允许双向：(-8, 8) 表示 ±8°
        return _check_interval(v, "rotation_range", lo=-180.0, hi=180.0)

    @field_validator("stress_num_operators")
    @classmethod
    def _check_stress_count(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        return _check_interval(v, "stress_num_operators", lo=1, hi=19)

    def summary(self) -> str:
        parts = []
        if self.deform_enabled:
            parts.append(f"按压变形 {self.pressure_range[0]:.2f}~{self.pressure_range[1]:.2f}")
        if self.sensor_enabled and self.sensor_type != "none":
            parts.append(f"传感器={self.sensor_type}×{self.sensor_strength:g}")
        if self.stress_enabled:
            n_on = sum(1 for on in self.stress_operators.values() if on)
            parts.append(f"退化算子 {n_on} 选 {self.stress_num_operators[0]}~{self.stress_num_operators[1]}")
        return " | ".join(parts) if parts else "无"

# ---------------------------------------------------------------------------
# 5. TTA —— 设计文档 F4
# ---------------------------------------------------------------------------
#: 每个算子的参数滑块定义：(显示名, 最小, 最大, 步长, 单位)
TTA_PARAM_SPECS: Dict[str, Dict[str, Tuple[str, float, float, float, str]]] = {
    "rotate": {"angle": ("角度绝对值上限", 0, 180, 1, "°")},
    "translate": {"shift": ("平移比例上限", 0.0, 0.5, 0.01, "×尺寸")},
    "scale": {"factor": ("缩放范围 ±", 0.0, 0.6, 0.01, "×")},
    "shear": {"amount": ("错切绝对值上限", 0.0, 0.6, 0.01, "×")},
    "elastic": {"alpha": ("形变强度", 0.0, 40, 1, "px"), "sigma": ("形变尺度", 4, 120, 1, "px")},
    "perspective": {"amount": ("透视强度上限", 0.0, 0.25, 0.005, "×")},
    "brightness": {"delta": ("亮度偏移 ±", 0.0, 0.6, 0.01, "×")},
    "contrast": {"factor": ("对比度范围 ±", 0.0, 0.8, 0.01, "×")},
    "gamma": {"gamma": ("Gamma 范围", 0.2, 3.0, 0.05, "×")},
    "blur": {"sigma": ("模糊强度上限", 0.0, 4.0, 0.1, "px")},
    "motion_blur": {"length": ("运动长度上限", 1, 25, 1, "px")},
    "gaussian_noise": {"sigma": ("噪声标准差上限", 0.0, 0.15, 0.005, "×")},
    "poisson": {"scale": ("泊松强度", 10, 400, 10, "×")},
    "salt_pepper": {"amount": ("噪声比例上限", 0.0, 0.15, 0.005, "×")},
    "jpeg": {"quality": ("JPEG 质量下界", 5, 100, 1, "")},
    "occlusion_rect": {"count": ("遮挡块数量上限", 1, 8, 1, "块"), "area": ("单块面积比例上限", 0.01, 0.3, 0.01, "×")},
    "occlusion_ellipse": {"count": ("遮挡椭圆数量上限", 1, 8, 1, "个"), "area": ("单个面积比例上限", 0.01, 0.3, 0.01, "×")},
    "scratch": {"count": ("划痕数量上限", 1, 12, 1, "条"), "width": ("划痕宽度上限", 1, 12, 1, "px")},
    "sensor_optical": {"strength": ("强度倍数", 0.0, 3.0, 0.1, "×")},
    "sensor_capacitive": {"strength": ("强度倍数", 0.0, 3.0, 0.1, "×")},
    "sensor_thermal": {"strength": ("强度倍数", 0.0, 3.0, 0.1, "×")},
}

#: 算子中文名与分类（GUI 勾选列表用）
TTA_CATALOG: Tuple[Tuple[str, str, str], ...] = (
    # (分类, 算子名, 中文名)
    ("几何", "rotate", "旋转"),
    ("几何", "translate", "平移"),
    ("几何", "scale", "缩放"),
    ("几何", "shear", "错切"),
    ("几何", "elastic", "弹性形变"),
    ("几何", "perspective", "透视"),
    ("光度", "brightness", "亮度"),
    ("光度", "contrast", "对比度"),
    ("光度", "gamma", "Gamma"),
    ("光度", "blur", "高斯模糊"),
    ("光度", "motion_blur", "运动模糊"),
    ("噪声", "gaussian_noise", "高斯噪声"),
    ("噪声", "poisson", "泊松噪声"),
    ("噪声", "salt_pepper", "椒盐噪声"),
    ("压缩", "jpeg", "JPEG 压缩"),
    ("遮挡", "occlusion_rect", "矩形遮挡"),
    ("遮挡", "occlusion_ellipse", "椭圆遮挡"),
    ("遮挡", "scratch", "划痕遮挡"),
    ("传感器", "sensor_optical", "光学传感器"),
    ("传感器", "sensor_capacitive", "电容传感器"),
    ("传感器", "sensor_thermal", "热感传感器"),
)


class TTASection(_Section):
    """TTA 引擎配置：勾选算子 + 参数范围 + 组合策略。"""

    enabled: bool = Field(default=True, description="是否启用 TTA 阶段")
    mode: Literal["single", "chain", "mixed"] = Field(
        default="mixed",
        description="single=每个 TTA 只取 1 个算子；chain=每个 TTA 组合多个；mixed=随机",
    )
    chain_length: Tuple[int, int] = Field(default=(1, 2), description="chain 模式下的算子数量")
    chain_probability: float = Field(default=0.35, ge=0, le=1, description="mixed 模式下走 chain 的概率")
    max_distortion: float = Field(
        default=0.75, ge=0.05, le=1.0, description="最大累计失真度（超过则截断算子链）"
    )
    operators: List[str] = Field(
        default_factory=lambda: [
            "rotate",
            "scale",
            "brightness",
            "contrast",
            "blur",
            "gaussian_noise",
        ],
        description="启用的算子名列表（顺序即 GUI 展示顺序，保存/加载幂等）",
    )
    params: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="算子 -> 参数范围覆盖，例如 {'rotate': {'angle': 8.0}}",
    )

    @field_validator("operators")
    @classmethod
    def _known(cls, v: List[str]) -> List[str]:
        """校验算子名合法并去重，**保持用户给定的顺序**。

        刻意不做「按目录顺序重排」：那会让「保存 → 加载」改变字段值，
        破坏配置文件幂等性（也会让配置指纹无谓地变化）。
        目录顺序只用于 GUI 展示，不用于规范化存储。
        """
        known = {name for _, name, _ in TTA_CATALOG}
        unknown = [n for n in v if n not in known]
        if unknown:
            raise ValueError(f"未知 TTA 算子: {unknown}")
        ordered: List[str] = []
        for name in v:
            if name not in ordered:
                ordered.append(name)
        return ordered

    @field_validator("params")
    @classmethod
    def _params_known(cls, v: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        known = {name for _, name, _ in TTA_CATALOG}
        unknown = [n for n in v if n not in known]
        if unknown:
            raise ValueError(f"未知 TTA 算子的参数: {unknown}")
        return v

    @model_validator(mode="after")
    def _check_chain(self) -> "TTASection":
        _check_interval(self.chain_length, "chain_length", lo=1, hi=10)
        if self.enabled and not self.operators:
            raise ValueError("启用 TTA 时至少需要勾选一个算子")
        return self

    def params_for(self, operator: str) -> Dict[str, Any]:
        return dict(self.params.get(operator, {}))

    def summary(self) -> str:
        if not self.enabled:
            return "已关闭"
        labels = {n: zh for _, n, zh in TTA_CATALOG}
        names = "、".join(labels.get(n, n) for n in self.operators[:5])
        more = f" 等 {len(self.operators)} 个" if len(self.operators) > 5 else ""
        return f"{self.mode} | {names}{more}"


# ---------------------------------------------------------------------------
# 6. 质量 与 细节点 —— 设计文档 4.7
# ---------------------------------------------------------------------------
class QualitySection(_Section):
    """质量评估与阈值过滤。"""

    enabled: bool = Field(default=True, description="启用质量评估")
    min_score: float = Field(default=0.0, ge=0, le=100, description="质量阈值，低于则丢弃")
    min_usable_area: float = Field(default=0.0, ge=0, le=1, description="最小有效面积占比")
    reject_on_fail: bool = Field(default=False, description="低于阈值时丢弃样本（否则仅标注）")

    def summary(self) -> str:
        if not self.enabled:
            return "已关闭"
        return f"阈值 {self.min_score:.0f}/100" + ("（超限丢弃）" if self.reject_on_fail else "（仅标注）")


class MinutiaeSection(_Section):
    """细节点提取与模板导出。"""

    enabled: bool = True
    min_distance: float = Field(
        default=16.0,
        ge=0,
        le=64,
        description=(
            "细节点间最小距离 (px)。500 dpi 下脊线周期约 10px，"
            "取 16px ≈ 1.5 个周期，可去掉同一处脊线事件的重复检出；"
            "调小会显著增加细节点数"
        ),
    )
    border_margin: int = Field(default=6, ge=0, le=64, description="边界裁剪边距 (px)")
    max_count: int = Field(default=200, ge=1, le=5000, description="单图最多输出细节点数")
    export_iso: bool = Field(default=False, description="导出 ISO/IEC 19794-2 风格模板")
    compute_angle: bool = Field(default=True, description="计算细节点方向（依赖方向场估计）")

    def summary(self) -> str:
        return f"最小间距 {self.min_distance:.0f}px" + ("，ISO模板" if self.export_iso else "")


class AnnotationSection(_Section):
    """标注产物（掩模 / 方向场 / 中心线）输出开关。"""

    export_mask: bool = True
    export_orientation: bool = True
    export_centerline: bool = False
    export_quality_map: bool = False

    def summary(self) -> str:
        on = [
            n
            for n, v in (
                ("掩模", self.export_mask),
                ("方向场", self.export_orientation),
                ("中心线", self.export_centerline),
                ("质量图", self.export_quality_map),
            )
            if v
        ]
        return "、".join(on) if on else "仅图像"


# ---------------------------------------------------------------------------
# 7. 划分 —— 设计文档 F9
# ---------------------------------------------------------------------------
class SplitSection(_Section):
    """数据集划分（默认按身份，防泄漏）。"""

    enabled: bool = True
    strategy: Literal["by_identity", "by_finger", "random"] = Field(
        default="by_identity", description="划分维度；by_identity 可防止身份泄漏"
    )
    ratios: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    write_csv: bool = True

    @field_validator("ratios")
    @classmethod
    def _check(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("划分比例不能为空")
        unknown = set(v) - {"train", "val", "test"}
        if unknown:
            raise ValueError(f"未知划分: {sorted(unknown)}")
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"划分比例之和必须为 1.0，当前为 {total:.4f}")
        return {k: float(w) for k, w in v.items()}

    def summary(self) -> str:
        if not self.enabled:
            return "不划分"
        return f"{self.strategy} " + "/".join(f"{k} {v:.0%}" for k, v in self.ratios.items())


# ---------------------------------------------------------------------------
# 8. 运行 —— 设计文档 4.3
# ---------------------------------------------------------------------------
class RuntimeSection(_Section):
    """调度与可靠性。"""

    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    workers: int = Field(default=4, ge=1, le=64, description="并行工作进程数")
    retries: int = Field(default=2, ge=0, le=10, description="单个样本失败重试次数")
    timeout_s: float = Field(default=300.0, gt=0, le=3600, description="单样本超时（秒）")
    resume: bool = True
    keep_failed: bool = True
    cache_masters: bool = Field(default=True, description="复用已生成的主指纹")
    log_dir: Path = Field(default=Path("outputs/logs"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def summary(self) -> str:
        return (
            f"seed={self.seed} | {self.workers} 并行单元 | 重试 {self.retries} 次"
            + (" | 断点续传" if self.resume else "")
        )


# ---------------------------------------------------------------------------
# 9. 导出 —— 设计文档 F10
# ---------------------------------------------------------------------------
class ExportSection(_Section):
    """导出与打包。"""

    write_sha256: bool = True
    write_readme: bool = True
    write_environment: bool = True
    write_stats: bool = True
    image_format: Literal["png", "tiff"] = "png"
    png_compression: int = Field(default=3, ge=0, le=9)
    copy_images_into_split: bool = Field(
        default=False, description="把图像复制进 splits/<split>/ 结构（会产生冗余副本）"
    )

    def summary(self) -> str:
        on = [n for n, v in (("校验和", self.write_sha256), ("README", self.write_readme),
                            ("统计报告", self.write_stats), ("环境快照", self.write_environment)) if v]
        return f"{self.image_format.upper()} | " + "、".join(on)


# ---------------------------------------------------------------------------
# 顶层配置
# ---------------------------------------------------------------------------
class AppConfig(_Section):
    """顶层配置对象 —— 与设计文档 ``DatasetConfig`` 对应并做了完整展开。"""

    schema_version: str = Field(default="2.0")
    dataset: DatasetSection = Field(default_factory=DatasetSection)
    classes: ClassDistributionSection = Field(default_factory=ClassDistributionSection)
    backend: BackendSection = Field(default_factory=BackendSection)
    impression: ImpressionSection = Field(default_factory=ImpressionSection)
    tta: TTASection = Field(default_factory=TTASection)
    quality: QualitySection = Field(default_factory=QualitySection)
    minutiae: MinutiaeSection = Field(default_factory=MinutiaeSection)
    annotation: AnnotationSection = Field(default_factory=AnnotationSection)
    split: SplitSection = Field(default_factory=SplitSection)
    runtime: RuntimeSection = Field(default_factory=RuntimeSection)
    export: ExportSection = Field(default_factory=ExportSection)

    # -- 便捷属性 ---------------------------------------------------------
    @property
    def dataset_dir(self) -> Path:
        return Path(self.dataset.output_root) / self.dataset.name

    def resolved_image_size(self) -> Tuple[int, int]:
        return self.backend.resolved_image_size()

    def estimate(self) -> Dict[str, int]:
        """估算工作量：主指纹数 / 捺印数 / 样本数（含 TTA）。"""
        d = self.dataset
        return {
            "masters": d.total_masters,
            "impressions": d.total_impressions,
            "samples": d.total_samples,
        }

    def summary_lines(self) -> List[str]:
        return [
            f"数据集      : {self.dataset.name}",
            f"规模        : {self.dataset.summary()}",
            f"输出        : {self.dataset_dir}",
            f"类别分布    : {self.classes.summary()}",
            f"分辨率      : {self.resolved_image_size()[0]}×{self.resolved_image_size()[1]} @ {self.backend.ppi}ppi",
            f"后端        : {self.backend.name}",
            f"捺印        : {self.impression.summary()}",
            f"TTA         : {self.tta.summary()}",
            f"质量        : {self.quality.summary()}",
            f"细节点      : {self.minutiae.summary()}",
            f"划分        : {self.split.summary()}",
            f"运行        : {self.runtime.summary()}",
            f"导出        : {self.export.summary()}",
        ]
