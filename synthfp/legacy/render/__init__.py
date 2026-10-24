"""虚拟指纹合成内核（原项目验证过的算法，v2.0 保留）。

模块职责（单向依赖，无回环）：

    fingerprint_class  Henry 分类体系与奇异点配置
    shape_model        手指轮廓 5 参数椭圆弧模型
    singular_points    Core / Delta 放置
    orientation_field  Zero-Pole 方向场（弓型走正弦模型）
    frequency_map      多层 Perlin 频率图
    ridge_generator    Gabor 迭代脊线合成
"""

from __future__ import annotations

from .fingerprint_class import (
    FINGERPRINT_CLASSES,
    NATURAL_DISTRIBUTION,
    FingerprintClass,
    SingularPointConfig,
)
from .frequency_map import FrequencyMapGenerator
from .orientation_field import OrientationFieldModel, compute_orientation_field
from .ridge_generator import RidgeGenerator, build_gabor_bank, clear_gabor_cache
from .shape_model import FingerShapeModel, FingerShapeParams
from .singular_points import SingularPoint, SingularPointGenerator

__all__ = [
    "FINGERPRINT_CLASSES",
    "NATURAL_DISTRIBUTION",
    "FingerShapeModel",
    "FingerShapeParams",
    "FingerprintClass",
    "FrequencyMapGenerator",
    "OrientationFieldModel",
    "RidgeGenerator",
    "SingularPoint",
    "SingularPointConfig",
    "SingularPointGenerator",
    "build_gabor_bank",
    "clear_gabor_cache",
    "compute_orientation_field",
]
