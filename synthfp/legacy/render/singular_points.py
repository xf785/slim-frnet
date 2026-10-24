"""
奇异点生成器 — Core和Delta的确定性放置

根据指纹类别确定奇异点数量和位置约束，
在约束区域内随机放置奇异点。
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from .fingerprint_class import (
    FingerprintClass,
    FINGERPRINT_CLASSES,
    SingularPointConfig,
    NATURAL_DISTRIBUTION,
)

# 字符串键到枚举的映射 (避免重复创建)
_STR_TO_CLASS = {
    "arch": FingerprintClass.ARCH,
    "left_loop": FingerprintClass.LEFT_LOOP,
    "right_loop": FingerprintClass.RIGHT_LOOP,
    "whorl": FingerprintClass.WHORL,
    "twin_loop": FingerprintClass.TWIN_LOOP,
}


@dataclass
class SingularPoint:
    """奇异点"""
    x: float
    y: float
    sp_type: str        # 'core' 或 'delta'
    index: float        # Poincaré指数: +0.5 (core), -0.5 (delta)


class SingularPointGenerator:
    """
    奇异点生成器

    根据指纹类别在合理位置生成核心点和三角点。
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select_class(self, distribution: Optional[dict] = None) -> FingerprintClass:
        """按类别分布随机选择指纹类别"""
        if distribution is None:
            classes = list(NATURAL_DISTRIBUTION.keys())
            probs = list(NATURAL_DISTRIBUTION.values())
        else:
            # 将字符串键映射回枚举 (使用模块级常量)
            classes = [_STR_TO_CLASS[k] for k in distribution.keys()]
            probs = list(distribution.values())

        return self.rng.choice(classes, p=probs)

    def generate(
        self,
        fp_class: FingerprintClass,
        image_size: Tuple[int, int],
        jitter: float = 0.1,
    ) -> List[SingularPoint]:
        """
        为给定类别生成奇异点列表

        Args:
            fp_class: 指纹类别
            image_size: (width, height)
            jitter: 位置随机抖动 (相对图像尺寸的比例)

        Returns:
            [(x, y, type, index), ...] 奇异点列表
        """
        w, h = image_size
        config = FINGERPRINT_CLASSES[fp_class]
        points = []

        # 生成Core
        for i in range(config.num_cores):
            if i < len(config.core_positions):
                cx, cy = config.core_positions[i]
            else:
                cx, cy = 0.5, 0.5

            # 添加随机抖动
            x = cx * w + self.rng.uniform(-jitter, jitter) * w
            y = cy * h + self.rng.uniform(-jitter, jitter) * h
            # 约束在有效范围内
            x = np.clip(x, w * 0.1, w * 0.9)
            y = np.clip(y, h * 0.1, h * 0.9)

            points.append(SingularPoint(x=x, y=y, sp_type="core", index=0.5))

        # 生成Delta (y范围略窄于core, 因delta通常更靠近下方)
        for i in range(config.num_deltas):
            if i < len(config.delta_positions):
                dx, dy = config.delta_positions[i]
            else:
                dx, dy = 0.3, 0.6

            x = dx * w + self.rng.uniform(-jitter, jitter) * w
            y = dy * h + self.rng.uniform(-jitter, jitter) * h
            x = np.clip(x, w * 0.1, w * 0.9)
            # Delta y范围: [15%, 85%] (相比core的[10%, 90%]略窄)
            y = np.clip(y, h * 0.15, h * 0.85)

            points.append(SingularPoint(x=x, y=y, sp_type="delta", index=-0.5))

        return points
