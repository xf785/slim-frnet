"""
指纹类别参数化

支持的类别及其奇异点配置:
- Arch:       0 cores, 0 deltas (弓型)
- Left Loop:  1 core,  1 delta (左箕型)
- Right Loop: 1 core,  1 delta (右箕型)
- Whorl:      2 cores, 2 deltas (螺型)
- Twin Loop:  2 cores, 2 deltas (双箕型)
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from enum import Enum


class FingerprintClass(Enum):
    ARCH = "arch"
    LEFT_LOOP = "left_loop"
    RIGHT_LOOP = "right_loop"
    WHORL = "whorl"
    TWIN_LOOP = "twin_loop"


@dataclass
class SingularPointConfig:
    """奇异点配置"""
    num_cores: int
    num_deltas: int
    # 奇异点相对位置约束
    core_positions: List[Tuple[float, float]] = field(default_factory=list)
    delta_positions: List[Tuple[float, float]] = field(default_factory=list)


# Henry分类体系的奇异点配置
FINGERPRINT_CLASSES = {
    FingerprintClass.ARCH: SingularPointConfig(
        num_cores=0,
        num_deltas=0,
    ),
    FingerprintClass.LEFT_LOOP: SingularPointConfig(
        num_cores=1,
        num_deltas=1,
        core_positions=[(0.5, 0.4)],     # core在中心偏上
        delta_positions=[(0.2, 0.7)],     # delta在左下
    ),
    FingerprintClass.RIGHT_LOOP: SingularPointConfig(
        num_cores=1,
        num_deltas=1,
        core_positions=[(0.5, 0.4)],     # core在中心偏上
        delta_positions=[(0.8, 0.7)],     # delta在右下
    ),
    FingerprintClass.WHORL: SingularPointConfig(
        num_cores=2,
        num_deltas=2,
        core_positions=[(0.45, 0.45), (0.55, 0.5)],  # 两个core相邻
        delta_positions=[(0.2, 0.55), (0.8, 0.55)],   # delta在两侧
    ),
    FingerprintClass.TWIN_LOOP: SingularPointConfig(
        num_cores=2,
        num_deltas=2,
        core_positions=[(0.35, 0.4), (0.65, 0.4)],   # 两个core水平排列
        delta_positions=[(0.15, 0.6), (0.85, 0.6)],   # delta在下方两侧
    ),
}


# 自然分布 (Henry分类)
NATURAL_DISTRIBUTION = {
    FingerprintClass.ARCH: 0.05,
    FingerprintClass.LEFT_LOOP: 0.35,
    FingerprintClass.RIGHT_LOOP: 0.35,
    FingerprintClass.WHORL: 0.20,
    FingerprintClass.TWIN_LOOP: 0.05,
}
