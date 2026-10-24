"""
指纹方向场数学模型 — Zero-Pole (Sherlock-Monro) + Vizcaya-Gerhardt改进

核心思想: 将奇异点(core=极点, delta=零点)建模为复平面上的零点和极点,
方向场由有理函数导出。

数学公式:
  θ(x,y) = θ∞ + ½[ Σ arg(z-cᵢ) - Σ arg(z-dⱼ) ]  (mod π)

其中:
  - cᵢ: core位置 (Poincaré指数 = +1/2)
  - dⱼ: delta位置 (Poincaré指数 = -1/2)
  - θ∞: 全局背景方向

弓型指纹使用正弦模型 (无奇异点):
  Θ(x,y) = β · sin(f·x + g(y))
"""

import numpy as np
from typing import List, Tuple, Optional
from .singular_points import SingularPoint


class OrientationFieldModel:
    """指纹方向场数学模型 (Zero-Pole + Vizcaya-Gerhardt改进)"""

    def __init__(
        self,
        cores: List[Tuple[float, float]],
        deltas: List[Tuple[float, float]],
        bg_angle: float = 0.0,
    ):
        """
        Args:
            cores:  [(x1,y1), (x2,y2), ...] core坐标列表
            deltas: [(x1,y1), (x2,y2), ...] delta坐标列表
            bg_angle: 全局背景方向 (弧度)
        """
        self.cores = cores
        self.deltas = deltas
        self.bg_angle = bg_angle

        # 合并所有奇异点 (core=+1, delta=-1)
        self.all_points = (
            [(x, y, +1) for x, y in cores]
            + [(x, y, -1) for x, y in deltas]
        )

    def compute(self, h: int, w: int) -> np.ndarray:
        """
        计算方向场 θ(x, y) ∈ [0, π)

        Returns:
            orientation: (h, w) float32 ndarray, 值 ∈ [0, π)
        """
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        theta = np.full((h, w), self.bg_angle, dtype=np.float32)

        for x_sp, y_sp, sign in self.all_points:
            # 计算当前点到奇异点的角度
            angle = np.arctan2(yy - y_sp, xx - x_sp)
            theta += sign * 0.5 * angle

        # 归一化到 [0, π)
        theta = np.mod(theta, np.pi)

        return theta.astype(np.float32)

    @classmethod
    def arch_model(
        cls,
        h: int,
        w: int,
        beta: float = 0.5,
        freq: float = 2.0,
    ) -> np.ndarray:
        """
        弓型指纹正弦方向场模型 (无奇异点)

        Θ(x, y) = β · sin(2π·freq·x/w +π·y/h)

        Args:
            h, w: 图像尺寸
            beta: 最大弯曲角度 (弧度)
            freq: 空间频率参数

        Returns:
            orientation: (h, w) float32, ∈ [0, π)
        """
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        x_norm = xx / w
        y_norm = yy / h

        # 弓型: 方向场主要由x位置决定
        theta = beta * np.sin(2 * np.pi * freq * x_norm + np.pi * y_norm)
        theta = np.mod(theta + np.pi / 2, np.pi)

        return theta.astype(np.float32)

    @classmethod
    def from_singular_points(
        cls,
        singular_points: List[SingularPoint],
        image_size: Tuple[int, int],
        bg_angle: float = 0.0,
    ) -> "OrientationFieldModel":
        """从SingularPoint列表构造方向场模型

        Returns:
            OrientationFieldModel 实例 (无奇异点时返回空模型)
        """
        cores = []
        deltas = []
        for sp in singular_points:
            if sp.sp_type == "core":
                cores.append((sp.x, sp.y))
            elif sp.sp_type == "delta":
                deltas.append((sp.x, sp.y))

        # 无奇异点时返回空模型 (调用方应使用arch_model代替)
        return cls(cores=cores, deltas=deltas, bg_angle=bg_angle)


def compute_orientation_field(
    singular_points: List[SingularPoint],
    image_size: Tuple[int, int],
    fp_class: str,
    bg_angle_range: Tuple[float, float] = (-0.3, 0.3),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    便捷函数: 计算完整方向场

    Args:
        singular_points: 奇异点列表
        image_size: (width, height)
        fp_class: 指纹类别字符串
        bg_angle_range: 背景方向范围
        rng: 随机数生成器

    Returns:
        orientation: (H, W) float32, ∈ [0, π)
    """
    if rng is None:
        rng = np.random.default_rng()

    w, h = image_size
    bg_angle = rng.uniform(bg_angle_range[0], bg_angle_range[1])

    if fp_class == "arch" or not singular_points:
        # 弓型: 正弦模型
        beta = rng.uniform(0.3, 0.7)
        freq = rng.uniform(1.5, 2.5)
        return OrientationFieldModel.arch_model(h, w, beta=beta, freq=freq)

    cores = [(sp.x, sp.y) for sp in singular_points if sp.sp_type == "core"]
    deltas = [(sp.x, sp.y) for sp in singular_points if sp.sp_type == "delta"]

    model = OrientationFieldModel(
        cores=cores,
        deltas=deltas,
        bg_angle=bg_angle,
    )
    return model.compute(h, w)
