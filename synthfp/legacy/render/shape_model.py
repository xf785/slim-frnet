"""
手指轮廓模型 — 5参数椭圆弧模型 (精确匹配SFinGe)

SFinGe原始算法:
  手指外形由4段椭圆弧 + 1个矩形中段拼接而成。

  Shape参数:
    sL ──── sR        ← 上半椭圆 (左半径sL, 右半径sR)
  ┌──────────────┐
  │   矩形中段    │ sM ← 中间矩形区域高度
  ├──────────────┤
  │              │
  │  下半椭圆    │ sB ← 下方区域高度
  └──────────────┘

  OpenCV实现:
    ellipse(shape, Point(sL, sT),       Size(sL, sT), 0, 180, 270, 1, -1)   # 左上
    ellipse(shape, Point(sL-1, sT),     Size(sR, sT), 0, 270, 360, 1, -1)   # 右上
    ellipse(shape, Point(sL, sT+sM-1),  Size(sL, sB), 0, 90,  180, 1, -1)   # 左下
    ellipse(shape, Point(sL-1, sT+sM-1),Size(sR, sB), 0, 0,   90,  1, -1)   # 右下
    rectangle(shape, Rect(0, sT, sL+sR, sM), 1, -1)                         # 中段
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2


@dataclass
class FingerShapeParams:
    """手指轮廓5参数模型 (SFinGe标准)

    所有参数单位为像素。
    """
    sL: int     # 左半径 (left radius)
    sR: int     # 右半径 (right radius)
    sT: int     # 顶部弧高度 (top)
    sB: int     # 底部弧高度 (bottom)
    sM: int     # 中间矩形高度 (middle)

    @property
    def image_width(self) -> int:
        return self.sL + self.sR

    @property
    def image_height(self) -> int:
        return self.sT + self.sM + self.sB


class FingerShapeModel:
    """手指轮廓生成器 — 5参数椭圆弧模型 (SFinGe标准)

    使用OpenCV ellipse + rectangle直接绘制，
    与SFinGe DemoOCV/FinGe.cpp中的generate_shape()方法等效。

    Examples:
        >>> model = FingerShapeModel({})
        >>> params = model.sample_params(rng)
        >>> mask = model.generate(params)
    """

    def __init__(self, config: dict):
        self.config = config

    def sample_params(self, rng: np.random.Generator) -> FingerShapeParams:
        """
        从配置参数空间中随机采样手指形状参数。

        采样策略:
        - 手指宽度范围: [width_min, width_max] (默认300-500)
        - 手指高度范围: [height_min, height_max] (默认400-700)
        - 顶部弧比例: [top_curve_min, top_curve_max] (默认0.1-0.5)
        - 底部弧比例: [bottom_curve_min, bottom_curve_max] (默认0.05-0.3)
        """
        w_range = self.config.get("width_range", [300, 500])
        h_range = self.config.get("height_range", [400, 700])
        tc_range = self.config.get("top_curve_range", [0.1, 0.5])
        bc_range = self.config.get("bottom_curve_range", [0.05, 0.3])
        m_ratio_range = self.config.get("middle_ratio_range", [0.3, 0.7])

        # 手指不对称: 左侧稍宽模拟拇指
        asymmetry = rng.uniform(0.30, 0.55)

        finger_width = int(rng.uniform(w_range[0], w_range[1]))
        finger_height = int(rng.uniform(h_range[0], h_range[1]))
        top_curve = rng.uniform(tc_range[0], tc_range[1])
        bottom_curve = rng.uniform(bc_range[0], bc_range[1])
        middle_ratio = rng.uniform(m_ratio_range[0], m_ratio_range[1])

        # 5参数计算
        sL = int(finger_width * asymmetry)
        sR = finger_width - sL
        sT = int(finger_height * top_curve * 0.5)  # 顶部弧高度
        sB = int(finger_height * bottom_curve * 0.5)  # 底部弧高度
        sM = int(finger_height * middle_ratio)

        # 确保所有参数为正
        sL = max(sL, 20)
        sR = max(sR, 20)
        sT = max(sT, 10)
        sB = max(sB, 10)
        sM = max(sM, 50)

        return FingerShapeParams(sL=sL, sR=sR, sT=sT, sB=sB, sM=sM)

    def generate(
        self,
        params: FingerShapeParams,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        生成手指轮廓二值掩模 (SFinGe精确算法)

        使用OpenCV直接绘制4段椭圆弧 + 矩形中段。

        Args:
            params: 手指形状5参数
            image_size: 输出图像尺寸 (width, height)，None=使用params计算

        Returns:
            mask: 二值掩模 (H, W), dtype=uint8, {0, 255}
        """
        if image_size is None:
            img_w = params.image_width
            img_h = params.image_height
        else:
            img_w, img_h = image_size

        # 创建画布
        shape = np.zeros((img_h, img_w), dtype=np.uint8)

        # 将SFinGe参数映射到图像中心
        # 计算缩放因子 (如果image_size与params自然尺寸不同)
        natural_w = params.sL + params.sR
        natural_h = params.sT + params.sM + params.sB
        scale_w = img_w / max(natural_w, 1)
        scale_h = img_h / max(natural_h, 1)

        sL = int(params.sL * scale_w)
        sR = int(params.sR * scale_w)
        sT = int(params.sT * scale_h)
        sB = int(params.sB * scale_h)
        sM = int(params.sM * scale_h)

        # 确保不超出图像边界
        sL = min(sL, img_w - 1)
        sR = min(sR, img_w - sL)
        sT = min(sT, img_h - 1)
        sB = min(sB, img_h - sT - 1)
        sM = max(min(sM, img_h - sT - sB), 10)

        # SFinGe精确算法: 4段椭圆弧 + 矩形中段
        # 左上椭圆弧 (180°→270°)
        cv2.ellipse(
            shape, (sL, sT), (sL, sT), 0,
            180, 270, 255, -1, cv2.LINE_AA,
        )
        # 右上椭圆弧 (270°→360°)
        cv2.ellipse(
            shape, (sL - 1, sT), (sR, sT), 0,
            270, 360, 255, -1, cv2.LINE_AA,
        )
        # 左下椭圆弧 (90°→180°)
        cv2.ellipse(
            shape, (sL, sT + sM - 1), (sL, sB), 0,
            90, 180, 255, -1, cv2.LINE_AA,
        )
        # 右下椭圆弧 (0°→90°)
        cv2.ellipse(
            shape, (sL - 1, sT + sM - 1), (sR, sB), 0,
            0, 90, 255, -1, cv2.LINE_AA,
        )
        # 矩形中段
        cv2.rectangle(
            shape, (0, sT), (sL + sR, sT + sM), 255, -1,
        )

        # 二值化: 抗锯齿可能产生中间值, 统一转为{0,255}
        shape = (shape > 128).astype(np.uint8) * 255
        return shape
