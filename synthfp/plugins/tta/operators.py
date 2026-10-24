"""TTA 算子实现 —— 契约 §1.3 规定的全部 21 个算子。

分类与失真度
------------
===========  ==========================================
分类          算子
===========  ==========================================
几何          rotate / translate / scale / shear / elastic / perspective
光度          brightness / contrast / gamma / blur / motion_blur
噪声          gaussian_noise / poisson / salt_pepper
压缩          jpeg
遮挡          occlusion_rect / occlusion_ellipse / scratch
传感器        sensor_optical / sensor_capacitive / sensor_thermal
===========  ==========================================

统一约定（详见 :mod:`synthfp.plugins.tta.base`）：
输入 ``(H, W)`` 灰度图（``float32 [0, 1]``，背景=1.0；同时兼容 ``uint8``），
输出形状与 dtype 不变、值域安全、绝不修改输入、同参数必定同结果。

几何算子一律用 ``cv2.warpAffine`` / ``cv2.warpPerspective`` / ``cv2.remap``，
边界填充 ``cv2.BORDER_REPLICATE``，以免白背景指纹出现黑边。遮挡算子把遮挡区域
填充为图像背景（高分位灰度，默认 ``1.0``）并轻度羽化，绝不填 0。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from .base import TTAOperator, background_level, gaussian_blur, random_field
from .registry import TTARegistry

__all__ = ["OPERATORS", "create_operators"]


def _warp_affine(work: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """用 2x3 仿射矩阵变换图像（尺寸不变，边缘 replicate 外推）。"""
    height, width = work.shape[:2]
    return cv2.warpAffine(
        work,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ===========================================================================
# 几何
# ===========================================================================
@TTARegistry.register(
    "rotate", "旋转", "几何", 0.25, {"angle": ("角度", 0.0, 180.0, 0.5, "°")}
)
class RotateOperator(TTAOperator):
    """绕图像中心旋转（保持原尺寸，边界 replicate 外推）。

    算法：``cv2.getRotationMatrix2D`` 构造绕中心的旋转矩阵，再
    ``cv2.warpAffine`` 双线性重采样。角度随机取 ``±angle``，即 0~180°
    两个方向等概率。

    参考：OpenCV 几何变换；Shorten & Khoshgoftaar (2019) 数据增强综述。
    """

    signed_params = frozenset({"angle"})

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样旋转角度（度，带符号）。"""
        return {"angle": self._sample_signed("angle", rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定角度旋转图像。"""
        work, dtype = self._prepare(image)
        height, width = work.shape[:2]
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, float(params["angle"]), 1.0)
        return self._restore(_warp_affine(work, matrix), dtype)


@TTARegistry.register(
    "translate", "平移", "几何", 0.15, {"shift": ("平移比例", 0.0, 0.5, 0.01, "×")}
)
class TranslateOperator(TTAOperator):
    """平移（位移量按图像尺寸的比例给出，保持原尺寸）。

    算法：仿射矩阵 ``[[1, 0, shift·W], [0, 1, shift·H]]``。``shift`` 为带符号
    比例，``|shift| <= 0.5``，即最大平移半个画面；越界区域由
    ``BORDER_REPLICATE`` 用边缘（指纹背景，白色）填充。

    参考：OpenCV ``warpAffine``；Simard et al. (2003) 位移增强实践。
    """

    signed_params = frozenset({"shift"})

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样平移比例（带符号）。"""
        return {"shift": self._sample_signed("shift", rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定比例平移图像。"""
        work, dtype = self._prepare(image)
        height, width = work.shape[:2]
        shift = float(params["shift"])
        matrix = np.array(
            [[1.0, 0.0, shift * width], [0.0, 1.0, shift * height]], dtype=np.float32
        )
        return self._restore(_warp_affine(work, matrix), dtype)


@TTARegistry.register(
    "scale", "缩放", "几何", 0.20, {"factor": ("缩放比例", 0.0, 0.6, 0.01, "×")}
)
class ScaleOperator(TTAOperator):
    """以图像中心为原点缩放（放大或缩小，保持原尺寸）。

    算法：缩放系数 ``zoom = 1 + factor``，``factor`` 带符号且 ``|factor| <= 0.6``
    ⇒ ``zoom ∈ [0.4, 1.6]``；用 ``cv2.getRotationMatrix2D(center, 0, zoom)``
    构造“绕中心缩放”矩阵，``warpAffine`` + ``BORDER_REPLICATE`` 外推边缘，
    因此缩小后四周是背景白而不是黑边。

    参考：OpenCV ``getRotationMatrix2D`` 的 scale 参数（绕定点缩放）。
    """

    signed_params = frozenset({"factor"})

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样缩放比例（带符号）。"""
        return {"factor": self._sample_signed("factor", rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定比例绕中心缩放。"""
        work, dtype = self._prepare(image)
        height, width = work.shape[:2]
        zoom = 1.0 + float(params["factor"])
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, 0.0, float(zoom))
        return self._restore(_warp_affine(work, matrix), dtype)


@TTARegistry.register(
    "shear", "错切", "几何", 0.30, {"amount": ("错切量", 0.0, 0.6, 0.01, "×")}
)
class ShearOperator(TTAOperator):
    """水平错切（x 随 y 线性偏移，保持原尺寸）。

    算法：仿射矩阵 ``[[1, amount, -amount·(H-1)/2], [0, 1, 0]]``，即以图像
    竖直中线为不动轴做错切；``amount`` 带符号，``|amount| <= 0.6``。

    参考：OpenCV ``warpAffine``；Shorten & Khoshgoftaar (2019)。
    """

    signed_params = frozenset({"amount"})

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样错切量（带符号）。"""
        return {"amount": self._sample_signed("amount", rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定错切量变换图像。"""
        work, dtype = self._prepare(image)
        height, _width = work.shape[:2]
        amount = float(params["amount"])
        matrix = np.array(
            [[1.0, amount, -amount * (height - 1) / 2.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        return self._restore(_warp_affine(work, matrix), dtype)


@TTARegistry.register(
    "elastic",
    "弹性形变",
    "几何",
    0.45,
    {"alpha": ("位移强度", 0.0, 40.0, 0.5, "px"), "sigma": ("平滑尺度", 4.0, 120.0, 1.0, "px")},
)
class ElasticOperator(TTAOperator):
    """弹性形变（模拟皮肤按压时的非线性形变）。

    算法（Simard et al., 2003）：分别生成两幅单位标准差、相关长度为 ``sigma``
    的平滑随机场 ``dx, dy``（见 :func:`~synthfp.plugins.tta.base.random_field`），
    乘以 ``alpha`` 得到像素位移场，再用 ``cv2.remap`` 双线性重采样。

    Args:
        alpha: 位移幅度（像素，0~40）。
        sigma: 位移场平滑尺度（像素，4~120），越大形变越“大尺度”。

    参考：Simard, Steinkraus & Platt (2003), ICDAR, "Best practices for
    convolutional neural networks applied to visual document analysis"。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样位移幅度与平滑尺度。"""
        return {
            "alpha": self._sample("alpha", rng),
            "sigma": self._sample("sigma", rng),
            "_seed": self._draw_seed(rng),
        }

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按位移场做弹性形变。"""
        work, dtype = self._prepare(image)
        height, width = work.shape[:2]
        alpha = float(params["alpha"])
        sigma = float(params["sigma"])
        rng = self._rng_from(params)
        field_x = random_field((height, width), sigma, rng) * alpha
        field_y = random_field((height, width), sigma, rng) * alpha
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        warped = cv2.remap(
            work,
            grid_x + field_x,
            grid_y + field_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return self._restore(warped, dtype)


@TTARegistry.register(
    "perspective", "透视", "几何", 0.40, {"amount": ("透视强度", 0.0, 0.25, 0.01, "×")}
)
class PerspectiveOperator(TTAOperator):
    """透视变换（模拟手指与传感器平面不平行造成的梯形畸变）。

    算法：以图像四角为源点，每个角在 ``±amount·(W, H)`` 范围内随机抖动得到
    目标点，``cv2.getPerspectiveTransform`` 求 3x3 单应矩阵，
    ``cv2.warpPerspective`` 双线性重采样，``BORDER_REPLICATE`` 外推边缘。

    参考：OpenCV ``getPerspectiveTransform`` / ``warpPerspective``。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样透视强度。"""
        return {"amount": self._sample("amount", rng), "_seed": self._draw_seed(rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按随机四角抖动做透视变换。"""
        work, dtype = self._prepare(image)
        height, width = work.shape[:2]
        amount = float(params["amount"])
        rng = self._rng_from(params)
        src = np.array(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        )
        jitter = (rng.random((4, 2)).astype(np.float32) * 2.0 - 1.0) * np.array(
            [amount * width, amount * height], dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(src, src + jitter)
        warped = cv2.warpPerspective(
            work,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return self._restore(warped, dtype)


# ===========================================================================
# 光度
# ===========================================================================
@TTARegistry.register(
    "brightness", "亮度", "光度", 0.20, {"delta": ("亮度偏移", 0.0, 0.6, 0.01, "×")}
)
class BrightnessOperator(TTAOperator):
    """亮度偏移：``out = clip(image + delta)``，``delta`` 随机取 ``±``。

    参考：图像处理基础（灰度线性点运算）；Shorten & Khoshgoftaar (2019)。
    """

    signed_params = frozenset({"delta"})

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样亮度偏移量（带符号）。"""
        return {"delta": self._sample_signed("delta", rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定偏移量调整亮度。"""
        work, dtype = self._prepare(image)
        shifted = work + np.float32(params["delta"])
        return self._restore(shifted, dtype)


@TTARegistry.register(
    "contrast", "对比度", "光度", 0.25, {"factor": ("对比度系数", 0.0, 0.8, 0.01, "×")}
)
class ContrastOperator(TTAOperator):
    """对比度调整：``out = clip((image - 0.5) · (1 + factor) + 0.5)``。

    ``factor`` 带符号，``|factor| <= 0.8`` ⇒ 增益 ``∈ [0.2, 1.8]``，负值为
    压对比度、正值为提对比度；以 0.5 为中灰不动点。

    参考：灰度线性点运算（对比度/增益）；NFIQ2 中对比度对指纹质量的影响。
    """

    signed_params = frozenset({"factor"})

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样对比度系数（带符号）。"""
        return {"factor": self._sample_signed("factor", rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定增益调整对比度。"""
        work, dtype = self._prepare(image)
        gain = np.float32(1.0 + float(params["factor"]))
        adjusted = (work - np.float32(0.5)) * gain + np.float32(0.5)
        return self._restore(adjusted, dtype)


@TTARegistry.register(
    "gamma", "Gamma", "光度", 0.25, {"gamma": ("伽马值", 0.2, 3.0, 0.05, "×")}
)
class GammaOperator(TTAOperator):
    """伽马校正：``out = image ** gamma``（不取符号，仅单调非线性）。

    ``gamma < 1`` 提亮暗部、``gamma > 1`` 压暗暗部，用来模拟传感器非线性响应
    与不同成像增益。输入先 clip 到 ``[0, 1]`` 保证幂运算无 NaN。

    参考：ITU-R BT.709 传递函数；OpenCV ``cv2.LUT`` 伽马校正实践。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样伽马值。"""
        return {"gamma": self._sample("gamma", rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定伽马值做幂律变换。"""
        work, dtype = self._prepare(image)
        corrected = np.power(work, np.float32(float(params["gamma"])))
        return self._restore(corrected, dtype)


@TTARegistry.register(
    "blur", "高斯模糊", "光度", 0.35, {"sigma": ("高斯标准差", 0.0, 4.0, 0.1, "px")}
)
class BlurOperator(TTAOperator):
    """高斯模糊（离焦/运动/传感器点扩散函数造成的整体软化）。

    算法：``cv2.GaussianBlur``，核尺寸按 OpenCV 规则由 ``sigma`` 推导
    （见 :func:`~synthfp.plugins.tta.base.gaussian_blur`），
    ``sigma <= 0`` 时原样返回。

    参考：OpenCV ``GaussianBlur``；ISO/IEC 19794-4 采集要求中的模糊容忍度。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样高斯标准差。"""
        return {"sigma": self._sample("sigma", rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定 sigma 做高斯模糊。"""
        work, dtype = self._prepare(image)
        return self._restore(gaussian_blur(work, float(params["sigma"])), dtype)


@TTARegistry.register(
    "motion_blur", "运动模糊", "光度", 0.40, {"length": ("模糊长度", 1.0, 25.0, 1.0, "px")}
)
class MotionBlurOperator(TTAOperator):
    """线性运动模糊（手指/设备在曝光期间相对移动）。

    算法：在 ``length x length`` 核中画一条水平线，再用
    ``cv2.getRotationMatrix2D`` + ``cv2.warpAffine`` 把该线旋转到随机方向
    （方向由 ``_seed`` 决定），归一化后 ``cv2.filter2D`` 卷积。
    ``length <= 1`` 视为恒等。

    参考：图像退化模型 ``g = f * h + n``（运动模糊 PSF 为线段）；
    Gonzalez & Woods, Digital Image Processing, 4th ed., Ch. 5。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样运动模糊长度（方向由内部种子决定）。"""
        return {"length": self._sample_int("length", rng), "_seed": self._draw_seed(rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定长度与随机方向做运动模糊。"""
        work, dtype = self._prepare(image)
        length = int(params["length"])
        if length < 2:
            return self._restore(work, dtype)
        rng = self._rng_from(params)
        angle = float(rng.uniform(0.0, 180.0))
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0
        rotation = cv2.getRotationMatrix2D(
            ((length - 1) / 2.0, (length - 1) / 2.0), angle, 1.0
        )
        kernel = cv2.warpAffine(
            kernel,
            rotation,
            (length, length),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        total = float(kernel.sum())
        if total <= 1e-6:
            return self._restore(work, dtype)
        kernel = kernel / np.float32(total)
        blurred = cv2.filter2D(work, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        return self._restore(blurred, dtype)


# ===========================================================================
# 噪声
# ===========================================================================
@TTARegistry.register(
    "gaussian_noise", "高斯噪声", "噪声", 0.30, {"sigma": ("噪声强度", 0.0, 0.15, 0.005, "×")}
)
class GaussianNoiseOperator(TTAOperator):
    """加性高斯白噪声：``out = clip(image + N(0, sigma))``。

    ``sigma`` 为归一化灰度域（``[0, 1]``）的标准差，``0.15`` 已接近视觉上的
    强噪声。噪声场由 ``_seed`` 重建的 ``np.random.default_rng`` 生成，保证可复现。

    参考：传感器读出噪声模型（Gaussian read noise）；
    Gonzalez & Woods, Ch. 5.2。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样噪声强度。"""
        return {"sigma": self._sample("sigma", rng), "_seed": self._draw_seed(rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定强度叠加高斯噪声。"""
        work, dtype = self._prepare(image)
        sigma = float(params["sigma"])
        if sigma <= 0.0:
            return self._restore(work, dtype)
        rng = self._rng_from(params)
        noise = rng.normal(0.0, sigma, size=work.shape).astype(np.float32)
        return self._restore(work + noise, dtype)


@TTARegistry.register(
    "poisson", "泊松噪声", "噪声", 0.30, {"scale": ("光子尺度", 10.0, 400.0, 1.0, "×")}
)
class PoissonNoiseOperator(TTAOperator):
    """泊松散粒噪声（光子计数噪声）：``out = Poisson(image · scale) / scale``。

    ``scale`` 等效于“满量程光子数”，越小噪声越强（``10`` 极强、``400`` 轻微）。

    参考：光子到达服从泊松分布；Foi et al. (2008), "Practical Poissonian-
    Gaussian noise modeling and fitting for single-image raw-data", IEEE TIP。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样光子尺度。"""
        return {"scale": self._sample("scale", rng), "_seed": self._draw_seed(rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定光子尺度施加泊松噪声。"""
        work, dtype = self._prepare(image)
        scale = float(params["scale"])
        if scale <= 0.0:
            return self._restore(work, dtype)
        rng = self._rng_from(params)
        counts = rng.poisson(np.clip(work, 0.0, 1.0) * scale).astype(np.float32)
        return self._restore(counts / np.float32(scale), dtype)


@TTARegistry.register(
    "salt_pepper", "椒盐噪声", "噪声", 0.30, {"amount": ("噪声比例", 0.0, 0.15, 0.005, "×")}
)
class SaltPepperOperator(TTAOperator):
    """椒盐噪声：按 ``amount`` 比例随机把像素置为 0（椒）或 1（盐），各占一半。

    参考：脉冲噪声模型；Gonzalez & Woods, Digital Image Processing, Ch. 5.3。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样噪声像素比例。"""
        return {"amount": self._sample("amount", rng), "_seed": self._draw_seed(rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定比例施加椒盐噪声。"""
        work, dtype = self._prepare(image)
        amount = float(params["amount"])
        if amount <= 0.0:
            return self._restore(work, dtype)
        rng = self._rng_from(params)
        selected = rng.random(work.shape) < amount
        salty = rng.random(work.shape) < 0.5
        work[selected & salty] = 1.0
        work[selected & ~salty] = 0.0
        return self._restore(work, dtype)


# ===========================================================================
# 压缩
# ===========================================================================
@TTARegistry.register(
    "jpeg", "JPEG 压缩", "压缩", 0.35, {"quality": ("压缩质量", 5.0, 100.0, 1.0, "")}
)
class JpegOperator(TTAOperator):
    """JPEG 有损压缩（DCT 量化 + 8x8 块效应）。

    契约要求 ``quality`` 为**下界**：实际质量在 ``[quality, 100]`` 内均匀取整，
    ``sample_params`` 返回的是本次实际使用的质量值（仍落在 ``5~100`` 内），
    便于 manifest 精确复现。实现为 ``uint8`` 往返：``cv2.imencode('.jpg')`` →
    ``cv2.imdecode``，编码失败时原样返回。

    参考：ITU-T T.81 (JPEG)；ISO/IEC 19794-4 中压缩失真对识别率的影响。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样实际 JPEG 质量（``quality`` 为下界，向上随机）。"""
        lower = self._sample_int("quality", rng)
        effective = int(rng.integers(lower, 101))
        return {"quality": int(min(max(effective, 5), 100))}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """按给定质量做一次 JPEG 往返编解码。"""
        work, dtype = self._prepare(image)
        quality = int(min(max(int(params["quality"]), 1), 100))
        encoded_u8 = np.clip(np.rint(work * np.float32(255.0)), 0, 255).astype(np.uint8)
        ok, buffer = cv2.imencode(
            ".jpg", encoded_u8, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            return self._restore(work, dtype)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            return self._restore(work, dtype)
        return self._restore(decoded.astype(np.float32) / np.float32(255.0), dtype)


# ===========================================================================
# 遮挡
# ===========================================================================
def _blend_occlusion(work: np.ndarray, mask: np.ndarray, feather: float = 1.5) -> np.ndarray:
    """把遮挡掩模（``float32 [0,1]``）按背景色混合到图像上。

    遮挡区域填充为图像背景值（高分位灰度，默认 1.0），边缘做 ``feather`` 尺度
    的高斯羽化，避免出现生硬的白块或黑块。
    """
    filled = background_level(work)
    soft = gaussian_blur(mask, feather) if feather > 0 else mask
    soft = np.clip(soft, 0.0, 1.0).astype(np.float32)
    return work * (1.0 - soft) + np.float32(filled) * soft


@TTARegistry.register(
    "occlusion_rect",
    "矩形遮挡",
    "遮挡",
    0.50,
    {"count": ("块数", 1.0, 8.0, 1.0, "块"), "area": ("面积占比", 0.01, 0.3, 0.01, "×")},
)
class RectOcclusionOperator(TTAOperator):
    """矩形遮挡：随机放置若干轴对齐矩形（模拟贴纸、污渍、缺指按压）。

    算法：每块面积 = ``area · H · W · U(0.5, 1)``，随机长宽比与位置；遮挡区域
    填充图像背景值（高分位灰度）并轻度羽化，绝不填 0。

    参考：SFinGe 的“脏手指/缺损”退化；ISO/IEC 19794-4 有效面积定义。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样块数与面积占比。"""
        return {
            "count": self._sample_int("count", rng),
            "area": self._sample("area", rng),
            "_seed": self._draw_seed(rng),
        }

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """绘制随机矩形遮挡。"""
        work, dtype = self._prepare(image)
        height, width = work.shape[:2]
        rng = self._rng_from(params)
        mask = np.zeros((height, width), dtype=np.float32)
        total = float(height * width)
        for _ in range(int(params["count"])):
            scale = float(rng.uniform(0.5, 1.0))
            target = float(params["area"]) * scale * total
            aspect = float(rng.uniform(0.5, 2.0))
            block_h = int(min(max(2, round((target * aspect) ** 0.5)), height))
            block_w = int(min(max(2, round((target / aspect) ** 0.5)), width))
            top = int(rng.integers(0, max(1, height - block_h + 1)))
            left = int(rng.integers(0, max(1, width - block_w + 1)))
            mask[top : top + block_h, left : left + block_w] = 1.0
        return self._restore(_blend_occlusion(work, mask), dtype)


@TTARegistry.register(
    "occlusion_ellipse",
    "椭圆遮挡",
    "遮挡",
    0.50,
    {"count": ("个数", 1.0, 8.0, 1.0, "个"), "area": ("面积占比", 0.01, 0.3, 0.01, "×")},
)
class EllipseOcclusionOperator(TTAOperator):
    """椭圆遮挡：随机放置若干旋转椭圆（模拟油污、水渍、按压气泡）。

    算法：由目标面积反解半轴 ``a·b = area·H·W/π``，随机长宽比与旋转角，
    用 ``cv2.ellipse`` 实心绘制后与矩形遮挡同样做背景填充 + 羽化。

    参考：SFinGe 退化模块中的 blemish / stain 模型。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样椭圆个数与面积占比。"""
        return {
            "count": self._sample_int("count", rng),
            "area": self._sample("area", rng),
            "_seed": self._draw_seed(rng),
        }

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """绘制随机椭圆遮挡。"""
        work, dtype = self._prepare(image)
        height, width = work.shape[:2]
        rng = self._rng_from(params)
        mask = np.zeros((height, width), dtype=np.float32)
        total = float(height * width)
        for _ in range(int(params["count"])):
            scale = float(rng.uniform(0.5, 1.0))
            target = float(params["area"]) * scale * total
            aspect = float(rng.uniform(0.4, 2.5))
            axis_a = int(min(max(2, round((target * aspect / np.pi) ** 0.5)), width))
            axis_b = int(min(max(2, round((target / (aspect * np.pi)) ** 0.5)), height))
            center = (
                int(rng.integers(0, width)),
                int(rng.integers(0, height)),
            )
            angle = float(rng.uniform(0.0, 180.0))
            cv2.ellipse(
                mask,
                center,
                (axis_a, axis_b),
                angle,
                0.0,
                360.0,
                1.0,
                -1,
                cv2.LINE_8,
            )
        return self._restore(_blend_occlusion(work, mask), dtype)


@TTARegistry.register(
    "scratch",
    "划痕遮挡",
    "遮挡",
    0.40,
    {"count": ("条数", 1.0, 12.0, 1.0, "条"), "width": ("宽度", 1.0, 12.0, 1.0, "px")},
)
class ScratchOperator(TTAOperator):
    """划痕遮挡：绘制若干条穿过画面的亮线（模拟划伤玻璃/棱镜、指纹上的折痕）。

    算法：每条划痕从图像一侧边界上的随机点到对侧边界上的随机点连线，
    线宽取 ``width`` 像素，填充图像背景值并轻度羽化。

    参考：SFinGe 退化模块中的 scratch / crease 模型；光学采集的划痕伪影。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样划痕条数与线宽。"""
        return {
            "count": self._sample_int("count", rng),
            "width": self._sample_int("width", rng),
            "_seed": self._draw_seed(rng),
        }

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """绘制随机划痕遮挡。"""
        work, dtype = self._prepare(image)
        height, width = work.shape[:2]
        rng = self._rng_from(params)
        mask = np.zeros((height, width), dtype=np.float32)
        thickness = int(max(1, int(params["width"])))
        for _ in range(int(params["count"])):
            if float(rng.random()) < 0.5:
                start = (int(rng.integers(0, width)), 0)
                end = (int(rng.integers(0, width)), height - 1)
            else:
                start = (0, int(rng.integers(0, height)))
                end = (width - 1, int(rng.integers(0, height)))
            cv2.line(mask, start, end, 1.0, thickness, cv2.LINE_8)
        return self._restore(_blend_occlusion(work, mask, feather=0.8), dtype)


# ===========================================================================
# 传感器
# ===========================================================================
@TTARegistry.register(
    "sensor_optical", "光学传感器", "传感器", 0.30, {"strength": ("强度", 0.0, 3.0, 0.1, "×")}
)
class OpticalSensorOperator(TTAOperator):
    """光学（FTIR 棱镜/全内反射）指纹传感器成像特性。

    组合模型：径向暗角（vignette，余弦四次方衰减的简化平方律）+ 低对比度
    （散射导致调制传递下降）+ 轻微乘性散斑。``strength=0`` 为逐像素直通。

    参考：FTIR 光学指纹采集原理（棱镜 + 透镜成像）；
    Maltoni et al., Handbook of Fingerprint Recognition, 2nd ed., Ch. 2。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样传感器强度。"""
        return {"strength": self._sample("strength", rng), "_seed": self._draw_seed(rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """施加光学传感器成像特性。"""
        work, dtype = self._prepare(image)
        scale = float(min(max(float(params["strength"]), 0.0), 3.0))
        if scale <= 1e-6:
            return self._restore(work, dtype)
        height, width = work.shape[:2]
        rng = self._rng_from(params)
        grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        radius_x = (grid_x - (width - 1) / 2.0) / max((width - 1) / 2.0, 1.0)
        radius_y = (grid_y - (height - 1) / 2.0) / max((height - 1) / 2.0, 1.0)
        radius2 = radius_x * radius_x + radius_y * radius_y
        vignette = np.clip(1.0 - 0.075 * scale * radius2, 0.45, 1.0).astype(np.float32)
        result = work * vignette
        gain = 1.0 - 0.10 * scale
        result = np.float32(0.5) + (result - np.float32(0.5)) * np.float32(gain)
        speckle = gaussian_blur(
            rng.normal(0.0, 1.0, size=work.shape).astype(np.float32), 0.6
        )
        result = result * (1.0 + np.float32(0.04 * scale) * speckle)
        return self._restore(result, dtype)


@TTARegistry.register(
    "sensor_capacitive",
    "电容传感器",
    "传感器",
    0.30,
    {"strength": ("强度", 0.0, 3.0, 0.1, "×")},
)
class CapacitiveSensorOperator(TTAOperator):
    """电容（active capacitance）指纹传感器成像特性。

    组合模型：高频读出噪声 + 少量椒盐坏点 + 轻微行方向固定模式噪声（FPN）。
    ``strength=0`` 为逐像素直通。

    参考：电容式指纹传感器读出电路噪声（kTC + 1/f）；
    Maltoni et al., Handbook of Fingerprint Recognition, Ch. 2。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样传感器强度。"""
        return {"strength": self._sample("strength", rng), "_seed": self._draw_seed(rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """施加电容传感器成像特性。"""
        work, dtype = self._prepare(image)
        scale = float(min(max(float(params["strength"]), 0.0), 3.0))
        if scale <= 1e-6:
            return self._restore(work, dtype)
        height, _width = work.shape[:2]
        rng = self._rng_from(params)
        result = work + rng.normal(0.0, 1.0, size=work.shape).astype(np.float32) * np.float32(
            0.05 * scale
        )
        bad = rng.random(work.shape) < (0.004 * scale)
        salty = rng.random(work.shape) < 0.5
        result[bad & salty] = 1.0
        result[bad & ~salty] = 0.0
        row_noise = gaussian_blur(
            rng.normal(0.0, 1.0, size=(height, 1)).astype(np.float32), 1.0
        )
        result = result + row_noise * np.float32(0.02 * scale)
        return self._restore(result, dtype)


@TTARegistry.register(
    "sensor_thermal", "热感传感器", "传感器", 0.30, {"strength": ("强度", 0.0, 3.0, 0.1, "×")}
)
class ThermalSensorOperator(TTAOperator):
    """热感（滑动式热敏/温差传感）指纹传感器成像特性。

    组合模型：低频温度漂移 + 行方向条带（扫描行增益不一致）+ 非线性响应
    （``out = out ** (1 + 0.12·strength)``）。``strength=0`` 为逐像素直通。

    参考：热敏式指纹传感原理（热电/热容差成像）与扫描行 FPN；
    Maltoni et al., Handbook of Fingerprint Recognition, Ch. 2。
    """

    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样传感器强度。"""
        return {"strength": self._sample("strength", rng), "_seed": self._draw_seed(rng)}

    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """施加热感传感器成像特性。"""
        work, dtype = self._prepare(image)
        scale = float(min(max(float(params["strength"]), 0.0), 3.0))
        if scale <= 1e-6:
            return self._restore(work, dtype)
        height, width = work.shape[:2]
        rng = self._rng_from(params)
        drift = random_field((height, width), max(6.0, min(height, width) / 4.0), rng)
        result = work + drift * np.float32(0.06 * scale)
        stripes = gaussian_blur(
            rng.normal(0.0, 1.0, size=(height, 1)).astype(np.float32), 1.0
        )
        result = result + stripes * np.float32(0.05 * scale)
        result = np.clip(result, 0.0, 1.0)
        gamma = np.float32(1.0 + 0.12 * scale)
        result = np.power(result, gamma)
        return self._restore(result, dtype)


# ---------------------------------------------------------------------------
# 批量实例化
# ---------------------------------------------------------------------------
#: 全部注册名（按 ``operators.py`` 中的定义顺序）。
OPERATORS: List[str] = TTARegistry.all_names()


def create_operators(
    names: Sequence[str], params: Optional[Dict[str, dict]] = None
) -> List[TTAOperator]:
    """按名称批量实例化算子（保持 ``names`` 顺序）。

    Args:
        names: 注册名序列，如 ``["rotate", "gaussian_noise"]``。
        params: 可选，``{注册名: {参数名: 固定值}}``；未给出的算子在
            ``param_spec`` 范围内随机采样。

    Returns:
        算子实例列表，顺序与 ``names`` 完全一致。

    Raises:
        KeyError: 出现未注册的算子名（错误信息含可用算子列表）。
    """
    overrides: Dict[str, Any] = dict(params) if params else {}
    return [TTARegistry.create(name, overrides.get(name)) for name in names]
