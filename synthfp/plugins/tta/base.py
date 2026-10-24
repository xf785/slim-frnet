"""TTA 算子抽象基类与公共图像工具。

本模块定义 :class:`TTAOperator` —— TTA（Test-Time Augmentation，测试时增强）
引擎中所有算子的统一接口。TTA 的作用是把少量主指纹扩展成大规模、多样化的
样本集，因此每个算子都必须满足以下硬性约束（契约 §0 / §1.1）：

* **尺寸不变**：输出与输入形状一致 ``(H, W)``。几何变换用
  ``cv2.BORDER_REPLICATE`` 填充，避免白背景指纹出现黑边。
* **dtype 不变**：``float32`` 输入给出 ``float32`` 输出，``uint8`` 输入给出
  ``uint8`` 输出。
* **值域安全**：``float32`` 输出恒在 ``[0, 1]``，``uint8`` 输出恒在 ``[0, 255]``，
  绝不产生 ``NaN`` / ``Inf``。
* **不修改输入**：内部始终在输入副本上运算（见 :meth:`TTAOperator._prepare`）。
* **可复现**：随机数只从外部传入的 ``rng`` 采样，禁止使用 ``np.random.*``
  的全局状态；``_apply`` 是 ``(image, params)`` 的纯函数，同样的 ``params``
  必得同样的结果。

``_apply`` 内部若需要随机场（噪声、遮挡位置、散斑等），通过参数里的隐藏键
``"_seed"`` 携带的整数种子重建 ``np.random.default_rng``。隐藏键不写入
``param_spec``，因此 :meth:`TTAOperator.describe` 与 manifest 的
``tta_params`` 只暴露人类可读的公开参数。

参考文献：
    * Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). Best practices for
      convolutional neural networks applied to visual document analysis. ICDAR.
    * Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data
      Augmentation for Deep Learning. Journal of Big Data, 6(1), 60.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Dict, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "TTAOperator",
    "TTA_CATEGORIES",
    "gaussian_blur",
    "background_level",
    "random_field",
]

#: 契约 §1.3 规定的六个算子分类（注册表 :meth:`TTARegistry.by_category` 的键集合）。
TTA_CATEGORIES: Tuple[str, ...] = ("几何", "光度", "噪声", "压缩", "遮挡", "传感器")

#: 隐藏参数（内部键）前缀：不参与 ``param_spec`` 校验，不进入 ``describe``。
_INTERNAL_PREFIX = "_"


# ---------------------------------------------------------------------------
# 公共图像工具（供各算子复用，也便于单元测试）
# ---------------------------------------------------------------------------
def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """对 ``float32`` 图像做高斯模糊（边缘按 ``BORDER_REPLICATE`` 外推）。

    核尺寸按 OpenCV 内部规则 ``2 * round(4 * sigma) + 1`` 计算，并截断到
    ``[3, 121]``，以免 ``sigma`` 很大时（契约允许 elastic 的 sigma 到 120 px）
    产生上千像素的卷积核拖慢整个管道。

    Args:
        image: ``float32`` ``(H, W)`` 图像。
        sigma: 高斯标准差（像素）。``sigma <= 0`` 时返回输入的副本。

    Returns:
        模糊后的 ``float32`` ``(H, W)`` 新数组；``sigma <= 0`` 时为输入的副本。
    """
    sigma = float(sigma)
    if sigma <= 1e-3:
        return np.array(image, dtype=np.float32, copy=True)
    kernel = int(round(sigma * 4.0)) * 2 + 1
    kernel = int(min(max(kernel, 3), 121))
    blurred = cv2.GaussianBlur(
        np.asarray(image, dtype=np.float32),
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    return blurred


def background_level(image: np.ndarray, default: float = 1.0) -> float:
    """估计图像背景灰度（高分位灰度），用于遮挡区域的填充。

    Args:
        image: ``float32`` ``(H, W)`` 图像，范围 ``[0, 1]``。
        default: 无法估计时（空数组 / 非有限值）使用的默认背景值。

    Returns:
        背景灰度，恒在 ``[0, 1]``。合成指纹背景为白，故通常约等于 ``1.0``。
    """
    flat = np.asarray(image, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return float(default)
    level = float(np.quantile(flat, 0.98))
    if not np.isfinite(level):
        return float(default)
    return float(min(max(level, 0.0), 1.0))


def random_field(shape: Tuple[int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    """生成单位标准差的低频平滑随机场（空间相关长度约等于 ``sigma``）。

    做法：在间距为 ``sigma`` 的粗网格上采样标准正态噪声，用三次插值放大到目标
    尺寸，再做一次小尺度高斯平滑消除插值条纹，最后归一化到标准差 1 —— 这样
    调用方可以直接用幅度系数控制位移量。相比对全分辨率白噪声做 ``sigma``
    尺度的高斯卷积，该实现在 ``sigma`` 很大时依然很快。

    Args:
        shape: 输出尺寸 ``(H, W)``。
        sigma: 平滑尺度（像素），相关长度。
        rng: 随机源。

    Returns:
        ``float32`` ``(H, W)`` 随机场，标准差约 1，均值为 0。
    """
    height, width = int(shape[0]), int(shape[1])
    spacing = float(max(sigma, 1.0))
    coarse_h = int(min(max(2, round(height / spacing) + 1), max(height, 2)))
    coarse_w = int(min(max(2, round(width / spacing) + 1), max(width, 2)))
    coarse = rng.standard_normal((coarse_h, coarse_w)).astype(np.float32)
    if (coarse_h, coarse_w) != (height, width):
        field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        field = coarse
    field = gaussian_blur(field, max(0.8, spacing / 3.0))
    std = float(field.std())
    if std > 1e-6:
        field = field / np.float32(std)
    return np.ascontiguousarray(field, dtype=np.float32)


# ---------------------------------------------------------------------------
# 算子基类
# ---------------------------------------------------------------------------
class TTAOperator(abc.ABC):
    """TTA 算子抽象基类。

    子类必须声明五个类属性（通常由 :meth:`TTARegistry.register` 装饰器写入），
    并实现 :meth:`sample_params` 与 :meth:`_apply`。

    Attributes:
        name: 算子标识，等于注册名，如 ``"rotate"``。
        label: 中文名，如 ``"旋转"``。
        category: 分类，取值见 :data:`TTA_CATEGORIES`。
        distortion: 典型失真度 ``0~1``，供管道做累计失真截断。
        param_spec: ``{参数名: (中文名, 最小值, 最大值, 步长, 单位)}``。
        signed_params: 扩展属性 —— 取值范围随机取正负号的参数名集合
            （契约表格中标注“随机 ±”的参数）。这些参数由
            :meth:`_sample_signed` 采样，返回值落在 ``[-max, +max]``。
        params: 用户在构造时指定的固定值（覆盖随机采样）。
        last_params: 最近一次 :meth:`apply` 使用的完整参数，供 manifest 记录。
    """

    name: ClassVar[str] = ""
    label: ClassVar[str] = ""
    category: ClassVar[str] = ""
    distortion: ClassVar[float] = 0.0
    param_spec: ClassVar[Dict[str, Tuple[str, float, float, float, str]]] = {}
    signed_params: ClassVar[frozenset] = frozenset()

    def __init__(self, params: Optional[dict] = None) -> None:
        """构造算子。

        Args:
            params: 用户覆盖的固定值，如 ``{"angle": 8.0}``。只接受出现在
                ``param_spec`` 中的键，未知键会被忽略（避免配置笔误直接崩溃
                整条管道）；取值超出声明范围时会被裁剪回范围内。
        """
        self.params: Dict[str, Any] = {}
        if params:
            for key, value in params.items():
                if key in self.param_spec and value is not None:
                    self.params[key] = value
        self.last_params: Dict[str, Any] = {}

    # -- 参数采样 ---------------------------------------------------------
    def _range_of(self, key: str) -> Tuple[float, float, float, str]:
        """返回参数 ``key`` 的 ``(最小值, 最大值, 步长, 单位)``。"""
        item = self.param_spec[key]
        return float(item[1]), float(item[2]), float(item[3]), str(item[4])

    def _sample(self, key: str, rng: np.random.Generator) -> float:
        """在 ``param_spec`` 声明范围内采样连续参数（用户已给定则直接采用）。

        Args:
            key: 参数名。
            rng: 随机源。

        Returns:
            落在 ``[min, max]`` 内的浮点值，已按步长量化并 clip。
        """
        low, high, step, _unit = self._range_of(key)
        if key in self.params:
            value = float(self.params[key])
        else:
            raw = float(rng.uniform(low, high))
            value = low + round((raw - low) / step) * step if step > 0 else raw
        value = min(max(value, low), high)
        return round(float(value), 6)

    def _sample_int(self, key: str, rng: np.random.Generator) -> int:
        """在声明范围内采样整数参数（块数、条数、JPEG 质量等）。

        Args:
            key: 参数名。
            rng: 随机源。

        Returns:
            落在 ``[round(min), round(max)]`` 内的整数。
        """
        low, high, _step, _unit = self._range_of(key)
        low_i, high_i = int(round(low)), int(round(high))
        if key in self.params:
            value = int(round(float(self.params[key])))
        else:
            value = int(rng.integers(low_i, high_i + 1))
        return int(min(max(value, low_i), high_i))

    def _sample_signed(self, key: str, rng: np.random.Generator) -> float:
        """采样“随机 ±”参数：幅值取自 ``[min, max]``，符号由 ``rng`` 决定。

        若用户在 ``params`` 中显式给出该参数，则**直接采用**该值（允许负数，
        即用户可自行指定方向），只做范围裁剪到 ``[-max, +max]``。

        Args:
            key: 参数名。
            rng: 随机源。

        Returns:
            落在 ``[-max, +max]`` 内的浮点值。
        """
        _low, high, _step, _unit = self._range_of(key)
        if key in self.params:
            value = float(self.params[key])
            return round(float(min(max(value, -high), high)), 6)
        magnitude = self._sample(key, rng)
        signed = magnitude if float(rng.random()) < 0.5 else -magnitude
        return round(float(signed), 6)

    def _draw_seed(self, rng: np.random.Generator) -> int:
        """为 ``_apply`` 内部的随机场抽取一个整数种子（写入隐藏键 ``"_seed"``）。"""
        return int(rng.integers(0, 2**31 - 1))

    @staticmethod
    def _rng_from(params: dict) -> np.random.Generator:
        """由参数里的隐藏键 ``"_seed"`` 重建随机源（保证 ``_apply`` 可复现）。"""
        return np.random.default_rng(int(params.get("_seed", 0)))

    # -- 图像准备 / 还原 ---------------------------------------------------
    @staticmethod
    def _prepare(image: np.ndarray) -> Tuple[np.ndarray, np.dtype]:
        """把输入图像转成内部工作副本 ``float32 [0, 1]``。

        Args:
            image: ``(H, W)`` 灰度图，``uint8`` 或浮点（``[0, 1]``）。

        Returns:
            ``(work, dtype)``：``work`` 是全新副本（绝不与输入共享内存），
            ``dtype`` 是输入原始 dtype，供 :meth:`_restore` 还原。

        Raises:
            TypeError: ``image`` 不是 ``np.ndarray``。
            ValueError: ``image`` 不是二维灰度图。
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"TTA 算子只接受 np.ndarray，收到 {type(image)!r}")
        if image.ndim != 2:
            raise ValueError(f"TTA 算子只支持 (H, W) 灰度图，收到 shape={image.shape}")
        dtype = image.dtype
        if dtype == np.uint8:
            work = image.astype(np.float32) / np.float32(255.0)
        else:
            work = image.astype(np.float32, copy=True)
        np.clip(work, 0.0, 1.0, out=work)
        if not bool(np.isfinite(work).all()):
            work = np.nan_to_num(work, nan=1.0, posinf=1.0, neginf=0.0)
        return work, dtype

    @staticmethod
    def _restore(work: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """把内部 ``float32 [0, 1]`` 结果还原为输入 dtype（并 clip 到合法值域）。"""
        clipped = np.clip(np.asarray(work, dtype=np.float32), 0.0, 1.0)
        if not bool(np.isfinite(clipped).all()):
            clipped = np.nan_to_num(clipped, nan=1.0, posinf=1.0, neginf=0.0)
        if dtype == np.uint8:
            return np.rint(clipped * np.float32(255.0)).astype(np.uint8)
        return clipped.astype(dtype, copy=False)

    # -- 契约接口 ---------------------------------------------------------
    @abc.abstractmethod
    def sample_params(self, rng: np.random.Generator) -> dict:
        """采样一组完整参数。

        Args:
            rng: 随机源。

        Returns:
            参数字典。已在 ``self.params`` 中固定给出的参数必须直接采用，
            不再随机；所有取值都必须落在 ``param_spec`` 声明范围内
            （“随机 ±”的参数见 :meth:`_sample_signed`）。
        """

    @abc.abstractmethod
    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
        """用给定参数施加变换（纯函数，不消费随机源）。

        Args:
            image: 输入图像。
            params: :meth:`sample_params` 给出的参数字典。

        Returns:
            变换后的图像，形状与 dtype 同输入，值域安全。
        """

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """施加变换：``sample_params`` + ``_apply``，并记录 ``last_params``。

        Args:
            image: 输入图像 ``(H, W)``。
            rng: 随机源。

        Returns:
            变换后的图像，形状与 dtype 同输入。
        """
        params = self.sample_params(rng)
        result = self._apply(image, params)
        self.last_params = dict(params)
        return result

    # -- 描述 -------------------------------------------------------------
    @staticmethod
    def _format_value(value: Any, step: float) -> str:
        """按步长决定小数位数，把参数值格式化成人类可读字符串。"""
        number = float(value)
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        if step >= 1.0:
            digits = 0
        elif step >= 0.1:
            digits = 1
        elif step >= 0.01:
            digits = 2
        else:
            digits = 3
        return f"{number:.{digits}f}"

    def describe(self) -> str:
        """返回最近一次 :meth:`apply` 的人类可读描述，如 ``"旋转(角度=7.5°)"``。

        Returns:
            描述字符串；尚未调用过 :meth:`apply` 时退化为中文名（或注册名）。
        """
        fallback = self.label or self.name
        if not self.last_params:
            return fallback
        parts = []
        for key, item in self.param_spec.items():
            if key not in self.last_params:
                continue
            parts.append(f"{item[0]}={self._format_value(self.last_params[key], float(item[3]))}{item[4]}")
        return f"{fallback}({', '.join(parts)})" if parts else fallback

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<{type(self).__name__} name={self.name!r} params={self.params!r}>"
