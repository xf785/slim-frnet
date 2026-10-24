"""传感器成像模拟：光学 / 电容 / 热感 / 直通。

本模块模拟指纹采集传感器的成像特性，输入输出统一为 ``uint8 (H, W)`` 灰度图，
背景为白（255）。三类传感器按契约实现：

============  ==========================================================
``optical``   径向暗角（vignette）+ 低对比度 + 轻微散斑（乘性）
``capacitive`` 高频噪声 + 少量椒盐 + 轻微行噪声
``thermal``   低频漂移 + 行方向条带 + 非线性响应（墨迹域 Gamma）
============  ==========================================================

统一约束：

* ``strength=0.0`` 为**逐像素完全一致**的直通（直接返回入参副本，dtype/取值均不变）；
* 不修改传入数组；输出保证有限并落在合法值域内；
* 完全由传入的 ``rng`` 决定，同种子同结果，不使用任何全局随机函数；
* 为方便滤镜管线复用，``float`` 输入按 ``[0, 1]``（最大像素 >1.5 时按 ``[0, 255]``）
  解释，并返回**与输入相同的 dtype**；``uint8`` 输入返回 ``uint8``。

参考文献：

* Maltoni D. et al. *Handbook of Fingerprint Recognition*, 3rd ed., 2022, Ch. 2
  （光学 / 电容 / 热感传感器的成像特性与噪声模型）。
* 契约文档 ``CONTRACTS.md`` 第 1.3 节（传感器算子）与第 4.2 节。
"""

from __future__ import annotations

import abc

import cv2
import numpy as np

__all__ = [
    "SENSORS",
    "CapacitiveSensor",
    "NoneSensor",
    "OpticalSensor",
    "SensorProfile",
    "ThermalSensor",
    "get_sensor",
]

#: ``strength`` 上限，与 TTA 算子 param_spec 中的 ``0~3.0`` 保持一致。
_MAX_STRENGTH: float = 3.0

#: 浮点输入判为 ``[0, 1]`` 的阈值（超过则按 ``[0, 255]`` 解释）。
_UNIT_RANGE_PEAK: float = 1.5


def _clip_strength(strength: float) -> float:
    """把 ``strength`` 截断到 ``[0, _MAX_STRENGTH]``，非有限值按 0 处理。

    Args:
        strength: 原始强度。

    Returns:
        截断后的强度。
    """
    try:
        value = float(strength)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, _MAX_STRENGTH))


def _prepare(image: np.ndarray) -> tuple[np.ndarray, np.dtype, bool]:
    """把输入统一成 ``float32 [0, 255]`` 工作副本。

    Args:
        image: 输入灰度图，``uint8`` / 浮点 / 整型均可。

    Returns:
        ``(work, orig_dtype, unit_range)``：工作副本（新建数组，可安全就地修改）、
        原始 dtype、原始浮点值域是否为 ``[0, 1]``。

    Raises:
        ValueError: 输入不是二维灰度图时抛出。
        TypeError: dtype 不受支持时抛出。
    """
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"传感器算子仅支持 (H, W) 灰度图，收到形状 {arr.shape}")
    if arr.dtype == np.dtype(np.uint8):
        return arr.astype(np.float32), arr.dtype, False
    if arr.dtype.kind == "f":
        work = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        unit_range = bool(work.size == 0 or float(work.max()) <= _UNIT_RANGE_PEAK)
        if unit_range:
            work = work * np.float32(255.0)
        return np.clip(work, 0.0, 255.0), arr.dtype, unit_range
    if arr.dtype.kind in ("i", "u", "b"):
        work = np.clip(arr.astype(np.float32), 0.0, 255.0)
        return work, arr.dtype, False
    raise TypeError(f"不支持的图像 dtype：{arr.dtype}")


def _finalize(work: np.ndarray, dtype: np.dtype, unit_range: bool) -> np.ndarray:
    """把 ``float32 [0, 255]`` 工作数组还原为原始 dtype。

    Args:
        work: 处理后的工作数组。
        dtype: 目标 dtype（与输入一致）。
        unit_range: 目标浮点值域是否为 ``[0, 1]``。

    Returns:
        与输入 dtype 相同的有限数组。
    """
    clipped = np.clip(np.nan_to_num(work, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0)
    if dtype == np.dtype(np.uint8):
        return np.rint(clipped).astype(np.uint8)
    if dtype.kind == "f":
        out = clipped / np.float32(255.0) if unit_range else clipped
        return out.astype(dtype, copy=False)
    return np.rint(clipped).astype(dtype, copy=False)


def _low_freq_field(
    height: int,
    width: int,
    rng: np.random.Generator,
    sigma: float,
) -> np.ndarray:
    """生成单位标准差的低频平滑随机场（粗网格 + 三次上采样）。

    Args:
        height: 场高度 ``H``。
        width: 场宽度 ``W``。
        rng: 随机源。
        sigma: 空间相关长度（像素）。

    Returns:
        ``float32 (H, W)``，均值 0、标准差 1。
    """
    sig = max(1.0, float(sigma))
    step = max(2.0, sig * 0.5)
    gh = int(np.clip(np.ceil(height / step) + 1.0, 2, max(2, height)))
    gw = int(np.clip(np.ceil(width / step) + 1.0, 2, max(2, width)))
    raw = rng.standard_normal((gh, gw), dtype=np.float32)
    if (gh, gw) != (height, width):
        field = cv2.resize(raw, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        field = raw.astype(np.float32, copy=True)
    field = field - np.float32(field.mean())
    std = float(field.std())
    if std > 1e-6:
        field = field / np.float32(std)
    else:  # pragma: no cover - 极端退化输入
        field = np.zeros_like(field, dtype=np.float32)
    return field


def _radius_map(height: int, width: int) -> np.ndarray:
    """生成归一化径向距离图（中心 0，四角 1）。

    Args:
        height: 图像高度。
        width: 图像宽度。

    Returns:
        ``float32 (H, W)``，取值 ``[0, 1]``。
    """
    xx, yy = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    cx = np.float32(max((width - 1) / 2.0, 1.0))
    cy = np.float32(max((height - 1) / 2.0, 1.0))
    r2 = ((xx - np.float32((width - 1) / 2.0)) / cx) ** 2 + (
        (yy - np.float32((height - 1) / 2.0)) / cy
    ) ** 2
    return np.sqrt(r2 / 2.0).astype(np.float32, copy=False)


class SensorProfile(abc.ABC):
    """传感器成像模型基类。

    Attributes:
        name: 类属性，传感器标识（= 注册名）。
        label: 类属性，中文名。
    """

    name: str
    label: str

    @abc.abstractmethod
    def apply(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        strength: float = 1.0,
    ) -> np.ndarray:
        """对图像施加传感器成像效果。

        Args:
            image: 输入灰度图，``uint8 (H, W)``（也接受浮点，见模块文档）。
            rng: 随机源。
            strength: 效果强度，截断到 ``[0, 3]``；``0`` 时逐像素直通。

        Returns:
            与 ``image`` 形状、dtype 相同的新数组；不修改 ``image``。
        """

    def describe(self, strength: float = 1.0) -> str:
        """返回人类可读描述，例如 ``"光学传感器(强度=1.00)"``。

        Args:
            strength: 效果强度。

        Returns:
            描述字符串。
        """
        return f"{self.label}(强度={_clip_strength(strength):.2f})"

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"{type(self).__name__}(name={self.name!r})"


class NoneSensor(SensorProfile):
    """直通传感器：任何 ``strength`` 都返回逐像素一致的副本。"""

    name = "none"
    label = "无传感器"

    def apply(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        strength: float = 1.0,
    ) -> np.ndarray:
        """原样返回输入副本（不消耗 ``rng``，不修改入参）。

        Args:
            image: 输入灰度图。
            rng: 随机源（本实现不使用）。
            strength: 忽略。

        Returns:
            输入数组的独立副本。
        """
        del rng, strength
        return np.array(image, copy=True)


class OpticalSensor(SensorProfile):
    """光学传感器：径向暗角 + 低对比度 + 轻微散斑。

    暗角采用近似 ``cos^4`` 律的二次径向衰减；对比度压缩在「墨迹域」
    （``255 - x``）进行，因此白色背景不会被整体拉灰，只是脊线变浅。
    """

    name = "optical"
    label = "光学传感器"

    def apply(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        strength: float = 1.0,
    ) -> np.ndarray:
        """施加光学传感器效果。

        Args:
            image: 输入灰度图，``uint8 (H, W)``。
            rng: 随机源。
            strength: 强度，``0`` 时逐像素直通。

        Returns:
            处理后图像，dtype 与输入一致。
        """
        s = _clip_strength(strength)
        if s <= 0.0:
            return np.array(image, copy=True)

        work, dtype, unit_range = _prepare(image)
        h, w = work.shape

        # 1) 低对比度：在墨迹域压缩，背景保持白
        #
        # 系数从 0.30 降到 0.12：脊线周期约 10px 时，0.30 的压缩会把
        # 脊线从 ~48 抬到 ~110 以上，整张图看起来"发灰、发闷"；
        # 0.12 仍然模拟光学传感器的对比度损失，但保得住细节。
        gain = np.float32(max(0.15, 1.0 - 0.12 * s))
        work = np.float32(255.0) - (np.float32(255.0) - work) * gain

        # 2) 径向暗角（中心 1，四角 1 - 0.40 s）
        vignette = np.clip(1.0 - np.float32(0.40 * s) * _radius_map(h, w) ** 2, 0.05, 1.0)
        work = work * vignette

        # 3) 轻微乘性散斑（0.02 而不是 0.04：脊线变细后，4% 的散斑
        #    会在平坦的谷线区形成明显"砂纸感"，看着像脏，而不是真实采集）
        speckle = rng.standard_normal((h, w), dtype=np.float32) * np.float32(0.02 * s)
        work = work * (np.float32(1.0) + speckle)

        return _finalize(work, dtype, unit_range)


class CapacitiveSensor(SensorProfile):
    """电容传感器：高频噪声 + 少量椒盐 + 轻微行噪声。

    行噪声用一维随机游走（能量集中在低频）模拟逐行读出偏置，与逐像素高斯噪声
    的高频特性互补。
    """

    name = "capacitive"
    label = "电容传感器"

    def apply(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        strength: float = 1.0,
    ) -> np.ndarray:
        """施加电容传感器效果。

        Args:
            image: 输入灰度图，``uint8 (H, W)``。
            rng: 随机源。
            strength: 强度，``0`` 时逐像素直通。

        Returns:
            处理后图像，dtype 与输入一致。
        """
        s = _clip_strength(strength)
        if s <= 0.0:
            return np.array(image, copy=True)

        work, dtype, unit_range = _prepare(image)
        h, w = work.shape

        # 1) 高频读出噪声
        work = work + rng.standard_normal((h, w), dtype=np.float32) * np.float32(6.0 * s)

        # 2) 少量椒盐（坏点）
        amount = 0.004 * s
        selected = rng.random((h, w)) < amount
        values = np.where(
            rng.random((h, w)) < 0.5,
            np.float32(255.0),
            np.float32(0.0),
        )
        work = np.where(selected, values, work)

        # 3) 轻微行噪声（逐行偏置，随机游走 + 归一化）
        raw = np.cumsum(rng.standard_normal((h, 1), dtype=np.float32), axis=0)
        raw = raw - np.float32(raw.mean())
        std = float(raw.std())
        if std > 1e-6:
            raw = raw / np.float32(std)
        else:  # pragma: no cover - 极端退化输入
            raw = np.zeros_like(raw)
        work = work + raw * np.float32(2.5 * s)

        return _finalize(work, dtype, unit_range)


class ThermalSensor(SensorProfile):
    """热感传感器：低频漂移 + 行方向条带 + 非线性响应。

    非线性响应在墨迹域做 Gamma 变换（``gamma = 1 + 0.45 s``）：背景 ``255`` 保持不变，
    脊线变浅，模拟热感传感器偏低的反差；漂移与条带为加性低频分量。
    """

    name = "thermal"
    label = "热感传感器"

    def apply(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        strength: float = 1.0,
    ) -> np.ndarray:
        """施加热感传感器效果。

        Args:
            image: 输入灰度图，``uint8 (H, W)``。
            rng: 随机源。
            strength: 强度，``0`` 时逐像素直通。

        Returns:
            处理后图像，dtype 与输入一致。
        """
        s = _clip_strength(strength)
        if s <= 0.0:
            return np.array(image, copy=True)

        work, dtype, unit_range = _prepare(image)
        h, w = work.shape

        # 1) 非线性响应（墨迹域 Gamma，背景保持白）
        ink = np.clip(np.float32(255.0) - work, 0.0, 255.0) / np.float32(255.0)
        ink = np.float32(255.0) * np.power(ink, np.float32(1.0 + 0.45 * s))
        work = np.float32(255.0) - ink

        # 2) 低频漂移
        drift = _low_freq_field(h, w, rng, sigma=max(4.0, 0.05 * min(h, w)))
        work = work + drift * np.float32(9.0 * s)

        # 3) 行方向条带（沿行方向延伸，随 y 变化）
        rows = np.arange(h, dtype=np.float32)[:, None]
        period_a = max(6.0, 0.10 * h) * float(rng.uniform(0.8, 1.25))
        period_b = max(4.0, 0.035 * h) * float(rng.uniform(0.8, 1.25))
        phase_a = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_b = float(rng.uniform(0.0, 2.0 * np.pi))
        band = np.float32(0.65) * np.sin(
            np.float32(2.0 * np.pi) * rows / np.float32(period_a) + np.float32(phase_a)
        ) + np.float32(0.35) * np.sin(
            np.float32(2.0 * np.pi) * rows / np.float32(period_b) + np.float32(phase_b)
        )
        work = work + band * np.float32(5.0 * s)

        return _finalize(work, dtype, unit_range)


#: 传感器注册表：``{注册名: 类}``。
SENSORS: dict[str, type[SensorProfile]] = {
    "optical": OpticalSensor,
    "capacitive": CapacitiveSensor,
    "thermal": ThermalSensor,
    "none": NoneSensor,
}

#: 常见别名 -> 注册名。
_ALIASES: dict[str, str] = {
    "": "none",
    "identity": "none",
    "null": "none",
    "no": "none",
    "light": "optical",
    "condenser": "capacitive",
    "heat": "thermal",
}


def get_sensor(name: str) -> SensorProfile:
    """按名称实例化传感器；未知名回退 :class:`NoneSensor`。

    Args:
        name: 传感器注册名（大小写不敏感，支持少量别名）。

    Returns:
        传感器实例；名称无法识别时返回 :class:`NoneSensor` 实例。
    """
    key = str(name).strip().lower() if name is not None else "none"
    key = _ALIASES.get(key, key)
    return SENSORS.get(key, NoneSensor)()
