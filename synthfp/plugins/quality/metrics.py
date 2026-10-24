"""质量评估底层度量函数（纯 ``numpy`` + ``cv2``，无 scipy / scikit-image 依赖）。

本模块提供 NFIQ2 风格评分所需的五个度量：

* :func:`estimate_orientation` —— 梯度结构张量方向场 + 一致性
* :func:`ridge_clarity` —— Gabor 响应强度（脊线清晰度）
* :func:`local_contrast` —— 局部标准差（归一化对比度）
* :func:`usable_area` —— 有效（非饱和、非背景）面积占比
* :func:`estimate_ridge_period` —— 沿方向场正交方向的局部自相关（平均脊线周期）

约定（见 ``CONTRACTS.md`` 第 0 节）
-----------------------------------
* 输入图像可为 ``uint8``（0~255）或 ``float32``（0~1），自动识别；**背景为白**。
* 掩模 ``uint8 (H, W)``，取值 ``{0, 255}``；``mask=None`` 表示整幅图有效，
  而**全零掩模**表示没有任何有效像素。
* 所有函数不修改传入数组；遇到退化输入（空掩模 / 纯色图 / 全零图 / 任一边 < 3 px）
  一律安全返回 ``0.0``，不抛异常、不产生 ``NaN`` / ``Inf``。

参考文献
--------
* Hong, Wan & Jain, *Fingerprint Image Enhancement: Algorithm and Performance
  Evaluation*, IEEE TPAMI 20(8), 1998（方向场、Gabor 增强）。
* Bazen & Gerez, *Systematic Methods for the Computation of the Directional
  Fields and Singular Points of Fingerprints*, IEEE TPAMI 24(7), 2002
  （梯度结构张量与一致性）。
* Tabassi, Wilson & Watson, *NIST Fingerprint Image Quality (NFIQ)*, NISTIR 7151,
  2004（分项加权评分思路）。
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "to_float01",
    "estimate_orientation",
    "orientation_certainty",
    "ridge_clarity",
    "local_contrast",
    "usable_area",
    "estimate_ridge_period",
    "DEFAULT_RIDGE_PERIOD",
    "CONTRAST_REF",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_EPS = 1e-8
"""防止除零的小量。"""

MIN_SIDE = 3
"""任一边小于该值视为“不可评估”，所有度量返回 0.0。"""

DEFAULT_RIDGE_PERIOD = 10.0
"""脊线周期回退值（px）。可评估但测不到周期性时使用。"""

SAT_LOW = 0.02
"""饱和（过暗 / 背景为黑）阈值——低于该灰度视为无效像素。"""

SAT_HIGH = 0.98
"""饱和 / 背景（纯白）阈值——高于该灰度视为无效像素。"""

CONTRAST_REF = 0.25
"""对比度归一化参考值：局部标准差达到该值即记满分。"""

_PERIOD_MIN_TAU = 3
_PERIOD_MAX_TAU = 24
_PERIOD_PEAK_RATIO = 0.25
"""自相关峰值得超过归一化自相关 0 阶的该比例才算“测到周期”。"""

_GABOR_ORIENTATIONS = 8
"""Gabor 滤波器组的方向数。"""

_GABOR_SIGMA_SCALE = 0.55
"""高斯包络 sigma = 该系数 × 脊线周期。"""

_GABOR_TRUNCATE = 3.0
"""包络截断倍数：核半径 = ceil(截断倍数 × sigma)。"""

_BANK_CACHE: dict[tuple[float, int], tuple[np.ndarray, float]] = {}
"""Gabor 滤波器组缓存：``(周期, 方向数) -> (核, 峰值增益)``（只读，可安全共享）。"""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _to_gray_float(image: np.ndarray) -> np.ndarray:
    """把多通道图像转成 2 维 ``float32``（不做范围归一化）。"""
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] in (3, 4):
            code = cv2.COLOR_BGR2GRAY if arr.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            arr = cv2.cvtColor(arr.astype(np.float32, copy=False), code)
        else:
            arr = arr[:, :, 0]
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def to_float01(image: np.ndarray) -> np.ndarray:
    """把图像统一转换为 ``float32``、范围 ``[0, 1]`` 的单通道灰度副本。

    Args:
        image: 输入图像。支持 ``bool``、``uint8``(0~255)、``uint16``(0~65535)、
            ``float``(0~1 或 0~255 或任意范围) 以及 1/3/4 通道彩色图。

    Returns:
        ``float32`` 数组，形状 ``(H, W)``，取值 ``[0, 1]``，且保证全部有限。

    Note:
        返回值一定是新数组（不修改输入）；``NaN`` / ``Inf`` 会被安全替换。
    """
    arr = _to_gray_float(image)
    if arr.dtype == np.bool_:
        return arr.astype(np.float32, copy=True)
    if arr.dtype == np.uint8:
        return (arr.astype(np.float32) / 255.0).astype(np.float32)
    if arr.dtype == np.uint16:
        return (arr.astype(np.float32) / 65535.0).astype(np.float32)
    out = arr.astype(np.float32, copy=True)
    if out.size == 0:
        return out
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    lo = float(out.min())
    hi = float(out.max())
    if hi > 1.0 + 1e-6:
        if hi <= 255.0 + 1e-6 and lo >= -1e-6:
            out = out / 255.0
        elif hi - lo > _EPS:
            out = (out - lo) / (hi - lo)
        else:
            out = np.zeros_like(out)
    elif lo < -1e-6:
        if hi - lo > _EPS:
            out = (out - lo) / (hi - lo)
        else:
            out = np.zeros_like(out)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _region(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """返回与图像同形状的有效区域布尔掩模。

    Args:
        mask: ``None`` 表示整幅图有效；否则取非零像素为有效。形状不匹配时用
            最近邻缩放对齐（掩模只是粗粒度区域指示）。
        shape: 目标形状 ``(H, W)``。

    Returns:
        ``bool`` 数组，形状 ``(H, W)``；输入为空时返回全 ``False``。
    """
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        return np.zeros((max(h, 0), max(w, 0)), dtype=bool)
    if mask is None:
        return np.ones((h, w), dtype=bool)
    m = np.asarray(mask)
    if m.ndim == 3 and m.shape[2] >= 1:
        m = m[:, :, 0]
    if m.ndim == 0:
        return np.full((h, w), bool(m), dtype=bool)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    if m.shape != (h, w):
        if m.size == 0:
            return np.zeros((h, w), dtype=bool)
        m = cv2.resize(m.astype(np.uint8, copy=False), (w, h), interpolation=cv2.INTER_NEAREST)
    with np.errstate(invalid="ignore"):
        return np.asarray(m > 0, dtype=bool)


def _inner_region(region: np.ndarray, pad: int) -> np.ndarray:
    """把区域向内腐蚀 ``pad`` 像素，避免滤波边界效应对均值的影响。

    Args:
        region: 有效区域布尔掩模。
        pad: 腐蚀半径（px）。

    Returns:
        腐蚀后的区域；若腐蚀为空则退回原区域。
    """
    pad = int(max(0, pad))
    if pad <= 0 or region.size == 0:
        return region
    k = 2 * pad + 1
    if min(region.shape) <= k:
        return region
    kernel = np.ones((k, k), np.uint8)
    eroded = cv2.erode(
        region.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    inner = eroded > 0
    return inner if inner.any() else region


def _finite(value: float) -> float:
    """把标量夹到有限值（``NaN`` / ``Inf`` → ``0.0``）。"""
    v = float(value)
    return v if np.isfinite(v) else 0.0


def _local_stats(image: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """盒式滤波计算局部均值与局部标准差。

    Args:
        image: ``float32 (H, W)`` 图像。
        window: 窗口边长（px，会被夹到 >= 2）。

    Returns:
        ``(mean, std)``，均为 ``float32 (H, W)``，``std >= 0``。
    """
    h, w = image.shape
    k = int(max(2, min(int(window), max(2, min(h, w)))))
    ksize = (k, k)
    mean = cv2.boxFilter(image, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(
        image * image, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT
    )
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean.astype(np.float32, copy=False), np.sqrt(var).astype(np.float32)


# ---------------------------------------------------------------------------
# 方向场
# ---------------------------------------------------------------------------
def estimate_orientation(
    image: np.ndarray,
    block: int = 16,
    smooth_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """梯度法方向场估计。

    先在尺度 ``smooth_sigma`` 上低通以抑制噪声，再用 Sobel 算子求梯度，
    在 ``block × block`` 窗口内累加梯度结构张量：

    .. math::

        G_{xx}=\\sum g_x^2,\\quad G_{yy}=\\sum g_y^2,\\quad G_{xy}=\\sum g_x g_y

    脊线方向 :math:`\\theta` 与梯度主方向正交，一致性
    :math:`c=\\sqrt{(G_{xx}-G_{yy})^2+4G_{xy}^2}/(G_{xx}+G_{yy})\\in[0,1]`
    （Bazen & Gerez 2002）。

    Args:
        image: 输入图像，``uint8`` 或 ``float32``，形状 ``(H, W)``。
        block: 结构张量累加窗口边长（px），默认 16。
        smooth_sigma: 求梯度前的高斯平滑 sigma（px）；``<= 0`` 表示不平滑。

    Returns:
        ``(orientation, certainty)``：

        * ``orientation``：``float32 (H, W)``，弧度，取值 ``[0, π)``，
          表示**脊线走向**（0 = 水平脊线，π/2 = 垂直脊线）；
        * ``certainty``：``float32 (H, W)``，取值 ``[0, 1]``，梯度一致性。

    Note:
        纯色 / 空 / 极小图像返回全 0（不会产生 ``NaN``）。
    """
    img = to_float01(image)
    h, w = img.shape
    if h < MIN_SIDE or w < MIN_SIDE:
        zeros = np.zeros((h, w), dtype=np.float32)
        return zeros, zeros.copy()

    if smooth_sigma and float(smooth_sigma) > 0:
        blurred = cv2.GaussianBlur(img, (0, 0), float(smooth_sigma))
    else:
        blurred = img

    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    k = int(max(2, min(int(block), max(2, min(h, w)))))
    ksize = (k, k)
    gxx = cv2.boxFilter(gx * gx, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    gyy = cv2.boxFilter(gy * gy, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    gxy = cv2.boxFilter(gx * gy, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    den = gxx + gyy
    num = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4.0 * gxy * gxy, 0.0))
    certainty = np.clip(num / (den + _EPS), 0.0, 1.0)

    vx = 2.0 * gxy
    vy = gxx - gyy
    theta = 0.5 * np.arctan2(vx, vy) + 0.5 * np.pi
    theta = np.mod(theta, np.pi)
    theta = np.where(den > _EPS, theta, 0.0)

    orientation = np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
    orientation = np.mod(orientation, np.pi).astype(np.float32, copy=False)
    certainty = np.nan_to_num(certainty, nan=0.0, posinf=1.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    return orientation, certainty


def orientation_certainty(
    orientation: np.ndarray,
    certainty: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """掩模内方向一致性的均值。

    Args:
        orientation: ``float32 (H, W)`` 方向场（弧度）；仅用于剔除 ``NaN`` 像素，
            保持与 :func:`estimate_orientation` 的成对调用签名一致。
        certainty: ``float32 (H, W)`` 一致性图，取值 ``[0, 1]``。
        mask: 有效区域掩模；``None`` 表示全图有效。全零掩模返回 ``0.0``。

    Returns:
        一致性均值，``float``，取值 ``[0, 1]``；退化输入返回 ``0.0``。
    """
    if certainty is None:
        return 0.0
    cert = np.asarray(certainty)
    if cert.ndim != 2 or cert.size == 0:
        return 0.0
    h, w = cert.shape
    if min(h, w) < MIN_SIDE:
        return 0.0
    region = _region(mask, (h, w))
    if not region.any():
        return 0.0
    cert = np.nan_to_num(
        cert.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0
    )
    if orientation is not None:
        ori = np.asarray(orientation)
        if ori.shape == cert.shape:
            region = region & np.isfinite(ori)
            if not region.any():
                return 0.0
    value = np.clip(cert[region], 0.0, 1.0).mean()
    return float(min(max(_finite(value), 0.0), 1.0))


# ---------------------------------------------------------------------------
# Gabor 滤波器组（脊线清晰度）
# ---------------------------------------------------------------------------
def _gabor_bank(period: float, n_orient: int = _GABOR_ORIENTATIONS) -> tuple[np.ndarray, float]:
    """构造正交（quadrature）Gabor 滤波器组并计算其峰值增益。

    核函数（脊线方向 :math:`\\varphi`，调制轴垂直于脊线）::

        g_e(x, y) = exp(-(x² + y²) / (2σ²)) · cos(2π x'/λ)
        g_o(x, y) = exp(-(x² + y²) / (2σ²)) · sin(2π x'/λ)
        x' = -x·sin φ + y·cos φ,   σ = 0.55·λ

        每个核去直流后归一化到单位 L2 范数；``(g_e, g_o)`` 构成解析对，
        其能量 ``sqrt(R_e² + R_o²)`` 对匹配频率的正弦是空间常数。

    Args:
        period: 脊线周期 λ（px）。
        n_orient: 方向数（默认 8，即每 22.5° 一个方向）。

    Returns:
        ``(bank, gain)``：``bank`` 形状 ``(n_orient, 2, k, k)``（第 1 维为方向，
        第 2 维为偶/奇对称核）；``gain`` 为单位幅度匹配正弦的峰值响应幅值。

    Note:
        结果按 ``(period, n_orient)`` 缓存（只读，不随调用变化）。
    """
    key = (round(float(period), 4), int(n_orient))
    cached = _BANK_CACHE.get(key)
    if cached is not None:
        return cached

    lam = float(np.clip(abs(_finite(period)) or DEFAULT_RIDGE_PERIOD, 3.0, 40.0))
    sigma = float(np.clip(_GABOR_SIGMA_SCALE * lam, 1.0, 20.0))
    radius = int(np.ceil(_GABOR_TRUNCATE * sigma))
    ksize = 2 * radius + 1
    coords = np.arange(ksize, dtype=np.float32) - radius
    gx, gy = np.meshgrid(coords, coords)
    envelope = np.exp(-(gx * gx + gy * gy) / (2.0 * sigma * sigma)).astype(np.float32)
    k0 = 2.0 * np.pi / lam

    bank = np.zeros((int(n_orient), 2, ksize, ksize), dtype=np.float32)
    gain = 0.0
    for i in range(int(n_orient)):
        phi = np.pi * i / float(n_orient)
        ux, uy = -np.sin(phi), np.cos(phi)
        xr = (gx * ux + gy * uy).astype(np.float32)
        carrier_e = np.cos(k0 * xr).astype(np.float32)
        carrier_o = np.sin(k0 * xr).astype(np.float32)
        for j, carrier in enumerate((carrier_e, carrier_o)):
            kern = envelope * carrier
            kern -= kern.mean()
            norm = float(np.sqrt(np.sum(kern * kern)))
            if norm <= _EPS:
                kern = np.zeros_like(kern)
            else:
                kern = kern / norm
            bank[i, j] = kern
        # 单位幅度、方向匹配正弦的峰值响应 = |DTFT(k0 · u)|
        basis = np.exp(-1j * k0 * xr)
        gain = max(gain, abs(complex(np.sum(bank[i, 0] * basis))),
                   abs(complex(np.sum(bank[i, 1] * basis))))

    result = (bank, float(gain))
    if len(_BANK_CACHE) > 64:
        _BANK_CACHE.clear()
    _BANK_CACHE[key] = result
    return result


def ridge_clarity(
    image: np.ndarray,
    orientation: np.ndarray,
    mask: np.ndarray | None = None,
    period: float = DEFAULT_RIDGE_PERIOD,
) -> float:
    """Gabor 响应强度度量脊线清晰度。

    用匹配脊线周期的正交 Gabor 滤波器组滤波，取方向选择性响应能量，并用
    局部标准差（脊线幅度 :math:`\\sqrt{2}\\,\\sigma_{local}`）归一化：

    .. math::

        \\text{clarity}=\\frac{\\overline{E_{gabor}}}
        {\\sqrt{2}\\,G\\,\\overline{\\sigma_{local}}}

    其中 :math:`G` 是单位幅度匹配正弦的峰值响应（解析计算），因此理想正弦
    条纹图得分约为 1.0，白噪声因能量落在通带外而显著偏低。

    Args:
        image: 输入图像（``uint8`` 或 ``float32``）。
        orientation: ``float32 (H, W)`` 脊线方向场（弧度）；用于逐像素选择
            最匹配的滤波器方向。``None`` 或形状不符时改用滤波器组最大值。
        mask: 有效区域掩模；``None`` 表示全图有效。
        period: 脊线周期（px），默认 10.0。

    Returns:
        ``float``，取值 ``[0, 1]``；退化输入返回 ``0.0``。
    """
    img = to_float01(image)
    h, w = img.shape
    if h < MIN_SIDE or w < MIN_SIDE:
        return 0.0
    region = _region(mask, (h, w))
    if not region.any():
        return 0.0

    lam = _finite(period)
    if lam <= 0.0:
        lam = DEFAULT_RIDGE_PERIOD
    bank, gain = _gabor_bank(lam)
    if gain <= _EPS:
        return 0.0

    n_orient, _, ksize, _ = bank.shape
    energy = np.zeros((h, w), dtype=np.float32)
    use_orientation = (
        orientation is not None
        and np.asarray(orientation).shape == img.shape
        and min(h, w) >= MIN_SIDE
    )
    idx = None
    if use_orientation:
        ori = np.mod(
            np.nan_to_num(
                np.asarray(orientation, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            ),
            np.pi,
        )
        idx = np.rint(ori / (np.pi / float(n_orient))).astype(np.int32) % n_orient

    responses = np.empty((n_orient, h, w), dtype=np.float32)
    for i in range(n_orient):
        even = cv2.filter2D(
            img, cv2.CV_32F, bank[i, 0], borderType=cv2.BORDER_REFLECT
        )
        odd = cv2.filter2D(img, cv2.CV_32F, bank[i, 1], borderType=cv2.BORDER_REFLECT)
        responses[i] = np.sqrt(even * even + odd * odd)

    if idx is not None:
        # 方向场不可靠时的下限保护：不低于最强方向的 50%
        selected = np.take_along_axis(responses, idx[None, :, :], axis=0)[0]
        energy = np.maximum(selected, 0.5 * responses.max(axis=0))
    else:
        energy = responses.max(axis=0)

    _, local_std = _local_stats(img, int(round(1.5 * lam)) | 1)
    inner = _inner_region(region, ksize // 2 + 1)
    e_mean = _finite(float(energy[inner].mean()))
    s_mean = _finite(float(local_std[inner].mean()))
    if s_mean <= _EPS:
        return 0.0
    clarity = e_mean / (np.sqrt(2.0) * gain * s_mean + _EPS)
    return float(min(max(_finite(clarity), 0.0), 1.0))


# ---------------------------------------------------------------------------
# 对比度 / 有效面积
# ---------------------------------------------------------------------------
def local_contrast(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    block: int = 16,
) -> float:
    """局部标准差均值（归一化到 ``[0, 1]``）。

    局部标准差用 ``block × block`` 盒式滤波在同一窗口内计算
    :math:`\\sqrt{E[I^2]-E[I]^2}`，再对有效区域（向内腐蚀半个窗口以避开
    边界）取均值，除以参考值 :data:`CONTRAST_REF` 并截断到 ``[0, 1]``。

    Args:
        image: 输入图像（``uint8`` 或 ``float32``）。
        mask: 有效区域掩模；``None`` 表示全图有效。
        block: 局部统计窗口边长（px），默认 16。

    Returns:
        ``float``，取值 ``[0, 1]``；纯色图 / 空掩模 / 极小图返回 ``0.0``。
    """
    img = to_float01(image)
    h, w = img.shape
    if h < MIN_SIDE or w < MIN_SIDE:
        return 0.0
    region = _region(mask, (h, w))
    if not region.any():
        return 0.0
    k = int(max(2, min(int(block), max(2, min(h, w)))))
    _, std = _local_stats(img, k)
    if not np.any(std[region] > _EPS):
        return 0.0
    inner = _inner_region(region, k // 2)
    value = _finite(float(std[inner].mean())) / CONTRAST_REF
    return float(min(max(value, 0.0), 1.0))


def usable_area(
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """有效（非饱和、非背景）面积占比。

    有效像素定义为 ``SAT_LOW < I < SAT_HIGH``：过亮（纯白背景）与过暗
    （饱和黑块）都视为无效。整幅图完全没有灰度变化（纯色）时视为不含信息，
    返回 ``0.0``。

    Args:
        image: 输入图像（``uint8`` 或 ``float32``）。
        mask: 有效区域掩模；``None`` 表示整幅图作为分母。

    Returns:
        ``float``，取值 ``[0, 1]``；全零图 / 纯色图 / 空掩模 / 极小图返回 ``0.0``。
    """
    img = to_float01(image)
    h, w = img.shape
    if h < MIN_SIDE or w < MIN_SIDE:
        return 0.0
    region = _region(mask, (h, w))
    if not region.any():
        return 0.0
    values = img[region]
    if values.size == 0:
        return 0.0
    if float(values.max() - values.min()) <= _EPS:
        # 纯色区域不携带脊线信息
        return 0.0
    valid = (img > SAT_LOW) & (img < SAT_HIGH)
    ratio = _finite(float(valid[region].sum()) / float(region.sum()))
    return float(min(max(ratio, 0.0), 1.0))


# ---------------------------------------------------------------------------
# 脊线周期
# ---------------------------------------------------------------------------
def estimate_ridge_period(
    image: np.ndarray,
    orientation: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """沿方向场正交方向做局部自相关，估计平均脊线周期（px）。

    对每个滞后 :math:`\\tau`，把去局部均值后的图像沿**垂直于脊线**的方向
    平移 :math:`\\tau` 像素（方向场驱动的 ``cv2.remap``），计算归一化自相关::

        C(τ) = Σ I₀(p)·I₀(p + τ·n(p)) / Σ I₀(p)²

    取首个显著局部极大（``C > 0.25``）的滞后作为周期，并做抛物线插值细化。

    Args:
        image: 输入图像（``uint8`` 或 ``float32``）。
        orientation: ``float32 (H, W)`` 脊线方向场（弧度）；``None`` 时内部调用
            :func:`estimate_orientation` 现场估计。
        mask: 有效区域掩模；``None`` 表示全图有效。

    Returns:
        ``float``：平均脊线周期（px）。

        * 退化输入（空掩模 / 纯色图 / 全零图 / 极小图）返回 ``0.0``；
        * 可评估但测不到明显周期性时返回默认回退值
          :data:`DEFAULT_RIDGE_PERIOD`（10.0）。
    """
    img = to_float01(image)
    h, w = img.shape
    if h < 4 * MIN_SIDE or w < 4 * MIN_SIDE:
        return 0.0
    region = _region(mask, (h, w))
    if not region.any():
        return 0.0

    if orientation is None or np.asarray(orientation).shape != img.shape:
        theta, _ = estimate_orientation(img)
    else:
        theta = np.mod(
            np.nan_to_num(
                np.asarray(orientation, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            ),
            np.pi,
        )

    # 去局部均值（抑制低频背景）
    win = int(min(61, max(9, (min(h, w) // 4) | 1)))
    mean, std = _local_stats(img, win)
    centered = img - mean
    inner = _inner_region(region, _PERIOD_MAX_TAU + 2)
    denom = _finite(float(np.sum(centered[inner] * centered[inner])))
    if denom <= _EPS or _finite(float(std[region].mean())) <= _EPS:
        return 0.0

    nx = -np.sin(theta).astype(np.float32)
    ny = np.cos(theta).astype(np.float32)
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )

    taus = np.arange(_PERIOD_MIN_TAU, _PERIOD_MAX_TAU + 1, dtype=np.float32)
    corr = np.zeros(taus.size, dtype=np.float32)
    for i, tau in enumerate(taus):
        map_x = (grid_x + tau * nx).astype(np.float32)
        map_y = (grid_y + tau * ny).astype(np.float32)
        shifted = cv2.remap(
            centered,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        corr[i] = np.sum(centered[inner] * shifted[inner]) / denom

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(corr)) or float(np.max(corr)) <= _PERIOD_PEAK_RATIO:
        return DEFAULT_RIDGE_PERIOD

    best = -1
    # 从第二个滞后开始扫描，保证总能与左侧滞后比较（避免把单调下降段误判为峰）
    for i in range(1, corr.size):
        left = corr[i - 1]
        right = corr[i + 1] if i + 1 < corr.size else -np.inf
        if corr[i] >= left and corr[i] >= right and corr[i] > _PERIOD_PEAK_RATIO:
            best = i
            break
    if best < 0:
        return DEFAULT_RIDGE_PERIOD

    tau = float(taus[best])
    if 0 < best < corr.size - 1:
        c0, c1, c2 = float(corr[best - 1]), float(corr[best]), float(corr[best + 1])
        curv = c0 - 2.0 * c1 + c2
        if abs(curv) > _EPS:
            tau += float(np.clip(0.5 * (c0 - c2) / curv, -1.0, 1.0))
    period = _finite(tau)
    if period <= 0.0:
        return DEFAULT_RIDGE_PERIOD
    return float(np.clip(period, 1.0, 100.0))
