"""捺印形变：皮肤压力位移场与几何组合变换。

本模块模拟手指按压采集面时的皮肤形变（捺印效果）。核心是**双曲正切径向压力模型**：
手指中心受压缩、边缘向外扩展，再叠加低频平滑随机场（弹性抖动），
最后用 :func:`cv2.remap` 做一次重采样，避免多次插值累积模糊。

坐标约定（全模块统一）：

* 位移场 ``(dx, dy)`` 为 ``float32``、形状 ``(H, W)``、单位像素；
* 方向为 **目标像素 -> 源像素** 的采样偏移，即 ``map_x = x + dx``、``map_y = y + dy``，
  与 :func:`cv2.remap` / :func:`cv2.warpAffine` 的约定一致；
* 旋转角单位为**度**，矩阵形式与 :func:`cv2.getRotationMatrix2D` 一致，
  且采样方向与 :func:`cv2.warpAffine`（内部对矩阵求逆）保持一致。

数值约定：所有函数都不修改传入数组，且返回值保证有限（无 NaN/Inf）。

参考文献：

* Maltoni D., Maio D., Jain A. K., Prabhakar S. *Handbook of Fingerprint
  Recognition*, 3rd ed., Springer, 2022, Ch. 2（皮肤弹性形变与捺印失真建模）。
* Cappelli R., Maio D., Maltoni D. "SFinGe: an Approach to Synthetic Fingerprint
  Generation", *Proc. International Workshop on Biometric Technologies*, 2004.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

__all__ = [
    "deform_image",
    "impression_transform",
    "map_coordinates",
    "pressure_field",
]

#: 双曲正切径向模型的斜率 ``k``（见契约 4.1 的 scale 公式）。
_PRESSURE_K: float = 4.0

#: ``pressure`` 的上限；超出后截断，保证 scale 曲线不被无限放大。
_PRESSURE_MAX: float = 1.0

#: ``impression_transform`` 中弹性抖动的默认空间相关长度（像素）。
_DEFAULT_ELASTIC_SIGMA: float = 28.0

#: ``impression_transform`` 中弹性抖动的默认幅度（像素，单位标准差）。
_DEFAULT_ELASTIC_ALPHA: float = 9.0

#: 传给 ``cv2.remap`` 的坐标绝对值上限，防止定点化溢出（超界像素本就取边界值）。
_COORD_LIMIT: float = float(1 << 20)


def _positive_int(value: int | float, name: str) -> int:
    """把尺寸参数校验为正整数。

    Args:
        value: 待校验的尺寸值。
        name: 参数名（用于错误信息）。

    Returns:
        校验通过的正整数。

    Raises:
        ValueError: 尺寸不是正整数时抛出。
    """
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 防御式分支
        raise ValueError(f"{name} 必须为正整数，收到 {value!r}") from exc
    if out <= 0:
        raise ValueError(f"{name} 必须为正整数，收到 {value!r}")
    return out


def _finite(value: float, default: float = 0.0) -> float:
    """把非有限标量折叠为默认值。"""
    out = float(value)
    return out if np.isfinite(out) else float(default)


def _sanitize(field: np.ndarray) -> np.ndarray:
    """把数组转为 ``float32`` 并清除 NaN/Inf（不修改入参）。"""
    out = np.nan_to_num(np.asarray(field, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(out, dtype=np.float32)


def _coordinate_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """生成 ``float32`` 的像素坐标网格。

    Args:
        height: 图像高度 ``H``。
        width: 图像宽度 ``W``。

    Returns:
        ``(xx, yy)``：均为 ``float32 (H, W)``，``xx`` 沿列方向递增，``yy`` 沿行方向递增。
    """
    xx, yy = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return xx, yy


def _smooth_random_field(
    height: int,
    width: int,
    rng: np.random.Generator,
    sigma: float,
) -> np.ndarray:
    """生成单位标准差的低频平滑随机场（弹性抖动基底）。

    实现方式为「粗网格采样 + 三次插值上采样」：相关系长度约等于 ``sigma`` 像素，
    比大核高斯模糊快得多，且完全由 ``rng`` 决定，可复现。

    Args:
        height: 场的高度 ``H``。
        width: 场的宽度 ``W``。
        rng: 随机源。
        sigma: 期望的空间相关长度（像素），``<=0`` 时按 1.0 处理。

    Returns:
        ``float32 (H, W)``，均值 0、标准差 1（退化情形返回全 0）、取值裁剪到 ``[-4, 4]``。
    """
    sig = _finite(sigma, 1.0)
    if sig <= 0.0:
        sig = 1.0
    step = max(2.0, sig * 0.5)
    gh = int(np.clip(np.ceil(height / step) + 1.0, 2, height))
    gw = int(np.clip(np.ceil(width / step) + 1.0, 2, width))

    raw = rng.standard_normal((gh, gw), dtype=np.float32)
    if (gh, gw) != (height, width):
        field = cv2.resize(raw, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        field = raw.astype(np.float32, copy=True)

    field = field - np.float32(field.mean())
    std = float(field.std())
    if std > 1e-6:
        field = field / np.float32(std)
    else:  # pragma: no cover - 仅在极端退化输入下触发
        field = np.zeros_like(field, dtype=np.float32)
    return np.clip(field, -4.0, 4.0).astype(np.float32, copy=False)


def pressure_field(
    height: int,
    width: int,
    pressure: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    *,
    elastic_sigma: float = _DEFAULT_ELASTIC_SIGMA,
    elastic_alpha: float = _DEFAULT_ELASTIC_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """按压力度生成位移场。

    模型：双曲正切径向模型（中心受压缩、边缘向外扩展）::

        r_norm = 归一化到 [0, 1] 的到手指中心的距离
        scale(r) = 1 + p * (tanh(k * (r_norm - 0.5)) * 2)   # 中心 < 1, 边缘 > 1
        dx = (scale - 1) * (x - cx),  dy = (scale - 1) * (y - cy)

    即 ``scale < 1`` 时源采样坐标被压缩向中心，``scale > 1`` 时源采样坐标向外扩展。
    随后叠加低频平滑随机场（弹性抖动）模拟皮肤的非刚性形变；抖动幅度同样与
    ``pressure`` 成正比，因此整场严格线性于 ``pressure``：

    * ``pressure=0`` 时返回**严格全零**的位移场（可直接作为直通）；
    * 中心区域的平均位移随 ``pressure`` 单调递增。

    手指中心默认取图像中心；给定 ``mask`` 时改用掩模质心，``r_norm`` 的归一化半径
    取掩模内像素到中心的最大距离（无掩模时取半对角线）。掩模之外的位移严格置 0。

    Args:
        height: 图像高度 ``H``。
        width: 图像宽度 ``W``。
        pressure: 按压力度，截断到 ``[0, 1]``；建议实际使用 ``[0, 0.5]``，
            过大（接近 1）时中心 ``scale`` 逼近 0，会出现折叠式极端形变。
        rng: 随机源（:class:`numpy.random.Generator`）。
        mask: 可选手指掩模，``uint8 (H, W)``，非 0 视为手指区域。
        elastic_sigma: 弹性抖动的空间相关长度（像素），默认 28.0。
        elastic_alpha: 弹性抖动的幅度（像素，单位标准差），默认 9.0。

    Returns:
        ``(dx, dy)``：``float32 (H, W)``，单位为像素（目标像素 -> 源像素的偏移）。

    Raises:
        ValueError: 尺寸非正或 ``mask`` 形状不匹配时抛出。
    """
    h = _positive_int(height, "height")
    w = _positive_int(width, "width")

    p = float(np.clip(_finite(pressure, 0.0), 0.0, _PRESSURE_MAX))

    weight: np.ndarray | None = None
    if mask is not None:
        m = np.asarray(mask)
        if m.shape != (h, w):
            raise ValueError(f"mask 形状 {m.shape} 与位移场 {(h, w)} 不一致")
        inside = m > 0
        weight = inside.astype(np.float32)

    # ---- 手指中心与归一化半径 ----------------------------------------------
    cx = np.float32((w - 1) / 2.0)
    cy = np.float32((h - 1) / 2.0)
    ref = float(np.hypot(float(cx), float(cy)))
    if weight is not None:
        ys, xs = np.nonzero(weight > 0)
        if xs.size > 0:
            cx = np.float32(xs.mean())
            cy = np.float32(ys.mean())
            ref = float(np.hypot(xs - float(cx), ys - float(cy)).max())
    if not np.isfinite(ref) or ref < 1.0:
        ref = max(1.0, float(np.hypot(w, h)))

    # ---- 双曲正切径向模型 --------------------------------------------------
    xx, yy = _coordinate_grid(h, w)
    ux = xx - cx
    uy = yy - cy
    r_norm = np.sqrt(ux * ux + uy * uy) / np.float32(ref)
    scale = 1.0 + p * 2.0 * np.tanh(_PRESSURE_K * (r_norm - np.float32(0.5)))
    dx = ((scale - 1.0) * ux).astype(np.float32, copy=False)
    dy = ((scale - 1.0) * uy).astype(np.float32, copy=False)

    # ---- 低频弹性抖动（幅度与 pressure 成正比） ----------------------------
    amp = np.float32(_finite(elastic_alpha, _DEFAULT_ELASTIC_ALPHA) * p)
    dx += _smooth_random_field(h, w, rng, elastic_sigma) * amp
    dy += _smooth_random_field(h, w, rng, elastic_sigma) * amp

    if weight is not None:
        dx *= weight
        dy *= weight

    return _sanitize(dx), _sanitize(dy)


def deform_image(
    image: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    border: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """用 :func:`cv2.remap` 施加位移场，保持尺寸与 dtype。

    Args:
        image: 输入图像，``(H, W)`` 灰度或 ``(H, W, C)`` 多通道；``uint8`` /
            ``float32`` 直接重采样，其他 dtype 先转 ``float32`` 再转回。
        dx: 水平位移场，``float32 (H, W)``，单位像素。
        dy: 垂直位移场，``float32 (H, W)``，单位像素。
        border: 边界填充模式，默认 :data:`cv2.BORDER_REPLICATE`
            （指纹背景为白，避免出现黑边）。

    Returns:
        与 ``image`` 形状、dtype 相同的新数组；不修改 ``image``。

    Raises:
        ValueError: ``image`` 维度不是 2/3，或位移场形状与图像不匹配时抛出。
    """
    src = np.asarray(image)
    if src.ndim not in (2, 3):
        raise ValueError(f"image 必须是 (H, W) 或 (H, W, C)，收到形状 {src.shape}")
    h, w = int(src.shape[0]), int(src.shape[1])

    fx = _sanitize(dx)
    fy = _sanitize(dy)
    if fx.shape != (h, w) or fy.shape != (h, w):
        raise ValueError(
            f"位移场形状 {(fx.shape, fy.shape)} 与图像 {(h, w)} 不一致"
        )

    xx, yy = _coordinate_grid(h, w)
    map_x = np.clip(xx + fx, -_COORD_LIMIT, _COORD_LIMIT)
    map_y = np.clip(yy + fy, -_COORD_LIMIT, _COORD_LIMIT)

    native = src.dtype in (np.dtype(np.uint8), np.dtype(np.float32))
    work = src if native else src.astype(np.float32)
    out = cv2.remap(
        work,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=int(border),
    )
    if out.dtype != src.dtype:
        out = out.astype(src.dtype)
    if out.dtype.kind == "f":
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def impression_transform(
    height: int,
    width: int,
    pressure: float,
    rotation: float,
    shift: tuple[float, float],
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
    elastic_sigma: float = _DEFAULT_ELASTIC_SIGMA,
    elastic_alpha: float = _DEFAULT_ELASTIC_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """组合变换：旋转 + 平移（仿射）+ 压力位移场。

    仿射部分绕图像中心旋转、再整体平移，压力部分由 :func:`pressure_field` 给出，
    两者**线性相加**。为便于与 OpenCV 组合，仿射部分等价于
    ``cv2.warpAffine(image, M, (width, height))``，其中 ``M`` 由
    :func:`cv2.getRotationMatrix2D` 给出（注意 ``warpAffine`` 内部会对 ``M`` 求逆，
    本函数已按同样约定取采样方向）::

        M = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), rotation, 1.0)
        M[0, 2] += shift[0]; M[1, 2] += shift[1]

    当 ``pressure=0``、``rotation=0``、``shift=(0, 0)`` 时返回严格全零场，
    此时 :func:`deform_image` 为逐像素直通。

    Args:
        height: 图像高度 ``H``。
        width: 图像宽度 ``W``。
        pressure: 按压力度，见 :func:`pressure_field`。
        rotation: 旋转角度（**度**），正值方向与 :func:`cv2.getRotationMatrix2D` 一致
            （逆时针）；对图像内容而言即「逆时针旋转 rotation 度」。
        shift: 内容平移量 ``(tx, ty)``（像素，+x 向右、+y 向下）；传入 ``None``
            等价于 ``(0, 0)``。位移场是「目标 -> 源」方向，内部取负号。
        rng: 随机源。
        mask: 可选手指掩模，仅约束压力位移场（仿射部分作用于整图）。
        elastic_sigma: 弹性抖动的空间相关长度（像素）。
        elastic_alpha: 弹性抖动的幅度（像素，单位标准差）。

    Returns:
        ``(dx, dy)``：``float32 (H, W)`` 总位移场（像素），可直接用于
        :func:`deform_image` 与 :func:`map_coordinates`。

    Raises:
        ValueError: 尺寸非正、``mask`` 形状不匹配或 ``shift`` 长度不为 2 时抛出。
    """
    h = _positive_int(height, "height")
    w = _positive_int(width, "width")

    dx, dy = pressure_field(
        h,
        w,
        pressure,
        rng,
        mask,
        elastic_sigma=elastic_sigma,
        elastic_alpha=elastic_alpha,
    )

    theta = np.deg2rad(_finite(rotation, 0.0))
    ca = np.float32(np.cos(theta))
    sa = np.float32(np.sin(theta))

    if shift is None:
        sx, sy = 0.0, 0.0
    else:
        try:
            sx, sy = float(shift[0]), float(shift[1])
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError(f"shift 必须是长度为 2 的序列，收到 {shift!r}") from exc
        sx, sy = _finite(sx, 0.0), _finite(sy, 0.0)

    xx, yy = _coordinate_grid(h, w)
    # 与 cv2.warpAffine 一致：先平移到源点，再绕图像中心做逆旋转采样
    ux = xx - (np.float32((w - 1) / 2.0) + np.float32(sx))
    uy = yy - (np.float32((h - 1) / 2.0) + np.float32(sy))

    dx += (ca - np.float32(1.0)) * ux - sa * uy - np.float32(sx)
    dy += sa * ux + (ca - np.float32(1.0)) * uy - np.float32(sy)

    return _sanitize(dx), _sanitize(dy)


def map_coordinates(
    dx: np.ndarray,
    dy: np.ndarray,
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """把源图坐标经位移场映射到目标图坐标（用于标注同步变换）。

    对每个点 ``p``，在位移场中双线性插值得到 ``(ddx, ddy)``，返回 ``(x+ddx, y+ddy)``。
    越界点被裁剪到图像范围内；零位移场时为恒等映射。

    注意：本函数按契约把插值结果**相加**（源 -> 目标）。同一位移场在
    :func:`deform_image` 中按「目标 -> 源」重采样，两者互为近似逆映射；若需让标注
    与形变后的图像严格同步，请传入取负后的位移场 ``(-dx, -dy)``。

    Args:
        dx: 水平位移场，``float32 (H, W)``。
        dy: 垂直位移场，``float32 (H, W)``。
        points: 源图坐标序列 ``[(x, y), ...]``，允许浮点。

    Returns:
        ``[(x', y'), ...]``，与 ``points`` 等长；元素为 Python ``float``，
        已裁剪到 ``[0, W-1] x [0, H-1]``。

    Raises:
        ValueError: 两个位移场形状不一致、不是二维，或 ``points`` 元素长度不为 2 时抛出。
    """
    fx = _sanitize(dx)
    fy = _sanitize(dy)
    if fx.ndim != 2 or fy.ndim != 2 or fx.shape != fy.shape:
        raise ValueError(f"位移场必须是同形状的二维数组，收到 {fx.shape} / {fy.shape}")

    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return []
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points 必须是 (N, 2) 的坐标序列，收到形状 {pts.shape}")

    h, w = int(fx.shape[0]), int(fx.shape[1])
    x = np.clip(np.nan_to_num(pts[:, 0], nan=0.0, posinf=0.0, neginf=0.0), 0.0, float(w - 1))
    y = np.clip(np.nan_to_num(pts[:, 1], nan=0.0, posinf=0.0, neginf=0.0), 0.0, float(h - 1))

    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    def _sample(field: np.ndarray) -> np.ndarray:
        v00 = field[y0, x0]
        v01 = field[y0, x1]
        v10 = field[y1, x0]
        v11 = field[y1, x1]
        top = v00 * (1.0 - wx) + v01 * wx
        bottom = v10 * (1.0 - wx) + v11 * wx
        return top * (1.0 - wy) + bottom * wy

    out_x = x + _sample(fx).astype(np.float64)
    out_y = y + _sample(fy).astype(np.float64)
    out_x = np.clip(np.nan_to_num(out_x, nan=0.0, posinf=0.0, neginf=0.0), 0.0, float(w - 1))
    out_y = np.clip(np.nan_to_num(out_y, nan=0.0, posinf=0.0, neginf=0.0), 0.0, float(h - 1))

    return [(float(a), float(b)) for a, b in zip(out_x, out_y, strict=True)]
