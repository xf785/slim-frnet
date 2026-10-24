"""通用图像工具函数 — 包含Unicode安全的文件I/O"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Unicode安全的图像文件I/O (解决Windows中文路径问题)
# ---------------------------------------------------------------------------

def imread_unicode(filepath, flags=cv2.IMREAD_UNCHANGED):
    """
    读取图像 — 支持Unicode/中文路径 (替代cv2.imread)

    工作原理: 用numpy从文件读取字节流, 然后通过cv2.imdecode解码,
    避免OpenCV C++后端对Unicode路径的处理缺陷。

    Args:
        filepath: 图像文件路径 (str或Path, 支持中文)
        flags: cv2.IMREAD_GRAYSCALE / cv2.IMREAD_UNCHANGED / cv2.IMREAD_COLOR

    Returns:
        image: ndarray, 失败时返回None
    """
    filepath = str(Path(filepath))
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def imwrite_unicode(filepath, img, params=None, ext=None):
    """
    写入图像 — 支持Unicode/中文路径 (替代cv2.imwrite)

    v2.0 增强：新增 ``ext`` 参数。原子写入需要先写 ``<stem>.tmp`` 再 rename，
    但 OpenCV 的 ``imencode`` 会按路径扩展名查找编码器，``.tmp`` 会直接失败；
    因此调用方可以显式指定真实的编码格式。

    Args:
        filepath: 目标文件路径 (str或Path, 支持中文)
        img: 图像ndarray
        params: 编码参数 (如 [cv2.IMWRITE_PNG_COMPRESSION, 3])
        ext: 编码格式 (如 '.png')；为 None 时从 filepath 推断

    Returns:
        success: bool
    """
    filepath = str(Path(filepath))
    if ext is None:
        ext = Path(filepath).suffix.lower()
    if not ext:
        ext = '.png'
    # 根据扩展名确定编码格式
    try:
        success, data = cv2.imencode(ext, img, params or [])
        if success:
            data.tofile(filepath)
            return True
        return False
    except Exception:
        return False


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """
    min-max归一化 → [0, 1]

    Args:
        image: 输入图像, 任意dtype

    Returns:
        normalized: float32 ndarray, range=[0, 1]
    """
    img = image.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - vmin) / (vmax - vmin)).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    将[0,1]浮点图像转换为uint8 [0,255]

    Args:
        image: float32 ndarray, range=[0, 1]

    Returns:
        uint8 ndarray, range=[0, 255]
    """
    img = np.clip(image, 0, 1)
    return (img * 255).astype(np.uint8)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    确保图像为单通道灰度图

    Args:
        image: 输入图像 (可以是RGB/BGR)

    Returns:
        grayscale: 单通道灰度图
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        if image.shape[2] == 1:
            return image[:, :, 0]
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"不支持的图像维度: {image.shape}")


def resize_preserve_aspect(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """
    保持纵横比的resize (padding to fit)

    Args:
        image: 输入图像
        target_size: (width, height)
        interpolation: cv2插值方法

    Returns:
        resized: 调整尺寸后的图像
    """
    h, w = image.shape[:2]
    tw, th = target_size

    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Padding
    if new_w != tw or new_h != th:
        pad_w = (tw - new_w) // 2
        pad_h = (th - new_h) // 2
        if image.ndim == 2:
            resized = cv2.copyMakeBorder(
                resized, pad_h, th - new_h - pad_h,
                pad_w, tw - new_w - pad_w,
                cv2.BORDER_CONSTANT, value=0
            )
        else:
            resized = cv2.copyMakeBorder(
                resized, pad_h, th - new_h - pad_h,
                pad_w, tw - new_w - pad_w,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

    return resized


def gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    """生成2D高斯核"""
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return (kernel / kernel.sum()).astype(np.float32)
