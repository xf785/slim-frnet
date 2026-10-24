"""标注数组的紧凑量化编解码。

方向场与频率图若直接存 ``float32`` 的 ``.npy``，单个样本就要 600 KB 以上；
本模块把它们量化成 ``uint8`` PNG（体积降到约 1%，且任何图像工具都能查看），
需要时用伴随存储的 ``*_meta.json`` 还原到物理量。

约定：
* 方向场：弧度 ``[0, π)`` → ``uint8``，``value = round(angle / π * 255)``，
  量化步长约 0.7°，远小于方向场的估计精度。
* 频率图：``[min_freq, max_freq]`` 线性映射到 ``uint8``，步长由伴随元数据记录。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

__all__ = ["encode_orientation", "decode_orientation", "encode_float_map", "decode_float_map",
           "save_array_png", "load_array_png"]

_PI = float(np.pi)


# ---------------------------------------------------------------------------
# 方向场
# ---------------------------------------------------------------------------
def encode_orientation(orientation: np.ndarray) -> np.ndarray:
    """方向场 ``float32 [0, π)`` → ``uint8 [0, 255]``。"""
    ang = np.mod(np.asarray(orientation, dtype=np.float32), _PI)
    return np.clip(np.round(ang / _PI * 255.0), 0, 255).astype(np.uint8)


def decode_orientation(encoded: np.ndarray) -> np.ndarray:
    """``uint8`` → 方向场 ``float32 [0, π)`` 弧度。"""
    return (encoded.astype(np.float32) / 255.0 * _PI).astype(np.float32)


# ---------------------------------------------------------------------------
# 任意浮点图（频率图 / 质量图）
# ---------------------------------------------------------------------------
def encode_float_map(
    data: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """浮点图 → ``uint8`` + 反量化元数据。

    Args:
        data: 任意 ``float32/float64`` 二维数组。
        vmin / vmax: 映射范围；给 ``None`` 时取数据自身的 0.5% ~ 99.5% 分位。

    Returns:
        ``(encoded, meta)``，``meta`` 含 ``vmin``/``vmax``。
    """
    arr = np.asarray(data, dtype=np.float32)
    if vmin is None or vmax is None:
        lo = float(np.percentile(arr, 0.5)) if arr.size else 0.0
        hi = float(np.percentile(arr, 99.5)) if arr.size else 1.0
    else:
        lo, hi = float(vmin), float(vmax)
    if hi - lo < 1e-12:
        hi = lo + 1e-6
    enc = np.clip(np.round((arr - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)
    return enc, {"vmin": lo, "vmax": hi}


def decode_float_map(encoded: np.ndarray, meta: Dict[str, float]) -> np.ndarray:
    """``uint8`` + 元数据 → 原始量纲的 ``float32`` 数组。"""
    lo = float(meta.get("vmin", 0.0))
    hi = float(meta.get("vmax", 1.0))
    return (encoded.astype(np.float32) / 255.0 * (hi - lo) + lo).astype(np.float32)


# ---------------------------------------------------------------------------
# 落盘（原子写，Unicode 安全）
# ---------------------------------------------------------------------------
def save_array_png(path: Path, data: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
    """把 ``uint8`` 数组写成 PNG；``meta`` 非空时同时写 ``<stem>_meta.json``。"""
    from ..legacy.util.image_utils import imwrite_unicode

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件名不能以 .png 结尾之前带额外后缀（OpenCV 会按扩展名找编码器失败）
    tmp = path.with_name(f"{path.stem}.tmp")
    if not imwrite_unicode(str(tmp), data, ext=".png"):
        raise OSError(f"写入数组 PNG 失败: {path}")
    tmp.replace(path)
    if meta:
        meta_path = path.with_name(path.stem + "_meta.json")
        meta_tmp = meta_path.with_name(f"{meta_path.stem}.tmp")
        meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_tmp.replace(meta_path)


def load_array_png(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """读取数组 PNG（以及同名伴随元数据）。"""
    from ..legacy.util.image_utils import imread_unicode

    path = Path(path)
    arr = imread_unicode(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"无法读取: {path}")
    meta: Dict[str, Any] = {}
    meta_path = path.with_name(path.stem + "_meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    return arr, meta
