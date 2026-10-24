"""ISO/IEC 19794-2:2005 细节点模板（简约格式，格式标识 0）导出。

字节布局（全部整数为**大端**）::

    头部 24 字节
      偏移  0  4B  格式标识 "FMR\\0"            (0x46 0x4D 0x52 0x00)
      偏移  4  4B  版本号   " 20\\0"            (0x20 0x32 0x30 0x00)
      偏移  8  4B  记录总长度 = 24 + 6n
      偏移 12  2B  采集设备 ID（本实现固定 0）
      偏移 14  2B  图像宽度（px）
      偏移 16  2B  图像高度（px）
      偏移 18  2B  水平分辨率（ppi）
      偏移 20  2B  垂直分辨率（ppi）
      偏移 22  1B  细节点个数 n（≤ 255）
      偏移 23  1B  预留（0）

    每个细节点 6 字节
      字节 0  [b7..b6] 类型码 | [b5..b4] X 高 2 位 | [b3..b2] Y 高 2 位 | [b1..b0] 预留
      字节 1  X 低 8 位
      字节 2  Y 低 8 位
      字节 3  方向，8 位量化（0~255 对应 [0, 2π)）
      字节 4  质量，0~100
      字节 5  预留（0）

说明
----
* 类型码直接取 :attr:`~synthfp.core.types.Minutia.type_code`（本项目约定
  ``0 = ending``、``1 = bifurcation``），与 ``core/types.py`` 保持单一数据源。
* X / Y 各 10 位（0~1023），覆盖本项目 500~640 px 的图像尺寸；超范围会被裁剪。
* 分辨率与尺寸按 16 位无符号整数存储，同样做裁剪保护。
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Sequence

from ...core.types import Minutia

__all__ = [
    "to_iso_template",
    "ISO_HEADER_SIZE",
    "ISO_MINUTIA_SIZE",
    "ISO_FORMAT_IDENTIFIER",
    "ISO_VERSION",
]

ISO_HEADER_SIZE = 24
"""FMR 记录头部长度（字节）。"""

ISO_MINUTIA_SIZE = 6
"""单条细节点记录长度（字节）。"""

ISO_FORMAT_IDENTIFIER = b"FMR\x00"
"""格式标识（格式标识 0 = 简约格式）。"""

ISO_VERSION = b" 20\x00"
"""版本号 " 20"（ISO/IEC 19794-2:2005）。"""

_MAX_COORD = 1023
"""10 位坐标上限。"""

_MAX_COUNT = 255
"""1 字节细节点计数上限。"""

_TWO_PI = 6.283185307179586


def _clip_int(value: float, low: int, high: int) -> int:
    """把数值安全地转成 ``[low, high]`` 内的整数（``NaN`` → ``low``）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if number != number or number in (float("inf"), float("-inf")):
        return low
    return int(min(max(int(round(number)), low), high))


def _angle_byte(angle: float) -> int:
    """把弧度角量化到 8 位（``0~255`` 均匀覆盖 ``[0, 2π)``）。"""
    try:
        value = float(angle)
    except (TypeError, ValueError):
        return 0
    if value != value or value in (float("inf"), float("-inf")):
        return 0
    value = value % _TWO_PI
    return int(round(value / _TWO_PI * 256.0)) % 256


def _quality_byte(quality: float) -> int:
    """把 ``[0, 1]`` 的质量映射到 ``0~100``。"""
    try:
        value = float(quality)
    except (TypeError, ValueError):
        return 0
    if value != value or value in (float("inf"), float("-inf")):
        return 0
    return _clip_int(min(max(value, 0.0), 1.0) * 100.0, 0, 100)


def to_iso_template(
    minutiae: Sequence[Minutia],
    width: int,
    height: int,
    ppi: int = 500,
) -> bytes:
    """把细节点列表编码为 ISO/IEC 19794-2:2005 简约格式字节流。

    Args:
        minutiae: :class:`~synthfp.core.types.Minutia` 序列（或任何具备
            ``x`` / ``y`` / ``angle`` / ``quality`` / ``type_code`` 属性的对象）。
            ``None`` 视为空序列。
        width: 图像宽度（px），存储为 16 位无符号整数。
        height: 图像高度（px），存储为 16 位无符号整数。
        ppi: 分辨率（ppi），默认 500。

    Returns:
        ``bytes``，长度恒为 ``24 + 6n``，其中 ``n = min(len(minutiae), 255)``。

    Note:
        超出 10 位范围的坐标会被裁剪；``angle`` 先对 ``2π`` 取模再量化；
        ``quality`` 由 ``[0, 1]`` 线性映射到 ``0~100``。函数不修改入参。
    """
    records: Iterable[Minutia] = [] if minutiae is None else minutiae
    items = list(records)
    count = min(len(items), _MAX_COUNT)

    total = ISO_HEADER_SIZE + ISO_MINUTIA_SIZE * count
    header = b"".join(
        (
            ISO_FORMAT_IDENTIFIER,
            ISO_VERSION,
            struct.pack(">I", total),
            struct.pack(">H", 0),  # 采集设备 ID（保留）
            struct.pack(">H", _clip_int(width, 0, 65535)),
            struct.pack(">H", _clip_int(height, 0, 65535)),
            struct.pack(">H", _clip_int(ppi, 0, 65535)),
            struct.pack(">H", _clip_int(ppi, 0, 65535)),
            struct.pack(">B", count),
            struct.pack(">B", 0),  # 预留
        )
    )

    body = bytearray()
    for minutia in items[:count]:
        x = _clip_int(getattr(minutia, "x", 0), 0, _MAX_COORD)
        y = _clip_int(getattr(minutia, "y", 0), 0, _MAX_COORD)
        type_code = _clip_int(getattr(minutia, "type_code", 0), 0, 3)
        angle = _angle_byte(getattr(minutia, "angle", 0.0))
        quality = _quality_byte(getattr(minutia, "quality", 1.0))
        body.append(((type_code & 0x03) << 6) | (((x >> 8) & 0x03) << 4) | (((y >> 8) & 0x03) << 2))
        body.append(x & 0xFF)
        body.append(y & 0xFF)
        body.append(angle)
        body.append(quality)
        body.append(0)

    return header + bytes(body)
