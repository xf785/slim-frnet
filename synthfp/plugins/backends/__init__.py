"""生成后端插件 —— 设计文档 4.5。

统一接口 :class:`FingerprintBackend`，内置实现：

* ``SFinGeBackend``  原项目验证过的合成内核（虚拟指纹全合成）
* ``DummyBackend``   测试用占位图，零开销

后端**只负责生成主指纹**（一次完整合成，可得掩模 / 方向场 / 频率图 /
奇异点）。捺印（按压变形 + 传感器）与 TTA 由管道后续阶段处理，
这样主指纹可以复用，避免重复付出最昂贵的合成开销。
"""

from __future__ import annotations

from .base import BackendRegistry, FingerprintBackend, get_backend, list_backends
from .dummy import DummyBackend
from .sfinge import SFinGeBackend

__all__ = [
    "BackendRegistry",
    "DummyBackend",
    "FingerprintBackend",
    "SFinGeBackend",
    "get_backend",
    "list_backends",
]
