"""``synthfp.plugins.minutiae`` —— 细节点提取与 ISO 模板导出插件。

公开 API::

    from synthfp.plugins.minutiae import (
        MinutiaeExtractor,     # 骨架化 + Crossing Number 提取器
        zhang_suen,            # 向量化 Zhang-Suen 细化
        skeleton_minutiae,     # 骨架上的细节点候选（CN==1 / CN==3）
        to_iso_template,       # ISO/IEC 19794-2:2005 简约格式字节流
    )
"""

from __future__ import annotations

from .extractor import MinutiaeExtractor
from .iso_template import to_iso_template
from .skeleton import (
    as_binary,
    crossing_numbers,
    geodesic_distance,
    neighbour_codes,
    prune_short_branches,
    skeleton_minutiae,
    zhang_suen,
)

__all__ = [
    "MinutiaeExtractor",
    "zhang_suen",
    "skeleton_minutiae",
    "prune_short_branches",
    "geodesic_distance",
    "to_iso_template",
    "as_binary",
    "neighbour_codes",
    "crossing_numbers",
]
