"""synthfp.plugins —— 插件层。

子包一览（各自独立，互不依赖）::

    tta/          TTA 引擎：21 个几何/光度/噪声/压缩/遮挡/传感器算子
    quality/      质量评估：方向场、脊线清晰度、NFIQ2 风格综合评分
    minutiae/     细节点提取：Zhang-Suen 细化 + Crossing Number
    impressions/  捺印效果：压力形变 + 传感器成像
    backends/     合成后端（SFinGe 风格生成内核）

注意：本文件只作为包标记，**不导入任何子模块**，避免某个子包不可用时
影响其他子包的导入。
"""

from __future__ import annotations

__all__: list[str] = []
