"""原项目已验证的生成内核（保留资产，接口层做最小改造）。

* ``render/``  —— 虚拟指纹合成内核
  手指轮廓 5 参数模型 → 奇异点 → 方向场（Zero-Pole）→ 频率图（Perlin 多层）
  → Gabor 迭代脊线合成。这部分算法经过原项目验证，**算法逻辑一行不改**，
  仅修正导入路径、向量化瓶颈与去除与标注阶段的重复计算。

* ``degrade/`` —— 19 个指纹退化算子（干湿、污渍、断线、边缘粗糙、传感器噪声等），
  在 v2.0 中作为「捺印阶段」的随机压力链复用。

* ``util/``    —— Unicode 安全的图像读写等通用工具。

这些模块是纯算法库，不含任何业务编排逻辑；编排逻辑在 ``synthfp.pipeline`` 与
``synthfp.app`` 中。新算子请写到 ``synthfp.plugins`` 而不是这里。
"""

from __future__ import annotations

__all__ = ["render", "degrade", "util"]
