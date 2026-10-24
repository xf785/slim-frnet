"""TTA 链选择器 —— 设计文档 4.6「随机种子由样本 ID 派生，保证可复现」。

三种模式：

* ``single``  每个样本只取 1 个算子；
* ``chain``   每个样本组合 ``chain_length`` 个算子；
* ``mixed``   按 ``chain_probability`` 决定走 single 还是 chain。

算子链在**样本种子**上采样，因此同一个样本号在任何机器上都会得到
完全相同的算子序列 —— 这是可复现性的必要条件。

约定：``tta_index == 0`` 的样本保留**未变换的原始捺印**，作为数据集内的
干净对照；所有 TTA 变换从 ``tta_index >= 1`` 开始。
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..config.schema import AppConfig
from ..core.seeds import derive

__all__ = ["select_tta_chain", "tta_chain_keys", "plan_tta_chains"]


def _chain_length(config: AppConfig, rng: np.random.Generator) -> int:
    """按模式决定本次链长。"""
    mode = config.tta.mode
    lo, hi = int(config.tta.chain_length[0]), int(config.tta.chain_length[1])
    lo = max(lo, 1)
    hi = max(hi, lo)
    if mode == "single":
        return 1
    if mode == "chain":
        return int(rng.integers(lo, hi + 1))
    # mixed
    if rng.random() < float(config.tta.chain_probability):
        return int(rng.integers(max(lo, 2) if hi >= 2 else 1, hi + 1))
    return 1


def _distortion_of(name: str) -> float:
    from ..plugins.tta import TTARegistry

    try:
        return float(TTARegistry.get(name).distortion)
    except Exception:  # pragma: no cover
        return 0.2


def select_tta_chain(
    config: AppConfig, sample_seed: int, sample_id: str = "", tta_index: int = 0
) -> List[object]:
    """为某个样本挑选并实例化 TTA 算子链。

    Args:
        config: 配置。
        sample_seed: 该样本的稳定种子（``SeedTree.tta_seed`` 产出）。
        sample_id: 样本标识（参与种子派生，便于日志溯源）。
        tta_index: TTA 序号；为 0 时返回空链（保留原始捺印）。

    Returns:
        算子实例列表（可能为空）。
    """
    if not config.tta.enabled or tta_index == 0:
        return []

    from ..plugins.tta import TTARegistry  # 延迟导入：插件层按需加载

    available = [n for n in config.tta.operators if TTARegistry.is_registered(n)]
    if not available:
        return []

    rng = np.random.default_rng(derive(sample_seed, "tta-chain", sample_id, tta_index))
    length = min(_chain_length(config, rng), len(available))
    picked = [str(x) for x in rng.choice(available, size=length, replace=False)]
    # 先轻后重，避免重算子把轻算子想保留的细节抹掉
    picked.sort(key=_distortion_of)

    operators: List[object] = []
    budget = float(config.tta.max_distortion)
    used = 0.0
    for name in picked:
        d = _distortion_of(name)
        if operators and used + d > budget:
            break
        try:
            operators.append(TTARegistry.create(name, config.tta.params_for(name)))
            used += d
        except Exception:  # pragma: no cover
            continue
    return operators


def tta_chain_keys(
    config: AppConfig, sample_seed: int, sample_id: str = "", tta_index: int = 0
) -> List[str]:
    """只返回算子名列表（不实例化），供计划预览与 GUI 展示。"""
    return [op.name for op in select_tta_chain(config, sample_seed, sample_id, tta_index)]


def plan_tta_chains(config: AppConfig, sample_seeds) -> List[List[str]]:
    """批量预览：``[(seed, sample_id, tta_index)] -> [[算子名, ...], ...]``。"""
    return [
        tta_chain_keys(config, seed, sid, idx) for seed, sid, idx in sample_seeds
    ]
