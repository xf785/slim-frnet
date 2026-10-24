"""样本管道（设计文档 4.4）。

本包只做编排，不含任何生成算法：

* :mod:`synthfp.pipeline.chain`  —— TTA 算子链的可复现选择
* :mod:`synthfp.pipeline.runner` —— 一个主指纹条目走完整个管道
* :mod:`synthfp.pipeline.stages` —— 各阶段的纯函数实现
"""

from __future__ import annotations

from .chain import plan_tta_chains, select_tta_chain, tta_chain_keys
from .runner import PipelineRunner

__all__ = ["PipelineRunner", "plan_tta_chains", "select_tta_chain", "tta_chain_keys"]
