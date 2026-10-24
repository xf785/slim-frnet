"""应用服务层：GUI 与 CLI 的唯一入口（设计文档 3.3）。

* :class:`GenerationService` —— 批量生成、暂停/恢复/停止、断点续传
* :class:`PreviewService`    —— 单张预览、TTA 效果对比（不落盘）
* :func:`export_dataset`     —— 导出 README / 统计 / 划分 / 校验和
* :func:`compute_dataset_stats` —— 数据集统计报告
"""

from __future__ import annotations

from .exporter import export_dataset, write_checksums, write_readme
from .services import (
    GenerationCallbacks,
    GenerationService,
    PreviewResult,
    PreviewService,
    ProgressEvent,
    config_fingerprint,
)
from .stats import compute_dataset_stats, format_stats_text, summarize_numbers

__all__ = [
    "GenerationCallbacks",
    "GenerationService",
    "PreviewResult",
    "PreviewService",
    "ProgressEvent",
    "compute_dataset_stats",
    "config_fingerprint",
    "export_dataset",
    "format_stats_text",
    "summarize_numbers",
    "write_checksums",
    "write_readme",
]
