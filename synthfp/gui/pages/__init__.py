"""GUI 页面集合（设计文档 3.2 的 8 个页面）。"""

from __future__ import annotations

from .backend_page import BackendPage
from .base import BasePage
from .dataset_page import DatasetPage
from .export_page import ExportPage
from .monitor_page import MonitorPage
from .preview_page import PreviewPage
from .quality_page import QualityPage
from .stats_page import StatsPage
from .tta_page import TTAPage

#: 导航顺序（与设计文档 3.2 的页面编号一致）
PAGE_CLASSES = (
    DatasetPage,   # 1. 数据集配置
    BackendPage,   # 2. 后端配置
    TTAPage,       # 3. TTA 配置
    QualityPage,   # 4. 质量与标注
    MonitorPage,   # 5. 生成监控
    PreviewPage,   # 6. 结果预览
    StatsPage,     # 7. 统计报告
    ExportPage,    # 8. 导出
)

__all__ = [
    "BasePage",
    "BackendPage",
    "DatasetPage",
    "ExportPage",
    "MonitorPage",
    "PAGE_CLASSES",
    "PreviewPage",
    "QualityPage",
    "StatsPage",
    "TTAPage",
]
