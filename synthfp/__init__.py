"""synthfp —— 轻量级 SFinGe 合成指纹数据集合成软件。

分层结构（自上而下）::

    gui/            PySide6 图形界面（一级入口）
    app/            应用服务层：任务编排、进度信号、导出
    core/           核心域层：Planner / Scheduler / Manifest / Storage
    pipeline/       样本管道：impression -> tta -> quality -> minutiae -> storage
    plugins/        插件层：backends / impressions / tta / quality / minutiae
    config/         配置系统：Pydantic schema + YAML 读写 + 预设
    legacy/         原项目已验证的生成内核（shape/orientation/frequency/ridge/退化算子）
"""

from __future__ import annotations

__version__ = "2.0.0"
__all__ = ["__version__"]
