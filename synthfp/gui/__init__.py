"""图形界面层 —— 设计文档 3。

本包是应用的一级入口。GUI **不直接调用**任何生成逻辑，
只通过 :mod:`synthfp.app`（应用服务层）与 :mod:`synthfp.gui.qt_bridge`（信号桥）
工作，因此核心层可以在无图形环境下独立运行与测试。

**Qt 绑定**：所有 Qt 访问都经过 :mod:`synthfp.gui.qt`。它优先用 PySide6
（设计文档 3.5 的选型），找不到时自动回退 PyQt6；本包内**其它模块一律不得**
直接 ``import PySide6`` / ``import PyQt6``（有单元测试守着这条约束）。

页面结构（``QStackedWidget`` + 左侧导航）::

    1. 数据集配置   规模 / 类别比例 / 输出目录
    2. 后端配置     生成内核参数与分辨率
    3. TTA 配置     算子勾选 + 参数滑块 + 效果对比预览
    4. 质量与标注   质量阈值 / 细节点 / 标注产物
    5. 生成监控     进度 / 任务表 / 日志 / 暂停恢复停止
    6. 结果预览     缩略图网格 + 大图 + 细节点与方向场叠加
    7. 统计报告     类别分布 / 质量分布 / 细节点分布 / 划分
    8. 导出         打包 / README / 校验和 / 数据集结构
"""

from __future__ import annotations

__all__ = ["run_gui"]


def run_gui(config=None):  # pragma: no cover - 延迟导入，避免无 Qt 环境时导入失败
    """启动图形界面（供 CLI 与 ``synthfp-gui`` 脚本调用）。"""
    from .app import run_gui as _run

    return _run(config)
