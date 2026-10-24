"""Qt 服务桥 —— 把应用服务层的**回调**转成 Qt **信号**。

设计文档 3.3 要求「GUI 不直接调用生成逻辑，只调用 Application 服务层」。
服务层是纯 Python（无 Qt 依赖），本模块是两者之间唯一的适配点：

    GenerationService(callbacks)  ──信号──▶  GUI 控件

线程模型：``GenerationService.start(blocking=True)`` 在 ``QThread`` 里跑，
回调发生在工作线程；Qt 信号跨线程投递是**队列式**的，
因此槽函数一定在主线程执行，可以直接操作控件。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..app.services import (
    GenerationCallbacks,
    GenerationService,
    PreviewResult,
)
from ..config.schema import AppConfig
from ..core.types import RunStats
from .qt import QObject, QThread, Signal

__all__ = ["GenerationWorker", "GenerationController", "PreviewWorker"]


# ---------------------------------------------------------------------------
# 生成工作线程
# ---------------------------------------------------------------------------
class GenerationWorker(QThread):
    """在后台线程里跑一遍完整的生成流程。"""

    progress = Signal(object)  # ProgressEvent
    sample_done = Signal(object)  # SampleResult
    log = Signal(str, str)  # level, message
    phase = Signal(str)  # 阶段名
    planned = Signal(dict)  # 计划摘要
    finished_run = Signal(object)  # RunStats

    def __init__(self, service: GenerationService, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._service = service

    @property
    def service(self) -> GenerationService:
        return self._service

    def run(self) -> None:  # noqa: D102 - QThread 约定
        self._service.start(blocking=True)


# ---------------------------------------------------------------------------
# 控制器：对 GUI 暴露的高层 API
# ---------------------------------------------------------------------------
class GenerationController(QObject):
    """GUI 侧统一入口：启动 / 暂停 / 恢复 / 停止 / 进度 / 日志。

    所有方法都必须在主线程调用。
    """

    progress = Signal(object)  # ProgressEvent
    sample_done = Signal(object)  # SampleResult
    log = Signal(str, str)  # level, message
    phase = Signal(str)
    planned = Signal(dict)
    finished = Signal(object)  # RunStats
    running_changed = Signal(bool)
    paused_changed = Signal(bool)

    def __init__(self, config: AppConfig, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._worker: Optional[GenerationWorker] = None
        self._service: Optional[GenerationService] = None
        self._paused = False

    # -- 状态 -------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def stats(self) -> RunStats:
        if self._service is None:
            return RunStats()
        return self._service.stats

    @property
    def service(self) -> Optional[GenerationService]:
        return self._service

    def set_config(self, config: AppConfig) -> None:
        if self.is_running:
            raise RuntimeError("生成中不能修改配置")
        self._config = config

    # -- 生命周期 ---------------------------------------------------------
    def start(self, export_after_run: bool = True) -> bool:
        """启动生成；运行中或配置非法时返回 ``False``。"""
        if self.is_running:
            self.log.emit("WARNING", "已有任务在运行")
            return False

        callbacks = GenerationCallbacks(
            on_phase=self.phase.emit,
            on_plan=self.planned.emit,
            on_progress=self.progress.emit,
            on_sample=self.sample_done.emit,
            on_log=self.log.emit,
            on_finished=self._on_finished,
        )
        self._service = GenerationService(self._config, callbacks, export_after_run=export_after_run)
        self._worker = GenerationWorker(self._service)
        self._worker.finished.connect(self._on_thread_finished)
        self._paused = False
        self._worker.start()
        self.running_changed.emit(True)
        return True

    def pause(self) -> None:
        if self._service is not None and self.is_running and not self._paused:
            self._service.pause()
            self._paused = True
            self.paused_changed.emit(True)

    def resume(self) -> None:
        if self._service is not None and self._paused:
            self._service.resume()
            self._paused = False
            self.paused_changed.emit(False)

    def toggle_pause(self) -> None:
        self.resume() if self._paused else self.pause()

    def stop(self) -> None:
        if self._service is not None and self.is_running:
            self._service.stop()
            self._paused = False
            self.paused_changed.emit(False)

    def wait(self, timeout_ms: int = 5000) -> bool:
        if self._worker is None:
            return True
        return self._worker.wait(timeout_ms)

    # -- 断开 -------------------------------------------------------------
    def _on_finished(self, stats: RunStats) -> None:
        self.finished.emit(stats)

    def _on_thread_finished(self) -> None:
        self.running_changed.emit(False)
        self.planned.emit({})


# ---------------------------------------------------------------------------
# 预览工作线程
# ---------------------------------------------------------------------------
class PreviewWorker(QThread):
    """后台渲染单张预览（避免 GUI 卡顿）。"""

    rendered = Signal(object)  # PreviewResult
    failed = Signal(str)

    def __init__(
        self,
        service: Any,
        kwargs: Dict[str, Any],
        grid_operators: Optional[List[str]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._kwargs = kwargs
        self._grid = grid_operators

    def run(self) -> None:  # noqa: D102
        try:
            result: PreviewResult = self._service.render(**self._kwargs)
            if self._grid:
                result.tta_grid = self._service.render_tta_grid(
                    self._grid,
                    identity_id=self._kwargs.get("identity_id", "000001"),
                    finger=self._kwargs.get("finger", "left_thumb"),
                    fp_class=self._kwargs.get("fp_class"),
                    base_seed=self._kwargs.get("seed"),
                )
            self.rendered.emit(result)
        except Exception as exc:  # pragma: no cover - 兜底
            self.failed.emit(f"{type(exc).__name__}: {exc}")
