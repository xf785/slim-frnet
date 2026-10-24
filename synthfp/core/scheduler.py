"""Scheduler —— 设计文档 4.3。

* 任务粒度：**一个主指纹（含其全部捺印与 TTA 样本）**。
  设计文档原定「一个 TTA 样本」，但主指纹合成是整条管道里最贵的一步；
  以主指纹为粒度可以让它天然只算一次（同进程内复用），把总开销降低 3~8 倍。
* 并行后端：``ThreadPoolExecutor``。重活是 OpenCV 的滤波 / 重映射，
  这些调用会释放 GIL，因此线程池能拿到真实并行度；而进程池在 Windows 上
  需要排队 + pickle 传输 numpy 数组，并且模块级 Gabor 缓存无法跨进程复用，
  实测反而更慢。``workers=1`` 时完全串行，便于调试与精确计时。
* 断点续传：Entry 内所有样本都已完成则整条跳过；部分完成则只补缺失样本。
* 失败隔离：单样本异常重试 ``retries`` 次后写入 ``failed.jsonl``，不影响其它样本。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config.schema import AppConfig
from .manifest import Manifest
from .planner import Plan, PlanEntry
from .storage import DatasetLayout
from .types import SampleResult, SampleSpec

logger = logging.getLogger(__name__)

__all__ = ["SchedulerHooks", "Scheduler", "EntryOutcome"]


@dataclass
class SchedulerHooks:
    """调度事件回调（纯 Python，GUI 层负责转成 Qt 信号）。

    Attributes:
        on_start: ``(total_samples, total_entries)``
        on_entry_start: ``(entry_index, master_id, n_samples)``
        on_sample: ``(SampleResult,)``
        on_entry_done: ``(entry_index, ok, failed, skipped)``
        on_log: ``(level, message)``
    """

    on_start: Optional[Callable[[int, int], None]] = None
    on_entry_start: Optional[Callable[[int, str, int], None]] = None
    on_sample: Optional[Callable[[SampleResult], None]] = None
    on_entry_done: Optional[Callable[[int, int, int, int], None]] = None
    on_log: Optional[Callable[[str, str], None]] = None

    def log(self, level: str, message: str) -> None:
        if self.on_log is not None:
            try:
                self.on_log(level, message)
            except Exception:  # pragma: no cover - 回调不得影响生成
                logger.debug("日志回调异常", exc_info=True)


@dataclass
class EntryOutcome:
    """单个主指纹条目的处理结果。"""

    entry_index: int
    master_id: str
    results: List[SampleResult] = field(default_factory=list)
    skipped: int = 0
    failed: int = 0
    duration_s: float = 0.0

    @property
    def ok(self) -> int:
        return sum(1 for r in self.results if r.status != "failed")


class Scheduler:
    """把计划条目分发到工作线程，并汇总结果。"""

    def __init__(
        self,
        config: AppConfig,
        layout: DatasetLayout,
        manifest: Manifest,
        hooks: Optional[SchedulerHooks] = None,
        stop_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
    ) -> None:
        """
        Args:
            config: 配置。
            layout: 输出布局。
            manifest: 清单（负责写入与断点续传查询）。
            hooks: 事件回调。
            stop_event: 置位后停止派发新任务并等待在跑的条目收尾。
            pause_event: 置位后暂停派发新任务。
        """
        self.config = config
        self.layout = layout
        self.manifest = manifest
        self.hooks = hooks or SchedulerHooks()
        self.stop_event = stop_event or threading.Event()
        self.pause_event = pause_event or threading.Event()
        self.config_fingerprint: str = ""
        #: 每个工作线程各自持有一个 PipelineRunner（runner 内有不可共享的资源）
        self._local = threading.local()

    # -- 工作线程局部资源 -------------------------------------------------
    def _runner_for_thread(self):
        runner = getattr(self._local, "runner", None)
        if runner is None:
            from ..pipeline.runner import PipelineRunner

            runner = PipelineRunner(self.config, self.layout)
            self._local.runner = runner
        return runner

    # -- 主入口 -----------------------------------------------------------
    def run(self, plan: Plan) -> Dict[str, Any]:
        """执行整个计划，返回运行统计字典。"""
        entries = list(plan.entries)
        total_samples = sum(len(e.samples) for e in entries)
        workers = max(1, int(self.config.runtime.workers))

        self.hooks.on_start and self.hooks.on_start(total_samples, len(entries))
        self.hooks.log(
            "INFO",
            f"开始生成：{len(entries)} 个主指纹 / {total_samples} 个样本，"
            f"{workers} 个并行工作单元",
        )

        started = time.perf_counter()
        agg = {"ok": 0, "rejected": 0, "failed": 0, "skipped": 0, "processed": 0}

        if workers == 1:
            for idx, entry in enumerate(entries):
                if self._should_stop():
                    self.hooks.log("WARNING", "收到停止请求，剩余任务已取消")
                    break
                self._wait_if_paused()
                outcome = self._run_entry(idx, entry)
                self._absorb(outcome, agg)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="synthfp") as pool:
                futures: Dict[Future, int] = {}
                for idx, entry in enumerate(entries):
                    if self._should_stop():
                        break
                    self._wait_if_paused()
                    futures[pool.submit(self._run_entry, idx, entry)] = idx
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        outcome = fut.result()
                    except Exception as exc:  # pragma: no cover - 兜底
                        logger.exception("条目 %s 异常", idx)
                        self.hooks.log("ERROR", f"主指纹条目 #{idx} 异常: {exc}")
                        continue
                    self._absorb(outcome, agg)

        elapsed = time.perf_counter() - started
        agg["elapsed_s"] = round(elapsed, 3)
        self.hooks.log(
            "INFO",
            f"生成结束：成功 {agg['ok']}，超质量阈值丢弃 {agg['rejected']}，"
            f"失败 {agg['failed']}，跳过 {agg['skipped']}，耗时 {elapsed:.1f}s",
        )
        return agg

    # -- 内部 -------------------------------------------------------------
    def _absorb(self, outcome: EntryOutcome, agg: Dict[str, int]) -> None:
        agg["skipped"] += outcome.skipped
        for res in outcome.results:
            agg["processed"] += 1
            if res.status == "ok":
                agg["ok"] += 1
            elif res.status == "rejected":
                agg["rejected"] += 1
            else:
                agg["failed"] += 1

    def _should_stop(self) -> bool:
        return self.stop_event.is_set()

    def _wait_if_paused(self) -> None:
        """暂停：在任务边界处以 100ms 粒度等待。"""
        while self.pause_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.1)

    def _run_entry(self, index: int, entry: PlanEntry) -> EntryOutcome:
        """处理一个主指纹条目（含全部捺印与 TTA 样本）。"""
        outcome = EntryOutcome(entry_index=index, master_id=entry.master_id)
        pending = [s for s in entry.samples if not self._is_done(s)]
        outcome.skipped = len(entry.samples) - len(pending)
        if not pending:
            return outcome

        retries = int(self.config.runtime.retries)
        t0 = time.perf_counter()
        self.hooks.on_entry_start and self.hooks.on_entry_start(index, entry.master_id, len(pending))
        if outcome.skipped:
            self.hooks.log(
                "INFO",
                f"[{entry.master_id}] 断点续传：跳过 {outcome.skipped} 个已完成样本，"
                f"补生成 {len(pending)} 个",
            )

        try:
            results = self._runner_for_thread().run_entry(
                entry, pending, progress=self._emit_sample
            )
        except Exception as exc:  # 主指纹整体失败：整条条目的全部样本都记为失败
            logger.exception("主指纹 %s 生成失败", entry.master_id)
            self.hooks.log("ERROR", f"[{entry.master_id}] 主指纹生成失败: {exc}")
            results = []
            for spec in pending:
                res = SampleResult(spec=spec, status="failed", error=str(exc))
                self.manifest.append_failure(spec.sample_id, str(exc), attempt=1)
                self._emit_sample(res)
                results.append(res)
            outcome.failed = len(pending)
        else:
            for res in results:
                if res.status == "failed":
                    if res.attempt >= retries + 1:
                        self.manifest.append_failure(res.spec.sample_id, res.error, res.attempt)
                    outcome.failed += 1

        outcome.results = list(results)
        outcome.duration_s = time.perf_counter() - t0
        self.hooks.on_entry_done and self.hooks.on_entry_done(
            index, sum(1 for r in outcome.results if r.status == "ok"), outcome.failed, outcome.skipped
        )
        return outcome

    def _emit_sample(self, result: SampleResult) -> None:
        if result.status != "failed":
            try:
                self.manifest.append(
                    result.to_manifest_row(self.config_fingerprint)
                )
            except Exception as exc:  # pragma: no cover
                logger.exception("写入 manifest 失败")
                self.hooks.log("ERROR", f"写入 manifest 失败: {exc}")
        self.hooks.on_sample and self.hooks.on_sample(result)

    def _is_done(self, spec: SampleSpec) -> bool:
        """断点续传判定：manifest 已记录 **且** 图像仍在磁盘上。"""
        if not self.config.runtime.resume:
            return False
        if not self.manifest.is_done(spec.sample_id):
            return False
        image = self.layout.resolve(
            str(Path("images") / spec.relative_path)
        )
        if image.exists():
            return True
        # 兼容 image_format=tiff 的配置
        return image.with_suffix(".tiff").exists() or image.with_suffix(".tif").exists()
