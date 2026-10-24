"""应用服务层 —— GUI 与 CLI 都只通过这一层访问核心域（设计文档 3.3）。

GUI **不直接调用**任何生成逻辑，只调用本层的服务对象，
服务对象通过**回调**上报进度（Qt 适配器再把回调转成信号）。
这样核心层完全不依赖 PySide6，可以在无 GUI 环境（服务器 / 测试）里跑通。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..config.loader import dump_yaml, save_config
from ..config.schema import AppConfig
from ..core.manifest import Manifest
from ..core.planner import Plan, Planner
from ..core.scheduler import Scheduler, SchedulerHooks
from ..core.seeds import stable_seed
from ..core.storage import DatasetLayout, atomic_write_text, environment_snapshot
from ..core.types import RunStats, SampleResult

logger = logging.getLogger(__name__)

__all__ = [
    "config_fingerprint",
    "ProgressEvent",
    "GenerationCallbacks",
    "GenerationService",
    "PreviewService",
]


# ---------------------------------------------------------------------------
# 配置指纹（可复现四要素之一）
# ---------------------------------------------------------------------------
def config_fingerprint(config: AppConfig) -> str:
    """配置内容的 SHA-256（前 16 位），用于判断两次运行是否同源。"""
    payload = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _needed_modalities(config: AppConfig) -> list:
    """按配置算出**真正会被写入**的产物目录。

    只建需要的目录：早先无条件建 ``frequency`` / ``centerline`` / ``quality``，
    而这些只有在对应开关打开时才会写文件，结果每次生成都留下几个空目录，
    让人误以为生成不完整。
    """
    modalities = ["images", "mask"]
    if config.annotation.export_orientation:
        modalities.append("orientation")
    if config.annotation.export_centerline:
        modalities.append("centerline")
    if config.annotation.export_quality_map:
        modalities.append("quality")
    if config.minutiae.enabled:
        modalities.append("minutiae")
    return modalities


# ---------------------------------------------------------------------------
# 事件与回调
# ---------------------------------------------------------------------------
@dataclass
class ProgressEvent:
    """一次进度上报。"""

    stats: RunStats
    result: Optional[SampleResult] = None

    @property
    def percent(self) -> float:
        return 100.0 * self.stats.processed / self.stats.total if self.stats.total else 0.0


@dataclass
class GenerationCallbacks:
    """GUI / CLI 注入的回调集合（任一可为 ``None``）。"""

    on_phase: Optional[Callable[[str], None]] = None
    on_plan: Optional[Callable[[Dict[str, Any]], None]] = None
    on_progress: Optional[Callable[[ProgressEvent], None]] = None
    on_sample: Optional[Callable[[SampleResult], None]] = None
    on_log: Optional[Callable[[str, str], None]] = None
    on_finished: Optional[Callable[[RunStats], None]] = None


# ---------------------------------------------------------------------------
# 生成服务
# ---------------------------------------------------------------------------
class GenerationService:
    """批量生成服务：计划 → 调度 → 写入 manifest → 导出。

    典型用法（GUI 在后台线程里跑 ``start``）::

        svc = GenerationService(config, callbacks)
        svc.start()          # 阻塞直到完成 / 停止
        svc.pause(); svc.resume(); svc.stop()

    Attributes:
        stats: 实时统计（GUI 状态栏直接读）。
    """

    def __init__(
        self, config: AppConfig, callbacks: Optional[GenerationCallbacks] = None,
        export_after_run: bool = True,
    ) -> None:
        self.config = config
        self.callbacks = callbacks or GenerationCallbacks()
        self.export_after_run = export_after_run
        self.layout = DatasetLayout(config.dataset_dir)
        self.plan: Optional[Plan] = None
        self.manifest: Optional[Manifest] = None
        self.stats = RunStats(status="idle", output_dir=str(self.layout.root))
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.fingerprint = config_fingerprint(config)
        self._started_at = 0.0

    # -- 生命周期 ---------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return self._pause.is_set()

    def build_plan(self) -> Plan:
        """构造（并缓存）生成计划。"""
        self.plan = Planner(self.config, self.layout).build()
        return self.plan

    def start(self, blocking: bool = True) -> None:
        """开始生成。

        Args:
            blocking: ``True`` 时在当前线程执行（调用方自己放进后台线程）；
                ``False`` 时服务自己起一个后台线程。
        """
        if self.is_running:
            raise RuntimeError("生成任务已在运行")
        self._stop.clear()
        self._pause.clear()
        if blocking:
            self._run()
        else:
            self._thread = threading.Thread(target=self._run, name="synthfp-gen", daemon=True)
            self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """等待后台线程结束。"""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def pause(self) -> None:
        """暂停派发新任务（在跑的条目会收尾）。"""
        self._pause.set()
        self.stats.status = "paused"
        self._log("WARNING", "已暂停：当前条目完成后不再派发新任务")

    def resume(self) -> None:
        """恢复。"""
        self._pause.clear()
        self.stats.status = "running"
        self._log("INFO", "已恢复")

    def stop(self) -> None:
        """请求停止（在跑的条目会收尾，已完成的样本完整保留）。"""
        self._stop.set()
        self._pause.clear()
        self.stats.status = "stopped"
        self._log("WARNING", "已请求停止：等待当前条目收尾…")

    # -- 主流程 -----------------------------------------------------------
    def _run(self) -> None:
        cfg = self.config
        try:
            self._phase("准备")
            self.layout.create(_needed_modalities(cfg))
            self._write_run_metadata()

            plan = self.plan or self.build_plan()
            plan.write(self.layout.plan)
            self._emit_plan(plan)

            self.manifest = Manifest(self.layout, resume=cfg.runtime.resume)
            self.manifest.open()

            scheduler = Scheduler(
                cfg,
                self.layout,
                self.manifest,
                hooks=self._hooks(),
                stop_event=self._stop,
                pause_event=self._pause,
            )
            scheduler.config_fingerprint = self.fingerprint

            self.stats = RunStats(
                total=len(plan),
                status="running",
                output_dir=str(self.layout.root),
                started_at=time.time(),
            )
            self._started_at = self.stats.started_at
            self._emit_progress()

            self._phase("生成")
            agg = scheduler.run(plan)

            self.manifest.close()
            self._finish(agg)
        except Exception as exc:  # 生成整体失败
            logger.exception("生成过程失败")
            self.stats.status = "failed"
            self.stats.message = f"{type(exc).__name__}: {exc}"
            self._log("ERROR", f"生成失败: {exc}")
            if self.manifest is not None:
                self.manifest.close()
            self._finished()

    def _finish(self, agg: Dict[str, Any]) -> None:
        stats = self.stats
        stats.ok = int(agg.get("ok", 0))
        stats.rejected = int(agg.get("rejected", 0))
        stats.failed = int(agg.get("failed", 0))
        stats.skipped = int(agg.get("skipped", 0))
        stats.elapsed_s = float(agg.get("elapsed_s", 0.0))
        stats.finished_at = time.time()
        stats.done = stats.ok + stats.rejected + stats.failed
        stats.status = "stopped" if self._stop.is_set() else "finished"

        if self.export_after_run and stats.status == "finished" and stats.ok > 0:
            self._phase("导出")
            try:
                from .exporter import export_dataset
                from .stats import compute_dataset_stats

                export_dataset(self.config, self.layout, fingerprint=self.fingerprint)
                compute_dataset_stats(self.config, self.layout, write=True)
            except Exception as exc:  # 导出失败不应吞掉生成成果
                logger.exception("导出阶段失败")
                self._log("ERROR", f"导出阶段失败（数据本身已完整）: {exc}")
        self._finished()

    # -- 元数据 -----------------------------------------------------------
    def _write_run_metadata(self) -> None:
        cfg = self.config
        try:
            save_config(cfg, self.layout.config_resolved)
            env = environment_snapshot(self.fingerprint)
            env["run"] = {
                "dataset": cfg.dataset.name,
                "seed": cfg.runtime.seed,
                "estimate": cfg.estimate(),
                "summary": cfg.summary_lines(),
            }
            atomic_write_text(
                self.layout.environment, json.dumps(env, ensure_ascii=False, indent=2)
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("写入运行元数据失败: %s", exc)

    def dry_run(self) -> Dict[str, Any]:
        """只构造计划并返回摘要，不生成任何文件（GUI「预览计划」用）。"""
        plan = self.build_plan()
        summary = plan.summary()
        summary["fingerprints"] = {
            "masters": summary["masters"],
            "impressions": summary["impressions"],
            "samples": summary["samples"],
        }
        summary["config_fingerprint"] = self.fingerprint
        return summary

    # -- 回调桥接 ---------------------------------------------------------
    def _phase(self, name: str) -> None:
        if self.callbacks.on_phase:
            self.callbacks.on_phase(name)

    def _emit_plan(self, plan: Plan) -> None:
        if self.callbacks.on_plan:
            self.callbacks.on_plan(plan.summary())

    def _emit_progress(self) -> None:
        if self.callbacks.on_progress:
            self.callbacks.on_progress(ProgressEvent(stats=self.stats))

    def _finished(self) -> None:
        if self.callbacks.on_finished:
            self.callbacks.on_finished(self.stats)

    def _log(self, level: str, message: str) -> None:
        if self.callbacks.on_log:
            try:
                self.callbacks.on_log(level, message)
            except Exception:  # pragma: no cover
                logger.debug("日志回调异常", exc_info=True)
        logger.log(getattr(logging, level, logging.INFO), message)

    def _hooks(self) -> SchedulerHooks:
        """把调度事件转成服务回调。"""

        def on_sample(result: SampleResult) -> None:
            with self._lock:
                if result.status == "ok":
                    self.stats.ok += 1
                elif result.status == "rejected":
                    self.stats.rejected += 1
                else:
                    self.stats.failed += 1
                self.stats.elapsed_s = time.time() - self._started_at if self._started_at else 0.0
            if self.callbacks.on_sample:
                try:
                    self.callbacks.on_sample(result)
                except Exception:  # pragma: no cover
                    logger.debug("样本回调异常", exc_info=True)
            self._emit_progress()

        def on_log(level: str, message: str) -> None:
            self._log(level, message)

        return SchedulerHooks(on_sample=on_sample, on_log=on_log)


# ---------------------------------------------------------------------------
# 预览服务（不写文件，GUI 实时预览用）
# ---------------------------------------------------------------------------
@dataclass
class PreviewResult:
    """单张预览结果。"""

    master: Any = None
    image: Any = None
    mask: Any = None
    orientation: Any = None
    effects: List[Dict[str, Any]] = field(default_factory=list)
    tta_operators: List[str] = field(default_factory=list)
    tta_params: Dict[str, Any] = field(default_factory=dict)
    quality: Any = None
    minutiae: List[Any] = field(default_factory=list)
    tta_grid: List[Any] = field(default_factory=list)
    duration_s: float = 0.0
    error: str = ""


class PreviewService:
    """生成单张样本供 GUI 预览 / TTA 效果对比，完全不落盘。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def render(
        self,
        identity_id: str = "000001",
        finger: str = "left_thumb",
        fp_class: Optional[str] = None,
        seed: Optional[int] = None,
        tta_index: int = 0,
        impression_index: int = 1,
        with_annotation: bool = True,
    ) -> PreviewResult:
        """渲染一张样本。

        Args:
            fp_class: 为 ``None`` 时按配置的类别分布采样。
            seed: 为 ``None`` 时由身份 + 手指 + 种子派生。
            tta_index: ``0`` 表示原始捺印；``>=1`` 表示施加 TTA 链。
        """
        import numpy as np

        from ..core.seeds import SeedTree
        from ..pipeline import chain as chain_mod
        from ..pipeline import stages

        cfg = self.config
        t0 = time.perf_counter()
        result = PreviewResult()
        try:
            tree = SeedTree(cfg.runtime.seed, cfg.dataset.name)
            names, weights = cfg.classes.weights()
            if fp_class is None:
                fp_class = str(np.random.default_rng(cfg.runtime.seed).choice(names, p=weights))
            if seed is None:
                seed = tree.finger_seed(identity_id, finger)

            from ..core.types import MasterSpec
            from ..plugins.backends import get_backend

            spec = MasterSpec(
                master_id=f"{identity_id}_{finger}",
                identity_id=identity_id,
                finger=finger,
                finger_index=0,
                fp_class=fp_class,
                image_size=cfg.resolved_image_size(),
                seed=int(seed),
            )
            backend = get_backend(cfg.backend.name, cfg.backend.model_dump(mode="python"))
            master = backend.generate_master(spec)
            result.master = master

            # 捺印
            imp_seed = tree.impression_seed(identity_id, finger, impression_index)
            rng = np.random.default_rng(imp_seed)
            composer = stages.build_stress_composer(cfg)
            image, effects = stages.apply_impression(
                master, master.image, rng, cfg, composer=composer,
                sample_key=f"{spec.master_id}:imp_{impression_index:02d}",
            )
            result.effects = effects

            # TTA
            sample_seed = tree.tta_seed(identity_id, finger, impression_index, tta_index)
            operators = chain_mod.select_tta_chain(
                cfg, sample_seed, f"{spec.master_id}_imp{impression_index:02d}_tta{tta_index:02d}",
                tta_index,
            )
            image, op_names, op_params, _ = stages.apply_tta(image, operators, cfg, tta_index)
            image = stages.finalize_image(image, cfg)
            result.tta_operators = op_names
            result.tta_params = op_params
            result.image = image

            mask = np.where(image < 245, np.uint8(255), np.uint8(0))
            if operators:
                rng2 = np.random.default_rng(sample_seed)
                warped = mask.astype(np.float32) / 255.0
                for op in operators:
                    try:
                        warped = op.apply(warped, rng2)
                    except Exception:
                        continue
                mask = (np.clip(warped, 0, 1) * 255 > 127).astype(np.uint8) * 255
            result.mask = mask

            if with_annotation:
                result.orientation = stages.estimate_orientation(image)
                result.quality = stages.evaluate_quality(image, mask, result.orientation, cfg)
                result.minutiae = stages.extract_minutiae(image, mask, result.orientation, cfg)
                if result.quality is not None:
                    result.quality.minutiae_count = len(result.minutiae)
        except Exception as exc:
            logger.exception("预览渲染失败")
            result.error = f"{type(exc).__name__}: {exc}"
        result.duration_s = time.perf_counter() - t0
        return result

    def render_tta_grid(
        self, operators: List[str], identity_id: str = "000001", finger: str = "left_thumb",
        fp_class: Optional[str] = None, base_seed: Optional[int] = None,
    ) -> List[tuple]:
        """对同一张捺印逐个施加指定算子，返回 ``[(算子名, 图像), ...]``。

        用于 GUI 的 TTA 页面「实时预览效果」。
        """
        import numpy as np

        from ..plugins.tta import TTARegistry

        cfg = self.config
        base = self.render(identity_id, finger, fp_class, base_seed, tta_index=0,
                           with_annotation=False)
        if base.error or base.image is None:
            return []
        out: List[tuple] = [("原始", base.image)]
        src = base.image.astype(np.float32) / 255.0
        for name in operators:
            if not TTARegistry.is_registered(name):
                continue
            try:
                op = TTARegistry.create(name, cfg.tta.params_for(name))
                rng = np.random.default_rng(
                    stable_seed(name, base_seed, identity_id, finger) % (2**31)
                )
                img = np.clip(op.apply(src, rng), 0, 1)
                out.append((getattr(op, "label", name), (img * 255).astype(np.uint8)))
            except Exception as exc:  # pragma: no cover
                logger.warning("TTA 预览 %s 失败: %s", name, exc)
        return out

    def yaml_text(self) -> str:
        return dump_yaml(self.config)
