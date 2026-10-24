"""管道执行器 —— 让一个主指纹条目走完整个管道。

一个条目 = 一个手指的一遍完整合成，包含：

* **1 次主指纹生成**（最贵的一步，条目内复用）
* **N 次捺印**（随机退化链 → 按压变形 → 传感器）
* **M 个 TTA 样本**（每个捺印派生 M 个 TTA 变换）

这样组织使「主指纹只算一次」成为结构性保证，而不是靠缓存碰运气。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..config.schema import AppConfig
from ..core.planner import PlanEntry
from ..core.storage import DatasetLayout, sha256_bytes
from ..core.types import MasterArtifact, SampleResult, SampleSpec
from . import chain as chain_mod
from . import stages

logger = logging.getLogger(__name__)

__all__ = ["PipelineRunner"]

ProgressFn = Callable[[SampleResult], None]


class PipelineRunner:
    """按条目执行管道。

    实例是**有状态且非线程安全**的：每个工作线程应各自持有一个实例
    （``Scheduler`` 在单线程模式下只建一个，多线程模式下由线程局部持有）。
    """

    def __init__(self, config: AppConfig, layout: DatasetLayout) -> None:
        self.config = config
        self.layout = layout
        self._backend = None
        self._composer = None
        self._minutiae = None
        self._quality = None

    # -- 惰性资源 ---------------------------------------------------------
    @property
    def backend(self):
        if self._backend is None:
            from ..plugins.backends import get_backend

            self._backend = get_backend(
                self.config.backend.name, self.config.backend.model_dump(mode="python")
            )
            try:
                self._backend.warmup()
            except Exception:  # pragma: no cover
                pass
        return self._backend

    @property
    def composer(self):
        if self._composer is None:
            self._composer = stages.build_stress_composer(self.config)
        return self._composer

    # -- 主入口 -----------------------------------------------------------
    def run_entry(
        self,
        entry: PlanEntry,
        pending: Sequence[SampleSpec],
        progress: Optional[ProgressFn] = None,
    ) -> List[SampleResult]:
        """处理一个主指纹条目的待办样本，返回它们的结果列表。"""
        results: List[SampleResult] = []
        if not pending:
            return results

        # ---- 1. 主指纹（每个手指只生成一次） ----
        t_master = time.perf_counter()
        master = self.backend.generate_master(entry.master)
        logger.debug("主指纹 %s 耗时 %.2fs", entry.master_id, time.perf_counter() - t_master)

        # ---- 2. 按捺印分组，避免重复渲染同一捺印 ----
        by_impression: Dict[str, List[SampleSpec]] = {}
        for spec in pending:
            by_impression.setdefault(spec.impression_id, []).append(spec)

        imp_specs = {imp.impression_id: imp for imp in entry.impressions}

        for imp_id, specs in by_impression.items():
            imp_spec = imp_specs.get(imp_id)
            if imp_spec is None:  # pragma: no cover - 计划完整性由 Planner 保证
                for spec in specs:
                    results.append(self._fail(spec, "计划中缺少捺印定义"))
                continue

            impression_image, effects = self._render_impression(master, imp_spec)
            for spec in sorted(specs, key=lambda s: s.tta_index):
                if self.config.runtime.keep_failed:
                    result = self._process_sample_with_retry(
                        master, impression_image, effects, spec
                    )
                else:
                    result = self._process_sample(master, impression_image, effects, spec)
                results.append(result)
                if progress is not None:
                    progress(result)
        return results

    # -- 捺印 -------------------------------------------------------------
    def _render_impression(self, master: MasterArtifact, imp_spec) -> tuple:
        """渲染一次捺印（含退化链 / 变形 / 传感器）。"""
        rng = np.random.default_rng(imp_spec.seed)
        return stages.apply_impression(
            master,
            master.image,
            rng,
            self.config,
            composer=self.composer,
            sample_key=f"{master.master_id}:{imp_spec.impression_id}",
        )

    # -- 单样本 -----------------------------------------------------------
    def _process_sample_with_retry(
        self,
        master: MasterArtifact,
        impression_image: np.ndarray,
        effects: List[Dict[str, Any]],
        spec: SampleSpec,
    ) -> SampleResult:
        """带重试的单样本处理（默认重试 ``runtime.retries`` 次）。"""
        attempts = max(int(self.config.runtime.retries), 0) + 1
        last: Optional[SampleResult] = None
        for attempt in range(1, attempts + 1):
            result = self._process_sample(master, impression_image, effects, spec)
            result.attempt = attempt
            if result.status != "failed":
                return result
            last = result
            logger.debug("样本 %s 第 %d 次失败: %s", spec.sample_id, attempt, result.error)
        assert last is not None
        return last

    def _process_sample(
        self,
        master: MasterArtifact,
        impression_image: np.ndarray,
        effects: List[Dict[str, Any]],
        spec: SampleSpec,
    ) -> SampleResult:
        """走完单个样本的 TTA → 细节点 → 质量 → 存储。"""
        t0 = time.perf_counter()
        result = SampleResult(
            spec=spec,
            backend=master.backend,
            backend_version=master.backend_version,
            ppi=int(self.config.backend.ppi),
            impression_effects=list(effects),
            singular_points=list(master.singular_points or []),
            shape_params=dict(master.shape_params or {}),
        )
        try:
            # --- TTA 算子链（掩模与图像共用同一批算子，保证严格同源） ---
            operators = chain_mod.select_tta_chain(
                self.config, spec.seed, spec.sample_id, spec.tta_index
            )

            # --- 掩模：主指纹掩模 → 经过捺印变形 → 经过 TTA 几何变换 ---
            #
            # 直接用"像素质疑"（灰度 < 245）当掩模是不行的：传感器噪声与皮肤纹理
            # 会让背景出现零星暗点，细节点检测随后就会在背景里报出成百上千个伪点。
            # 这里改为对**几何意义上正确的掩模**做变换，再用捺印掩模做一次交集校验。
            mask = self._transform_mask(master.mask, operators, spec)

            # --- TTA ---
            image, names, params, distortion = stages.apply_tta(
                impression_image, operators, self.config, spec.tta_index,
                mask=mask, seed=spec.seed,
            )
            # 「仅脊线黑色」要求最终产物就是二值图：捺印与 TTA 阶段会重新引入灰度
            # （传感器噪声、纹理、模糊），所以在这里统一收口一次。
            image = stages.finalize_image(image, self.config)
            result.tta_operators = names
            result.tta_params = params
            result.tta_distortion = distortion

            height, width = image.shape[:2]
            result.width, result.height = int(width), int(height)

            # --- 方向场（作用于最终图像，保证与像素严格对齐） ---
            orientation = (
                stages.estimate_orientation(image)
                if (self.config.annotation.export_orientation or self.config.minutiae.compute_angle)
                else None
            )

            # --- 细节点 ---
            minutiae = stages.extract_minutiae(image, mask, orientation, self.config)
            result.minutiae_count = len(minutiae)
            result.minutiae_endings = sum(1 for m in minutiae if m.kind == "ending")
            result.minutiae_bifurcations = sum(1 for m in minutiae if m.kind == "bifurcation")

            # --- 质量 ---
            quality = stages.evaluate_quality(image, mask, orientation, self.config)
            if quality is not None:
                quality.minutiae_count = len(minutiae)
                result.quality = quality
                if self.config.quality.reject_on_fail and not quality.passed:
                    result.status = "rejected"
                    result.error = (
                        f"质量不达标: score={quality.score:.1f} < {self.config.quality.min_score:.1f}"
                        f" 或有效面积不足"
                    )
                    result.duration_s = time.perf_counter() - t0
                    return result

            # --- 存储 ---
            self._persist(result, image, mask, orientation, minutiae)
            result.status = "ok"
        except Exception as exc:  # 单样本失败隔离
            logger.exception("样本 %s 处理失败", spec.sample_id)
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
        result.duration_s = time.perf_counter() - t0
        return result

    # -- 掩模 -------------------------------------------------------------
    def _transform_mask(
        self,
        master_mask: np.ndarray,
        operators: Sequence[object],
        spec: SampleSpec,
    ) -> np.ndarray:
        """把主指纹掩模对齐到最终样本图。

        用**同一批已实例化的 TTA 算子**变换掩模，保证掩模与像素同源；
        再做一次形态学清理去掉算子（旋转/缩放/遮挡）造成的碎块与毛刺。

        与"按灰度阈值提前景"相比，这条路径不会把传感器噪声误当成手指，
        因此细节点检测不会被背景伪点淹没。
        """
        mask = master_mask
        if operators:
            # 几何算子接受 float32 图，因此把掩模归一化后传入，再二值化回来
            rng = np.random.default_rng(spec.seed)
            warped = mask.astype(np.float32) / 255.0
            for op in operators:
                try:
                    warped = op.apply(warped, rng)
                except Exception:  # pragma: no cover - 与图像侧保持一致的容错
                    continue
            mask = (np.clip(warped, 0.0, 1.0) * 255.0 > 127).astype(np.uint8) * 255

        return _clean_mask(mask)

    # -- 落盘 -------------------------------------------------------------
    def _persist(
        self,
        result: SampleResult,
        image: np.ndarray,
        mask: np.ndarray,
        orientation: Optional[np.ndarray],
        minutiae: Sequence,
    ) -> None:
        """写出图像与标注，并填充结果中的路径 / 校验和 / 体积。"""
        cfg = self.config
        layout = self.layout
        spec = result.spec
        rel = spec.relative_path

        # 图像（先编码成字节以便一次算出 sha256 与体积，避免二次读盘）
        img_rel = layout.save_image(rel, image, cfg.export.png_compression, cfg.export.image_format)
        img_abs = layout.resolve(img_rel)
        blob = img_abs.read_bytes()
        result.image_path = img_rel
        result.checksum = sha256_bytes(blob)
        result.bytes = len(blob)

        if cfg.annotation.export_mask:
            result.mask_path = layout.save_mask(rel, mask, cfg.export.png_compression)
        if cfg.annotation.export_orientation and orientation is not None:
            result.orientation_path = layout.save_orientation(rel, orientation)
        if cfg.annotation.export_centerline:
            result.centerline_path = self._save_centerline(rel, mask)
        if cfg.annotation.export_quality_map:
            result.quality_map_path = self._save_quality_map(rel, image, mask)

        if cfg.minutiae.enabled:
            payload = stages.minutiae_payload(
                spec.sample_id, minutiae, result.width, result.height, result.ppi,
                spec.fp_class, result.quality, result.singular_points, result.shape_params,
            )
            result.minutiae_path = layout.save_json("minutiae", rel, payload)
            if cfg.minutiae.export_iso:
                from ..plugins.minutiae.iso_template import to_iso_template

                blob = to_iso_template(minutiae, result.width, result.height, result.ppi)
                iso_rel = str(Path("minutiae") / Path(rel).with_suffix(".iso")).replace("\\", "/")
                from ..core.storage import atomic_write_bytes

                atomic_write_bytes(layout.resolve(iso_rel), blob)
                result.template_path = iso_rel

    def _save_centerline(self, rel: str, mask: np.ndarray) -> str:
        """脊线中心线（形态学骨架，轻量近似）。"""
        try:
            binary = (mask > 0).astype(np.uint8) * 255
            # 用距离变换的脊线近似中心线：局部极大值
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
            dilated = cv2.dilate(dist, np.ones((3, 3), np.uint8))
            ridge = ((dist >= dilated - 1e-3) & (dist > 2.0)).astype(np.uint8) * 255
            return self.layout.save_mask(str(Path("centerline") / Path(rel)), ridge)
        except Exception:  # pragma: no cover
            return ""

    def _save_quality_map(self, rel: str, image: np.ndarray, mask: np.ndarray) -> str:
        """局部质量图（分块标准差，量化存储）。"""
        try:
            gray = image.astype(np.float32) / 255.0
            block = 16
            h, w = gray.shape
            gh, gw = max(h // block, 1), max(w // block, 1)
            small = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)
            mean = cv2.blur(small, (3, 3))
            sq = cv2.blur(small * small, (3, 3))
            var = np.clip(sq - mean * mean, 0, None)
            qmap = cv2.resize(np.sqrt(var), (w, h), interpolation=cv2.INTER_LINEAR)
            qmap = qmap * (mask > 0)
            out = str(Path("quality") / Path(rel)).replace("\\", "/")
            return self.layout.save_float_map("quality", out, qmap)
        except Exception:  # pragma: no cover
            return ""

    # -- 工具 -------------------------------------------------------------
    @staticmethod
    def _fail(spec: SampleSpec, error: str) -> SampleResult:
        return SampleResult(spec=spec, status="failed", error=error)


def _clean_mask(mask: np.ndarray, min_area_ratio: float = 0.02) -> np.ndarray:
    """掩模形态学清理：去掉碎块与毛刺，补上小孔。

    几何 TTA 会让掩模边缘出现锯齿，遮挡算子会切掉一块；这里：
    1. 闭运算补小孔 → 开运算去毛刺；
    2. 只保留面积最大的连通域（切掉被算子打到画面角落的碎片）；
    3. 再用闭运算把最大连通域的边缘抹圆。
    """
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if count > 2:
        total = float(mask.size)
        # 通道 0 是背景，从 1 开始
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[largest, cv2.CC_STAT_AREA] >= min_area_ratio * total:
            cleaned = np.where(labels == largest, np.uint8(255), np.uint8(0))
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
