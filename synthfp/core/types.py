"""核心领域数据结构 —— 单一数据源（设计文档 0.2「单一数据源」原则）。

这些 dataclass 是 Planner / Pipeline / Manifest / Storage / GUI 之间唯一的
数据交换格式。任何模块都不得私自定义另一套字段名。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "MasterSpec",
    "ImpressionSpec",
    "SampleSpec",
    "MasterArtifact",
    "ImpressionArtifact",
    "Minutia",
    "QualityReport",
    "SampleResult",
    "RunStats",
]


# ---------------------------------------------------------------------------
# 计划（Planner 产出）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MasterSpec:
    """主指纹（一个手指一遍完整合成）的生成规格。"""

    master_id: str
    identity_id: str
    finger: str
    finger_index: int
    fp_class: str
    image_size: Tuple[int, int]
    seed: int
    split: str = "train"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImpressionSpec:
    """一次捺印（同一主指纹的一次按压采集）。"""

    impression_id: str
    master_id: str
    index: int
    seed: int
    pressure: float
    rotation: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SampleSpec:
    """一个最终样本 = 一次捺印 + 一次 TTA 变换。

    这是调度器的任务粒度（设计文档 4.3）。
    """

    sample_id: str
    master_id: str
    impression_id: str
    identity_id: str
    finger: str
    fp_class: str
    impression_index: int
    tta_index: int
    seed: int
    relative_path: str
    split: str = "train"
    is_base: bool = False  # TTA 序号 0 表示未变换的原始捺印

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 运行期产物
# ---------------------------------------------------------------------------
@dataclass
class MasterArtifact:
    """后端输出的主指纹（内存态，不落盘）。"""

    master_id: str
    fp_class: str
    image: Any  # np.ndarray uint8 (H, W)
    mask: Any  # np.ndarray uint8 (H, W) {0,255}
    orientation: Any  # np.ndarray float32 (H, W) 弧度 ∈ [0, π)
    frequency: Any  # np.ndarray float32 (H, W) 频率
    singular_points: List[Dict[str, Any]] = field(default_factory=list)
    shape_params: Dict[str, int] = field(default_factory=dict)
    seed: int = 0
    backend: str = "sfinge"
    backend_version: str = ""


@dataclass
class ImpressionArtifact:
    """捺印产物：图像 + 掩模 + 施加的效果记录。"""

    impression_id: str
    image: Any  # np.ndarray uint8
    mask: Any  # np.ndarray uint8
    effects: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 标注
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Minutia:
    """细节点（x, y, 方向, 类型, 质量）。"""

    x: int
    y: int
    angle: float = 0.0  # 弧度 ∈ [0, 2π)
    kind: str = "ending"  # ending | bifurcation
    quality: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "angle": round(float(self.angle), 4),
            "type": self.kind,
            "quality": round(float(self.quality), 3),
        }

    @property
    def type_code(self) -> int:
        """ISO/IEC 19794-2 细节点类型编码。"""
        return 0 if self.kind == "ending" else 1


@dataclass
class QualityReport:
    """质量评估结果（自研 NFIQ2 风格评分，无需外部依赖）。"""

    score: float = 0.0  # 0-100 综合分
    level: int = 0  # 1(最好) - 5(最差)
    clarity: float = 0.0  # 脊线清晰度
    contrast: float = 0.0  # 对比度
    orientation_certainty: float = 0.0  # 方向一致性
    usable_area: float = 0.0  # 有效面积占比
    ridge_period: float = 0.0  # 平均脊线周期 (px)
    minutiae_count: int = 0
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "level": int(self.level),
            "clarity": round(self.clarity, 4),
            "contrast": round(self.contrast, 4),
            "orientation_certainty": round(self.orientation_certainty, 4),
            "usable_area": round(self.usable_area, 4),
            "ridge_period": round(self.ridge_period, 3),
            "minutiae_count": int(self.minutiae_count),
            "passed": bool(self.passed),
        }


# ---------------------------------------------------------------------------
# 样本结果（Pipeline -> Manifest）
# ---------------------------------------------------------------------------
@dataclass
class SampleResult:
    """一个样本走完管道后的完整结果。"""

    spec: SampleSpec
    status: str = "ok"  # ok | rejected | failed
    image_path: str = ""
    width: int = 0
    height: int = 0
    mask_path: str = ""
    orientation_path: str = ""
    centerline_path: str = ""
    quality_map_path: str = ""
    minutiae_path: str = ""
    template_path: str = ""
    checksum: str = ""
    bytes: int = 0
    quality: Optional[QualityReport] = None
    minutiae_count: int = 0
    minutiae_endings: int = 0
    minutiae_bifurcations: int = 0
    tta_operators: List[str] = field(default_factory=list)
    tta_params: Dict[str, Any] = field(default_factory=dict)
    tta_distortion: float = 0.0
    impression_effects: List[Dict[str, Any]] = field(default_factory=list)
    singular_points: List[Dict[str, Any]] = field(default_factory=list)
    shape_params: Dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    backend_version: str = ""
    ppi: int = 500
    error: str = ""
    duration_s: float = 0.0
    attempt: int = 1

    # -- 序列化 -----------------------------------------------------------
    def to_manifest_row(self, config_fingerprint: str = "") -> Dict[str, Any]:
        """转为 ``dataset_manifest.jsonl`` 的一行（设计文档 11.1）。"""
        spec = self.spec
        row: Dict[str, Any] = {
            "sample_id": spec.sample_id,
            "identity_id": spec.identity_id,
            "finger": spec.finger,
            "master_id": spec.master_id,
            "impression_id": spec.impression_id,
            "impression_index": spec.impression_index,
            "tta_id": f"tta_{spec.tta_index:02d}",
            "tta_index": spec.tta_index,
            "is_base": spec.is_base,
            "class_label": spec.fp_class,
            "split": spec.split,
            "status": self.status,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "ppi": self.ppi,
            "nfiq2": round(self.quality.score, 2) if self.quality else None,
            "quality_level": self.quality.level if self.quality else None,
            "quality_detail": self.quality.to_dict() if self.quality else None,
            "minutiae_count": self.minutiae_count,
            "minutiae_endings": self.minutiae_endings,
            "minutiae_bifurcations": self.minutiae_bifurcations,
            "minutiae_path": self.minutiae_path,
            "template_path": self.template_path,
            "tta_operator": self.tta_operators[0] if self.tta_operators else "none",
            "tta_operators": self.tta_operators,
            "tta_params": self.tta_params,
            "tta_distortion": round(self.tta_distortion, 4),
            "impression_effects": self.impression_effects,
            "singular_points": self.singular_points,
            "shape_params": self.shape_params,
            "seed": spec.seed,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "sha256": self.checksum,
            "bytes": self.bytes,
            "duration_s": round(self.duration_s, 4),
            "attempt": self.attempt,
        }
        if self.mask_path:
            row["mask_path"] = self.mask_path
        if self.orientation_path:
            row["orientation_path"] = self.orientation_path
        if self.centerline_path:
            row["centerline_path"] = self.centerline_path
        if self.quality_map_path:
            row["quality_map_path"] = self.quality_map_path
        if config_fingerprint:
            row["config_fingerprint"] = config_fingerprint
        if self.error:
            row["error"] = self.error
        return row

    def to_json(self) -> str:
        return json.dumps(self.to_manifest_row(), ensure_ascii=False)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _finger_index_of(spec: SampleSpec) -> int:
    """从 ``master_id`` 反解手指序号（manifest 字段用）。"""
    parts = spec.master_id.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0
# ---------------------------------------------------------------------------
# 运行统计
# ---------------------------------------------------------------------------
@dataclass
class RunStats:
    """一次运行的总统计（GUI 状态栏 / 统计报告用）。"""

    total: int = 0
    done: int = 0
    ok: int = 0
    rejected: int = 0
    failed: int = 0
    skipped: int = 0
    elapsed_s: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    status: str = "idle"  # idle | running | paused | stopped | finished | failed
    output_dir: str = ""
    message: str = ""

    @property
    def processed(self) -> int:
        return self.ok + self.rejected + self.failed

    @property
    def rate(self) -> float:
        """样本/秒。"""
        return self.processed / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def eta_s(self) -> float:
        r = self.rate
        return (self.total - self.processed) / r if r > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rate"] = round(self.rate, 3)
        d["eta_s"] = round(self.eta_s, 1)
        d["processed"] = self.processed
        return d
