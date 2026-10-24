"""退化引擎 — 模式A核心"""
from .base import DegradationOperator, DegradationParams
from .registry import DegradationRegistry
from .composer import DegradationComposer
from .recorder import DegradationRecorder

# 导入所有退化算子以触发注册
from .operators import (
    DownsampleOperator,
    GaussianBlurOperator,
    MotionBlurOperator,
    PoissonNoiseOperator,
    SpeckleNoiseOperator,
    ElasticDeformationOperator,
    RandomErasureOperator,
    InkSmudgeOperator,
    JPEGCompressionOperator,
    ContrastReductionOperator,
    DirtStainOperator,
    BlemishSpotOperator,
    GhostingOperator,
    PartialMissingOperator,
    DefocusBlurOperator,
    EdgeFeatheringOperator,
    RidgeEdgeRoughnessOperator,
    RidgeBreaksOperator,
    SensorNoiseOperator,
)

__all__ = [
    "DegradationOperator",
    "DegradationParams",
    "DegradationRegistry",
    "DegradationComposer",
    "DegradationRecorder",
]
