"""核心域层：计划 / 调度 / 管道编排 / 清单 / 存储。

本层**不依赖** PySide6，可在无图形环境（服务器、CI）中独立运行。
"""

from __future__ import annotations

from .manifest import Manifest, read_jsonl
from .planner import Plan, PlanEntry, Planner, assign_splits, expand_class_assignment
from .scheduler import EntryOutcome, Scheduler, SchedulerHooks
from .seeds import SeedTree, derive, stable_seed
from .storage import DatasetLayout, environment_snapshot, sha256_file
from .types import (
    ImpressionArtifact,
    ImpressionSpec,
    MasterArtifact,
    MasterSpec,
    Minutia,
    QualityReport,
    RunStats,
    SampleResult,
    SampleSpec,
)

__all__ = [
    "DatasetLayout",
    "EntryOutcome",
    "ImpressionArtifact",
    "ImpressionSpec",
    "Manifest",
    "MasterArtifact",
    "MasterSpec",
    "Minutia",
    "Plan",
    "PlanEntry",
    "Planner",
    "QualityReport",
    "RunStats",
    "SampleResult",
    "SampleSpec",
    "Scheduler",
    "SchedulerHooks",
    "SeedTree",
    "assign_splits",
    "derive",
    "environment_snapshot",
    "expand_class_assignment",
    "read_jsonl",
    "sha256_file",
    "stable_seed",
]
