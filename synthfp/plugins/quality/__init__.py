"""``synthfp.plugins.quality`` —— 质量评估插件。

公开 API::

    from synthfp.plugins.quality import (
        QualityEvaluator,        # NFIQ2-lite 综合评分器
        estimate_orientation,    # 方向场 + 一致性
        orientation_certainty,
        ridge_clarity,           # Gabor 响应强度
        local_contrast,
        usable_area,
        estimate_ridge_period,
        score_to_level,
        ridge_period_score,
    )
"""

from __future__ import annotations

from .evaluator import (
    RIDGE_PERIOD_DECAY,
    RIDGE_PERIOD_MAX,
    RIDGE_PERIOD_MIN,
    WEIGHTS,
    QualityEvaluator,
    ridge_period_score,
    score_to_level,
)
from .metrics import (
    CONTRAST_REF,
    DEFAULT_RIDGE_PERIOD,
    estimate_orientation,
    estimate_ridge_period,
    local_contrast,
    orientation_certainty,
    ridge_clarity,
    to_float01,
    usable_area,
)

__all__ = [
    "QualityEvaluator",
    "score_to_level",
    "ridge_period_score",
    "WEIGHTS",
    "RIDGE_PERIOD_MIN",
    "RIDGE_PERIOD_MAX",
    "RIDGE_PERIOD_DECAY",
    "estimate_orientation",
    "orientation_certainty",
    "ridge_clarity",
    "local_contrast",
    "usable_area",
    "estimate_ridge_period",
    "to_float01",
    "DEFAULT_RIDGE_PERIOD",
    "CONTRAST_REF",
]
