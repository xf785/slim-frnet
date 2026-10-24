"""自研 NFIQ2 风格综合质量评分器（无需外部依赖）。

评分公式（见 ``CONTRACTS.md`` 第 2.2 节）::

    score = 100 · (0.35·clarity + 0.20·contrast + 0.25·orientation_certainty
                   + 0.10·usable_area + 0.10·ridge_period_score)

    level:  score ≥ 80 → 1，≥ 65 → 2，≥ 50 → 3，≥ 35 → 4，否则 5
    passed: score ≥ min_score 且 usable_area ≥ min_usable_area

权重与分箱参照 NIST NFIQ（Tabassi et al., NISTIR 7151, 2004）的分项加权思路，
并结合合成指纹场景做了简化：脊线清晰度与方向一致性占主导，面积与周期作为
辅助约束。
"""

from __future__ import annotations

import numpy as np

from ...core.types import QualityReport
from . import metrics as _metrics

__all__ = [
    "QualityEvaluator",
    "score_to_level",
    "ridge_period_score",
    "WEIGHTS",
    "RIDGE_PERIOD_MIN",
    "RIDGE_PERIOD_MAX",
    "RIDGE_PERIOD_DECAY",
]

WEIGHTS = {
    "clarity": 0.35,
    "contrast": 0.20,
    "orientation_certainty": 0.25,
    "usable_area": 0.10,
    "ridge_period": 0.10,
}
"""五个分项的权重（和为 1.0）。"""

RIDGE_PERIOD_MIN = 6.0
"""脊线周期满分区间下界（px）。"""

RIDGE_PERIOD_MAX = 16.0
"""脊线周期满分区间上界（px）。"""

RIDGE_PERIOD_DECAY = 6.0
"""偏离满分区间后线性衰减到 0 所需的像素数。"""

_LEVEL_BINS = ((80.0, 1), (65.0, 2), (50.0, 3), (35.0, 4))
"""``(分数下界, 等级)``，从优到劣。"""


def score_to_level(score: float) -> int:
    """把 0~100 的综合分映射到 5 级质量等级。

    Args:
        score: 综合分，可为任意实数（自动夹到 ``[0, 100]``）。

    Returns:
        质量等级 ``int``：``1``（最好）~ ``5``（最差）。
    """
    value = float(score)
    if not np.isfinite(value):
        return 5
    for lower, level in _LEVEL_BINS:
        if value >= lower:
            return level
    return 5


def ridge_period_score(period: float) -> float:
    """把脊线周期映射到 ``[0, 1]`` 的分项得分。

    周期落在 ``[RIDGE_PERIOD_MIN, RIDGE_PERIOD_MAX]``（6~16 px）记 1.0，
    区间外每偏离 1 px 线性扣减 ``1 / RIDGE_PERIOD_DECAY``，衰减到 0 为止。

    Args:
        period: 脊线周期（px），``<= 0`` 或非有限值记为 0 分。

    Returns:
        ``float``，取值 ``[0, 1]``。
    """
    p = float(period)
    if not np.isfinite(p) or p <= 0.0:
        return 0.0
    if RIDGE_PERIOD_MIN <= p <= RIDGE_PERIOD_MAX:
        return 1.0
    deviation = RIDGE_PERIOD_MIN - p if p < RIDGE_PERIOD_MIN else p - RIDGE_PERIOD_MAX
    return float(min(max(1.0 - deviation / RIDGE_PERIOD_DECAY, 0.0), 1.0))


class QualityEvaluator:
    """自研 NFIQ2 风格综合质量评分器（无需外部依赖）。

    管道按契约调用::

        evaluator = QualityEvaluator(min_score=50.0, min_usable_area=0.4,
                                     reject_on_fail=True)
        report = evaluator.evaluate(image, mask, orientation)
        if report.passed:
            ...

    Attributes:
        min_score: 通过阈值（``score >= min_score``）。
        min_usable_area: 通过阈值（``usable_area >= min_usable_area``）。
        reject_on_fail: 是否由 pipeline 依据 ``passed`` 丢弃样本（本类只记录标志）。
    """

    def __init__(
        self,
        min_score: float = 0.0,
        reject_on_fail: bool = False,
        min_usable_area: float = 0.0,
        count_minutiae: bool = True,
    ) -> None:
        """初始化评分器。

        Args:
            min_score: 通过所需的最低综合分（0~100），默认 0。
            reject_on_fail: ``True`` 时由 pipeline 依据 ``passed`` 丢弃样本，默认 ``False``。
            min_usable_area: 通过所需的最低有效面积占比（0~1），默认 0。
            count_minutiae: 是否在评分内部调用细节点提取器填充
                ``minutiae_count``。pipeline 自己会提一遍细节点并回填该字段，
                因此那里传 ``False`` 可省掉一次冗余提取（500×500 约 40 ms/样本）。
        """
        self.min_score = float(min_score) if np.isfinite(float(min_score)) else 0.0
        self.min_usable_area = (
            float(min_usable_area) if np.isfinite(float(min_usable_area)) else 0.0
        )
        self.reject_on_fail = bool(reject_on_fail)
        self.count_minutiae = bool(count_minutiae)

    @property
    def name(self) -> str:
        """评分器标识名。"""
        return "synthfp-nfiq2-lite"

    # -- 内部 -------------------------------------------------------------
    @staticmethod
    def _fill_minutiae_count(
        image: np.ndarray,
        mask: np.ndarray | None,
        orientation: np.ndarray | None,
    ) -> int:
        """尽力填充细节点数量（细节点模块不可用时返回 0）。"""
        try:
            from ..minutiae.extractor import MinutiaeExtractor

            return int(len(MinutiaeExtractor().extract(image, mask, orientation)))
        except Exception:  # pragma: no cover - 细节点模块缺失/异常时安全降级
            return 0

    # -- 公开 API ---------------------------------------------------------
    def evaluate(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ) -> QualityReport:
        """评估一幅指纹图像的综合质量。

        Args:
            image: 输入图像，``uint8 (H, W)``（背景为白）或 ``float32 (H, W)``
                取值 ``[0, 1]``。
            mask: 有效区域掩模 ``uint8 (H, W)`` 取值 ``{0, 255}``；``None``
                表示整幅图有效，全零掩模会得到 0 分。
            orientation: 预计算的方向场 ``float32 (H, W)``（弧度）；``None`` 时
                内部用 :func:`~synthfp.plugins.quality.metrics.estimate_orientation`
                估计。

        Returns:
            :class:`~synthfp.core.types.QualityReport`，所有字段均为有限值：
            ``score ∈ [0, 100]``、``level ∈ [1, 5]``、各分项 ``∈ [0, 1]``、
            ``ridge_period >= 0``、``minutiae_count >= 0``。

        Note:
            退化输入（空掩模 / 纯色图 / 全零图 / 任意边 < 3 px）不会抛异常，
            各分项与总分均为 0.0。``minutiae_count`` 通过内部调用细节点提取器
            填充（若细节点模块不可用则保持 0）。
        """
        gray = _metrics.to_float01(image)
        height, width = gray.shape

        estimated, certainty = _metrics.estimate_orientation(gray)
        if (
            orientation is not None
            and np.asarray(orientation).shape == gray.shape
            and min(gray.shape) >= _metrics.MIN_SIDE
        ):
            field = np.mod(
                np.nan_to_num(
                    np.asarray(orientation, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                np.pi,
            ).astype(np.float32, copy=False)
        else:
            field = estimated

        period = _metrics.estimate_ridge_period(gray, field, mask)
        clarity = _metrics.ridge_clarity(gray, field, mask, period=period or 10.0)
        contrast = _metrics.local_contrast(gray, mask)
        cert = _metrics.orientation_certainty(field, certainty, mask)
        area = _metrics.usable_area(gray, mask)
        period_score = ridge_period_score(period)

        score = 100.0 * (
            WEIGHTS["clarity"] * clarity
            + WEIGHTS["contrast"] * contrast
            + WEIGHTS["orientation_certainty"] * cert
            + WEIGHTS["usable_area"] * area
            + WEIGHTS["ridge_period"] * period_score
        )
        score = float(min(max(score if np.isfinite(score) else 0.0, 0.0), 100.0))
        level = score_to_level(score)
        passed = bool(score >= self.min_score and area >= self.min_usable_area)
        count = 0
        if self.count_minutiae and min(gray.shape) >= 8:
            count = self._fill_minutiae_count(gray, mask, field)

        return QualityReport(
            score=score,
            level=level,
            clarity=float(clarity),
            contrast=float(contrast),
            orientation_certainty=float(cert),
            usable_area=float(area),
            ridge_period=float(period),
            minutiae_count=int(max(0, count)),
            passed=passed,
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"QualityEvaluator(min_score={self.min_score}, "
            f"min_usable_area={self.min_usable_area}, "
            f"reject_on_fail={self.reject_on_fail})"
        )
