"""``synthfp.plugins.quality`` 的质量评估测试（契约第 2.3 节）。

覆盖：方向场一致性、脊线清晰度、对比度、有效面积、脊线周期、分箱边界、
评分确定性、退化输入（空掩模 / 纯色 / 全零 / 单像素）与
“理想条纹高分、白噪声低分”的合理性检查。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from synthfp.core.types import QualityReport
from synthfp.plugins.quality import (
    CONTRAST_REF,
    QualityEvaluator,
    estimate_orientation,
    estimate_ridge_period,
    local_contrast,
    orientation_certainty,
    ridge_clarity,
    ridge_period_score,
    score_to_level,
    to_float01,
    usable_area,
)

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
SIZE = 256


def _vertical_ridges(size: int = SIZE, period: float = 10.0, amplitude: float = 0.45):
    """垂直脊线（灰度沿 x 呈余弦变化），float32 [0,1]。"""
    xx = np.arange(size, dtype=np.float32)[None, :]
    row = 0.5 + amplitude * np.cos(2.0 * np.pi * xx / period)
    return np.repeat(row.astype(np.float32), size, axis=0)


def _horizontal_ridges(size: int = SIZE, period: float = 10.0, amplitude: float = 0.45):
    """水平脊线（灰度沿 y 呈余弦变化），float32 [0,1]。"""
    return _vertical_ridges(size, period, amplitude).T.copy()


def _uniform_noise(size: int = SIZE, seed: int = 0, low: float = 0.05, high: float = 0.95):
    """均匀白噪声，与条纹图同量程。"""
    rng = np.random.default_rng(seed)
    return (low + (high - low) * rng.random((size, size))).astype(np.float32)


def _ellipse_mask(size: int = SIZE, axes=(60, 100)) -> np.ndarray:
    """椭圆掩模 uint8 {0,255}。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), axes, 0, 0, 360, 255, -1)
    return mask


def _circular_diff(angle: float, expected: float) -> float:
    """方向场的圆距离（模 π）。"""
    return float(abs((angle - expected + np.pi / 2) % np.pi - np.pi / 2))


def _ridge_grid(size: int = 256, spacing: int = 12):
    """带断口的平行脊线（模拟真实指纹，用于细节点计数）。"""
    image = np.full((size, size), 255, np.uint8)
    for index, y in enumerate(range(30, size - 26, spacing)):
        cv2.line(image, (40, y), (size - 46, y), 0, 3)
        if index % 3 == 0:
            cv2.line(image, (110, y), (126, y), 255, 5)
    return image


# ---------------------------------------------------------------------------
# to_float01
# ---------------------------------------------------------------------------
def test_to_float01_supported_inputs():
    uint8_img = np.array([[0, 128, 255]], dtype=np.uint8)
    assert to_float01(uint8_img).dtype == np.float32
    np.testing.assert_allclose(to_float01(uint8_img)[0], [0.0, 128 / 255, 1.0], atol=1e-6)

    bool_img = np.array([[True, False]], dtype=bool)
    np.testing.assert_allclose(to_float01(bool_img)[0], [1.0, 0.0])

    float_255 = np.array([[0.0, 51.0, 255.0]], dtype=np.float32)
    np.testing.assert_allclose(to_float01(float_255)[0], [0.0, 0.2, 1.0], atol=1e-6)

    color = np.zeros((2, 2, 3), dtype=np.uint8)
    assert to_float01(color).shape == (2, 2)

    broken = np.array([[np.nan, np.inf, 0.5]], dtype=np.float32)
    result = to_float01(broken)
    assert np.all(np.isfinite(result))
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_to_float01_does_not_modify_input():
    original = np.array([[10, 200], [255, 0]], dtype=np.uint8)
    backup = original.copy()
    to_float01(original)
    np.testing.assert_array_equal(original, backup)


# ---------------------------------------------------------------------------
# 方向场
# ---------------------------------------------------------------------------
def test_estimate_orientation_shape_dtype_and_range():
    orientation, certainty = estimate_orientation(_vertical_ridges())
    assert orientation.shape == (SIZE, SIZE)
    assert certainty.shape == (SIZE, SIZE)
    assert orientation.dtype == np.float32 and certainty.dtype == np.float32
    assert np.all(np.isfinite(orientation)) and np.all(np.isfinite(certainty))
    assert orientation.min() >= 0.0 and orientation.max() < np.pi
    assert certainty.min() >= 0.0 and certainty.max() <= 1.0


def test_estimate_orientation_axis_convention():
    vertical, _ = estimate_orientation(_vertical_ridges())
    horizontal, _ = estimate_orientation(_horizontal_ridges())
    center = np.s_[64:192, 64:192]
    assert _circular_diff(float(np.median(vertical[center])), np.pi / 2) < 0.05
    assert _circular_diff(float(np.median(horizontal[center])), 0.0) < 0.05


def test_estimate_orientation_does_not_modify_input():
    image = _vertical_ridges()
    backup = image.copy()
    estimate_orientation(image)
    np.testing.assert_array_equal(image, backup)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((64, 64), dtype=np.float32),
        np.full((64, 64), 0.5, dtype=np.float32),
        np.full((1, 1), 0.3, dtype=np.float32),
        np.zeros((2, 7), dtype=np.uint8),
    ],
)
def test_estimate_orientation_degenerate_inputs(image):
    orientation, certainty = estimate_orientation(image)
    assert orientation.shape == image.shape
    assert certainty.shape == image.shape
    assert np.all(np.isfinite(orientation)) and np.all(np.isfinite(certainty))
    assert float(np.abs(orientation).max()) == 0.0
    assert float(np.abs(certainty).max()) == 0.0


def test_orientation_certainty_ridges_vs_noise():
    orientation, certainty = estimate_orientation(_vertical_ridges())
    ridge_score = orientation_certainty(orientation, certainty)
    noise_orientation, noise_certainty = estimate_orientation(_uniform_noise())
    noise_score = orientation_certainty(noise_orientation, noise_certainty)
    assert 0.0 <= noise_score < ridge_score <= 1.0
    assert ridge_score > 0.9
    assert noise_score < 0.5


def test_orientation_certainty_mask_and_degenerate():
    orientation, certainty = estimate_orientation(_vertical_ridges())
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 40, 255, -1)
    assert 0.0 <= orientation_certainty(orientation, certainty, mask) <= 1.0
    # 全零掩模 → 0.0
    assert orientation_certainty(orientation, certainty, np.zeros_like(mask)) == 0.0
    # 空 / 单像素 certainty → 0.0，且不抛异常
    assert orientation_certainty(orientation, np.zeros((0, 0), np.float32)) == 0.0
    assert orientation_certainty(None, np.ones((1, 1), np.float32)) == 0.0
    assert orientation_certainty(None, None) == 0.0


# ---------------------------------------------------------------------------
# 对比度 / 有效面积
# ---------------------------------------------------------------------------
def test_local_contrast_values():
    assert local_contrast(_vertical_ridges()) > 0.9
    assert 0.8 <= local_contrast(_uniform_noise()) <= 1.0
    assert local_contrast(np.zeros((64, 64), np.float32)) == 0.0
    assert local_contrast(np.full((64, 64), 0.42, np.float32)) == 0.0
    # 与参考值一致：均匀方波（0/1）的局部标准差为 0.5 → 截断到 1.0
    square = np.zeros((64, 64), np.float32)
    square[:, ::2] = 1.0
    assert local_contrast(square) == pytest.approx(1.0)
    assert CONTRAST_REF > 0.0


def test_local_contrast_mask_and_small_inputs():
    image = _vertical_ridges()
    empty = np.zeros((SIZE, SIZE), dtype=np.uint8)
    assert local_contrast(image, empty) == 0.0
    mask = _ellipse_mask(axes=(40, 40))
    assert 0.0 < local_contrast(image, mask) <= 1.0
    assert local_contrast(np.full((1, 1), 0.5, np.float32)) == 0.0
    assert local_contrast(np.full((2, 9), 0.5, np.float32)) == 0.0


def test_usable_area_values():
    noise = _uniform_noise()
    assert usable_area(noise) > 0.95
    assert usable_area(np.ones((32, 32), np.float32)) == 0.0  # 纯白背景
    assert usable_area(np.zeros((32, 32), np.float32)) == 0.0  # 全零 → 饱和
    assert usable_area(np.full((32, 32), 0.5, np.float32)) == 0.0  # 纯色无信息
    assert usable_area(np.full((1, 1), 0.5, np.float32)) == 0.0
    # 掩模外不计入
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[64:192, 64:192] = 255
    assert usable_area(noise, mask) > 0.95
    assert usable_area(noise, np.zeros_like(mask)) == 0.0


# ---------------------------------------------------------------------------
# 脊线周期
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("period", [6.0, 8.0, 10.0, 12.0, 16.0])
def test_estimate_ridge_period_matches_synthetic_ridges(period):
    image = _vertical_ridges(period=period)
    orientation, _ = estimate_orientation(image)
    estimated = estimate_ridge_period(image, orientation)
    assert estimated == pytest.approx(period, abs=0.6)


def test_estimate_ridge_period_fallbacks():
    # 可评估但无周期性 → 契约规定的默认回退值 10.0
    noise = _uniform_noise()
    orientation, _ = estimate_orientation(noise)
    assert estimate_ridge_period(noise, orientation) == pytest.approx(10.0)
    # 退化输入 → 0.0
    constant = np.full((SIZE, SIZE), 0.5, np.float32)
    assert estimate_ridge_period(constant, np.zeros((SIZE, SIZE), np.float32)) == 0.0
    assert estimate_ridge_period(np.zeros((SIZE, SIZE), np.float32), None) == 0.0
    image = _vertical_ridges()
    assert estimate_ridge_period(image, None, np.zeros((SIZE, SIZE), np.uint8)) == 0.0
    assert estimate_ridge_period(np.full((1, 1), 0.5, np.float32), None) == 0.0


def test_estimate_ridge_period_estimates_orientation_when_missing():
    image = _vertical_ridges(period=9.0)
    assert estimate_ridge_period(image, None) == pytest.approx(9.0, abs=0.6)


# ---------------------------------------------------------------------------
# 脊线清晰度
# ---------------------------------------------------------------------------
def test_ridge_clarity_ridges_vs_noise():
    ridges = _vertical_ridges()
    orientation, _ = estimate_orientation(ridges)
    ridge_score = ridge_clarity(ridges, orientation)
    noise = _uniform_noise()
    noise_orientation, _ = estimate_orientation(noise)
    noise_score = ridge_clarity(noise, noise_orientation)
    assert ridge_score > 0.8
    assert noise_score < 0.3
    assert noise_score < ridge_score


def test_ridge_clarity_period_sensitivity_and_orientation_optional():
    ridges = _vertical_ridges(period=10.0)
    orientation, _ = estimate_orientation(ridges)
    assert ridge_clarity(ridges, orientation, period=10.0) > 0.8
    assert ridge_clarity(ridges, None, period=10.0) > 0.8
    # 周期明显错配时清晰度下降
    mismatched = ridge_clarity(ridges, orientation, period=4.0)
    assert mismatched < ridge_clarity(ridges, orientation, period=10.0)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((64, 64), np.float32),
        np.full((64, 64), 0.7, np.float32),
        np.zeros((1, 1), np.float32),
        np.zeros((2, 2), np.float32),
    ],
)
def test_ridge_clarity_degenerate_inputs(image):
    orientation, _ = estimate_orientation(image)
    assert ridge_clarity(image, orientation) == 0.0


def test_ridge_clarity_mask_restricts_region():
    image = _vertical_ridges()
    orientation, _ = estimate_orientation(image)
    full = ridge_clarity(image, orientation)
    assert full > 0.8
    assert ridge_clarity(image, orientation, np.zeros((SIZE, SIZE), np.uint8)) == 0.0
    # 在图上叠加一块噪声，若掩模只取干净区域，得分仍高
    mixed = image.copy()
    mixed[:128, :] = _uniform_noise()[:128, :]
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[140:250, 10:250] = 255
    assert ridge_clarity(mixed, orientation, mask) > 0.7


# ---------------------------------------------------------------------------
# 分箱与单项得分
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("score", "level"),
    [
        (100.0, 1),
        (80.0, 1),
        (79.99, 2),
        (65.0, 2),
        (64.99, 3),
        (50.0, 3),
        (49.99, 4),
        (35.0, 4),
        (34.99, 5),
        (0.0, 5),
        (-5.0, 5),
        (float("nan"), 5),
    ],
)
def test_score_to_level_bins(score, level):
    assert score_to_level(score) == level


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        (0.0, 0.0),
        (-3.0, 0.0),
        (2.0, 1.0 / 3.0),
        (3.0, 0.5),
        (6.0, 1.0),
        (10.0, 1.0),
        (16.0, 1.0),
        (19.0, 0.5),
        (22.0, 0.0),
        (40.0, 0.0),
        (float("nan"), 0.0),
    ],
)
def test_ridge_period_score(period, expected):
    assert ridge_period_score(period) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# 综合评分器
# ---------------------------------------------------------------------------
def test_evaluator_name_and_attributes():
    evaluator = QualityEvaluator()
    assert evaluator.name == "synthfp-nfiq2-lite"
    assert evaluator.min_score == 0.0
    assert evaluator.min_usable_area == 0.0
    assert evaluator.reject_on_fail is False
    tuned = QualityEvaluator(min_score=50.0, reject_on_fail=True, min_usable_area=0.3)
    assert tuned.min_score == 50.0 and tuned.min_usable_area == 0.3 and tuned.reject_on_fail


def test_evaluator_report_types_and_ranges():
    evaluator = QualityEvaluator()
    report = evaluator.evaluate(_vertical_ridges())
    assert isinstance(report, QualityReport)
    assert 0.0 <= report.score <= 100.0
    assert report.level in (1, 2, 3, 4, 5)
    for value in (
        report.clarity,
        report.contrast,
        report.orientation_certainty,
        report.usable_area,
    ):
        assert 0.0 <= value <= 1.0
    assert report.ridge_period >= 0.0
    assert report.minutiae_count >= 0
    assert report.level == score_to_level(report.score)
    assert isinstance(report.to_dict(), dict)


@pytest.mark.parametrize("period", [10.0, 12.0])
def test_evaluator_ideal_ridges_score_high(period):
    """契约合理性检查：理想合成脊线图综合分必须 ≥ 55。"""
    evaluator = QualityEvaluator()
    report = evaluator.evaluate(_vertical_ridges(period=period))
    assert report.score >= 55.0
    assert report.orientation_certainty > 0.9
    assert report.usable_area > 0.9
    assert report.level <= 2


def test_evaluator_noise_scores_notably_lower():
    evaluator = QualityEvaluator()
    ridge_report = evaluator.evaluate(_vertical_ridges())
    noise_report = evaluator.evaluate(_uniform_noise())
    assert ridge_report.score - noise_report.score >= 20.0
    assert noise_report.clarity < ridge_report.clarity
    assert noise_report.orientation_certainty < ridge_report.orientation_certainty


def test_evaluator_is_deterministic():
    evaluator = QualityEvaluator()
    image = _uniform_noise(seed=3)
    first = evaluator.evaluate(image)
    second = evaluator.evaluate(image)
    assert first.to_dict() == second.to_dict()
    assert first.score == second.score
    assert first.minutiae_count == second.minutiae_count


def test_evaluator_accepts_precomputed_orientation_and_uint8():
    evaluator = QualityEvaluator()
    float_image = _vertical_ridges()
    orientation, _ = estimate_orientation(float_image)
    with_field = evaluator.evaluate(float_image, None, orientation)
    without_field = evaluator.evaluate(float_image)
    assert with_field.score == pytest.approx(without_field.score, abs=1e-6)
    uint8_image = np.rint(float_image * 255.0).astype(np.uint8)
    assert evaluator.evaluate(uint8_image).score == pytest.approx(without_field.score, abs=1.0)


def test_evaluator_passed_thresholds():
    evaluator = QualityEvaluator()
    ridges = _vertical_ridges()
    assert evaluator.evaluate(ridges).passed is True
    strict = QualityEvaluator(min_score=99.9)
    assert strict.evaluate(ridges).passed is True  # 理想条纹接近满分
    assert strict.evaluate(_uniform_noise()).passed is False
    area_strict = QualityEvaluator(min_usable_area=0.5)
    blank = np.ones((64, 64), np.float32)
    assert area_strict.evaluate(blank).usable_area == 0.0
    assert area_strict.evaluate(blank).passed is False


def test_evaluator_micro_texture_can_fail_area_threshold():
    evaluator = QualityEvaluator(min_score=40.0, min_usable_area=0.6, reject_on_fail=True)
    assert evaluator.reject_on_fail is True
    report = evaluator.evaluate(_vertical_ridges(), _ellipse_mask(axes=(60, 100)))
    assert report.passed == (report.score >= 40.0 and report.usable_area >= 0.6)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((64, 64), np.float32),
        np.full((64, 64), 0.5, np.float32),
        np.full((1, 1), 128, np.uint8),
        np.full((2, 3), 0.9, np.float32),
        np.full((16, 16), np.nan, np.float32),
    ],
)
def test_evaluator_degenerate_inputs_are_safe(image):
    evaluator = QualityEvaluator()
    report = evaluator.evaluate(image)
    assert report.score == 0.0
    assert report.level == 5
    assert report.minutiae_count == 0
    assert np.isfinite(report.score) and np.isfinite(report.ridge_period)
    for value in (
        report.clarity,
        report.contrast,
        report.orientation_certainty,
        report.usable_area,
    ):
        assert value == 0.0


def test_evaluator_empty_and_mismatched_mask():
    evaluator = QualityEvaluator()
    image = _vertical_ridges()
    empty = evaluator.evaluate(image, np.zeros((SIZE, SIZE), np.uint8))
    assert empty.score == 0.0
    assert empty.passed is True  # 默认阈值下 0 分也算“不拒绝”
    strict = QualityEvaluator(min_score=1.0).evaluate(image, np.zeros((SIZE, SIZE), np.uint8))
    assert strict.passed is False
    # 形状不匹配的掩模不应抛异常
    mismatched = evaluator.evaluate(image, np.ones((64, 64), np.uint8))
    assert 0.0 <= mismatched.score <= 100.0


def test_evaluator_mask_reduces_effective_area():
    evaluator = QualityEvaluator()
    image = np.concatenate(
        [_vertical_ridges()[:, :128], _uniform_noise()[:, :128]], axis=1
    ).astype(np.float32)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[:, :120] = 255
    masked = evaluator.evaluate(image, mask)
    unmasked = evaluator.evaluate(image)
    assert masked.clarity > unmasked.clarity
    assert masked.score > unmasked.score


def test_evaluator_minutiae_count_matches_extractor():
    from synthfp.plugins.minutiae import MinutiaeExtractor

    evaluator = QualityEvaluator()
    image = _ridge_grid()
    report = evaluator.evaluate(image)
    expected = len(MinutiaeExtractor().extract(image))
    assert report.minutiae_count == expected
    assert report.minutiae_count > 0
    # 理想条纹图无细节点
    assert evaluator.evaluate(_vertical_ridges()).minutiae_count == 0
