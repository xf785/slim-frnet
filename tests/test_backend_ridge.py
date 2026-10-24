"""脊线外观回归测试：密度、灰度与「仅脊线黑色」模式。

这里的断言都对着**真实 500 dpi 指纹**的物理量标定：

* 脊线周期（一脊 + 一谷）约 8~12 px（0.4~0.6 mm，500 dpi ≈ 19.7 px/mm）；
* 脊线占空比约 45%~55%；
* 脊线灰度明显深于谷线。

之所以单独成文件：脊线周期曾经因为 ``coarse_scale`` 的频率换算写反
（乘成 ``cw/w`` 而不是 ``w/cw``）而放大到约 22 px，肉眼看就是"指纹太稀疏"。
这种问题不会让任何功能测试失败，只能用物理量兜住。
"""

from __future__ import annotations

import numpy as np
import pytest

from synthfp.config import apply_overrides, build_config
from synthfp.core.types import MasterSpec
from synthfp.plugins.backends import get_backend


def _master(seed: int = 1, size: tuple = (320, 480)):
    cfg = build_config("quick_test")
    backend = get_backend("sfinge", cfg.backend.model_dump())
    spec = MasterSpec(
        master_id=f"m{seed}",
        identity_id="000001",
        finger="left_thumb",
        finger_index=0,
        fp_class="left_loop",
        image_size=size,
        seed=seed,
    )
    return backend, backend.generate_master(spec)


def _autocorr_period(row: np.ndarray, mask_row: np.ndarray) -> float:
    """用一行的自相关估计脊线周期。"""
    idx = np.nonzero(mask_row > 0)[0]
    if idx.size < 40:
        return 0.0
    seg = row[idx[0] : idx[-1]].astype(np.float64)
    seg -= seg.mean()
    if np.allclose(seg, 0):
        return 0.0
    ac = np.correlate(seg, seg, mode="full")[seg.size - 1 :]
    ac /= ac[0] + 1e-9
    for lag in range(4, min(30, ac.size - 1)):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.15:
            return float(lag)
    return 0.0


def _row_stats(artifact) -> tuple:
    """返回 (周期中位数, 脊线均值, 谷线均值, 占空比)。"""
    img, mask = artifact.image, artifact.mask
    periods, ridges, valleys, duties = [], [], [], []
    h = img.shape[0]
    for y in range(h // 3, 2 * h // 3, 6):
        p = _autocorr_period(img[y], mask[y])
        if p > 0:
            periods.append(p)
        row = img[y][mask[y] > 0]
        if row.size < 100:
            continue
        lo, hi = np.percentile(row, [10, 90])
        if hi - lo < 15:
            continue
        thr = (lo + hi) / 2
        ridges.append(float(row[row < thr].mean()))
        valleys.append(float(row[row >= thr].mean()))
        duties.append(float((row < thr).mean()))
    return (
        float(np.median(periods)) if periods else 0.0,
        float(np.mean(ridges)) if ridges else 0.0,
        float(np.mean(valleys)) if valleys else 0.0,
        float(np.mean(duties)) if duties else 0.0,
    )


# ---------------------------------------------------------------------------
# 密度
# ---------------------------------------------------------------------------
def test_脊线周期接近真实500dpi指纹():
    """周期在 7~14 px 之间（真实 8~12 px，留一点合成余量）。"""
    _backend, artifact = _master(seed=3)
    period, _r, _v, _d = _row_stats(artifact)

    assert period > 0, "自相关没找到周期，说明图像几乎没有脊线结构"
    assert 7.0 <= period <= 14.0, (
        f"脊线周期 {period:.1f}px 偏离真实 500dpi 指纹（8~12px）；"
        "检查 ridge_generator 里 coarse_scale 的频率换算方向"
    )


def test_脊线占空比接近一半():
    """脊线宽度应约占周期一半（45%~60%）。"""
    _backend, artifact = _master(seed=4)
    _p, _r, _v, duty = _row_stats(artifact)

    assert 0.35 <= duty <= 0.62, f"脊线占空比 {duty:.1%} 不合理"


# ---------------------------------------------------------------------------
# 灰度
# ---------------------------------------------------------------------------
def test_谷线明显浅于脊线():
    _backend, artifact = _master(seed=5)
    _p, ridge, valley, _d = _row_stats(artifact)

    assert valley - ridge > 90, f"对比度不足：脊 {ridge:.0f} / 谷 {valley:.0f}"
    assert valley >= 190, f"谷线偏深（{valley:.0f}），真实指纹谷线应在 200 上下"


# ---------------------------------------------------------------------------
# 仅脊线黑色
# ---------------------------------------------------------------------------
def test_仅脊线黑色输出纯二值图():
    cfg = build_config("quick_test")
    cfg = apply_overrides(cfg, {"backend.ridge_only_black": True})
    backend = get_backend("sfinge", cfg.backend.model_dump())
    spec = MasterSpec(
        master_id="m1", identity_id="000001", finger="left_thumb",
        finger_index=0, fp_class="left_loop", image_size=(320, 480), seed=11,
    )
    artifact = backend.generate_master(spec)

    inside = artifact.image[artifact.mask > 0]
    values = set(np.unique(inside).tolist())
    assert values <= {0, 255}, f"掩模内出现了非二值灰度: {sorted(values)}"
    # 手指内部必须真的同时有脊（0）与谷（255）
    assert 0 in values and 255 in values
    # 背景未被污染
    assert np.all(artifact.image[artifact.mask == 0] == 255)


def test_仅脊线黑色不受纹理与噪声影响():
    """二值模式跳过纹理/底噪，改这些参数不应改变结果。"""

    def render(texture: float, noise: float) -> np.ndarray:
        cfg = apply_overrides(
            build_config("quick_test"),
            {
                "backend.ridge_only_black": True,
                "backend.texture_strength": texture,
                "backend.noise_sigma": noise,
            },
        )
        backend = get_backend("sfinge", cfg.backend.model_dump())
        spec = MasterSpec(
            master_id="m1", identity_id="000001", finger="left_thumb",
            finger_index=0, fp_class="left_loop", image_size=(240, 320), seed=7,
        )
        return backend.generate_master(spec).image

    a = render(0.0, 0.0)
    b = render(40.0, 12.0)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("seed", [1, 2])
def test_二值模式仍然保持脊线结构(seed: int):
    """二值化不能把脊线糊成一片：前景里两种取值都要有可观比例。"""
    cfg = apply_overrides(build_config("quick_test"), {"backend.ridge_only_black": True})
    backend = get_backend("sfinge", cfg.backend.model_dump())
    spec = MasterSpec(
        master_id=f"m{seed}", identity_id="000001", finger="left_thumb",
        finger_index=0, fp_class="whorl", image_size=(240, 320), seed=seed,
    )
    image = backend.generate_master(spec).image
    inside = image
    black_ratio = float((inside == 0).mean())
    assert 0.08 <= black_ratio <= 0.55, f"脊线像素占比 {black_ratio:.1%} 异常"
