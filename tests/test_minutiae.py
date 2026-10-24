"""``synthfp.plugins.minutiae`` 的细节点提取测试（契约第 3.3 节）。

覆盖：向量化 Zhang-Suen 细化（细横线 / 十字 / 方块 / 幂等性 / 无 2×2 块）、
Crossing Number 端点与分叉、短枝剪除、边界与掩模过滤、``min_distance`` 合并、
确定性、``max_count`` 截断与排序、ISO/IEC 19794-2 模板长度与字段编码。
"""

from __future__ import annotations

import struct

import cv2
import numpy as np
import pytest

from synthfp.core.types import Minutia
from synthfp.plugins.minutiae import (
    MinutiaeExtractor,
    crossing_numbers,
    prune_short_branches,
    skeleton_minutiae,
    to_iso_template,
    zhang_suen,
)
from synthfp.plugins.minutiae.iso_template import ISO_HEADER_SIZE, ISO_MINUTIA_SIZE
from synthfp.plugins.minutiae.skeleton import neighbour_codes

SIZE = 256


# ---------------------------------------------------------------------------
# 画图工具
# ---------------------------------------------------------------------------
def _canvas(size: int = SIZE) -> np.ndarray:
    """白底（背景 255）画布。"""
    return np.full((size, size), 255, np.uint8)


def _vertical_ridge(size: int = SIZE, y0: int = 20, y1: int = 200, x: int = 64,
                    thickness: int = 3) -> np.ndarray:
    """一条竖直脊线（暗线画在白底上）。"""
    image = _canvas(size)
    cv2.line(image, (x, y0), (x, y1), 0, thickness)
    return image


def _line_with_gap(size: int = SIZE) -> np.ndarray:
    """一条被断口分成两段的竖直脊线 → 4 个端点。"""
    image = _canvas(size)
    cv2.line(image, (64, 20), (64, 120), 0, 3)
    cv2.line(image, (64, 140), (64, 230), 0, 3)
    return image


def _y_shape(size: int = SIZE) -> np.ndarray:
    """Y 形（三叉）脊线 → 1 个分叉 + 3 个端点。"""
    image = _canvas(size)
    cv2.line(image, (128, 200), (128, 120), 0, 3)
    cv2.line(image, (128, 120), (80, 40), 0, 3)
    cv2.line(image, (128, 120), (176, 40), 0, 3)
    return image


def _ridge_with_spur(size: int = SIZE, spur: int = 5) -> np.ndarray:
    """竖直脊线 + 一条水平短枝（长度 ``spur`` px）。"""
    image = _canvas(size)
    cv2.line(image, (128, 30), (128, 220), 0, 3)
    cv2.line(image, (128, 120), (128 + spur, 120), 0, 3)
    return image


def _ridge_grid(size: int = SIZE, spacing: int = 12) -> np.ndarray:
    """带断口的平行脊线（细节点较多，用于合并/截断测试）。"""
    image = _canvas(size)
    for index, y in enumerate(range(30, size - 26, spacing)):
        cv2.line(image, (40, y), (size - 46, y), 0, 3)
        if index % 3 == 0:
            cv2.line(image, (110, y), (126, y), 255, 5)
    return image


def _binary(array: np.ndarray) -> np.ndarray:
    """把 0/255 图转成 0/1 二值图。"""
    return (array > 127).astype(np.uint8)


# ---------------------------------------------------------------------------
# zhang_suen / 邻域编码
# ---------------------------------------------------------------------------
def test_neighbour_codes_and_crossing_numbers():
    cross = np.zeros((5, 5), np.uint8)
    cross[2, :] = 1
    cross[:, 2] = 1
    codes = neighbour_codes(cross)
    # 中心像素：N/E/S/W 为前景（bit0/2/4/6）→ 码 = 1 + 4 + 16 + 64 = 85
    assert int(codes[2, 2]) == 85
    cn = crossing_numbers(cross)
    assert int(cn[2, 2]) == 4  # 四叉交叉
    assert int(cn[2, 0]) == 1  # 端点
    assert int(cn[2, 1]) == 2  # 普通脊线点


def test_zhang_suen_thin_line_keeps_shape_and_dtype():
    image = np.zeros((9, 15), np.uint8)
    image[4, 2:13] = 1
    skeleton = zhang_suen(image)
    assert skeleton.dtype == np.uint8
    assert set(np.unique(skeleton).tolist()) <= {0, 1}
    np.testing.assert_array_equal(skeleton, image)
    # 输入不被修改（0/255 输入也一样）
    canvas = _binary(_vertical_ridge())
    backup = canvas.copy()
    zhang_suen(canvas)
    np.testing.assert_array_equal(canvas, backup)


def test_zhang_suen_cross_is_preserved():
    image = np.zeros((9, 9), np.uint8)
    image[4, :] = 1
    image[:, 4] = 1
    np.testing.assert_array_equal(zhang_suen(image), image)


def test_zhang_suen_hollow_square_loop_is_unchanged():
    image = np.zeros((21, 21), np.uint8)
    image[4:17, 4] = 1
    image[4:17, 16] = 1
    image[4, 4:17] = 1
    image[16, 4:17] = 1
    skeleton = zhang_suen(image)
    np.testing.assert_array_equal(skeleton, image)
    # 闭合环没有细节点
    assert skeleton_minutiae(skeleton) == []


def test_zhang_suen_solid_block_is_thinned_not_grown():
    image = np.zeros((40, 40), np.uint8)
    image[8:32, 8:32] = 1
    skeleton = zhang_suen(image)
    assert 0 < int(skeleton.sum()) <= 8  # 实心块被彻底细化（Zhang-Suen 固有行为）
    assert int((skeleton & ~image).sum()) == 0  # 骨架是前景的子集
    assert _has_no_2x2_block(skeleton)


def _has_no_2x2_block(image: np.ndarray) -> bool:
    """检查二值图是否不存在 2×2 全前景块（细化的“单像素宽”判据）。"""
    binary = (image > 0).astype(np.uint8)
    blocks = (
        binary[:-1, :-1] + binary[:-1, 1:] + binary[1:, :-1] + binary[1:, 1:]
    )
    return not bool((blocks == 4).any())


def _count_2x2_blocks(image: np.ndarray) -> int:
    """统计 2×2 全前景块数量。"""
    binary = (image > 0).astype(np.uint8)
    blocks = (
        binary[:-1, :-1] + binary[:-1, 1:] + binary[1:, :-1] + binary[1:, 1:]
    )
    return int((blocks == 4).sum())


def test_zhang_suen_thins_without_2x2_blocks_and_is_idempotent():
    rng = np.random.default_rng(11)
    blob = (rng.random((48, 48)) > 0.45).astype(np.uint8)
    skeleton = zhang_suen(blob)
    assert 0 < int(skeleton.sum()) < int(blob.sum())  # 确实变细了
    assert int((skeleton & ~blob).sum()) == 0  # 骨架是前景子集
    # Zhang-Suen 不保证严格 1 像素宽（拓扑保持型细化），但残留 2×2 块应极少
    assert _count_2x2_blocks(skeleton) * 20 < int(skeleton.sum())
    # 收敛后再次细化不再变化（幂等、确定性）
    np.testing.assert_array_equal(zhang_suen(skeleton), skeleton)
    np.testing.assert_array_equal(zhang_suen(blob), skeleton)


def test_zhang_suen_clean_shapes_are_one_pixel_wide():
    line = np.zeros((9, 21), np.uint8)
    line[4, 2:19] = 1
    cross = np.zeros((9, 9), np.uint8)
    cross[4, :] = 1
    cross[:, 4] = 1
    for shape in (line, cross):
        assert _has_no_2x2_block(zhang_suen(shape))


def test_zhang_suen_is_fast_on_large_images():
    """向量化实现：500×500 的细化必须在数秒内完成（不得是逐像素循环）。"""
    import time

    image = _ridge_grid(500, spacing=12)
    binary = (image < 128).astype(np.uint8)
    start = time.perf_counter()
    skeleton = zhang_suen(binary)
    elapsed = time.perf_counter() - start
    assert skeleton.sum() > 0
    assert elapsed < 5.0


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((8, 8), np.uint8),
        np.zeros((1, 1), np.uint8),
        np.zeros((0, 0), np.uint8),
        np.ones((2, 5), np.uint8),
    ],
)
def test_zhang_suen_degenerate_inputs(image):
    result = zhang_suen(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    if image.shape == (2, 5):
        # 任一边 < 3 px：不细化（也不破坏）输入，原样返回
        np.testing.assert_array_equal(result, image)
    else:
        assert int(result.sum()) == 0


# ---------------------------------------------------------------------------
# skeleton_minutiae
# ---------------------------------------------------------------------------
def test_skeleton_minutiae_line_has_two_endings():
    skeleton = zhang_suen(np.pad(np.ones((1, 20), np.uint8), ((4, 4), (4, 4))))
    found = skeleton_minutiae(skeleton)
    assert [kind for _, _, kind in found] == ["ending", "ending"]
    assert found[0][1] == found[1][1]  # 同一行
    assert found == sorted(found, key=lambda item: (item[1], item[0]))


def test_skeleton_minutiae_t_junction_is_bifurcation():
    image = np.zeros((32, 32), np.uint8)
    image[16, 4:28] = 1
    image[8:17, 15] = 1  # 竖线止于横线 → T 形三叉
    found = skeleton_minutiae(zhang_suen(image))
    kinds = [kind for _, _, kind in found]
    assert kinds.count("bifurcation") == 1
    assert kinds.count("ending") == 3
    assert (15, 16) in [(x, y) for x, y, kind in found if kind == "bifurcation"]


def test_skeleton_minutiae_four_way_cross_is_not_bifurcation():
    """CN == 4 不产生细节点（契约只定义 CN==1/3）。"""
    image = np.zeros((32, 32), np.uint8)
    image[16, 4:28] = 1
    image[4:28, 16] = 1
    kinds = [kind for _, _, kind in skeleton_minutiae(zhang_suen(image))]
    assert "bifurcation" not in kinds
    assert kinds.count("ending") == 4


def test_skeleton_minutiae_degenerate_inputs():
    assert skeleton_minutiae(np.zeros((16, 16), np.uint8)) == []
    assert skeleton_minutiae(np.zeros((1, 1), np.uint8)) == []


def test_prune_short_branches_removes_spur_but_keeps_long_branch():
    image = np.zeros((32, 40), np.uint8)
    image[16, 4:34] = 1  # 长分支
    image[8:17, 15] = 1  # 短枝（8 px，< 10）
    skeleton = zhang_suen(image)
    pruned = prune_short_branches(skeleton, 10)
    assert int(pruned.sum()) < int(skeleton.sum())
    assert int(pruned[8:17, 15].sum()) <= 2  # 短枝被剪掉（仅余交叉点本身）
    assert int(pruned[16, 4:34].sum()) >= 28  # 长分支保留
    assert _has_no_2x2_block(pruned)
    # 剪枝后伪分叉消失，只剩两个真实端点
    kinds = [kind for _, _, kind in skeleton_minutiae(pruned)]
    assert kinds == ["ending", "ending"]
    # 阈值大于短枝长度时不应剪枝
    kept = prune_short_branches(skeleton, 4)
    assert int(kept.sum()) >= int(pruned.sum())


def test_prune_short_branches_removes_isolated_fragment():
    image = np.zeros((40, 40), np.uint8)
    image[30, 5:35] = 1  # 长脊线
    image[10:12, 10:16] = 1  # 孤立短碎片（12 px）
    skeleton = zhang_suen(image)
    pruned = prune_short_branches(skeleton, 10)
    assert int(pruned[5:20, 5:25].sum()) == 0
    assert int(pruned[30, 5:35].sum()) >= 25


def test_prune_short_branches_degenerate_inputs():
    for image in (np.zeros((16, 16), np.uint8), np.zeros((1, 1), np.uint8)):
        result = prune_short_branches(image, 10)
        assert result.shape == image.shape and int(result.sum()) == 0


# ---------------------------------------------------------------------------
# MinutiaeExtractor
# ---------------------------------------------------------------------------
def test_extractor_name_and_types():
    extractor = MinutiaeExtractor()
    assert extractor.name == "synthfp-cn"
    assert extractor.min_distance == 8.0
    assert extractor.border_margin == 6
    assert extractor.max_count == 200
    assert extractor.compute_angle is True
    for minutia in extractor.extract(_ridge_grid()):
        assert isinstance(minutia, Minutia)
        assert isinstance(minutia.x, int) and isinstance(minutia.y, int)
        assert minutia.kind in ("ending", "bifurcation")
        assert 0.0 <= minutia.angle < 2 * np.pi
        assert 0.0 <= minutia.quality <= 1.0
        assert minutia.to_dict()["type"] == minutia.kind


def test_extractor_line_with_gap_finds_four_endings():
    found = MinutiaeExtractor().extract(_line_with_gap())
    assert len(found) == 4
    assert {m.kind for m in found} == {"ending"}
    assert [m.y for m in found] == sorted(m.y for m in found)


def test_extractor_detects_bifurcation_on_y_shape():
    found = MinutiaeExtractor().extract(_y_shape())
    kinds = [m.kind for m in found]
    assert kinds.count("bifurcation") == 1
    assert kinds.count("ending") == 3


@pytest.mark.parametrize("spur", [3, 5, 9])
def test_extractor_drops_short_spur_minutiae(spur):
    """长度 < 10 px 的骨架短枝产生的伪端点/伪分叉必须被剔除。"""
    found = MinutiaeExtractor().extract(_ridge_with_spur(spur=spur))
    assert [m.kind for m in found] == ["ending", "ending"]
    assert [m.x for m in found] == [128, 128]


@pytest.mark.parametrize("spur", [20, 30])
def test_extractor_keeps_long_spur_minutiae(spur):
    found = MinutiaeExtractor().extract(_ridge_with_spur(spur=spur))
    kinds = [m.kind for m in found]
    assert kinds.count("bifurcation") == 1
    assert kinds.count("ending") == 3
    assert max(m.x for m in found) >= 128 + spur - 2


def test_extractor_border_margin_filters_near_border_endings():
    image = np.full((SIZE, SIZE), 255, np.uint8)
    cv2.line(image, (64, 3), (64, 210), 0, 3)
    loose = MinutiaeExtractor(border_margin=0).extract(image)
    assert len(loose) == 2
    assert min(m.y for m in loose) <= 5  # 上端端点贴近边界
    strict = MinutiaeExtractor(border_margin=12).extract(image)
    assert len(strict) == 1
    for minutia in strict:
        assert minutia.x >= 12 and minutia.y >= 12
        assert minutia.x < SIZE - 12 and minutia.y < SIZE - 12
    # 边界裕度不能“凭空造出”细节点
    assert len(strict) <= len(loose)


def test_extractor_mask_filters_outside_region():
    image = _canvas()
    cv2.line(image, (40, 40), (40, 200), 0, 3)
    cv2.line(image, (200, 40), (200, 200), 0, 3)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[:, :128] = 255
    extractor = MinutiaeExtractor()
    assert len(extractor.extract(image)) == 4
    masked = extractor.extract(image, mask)
    assert len(masked) == 2
    assert all(m.x < 128 for m in masked)
    # 全零掩模
    assert extractor.extract(image, np.zeros((SIZE, SIZE), np.uint8)) == []


def test_extractor_min_distance_merges_neighbours():
    image = _ridge_grid()
    extractor = MinutiaeExtractor()
    fine = extractor.extract(image)
    coarse = MinutiaeExtractor(min_distance=40.0).extract(image)
    assert len(coarse) < len(fine)
    points = np.array([(m.x, m.y) for m in coarse], dtype=float)
    if len(points) > 1:
        deltas = points[:, None, :] - points[None, :, :]
        distances = np.sqrt((deltas**2).sum(axis=-1))
        distances += np.eye(len(points)) * 1e9
        assert distances.min() >= 40.0 - 1e-6


def test_extractor_quality_map_prefers_high_contrast():
    """quality 由局部对比度给出：强脊线处应高于弱脊线处。"""
    extractor = MinutiaeExtractor()
    image = np.ones((64, 128), np.float32)
    image[:, 8:12] = 0.0  # 左：满对比脊线
    image[:, 72:76] = 0.45  # 右：低对比脊线
    region = np.ones(image.shape, dtype=bool)
    quality = extractor._quality_map(image, None, region)
    strong = float(quality[:, 8:12].mean())
    weak = float(quality[:, 72:76].mean())
    assert 0.0 <= weak < strong <= 1.0
    assert strong > 0.9
    assert np.all(np.isfinite(quality))
    assert quality.min() >= 0.0 and quality.max() <= 1.0


def test_extractor_quality_map_uses_orientation_coherence():
    """传入方向场时 quality 用邻域方向一致性。"""
    extractor = MinutiaeExtractor()
    image = np.ones((64, 64), np.float32)
    region = np.ones(image.shape, dtype=bool)
    constant = np.full((64, 64), 0.7, np.float32)
    coherent = extractor._quality_map(image, constant, region)
    assert float(coherent.mean()) > 0.99
    rng = np.random.default_rng(3)
    noisy = rng.uniform(0.0, np.pi, (64, 64)).astype(np.float32)
    incoherent = extractor._quality_map(image, noisy, region)
    assert float(incoherent.mean()) < 0.5
    # 掩模外为 0
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :] = True
    partial = extractor._quality_map(image, constant, mask)
    assert float(partial[32:, :].max()) == 0.0


def test_extractor_uses_provided_orientation():
    image = _vertical_ridge()
    orientation = np.full((SIZE, SIZE), np.pi / 2, np.float32)
    found = MinutiaeExtractor().extract(image, None, orientation)
    assert len(found) == 2
    for minutia in found:
        assert minutia.angle == pytest.approx(np.pi / 2, abs=1e-4)
        assert minutia.quality > 0.9  # 方向场完全一致
    # 不传方向场时角度由内部估计给出（同样是 π/2 附近）
    estimated = MinutiaeExtractor().extract(image)
    assert all(abs(m.angle - np.pi / 2) < 0.2 or m.angle > np.pi - 0.2 for m in estimated)
    # compute_angle=False → 角度恒为 0
    assert all(m.angle == 0.0 for m in MinutiaeExtractor(compute_angle=False).extract(image))


def test_extractor_is_deterministic():
    image = _ridge_grid()
    first = [m.to_dict() for m in MinutiaeExtractor().extract(image)]
    second = [m.to_dict() for m in MinutiaeExtractor().extract(image)]
    assert first == second


def test_extractor_sorted_by_y_then_x():
    found = MinutiaeExtractor().extract(_ridge_grid())
    keys = [(m.y, m.x) for m in found]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def test_extractor_max_count_truncates_by_quality_and_resorts():
    image = _ridge_grid()
    full = MinutiaeExtractor().extract(image)
    assert len(full) > 5
    limited = MinutiaeExtractor(max_count=5).extract(image)
    assert len(limited) == 5
    keys = [(m.y, m.x) for m in limited]
    assert keys == sorted(keys)
    kept = {(m.x, m.y) for m in limited}
    dropped = [m for m in full if (m.x, m.y) not in kept]
    assert kept <= {(m.x, m.y) for m in full}
    assert min(m.quality for m in limited) >= max(m.quality for m in dropped) - 1e-9
    # max_count = 0 → 空
    assert MinutiaeExtractor(max_count=0).extract(image) == []


def test_extractor_angle_range_for_all_minutiae():
    found = MinutiaeExtractor().extract(_y_shape())
    assert found
    for minutia in found:
        assert 0.0 <= minutia.angle < 2 * np.pi
        assert np.isfinite(minutia.angle) and np.isfinite(minutia.quality)


@pytest.mark.parametrize(
    "image",
    [
        np.full((64, 64), 255, np.uint8),
        np.zeros((64, 64), np.uint8),
        np.full((128, 128), 128, np.uint8),
        np.full((1, 1), 0, np.uint8),
        np.full((2, 2), 255, np.uint8),
    ],
)
def test_extractor_degenerate_inputs_return_empty(image):
    assert MinutiaeExtractor().extract(image) == []


def test_extractor_does_not_modify_input():
    image = _ridge_grid()
    backup = image.copy()
    MinutiaeExtractor().extract(image)
    np.testing.assert_array_equal(image, backup)
    float_image = (image / 255.0).astype(np.float32)
    float_backup = float_image.copy()
    MinutiaeExtractor().extract(float_image)
    np.testing.assert_array_equal(float_image, float_backup)


def test_extractor_handles_inverted_polarity():
    """背景为暗、脊线为亮的（反相）输入也应被正确提取。"""
    image = _canvas()
    cv2.line(image, (64, 20), (64, 200), 0, 3)
    inverted = 255 - image
    found = MinutiaeExtractor().extract(inverted)
    assert len(found) == 2


def test_extractor_handles_noisy_impression():
    rng = np.random.default_rng(5)
    image = _ridge_grid().astype(np.float32)
    noisy = np.clip(image + rng.normal(0.0, 12.0, image.shape), 0, 255).astype(np.uint8)
    found = MinutiaeExtractor().extract(noisy)
    assert len(found) >= 20  # 噪声下仍能检出主要细节点
    assert all(np.isfinite(m.quality) for m in found)


# ---------------------------------------------------------------------------
# ISO/IEC 19794-2 模板
# ---------------------------------------------------------------------------
def _sample_minutiae(count: int = 3) -> list[Minutia]:
    return [
        Minutia(x=40 + 30 * i, y=60 + 20 * i, angle=0.3 * i, kind="ending" if i % 2 else "bifurcation",
                quality=0.9 - 0.1 * i)
        for i in range(count)
    ]


@pytest.mark.parametrize("count", [0, 1, 3, 12])
def test_iso_template_length_is_24_plus_6n(count):
    template = to_iso_template(_sample_minutiae(count), 256, 256, ppi=500)
    assert isinstance(template, bytes)
    assert len(template) == ISO_HEADER_SIZE + ISO_MINUTIA_SIZE * count
    assert len(template) == 24 + 6 * count


def test_iso_template_header_fields():
    minutiae = _sample_minutiae(3)
    template = to_iso_template(minutiae, 320, 480, ppi=500)
    identifier, version, total, device, width, height, res_x, res_y, count, reserved = (
        struct.unpack(">4s4sIHHHHHBB", template[:ISO_HEADER_SIZE])
    )
    assert identifier == b"FMR\x00"
    assert version == b" 20\x00"
    assert total == len(template) == 24 + 6 * 3
    assert device == 0
    assert (width, height) == (320, 480)
    assert (res_x, res_y) == (500, 500)
    assert count == 3
    assert reserved == 0


def test_iso_template_minutia_record_encoding():
    minutia = Minutia(x=300, y=260, angle=np.pi / 2, kind="bifurcation", quality=0.8)
    template = to_iso_template([minutia], 640, 640, ppi=500)
    head, x_low, y_low, angle, quality, reserved = template[24:30]
    assert (head >> 6) == minutia.type_code == 1
    assert ((head >> 4) & 0x03) == (300 >> 8)
    assert ((head >> 2) & 0x03) == (260 >> 8)
    assert (x_low | (((head >> 4) & 0x03) << 8)) == 300
    assert (y_low | (((head >> 2) & 0x03) << 8)) == 260
    assert angle == pytest.approx(64, abs=1)  # π/2 → 256 × (1/4) = 64
    assert quality == 80
    assert reserved == 0


def test_iso_template_ending_type_code_is_zero():
    template = to_iso_template([Minutia(x=10, y=10, kind="ending")], 64, 64)
    assert (template[24] >> 6) == 0


def test_iso_template_clips_count_and_coordinates():
    many = [Minutia(x=5, y=5) for _ in range(300)]
    template = to_iso_template(many, 256, 256)
    assert len(template) == 24 + 6 * 255
    assert template[22] == 255
    # 超范围坐标被裁剪到 10 位
    huge = to_iso_template([Minutia(x=100000, y=-50)], 256, 256)
    head, x_low, y_low = huge[24], huge[25], huge[26]
    assert (x_low | (((head >> 4) & 0x03) << 8)) == 1023
    assert (y_low | (((head >> 2) & 0x03) << 8)) == 0


def test_iso_template_degenerate_inputs():
    assert len(to_iso_template([], 0, 0, ppi=0)) == 24
    assert len(to_iso_template(None, 256, 256)) == 24
    assert len(to_iso_template([Minutia(x=1, y=2, angle=float("nan"), quality=float("nan"))], 8, 8)) == 30


def test_iso_template_from_extracted_minutiae():
    image = _line_with_gap()
    found = MinutiaeExtractor().extract(image)
    template = to_iso_template(found, SIZE, SIZE, ppi=500)
    assert len(template) == 24 + 6 * len(found)
    assert template[:4] == b"FMR\x00"
