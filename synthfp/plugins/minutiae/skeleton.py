"""向量化 Zhang-Suen 细化 + Crossing Number 细节点候选检测（纯 ``numpy``）。

实现要点
--------
* **禁止逐像素 Python 循环**：8 邻域通过带 1 像素零边界的 8 个切片视图取得，
  打包成 ``0~255`` 的邻域码，再用两张 256 项查找表并行判定两轮子迭代的
  可删除像素（Zhang & Suen, *A fast parallel algorithm for thinning digital
  patterns*, CACM 27(3), 1984）。
* Crossing Number（Rutovitz 定义）= 8 邻域循环序列中 ``0→1`` 跳变次数：
  ``CN == 1`` → 端点（ridge ending），``CN == 3`` → 分叉点（bifurcation）。

复杂度：每轮迭代为 O(H·W) 的常数次数组运算，500×500 图像一次细化通常
15 次迭代以内收敛。
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "as_binary",
    "neighbour_codes",
    "crossing_numbers",
    "zhang_suen",
    "skeleton_minutiae",
    "prune_short_branches",
]


def as_binary(array: np.ndarray) -> np.ndarray:
    """把任意输入转为 ``uint8``、取值 ``{0, 1}`` 的二值图副本。

    Args:
        array: 输入数组，非零视为前景（支持 ``bool`` / ``{0,1}`` / ``{0,255}``）。

    Returns:
        ``uint8 (H, W)``，仅含 0 与 1；输入不会被修改。
    """
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"as_binary 需要 2 维数组，实际形状 {arr.shape}")
    return (arr != 0).astype(np.uint8)


def _build_zhang_suen_luts() -> tuple[np.ndarray, np.ndarray]:
    """预计算 Zhang-Suen 两张 256 项删除查找表。

    邻域位序为 ``bit0..bit7 = P2(N), P3(NE), P4(E), P5(SE), P6(S), P7(SW),
    P8(W), P9(NW)``。

    Returns:
        ``(lut1, lut2)``，均为 ``uint8 (256,)``，1 表示该邻域模式下可删除。
    """
    lut1 = np.zeros(256, dtype=np.uint8)
    lut2 = np.zeros(256, dtype=np.uint8)
    for code in range(256):
        p = [(code >> i) & 1 for i in range(8)]  # p[0]=P2 ... p[7]=P9
        b = sum(p)
        if b < 2 or b > 6:
            continue
        transitions = sum(1 for i in range(8) if p[i] == 0 and p[(i + 1) % 8] == 1)
        if transitions != 1:
            continue
        p2, p3, p4, p5, p6, p7, p8, p9 = p
        if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
            lut1[code] = 1
        if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
            lut2[code] = 1
    return lut1, lut2


def _build_transition_lut() -> np.ndarray:
    """预计算 Crossing Number 查找表（循环 0→1 跳变次数）。

    Returns:
        ``uint8 (256,)``，取值 ``0~8``。
    """
    lut = np.zeros(256, dtype=np.uint8)
    for code in range(256):
        p = [(code >> i) & 1 for i in range(8)]
        lut[code] = sum(1 for i in range(8) if p[i] == 0 and p[(i + 1) % 8] == 1)
    return lut


_ZS_LUT1, _ZS_LUT2 = _build_zhang_suen_luts()
_TRANSITION_LUT = _build_transition_lut()


def neighbour_codes(binary: np.ndarray) -> np.ndarray:
    """把 8 邻域打包成 ``0~255`` 的整数码（图像外视为背景）。

    Args:
        binary: 二值图（非零为前景），形状 ``(H, W)``。

    Returns:
        ``uint8 (H, W)``，位序 ``bit0..bit7 = P2(N), P3(NE), P4(E), P5(SE),
        P6(S), P7(SW), P8(W), P9(NW)``。
    """
    img = as_binary(binary)
    padded = np.pad(img, 1, mode="constant")
    p2 = padded[0:-2, 1:-1]  # N
    p3 = padded[0:-2, 2:]  # NE
    p4 = padded[1:-1, 2:]  # E
    p5 = padded[2:, 2:]  # SE
    p6 = padded[2:, 1:-1]  # S
    p7 = padded[2:, 0:-2]  # SW
    p8 = padded[1:-1, 0:-2]  # W
    p9 = padded[0:-2, 0:-2]  # NW
    code = (
        p2
        | (p3 << np.uint8(1))
        | (p4 << np.uint8(2))
        | (p5 << np.uint8(3))
        | (p6 << np.uint8(4))
        | (p7 << np.uint8(5))
        | (p8 << np.uint8(6))
        | (p9 << np.uint8(7))
    )
    return code.astype(np.uint8, copy=False)


def crossing_numbers(binary: np.ndarray) -> np.ndarray:
    """计算每个前景像素的 Crossing Number（0~8）。

    Args:
        binary: 二值图（非零为前景），形状 ``(H, W)``。

    Returns:
        ``uint8 (H, W)``；背景像素的 CN 也为 0，需与前景掩模配合使用。
    """
    return _TRANSITION_LUT[neighbour_codes(binary)]


def zhang_suen(binary: np.ndarray, max_iter: int | None = None) -> np.ndarray:
    """向量化 Zhang-Suen 细化（骨架化）。

    两轮子迭代交替执行，每轮用 :func:`neighbour_codes` 打包邻域并查表并行
    删除满足条件的边界像素，直至没有任何像素被删除为止。

    Args:
        binary: 二值图（非零为前景），形状 ``(H, W)``，``bool`` / ``uint8`` /
            ``{0,255}`` 均可。
        max_iter: 最大完整迭代次数；``None`` 表示直到收敛（内部上限
            ``min(2·max(H, W) + 8, 400)``，用于兜底病态的实心大色块）。

    Returns:
        ``uint8 (H, W)``，取值 ``{0, 1}``，1 为细化后的脊线骨架。
        空输入 / 任一边 < 3 px 时原样返回。

    Note:
        不修改输入数组；结果为 8 连通、单像素宽（2×2 实心块会被并行删除，
        与标准 Zhang-Suen 行为一致）。
    """
    current = as_binary(binary)
    height, width = current.shape
    if height < 3 or width < 3 or not current.any():
        return current

    if max_iter is None:
        limit = int(min(2 * max(height, width) + 8, 400))
    else:
        limit = int(max(0, max_iter))

    for _ in range(limit):
        previous = int(current.sum())
        for lut in (_ZS_LUT1, _ZS_LUT2):
            removable = lut[neighbour_codes(current)]
            if removable.any():
                current = current & np.logical_not(removable).astype(np.uint8)
        if int(current.sum()) == previous:
            break
    return current.astype(np.uint8, copy=False)


def skeleton_minutiae(skeleton: np.ndarray) -> list[tuple[int, int, str]]:
    """在骨架上检测细节点候选（未做任何伪细节点过滤）。

    ``CN == 1`` → ``"ending"``；``CN == 3`` → ``"bifurcation"``。
    ``CN == 0``（孤立点）、``CN == 2``（普通脊线点）、``CN >= 4``（四叉交叉）
    均不产生细节点。

    Args:
        skeleton: 骨架图（非零为前景），形状 ``(H, W)``。

    Returns:
        ``[(x, y, kind), ...]``，按 ``(y, x)`` 升序排列以保证确定性；
        ``kind ∈ {"ending", "bifurcation"}``。
    """
    skel = as_binary(skeleton)
    height, width = skel.shape
    if height < 3 or width < 3 or not skel.any():
        return []
    cn = crossing_numbers(skel)
    foreground = skel > 0
    ys, xs = np.nonzero(foreground & (cn == 1))
    results: list[tuple[int, int, str]] = [
        (int(x), int(y), "ending") for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
    ]
    ys, xs = np.nonzero(foreground & (cn == 3))
    results.extend(
        (int(x), int(y), "bifurcation")
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
    )
    results.sort(key=lambda item: (item[1], item[0]))
    return results


def geodesic_distance(mask: np.ndarray, seeds: np.ndarray, cap: int) -> np.ndarray:
    """骨架上的测地（沿骨架）距离场，超过 ``cap`` 记为 ``cap + 1``。

    用 8 邻域膨胀在骨架内部做“火烧”传播：每轮前进一步，因此得到的是沿骨架
    的像素步数，而不是欧氏距离（后者会在平行脊线之间“跳线”）。

    Args:
        mask: 骨架（非零为前景），形状 ``(H, W)``。
        seeds: 种子像素布尔掩膜（距离 0）。
        cap: 最大传播步数。

    Returns:
        ``int32 (H, W)`` 距离场；``mask`` 之外恒为 ``cap + 1``。
    """
    binary = as_binary(mask)
    limit = int(max(0, cap))
    distance = np.full(binary.shape, limit + 1, dtype=np.int32)
    seed_mask = np.asarray(seeds, dtype=bool) & (binary > 0)
    if not seed_mask.any():
        return distance
    distance[seed_mask] = 0
    frontier = seed_mask
    kernel = np.ones((3, 3), np.uint8)
    for step in range(1, limit + 1):
        reached = cv2.dilate(
            frontier.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        frontier = (reached > 0) & (binary > 0) & (distance > limit)
        if not frontier.any():
            break
        distance[frontier] = step
    return distance


def prune_short_branches(
    skeleton: np.ndarray, min_length: int = 10, max_passes: int | None = None
) -> np.ndarray:
    """剪除长度小于 ``min_length`` 的骨架短枝（毛刺）与孤立短碎片。

    两步（全部向量化，无逐像素行走）：

    1. **毛刺**：迭代删除“到最近交叉点（``CN >= 3``）的测地距离 <=
       ``min_length``”的端点，直到没有可删端点（最多 ``min_length + 1`` 轮）。
       短枝被逐层吃掉后，原本由毛刺造成的分叉点会退化为普通脊线点或端点，
       而脊线末端因毛刺形成的小叉会收敛成一个干净的真实端点。
    2. **孤立短碎片**：删掉不含任何交叉点且像素数 < ``min_length`` 的
       8 连通分量（噪声碎片）。

    说明：本方法对 8 连通“阶梯”骨架（``cv2.line`` 画出的斜线）鲁棒——
    阶梯处像素有两个同属一条分支的后继，逐像素行走法会误判为分叉。

    Args:
        skeleton: 骨架图（非零为前景），形状 ``(H, W)``。
        min_length: 短枝长度阈值（px），默认 10。
        max_passes: 毛刺删除的最大轮数；``None`` 表示 ``min_length + 1``。

    Returns:
        ``uint8 (H, W)`` 剪枝后的骨架（取值 ``{0, 1}``）；输入不会被修改。
    """
    current = as_binary(skeleton)
    height, width = current.shape
    if height < 3 or width < 3 or not current.any() or int(min_length) <= 1:
        return current
    limit = int(min_length)
    passes = limit + 1 if max_passes is None else int(max(1, max_passes))

    for _ in range(passes):
        cn = crossing_numbers(current)
        junctions = (current > 0) & (cn >= 3)
        if not junctions.any():
            break
        distance = geodesic_distance(current, junctions, limit)
        tips = (current > 0) & (cn == 1) & (distance <= limit)
        if not tips.any():
            break
        current = (current & ~tips).astype(np.uint8)

    if current.any():
        cn = crossing_numbers(current)
        junctions = (current > 0) & (cn >= 3)
        count, labels = cv2.connectedComponents(current, connectivity=8)
        if count > 1:
            sizes = np.bincount(labels.ravel(), minlength=count)
            has_junction = np.zeros(count, dtype=bool)
            if junctions.any():
                has_junction[np.unique(labels[junctions])] = True
            fragments = (sizes < limit) & ~has_junction
            fragments[0] = False
            if fragments.any():
                current = (current & ~fragments[labels]).astype(np.uint8)
    return current
