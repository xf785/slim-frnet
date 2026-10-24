"""基于骨架化 + Crossing Number 的细节点提取器（含伪细节点过滤）。

流程（见 ``CONTRACTS.md`` 第 3.1 节）::

    局部均值归一化 + Otsu 二值化（脊线为前景）
      → 向量化 Zhang-Suen 细化
      → 短枝剪除（长度 < 10 px 的毛刺 / 孤立碎片）
      → Crossing Number（CN==1 端点 / CN==3 分叉）
      → 伪细节点过滤（边界、掩模、近邻合并）
      → 质量与方向赋值
      → 按 (y, x) 排序、按 max_count 截断

参考文献
--------
* Zhang & Suen, CACM 27(3), 1984（细化）。
* Rutovitz, *Pattern Recognition*, J. Roy. Statist. Soc. 129, 1966
  （Crossing Number 细节点判定）。
* Ratha, Chen & Jain, *Adaptive flow orientation-based feature extraction in
  fingerprint images*, Pattern Recognition 28(11), 1995（伪细节点过滤）。

已知限制
--------
* 二值化是“局部均值归零 + 全图 Otsu”，因此当同一幅图里同时存在极强脊线与
  极弱脊线时，极弱脊线可能落在 Otsu 阈值之外而漏检（掩模 / 方向场不参与
  阈值选择）。
* 图像四周明显偏暗（``< 0.4``）时按“背景为暗”反相处理，以兼容反相输入；
  若真实指纹的暗背景恰好延伸至图像边缘，极性判断可能出错。
"""

from __future__ import annotations

import cv2
import numpy as np

from ...core.types import Minutia
from ..quality.metrics import (
    CONTRAST_REF,
    _local_stats,  # 同项目内部工具复用：局部均值/标准差
    estimate_orientation,
    to_float01,
)
from .skeleton import prune_short_branches, skeleton_minutiae, zhang_suen

__all__ = [
    "MinutiaeExtractor",
    "MIN_BRANCH_LENGTH",
    "QUALITY_WINDOW",
    "DEFAULT_MIN_DISTANCE",
    "DEFAULT_BORDER_MARGIN",
    "DEFAULT_MAX_COUNT",
]

MIN_BRANCH_LENGTH = 10
"""短枝剔除阈值（px）：长度小于该值的骨架短枝产生的细节点一律丢弃。"""

QUALITY_WINDOW = 15
"""quality 估计邻域边长（px，即契约中的 “cn 邻域”）。"""

DEFAULT_MIN_DISTANCE = 8.0
"""默认近邻合并距离（px）。"""

DEFAULT_BORDER_MARGIN = 6
"""默认边界安全距离（px）。"""

DEFAULT_MAX_COUNT = 200
"""默认细节点数量上限（与 ISO 模板 1 字节计数上限 255 兼容）。"""

_TWO_PI = 2.0 * np.pi


class MinutiaeExtractor:
    """基于骨架化 + Crossing Number 的细节点提取器（含伪细节点过滤）。

    典型用法::

        extractor = MinutiaeExtractor(min_distance=8.0, border_margin=6)
        minutiae = extractor.extract(image_uint8, mask, orientation)

    Attributes:
        min_distance: 近邻合并距离（px），相互距离小于该值的细节点只保留
            ``quality`` 最高的一个。
        border_margin: 边界安全距离（px），距图像边界不足该值的细节点被丢弃。
        max_count: 细节点数量上限，超出时按 ``quality`` 降序截断。
        compute_angle: 是否计算细节点方向（取局部方向场）。
    """

    def __init__(
        self,
        min_distance: float = DEFAULT_MIN_DISTANCE,
        border_margin: int = DEFAULT_BORDER_MARGIN,
        max_count: int = DEFAULT_MAX_COUNT,
        compute_angle: bool = True,
    ) -> None:
        """初始化提取器。

        Args:
            min_distance: 近邻合并距离（px），默认 8.0。
            border_margin: 边界安全距离（px），默认 6。
            max_count: 细节点数量上限，默认 200。
            compute_angle: 是否计算方向，默认 ``True``。
        """
        self.min_distance = float(min_distance) if np.isfinite(float(min_distance)) else 0.0
        self.min_distance = max(0.0, self.min_distance)
        self.border_margin = int(max(0, border_margin))
        self.max_count = int(max(0, max_count))
        self.compute_angle = bool(compute_angle)

    @property
    def name(self) -> str:
        """提取器标识名。"""
        return "synthfp-cn"

    # -- 二值化 -----------------------------------------------------------
    @staticmethod
    def _binarize(image: np.ndarray, region: np.ndarray) -> np.ndarray:
        """局部均值归一化 + Otsu 二值化，脊线为前景（1）。

        Args:
            image: ``float32 (H, W)``，取值 ``[0, 1]``，背景为白。
            region: 有效区域布尔掩模。

        Returns:
            ``uint8 (H, W)``，取值 ``{0, 1}``。
        """
        height, width = image.shape
        border = np.concatenate(
            [image[0, :], image[-1, :], image[:, 0], image[:, -1]]
        )
        dark_background = float(border.mean()) < 0.4
        inverted = image.copy() if dark_background else (1.0 - image)

        if not region.all():
            fill = float(np.median(inverted[region])) if region.any() else 0.0
            inverted = np.where(region, inverted, fill)

        window = int(min(31, max(9, (min(height, width) // 8) | 1)))
        local_mean = cv2.boxFilter(
            inverted, -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT
        )
        normalized = inverted - local_mean
        values = normalized[region]
        if values.size == 0:
            return np.zeros((height, width), dtype=np.uint8)
        low = float(values.min())
        high = float(values.max())
        if high - low <= 1e-6:
            return np.zeros((height, width), dtype=np.uint8)
        scaled = ((normalized - low) / (high - low) * 255.0).astype(np.uint8)
        _, thresholded = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (thresholded > 0) & (normalized > 0.0) & region
        return binary.astype(np.uint8)

    # -- 区域掩模 ---------------------------------------------------------
    @staticmethod
    def _region(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
        """把掩模规范成与图像同形状的布尔区域（``None`` 表示全图）。

        Args:
            mask: 掩模数组或 ``None``；形状不符或为空时退回全图。
            shape: 目标形状 ``(H, W)``。

        Returns:
            ``bool (H, W)``。
        """
        height, width = int(shape[0]), int(shape[1])
        if mask is None:
            return np.ones((height, width), dtype=bool)
        array = np.asarray(mask)
        if array.ndim == 3 and array.shape[2] >= 1:
            array = array[:, :, 0]
        if array.shape != (height, width):
            if array.size == 0:
                return np.zeros((height, width), dtype=bool)
            array = cv2.resize(
                array.astype(np.uint8, copy=False),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        with np.errstate(invalid="ignore"):
            return np.asarray(array > 0, dtype=bool)

    # -- 质量图 -----------------------------------------------------------
    @staticmethod
    def _quality_map(
        image: np.ndarray,
        orientation: np.ndarray | None,
        region: np.ndarray,
    ) -> np.ndarray:
        """计算逐像素 quality：方向一致性（有方向场）或局部对比度。

        Args:
            image: ``float32 (H, W)`` 图像。
            orientation: 方向场；``None`` 或形状不符时改用局部对比度。
            region: 有效区域布尔掩模（掩模外置 0）。

        Returns:
            ``float32 (H, W)``，取值 ``[0, 1]``。
        """
        height, width = image.shape
        k = int(min(QUALITY_WINDOW, max(3, min(height, width))) | 1)
        ksize = (k, k)
        use_orientation = (
            orientation is not None and np.asarray(orientation).shape == image.shape
        )
        if use_orientation:
            theta = np.mod(
                np.nan_to_num(
                    np.asarray(orientation, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                np.pi,
            )
            ex = cv2.boxFilter(
                np.cos(2.0 * theta), -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT
            )
            ey = cv2.boxFilter(
                np.sin(2.0 * theta), -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT
            )
            quality = np.sqrt(ex * ex + ey * ey)
        else:
            _, std = _local_stats(image, k)
            quality = std / CONTRAST_REF
        quality = np.clip(np.nan_to_num(quality, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        quality = np.where(region, quality, 0.0)
        return quality.astype(np.float32, copy=False)

    # -- 近邻合并 ---------------------------------------------------------
    def _merge_close(self, items: list[tuple]) -> list[tuple]:
        """按 quality 降序贪心抑制，仅保留相互距离 ≥ ``min_distance`` 的细节点。

        Args:
            items: ``[(x, y, kind, quality, angle), ...]``。

        Returns:
            过滤后的同构列表（保持输入顺序无关的确定性）。
        """
        if self.min_distance <= 0.0 or len(items) < 2:
            return list(items)
        order = sorted(
            range(len(items)), key=lambda i: (-items[i][3], items[i][1], items[i][0])
        )
        kept: list[int] = []
        limit_sq = self.min_distance * self.min_distance
        for index in order:
            x, y = items[index][0], items[index][1]
            ok = True
            for other in kept:
                dx = x - items[other][0]
                dy = y - items[other][1]
                if dx * dx + dy * dy < limit_sq:
                    ok = False
                    break
            if ok:
                kept.append(index)
        kept.sort()
        return [items[i] for i in kept]

    # -- 公开 API ---------------------------------------------------------
    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ) -> list[Minutia]:
        """从指纹图像中提取细节点。

        Args:
            image: 输入图像 ``uint8 (H, W)``（背景为白）或 ``float32 (H, W)``
                取值 ``[0, 1]``。
            mask: 有效区域掩模 ``uint8 (H, W)`` 取值 ``{0, 255}``；``None``
                表示整幅图有效，全零掩模返回空列表。
            orientation: 预计算的方向场 ``float32 (H, W)``（弧度）。传入时
                ``quality`` 用邻域方向一致性；未传入时用局部对比度，
                且 ``compute_angle=True`` 时内部调用
                :func:`~synthfp.plugins.quality.metrics.estimate_orientation`
                现场估计。

        Returns:
            :class:`~synthfp.core.types.Minutia` 列表，按 ``(y, x)`` 升序排列，
            长度不超过 ``max_count``；``angle ∈ [0, 2π)``、``quality ∈ [0, 1]``。
            退化输入（空掩模 / 纯色图 / 极小图 / ``max_count=0``）返回 ``[]``。
        """
        if self.max_count <= 0:
            return []
        gray = to_float01(image)
        height, width = gray.shape
        if height < 3 or width < 3:
            return []

        region = self._region(mask, gray.shape)
        if not region.any():
            return []

        binary = self._binarize(gray, region)
        # 先剪除长度 < MIN_BRANCH_LENGTH 的骨架短枝，再做 Crossing Number 检测：
        # 这样毛刺造成的伪端点/伪分叉在检测前就已被消除，且脊线末端的
        # “毛刺小叉”会在剪枝后退化为一个干净的真实端点。
        skeleton = prune_short_branches(zhang_suen(binary), MIN_BRANCH_LENGTH)
        if not skeleton.any():
            return []
        candidates = skeleton_minutiae(skeleton)
        if not candidates:
            return []

        quality_map = self._quality_map(gray, orientation, region)
        field: np.ndarray | None = None
        if orientation is not None and np.asarray(orientation).shape == gray.shape:
            field = np.mod(
                np.nan_to_num(
                    np.asarray(orientation, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                np.pi,
            ).astype(np.float32, copy=False)
        elif self.compute_angle:
            field, _ = estimate_orientation(gray)

        margin = self.border_margin
        items: list[tuple[int, int, str, float, float]] = []
        for x, y, kind in candidates:
            if x < margin or y < margin or x >= width - margin or y >= height - margin:
                continue
            if not region[y, x]:
                continue
            quality = float(quality_map[y, x])
            if not np.isfinite(quality):
                quality = 0.0
            quality = float(min(max(quality, 0.0), 1.0))
            if self.compute_angle and field is not None:
                angle = float(np.mod(float(field[y, x]), _TWO_PI))
                if not np.isfinite(angle):
                    angle = 0.0
            else:
                angle = 0.0
            items.append((int(x), int(y), str(kind), quality, angle))

        items = self._merge_close(items)
        items.sort(key=lambda item: (item[1], item[0]))
        if len(items) > self.max_count:
            items.sort(key=lambda item: (-item[3], item[1], item[0]))
            items = items[: self.max_count]
            items.sort(key=lambda item: (item[1], item[0]))

        return [
            Minutia(x=x, y=y, angle=angle, kind=kind, quality=quality)
            for x, y, kind, quality, angle in items
        ]
