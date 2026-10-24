"""SFinGe Gabor 迭代脊线合成器。

算法出处：Cappelli, R. et al. "Synthetic Fingerprint-Image Generation." ICPR 2000.

迭代过程：
1. 在手指区域内随机撒极少量种子像素（默认 0.1%）；
2. 每一轮：对当前图案按每个像素局部的 (方向, 频率) 选择对应 Gabor 核做卷积，
   取响应 > 0 的像素作为下一轮图案；
3. 直到图案的像素数变化率小于收敛阈值，或达到迭代上限。

v2.0 的重构（算法不变，只优化工程实现）：
* Gabor 核库改为模块级 ``lru_cache`` 缓存，多变体 / 多进程复用，省去反复重建核的开销；
* 方向场按 ``[0, π)``（脊线切向）离散为 36 个方向，与 ``orientation_field`` 的约定一致；
* 逐核循环内部为向量化布尔赋值；
* **移除内嵌的 Zhang-Suen 细化与细节点检测**：这部分在 v2.0 中由独立的
  ``synthfp.plugins.minutiae`` 插件在最终样本图上完成，避免每个主指纹重复
  付出一遍 O(200×H×W) 的 Python 逐像素循环（原实现在 320×480 上耗时数十秒）。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["RidgeGenerator", "build_gabor_bank", "clear_gabor_cache"]


# ---------------------------------------------------------------------------
# Gabor 核库（跨实例 / 跨进程复用）
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def build_gabor_bank(
    min_freq: float,
    max_freq: float,
    n_directions: int,
    n_frequencies: int,
    half_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造 Gabor 核库。

    ``kernel(θ, f) = Gaussian(u, v) · cos(2πf·u) - mean``，其中 ``u`` 为沿 θ 方向
    的坐标。θ 取脊线**切向**，因此滤波在某点产生正响应代表该点位于脊线上。

    Args:
        min_freq: 最低空间频率（脊线/像素）。
        max_freq: 最高空间频率。
        n_directions: 方向离散数。
        n_frequencies: 频率离散数。
        half_k: 核半径（核尺寸 = ``2*half_k+1``）。

    Returns:
        ``(bank, theta_values, freq_values)``；
        bank 形状 ``(n_directions*n_frequencies, 2*half_k+1, 2*half_k+1)``，
        索引为 ``d_idx * n_frequencies + f_idx``。
    """
    kernel_size = 2 * half_k + 1
    thetas = np.linspace(0.0, np.pi, n_directions, endpoint=False)
    freqs = np.linspace(min_freq, max_freq, n_frequencies)

    bank = np.empty((n_directions * n_frequencies, kernel_size, kernel_size), dtype=np.float32)
    yc, xc = np.mgrid[-half_k : half_k + 1, -half_k : half_k + 1].astype(np.float32)

    for d_idx, theta in enumerate(thetas):
        ct, st = float(np.cos(theta)), float(np.sin(theta))
        # u: 沿脊线切向；v: 垂直脊线切向
        u = xc * ct + yc * st
        v = -xc * st + yc * ct
        for f_idx, freq in enumerate(freqs):
            f = max(float(freq), 0.01)
            sigma = np.sqrt(max(-9.0 / (8.0 * f * f * np.log(0.001)), 0.25))
            gauss = np.exp(-0.5 * (u * u + v * v) / (sigma * sigma))
            kernel = (gauss * np.cos(2.0 * np.pi * f * u)).astype(np.float32)
            bank[d_idx * n_frequencies + f_idx] = kernel - kernel.mean()

    return bank, thetas, freqs


def clear_gabor_cache() -> None:
    """清空 Gabor 核库缓存。"""
    build_gabor_bank.cache_clear()


# ---------------------------------------------------------------------------
# 脊线生成器
# ---------------------------------------------------------------------------
class RidgeGenerator:
    """SFinGe Gabor 迭代脊线合成器。

    Example:
        >>> gen = RidgeGenerator(min_freq=1 / 15, max_freq=1 / 5, max_iterations=80)
        >>> image = gen.generate(orientation, frequency_map, finger_mask, rng)
    """

    def __init__(
        self,
        n_iterations: int = 80,
        seed_density: float = 0.001,
        gabor_filter_size: int = 10,
        n_directions: int = 36,
        n_frequencies: int = 5,
        convergence_eps: float = 1e-3,
        min_freq: float = 1.0 / 15.0,
        max_freq: float = 1.0 / 5.0,
        max_iterations: Optional[int] = None,
        coarse_scale: float = 0.7,
        **_: object,
    ) -> None:
        """
        Args:
            n_iterations: 迭代上限（``max_iterations`` 给出时以后者为准）。
            seed_density: 初始种子像素比例。
            gabor_filter_size: Gabor 核半径。
            n_directions: 方向离散数。
            n_frequencies: 频率离散数。指纹的局部脊线周期变化平缓，5 档足以表达；
                档数直接决定每次迭代的卷积次数，是吞吐的第一杠杆。
            convergence_eps: 收敛阈值（像素变化率）。
            min_freq / max_freq: 频率范围，由脊宽决定（``freq = 1 / width``）。
            max_iterations: 迭代上限（向后兼容旧调用）。
            coarse_scale: 迭代分辨率比例（0.3~1.0）。迭代在缩放后的网格上进行，
                最后再放大回原尺寸做灰度渲染。0.7 在几乎无肉眼差异的前提下
                把卷积量降到约一半；置 1.0 可得到最锐利的脊线。
        """
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.max_iterations = int(max_iterations if max_iterations is not None else n_iterations)
        self.n_iterations = self.max_iterations
        self.seed_density = float(seed_density)
        self.half_k = int(gabor_filter_size)
        self.kernel_size = 2 * self.half_k + 1
        self.n_directions = int(n_directions)
        self.n_frequencies = max(int(n_frequencies), 2)
        self.convergence_eps = float(convergence_eps)
        self.coarse_scale = float(min(max(coarse_scale, 0.3), 1.0))

        # 外观参数
        self._ridge_color: Tuple[int, int] = (25, 70)
        self._valley_color: Tuple[int, int] = (210, 248)
        self._texture_strength = 10.0
        self._noise_sigma = 1.5
        self._mask_feather_width = 12
        self._ridge_only_black = False

        self._debug_steps: List[np.ndarray] = []
        self._iterations_used = 0

    # -- 参数设置（链式友好） ---------------------------------------------
    def set_ridge_width(self, min_width: int, max_width: int) -> "RidgeGenerator":
        """按脊线宽度（像素）设定频率范围：``freq = 1 / width``。"""
        self.min_freq = 1.0 / max(int(max_width), 1)
        self.max_freq = 1.0 / max(int(min_width), 1)
        return self

    def set_ridge_only_black(self, enabled: bool = True) -> "RidgeGenerator":
        """仅脊线显示为黑色：输出纯二值图（脊线 0 / 谷线 255）。"""
        self._ridge_only_black = bool(enabled)
        return self

    def set_ridge_color(self, min_val: int, max_val: int) -> "RidgeGenerator":
        self._ridge_color = (int(min_val), int(max_val))
        return self

    def set_valley_color(self, min_val: int, max_val: int) -> "RidgeGenerator":
        self._valley_color = (int(min_val), int(max_val))
        return self

    def set_texture_strength(self, strength: float) -> "RidgeGenerator":
        self._texture_strength = float(max(strength, 0.0))
        return self

    def set_noise_sigma(self, sigma: float) -> "RidgeGenerator":
        self._noise_sigma = float(max(sigma, 0.0))
        return self

    def set_mask_feather_width(self, width: int) -> "RidgeGenerator":
        self._mask_feather_width = int(max(width, 0))
        return self

    @property
    def frequency_range(self) -> Tuple[float, float]:
        return (self.min_freq, self.max_freq)

    @property
    def iterations_used(self) -> int:
        return self._iterations_used

    # -- 主流程 -----------------------------------------------------------
    def generate(
        self,
        orientation_field: np.ndarray,
        frequency_map: np.ndarray,
        finger_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """合成指纹灰度图（uint8，背景为 255）。

        Args:
            orientation_field: ``(H, W)`` float32，脊线切向 ``[0, π)`` 弧度。
            frequency_map: ``(H, W)`` float32，脊线频率（脊线/像素）。
            finger_mask: ``(H, W)`` uint8 ``{0, 255}``。
            rng: 随机源。

        Returns:
            ``(H, W)`` uint8 灰度图，手指区域内为脊线纹理，区域外为 255。
        """
        h, w = int(orientation_field.shape[0]), int(orientation_field.shape[1])
        mask_bool = finger_mask > 0
        if not mask_bool.any():
            return np.full((h, w), 255, dtype=np.uint8)

        pattern = self._iterate(orientation_field, frequency_map, mask_bool, rng)
        gray = self._to_grayscale(pattern, mask_bool, rng, h, w)
        if not self._ridge_only_black:
            # 二值模式要的是"真二值"，羽化会在指边制造过渡灰，所以跳过
            gray = self._feather_edges(gray, mask_bool)
        gray[~mask_bool] = 255
        return gray

    def ridge_pattern(
        self,
        orientation_field: np.ndarray,
        frequency_map: np.ndarray,
        finger_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """只返回二值脊线图案（``{0,255}`` uint8，手指区域内有效）。

        标注阶段（细节点 / 中心线）用它作为输入，可避免从灰度图重新二值化。
        """
        mask_bool = finger_mask > 0
        pattern = self._iterate(orientation_field, frequency_map, mask_bool, rng)
        out = (pattern > 0.5).astype(np.uint8) * 255
        out[~mask_bool] = 0
        return out

    # -- 内部 -------------------------------------------------------------
    def _iterate(
        self,
        orientation_field: np.ndarray,
        frequency_map: np.ndarray,
        mask_bool: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """执行 Gabor 迭代，返回与输入同尺寸的 ``[0, 1]`` float32 图案。

        迭代在 ``coarse_scale`` 缩放后的网格上进行（卷积代价与像素数成正比），
        收敛后把图案放大回原尺寸，供后续灰度渲染使用。
        """
        h, w = orientation_field.shape
        if not mask_bool.any():
            self._iterations_used = 0
            return np.zeros((h, w), dtype=np.float32)

        scale = self.coarse_scale
        if scale < 0.999:
            ch, cw = max(int(round(h * scale)), 16), max(int(round(w * scale)), 16)
            # 频率是"每像素"量。
            #
            # 同一根手指在缩放后的网格上占 cw 个像素而不是 w 个，脊线间距
            # 也按 same 比例缩成 S*scale，因此频率必须变成 f/scale = f*(w/cw)。
            # 早先误写成 f*(cw/w)（乘反了），脊线周期被放大 1/scale² ≈ 2 倍，
            # 生成出来的指纹明显比真实指纹稀疏。
            ori_s = cv2.resize(orientation_field.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR)
            freq_s = (
                cv2.resize(frequency_map.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR)
                * (w / float(cw))
            )
            mask_s = cv2.resize(mask_bool.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST) > 0
        else:
            ori_s, freq_s, mask_s = orientation_field, frequency_map, mask_bool
            ch, cw = h, w

        bank, _, _ = build_gabor_bank(
            self.min_freq, self.max_freq, self.n_directions, self.n_frequencies, self.half_k
        )
        kernel_map = self._kernel_index_map(ori_s, freq_s, mask_s)

        pattern = (rng.random((ch, cw)) < self.seed_density).astype(np.float32)
        self._debug_steps = []
        self._iterations_used = 0

        for iteration in range(self.max_iterations):
            prev_nz = int(np.count_nonzero(pattern))
            nxt = np.zeros((ch, cw), dtype=np.float32)
            for k_idx in np.unique(kernel_map[mask_s]):
                pixel_mask = kernel_map == k_idx
                response = cv2.filter2D(pattern, -1, bank[int(k_idx)])
                nxt[pixel_mask] = (response[pixel_mask] > 0.0).astype(np.float32)
            pattern = nxt
            self._iterations_used = iteration + 1

            nz = int(np.count_nonzero(pattern))
            if abs(nz - prev_nz) / max(ch * cw, 1) <= self.convergence_eps:
                break
            if iteration % 20 == 0 and len(self._debug_steps) < 8:
                self._debug_steps.append(pattern.copy())

        if len(self._debug_steps) < 8:
            self._debug_steps.append(pattern.copy())

        if (ch, cw) != (h, w):
            # 双三次上采样后再二值化，边缘比最近邻平滑
            pattern = cv2.resize(pattern, (w, h), interpolation=cv2.INTER_CUBIC)
        return pattern

    def _kernel_index_map(
        self, orientation: np.ndarray, freq_map: np.ndarray, mask_bool: np.ndarray
    ) -> np.ndarray:
        """把每个像素映射到 Gabor 核库索引（区域外为 0，无意义）。"""
        h, w = orientation.shape
        kmap = np.zeros((h, w), dtype=np.int32)
        if not mask_bool.any():
            return kmap
        ang = np.mod(orientation[mask_bool].astype(np.float32), np.pi)
        d_idx = np.clip(
            np.floor(ang / np.pi * self.n_directions).astype(np.int32), 0, self.n_directions - 1
        )
        span = max(self.max_freq - self.min_freq, 1e-6)
        fm = freq_map[mask_bool].astype(np.float32)
        f_idx = np.clip(
            np.floor((fm - self.min_freq) / span * self.n_frequencies).astype(np.int32),
            0,
            self.n_frequencies - 1,
        )
        kmap[mask_bool] = d_idx * self.n_frequencies + f_idx
        return kmap

    def _to_grayscale(
        self,
        pattern: np.ndarray,
        mask_bool: np.ndarray,
        rng: np.random.Generator,
        h: int,
        w: int,
    ) -> np.ndarray:
        """二值图案 → 带皮肤纹理与噪声的灰度图。

        映射分三步，**顺序很关键**：

        1. 图案轻度模糊后线性映射到 ``[ridge_mid, valley_mid]``；
        2. 叠加多尺度皮肤纹理与高斯底噪（这些是"采集噪声"，属于手指/设备的一部分）；
        3. 才在手指区域内做 1%–99% 分位拉伸，把对比度**拉满到配置的灰度范围**。

        第 3 步必须在最后：如果先拉伸再加噪声，噪声会把已经拉开的动态范围再次压扁，
        实测会把脊线对比度从 ~160 灰度级压到 ~35，指纹看起来一片灰。

        ``ridge_only_black`` 打开时走另一条分支：直接输出**纯二值图**
        （脊线 0、谷线 255），既不叠纹理也不加噪、不做分位拉伸。
        这是给分割/细化/细节点算法准备的最干净输入，也是很多论文里
        "二值脊线图"的形态。

        注意这个分支**不做边缘羽化**：羽化会在指边制造 0~255 之间的过渡灰，
        二值图就名不副实了。二值图本来就要硬边。
        """
        if self._ridge_only_black:
            # 图案是"粗网格迭代 + 双三次放大"来的，边界带一小段 0~1 的斜坡。
            # 直接取 ``> 0`` 会把斜坡全算成脊线，脊线明显变胖、谷线只剩细缝；
            # 先轻微平滑再以 0.5 为界，脊/谷宽度才接近各占一半。
            soft = cv2.GaussianBlur(pattern.astype(np.float32), (3, 3), 0.8)
            return np.where(soft >= 0.5, 0, 255).astype(np.uint8)

        rc_mid = (self._ridge_color[0] + self._ridge_color[1]) / 2.0
        vc_mid = (self._valley_color[0] + self._valley_color[1]) / 2.0

        # --- 1. 图案 → 灰度（用稍大的核把二值边缘化成脊线剖面） ---
        soft = cv2.GaussianBlur(pattern.astype(np.float32), (5, 5), 1.0)
        img = soft * (vc_mid - rc_mid) + rc_mid

        # --- 2. 皮肤纹理与底噪 ---
        fine = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (5, 5), 1.2) * 0.18
        mh, mw = max(1, h // 3), max(1, w // 3)
        med = (
            cv2.resize(
                cv2.GaussianBlur(rng.normal(0, 1, (mh, mw)).astype(np.float32), (7, 7), 2.0),
                (w, h),
            )
            * 0.12
        )
        ch, cw = max(1, h // 6), max(1, w // 6)
        coarse = (
            cv2.resize(
                cv2.GaussianBlur(rng.normal(0, 1, (ch, cw)).astype(np.float32), (15, 15), 4.0),
                (w, h),
            )
            * 0.08
        )
        texture = fine + med + coarse
        t_min, t_max = float(texture.min()), float(texture.max())
        if t_max - t_min > 1e-8:
            texture = (texture - t_min) / (t_max - t_min) * 2.0 - 1.0
        img = img + texture * self._texture_strength
        if self._noise_sigma > 0:
            img = img + rng.normal(0, self._noise_sigma, (h, w))

        # --- 3. 区域内分位拉伸，把动态范围铺满配置的灰度区间 ---
        if mask_bool.any():
            vals = img[mask_bool]
            lo, hi = np.percentile(vals, [1.0, 99.0])
            if hi - lo > 5.0:
                # 目标区间略微外扩，使 1%/99% 分位正好落在灰度上下界上
                out_lo = max(rc_mid - (vc_mid - rc_mid) * 0.08, 0.0)
                out_hi = min(vc_mid + (vc_mid - rc_mid) * 0.08, 255.0)
                img = img.copy()
                img[mask_bool] = (vals - lo) / (hi - lo) * (out_hi - out_lo) + out_lo
        return np.clip(img, 0, 255).astype(np.uint8)

    def _feather_edges(self, gray: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        """手指边缘羽化过渡到背景（避免硬边）。"""
        fw = self._mask_feather_width
        if fw <= 0:
            return gray
        mask_u8 = mask_bool.astype(np.uint8) * 255
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        feather = np.clip(dist / float(fw), 0.0, 1.0).astype(np.float32)
        ks = fw * 2 + 1
        feather = cv2.GaussianBlur(feather, (ks, ks), fw / 3.0)
        out = gray.astype(np.float32) * feather + 255.0 * (1.0 - feather)
        return np.clip(out, 0, 255).astype(np.uint8)

    # -- 诊断 -------------------------------------------------------------
    def get_debug_steps(self) -> List[np.ndarray]:
        """返回迭代过程中的中间图案（GUI 分步预览用）。"""
        return self._debug_steps
