"""
脊线频率图生成器 — 基于多层Perlin噪声 (SFinGe标准算法)

SFinGe原始算法:
1. 生成多层不同分辨率的随机噪声 (3层: 5×5, 6×6, 7×7)
2. 双三次插值 (INTER_CUBIC) 放大至目标尺寸
3. 三层取平均值 (各1/3权重)
4. 频率范围受ridge_width参数约束

改进:
- 使用真正的Perlin噪声 (hash函数 + 余弦插值 + 倍频叠加)
- 奇异点附近频率局部变化
- 高斯平滑 + 频率范围约束

参考文献:
  Perlin, K. "An Image Synthesizer." SIGGRAPH, 1985.
  Cappelli, R. "SFinGe: an Approach to Synthetic Fingerprint Generation." 2004.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class FrequencyMapGenerator:
    """
    脊线频率图生成器 — Perlin噪声多层叠加 (SFinGe标准算法)

    Examples:
        >>> gen = FrequencyMapGenerator(base_frequency=0.125)
        >>> freq_map = gen.generate(h, w, singular_points, rng)
    """

    def __init__(
        self,
        base_frequency: float = 0.125,
        freq_variation: float = 0.03,
        singularity_effect: float = 0.3,
        perlin_octaves: int = 4,
        perlin_persistence: float = 0.5,
        perlin_lacunarity: float = 2.0,
    ):
        """
        Args:
            base_frequency: 基础脊线频率 (ridges/pixel), 500dpi下≈0.125
            freq_variation: 频率变化幅度
            singularity_effect: 奇异点影响强度 [0, 1]
            perlin_octaves: Perlin噪声倍频层数
            perlin_persistence: Perlin幅度衰减率 (典型值0.5)
            perlin_lacunarity: Perlin频率倍增率 (典型值2.0)
        """
        self.base_freq = base_frequency
        self.freq_var = freq_variation
        self.singularity_effect = singularity_effect
        self.perlin_octaves = perlin_octaves
        self.perlin_persistence = perlin_persistence
        self.perlin_lacunarity = perlin_lacunarity

    def generate(
        self,
        h: int,
        w: int,
        singular_points: List[Tuple[float, float]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        生成脊线频率图 F(x, y)

        Args:
            h, w: 图像尺寸
            singular_points: 奇异点坐标列表 [(x, y), ...]
            rng: 随机数生成器

        Returns:
            freq_map: (H, W) float32, 值 ∈ [base_freq*0.5, base_freq*2.0]
        """
        # 1. 生成多层Perlin噪声叠加 (SFinGe方法)
        # 3个不同分辨率的随机种子层
        resolutions = [5, 6, 7]  # SFinGe原始分辨率

        freq_map = np.zeros((h, w), dtype=np.float64)

        for res in resolutions:
            # 在低分辨率上生成Perlin噪声
            perlin_layer = self._generate_perlin_noise(res, res, rng)
            perlin_layer = perlin_layer.astype(np.float64)

            # 范围映射到 [base_freq - var, base_freq + var]
            perlin_layer = self.base_freq + (perlin_layer - 0.5) * 2 * self.freq_var

            # 双三次插值放大到目标尺寸 (SFinGe使用INTER_CUBIC)
            perlin_layer = cv2.resize(
                perlin_layer, (w, h), interpolation=cv2.INTER_CUBIC
            )

            # 累加 (3层各占1/3权重)
            freq_map += perlin_layer / 3.0

        # 2. 低频全局扰动 (额外多尺度细节)
        octave_noise = self._generate_multi_octave_noise(h, w, rng)
        freq_map += octave_noise * self.freq_var * 0.5

        # 3. 奇异点附近频率增加 (奇异点周围脊线更密集)
        if singular_points:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            for sx, sy in singular_points:
                dist = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
                ridge_period = 1.0 / max(self.base_freq, 0.01)
                influence_radius = ridge_period * 3.0
                # 高斯衰减
                effect = np.exp(-(dist**2) / (2 * influence_radius**2))
                freq_map += self.singularity_effect * self.base_freq * effect

        # 4. 轻度高斯平滑 (保持空间连续性)
        ksize = min(15, (min(h, w) // 16) | 1)  # 确保奇数
        if ksize >= 3:
            freq_map = cv2.GaussianBlur(freq_map.astype(np.float32), (ksize, ksize), 3)

        # 5. 频率范围约束
        freq_map = np.clip(
            freq_map,
            self.base_freq * 0.5,
            self.base_freq * 2.0,
        )

        return freq_map.astype(np.float32)

    # ------------------------------------------------------------------
    # Perlin噪声实现 (匹配SFinGe Ve2D/misc.cpp)
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_noise(x: int, y: int, seed: int) -> float:
        """
        哈希噪声函数 — 基于整数坐标的伪随机

        匹配SFinGe的 findnoise2():
          n = (x + y*57)
          n = (n<<13) ^ n
          nn = (n*(n*n*60493 + 19990303) + 1376312589) & 0x7fffffff
          return 1.0 - nn/1073741824.0
        """
        n = (x + y * 57 + seed) & 0xFFFFFFFF
        n = ((n << 13) ^ n) & 0xFFFFFFFF
        # 转为 Python int 避免 numpy 溢出警告
        n_int = int(n)
        nn = (n_int * (n_int * n_int * 60493 + 19990303) + 1376312589) & 0x7FFFFFFF
        return 1.0 - (nn / 1073741824.0)

    @staticmethod
    def _cosine_interpolate(a: float, b: float, t: float) -> float:
        """
        余弦插值 — 匹配SFinGe的 interpolate()

        f = (1 - cos(t*π)) * 0.5
        return a*(1-f) + b*f
        """
        ft = t * np.pi
        f = (1.0 - np.cos(ft)) * 0.5
        return a * (1.0 - f) + b * f

    def _perlin_noise_2d(self, x: float, y: float, seed: int) -> float:
        """
        2D Perlin噪声函数 (标量版本, 保留向后兼容)

        匹配SFinGe的 noise():
        1. 获取整数部分 (floor)
        2. 计算四个角的hash值
        3. 余弦插值 (x方向 → y方向)
        """
        floor_x = int(np.floor(x))
        floor_y = int(np.floor(y))
        frac_x = x - floor_x
        frac_y = y - floor_y

        s = self._hash_noise(floor_x, floor_y, seed)
        t = self._hash_noise(floor_x + 1, floor_y, seed)
        u = self._hash_noise(floor_x, floor_y + 1, seed)
        v = self._hash_noise(floor_x + 1, floor_y + 1, seed)

        int1 = self._cosine_interpolate(s, t, frac_x)
        int2 = self._cosine_interpolate(u, v, frac_x)
        return self._cosine_interpolate(int1, int2, frac_y)

    def _perlin_noise_2d_vectorized(
        self, x_grid: np.ndarray, y_grid: np.ndarray, seed: int
    ) -> np.ndarray:
        """
        向量化2D Perlin噪声 — 同时计算所有像素

        与_perlin_noise_2d功能相同, 但使用numpy向量化操作。
        """
        floor_x = np.floor(x_grid).astype(np.int64)
        floor_y = np.floor(y_grid).astype(np.int64)
        frac_x = x_grid - floor_x
        frac_y = y_grid - floor_y

        # 四个角的hash值 (向量化)
        s = np.vectorize(self._hash_noise)(floor_x, floor_y, seed)
        t = np.vectorize(self._hash_noise)(floor_x + 1, floor_y, seed)
        u = np.vectorize(self._hash_noise)(floor_x, floor_y + 1, seed)
        v = np.vectorize(self._hash_noise)(floor_x + 1, floor_y + 1, seed)

        # x方向余弦插值
        ft_x = frac_x * np.pi
        f_x = (1.0 - np.cos(ft_x)) * 0.5
        int1 = s * (1.0 - f_x) + t * f_x
        int2 = u * (1.0 - f_x) + v * f_x

        # y方向余弦插值
        ft_y = frac_y * np.pi
        f_y = (1.0 - np.cos(ft_y)) * 0.5
        return int1 * (1.0 - f_y) + int2 * f_y

    def _generate_perlin_noise(
        self, h: int, w: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        生成单层Perlin噪声图像 (向量化实现)

        使用倍频叠加 (octaves):
          noise(x,y) = Σ(a=0→octaves-1) noise2d(x·freq_a, y·freq_a) × ampl_a
          其中 freq_a = lacunarity^a, ampl_a = persistence^a
        """
        seed = rng.integers(0, 2**31 - 1)
        zoom = max(h, w) / 2.0

        noise_img = np.zeros((h, w), dtype=np.float64)

        # 创建坐标网格 (向量化, 替代逐像素循环)
        y_indices = np.arange(h, dtype=np.float64).reshape(-1, 1)
        x_indices = np.arange(w, dtype=np.float64).reshape(1, -1)
        x_norm = x_indices / w  # (1, W)
        y_norm = y_indices / h  # (H, 1)

        for octave in range(self.perlin_octaves):
            frequency = self.perlin_lacunarity**octave
            amplitude = self.perlin_persistence**octave

            # 计算所有像素的噪声坐标 (广播)
            nx = x_norm * zoom * frequency  # (1, W) -> broadcasts to (H, W)
            ny = y_norm * zoom * frequency  # (H, 1) -> broadcasts to (H, W)

            noise_img += (
                self._perlin_noise_2d_vectorized(nx, ny, seed + octave) * amplitude
            )

        # 归一化到 [0, 1]
        n_min, n_max = noise_img.min(), noise_img.max()
        if n_max - n_min > 1e-8:
            noise_img = (noise_img - n_min) / (n_max - n_min)

        return noise_img

    def _generate_multi_octave_noise(
        self, h: int, w: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        生成多倍频Perlin噪声 (用于低频全局扰动, 向量化实现)

        匹配SFinGe的 render_clouds():
        在不同空间频率上叠加噪声，模拟自然纹理的分形特征。
        """
        seed = rng.integers(0, 2**31 - 1)
        zoom = max(h, w) / 4.0  # 低频 → 大zoom

        noise = np.zeros((h, w), dtype=np.float64)

        for octave in range(self.perlin_octaves):
            frequency = self.perlin_lacunarity**octave
            amplitude = self.perlin_persistence**octave

            # 降低分辨率以提升性能 (低频不需要全分辨率)
            sub_h = max(1, h // (2**octave))
            sub_w = max(1, w // (2**octave))

            # 向量化计算子分辨率噪声
            y_indices = np.arange(sub_h, dtype=np.float64).reshape(-1, 1)
            x_indices = np.arange(sub_w, dtype=np.float64).reshape(1, -1)
            nx = (x_indices / sub_w) * zoom * frequency
            ny = (y_indices / sub_h) * zoom * frequency
            sub_noise = (
                self._perlin_noise_2d_vectorized(nx, ny, seed + octave) * amplitude
            )

            # 放大到目标尺寸
            if (sub_h, sub_w) != (h, w):
                sub_noise = cv2.resize(
                    sub_noise, (w, h), interpolation=cv2.INTER_LINEAR
                )

            noise += sub_noise

        # 归一化到 [-1, 1] (零中心)
        n_min, n_max = noise.min(), noise.max()
        if n_max - n_min > 1e-8:
            noise = (noise - n_min) / (n_max - n_min) * 2.0 - 1.0

        return noise.astype(np.float64)
