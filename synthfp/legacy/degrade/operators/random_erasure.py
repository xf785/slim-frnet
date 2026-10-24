"""随机遮挡 — 模拟局部缺失/疤痕/污垢"""

import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry


@DegradationRegistry.register("random_erasure")
class RandomErasureOperator(DegradationOperator):
    """随机遮挡: 模拟局部缺失、疤痕或污垢覆盖"""

    @property
    def name(self) -> str:
        return "random_erasure"

    def sample_params(self, rng: np.random.Generator) -> dict:
        p = self.config.get("p", 0.3)
        area_range = self.config.get("area_range", [0.02, 0.10])
        area_ratio = rng.uniform(area_range[0], area_range[1])
        return {"p": p, "area_ratio": round(area_ratio, 4)}

    def apply(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(
            operator_name=self.name, params=params
        )

        p = params["p"]
        area_ratio = params["area_ratio"]
        h, w = image.shape

        # p 表示施加遮挡的概率 (修正语义: p=0.3 表示30%概率施加遮挡)
        if rng.random() >= p:
            return image.copy()

        # 生成随机遮挡区域
        mask = np.ones((h, w), dtype=np.float32)
        num_erasures = rng.integers(1, 5)
        total_pixels = h * w
        # 将area_ratio分配给多个遮挡区域, 每个区域面积 = 总面积 / num_erasures
        per_erasure_ratio = area_ratio / num_erasures

        for _ in range(num_erasures):
            cx = rng.integers(0, w)
            cy = rng.integers(0, h)
            radius = int(np.sqrt(total_pixels * per_erasure_ratio / np.pi))
            radius = max(5, min(radius, min(w, h) // 3))

            # 椭圆形遮挡
            rw = rng.integers(radius // 2, radius)
            rh = rng.integers(radius // 2, radius)
            angle = rng.uniform(0, 180)

            yy, xx = np.ogrid[:h, :w]
            x_rot = (xx - cx) * np.cos(np.deg2rad(angle)) + (yy - cy) * np.sin(np.deg2rad(angle))
            y_rot = -(xx - cx) * np.sin(np.deg2rad(angle)) + (yy - cy) * np.cos(np.deg2rad(angle))
            ellipse = (x_rot / rw) ** 2 + (y_rot / rh) ** 2 <= 1
            mask[ellipse] = 0

        # 遮挡区域填充噪声
        fill_value = rng.uniform(0.1, 0.3)
        noise = rng.normal(fill_value, 0.05, (h, w)).astype(np.float32)

        degraded = image * mask + noise * (1 - mask)
        return np.clip(degraded, 0, 1).astype(np.float32)
