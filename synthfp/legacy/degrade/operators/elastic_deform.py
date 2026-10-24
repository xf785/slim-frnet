"""弹性变形 — 模拟手指按压皮肤变形"""

import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry


@DegradationRegistry.register("elastic_deformation")
class ElasticDeformationOperator(DegradationOperator):
    """弹性变形: 模拟手指按压时的皮肤变形"""

    @property
    def name(self) -> str:
        return "elastic_deformation"

    def sample_params(self, rng: np.random.Generator) -> dict:
        alpha_range = self.config.get("alpha", [10, 50])
        sigma_range = self.config.get("sigma", [3, 6])
        alpha = rng.uniform(alpha_range[0], alpha_range[1])
        sigma = rng.uniform(sigma_range[0], sigma_range[1])
        return {"alpha": round(alpha, 1), "sigma": round(sigma, 1)}

    def apply(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(
            operator_name=self.name, params=params
        )

        alpha = params["alpha"]
        sigma = params["sigma"]
        h, w = image.shape

        # 生成随机位移场
        dx = rng.uniform(-1, 1, (h, w)).astype(np.float32)
        dy = rng.uniform(-1, 1, (h, w)).astype(np.float32)

        # 高斯平滑位移场 (控制变形的空间连续性)
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        # 构建映射网格并重映射
        x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        deformed = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return np.clip(deformed, 0, 1).astype(np.float32)
