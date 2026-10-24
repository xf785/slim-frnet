"""双三次下采样 — 模拟低分辨率采集"""

import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry


@DegradationRegistry.register("downsample")
class DownsampleOperator(DegradationOperator):
    """双三次下采样 + 上采样回原始尺寸 → 损失高频细节"""

    @property
    def name(self) -> str:
        return "downsample"

    def sample_params(self, rng: np.random.Generator) -> dict:
        scale_factors = self.config.get("scale_factors", [2, 4])
        scale = int(rng.choice(scale_factors))
        return {"scale": scale}

    def apply(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(
            operator_name=self.name, params=params
        )

        scale = params["scale"]
        h, w = image.shape

        # 验证图像尺寸是否足够下采样
        if w // scale < 2 or h // scale < 2:
            # 图像太小无法下采样, 返回原图
            return image.copy().astype(np.float32)

        # 下采样
        small = cv2.resize(
            image, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC
        )
        # 上采样回原始尺寸
        degraded = cv2.resize(
            small, (w, h), interpolation=cv2.INTER_CUBIC
        )
        return np.clip(degraded, 0, 1).astype(np.float32)
