"""墨迹污渍 — 模拟油墨涂抹/污渍"""

import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry


@DegradationRegistry.register("ink_smudge")
class InkSmudgeOperator(DegradationOperator):
    """墨迹污渍: 模拟油墨涂抹、污渍等局部退化"""

    @property
    def name(self) -> str:
        return "ink_smudge"

    def sample_params(self, rng: np.random.Generator) -> dict:
        intensity_range = self.config.get("intensity", [0.1, 0.4])
        intensity = rng.uniform(intensity_range[0], intensity_range[1])
        return {"intensity": round(intensity, 2)}

    def apply(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(
            operator_name=self.name, params=params
        )

        intensity = params["intensity"]
        h, w = image.shape

        # 生成随机污渍区域 (高斯混合)
        num_smudges = rng.integers(1, 5)
        smudge_mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(num_smudges):
            cx = rng.integers(0, w)
            cy = rng.integers(0, h)
            radius = rng.integers(20, 80)

            yy, xx = np.ogrid[:h, :w]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

            # 柔和的径向渐变污渍
            smudge = np.exp(-(dist**2) / (2 * (radius / 3) ** 2))
            smudge = smudge.astype(np.float32)
            smudge_mask = np.maximum(smudge_mask, smudge)

        # 污渍区域: 降低对比度 + 加暗 + 局部模糊
        smudge_mask = smudge_mask * intensity
        blurred = cv2.GaussianBlur(image, (15, 15), 5)
        result = image * (1 - smudge_mask) + blurred * smudge_mask
        result = result * (1 - smudge_mask * 0.5)  # 稍暗

        return np.clip(result, 0, 1).astype(np.float32)
