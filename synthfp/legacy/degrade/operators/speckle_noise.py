"""散斑噪声 — 模拟相干光源散斑噪声"""

import numpy as np
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry


@DegradationRegistry.register("speckle_noise")
class SpeckleNoiseOperator(DegradationOperator):
    """散斑噪声: 乘性噪声, 常见于激光/相干光成像"""

    @property
    def name(self) -> str:
        return "speckle_noise"

    def sample_params(self, rng: np.random.Generator) -> dict:
        std_range = self.config.get("std", [0.01, 0.05])
        std = rng.uniform(std_range[0], std_range[1])
        return {"std": round(std, 4)}

    def apply(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(
            operator_name=self.name, params=params
        )

        std = params["std"]
        noise = rng.normal(0, std, image.shape).astype(np.float32)
        degraded = image + image * noise  # 乘性噪声
        return np.clip(degraded, 0, 1).astype(np.float32)
