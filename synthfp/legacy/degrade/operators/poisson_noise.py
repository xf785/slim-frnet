"""泊松噪声 — 模拟CMOS/CCD传感器光子噪声"""

import numpy as np
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry


@DegradationRegistry.register("poisson_noise")
class PoissonNoiseOperator(DegradationOperator):
    """泊松噪声: 模拟光子散粒噪声 (λ越小噪声越强)"""

    @property
    def name(self) -> str:
        return "poisson_noise"

    def sample_params(self, rng: np.random.Generator) -> dict:
        lam_range = self.config.get("lam", [5, 30])
        lam = rng.uniform(lam_range[0], lam_range[1])
        return {"lam": round(lam, 1)}

    def apply(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(
            operator_name=self.name, params=params
        )

        lam = params["lam"]
        # 将图像缩放到泊松过程的强度范围
        scaled = image * lam
        noisy = rng.poisson(scaled).astype(np.float32) / lam
        return np.clip(noisy, 0, 1).astype(np.float32)
