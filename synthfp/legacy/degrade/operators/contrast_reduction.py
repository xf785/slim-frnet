"""对比度降低 — 模拟光照不均/低对比度"""

import numpy as np
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry


@DegradationRegistry.register("contrast_reduction")
class ContrastReductionOperator(DegradationOperator):
    """对比度降低: 模拟光照不均或低对比度采集"""

    @property
    def name(self) -> str:
        return "contrast_reduction"

    def sample_params(self, rng: np.random.Generator) -> dict:
        factor_range = self.config.get("factor", [0.3, 0.7])
        factor = rng.uniform(factor_range[0], factor_range[1])
        return {"factor": round(factor, 2)}

    def apply(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(
            operator_name=self.name, params=params
        )

        factor = params["factor"]
        # 对比度降低: I' = I·factor + mean(I)·(1-factor)
        mean_val = image.mean()
        degraded = image * factor + mean_val * (1 - factor)
        return np.clip(degraded, 0, 1).astype(np.float32)
