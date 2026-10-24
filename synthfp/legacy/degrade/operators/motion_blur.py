"""运动模糊 — 模拟手指滑动/运动模糊"""

import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry


@DegradationRegistry.register("motion_blur")
class MotionBlurOperator(DegradationOperator):
    """运动模糊: 模拟手指滑动"""

    @property
    def name(self) -> str:
        return "motion_blur"

    def sample_params(self, rng: np.random.Generator) -> dict:
        kernel_sizes = self.config.get("kernel_sizes", [5, 7, 9, 11])
        kernel_size = int(rng.choice(kernel_sizes))
        angle = rng.uniform(0, 180)
        return {"kernel_size": kernel_size, "angle": round(angle, 1)}

    def apply(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(
            operator_name=self.name, params=params
        )

        kernel_size = params["kernel_size"]
        angle = params["angle"]

        # 创建运动模糊核
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        rad = np.deg2rad(angle)

        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * np.cos(rad))
            y = int(center + offset * np.sin(rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0

        kernel /= kernel.sum()

        degraded = cv2.filter2D(image, -1, kernel)
        return np.clip(degraded, 0, 1).astype(np.float32)
