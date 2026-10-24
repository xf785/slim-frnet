"""传感器模式噪声 — 模拟采集设备固定模式噪声和像素级缺陷"""
import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry

@DegradationRegistry.register("sensor_noise")
class SensorNoiseOperator(DegradationOperator):
    """传感器噪声: 模拟CMOS/CCD固定模式噪声、坏点、行噪声和热噪声"""
    @property
    def name(self) -> str:
        return "sensor_noise"

    def sample_params(self, rng: np.random.Generator) -> dict:
        return {
            "fpn_std": round(rng.uniform(*self.config.get("fpn_std", [0.005, 0.03])), 4),
            "dead_pixel_ratio": round(rng.uniform(*self.config.get("dead_pixel_ratio", [0.0, 0.002])), 4),
            "row_noise_std": round(rng.uniform(*self.config.get("row_noise_std", [0.0, 0.02])), 4),
            "thermal_std": round(rng.uniform(*self.config.get("thermal_std", [0.005, 0.02])), 4),
        }

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(operator_name=self.name, params=params)
        h, w = image.shape
        result = image.copy().astype(np.float32)
        # FPN
        fpn_std = params["fpn_std"]
        if fpn_std > 0:
            fpn_h, fpn_w = max(1, h // 8), max(1, w // 8)
            fpn_lowres = rng.normal(1.0, fpn_std, (fpn_h, fpn_w)).astype(np.float32)
            fpn = cv2.GaussianBlur(cv2.resize(fpn_lowres, (w, h), interpolation=cv2.INTER_CUBIC), (5, 5), 2.0)
            active = (result > 0.05) & (result < 0.95)
            result[active] = result[active] * fpn[active]
        # Dead pixels
        dead_ratio = params["dead_pixel_ratio"]
        if dead_ratio > 0:
            n_dead = int(h * w * dead_ratio)
            if n_dead > 0:
                dead_y = rng.integers(0, h, n_dead)
                dead_x = rng.integers(0, w, n_dead)
                for dy, dx in zip(dead_y, dead_x):
                    result[dy, dx] = 0.0 if rng.random() < 0.5 else 1.0
                    y1, y2 = max(0, dy - 2), min(h, dy + 3)
                    x1, x2 = max(0, dx - 2), min(w, dx + 3)
                    neighbors = result[y1:y2, x1:x2]
                    if neighbors.size > 4: result[dy, dx] = np.median(neighbors)
        # Row noise
        row_noise_std = params["row_noise_std"]
        if row_noise_std > 0:
            row_gain = cv2.GaussianBlur(rng.normal(1.0, row_noise_std, h).astype(np.float32).reshape(-1, 1), (5, 1), 1.5).flatten()
            for i in range(h):
                active_row = (result[i, :] > 0.05) & (result[i, :] < 0.95)
                result[i, active_row] = result[i, active_row] * row_gain[i]
        # Thermal
        thermal_std = params["thermal_std"]
        if thermal_std > 0:
            thermal = rng.normal(0, thermal_std, (h, w)).astype(np.float32)
            dark_areas = (result < 0.4).astype(np.float32)
            result += thermal * (1.0 + dark_areas * 2.0)
        return np.clip(result, 0, 1).astype(np.float32)
