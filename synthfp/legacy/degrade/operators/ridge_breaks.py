"""脊线断裂 — 模拟干燥手指、低压力按压导致的脊线不连续/缺口"""
import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry

@DegradationRegistry.register("ridge_breaks")
class RidgeBreaksOperator(DegradationOperator):
    """脊线断裂: 模拟干燥手指或低压力按压导致的脊线微小断裂"""
    @property
    def name(self) -> str:
        return "ridge_breaks"

    def sample_params(self, rng: np.random.Generator) -> dict:
        density_range = self.config.get("density", [0.005, 0.04])
        break_width_range = self.config.get("break_width", [1, 4])
        types = self.config.get("break_types", ["point", "line"])
        density = rng.uniform(density_range[0], density_range[1])
        break_width = int(rng.integers(break_width_range[0], break_width_range[1] + 1))
        break_type = str(rng.choice(types))
        return {"density": round(density, 4), "break_width": break_width, "break_type": break_type}

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(operator_name=self.name, params=params)
        density, break_width, break_type = params["density"], params["break_width"], params["break_type"]
        h, w = image.shape
        result = image.copy()
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        finger_area = binary > 0
        if not finger_area.any(): return result
        n_breaks = int(np.count_nonzero(finger_area) * density)
        if break_type == "point":
            for _ in range(n_breaks):
                cx = rng.integers(0, w); cy = rng.integers(0, h)
                radius = rng.uniform(0.5, float(break_width))
                y1, y2 = max(0, cy - break_width - 1), min(h, cy + break_width + 1)
                x1, x2 = max(0, cx - break_width - 1), min(w, cx + break_width + 1)
                yy, xx = np.ogrid[y1:y2, x1:x2]
                dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                hole = dist <= radius
                region = result[y1:y2, x1:x2][hole]
                if region.size > 0 and region.mean() < 0.6:
                    result[y1:y2, x1:x2][hole] = region * 0.2 + 0.65 * 0.8
        elif break_type == "line":
            for _ in range(n_breaks):
                cx = rng.integers(0, w); cy = rng.integers(0, h)
                if not (0 <= cy < h and 0 <= cx < w): continue
                theta = rng.uniform(0, np.pi)
                line_length = rng.integers(2, break_width + 3)
                for t in range(-line_length, line_length + 1):
                    px = int(cx + t * np.cos(theta))
                    py = int(cy + t * np.sin(theta))
                    if 0 <= px < w and 0 <= py < h:
                        val = result[py, px]
                        if val < 0.55:
                            result[py, px] = val * 0.3 + 0.6 * 0.7
        return np.clip(result, 0, 1).astype(np.float32)
