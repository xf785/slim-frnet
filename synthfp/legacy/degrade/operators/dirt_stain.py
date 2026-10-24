"""污渍 — 模拟采集设备表面灰尘/油污等不规则污渍"""
import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry

@DegradationRegistry.register("dirt_stain")
class DirtStainOperator(DegradationOperator):
    """污渍: 模拟采集设备玻璃表面或手指上的灰尘、油污等不规则污渍"""
    @property
    def name(self) -> str:
        return "dirt_stain"

    def sample_params(self, rng: np.random.Generator) -> dict:
        num_range = self.config.get("num_stains", [2, 8])
        size_range = self.config.get("size_range", [15, 80])
        opacity_range = self.config.get("opacity", [0.15, 0.55])
        darkness_range = self.config.get("darkness", [0.05, 0.30])
        num = int(rng.integers(num_range[0], num_range[1] + 1))
        opacity = rng.uniform(opacity_range[0], opacity_range[1])
        darkness = rng.uniform(darkness_range[0], darkness_range[1])
        return {"num_stains": num, "min_size": size_range[0], "max_size": size_range[1],
                "opacity": round(opacity, 3), "darkness": round(darkness, 3)}

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(operator_name=self.name, params=params)
        h, w = image.shape
        num, min_sz, max_sz = params["num_stains"], params["min_size"], params["max_size"]
        opacity, darkness = params["opacity"], params["darkness"]
        stain_mask = np.zeros((h, w), dtype=np.float32)
        for _ in range(num):
            cx, cy = rng.integers(0, w), rng.integers(0, h)
            rx = rng.integers(min_sz, max_sz + 1)
            ry = rng.integers(min_sz, max_sz + 1)
            angle = rng.uniform(0, 180)
            yy, xx = np.ogrid[:h, :w]
            x_rot = (xx - cx) * np.cos(np.deg2rad(angle)) + (yy - cy) * np.sin(np.deg2rad(angle))
            y_rot = -(xx - cx) * np.sin(np.deg2rad(angle)) + (yy - cy) * np.cos(np.deg2rad(angle))
            ellipse = ((x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1).astype(np.float32)
            dist = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
            edge_zone = (dist >= 0.5) & (dist <= 1.5)
            perturb = rng.uniform(0.7, 1.3, (max(1, 2 * ry), max(1, 2 * rx)))
            perturb = cv2.resize(perturb, (w, h))
            stain = ellipse.copy()
            stain[edge_zone] = (perturb[edge_zone] > dist[edge_zone]).astype(np.float32)
            fade_mask = (dist >= 0.8) & (dist <= 1.2)
            if fade_mask.any():
                fade = np.clip((1.2 - dist[fade_mask]) / 0.4, 0, 1)
                stain[fade_mask] = stain[fade_mask] * fade
            stain_mask = np.maximum(stain_mask, stain * opacity)
        stain_mask = np.clip(stain_mask, 0, 0.85)
        degraded = image * (1.0 - stain_mask * darkness / 0.55)
        dirt_texture = rng.normal(0, 0.03, (h, w)).astype(np.float32)
        degraded = degraded + dirt_texture * stain_mask * 0.3
        return np.clip(degraded, 0, 1).astype(np.float32)
