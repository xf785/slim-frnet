"""残影 — 模拟采集设备上之前按压留下的残留指纹影像叠加"""
import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry

@DegradationRegistry.register("ghosting")
class GhostingOperator(DegradationOperator):
    """残影: 模拟采集设备玻璃面上前一次按压留下的残留指纹影像"""
    @property
    def name(self) -> str:
        return "ghosting"

    def sample_params(self, rng: np.random.Generator) -> dict:
        num_range = self.config.get("num_ghosts", [1, 3])
        offset_range = self.config.get("offset_px", [15, 60])
        opacity_range = self.config.get("opacity", [0.08, 0.30])
        angle_range = self.config.get("angle_deg", [-15, 15])
        blur_range = self.config.get("blur_sigma", [0.5, 2.5])
        num = int(rng.integers(num_range[0], num_range[1] + 1))
        dx = int(rng.integers(offset_range[0], offset_range[1] + 1)) * rng.choice([-1, 1])
        dy = int(rng.integers(offset_range[0], offset_range[1] + 1)) * rng.choice([-1, 1])
        opacity = rng.uniform(opacity_range[0], opacity_range[1])
        angle = rng.uniform(angle_range[0], angle_range[1])
        blur_sigma = rng.uniform(blur_range[0], blur_range[1])
        return {"num_ghosts": num, "offset_dx": dx, "offset_dy": dy,
                "opacity": round(opacity, 3), "angle_deg": round(angle, 1),
                "blur_sigma": round(blur_sigma, 2)}

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(operator_name=self.name, params=params)
        h, w = image.shape
        num, opacity, angle, blur_sigma = params["num_ghosts"], params["opacity"], params["angle_deg"], params["blur_sigma"]
        result = image.copy().astype(np.float32)
        for i in range(num):
            layer_opacity = opacity * (0.7 ** i)
            layer_dx = params["offset_dx"] + rng.integers(-8, 9)
            layer_dy = params["offset_dy"] + rng.integers(-8, 9)
            layer_angle = angle + rng.uniform(-5, 5)
            layer_blur = max(0.3, blur_sigma - i * 0.3)
            center = (w / 2, h / 2)
            rot_mat = cv2.getRotationMatrix2D(center, layer_angle, 1.0)
            rot_mat[0, 2] += layer_dx
            rot_mat[1, 2] += layer_dy
            ghost = cv2.warpAffine(image, rot_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.5)
            if layer_blur > 0.3:
                ks = int(max(3, layer_blur * 4))
                if ks % 2 == 0: ks += 1
                ghost = cv2.GaussianBlur(ghost, (ks, ks), layer_blur)
            mean_val = ghost.mean()
            ghost = ghost * 0.7 + mean_val * 0.3
            result = result * (1.0 - layer_opacity) + ghost * layer_opacity
        return np.clip(result, 0, 1).astype(np.float32)
