"""散焦模糊 — 支持均匀模糊和径向焦点外模糊"""
import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry

@DegradationRegistry.register("defocus_blur")
class DefocusBlurOperator(DegradationOperator):
    """散焦模糊: 模拟相机镜头失焦或光圈衍射导致的盘状模糊, 支持uniform和radial模式"""
    @property
    def name(self) -> str:
        return "defocus_blur"

    def sample_params(self, rng: np.random.Generator) -> dict:
        radius_range = self.config.get("radius", [3, 15])
        blades = int(self.config.get("aperture_blades", 0))
        focal_mode = self.config.get("focal_mode", "uniform")
        focal_x = self.config.get("focal_x", 0.5)
        focal_y = self.config.get("focal_y", 0.5)
        radius = rng.uniform(radius_range[0], radius_range[1])
        return {"radius": round(radius, 1), "aperture_blades": blades,
                "focal_mode": focal_mode, "focal_x": round(focal_x, 2), "focal_y": round(focal_y, 2)}

    def _make_disk_kernel(self, radius, blades=0):
        r_int = int(np.ceil(radius))
        size = 2 * r_int + 1
        yy, xx = np.ogrid[-r_int:r_int + 1, -r_int:r_int + 1]
        dist = np.sqrt(xx ** 2 + yy ** 2)
        if blades >= 3:
            angle = np.arctan2(yy, xx)
            modulation = 1.0 + 0.15 * np.cos(blades * angle)
            kernel = (dist <= radius * modulation).astype(np.float32)
        else:
            kernel = (dist <= radius).astype(np.float32)
        edge_half = 0.5
        inner = (dist <= radius - edge_half).astype(np.float32)
        outer = (dist <= radius + edge_half).astype(np.float32)
        edge = outer - inner
        if edge.any():
            fade = np.clip((radius + edge_half - dist) / (2 * edge_half), 0, 1)
            kernel = inner + edge * fade
        total = kernel.sum()
        if total > 0: kernel /= total
        return kernel.astype(np.float32)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(operator_name=self.name, params=params)
        radius, blades = params["radius"], params["aperture_blades"]
        focal_mode = params.get("focal_mode", "uniform")
        if focal_mode == "radial":
            return self._apply_radial(image, radius, blades, params)
        kernel = self._make_disk_kernel(radius, blades)
        return np.clip(cv2.filter2D(image, -1, kernel), 0, 1).astype(np.float32)

    def _apply_radial(self, image, max_radius, blades, params):
        h, w = image.shape
        cx = int(params.get("focal_x", 0.5) * w)
        cy = int(params.get("focal_y", 0.5) * h)
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
        max_dist = np.sqrt(max(cx, w - cx) ** 2 / w ** 2 + max(cy, h - cy) ** 2 / h ** 2)
        dist_norm = np.clip(dist / max(max_dist, 0.01), 0, 1).astype(np.float32)
        local_radius = 0.5 + dist_norm * (max_radius - 0.5)
        n_levels = max(3, int(max_radius))
        level_bins = np.linspace(0, 1, n_levels + 1)
        result = np.zeros_like(image)
        weight_sum = np.zeros_like(image)
        for i in range(n_levels):
            r_low = 0.5 + level_bins[i] * (max_radius - 0.5)
            r_high = 0.5 + level_bins[i + 1] * (max_radius - 0.5)
            r_mid = (r_low + r_high) / 2.0
            level_weight = np.clip(1.0 - np.abs(local_radius - r_mid) / max(r_high - r_low, 0.01), 0, 1).astype(np.float32)
            if level_weight.max() < 0.01: continue
            kernel = self._make_disk_kernel(r_mid, blades)
            blurred = cv2.filter2D(image, -1, kernel)
            result += blurred * level_weight
            weight_sum += level_weight
        mask = weight_sum > 0.01
        result[mask] /= weight_sum[mask]
        result[~mask] = image[~mask]
        return np.clip(result, 0, 1).astype(np.float32)
