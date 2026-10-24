"""高斯模糊 — 支持均匀模糊和径向焦点外模糊"""
import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry

@DegradationRegistry.register("gaussian_blur")
class GaussianBlurOperator(DegradationOperator):
    """高斯模糊: 模拟光学散焦, 支持uniform和radial模式"""
    @property
    def name(self) -> str:
        return "gaussian_blur"

    def sample_params(self, rng: np.random.Generator) -> dict:
        sigma_range = self.config.get("sigma", [0.5, 3.0])
        kernel_sizes = self.config.get("kernel_size", [3, 5, 7])
        sigma = rng.uniform(sigma_range[0], sigma_range[1])
        kernel = int(rng.choice(kernel_sizes))
        return {"sigma": round(sigma, 2), "kernel": kernel,
                "focal_mode": self.config.get("focal_mode", "uniform"),
                "focal_x": round(self.config.get("focal_x", 0.5), 2),
                "focal_y": round(self.config.get("focal_y", 0.5), 2)}

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(operator_name=self.name, params=params)
        kernel, sigma = params["kernel"], params["sigma"]
        if params.get("focal_mode") == "radial":
            return self._apply_radial(image, kernel, sigma, params)
        return np.clip(cv2.GaussianBlur(image, (kernel, kernel), sigma), 0, 1).astype(np.float32)

    def _apply_radial(self, image, kernel, max_sigma, params):
        h, w = image.shape
        cx = int(params.get("focal_x", 0.5) * w)
        cy = int(params.get("focal_y", 0.5) * h)
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt(((xx - cx).astype(np.float32) / w) ** 2 + ((yy - cy).astype(np.float32) / h) ** 2)
        max_dist = np.sqrt(max(cx, w - cx) ** 2 / w ** 2 + max(cy, h - cy) ** 2 / h ** 2)
        dist_norm = np.clip(dist / max(max_dist, 0.01), 0, 1).astype(np.float32)
        blur_weight = np.clip((dist_norm - 0.12) / 0.88, 0, 1)
        blur_weight = cv2.GaussianBlur(blur_weight, (31, 31), 10)
        local_sigma = 0.3 + blur_weight * (max_sigma - 0.3)
        n_levels = min(8, max(3, int(max_sigma * 2)))
        result, weight_sum = np.zeros_like(image), np.zeros_like(image)
        for i in range(n_levels - 1):
            s_low = 0.3 + i * max_sigma / n_levels
            s_high = 0.3 + (i + 1) * max_sigma / n_levels
            s_mid = (s_low + s_high) / 2
            lw = np.clip(1.0 - np.abs(local_sigma - s_mid) / max(s_high - s_low, 0.01), 0, 1).astype(np.float32)
            if lw.max() < 0.01: continue
            ks = max(3, min(int(s_mid * 4 + 1), kernel))
            if ks % 2 == 0: ks += 1
            blurred = cv2.GaussianBlur(image, (ks, ks), s_mid)
            result += blurred * lw; weight_sum += lw
        valid = weight_sum > 0.01
        result[valid] /= weight_sum[valid]
        result[~valid] = image[~valid]
        return np.clip(result, 0, 1).astype(np.float32)
