"""脊线边缘柔化 — 模拟传感器光学散射导致脊线边界灰度渐变"""
import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry

@DegradationRegistry.register("ridge_edge_roughness")
class RidgeEdgeRoughnessOperator(DegradationOperator):
    """脊线边缘柔化: 模拟传感器光学散射导致脊线边界灰度渐变"""
    @property
    def name(self) -> str:
        return "ridge_edge_roughness"

    def sample_params(self, rng: np.random.Generator) -> dict:
        roughness_range = self.config.get("roughness", [0.05, 0.35])
        return {"roughness": round(rng.uniform(roughness_range[0], roughness_range[1]), 3)}

    def _find_ridge_edges(self, image):
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        grad_x = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_threshold = np.percentile(grad_mag, 85)
        edge_zone = (grad_mag > edge_threshold).astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        edge_zone = cv2.dilate(edge_zone, kernel, iterations=1)
        return cv2.GaussianBlur(edge_zone, (3, 3), 0.6)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(operator_name=self.name, params=params)
        roughness = params["roughness"]
        edge_zone = self._find_ridge_edges(image)
        blur_sigma = 0.3 + roughness * 1.2
        blur_ksize = max(3, min(int(blur_sigma * 3), 5))
        if blur_ksize % 2 == 0: blur_ksize += 1
        blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), blur_sigma)
        blend_strength = roughness * 0.7
        blend = np.clip((edge_zone * blend_strength).astype(np.float32), 0, 0.5)
        result = image * (1.0 - blend) + blurred * blend
        return np.clip(result, 0, 1).astype(np.float32)
