"""淡化羽化 — 支持边缘淡化和径向焦点外淡化"""
import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry

@DegradationRegistry.register("edge_feathering")
class EdgeFeatheringOperator(DegradationOperator):
    """淡化羽化: 模拟手指边缘按压不均匀或焦点外淡化, 支持edge和radial模式"""
    @property
    def name(self) -> str:
        return "edge_feathering"

    def sample_params(self, rng: np.random.Generator) -> dict:
        feather_width_range = self.config.get("feather_width", [10, 60])
        intensity_range = self.config.get("intensity", [0.2, 0.8])
        feather_width = rng.uniform(feather_width_range[0], feather_width_range[1])
        intensity = rng.uniform(intensity_range[0], intensity_range[1])
        asymmetric_flag = rng.random() < self.config.get("asymmetric_prob", 0.6)
        bias_angle = rng.uniform(*self.config.get("direction_bias", [0, 360]))
        feather_mode = self.config.get("feather_mode", "edge")
        return {"feather_width_px": round(feather_width, 1), "intensity": round(intensity, 3),
                "asymmetric": asymmetric_flag, "bias_angle_deg": round(bias_angle, 1),
                "feather_mode": feather_mode,
                "focal_x": round(self.config.get("focal_x", 0.5), 2),
                "focal_y": round(self.config.get("focal_y", 0.5), 2)}

    def _estimate_fingerprint_mask(self, image):
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if img_uint8.std() < 10: return np.ones(image.shape, dtype=np.float32)
        _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            binary = (labels == largest_label).astype(np.uint8) * 255
        return binary.astype(np.float32) / 255.0

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(operator_name=self.name, params=params)
        h, w = image.shape
        if params.get("feather_mode") == "radial":
            return self._apply_radial(image, params, h, w)
        return self._apply_edge(image, params, h, w)

    def _apply_edge(self, image, params, h, w):
        fw, intensity = params["feather_width_px"], params["intensity"]
        asymmetric, bias_angle = params["asymmetric"], params["bias_angle_deg"]
        finger_mask = self._estimate_fingerprint_mask(image)
        mask_uint8 = (finger_mask * 255).astype(np.uint8)
        dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        feather_weight = np.clip(1.0 - dist / max(fw, 1.0), 0, 1).astype(np.float32)
        if asymmetric:
            yy, xx = np.ogrid[:h, :w]
            angle_map = np.rad2deg(np.arctan2(yy - h / 2, xx - w / 2))
            angle_diff = np.abs((angle_map - bias_angle + 180) % 360 - 180)
            direction_mod = 0.5 + 0.5 * np.cos(np.deg2rad(angle_diff))
            feather_weight = feather_weight * direction_mod
        feather_weight = cv2.GaussianBlur(feather_weight, (15, 15), fw / 4)
        bg_ref = np.percentile(image[finger_mask > 0.01], 10) if finger_mask.sum() > 100 else image.mean()
        mean_img = np.full_like(image, image.mean())
        reduced = image * (1.0 - intensity * feather_weight) + mean_img * intensity * feather_weight
        bg_comp = np.full_like(image, bg_ref)
        degraded = reduced * (1.0 - intensity * 0.5 * feather_weight) + bg_comp * intensity * 0.5 * feather_weight
        if intensity > 0.3:
            lb = cv2.GaussianBlur(degraded, (5, 5), 1.5)
            bm = (feather_weight * intensity * 0.5).astype(np.float32)
            degraded = degraded * (1.0 - bm) + lb * bm
        return np.clip(degraded, 0, 1).astype(np.float32)

    def _apply_radial(self, image, params, h, w):
        intensity, fw = params["intensity"], params["feather_width_px"]
        cx = int(params.get("focal_x", 0.5) * w)
        cy = int(params.get("focal_y", 0.5) * h)
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt(((xx - cx).astype(np.float32) / w) ** 2 + ((yy - cy).astype(np.float32) / h) ** 2)
        max_dist = np.sqrt(max(cx, w - cx) ** 2 / w ** 2 + max(cy, h - cy) ** 2 / h ** 2)
        dist_norm = np.clip(dist / max(max_dist, 0.01), 0, 1).astype(np.float32)
        transition_start = 0.15
        radial_weight = np.clip((dist_norm - transition_start) / max(1.0 - transition_start, 0.01), 0, 1)
        radial_weight = cv2.GaussianBlur(radial_weight, (21, 21), fw / 5)
        feather_effect = np.clip(radial_weight * intensity, 0, 1).astype(np.float32)
        bg_val = np.percentile(image, 15)
        bg_component = np.full_like(image, bg_val)
        degraded = image * (1.0 - feather_effect) + bg_component * feather_effect
        return np.clip(degraded, 0, 1).astype(np.float32)
