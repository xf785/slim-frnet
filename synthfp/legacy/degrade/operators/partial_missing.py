"""部分缺失 — 模拟手指部分区域未接触到采集面或图像损坏"""
import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry

@DegradationRegistry.register("partial_missing")
class PartialMissingOperator(DegradationOperator):
    """部分缺失: 模拟手指部分区域未接触到采集设备, 或图像数据损坏/丢失"""
    @property
    def name(self) -> str:
        return "partial_missing"

    def sample_params(self, rng: np.random.Generator) -> dict:
        area_range = self.config.get("area_ratio", [0.05, 0.35])
        num_range = self.config.get("num_regions", [1, 3])
        area_ratio = rng.uniform(area_range[0], area_range[1])
        num = int(rng.integers(num_range[0], num_range[1] + 1))
        positions = self.config.get("positions", ["edge", "center", "random"])
        position = str(rng.choice(positions))
        fill_types = self.config.get("fill_types", ["background", "noise"])
        fill = str(rng.choice(fill_types))
        return {"area_ratio": round(area_ratio, 3), "num_regions": num, "position": position, "fill_type": fill}

    def _generate_natural_mask(self, h, w, area_ratio, position, rng):
        mask = np.ones((h, w), dtype=np.float32)
        target_pixels = int(h * w * area_ratio)
        if position == "edge":
            edge = rng.choice(["top", "bottom", "left", "right"])
            if edge == "top": cx, cy = rng.integers(w // 4, 3 * w // 4), rng.integers(0, h // 3)
            elif edge == "bottom": cx, cy = rng.integers(w // 4, 3 * w // 4), rng.integers(2 * h // 3, h)
            elif edge == "left": cx, cy = rng.integers(0, w // 3), rng.integers(h // 4, 3 * h // 4)
            else: cx, cy = rng.integers(2 * w // 3, w), rng.integers(h // 4, 3 * h // 4)
        elif position == "center":
            cx, cy = rng.integers(w // 3, 2 * w // 3), rng.integers(h // 3, 2 * h // 3)
        else:
            cx, cy = rng.integers(0, w), rng.integers(0, h)
        est_radius = int(np.sqrt(target_pixels / np.pi) * 1.3)
        rx_base = int(min(est_radius * rng.uniform(0.6, 1.2), w * 0.4))
        ry_base = int(min(est_radius * rng.uniform(0.6, 1.2), h * 0.4))
        n_ellipses = rng.integers(2, 5)
        combined = np.zeros((h, w), dtype=np.float32)
        for i in range(n_ellipses):
            ecx = np.clip(cx + rng.integers(-rx_base // 2, rx_base // 2 + 1), 0, w - 1)
            ecy = np.clip(cy + rng.integers(-ry_base // 2, ry_base // 2 + 1), 0, h - 1)
            erx = max(5, rx_base + rng.integers(-rx_base // 3, rx_base // 3 + 1))
            ery = max(5, ry_base + rng.integers(-ry_base // 3, ry_base // 3 + 1))
            e_angle = rng.uniform(0, 180)
            yy, xx = np.ogrid[:h, :w]
            x_rot = (xx - ecx) * np.cos(np.deg2rad(e_angle)) + (yy - ecy) * np.sin(np.deg2rad(e_angle))
            y_rot = -(xx - ecx) * np.sin(np.deg2rad(e_angle)) + (yy - ecy) * np.cos(np.deg2rad(e_angle))
            dist = np.sqrt((x_rot / erx) ** 2 + (y_rot / ery) ** 2)
            perturb = rng.uniform(0.75, 1.25)
            combined = np.maximum(combined, (dist <= perturb).astype(np.float32))
        combined_uint8 = (combined * 255).astype(np.uint8)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined_uint8 = cv2.morphologyEx(combined_uint8, cv2.MORPH_CLOSE, kernel_close)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_uint8 = cv2.dilate(combined_uint8, kernel_dilate, iterations=1)
        combined_float = combined_uint8.astype(np.float32) / 255.0
        combined_float = cv2.GaussianBlur(combined_float, (15, 15), 5)
        return 1.0 - np.clip(combined_float, 0, 1)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(operator_name=self.name, params=params)
        h, w = image.shape
        area_ratio, num_regions = params["area_ratio"], params["num_regions"]
        position, fill_type = params["position"], params["fill_type"]
        per_region_area = area_ratio / num_regions
        total_missing = np.zeros((h, w), dtype=np.float32)
        for i in range(num_regions):
            pos = position if position != "random" or i == 0 else rng.choice(["edge", "center"])
            region_mask = self._generate_natural_mask(h, w, per_region_area, pos, rng)
            total_missing = np.maximum(total_missing, region_mask)
        total_missing = np.clip(total_missing, 0, 0.95)
        if fill_type == "background":
            ew = max(5, min(h, w) // 20)
            bg_value = np.mean([image[:ew, :].mean(), image[-ew:, :].mean(), image[:, :ew].mean(), image[:, -ew:].mean()])
            fill_image = np.full((h, w), bg_value, dtype=np.float32) + rng.normal(0, 0.01, (h, w)).astype(np.float32)
        else:
            fill_image = cv2.GaussianBlur(rng.uniform(0.1, 0.4, (h, w)).astype(np.float32), (11, 11), 4)
        edge_transition = cv2.GaussianBlur(total_missing, (7, 7), 3)
        edge_transition = np.clip(edge_transition, 0, 1)
        degraded = image * (1.0 - edge_transition) + fill_image * edge_transition
        return np.clip(degraded, 0, 1).astype(np.float32)
