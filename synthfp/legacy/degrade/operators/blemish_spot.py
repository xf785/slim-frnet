"""病斑 — 模拟手指皮肤病变、疤痕、老茧等局部异常"""
import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry

@DegradationRegistry.register("blemish_spot")
class BlemishSpotOperator(DegradationOperator):
    """病斑: 模拟手指皮肤上的病变、疤痕、老茧、水泡等异常区域"""
    @property
    def name(self) -> str:
        return "blemish_spot"

    def sample_params(self, rng: np.random.Generator) -> dict:
        num_range = self.config.get("num_spots", [1, 6])
        size_range = self.config.get("size_range", [8, 40])
        severity_range = self.config.get("severity", [0.2, 0.7])
        types = self.config.get("types", ["dark", "light", "blur"])
        num = int(rng.integers(num_range[0], num_range[1] + 1))
        severity = rng.uniform(severity_range[0], severity_range[1])
        spot_type = str(rng.choice(types))
        return {"num_spots": num, "min_size": size_range[0], "max_size": size_range[1],
                "severity": round(severity, 3), "type": spot_type}

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(operator_name=self.name, params=params)
        h, w = image.shape
        num, min_sz, max_sz = params["num_spots"], params["min_size"], params["max_size"]
        severity, spot_type = params["severity"], params["type"]
        result = image.copy()
        for _ in range(num):
            cx, cy = rng.integers(0, w), rng.integers(0, h)
            radius = rng.integers(min_sz, max_sz + 1)
            aspect = rng.uniform(0.7, 1.0)
            rx, ry = int(radius), int(radius * aspect)
            angle = rng.uniform(0, 180)
            yy, xx = np.ogrid[:h, :w]
            x_rot = (xx - cx) * np.cos(np.deg2rad(angle)) + (yy - cy) * np.sin(np.deg2rad(angle))
            y_rot = -(xx - cx) * np.sin(np.deg2rad(angle)) + (yy - cy) * np.cos(np.deg2rad(angle))
            dist = np.sqrt((x_rot / rx) ** 2 + (y_rot / ry) ** 2)
            transition = (0.85 - dist) / 0.45
            spot_mask = np.clip(transition, 0, 1).astype(np.float32)
            if rng.random() < 0.5:
                spot_mask = np.clip(spot_mask + rng.uniform(-0.08, 0.08), 0, 1)
            if spot_type == "dark":
                darken = 1.0 - spot_mask * severity * 0.6
                blurred = cv2.GaussianBlur(result, (7, 7), 2.5)
                result = result * (1 - spot_mask * severity * 0.5) + blurred * spot_mask * severity * 0.5
                result = result * darken + result * (1 - darken) * 0.5
            elif spot_type == "light":
                blurred = cv2.GaussianBlur(result, (5, 5), 1.5)
                blend = 1.0 - spot_mask * severity
                result = result * blend + blurred * (1.0 - blend) * 1.15
            elif spot_type == "blur":
                ks = int(9 + severity * 18)
                if ks % 2 == 0: ks += 1
                heavily_blurred = cv2.GaussianBlur(result, (ks, ks), ks / 3)
                result = result * (1.0 - spot_mask * severity) + heavily_blurred * spot_mask * severity
        return np.clip(result, 0, 1).astype(np.float32)
