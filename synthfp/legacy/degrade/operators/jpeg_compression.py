"""JPEG压缩伪影 — 模拟有损压缩"""

import io
import numpy as np
import cv2
from ..base import DegradationOperator, DegradationParams
from ..registry import DegradationRegistry


@DegradationRegistry.register("jpeg_compression")
class JPEGCompressionOperator(DegradationOperator):
    """JPEG压缩: 模拟有损压缩产生的块状伪影"""

    @property
    def name(self) -> str:
        return "jpeg_compression"

    def sample_params(self, rng: np.random.Generator) -> dict:
        quality_range = self.config.get("quality", [30, 75])
        quality = int(rng.integers(quality_range[0], quality_range[1] + 1))
        return {"quality": quality}

    def apply(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        params = self.sample_params(rng)
        self._last_params = DegradationParams(
            operator_name=self.name, params=params
        )

        quality = params["quality"]
        # 将float32 [0,1] → uint8 [0,255]
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        # JPEG压缩/解压
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, encoded = cv2.imencode(".jpg", img_uint8, encode_params)
        if not success:
            # 编码失败时返回原图
            return image.copy().astype(np.float32)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            # 解码失败时返回原图
            return image.copy().astype(np.float32)

        # 回到 float32 [0,1]
        degraded = decoded.astype(np.float32) / 255.0
        return np.clip(degraded, 0, 1).astype(np.float32)
