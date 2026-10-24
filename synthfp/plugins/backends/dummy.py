"""占位后端 —— 用于测试与流水线自检，不依赖合成内核。"""

from __future__ import annotations

import cv2
import numpy as np

from ...core.types import MasterArtifact, MasterSpec
from .base import BackendRegistry, FingerprintBackend

__all__ = ["DummyBackend"]


@BackendRegistry.register
class DummyBackend(FingerprintBackend):
    """生成确定性占位图案（椭圆掩模 + 正弦条纹），零随机耗时。

    用途：CI 冒烟测试、GUI 联调、管道性能基准。
    """

    name = "dummy"
    version = "2.0"
    description = "占位后端：正弦条纹 + 椭圆掩模，用于测试与流水线自检"

    def generate_master(self, spec: MasterSpec) -> MasterArtifact:
        width, height = int(spec.image_size[0]), int(spec.image_size[1])
        rng = np.random.default_rng(spec.seed)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (width // 2, height // 2),
            (max(width // 3, 8), max(height // 3, 8)),
            0,
            0,
            360,
            255,
            -1,
        )

        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        phase = rng.uniform(0, np.pi)
        period = float(rng.uniform(8.0, 14.0))
        field = 0.5 + 0.45 * np.cos(2 * np.pi * (xx * np.cos(phase) + yy * np.sin(phase)) / period)
        image = np.clip(field * 255.0, 0, 255).astype(np.uint8)
        image[mask == 0] = 255

        orientation = np.full((height, width), phase, dtype=np.float32)
        frequency = np.full((height, width), 1.0 / period, dtype=np.float32)

        return MasterArtifact(
            master_id=spec.master_id,
            fp_class=spec.fp_class,
            image=image,
            mask=mask,
            orientation=orientation,
            frequency=frequency,
            singular_points=[],
            shape_params={"sL": width // 2, "sR": width // 2, "sT": height // 4,
                          "sB": height // 4, "sM": height // 2},
            seed=int(spec.seed),
            backend=self.name,
            backend_version=self.version,
        )
