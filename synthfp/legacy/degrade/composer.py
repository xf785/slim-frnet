"""
退化算子组合器

负责:
1. 从配置的算子池中随机选择2~4个算子
2. 按预定义的类别顺序排序 (保证施加顺序合理)
3. 为每个算子创建独立RNG (可复现)
4. 记录完整退化参数链
"""

import hashlib
import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from .base import DegradationOperator, DegradationParams
from .registry import DegradationRegistry

logger = logging.getLogger("fingerprint_synth.degradation.composer")

# 算子施加顺序: downsample → blur → noise → deform → occlusion → compression
CATEGORY_ORDER = {
    "downsample": 0,
    "gaussian_blur": 1,
    "motion_blur": 1,
    "defocus_blur": 1,
    "poisson_noise": 2,
    "speckle_noise": 2,
    "sensor_noise": 2,
    "elastic_deformation": 3,
    "ghosting": 3,
    "ridge_edge_roughness": 3,
    "random_erasure": 4,
    "ink_smudge": 4,
    "dirt_stain": 4,
    "blemish_spot": 4,
    "partial_missing": 4,
    "ridge_breaks": 4,
    "contrast_reduction": 5,
    "jpeg_compression": 5,
    "edge_feathering": 5,
}


class DegradationComposer:
    """退化算子组合器"""

    def __init__(self, config: Dict[str, Any], rng: np.random.Generator):
        self.config = config
        self.rng = rng
        # DegradationRegistry uses @classmethod throughout; no instance needed

    def compose_chain(self) -> List[DegradationOperator]:
        """
        从配置的算子池中随机选择2~4个算子，构建退化链。

        Returns:
            按施加顺序排列的算子列表
        """
        # 1. 获取启用的算子列表
        operators_cfg = self.config.get("operators", {})
        enabled = [
            name
            for name, cfg in operators_cfg.items()
            if cfg.get("enabled", False)
        ]
        if not enabled:
            raise ValueError("至少启用一个退化算子")

        # 2. 随机确定退化数量
        n_min, n_max = self.config.get("num_operators", (2, 4))
        n = self.rng.integers(n_min, n_max + 1)
        n = min(n, len(enabled))

        # 3. 随机不放回采样
        selected = list(
            self.rng.choice(enabled, size=n, replace=False)
        )

        # 4. 按类别顺序排序
        selected = sorted(selected, key=lambda n: CATEGORY_ORDER.get(n, 99))

        # 5. 实例化算子 (使用类方法, 无需实例)
        operators = []
        for name in selected:
            op = DegradationRegistry.create(name, operators_cfg[name])
            operators.append(op)

        logger.debug(f"退化链: {' → '.join(op.name for op in operators)}")
        return operators

    def apply_chain(
        self, hr_image: np.ndarray, sample_id: str = ""
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        施加退化链，返回 (LR图像, 参数记录列表)

        Args:
            hr_image: 高质量原始图像 (H,W), float32, [0,1]
            sample_id: 样本标识 (用于派生独立种子)

        Returns:
            lr_image: 退化后低质量图像
            param_log: 退化参数记录 (JSON可序列化)
        """
        chain = self.compose_chain()
        degraded = hr_image.copy()
        param_log = []

        for op in chain:
            # 为每个算子创建独立RNG (使用SHA256确保跨进程可复现)
            seed_bytes = hashlib.sha256(
                f"{sample_id}:{op.name}".encode("utf-8")
            ).digest()[:4]
            hash_val = int.from_bytes(seed_bytes, "big")
            op_rng = np.random.default_rng(
                self.rng.integers(0, 2**31 - 1) ^ hash_val
            )
            degraded = op.apply(degraded, op_rng)
            if op.get_last_params() is not None:
                param_log.append(op.get_last_params().to_dict())

        return degraded.astype(np.float32), param_log
