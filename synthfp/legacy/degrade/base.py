"""退化算子抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class DegradationParams:
    """退化参数字典 — 序列化为JSON实现可复现"""

    operator_name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"operator": self.operator_name, "params": self.params}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DegradationParams":
        return cls(
            operator_name=data["operator"],
            params=data.get("params", {}),
        )


class DegradationOperator(ABC):
    """所有退化算子的抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._last_params: Optional[DegradationParams] = None

    @abstractmethod
    def apply(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """
        对输入图像施加退化。

        Args:
            image: 输入图像 (H, W), dtype=float32, range=[0, 1]
            rng:   独立的随机数生成器 (可复现性)

        Returns:
            degraded: 退化后图像 (H, W), dtype=float32, range=[0, 1]
        """
        ...

    @abstractmethod
    def sample_params(self, rng: np.random.Generator) -> Dict[str, Any]:
        """从配置参数空间中随机采样一组参数"""
        ...

    def get_last_params(self) -> Optional[DegradationParams]:
        """返回最近一次apply使用的参数 — 用于记录"""
        return self._last_params

    @property
    @abstractmethod
    def name(self) -> str:
        """算子唯一标识名"""
        ...

    def __repr__(self) -> str:
        return f"<{self.name}: {self._last_params}>"
