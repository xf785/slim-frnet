"""后端插件契约 —— 设计文档 4.5。

后端只做一件事：**把 :class:`MasterSpec` 渲染成一张主指纹**
（灰度图 + 掩模 + 方向场 + 频率图 + 奇异点 + 轮廓参数）。

后端不得写文件、不得接触 GUI、不得感知 TTA 与捺印。
"""

from __future__ import annotations

import abc
from typing import Dict, List, Type

from ...core.types import MasterArtifact, MasterSpec

__all__ = ["FingerprintBackend", "BackendRegistry", "get_backend", "list_backends"]


class FingerprintBackend(abc.ABC):
    """指纹生成后端抽象基类。"""

    #: 后端标识（与配置 ``backend.name`` 对应）
    name: str = "abstract"
    #: 后端版本（写入 manifest 的 ``backend_version``）
    version: str = "0"
    #: 人类可读描述（GUI 后端页签显示）
    description: str = ""

    def __init__(self, config: Dict | None = None) -> None:
        """
        Args:
            config: 后端配置字典（对应 ``BackendSection.model_dump()``）。
        """
        self.config: Dict = dict(config or {})

    @abc.abstractmethod
    def generate_master(self, spec: MasterSpec) -> MasterArtifact:
        """渲染一张主指纹。

        Args:
            spec: 主指纹规格（含稳定种子、类别、尺寸）。

        Returns:
            :class:`MasterArtifact`，图像为 ``uint8 (H, W)``、背景 255。
        """
        raise NotImplementedError

    def generate_impressions(self, master: MasterArtifact, count: int, seed: int):  # pragma: no cover
        """[已废弃] 捺印阶段由 ``synthfp.plugins.impressions`` 负责。

        保留此方法仅为兼容早期调用点，默认抛 ``NotImplementedError``。
        """
        raise NotImplementedError(
            "generate_impressions 已移至 synthfp.plugins.impressions（捺印插件）"
        )

    def warmup(self) -> None:
        """预热（可选）：提前构建 Gabor 核库等。"""
        return None

    def describe(self) -> str:
        return f"{self.name} v{self.version}" + (f" — {self.description}" if self.description else "")


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
class BackendRegistry:
    """后端注册表（插件式，见设计文档「可扩展」原则）。"""

    _backends: Dict[str, Type[FingerprintBackend]] = {}

    @classmethod
    def register(cls, backend_cls: Type[FingerprintBackend]) -> Type[FingerprintBackend]:
        """注册后端类（作为装饰器使用）。"""
        name = getattr(backend_cls, "name", "")
        if not name or name == "abstract":
            raise ValueError(f"后端类 {backend_cls.__name__} 必须定义非空 name")
        cls._backends[name] = backend_cls
        return backend_cls

    @classmethod
    def get(cls, name: str) -> Type[FingerprintBackend]:
        if name not in cls._backends:
            raise KeyError(f"未知后端 '{name}'；可用: {sorted(cls._backends)}")
        return cls._backends[name]

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._backends)

    @classmethod
    def describe_all(cls) -> List[Dict[str, str]]:
        return [
            {
                "name": n,
                "version": cls._backends[n].version,
                "description": cls._backends[n].description,
            }
            for n in cls.names()
        ]


def get_backend(name: str = "sfinge", config: Dict | None = None) -> FingerprintBackend:
    """按名称实例化后端。"""
    return BackendRegistry.get(name)(config)


def list_backends() -> List[str]:
    return BackendRegistry.names()
