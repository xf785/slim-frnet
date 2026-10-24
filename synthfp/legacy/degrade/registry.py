"""退化算子注册表 — 支持动态注册新算子"""

from typing import Dict, Type, Any
from .base import DegradationOperator


class DegradationRegistry:
    """退化算子注册表"""

    _operators: Dict[str, Type[DegradationOperator]] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器: 注册退化算子"""

        def decorator(operator_cls):
            cls._operators[name] = operator_cls
            return operator_cls

        return decorator

    @classmethod
    def create(
        cls, name: str, config: Dict[str, Any]
    ) -> DegradationOperator:
        """根据名称创建算子实例"""
        if name not in cls._operators:
            raise KeyError(
                f"未知退化算子 '{name}'. 可用: {list(cls._operators.keys())}"
            )
        return cls._operators[name](config)

    @classmethod
    def list_operators(cls) -> list:
        """列出所有注册的算子"""
        return list(cls._operators.keys())
