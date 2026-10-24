"""TTA 算子注册表。

算子通过 :meth:`TTARegistry.register` 装饰器注册；装饰器同时把 ``name`` /
``label`` / ``category`` / ``distortion`` / ``param_spec`` 写入算子类属性，
因此 GUI（按分类列算子）与管道（按名实例化）都只依赖注册表这一处真相。

设计要点：
    * 注册表是**类级全局状态**，注册发生在 ``operators`` 模块导入时；
    * 同名重复注册按“后者覆盖”处理（便于热重载与自定义算子覆盖内置算子）；
    * 未知算子名一律抛 :class:`KeyError`，并在错误信息里列出全部可用算子。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Type

from .base import TTAOperator

__all__ = ["TTARegistry"]


class TTARegistry:
    """TTA 算子注册表（全部为类方法，无需实例化）。"""

    _operators: Dict[str, Type[TTAOperator]] = {}

    # -- 注册 -------------------------------------------------------------
    @classmethod
    def register(
        cls,
        name: str,
        label: str,
        category: str,
        distortion: float,
        param_spec: Optional[dict] = None,
    ) -> Callable[[type], type]:
        """返回注册装饰器：登记算子类并写入其类属性。

        Args:
            name: 算子标识（=注册名），如 ``"rotate"``。
            label: 中文名，如 ``"旋转"``。
            category: 分类，取值见 :data:`~synthfp.plugins.tta.base.TTA_CATEGORIES`。
            distortion: 典型失真度 ``0~1``。
            param_spec: ``{参数名: (中文名, 最小值, 最大值, 步长, 单位)}``。

        Returns:
            类装饰器，原样返回被装饰的类。

        Raises:
            TypeError: 被装饰对象不是 :class:`TTAOperator` 的子类。
        """

        def decorator(operator_cls: type) -> type:
            if not (isinstance(operator_cls, type) and issubclass(operator_cls, TTAOperator)):
                raise TypeError(f"只能注册 TTAOperator 的子类，收到 {operator_cls!r}")
            operator_cls.name = str(name)
            operator_cls.label = str(label)
            operator_cls.category = str(category)
            operator_cls.distortion = float(distortion)
            operator_cls.param_spec = dict(param_spec) if param_spec else {}
            cls._operators[str(name)] = operator_cls
            return operator_cls

        return decorator

    # -- 查询 -------------------------------------------------------------
    @classmethod
    def create(cls, name: str, params: Optional[dict] = None) -> TTAOperator:
        """按名实例化算子。

        Args:
            name: 注册名。
            params: 用户覆盖的固定参数，如 ``{"angle": 8.0}``。

        Returns:
            算子实例。

        Raises:
            KeyError: 未注册的算子名（错误信息含可用算子列表）。
        """
        return cls.get(name)(params)

    @classmethod
    def get(cls, name: str) -> Type[TTAOperator]:
        """按名取算子类。

        Args:
            name: 注册名。

        Returns:
            算子类。

        Raises:
            KeyError: 未注册的算子名（错误信息含可用算子列表）。
        """
        try:
            return cls._operators[name]
        except KeyError:
            available = ", ".join(cls._operators.keys())
            raise KeyError(f"未知 TTA 算子 '{name}'；可用算子: [{available}]") from None

    @classmethod
    def all_names(cls) -> List[str]:
        """返回全部注册名（按注册顺序，即 ``operators.py`` 中的定义顺序）。"""
        return list(cls._operators.keys())

    @classmethod
    def by_category(cls) -> Dict[str, List[str]]:
        """按分类分组返回注册名。

        Returns:
            ``{分类: [注册名, ...]}``，分类顺序为首次出现的顺序。
        """
        grouped: Dict[str, List[str]] = {}
        for name, operator_cls in cls._operators.items():
            grouped.setdefault(operator_cls.category, []).append(name)
        return grouped

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """判断算子名是否已注册。"""
        return name in cls._operators
