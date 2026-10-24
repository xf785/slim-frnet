"""配置状态中心 —— 所有页面的唯一数据源。

GUI 的 8 个页面都是「配置的视图」：

* 每个页面通过 :meth:`ConfigStore.update` 修改配置（自动做 Pydantic 校验）；
* 任何修改都会发出 :attr:`ConfigStore.changed`，其它页面据此刷新；
* 校验失败时不落地，并把错误信息回传给调用方，实现「参数修改后实时校验」。

这样避免了页面之间互相引用，也让「配置面板 == 配置文件」在结构上成立。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ValidationError

from ..config import AppConfig, apply_overrides, build_config, dump_yaml, load_config, save_config
from ..config.loader import ConfigError
from .qt import QObject, Signal

__all__ = ["ConfigStore", "ConfigIssue"]


class ConfigIssue:
    """一条校验问题。"""

    __slots__ = ("field", "message")

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.field}: {self.message}"


class ConfigStore(QObject):
    """持有当前 :class:`AppConfig`，并在变更时广播。"""

    changed = Signal()  # 配置已变更
    loaded = Signal(str)  # 已从文件加载（路径）
    error = Signal(str)  # 校验或 IO 错误

    def __init__(self, config: Optional[AppConfig] = None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._config = config or build_config("default")
        self._path: Optional[Path] = None
        self._dirty = False
        self._suspend = 0

    # -- 读取 -------------------------------------------------------------
    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def dirty(self) -> bool:
        return self._dirty

    def summary(self) -> Dict[str, Any]:
        return self._config.estimate()

    # -- 修改 -------------------------------------------------------------
    def suspend(self) -> "_Suspend":
        """上下文管理器：临时抑制 ``changed`` 信号（批量更新时避免刷屏）。"""
        return _Suspend(self)

    def update(self, **overrides: Any) -> bool:
        """按点号路径批量更新配置（支持 ``dataset__identities`` 风格的键）。

        为了在 Python 里直接传嵌套键，这里同时接受：:

            store.update(**{"dataset.identities": 20})
            store.update(dataset={"identities": 20})     # 深合并

        Returns:
            ``True`` 表示校验通过并已生效。
        """
        flat: Dict[str, Any] = {}
        for key, value in overrides.items():
            if isinstance(value, dict) and "." not in key:
                flat.update(_flatten(value, prefix=key))
            else:
                flat[key] = value
        return self.update_flat(flat)

    def update_flat(self, flat: Dict[str, Any]) -> bool:
        """按扁平点号路径更新。"""
        try:
            new_config = apply_overrides(self._config, flat)
        except (ConfigError, ValidationError, ValueError, TypeError) as exc:
            self.error.emit(str(exc))
            return False
        # 值没变就不算"修改"：控件回填时常会以相同值重设一次，
        # 若一律标脏，用户什么都没动也会被问"是否保存"。
        if new_config.model_dump() == self._config.model_dump():
            return True
        self._config = new_config
        self._dirty = True
        if not self._suspend:
            self.changed.emit()
        return True

    def mark_clean(self) -> None:
        """把"有未保存修改"标记清掉（首屏初始化后调用）。"""
        self._dirty = False

    def replace(self, config: AppConfig, path: Optional[Path] = None) -> None:
        """整体替换配置（加载预设 / 打开文件时用）。"""
        self._config = config
        self._path = path
        self._dirty = False
        if not self._suspend:
            self.changed.emit()

    # -- 文件 -------------------------------------------------------------
    def load(self, path: Path) -> bool:
        try:
            config = load_config(path)
        except ConfigError as exc:
            self.error.emit(str(exc))
            return False
        except Exception as exc:  # pragma: no cover - 兜底
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return False
        self.replace(config, Path(path))
        self.loaded.emit(str(path))
        return True

    def load_preset(self, name: str) -> bool:
        try:
            self.replace(build_config(name))
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return False
        return True

    def save(self, path: Optional[Path] = None) -> bool:
        target = Path(path) if path else self._path
        if target is None:
            self.error.emit("尚未指定保存路径")
            return False
        try:
            save_config(self._config, target)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return False
        self._path = target
        self._dirty = False
        return True

    def to_yaml(self) -> str:
        return dump_yaml(self._config)

    def load_yaml_text(self, text: str) -> bool:
        """从 YAML 文本加载（内嵌编辑器用）。"""
        from ..config.loader import loads_config

        try:
            self.replace(loads_config(text))
        except ConfigError as exc:
            self.error.emit(str(exc))
            return False
        return True

    # -- 工厂 -------------------------------------------------------------
    @classmethod
    def from_preset(cls, name: str, parent: Optional[QObject] = None) -> "ConfigStore":
        return cls(build_config(name), parent)


class _Suspend:
    """``with store.suspend():`` 的上下文管理器。"""

    def __init__(self, store: ConfigStore) -> None:
        self._store = store

    def __enter__(self) -> ConfigStore:
        self._store._suspend += 1
        return self._store

    def __exit__(self, *exc: object) -> None:
        self._store._suspend = max(self._store._suspend - 1, 0)
        if self._store._suspend == 0:
            self._store.changed.emit()


def _flatten(mapping: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """``{"a": {"b": 1}}`` → ``{"a.b": 1}``（点号路径）。"""
    out: Dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out
