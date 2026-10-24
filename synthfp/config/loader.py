"""配置加载 / 保存 / 覆盖 —— YAML <-> AppConfig。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml
from pydantic import ValidationError

from .schema import AppConfig

__all__ = [
    "load_config",
    "save_config",
    "dump_yaml",
    "loads_config",
    "apply_overrides",
    "ConfigError",
    "to_plain",
]

_ENV_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """配置加载 / 校验失败。"""


# ---------------------------------------------------------------------------
# 环境变量与深合并
# ---------------------------------------------------------------------------
def _resolve_env(value: Any) -> Any:
    """递归替换 ``${VAR}`` / ``${VAR:-default}``。"""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, Mapping):
            out[key] = _deep_merge(out[key], val)  # type: ignore[arg-type]
        else:
            out[key] = val
    return out


def _dot_set(target: Dict[str, Any], path: str, value: Any) -> None:
    """``"tta.mode"`` -> ``target["tta"]["mode"] = value``。

    路径段既可以是字典键，也可以是**序列下标**（``backend.ridge_gray.0``）。
    tuple 不可变，因此遇到序列时要重建并回写到父节点 —— 早先这里只当作
    字典键处理，``ridge_gray.0`` 会把整个元组替换成 ``{"0": 25}``，
    导致这几个控件一改就校验失败。
    """
    _set_path(target, path.split("."), value)


def _set_path(node: Any, keys: List[str], value: Any) -> Any:
    """沿 ``keys`` 写入，返回（可能被重建的）节点。"""
    key = keys[0]
    if len(keys) == 1:
        if isinstance(node, dict):
            node[key] = value
            return node
        index = int(key)
        items = list(node)
        items[index] = value
        return tuple(items) if isinstance(node, tuple) else items

    if isinstance(node, dict):
        child = node.get(key)
        if not isinstance(child, (dict, list, tuple)):
            child = {}
        node[key] = _set_path(child, keys[1:], value)
        return node

    index = int(key)
    items = list(node)
    items[index] = _set_path(items[index], keys[1:], value)
    return tuple(items) if isinstance(node, tuple) else items


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------
def to_plain(obj: Any) -> Any:
    """把 Pydantic / Path / tuple 递归转成 YAML 友好的纯 Python 对象。"""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _raw_from_path(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"配置文件不存在: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - 取决于用户输入
        raise ConfigError(f"YAML 解析失败 ({p}): {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件顶层必须是映射 (mapping)，实际为 {type(raw).__name__}")
    return _resolve_env(raw)


def _validate(raw: Dict[str, Any]) -> AppConfig:
    try:
        return AppConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["配置校验失败："]
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "<root>"
        lines.append(f"  • {loc}: {err['msg']}")
    return "\n".join(lines)


def loads_config(text: str) -> AppConfig:
    """从 YAML 文本构造配置（GUI 内嵌编辑器用）。"""
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ConfigError("配置文本顶层必须是映射 (mapping)")
    return _validate(_resolve_env(raw))


def load_config(
    path: Union[str, Path] = "configs/default.yaml",
    overrides: Optional[Mapping[str, Any]] = None,
) -> AppConfig:
    """加载 YAML 配置并应用点号路径覆盖。

    Args:
        path: YAML 路径。
        overrides: ``{"runtime.seed": 7, "dataset.identities": 20}``。

    Note:
        支持 ``parent: <预设名>`` 继承。父配置先查 ``configs/presets/<name>.yaml``，
        找不到再回退到**内置预设**的覆盖片段，因此 ``parent: quick_test`` 也能生效。
    """
    raw = _raw_from_path(path)

    # 支持 ``parent: <preset-name>`` 继承：父配置打底，子文件整体覆盖
    parent = raw.pop("parent", None)
    if parent:
        raw = _deep_merge(_parent_overrides(str(parent)), raw)

    if overrides:
        for key, val in overrides.items():
            if val is not None:
                _dot_set(raw, key, val)

    return _validate(raw)


def _parent_overrides(name: str) -> Dict[str, Any]:
    """取父预设的原始覆盖片段（磁盘 YAML 优先，其次内置预设）。"""
    from .presets import PRESETS, preset_path

    parent_file = preset_path(name)
    if parent_file is not None:
        return _raw_from_path(parent_file)

    preset = PRESETS.get(name)
    if preset is None:
        available = ", ".join(sorted(PRESETS)) or "（无）"
        raise ConfigError(f"未知的父预设 '{name}'；可用内置预设: {available}")
    return dict(preset.overrides)


def save_config(config: AppConfig, path: Union[str, Path]) -> Path:
    """把配置写成 YAML（字段顺序与 schema 一致）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = to_plain(config.model_dump(mode="python"))
    out.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    return out


def dump_yaml(config: AppConfig) -> str:
    """配置 -> YAML 文本。"""
    return yaml.dump(
        to_plain(config.model_dump(mode="python")),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )


def apply_overrides(config: AppConfig, overrides: Mapping[str, Any]) -> AppConfig:
    """在已有配置对象上应用覆盖，返回**新对象**。

    合并语义（刻意分层，同时避免两种错误）：

    * **顶层小节**（``dataset`` / ``backend`` / ``tta`` … ）：**浅合并**
      （``section.update(patch)``）—— 只替换小节内给出的字段，其余原样保留。
      这是 ``synthfp generate -c my.yaml --preset quick_test`` 能保住
      ``my.yaml`` 里未被预设覆盖字段的关键。
    * **小节内的字段**：**整体替换**。像 ``classes.distribution`` /
      ``tta.operators`` 这样的字典或列表本身就是一个完整的值：
      预设写 ``distribution: {whorl: 1.0}`` 的语义是"只要螺型"，
      若递归合并就会变成"保留原来五类再加螺型"（比例和 1.8，直接校验失败）。

    ``None`` 一律忽略（表示"本次不覆盖"，便于 CLI 用 ``None`` 表达"未指定"）。
    """
    raw = config.model_dump(mode="python")
    for key, val in overrides.items():
        if val is None:
            continue
        if isinstance(val, Mapping) and "." not in key:
            current = raw.get(key)
            if isinstance(current, dict):
                merged = dict(current)
                merged.update(val)  # 浅合并：小节内字段整体替换
                raw[key] = merged
            else:
                raw[key] = dict(val)
        else:
            _dot_set(raw, key, val)
    return _validate(raw)
