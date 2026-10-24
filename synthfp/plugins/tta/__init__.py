"""TTA 引擎 —— 契约 §1 规定的 Test-Time Augmentation 算子集合。

公开 API::

    TTAOperator         算子抽象基类（name/label/category/distortion/param_spec、
                        sample_params、apply、_apply、last_params、describe）
    TTARegistry         注册表（register/create/get/all_names/by_category/is_registered）
    create_operators    按名批量实例化（保持顺序）
    OPERATORS           全部注册名列表（21 个）

导入本包即完成 ``operators.py`` 的加载与注册；调用方无需再显式 import 子模块。

用法::

    import numpy as np
    from synthfp.plugins.tta import TTARegistry, create_operators

    rng = np.random.default_rng(42)
    op = TTARegistry.create("rotate", {"angle": 8.0})
    out = op.apply(image, rng)          # image: float32 (H, W) ∈ [0, 1]
    print(op.describe())                # 旋转(角度=8°)
    print(op.last_params)               # {'angle': 8.0}

    pipeline = create_operators(["elastic", "jpeg", "sensor_optical"])
"""

from __future__ import annotations

# noqa: F401 —— 导入 operators 子模块是为了触发 21 个算子的注册（写入 TTARegistry），
# 这里的"未使用"导入是契约要求的行为，不是冗余。
from . import operators as _operators  # noqa: F401
from .base import TTAOperator
from .operators import OPERATORS, create_operators
from .registry import TTARegistry

__all__ = ["TTAOperator", "TTARegistry", "create_operators", "OPERATORS"]
