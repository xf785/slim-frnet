"""synthfp.plugins.impressions —— 捺印效果插件（皮肤压力形变 + 传感器模拟）。

两个子模块（见 ``CONTRACTS.md`` 第 4 节）::

    deform.py   压力位移场、仿射组合变换、重采样与标注坐标同步映射
    sensor.py   光学 / 电容 / 热感 / 直通 传感器成像模型

典型用法::

    import numpy as np
    from synthfp.plugins.impressions import impression_transform, deform_image, get_sensor

    rng = np.random.default_rng(0)
    dx, dy = impression_transform(320, 320, pressure=0.35, rotation=5.0,
                                  shift=(2.0, -3.0), rng=rng, mask=mask)
    pressed = deform_image(fingerprint, dx, dy)
    sensed = get_sensor("optical").apply(pressed, rng, strength=1.0)
"""

from __future__ import annotations

from synthfp.plugins.impressions.deform import (
    deform_image,
    impression_transform,
    map_coordinates,
    pressure_field,
)
from synthfp.plugins.impressions.sensor import (
    SENSORS,
    CapacitiveSensor,
    NoneSensor,
    OpticalSensor,
    SensorProfile,
    ThermalSensor,
    get_sensor,
)

__all__ = [
    "SENSORS",
    "CapacitiveSensor",
    "NoneSensor",
    "OpticalSensor",
    "SensorProfile",
    "ThermalSensor",
    "deform_image",
    "get_sensor",
    "impression_transform",
    "map_coordinates",
    "pressure_field",
]
