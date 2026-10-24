"""``synthfp.plugins.impressions`` 的单元测试（契约 4.3，≥ 12 个用例）。

覆盖：

* 位移场形状 / dtype / 有限性 / 确定性 / 掩模约束 / 压力单调性；
* ``deform_image`` 尺寸与 dtype 保持、零位移直通、不修改入参、越界报错；
* ``impression_transform`` 的仿射部分与 ``cv2.warpAffine`` 一致；
* ``map_coordinates`` 的恒等映射、常量场、双线性插值、越界裁剪；
* 每个传感器：值域、dtype、不修改入参、确定性、``strength=0`` 逐像素直通；
* 禁用依赖扫描（只允许 numpy / cv2）。
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from synthfp.plugins.impressions import (
    SENSORS,
    CapacitiveSensor,
    NoneSensor,
    OpticalSensor,
    SensorProfile,
    ThermalSensor,
    deform_image,
    get_sensor,
    impression_transform,
    map_coordinates,
    pressure_field,
)

HEIGHT, WIDTH = 96, 128
REAL_SENSORS = ("optical", "capacitive", "thermal")
ALL_SENSORS = ("optical", "capacitive", "thermal", "none")
PACKAGE_DIR = Path(__file__).resolve().parent.parent / "synthfp" / "plugins" / "impressions"


# --------------------------------------------------------------------------- #
# 测试辅助
# --------------------------------------------------------------------------- #
def make_rng(seed: int = 20240607) -> np.random.Generator:
    """构造确定性随机源。"""
    return np.random.default_rng(seed)


def ridge_image(height: int = HEIGHT, width: int = WIDTH, seed: int = 7) -> np.ndarray:
    """合成白底 + 深色脊线的指纹风格灰度图，``uint8``。"""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    ridges = 0.5 + 0.5 * np.cos(2.0 * np.pi * xx / 9.0 + 1.2 * np.sin(yy / 25.0))
    img = 255.0 - 150.0 * ridges
    img = img + np.random.default_rng(seed).normal(0.0, 3.0, (height, width))
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def ellipse_mask(height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """椭圆手指掩模，``uint8 {0, 255}``。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (width // 2, height // 2), (width // 3, height // 3), 0, 0, 360, 255, -1)
    return mask


def central_slice(height: int = HEIGHT, width: int = WIDTH, radius: int = 12) -> tuple[slice, slice]:
    """返回图像中心方形区域的切片。"""
    cy, cx = height // 2, width // 2
    return slice(cy - radius, cy + radius), slice(cx - radius, cx + radius)


# --------------------------------------------------------------------------- #
# pressure_field
# --------------------------------------------------------------------------- #
def test_pressure_field_shape_dtype_finite() -> None:
    dx, dy = pressure_field(HEIGHT, WIDTH, 0.4, make_rng())
    for field in (dx, dy):
        assert isinstance(field, np.ndarray)
        assert field.shape == (HEIGHT, WIDTH)
        assert field.dtype == np.float32
        assert np.isfinite(field).all()
        assert not np.isnan(field).any()


def test_pressure_field_zero_pressure_is_exactly_zero() -> None:
    dx, dy = pressure_field(HEIGHT, WIDTH, 0.0, make_rng())
    assert np.count_nonzero(dx) == 0
    assert np.count_nonzero(dy) == 0
    assert dx.dtype == np.float32 and dy.dtype == np.float32


def test_pressure_field_is_deterministic() -> None:
    a = pressure_field(HEIGHT, WIDTH, 0.5, make_rng(11))
    b = pressure_field(HEIGHT, WIDTH, 0.5, make_rng(11))
    c = pressure_field(HEIGHT, WIDTH, 0.5, make_rng(12))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_pressure_field_center_displacement_grows_with_pressure() -> None:
    rows, cols = central_slice()
    means: list[float] = []
    for pressure in (0.1, 0.2, 0.4, 0.8):
        dx, dy = pressure_field(HEIGHT, WIDTH, pressure, make_rng(3))
        means.append(float(np.hypot(dx[rows, cols], dy[rows, cols]).mean()))
    assert all(later > earlier for earlier, later in zip(means, means[1:])), means


def test_pressure_field_radial_model_signs() -> None:
    """中心 scale<1（源采样向中心压缩），边缘 scale>1（向外扩展）。"""
    pressure = 0.5
    dx, _ = pressure_field(HEIGHT, WIDTH, pressure, make_rng(5), elastic_alpha=0.0)
    cy, cx = HEIGHT // 2, WIDTH // 2
    ref = float(np.hypot((WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0))
    inner = dx[cy, cx + 4]
    outer = dx[cy, min(WIDTH - 1, cx + int(ref * 0.95))]
    assert inner < 0.0, inner
    assert outer > 0.0, outer


def test_pressure_field_mask_limits_and_zeroes_outside() -> None:
    mask = ellipse_mask()
    dx, dy = pressure_field(HEIGHT, WIDTH, 0.6, make_rng(9), mask)
    outside = mask == 0
    assert np.count_nonzero(dx[outside]) == 0
    assert np.count_nonzero(dy[outside]) == 0
    assert np.count_nonzero(dx[~outside]) > 0

    # 掩模形状不匹配必须报错
    with pytest.raises(ValueError):
        pressure_field(HEIGHT, WIDTH, 0.5, make_rng(), np.ones((3, 3), np.uint8))


def test_pressure_field_does_not_mutate_mask() -> None:
    mask = ellipse_mask()
    before = mask.copy()
    pressure_field(HEIGHT, WIDTH, 0.5, make_rng(4), mask)
    assert np.array_equal(mask, before)


def test_pressure_field_extreme_inputs_stay_finite() -> None:
    for pressure in (-3.0, 0.0, 1.0, 5.0, float("inf"), float("nan")):
        dx, dy = pressure_field(HEIGHT, WIDTH, pressure, make_rng(6))
        assert np.isfinite(dx).all() and np.isfinite(dy).all()

    with pytest.raises(ValueError):
        pressure_field(0, WIDTH, 0.5, make_rng())


# --------------------------------------------------------------------------- #
# deform_image
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_deform_image_preserves_shape_and_dtype(dtype: np.dtype) -> None:
    base = ridge_image()
    if dtype == np.float32:
        image = (base.astype(np.float32) / 255.0)
    else:
        image = base
    dx, dy = pressure_field(HEIGHT, WIDTH, 0.4, make_rng(21))
    out = deform_image(image, dx, dy)
    assert out.shape == image.shape
    assert out.dtype == image.dtype
    assert np.isfinite(out).all()


def test_deform_image_zero_field_is_pixel_exact_passthrough() -> None:
    image = ridge_image()
    dx, dy = pressure_field(HEIGHT, WIDTH, 0.0, make_rng(22))
    out = deform_image(image, dx, dy)
    assert np.array_equal(out, image)
    assert out is not image


def test_deform_image_does_not_mutate_input() -> None:
    image = ridge_image()
    before = image.copy()
    dx, dy = pressure_field(HEIGHT, WIDTH, 0.7, make_rng(23))
    deform_image(image, dx, dy)
    assert np.array_equal(image, before)


def test_deform_image_shape_mismatch_raises() -> None:
    image = ridge_image()
    dx, dy = pressure_field(HEIGHT, WIDTH, 0.3, make_rng(24))
    with pytest.raises(ValueError):
        deform_image(image, dx[:-1, :], dy)
    with pytest.raises(ValueError):
        deform_image(np.zeros((HEIGHT, WIDTH, 3, 2), np.uint8), dx, dy)


def test_deform_image_replicate_border_keeps_background_bright() -> None:
    image = ridge_image()
    dx, dy = impression_transform(HEIGHT, WIDTH, 0.0, 0.0, (40.0, 30.0), make_rng(25))
    out = deform_image(image, dx, dy)
    assert float(out.mean()) > 100.0  # BORDER_CONSTANT 会造出大片黑边
    assert int(out.min()) >= 0


# --------------------------------------------------------------------------- #
# impression_transform
# --------------------------------------------------------------------------- #
def test_impression_transform_identity_when_all_zero() -> None:
    dx, dy = impression_transform(HEIGHT, WIDTH, 0.0, 0.0, (0.0, 0.0), make_rng(31))
    assert dx.shape == (HEIGHT, WIDTH) and dx.dtype == np.float32
    assert np.count_nonzero(dx) == 0 and np.count_nonzero(dy) == 0
    image = ridge_image()
    assert np.array_equal(deform_image(image, dx, dy), image)


@pytest.mark.parametrize(
    ("rotation", "shift"),
    [(17.0, (0.0, 0.0)), (-8.5, (0.0, 0.0)), (0.0, (3.0, -4.0)), (25.0, (2.0, 5.0)), (90.0, (0.0, 0.0))],
)
def test_impression_transform_matches_warp_affine(rotation: float, shift: tuple[float, float]) -> None:
    image = ridge_image()
    dx, dy = impression_transform(HEIGHT, WIDTH, 0.0, rotation, shift, make_rng(32))
    matrix = cv2.getRotationMatrix2D(((WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0), rotation, 1.0)
    matrix[0, 2] += shift[0]
    matrix[1, 2] += shift[1]
    reference = cv2.warpAffine(
        image,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    mine = deform_image(image, dx, dy)
    assert int(np.abs(mine.astype(np.int32) - reference.astype(np.int32)).max()) <= 2


def test_impression_transform_adds_pressure_on_top_of_affine() -> None:
    affine_dx, affine_dy = impression_transform(HEIGHT, WIDTH, 0.0, 12.0, (1.0, 2.0), make_rng(33))
    dx, dy = impression_transform(HEIGHT, WIDTH, 0.5, 12.0, (1.0, 2.0), make_rng(33), ellipse_mask())
    assert np.isfinite(dx).all() and np.isfinite(dy).all()
    assert not np.allclose(dx, affine_dx)
    assert not np.allclose(dy, affine_dy)

    image = ridge_image()
    assert not np.array_equal(deform_image(image, dx, dy), deform_image(image, affine_dx, affine_dy))


def test_impression_transform_shift_moves_content_to_the_right() -> None:
    image = ridge_image()
    dx, dy = impression_transform(HEIGHT, WIDTH, 0.0, 0.0, (6.0, 0.0), make_rng(34))
    assert np.allclose(dx, -6.0)
    assert np.allclose(dy, 0.0)
    out = deform_image(image, dx, dy)
    # 内容右移 6 px：out[:, 6:] 应与原图 out 之前的区域一致
    assert np.array_equal(out[:, 6:], image[:, :-6])


# --------------------------------------------------------------------------- #
# map_coordinates
# --------------------------------------------------------------------------- #
def test_map_coordinates_identity_for_zero_field() -> None:
    dx, dy = pressure_field(HEIGHT, WIDTH, 0.0, make_rng(41))
    points = [(0.0, 0.0), (10.5, 20.25), (WIDTH - 1.0, HEIGHT - 1.0), (37.0, 12.0)]
    mapped = map_coordinates(dx, dy, points)
    assert isinstance(mapped, list) and len(mapped) == len(points)
    for (x, y), (mx, my) in zip(points, mapped):
        assert abs(mx - x) <= 0.51
        assert abs(my - y) <= 0.51
        assert isinstance(mx, float) and isinstance(my, float)


def test_map_coordinates_adds_constant_displacement() -> None:
    dx = np.full((HEIGHT, WIDTH), 5.0, np.float32)
    dy = np.full((HEIGHT, WIDTH), -3.0, np.float32)
    assert map_coordinates(dx, dy, [(10.0, 10.0)]) == [(15.0, 7.0)]
    assert map_coordinates(dx, dy, [(10.25, 20.75)]) == [(15.25, 17.75)]


def test_map_coordinates_bilinear_interpolation_is_exact_for_linear_field() -> None:
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    dx = (0.5 * xx + 0.25 * yy).astype(np.float32)
    dy = (-0.25 * xx + 0.125 * yy).astype(np.float32)
    x, y = 10.5, 20.25
    (mx, my), = map_coordinates(dx, dy, [(x, y)])
    assert mx == pytest.approx(x + 0.5 * x + 0.25 * y, abs=1e-3)
    assert my == pytest.approx(y - 0.25 * x + 0.125 * y, abs=1e-3)


def test_map_coordinates_clips_out_of_range_points() -> None:
    dx = np.full((HEIGHT, WIDTH), 50.0, np.float32)
    dy = np.full((HEIGHT, WIDTH), 50.0, np.float32)
    mapped = map_coordinates(dx, dy, [(-40.0, -40.0), (500.0, 900.0)])
    for (mx, my) in mapped:
        assert 0.0 <= mx <= WIDTH - 1
        assert 0.0 <= my <= HEIGHT - 1


def test_map_coordinates_empty_and_errors() -> None:
    dx = np.zeros((HEIGHT, WIDTH), np.float32)
    dy = np.zeros((HEIGHT, WIDTH), np.float32)
    assert map_coordinates(dx, dy, []) == []
    with pytest.raises(ValueError):
        map_coordinates(dx[:-1], dy, [(1.0, 1.0)])
    with pytest.raises(ValueError):
        map_coordinates(dx, dy, np.zeros((4, 3), np.float32))


# --------------------------------------------------------------------------- #
# sensor profile
# --------------------------------------------------------------------------- #
def test_sensor_registry_is_complete() -> None:
    assert set(SENSORS) == set(ALL_SENSORS)
    assert SENSORS["optical"] is OpticalSensor
    assert SENSORS["capacitive"] is CapacitiveSensor
    assert SENSORS["thermal"] is ThermalSensor
    assert SENSORS["none"] is NoneSensor
    for cls in SENSORS.values():
        assert issubclass(cls, SensorProfile)
        assert isinstance(cls.name, str) and cls.name
        assert isinstance(cls.label, str) and cls.label
    assert len({cls.name for cls in SENSORS.values()}) == len(SENSORS)
    assert len({cls.label for cls in SENSORS.values()}) == len(SENSORS)


@pytest.mark.parametrize("name", ALL_SENSORS)
def test_get_sensor_returns_instance(name: str) -> None:
    sensor = get_sensor(name)
    assert isinstance(sensor, SensorProfile)
    assert isinstance(sensor, SENSORS[name])
    assert sensor.name == name
    assert isinstance(sensor.describe(1.0), str) and sensor.label in sensor.describe(1.0)


@pytest.mark.parametrize("name", ["unknown", "", "NOPE", "sensor-x"])
def test_get_sensor_falls_back_to_none(name: str) -> None:
    sensor = get_sensor(name)
    assert isinstance(sensor, NoneSensor)
    assert sensor.name == "none"


@pytest.mark.parametrize("name", ALL_SENSORS)
def test_sensor_output_shape_dtype_and_range(name: str) -> None:
    image = ridge_image()
    out = get_sensor(name).apply(image, make_rng(51), strength=1.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == image.shape
    assert out.dtype == image.dtype == np.uint8
    assert int(out.min()) >= 0 and int(out.max()) <= 255
    assert np.isfinite(out.astype(np.float32)).all()


@pytest.mark.parametrize("name", ALL_SENSORS)
def test_sensor_zero_strength_is_pixel_exact_passthrough(name: str) -> None:
    image = ridge_image()
    before = image.copy()
    out = get_sensor(name).apply(image, make_rng(52), strength=0.0)
    assert np.array_equal(out, image)
    assert out is not image
    assert np.array_equal(image, before)


@pytest.mark.parametrize("name", ALL_SENSORS)
def test_sensor_does_not_mutate_input(name: str) -> None:
    image = ridge_image()
    before = image.copy()
    get_sensor(name).apply(image, make_rng(53), strength=1.7)
    assert np.array_equal(image, before)


@pytest.mark.parametrize("name", ALL_SENSORS)
def test_sensor_is_deterministic(name: str) -> None:
    image = ridge_image()
    first = get_sensor(name).apply(image, make_rng(54), strength=1.0)
    second = get_sensor(name).apply(image, make_rng(54), strength=1.0)
    third = get_sensor(name).apply(image, make_rng(55), strength=1.0)
    assert np.array_equal(first, second)
    if name != "none":
        assert not np.array_equal(first, third)


@pytest.mark.parametrize("name", REAL_SENSORS)
def test_sensor_changes_the_image(name: str) -> None:
    image = ridge_image()
    out = get_sensor(name).apply(image, make_rng(56), strength=1.0)
    assert not np.array_equal(out, image)
    assert float(np.abs(out.astype(np.int32) - image.astype(np.int32)).mean()) > 0.5


@pytest.mark.parametrize("name", REAL_SENSORS)
def test_sensor_effect_grows_with_strength(name: str) -> None:
    image = ridge_image()
    sensor = get_sensor(name)
    weak = sensor.apply(image, make_rng(57), strength=0.2)
    strong = sensor.apply(image, make_rng(57), strength=1.0)
    diff_weak = float(np.abs(weak.astype(np.int32) - image.astype(np.int32)).mean())
    diff_strong = float(np.abs(strong.astype(np.int32) - image.astype(np.int32)).mean())
    assert diff_strong > diff_weak


@pytest.mark.parametrize("strength", [-5.0, 0.0, 0.5, 3.0, 99.0, float("nan"), float("inf")])
def test_sensor_extreme_strength_is_bounded(strength: float) -> None:
    image = ridge_image()
    for name in ALL_SENSORS:
        out = get_sensor(name).apply(image, make_rng(58), strength=strength)
        assert out.dtype == np.uint8 and out.shape == image.shape
        assert int(out.min()) >= 0 and int(out.max()) <= 255


def test_none_sensor_is_always_passthrough() -> None:
    image = ridge_image()
    sensor = NoneSensor()
    for strength in (0.0, 0.5, 3.0, -1.0):
        out = sensor.apply(image, make_rng(59), strength=strength)
        assert np.array_equal(out, image)
        assert out is not image


@pytest.mark.parametrize("name", ALL_SENSORS)
def test_sensor_accepts_float_image_and_keeps_dtype(name: str) -> None:
    unit = (ridge_image().astype(np.float32) / 255.0)
    out = get_sensor(name).apply(unit, make_rng(60), strength=1.0)
    assert out.dtype == np.float32
    assert out.shape == unit.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    exact = get_sensor(name).apply(unit, make_rng(60), strength=0.0)
    assert np.array_equal(exact, unit)


@pytest.mark.parametrize("name", ALL_SENSORS)
def test_sensor_handles_tiny_and_flat_images(name: str) -> None:
    sensor = get_sensor(name)
    for image in (np.full((3, 3), 255, np.uint8), np.zeros((1, 9), np.uint8), np.full((8, 1), 128, np.uint8)):
        out = sensor.apply(image, make_rng(61), strength=1.5)
        assert out.shape == image.shape and out.dtype == np.uint8


def test_sensor_rejects_non_grayscale_input() -> None:
    with pytest.raises(ValueError):
        get_sensor("optical").apply(np.zeros((8, 8, 3), np.uint8), make_rng(62), strength=1.0)


# --------------------------------------------------------------------------- #
# 契约约束
# --------------------------------------------------------------------------- #
def test_imports_only_allow_numpy_and_cv2() -> None:
    forbidden = ("scipy", "skimage", "matplotlib", "pandas", "click", "tqdm", "PySide6")
    pattern = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", re.MULTILINE)
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        modules = pattern.findall(path.read_text(encoding="utf-8"))
        for module in modules:
            root = module.split(".")[0]
            assert root not in forbidden, f"{path.name} 引入了禁用依赖 {module}"
            assert root in ("abc", "collections", "cv2", "numpy", "synthfp", "__future__"), (
                f"{path.name} 引入了未授权依赖 {module}"
            )


def test_public_api_exports() -> None:
    import synthfp.plugins.impressions as pkg

    expected = {
        "pressure_field",
        "deform_image",
        "impression_transform",
        "map_coordinates",
        "SensorProfile",
        "OpticalSensor",
        "CapacitiveSensor",
        "ThermalSensor",
        "NoneSensor",
        "SENSORS",
        "get_sensor",
    }
    assert expected <= set(dir(pkg))
    assert expected <= set(pkg.__all__)
