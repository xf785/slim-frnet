"""TTA 引擎单元测试（契约 §1.5，≥ 30 个用例）。

覆盖范围：
    * 注册表完整性（恰好 21 个算子）与元数据（中文名 / 分类 / 失真度 / 参数规格）；
    * ``sample_params`` 的键集合、取值范围、确定性与 JSON 可序列化；
    * 用户固定参数覆盖、越界裁剪、“随机 ±”参数的符号分布；
    * ``apply`` 不改变 shape/dtype、值域安全（无 NaN/Inf）、不修改输入、可复现；
    * 几何算子保持尺寸、边界 replicate 不产生黑边、零参数为恒等；
    * 遮挡算子按背景色填充、不产生纯黑块；
    * 传感器算子 ``strength=0`` 逐像素直通；
    * ``create_operators`` 的顺序与未知名报错；
    * 依赖约束（只用 numpy / cv2 / 标准库）与静态检查（无全局 ``np.random.*``）。
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from synthfp.plugins import tta as tta_pkg
from synthfp.plugins.tta import OPERATORS, TTARegistry, TTAOperator, create_operators

# ---------------------------------------------------------------------------
# 常量与工具
# ---------------------------------------------------------------------------
#: 契约 §1.3 表格中的 21 个注册名（顺序与表格一致）。
EXPECTED_NAMES: tuple[str, ...] = (
    "rotate",
    "translate",
    "scale",
    "shear",
    "elastic",
    "perspective",
    "brightness",
    "contrast",
    "gamma",
    "blur",
    "motion_blur",
    "gaussian_noise",
    "poisson",
    "salt_pepper",
    "jpeg",
    "occlusion_rect",
    "occlusion_ellipse",
    "scratch",
    "sensor_optical",
    "sensor_capacitive",
    "sensor_thermal",
)

GEOMETRIC_NAMES: tuple[str, ...] = (
    "rotate",
    "translate",
    "scale",
    "shear",
    "elastic",
    "perspective",
)
OCCLUSION_NAMES: tuple[str, ...] = ("occlusion_rect", "occlusion_ellipse", "scratch")
SENSOR_NAMES: tuple[str, ...] = ("sensor_optical", "sensor_capacitive", "sensor_thermal")

CATEGORIES: frozenset[str] = frozenset({"几何", "光度", "噪声", "压缩", "遮挡", "传感器"})

TTA_DIR = Path(tta_pkg.__file__).resolve().parent
TTA_SOURCES = sorted(TTA_DIR.glob("*.py"))

ALLOWED_IMPORT_ROOTS = {"numpy", "cv2"} | set(sys.stdlib_module_names)
FORBIDDEN_IMPORT_ROOTS = {
    "scipy",
    "skimage",
    "sklearn",
    "matplotlib",
    "pandas",
    "click",
    "tqdm",
    "pyside6",
    "tensorflow",
    "torch",
}

#: 禁止在算子实现里出现的全局随机函数（只允许 ``np.random.default_rng``）。
GLOBAL_RANDOM_RE = re.compile(
    r"np\.random\.(?!default_rng\b)"
    r"(seed|rand|randn|random|random_sample|normal|uniform|integers|choice|"
    r"shuffle|permutation|poisson|gamma|beta|standard_normal|multivariate_normal)\b"
)


def make_image(height: int = 128, width: int = 128) -> np.ndarray:
    """构造“类指纹”图：中心圆形区域为正弦脊线，四周为白色背景。

    Returns:
        ``float32 (H, W)``，取值 ``[0.1, 1.0]``，背景恒为 ``1.0``。
    """
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    ridges = 0.5 + 0.40 * np.sin(2.0 * np.pi * (grid_x * 0.085 + 0.02 * grid_y))
    radius = np.sqrt(
        (grid_x - (width - 1) / 2.0) ** 2 + (grid_y - (height - 1) / 2.0) ** 2
    )
    inside = radius < min(height, width) * 0.40
    return np.clip(np.where(inside, ridges, 1.0), 0.0, 1.0).astype(np.float32)


def public_params(params: dict) -> dict:
    """剔除隐藏键（``_seed`` 等），只保留 ``param_spec`` 中的公开参数。"""
    return {key: value for key, value in params.items() if not str(key).startswith("_")}


def imported_roots(path: Path) -> list[str]:
    """用 AST 静态解析一个 Python 文件里的 import 根模块名（忽略文档字符串与相对导入）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # 相对导入（包内模块）
                continue
            if node.module:
                roots.append(node.module.split(".")[0])
    return roots


# ---------------------------------------------------------------------------
# 夹具（自包含，不依赖其他测试模块的共享夹具）
# ---------------------------------------------------------------------------
@pytest.fixture
def tta_rng() -> np.random.Generator:
    """函数级确定性随机源。"""
    return np.random.default_rng(20240517)


@pytest.fixture(scope="module")
def tta_image() -> np.ndarray:
    """``float32 (128, 128)`` 类指纹图。"""
    return make_image(128, 128)


@pytest.fixture(scope="module")
def tta_image_u8(tta_image: np.ndarray) -> np.ndarray:
    """``uint8`` 版本的类指纹图。"""
    return np.clip(np.rint(tta_image * 255.0), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def tta_white() -> np.ndarray:
    """``float32 (96, 96)`` 全白图。"""
    return np.ones((96, 96), dtype=np.float32)


@pytest.fixture(scope="module")
def tta_black() -> np.ndarray:
    """``float32 (64, 64)`` 全黑图。"""
    return np.zeros((64, 64), dtype=np.float32)


def new_operator(name: str, params: dict | None = None) -> TTAOperator:
    """按注册名创建算子实例。"""
    return TTARegistry.create(name, params)


# ===========================================================================
# 1. 注册表
# ===========================================================================
def test_registry_exposes_exactly_21_operators() -> None:
    """注册表恰好包含契约规定的 21 个算子，且无重复。"""
    assert len(OPERATORS) == 21
    assert len(set(OPERATORS)) == 21
    assert set(OPERATORS) == set(EXPECTED_NAMES)


def test_operators_list_matches_registry_order() -> None:
    """``OPERATORS`` 与注册表实际注册顺序完全一致。"""
    assert list(OPERATORS) == TTARegistry.all_names()
    assert list(OPERATORS) == list(EXPECTED_NAMES)


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_registry_metadata_complete(name: str) -> None:
    """每个算子的 name/label/category/distortion/param_spec 都符合契约。"""
    operator_cls = TTARegistry.get(name)
    assert operator_cls.name == name
    assert isinstance(operator_cls.label, str) and operator_cls.label
    assert operator_cls.label != name, "label 应为中文名"
    assert operator_cls.category in CATEGORIES
    assert 0.0 <= float(operator_cls.distortion) <= 1.0
    assert isinstance(operator_cls.param_spec, dict) and operator_cls.param_spec


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_registry_get_returns_operator_subclass(name: str) -> None:
    """``get`` 返回 TTAOperator 子类，且 ``is_registered`` 为真。"""
    operator_cls = TTARegistry.get(name)
    assert isinstance(operator_cls, type)
    assert issubclass(operator_cls, TTAOperator)
    assert TTARegistry.is_registered(name)
    assert isinstance(new_operator(name), TTAOperator)


def test_registry_is_registered_false_for_unknown() -> None:
    """未注册名返回 False。"""
    assert TTARegistry.is_registered("no_such_operator") is False
    assert TTARegistry.is_registered("") is False


def test_registry_by_category_covers_all() -> None:
    """``by_category`` 覆盖全部算子，且分类键集合等于契约六分类。"""
    grouped = TTARegistry.by_category()
    assert set(grouped) == CATEGORIES
    flat = [name for names in grouped.values() for name in names]
    assert sorted(flat) == sorted(EXPECTED_NAMES)
    assert grouped["几何"] == list(GEOMETRIC_NAMES)
    assert grouped["遮挡"] == list(OCCLUSION_NAMES)
    assert grouped["传感器"] == list(SENSOR_NAMES)
    assert grouped["压缩"] == ["jpeg"]


def test_registry_create_unknown_raises_keyerror_listing_names() -> None:
    """未知算子名抛 KeyError，且错误信息列出可用算子。"""
    with pytest.raises(KeyError) as excinfo:
        TTARegistry.create("rotatee")
    message = str(excinfo.value)
    assert "rotatee" in message
    assert "rotate" in message
    assert "sensor_thermal" in message


def test_registry_get_unknown_raises_keyerror() -> None:
    """``get`` 对未知名同样抛 KeyError。"""
    with pytest.raises(KeyError):
        TTARegistry.get("___missing___")


def test_registry_create_with_and_without_params() -> None:
    """``create`` 支持带/不带用户固定参数，且实例类型正确。"""
    plain = new_operator("rotate")
    assert plain.params == {}
    pinned = new_operator("rotate", {"angle": 8.0})
    assert pinned.params == {"angle": 8.0}
    assert isinstance(pinned, TTAOperator)


def test_registry_create_ignores_unknown_param_keys() -> None:
    """未知参数键被忽略（不崩溃），合法键保留。"""
    operator = TTARegistry.create("rotate", {"angle": 5.0, "nope": 1.0})
    assert operator.params == {"angle": 5.0}


def test_registry_register_decorator_writes_metadata() -> None:
    """``register`` 装饰器写入类属性并完成注册，测试后清理。"""
    name = "_pytest_dummy_operator"

    @TTARegistry.register(name, "测试算子", "光度", 0.1, {"x": ("参数", 0.0, 1.0, 0.1, "×")})
    class _DummyOperator(TTAOperator):
        def sample_params(self, rng: np.random.Generator) -> dict:
            return {"x": self._sample("x", rng)}

        def _apply(self, image: np.ndarray, params: dict) -> np.ndarray:
            work, dtype = self._prepare(image)
            return self._restore(work + np.float32(params["x"]) * 0.0, dtype)

    try:
        assert _DummyOperator.name == name
        assert _DummyOperator.label == "测试算子"
        assert _DummyOperator.category == "光度"
        assert _DummyOperator.distortion == pytest.approx(0.1)
        assert set(_DummyOperator.param_spec) == {"x"}
        assert TTARegistry.is_registered(name)
        instance = TTARegistry.create(name, {"x": 0.5})
        assert isinstance(instance, TTAOperator)
        assert public_params(instance.sample_params(tta_rng_factory())) == {"x": 0.5}
    finally:
        TTARegistry._operators.pop(name, None)
    assert not TTARegistry.is_registered(name)
    assert len(OPERATORS) == 21


def test_registry_register_rejects_non_operator_class() -> None:
    """``register`` 装饰非 TTAOperator 子类时抛 TypeError，且不污染注册表。"""

    class _NotAnOperator:
        pass

    with pytest.raises(TypeError):
        TTARegistry.register("_bad_operator", "坏算子", "光度", 0.1)(_NotAnOperator)
    assert not TTARegistry.is_registered("_bad_operator")


def tta_rng_factory() -> np.random.Generator:
    """供非 fixture 场景使用的确定性随机源。"""
    return np.random.default_rng(20240517)


# ===========================================================================
# 2. 参数规格与采样
# ===========================================================================
@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_param_spec_is_well_formed(name: str) -> None:
    """``param_spec`` 每项都是 ``(中文名, 最小值, 最大值, 步长, 单位)``。"""
    spec = TTARegistry.get(name).param_spec
    for key, item in spec.items():
        assert isinstance(key, str) and key
        assert len(item) == 5, f"{name}.{key} 必须是 5 元组"
        label, low, high, step, unit = item
        assert isinstance(label, str) and label
        assert isinstance(unit, str)
        assert float(low) <= float(high)
        assert float(step) > 0.0


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_sample_params_within_declared_range(name: str, tta_rng: np.random.Generator) -> None:
    """``sample_params`` 的公开参数全部落在声明范围内（± 参数按绝对值判定）。"""
    operator = new_operator(name)
    signed = set(getattr(operator, "signed_params", frozenset()))
    for _ in range(20):
        params = public_params(operator.sample_params(tta_rng))
        for key, item in operator.param_spec.items():
            low, high = float(item[1]), float(item[2])
            value = float(params[key])
            assert np.isfinite(value)
            if key in signed:
                assert low - 1e-9 <= abs(value) <= high + 1e-9
            else:
                assert low - 1e-9 <= value <= high + 1e-9


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_sample_params_keys_match_spec(name: str, tta_rng: np.random.Generator) -> None:
    """``sample_params`` 的公开键集合等于 ``param_spec`` 的键集合。"""
    operator = new_operator(name)
    params = operator.sample_params(tta_rng)
    assert set(public_params(params)) == set(operator.param_spec)
    assert isinstance(params, dict)


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_sample_params_is_deterministic(name: str) -> None:
    """同一随机种子两次采样结果完全一致。"""
    first = new_operator(name).sample_params(np.random.default_rng(7))
    second = new_operator(name).sample_params(np.random.default_rng(7))
    assert first == second


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_sample_params_json_serializable(name: str, tta_rng: np.random.Generator) -> None:
    """参数必须能直接写入 manifest（JSON 可序列化、非 NaN）。"""
    params = new_operator(name).sample_params(tta_rng)
    payload = json.dumps(params, ensure_ascii=False)
    assert isinstance(payload, str)
    for key, value in params.items():
        assert isinstance(value, (int, float, str, bool)) and not isinstance(value, bool)
        assert np.isfinite(float(value))
    assert json.loads(payload).keys() == params.keys()


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_user_override_is_used_verbatim(name: str, tta_rng: np.random.Generator) -> None:
    """用户固定的参数被直接采用（不再随机）。"""
    spec = TTARegistry.get(name).param_spec
    for key, item in spec.items():
        mid = (float(item[1]) + float(item[2])) / 2.0
        operator = new_operator(name, {key: mid})
        params = operator.sample_params(tta_rng)
        if name == "jpeg":
            # quality 是下界：实际质量在 [mid, 100] 内
            assert float(params[key]) >= mid - 0.51
            assert float(params[key]) <= 100.0
        else:
            assert float(params[key]) == pytest.approx(mid, abs=0.51)


@pytest.mark.parametrize(
    ("name", "key", "value", "expected"),
    [
        ("rotate", "angle", 999.0, 180.0),
        ("rotate", "angle", -999.0, -180.0),
        ("gamma", "gamma", 99.0, 3.0),
        ("gamma", "gamma", 0.0, 0.2),
        ("occlusion_rect", "count", 100.0, 8.0),
        ("scratch", "width", 0.0, 1.0),
        ("sensor_optical", "strength", 9.0, 3.0),
    ],
)
def test_override_out_of_range_is_clipped(
    name: str, key: str, value: float, expected: float, tta_rng: np.random.Generator
) -> None:
    """越界的用户固定值被裁剪到声明范围内。"""
    params = new_operator(name, {key: value}).sample_params(tta_rng)
    assert float(params[key]) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("name", ["rotate", "translate", "scale", "shear", "brightness", "contrast"])
def test_signed_parameters_take_both_signs(name: str) -> None:
    """标注“随机 ±”的参数在多次采样中同时出现正负值。"""
    operator = new_operator(name)
    key = next(iter(operator.param_spec))
    assert key in set(operator.signed_params)
    values = [float(operator.sample_params(np.random.default_rng(seed))[key]) for seed in range(200)]
    assert any(value > 0 for value in values)
    assert any(value < 0 for value in values)


def test_stochastic_parameters_actually_vary() -> None:
    """随机采样确实产生多组不同参数（不是常量）。"""
    operator = new_operator("rotate")
    values = {float(operator.sample_params(np.random.default_rng(seed))["angle"]) for seed in range(50)}
    assert len(values) > 5


# ===========================================================================
# 3. apply 通用行为
# ===========================================================================
@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_apply_preserves_shape_dtype_and_range(
    name: str, tta_image: np.ndarray, tta_rng: np.random.Generator
) -> None:
    """``apply`` 不改变形状/dtype，输出恒在 ``[0, 1]``。"""
    output = new_operator(name).apply(tta_image, tta_rng)
    assert isinstance(output, np.ndarray)
    assert output.shape == tta_image.shape
    assert output.dtype == tta_image.dtype == np.float32
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_apply_accepts_uint8_and_keeps_dtype(
    name: str, tta_image_u8: np.ndarray, tta_rng: np.random.Generator
) -> None:
    """``uint8`` 输入给出 ``uint8`` 输出，值域 ``[0, 255]``。"""
    output = new_operator(name).apply(tta_image_u8, tta_rng)
    assert output.shape == tta_image_u8.shape
    assert output.dtype == np.uint8
    assert int(output.min()) >= 0
    assert int(output.max()) <= 255


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_apply_never_returns_nan_or_inf(
    name: str, tta_image: np.ndarray, tta_rng: np.random.Generator
) -> None:
    """输出中不允许出现 NaN / Inf。"""
    output = new_operator(name).apply(tta_image, tta_rng)
    assert bool(np.isfinite(output).all())


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_apply_does_not_mutate_input(
    name: str, tta_image: np.ndarray, tta_rng: np.random.Generator
) -> None:
    """算子不得修改传入数组（float32 与 uint8 各查一次）。"""
    float_input = tta_image.copy()
    float_before = float_input.copy()
    output = new_operator(name).apply(float_input, tta_rng)
    assert np.array_equal(float_input, float_before)
    assert output is not float_input

    u8_input = np.clip(np.rint(tta_image * 255.0), 0, 255).astype(np.uint8)
    u8_before = u8_input.copy()
    new_operator(name).apply(u8_input, tta_rng)
    assert np.array_equal(u8_input, u8_before)


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_apply_is_deterministic(name: str, tta_image: np.ndarray) -> None:
    """同一种子 → 同一结果（两个独立实例）。"""
    first = new_operator(name).apply(tta_image, np.random.default_rng(31337))
    second = new_operator(name).apply(tta_image, np.random.default_rng(31337))
    assert np.array_equal(first, second)


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_apply_reproducible_from_last_params(
    name: str, tta_image: np.ndarray, tta_rng: np.random.Generator
) -> None:
    """``last_params`` 完整决定结果：用同一组参数重放必得同一图像。"""
    operator = new_operator(name)
    output = operator.apply(tta_image, tta_rng)
    replay = operator._apply(tta_image, dict(operator.last_params))
    assert np.array_equal(output, replay)


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_apply_records_last_params(
    name: str, tta_image: np.ndarray, tta_rng: np.random.Generator
) -> None:
    """``apply`` 把本次使用的完整参数写入 ``last_params``。"""
    operator = new_operator(name)
    assert operator.last_params == {}
    operator.apply(tta_image, tta_rng)
    assert set(public_params(operator.last_params)) == set(operator.param_spec)
    for key, value in operator.last_params.items():
        assert np.isfinite(float(value))


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_apply_handles_constant_images(
    name: str, tta_white: np.ndarray, tta_black: np.ndarray, tta_rng: np.random.Generator
) -> None:
    """全白 / 全黑输入不崩溃，输出仍满足形状、dtype 与值域约束。"""
    for image in (tta_white, tta_black):
        output = new_operator(name).apply(image, tta_rng)
        assert output.shape == image.shape
        assert output.dtype == image.dtype
        assert bool(np.isfinite(output).all())
        assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0


def test_apply_rejects_non_array(tta_rng: np.random.Generator) -> None:
    """非 ndarray 输入抛 TypeError。"""
    with pytest.raises(TypeError):
        new_operator("rotate").apply([[0.0, 1.0], [1.0, 0.0]], tta_rng)  # type: ignore[arg-type]


def test_apply_rejects_non_grayscale(tta_rng: np.random.Generator) -> None:
    """非二维（彩色）输入抛 ValueError。"""
    with pytest.raises(ValueError):
        new_operator("rotate").apply(np.zeros((8, 8, 3), dtype=np.float32), tta_rng)


def test_apply_sanitizes_non_finite_input(tta_rng: np.random.Generator) -> None:
    """输入中的 NaN/Inf 被清理，输出依然有限。"""
    image = make_image(32, 32)
    image[0, 0] = np.nan
    image[1, 1] = np.inf
    output = new_operator("brightness").apply(image, tta_rng)
    assert bool(np.isfinite(output).all())


# ===========================================================================
# 4. describe()
# ===========================================================================
def test_describe_example_format() -> None:
    """契约示例：角度 7.5° → ``旋转(角度=7.5°)``。"""
    operator = new_operator("rotate", {"angle": 7.5})
    operator.apply(make_image(32, 32), np.random.default_rng(0))
    assert operator.describe() == "旋转(角度=7.5°)"


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_describe_contains_label_and_parameter_labels(
    name: str, tta_image: np.ndarray, tta_rng: np.random.Generator
) -> None:
    """``describe`` 以中文名开头，并包含全部参数的中文名。"""
    operator = new_operator(name)
    operator.apply(tta_image, tta_rng)
    text = operator.describe()
    assert text.startswith(operator.label)
    for item in operator.param_spec.values():
        assert item[0] in text


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_describe_before_apply_returns_label(name: str) -> None:
    """未调用 ``apply`` 时 ``describe`` 退化为中文名，不抛异常。"""
    operator = new_operator(name)
    assert operator.describe() == operator.label


# ===========================================================================
# 5. 几何算子
# ===========================================================================
@pytest.mark.parametrize("name", GEOMETRIC_NAMES)
def test_geometric_operators_keep_size(
    name: str, tta_image: np.ndarray, tta_rng: np.random.Generator
) -> None:
    """几何算子变换后仍保持原图尺寸。"""
    output = new_operator(name).apply(tta_image, tta_rng)
    assert output.shape == tta_image.shape


@pytest.mark.parametrize("name", GEOMETRIC_NAMES)
def test_geometric_operators_with_fixed_params_change_or_keep_size(
    name: str, tta_image: np.ndarray
) -> None:
    """给定确定参数时几何算子输出尺寸不变（角度/位移非零时确实发生变换）。"""
    spec = TTARegistry.get(name).param_spec
    params = {key: float(item[2]) for key, item in spec.items()}
    operator = new_operator(name, params)
    output = operator._apply(tta_image, operator.sample_params(np.random.default_rng(5)))
    assert output.shape == tta_image.shape
    assert bool(np.isfinite(output).all())
    if name != "elastic":  # elastic 需 alpha>0，这里取上界故必然改变
        assert not np.array_equal(output, tta_image)


@pytest.mark.parametrize("name", GEOMETRIC_NAMES)
def test_geometric_operators_fill_with_background_not_black(
    name: str, tta_white: np.ndarray
) -> None:
    """边界用 ``BORDER_REPLICATE``：全白图变换后仍是全白（无黑边）。"""
    spec = TTARegistry.get(name).param_spec
    params = {key: float(item[2]) * 0.5 for key, item in spec.items()}
    operator = new_operator(name, params)
    output = operator._apply(tta_white, operator.sample_params(np.random.default_rng(11)))
    assert float(output.min()) >= 1.0 - 1e-6


def test_rotate_180_equals_flip(tta_image: np.ndarray) -> None:
    """180° 旋转等价于上下左右翻转。"""
    operator = new_operator("rotate", {"angle": 180.0})
    output = operator._apply(tta_image, {"angle": 180.0})
    assert np.allclose(output, tta_image[::-1, ::-1], atol=1e-4)


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("rotate", {"angle": 0.0}),
        ("translate", {"shift": 0.0}),
        ("scale", {"factor": 0.0}),
        ("shear", {"amount": 0.0}),
        ("elastic", {"alpha": 0.0, "sigma": 20.0}),
        ("blur", {"sigma": 0.0}),
        ("motion_blur", {"length": 1}),
        ("gaussian_noise", {"sigma": 0.0}),
        ("salt_pepper", {"amount": 0.0}),
        ("brightness", {"delta": 0.0}),
        ("contrast", {"factor": 0.0}),
        ("gamma", {"gamma": 1.0}),
    ],
)
def test_zero_strength_parameters_are_identity(
    name: str, params: dict, tta_image: np.ndarray
) -> None:
    """零强度参数应得到恒等变换（不引入任何额外退化）。"""
    operator = new_operator(name, params)
    sampled = operator.sample_params(np.random.default_rng(3))
    output = operator._apply(tta_image, sampled)
    assert np.allclose(output, tta_image, atol=1e-6)


# ===========================================================================
# 6. 遮挡算子
# ===========================================================================
@pytest.mark.parametrize("name", OCCLUSION_NAMES)
def test_occlusion_fills_background_not_black(name: str, tta_image: np.ndarray) -> None:
    """遮挡区域填充背景（高分位灰度），不产生纯黑块。"""
    operator = new_operator(name, {"count": 4, "area": 0.06})
    params = operator.sample_params(np.random.default_rng(17))
    output = operator._apply(tta_image, params)
    assert output.shape == tta_image.shape
    assert float(output.min()) >= float(tta_image.min()) - 1e-6
    assert np.count_nonzero(output < 0.05) <= np.count_nonzero(tta_image < 0.05)


@pytest.mark.parametrize("name", OCCLUSION_NAMES)
def test_occlusion_actually_covers_something(name: str, tta_image: np.ndarray) -> None:
    """遮挡算子必须真的改变图像（遮挡区域非空且更接近背景）。"""
    operator = new_operator(name, {"count": 3, "area": 0.05})
    params = operator.sample_params(np.random.default_rng(23))
    output = operator._apply(tta_image, params)
    changed = ~np.isclose(output, tta_image, atol=1e-6)
    assert changed.mean() > 0.005
    assert float(output[changed].mean()) > float(tta_image[changed].mean())


@pytest.mark.parametrize("name", OCCLUSION_NAMES)
def test_occlusion_params_effect_is_monotonic(name: str, tta_image: np.ndarray) -> None:
    """遮挡比例越大，被改变的像素越多。"""
    small = new_operator(name, {"count": 1, "area": 0.01})
    large = new_operator(name, {"count": 6, "area": 0.25})
    small_params = small.sample_params(np.random.default_rng(29))
    large_params = large.sample_params(np.random.default_rng(29))
    small_changed = (~np.isclose(small._apply(tta_image, small_params), tta_image, atol=1e-6)).mean()
    large_changed = (~np.isclose(large._apply(tta_image, large_params), tta_image, atol=1e-6)).mean()
    assert large_changed > small_changed


# ===========================================================================
# 7. 传感器算子
# ===========================================================================
@pytest.mark.parametrize("name", SENSOR_NAMES)
def test_sensor_zero_strength_is_exact_passthrough(name: str, tta_image: np.ndarray) -> None:
    """``strength=0`` 时传感器算子必须逐像素完全一致（直通）。"""
    operator = new_operator(name, {"strength": 0.0})
    params = operator.sample_params(np.random.default_rng(31))
    assert float(params["strength"]) == 0.0
    output = operator._apply(tta_image, params)
    assert np.array_equal(output, tta_image)


@pytest.mark.parametrize("name", SENSOR_NAMES)
def test_sensor_changes_image_at_full_strength(name: str, tta_image: np.ndarray) -> None:
    """``strength=3`` 时图像被明显改变，但仍在值域内、有限。"""
    operator = new_operator(name, {"strength": 3.0})
    params = operator.sample_params(np.random.default_rng(37))
    output = operator._apply(tta_image, params)
    assert not np.array_equal(output, tta_image)
    assert bool(np.isfinite(output).all())
    assert 0.0 <= float(output.min()) and float(output.max()) <= 1.0


# ===========================================================================
# 8. create_operators 与包导出
# ===========================================================================
def test_create_operators_keeps_order() -> None:
    """``create_operators`` 严格保持传入顺序。"""
    names = ["sensor_thermal", "rotate", "jpeg", "blur", "rotate"]
    operators = create_operators(names)
    assert [operator.name for operator in operators] == names
    assert all(isinstance(operator, TTAOperator) for operator in operators)


def test_create_operators_applies_params_mapping() -> None:
    """``params`` 映射按注册名分配到对应算子。"""
    operators = create_operators(
        ["rotate", "blur"], {"rotate": {"angle": 10.0}, "blur": {"sigma": 1.5}}
    )
    assert operators[0].params == {"angle": 10.0}
    assert operators[1].params == {"sigma": 1.5}
    assert float(operators[0].sample_params(np.random.default_rng(0))["angle"]) == pytest.approx(10.0)
    assert float(operators[1].sample_params(np.random.default_rng(0))["sigma"]) == pytest.approx(1.5)


def test_create_operators_unknown_name_raises_keyerror() -> None:
    """含未知名时抛 KeyError，错误信息列出可用算子。"""
    with pytest.raises(KeyError) as excinfo:
        create_operators(["rotate", "not_an_operator"])
    assert "not_an_operator" in str(excinfo.value)
    assert "blur" in str(excinfo.value)


def test_create_operators_empty_returns_empty_list() -> None:
    """空序列返回空列表。"""
    assert create_operators([]) == []


def test_create_operators_all_names_works(tta_image: np.ndarray) -> None:
    """一次性实例化全部 21 个算子并各施加一次，全部满足约束。"""
    operators = create_operators(list(OPERATORS))
    assert len(operators) == 21
    rng = np.random.default_rng(101)
    for operator in operators:
        output = operator.apply(tta_image, rng)
        assert output.shape == tta_image.shape
        assert output.dtype == np.float32
        assert bool(np.isfinite(output).all())


def test_package_exports_public_api() -> None:
    """``synthfp.plugins.tta`` 导出契约要求的四个公开符号。"""
    assert set(tta_pkg.__all__) == {"TTAOperator", "TTARegistry", "create_operators", "OPERATORS"}
    for symbol in tta_pkg.__all__:
        assert hasattr(tta_pkg, symbol)
    assert tta_pkg.OPERATORS is OPERATORS
    assert tta_pkg.TTARegistry is TTARegistry


def test_importing_package_registers_operators() -> None:
    """导入包即完成注册（无需显式 import operators 子模块）。"""
    assert len(TTARegistry.all_names()) == 21
    from synthfp.plugins.tta.operators import OPERATORS as operators_module_names

    assert list(operators_module_names) == list(EXPECTED_NAMES)


# ===========================================================================
# 9. 依赖与实现约束（静态检查）
# ===========================================================================
def test_tta_source_files_exist() -> None:
    """TTA 包至少包含 __init__/base/registry/operators 四个文件。"""
    assert {path.name for path in TTA_SOURCES} >= {
        "__init__.py",
        "base.py",
        "registry.py",
        "operators.py",
    }


def test_tta_only_imports_numpy_cv2_and_stdlib() -> None:
    """硬性依赖约束：只允许 numpy、cv2 与标准库。"""
    for path in TTA_SOURCES:
        for root in imported_roots(path):
            assert root in ALLOWED_IMPORT_ROOTS, f"{path.name} 引入了未授权依赖: {root}"


def test_tta_does_not_import_forbidden_packages() -> None:
    """显式禁止 scipy / scikit-image / matplotlib / pandas / click / tqdm / PySide6。"""
    for path in TTA_SOURCES:
        roots = {root.lower() for root in imported_roots(path)}
        assert not (roots & FORBIDDEN_IMPORT_ROOTS), f"{path.name} 引入了禁用依赖"


def test_tta_never_uses_global_random_state() -> None:
    """静态检查：实现中不得调用 ``np.random.*`` 全局函数（仅允许 ``default_rng``）。"""
    for path in TTA_SOURCES:
        text = path.read_text(encoding="utf-8")
        hits = GLOBAL_RANDOM_RE.findall(text)
        assert not hits, f"{path.name} 使用了全局随机函数: {hits}"


def test_helpers_produce_valid_random_field() -> None:
    """``random_field`` 生成零均值、单位标准差、有限值的平滑随机场。"""
    from synthfp.plugins.tta.base import random_field

    field = random_field((64, 64), 8.0, np.random.default_rng(5))
    assert field.shape == (64, 64)
    assert field.dtype == np.float32
    assert bool(np.isfinite(field).all())
    assert abs(float(field.mean())) < 0.5
    assert 0.5 < float(field.std()) < 1.5


def test_background_level_estimates_white_background() -> None:
    """``background_level`` 对白背景指纹图给出接近 1.0 的背景值。"""
    from synthfp.plugins.tta.base import background_level

    image = make_image(64, 64)
    assert float(background_level(image)) == pytest.approx(1.0, abs=0.05)
    assert float(background_level(np.zeros((8, 8), dtype=np.float32))) == pytest.approx(0.0)
