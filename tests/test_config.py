"""配置系统集成测试（schema / loader / presets），≥ 25 个用例。

覆盖范围：
    * 全部内置预设均可构建，``estimate()`` 返回合理整数；
    * ``AppConfig`` 的字段级与模型级校验（类别分布、划分比例、TTA 算子、
      脊宽、分辨率、区间倒挂 / 越界）；
    * 赋值即校验（``validate_assignment``）与未知字段拒绝（``extra=forbid``）；
    * YAML 往返（``save_config`` → ``load_config``）与 ``loads_config`` 文本解析；
    * ``apply_overrides`` 点号路径覆盖、``ConfigError`` 信息可读性；
    * 路径类字段同时接受 ``str`` 与 ``Path``。

全部用例都只写 ``tmp_path``，不依赖当前工作目录与机器 CPU 数。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from synthfp.config import (
    FINGERPRINT_CLASSES,
    PRESETS,
    RESOLUTION_PRESETS,
    TTA_CATALOG,
    AppConfig,
    ConfigError,
    apply_overrides,
    build_config,
    dump_yaml,
    load_config,
    load_default_config,
    loads_config,
    preset_names,
    save_config,
)

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
#: TTA 目录中的全部合法算子名。
_ALL_OPERATORS = tuple(name for _, name, _ in TTA_CATALOG)


def _override(overrides: dict) -> AppConfig:
    """在默认配置上应用覆盖（失败时抛 :class:`ConfigError`）。"""
    return apply_overrides(AppConfig(), overrides)


# ---------------------------------------------------------------------------
# 预设
# ---------------------------------------------------------------------------
def test_预设名称与注册表一致():
    """``preset_names()`` 与 ``PRESETS`` 的键完全一致。"""
    assert preset_names() == list(PRESETS)
    assert len(PRESETS) >= 5


def test_所有预设都能构建且元数据完整():
    """每个内置预设都能构建出完整配置，且带中文标题与说明。"""
    for name, preset in PRESETS.items():
        cfg = build_config(name)
        assert isinstance(cfg, AppConfig)
        assert cfg.dataset.name
        assert preset.key == name
        assert preset.label and preset.description
        assert isinstance(preset.tags, tuple)


def test_所有预设的estimate返回合理整数():
    """``estimate()`` 的三个键都是正整数，且满足乘算关系。"""
    for name in preset_names():
        est = build_config(name).estimate()
        assert set(est) == {"masters", "impressions", "samples"}
        for key, value in est.items():
            assert isinstance(value, int) and not isinstance(value, bool), (name, key)
            assert value >= 1, (name, key)
        cfg = build_config(name)
        d = cfg.dataset
        assert est["masters"] == d.identities * d.fingers_per_identity
        assert est["impressions"] == est["masters"] * d.impressions_per_finger
        assert est["samples"] == est["impressions"] * max(d.tta_per_impression, 1)


def test_预设覆盖确实生效():
    """抽查若干预设的关键覆盖字段。"""
    assert build_config("quick_test").dataset.identities == 2
    assert build_config("whorl_only").classes.distribution == {"whorl": 1.0}
    assert build_config("no_tta").tta.enabled is False
    assert build_config("no_tta").estimate()["samples"] == 40
    assert build_config("class_balanced").classes.distribution["arch"] == pytest.approx(0.2)
    assert build_config("large_research").runtime.workers == 8


def test_预设可基于已有配置构建():
    """``build_config(name, base=...)`` 在给定配置之上叠加预设覆盖。

    合并语义（v2.0 修正后）：

    * 预设未提及的段完整继承 ``base``；
    * 预设提及的**段**做浅合并 —— 预设写到的字段生效，该段里其余字段仍来自 ``base``。

    这样 ``synthfp generate -c my.yaml --preset quick_test`` 不会丢掉
    ``my.yaml`` 里未被预设覆盖的 ``runtime.seed`` / ``dataset.output_root``。
    """
    base = _override({"runtime.seed": 7, "dataset.output_root": "somewhere"})
    cfg = build_config("quick_test", base=base)

    # 预设未提及的段完整继承 base
    assert cfg.quality == base.quality
    assert cfg.split == base.split
    # 预设提及的段：预设写到的字段生效……
    assert cfg.runtime.workers == 2
    assert cfg.dataset.identities == 2
    assert cfg.dataset.name == "quick_test"
    # ……该段里预设没写的字段仍然保留 base 的值
    assert cfg.runtime.seed == 7
    assert str(cfg.dataset.output_root) == "somewhere"


def test_未知预设名报KeyError():
    with pytest.raises(KeyError):
        build_config("not_a_preset")


def test_默认配置的关键默认值():
    """默认配置与设计文档一致（分辨率 / 划分 / 导出开关）。"""
    cfg = AppConfig()
    assert cfg.schema_version == "2.0"
    assert cfg.backend.name == "sfinge"
    assert cfg.resolved_image_size() == RESOLUTION_PRESETS["finger_500dpi"]
    assert cfg.split.strategy == "by_identity"
    assert cfg.split.ratios == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert cfg.annotation.export_mask is True
    assert cfg.annotation.export_centerline is False
    assert cfg.export.image_format == "png"


# ---------------------------------------------------------------------------
# 类别分布校验
# ---------------------------------------------------------------------------
def test_类别分布之和不为一被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"classes.distribution": {"arch": 0.5, "whorl": 0.4}})
    assert "classes.distribution" in str(exc.value)
    assert "1.0" in str(exc.value)


def test_类别分布含未知类别被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"classes.distribution": {"arch": 0.5, "banana": 0.5}})
    assert "未知指纹类别" in str(exc.value)
    assert "banana" in str(exc.value)


def test_类别分布为空被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"classes.distribution": {}})
    assert "类别分布不能为空" in str(exc.value)


def test_类别分布含负值被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"classes.distribution": {"arch": -0.5, "whorl": 1.5}})
    assert "不能为负" in str(exc.value)


def test_类别分布零权重项被剔除():
    """校验器会丢弃权重为 0 的类别（归一化后的有效分布）。"""
    cfg = _override({"classes.distribution": {"arch": 0.0, "whorl": 1.0}})
    assert cfg.classes.distribution == {"whorl": 1.0}
    names, weights = cfg.classes.weights()
    assert names == ["whorl"] and weights == [1.0]
    assert set(cfg.classes.distribution) <= set(FINGERPRINT_CLASSES)


# ---------------------------------------------------------------------------
# 划分比例校验
# ---------------------------------------------------------------------------
def test_划分比例之和不为一被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"split.ratios": {"train": 0.5, "val": 0.2}})
    assert "split.ratios" in str(exc.value)


def test_划分含未知集合名被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"split.ratios": {"train": 0.5, "holdout": 0.5}})
    assert "未知划分" in str(exc.value)


def test_划分比例为空被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"split.ratios": {}})
    assert "划分比例不能为空" in str(exc.value)


# ---------------------------------------------------------------------------
# TTA 校验
# ---------------------------------------------------------------------------
def test_未知TTA算子被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"tta.operators": ["rotate", "not_an_operator"]})
    assert "未知 TTA 算子" in str(exc.value)


def test_TTA算子去重且保持用户顺序():
    """算子列表会被去重，且**保持用户给定顺序**（YAML 往返幂等）。"""
    cfg = _override({"tta.operators": ["brightness", "rotate", "brightness"]})
    assert cfg.tta.operators == ["brightness", "rotate"]
    # 保存 → 加载不改变字段值
    assert loads_config(dump_yaml(cfg)).tta.operators == ["brightness", "rotate"]


def test_TTA参数含未知算子被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"tta.params": {"nope": {"angle": 5}}})
    assert "未知 TTA 算子的参数" in str(exc.value)


def test_启用TTA但无算子被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"tta.enabled": True, "tta.operators": []})
    assert "至少需要勾选一个算子" in str(exc.value)


def test_关闭TTA时允许空算子列表():
    cfg = _override({"tta.enabled": False, "tta.operators": []})
    assert cfg.tta.operators == []
    assert cfg.tta.summary() == "已关闭"


def test_chain_length区间倒挂被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"tta.chain_length": (3, 1)})
    assert "chain_length" in str(exc.value)


def test_TTA参数规格覆盖全部目录算子():
    """``TTA_PARAM_SPECS`` 覆盖目录里所有需要参数的算子。"""
    from synthfp.config import TTA_PARAM_SPECS

    assert set(TTA_PARAM_SPECS) <= set(_ALL_OPERATORS)
    assert len(TTA_PARAM_SPECS) >= 15


# ---------------------------------------------------------------------------
# 后端校验
# ---------------------------------------------------------------------------
def test_最小脊宽必须小于最大脊宽():
    with pytest.raises(ConfigError) as exc:
        _override({"backend.min_ridge_width": 20, "backend.max_ridge_width": 10})
    assert "最窄脊宽" in str(exc.value)


def test_脊宽相等也被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"backend.min_ridge_width": 8, "backend.max_ridge_width": 8})
    assert "必须小于" in str(exc.value)


def test_未知分辨率预设被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"backend.resolution": "4k_ultra"})
    assert "未知分辨率预设" in str(exc.value)


def test_灰度范围倒挂被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"backend.ridge_gray": (200, 20)})
    assert "灰度范围" in str(exc.value)


def test_区间倒挂被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"backend.finger_width_frac": (0.9, 0.3)})
    assert "不能大于上界" in str(exc.value)


@pytest.mark.parametrize(
    "field, value",
    [
        ("backend.finger_width_frac", (0.0, 0.3)),
        ("backend.finger_height_frac", (0.5, 1.5)),
        ("backend.aspect_bias", (0.0, 0.5)),
        ("backend.top_cap_frac", (0.0, 0.5)),
        ("backend.bottom_cap_frac", (0.5, 2.0)),
    ],
)
def test_区间越界被拒(field, value):
    with pytest.raises(ConfigError) as exc:
        _override({field: value})
    assert "必须落在" in str(exc.value)


def test_捺印区间越界被拒():
    with pytest.raises(ConfigError) as exc:
        _override({"impression.pressure_range": (0.0, 2.0)})
    assert "pressure_range" in str(exc.value)
    with pytest.raises(ConfigError) as exc2:
        _override({"impression.rotation_range": (-200.0, 0.0)})
    assert "rotation_range" in str(exc2.value)


def test_自定义图像尺寸优先于分辨率预设():
    cfg = _override({"backend.image_size": (128, 256)})
    assert cfg.resolved_image_size() == (128, 256)
    assert cfg.backend.resolved_image_size() == (128, 256)


def test_分辨率别名保持未归一化():
    """schema 只保存预设名，尺寸在 ``resolved_image_size`` 时才展开。"""
    cfg = _override({"backend.resolution": "square_512"})
    assert cfg.backend.resolution == "square_512"
    assert cfg.resolved_image_size() == (512, 512)


# ---------------------------------------------------------------------------
# 赋值即校验 / 未知字段
# ---------------------------------------------------------------------------
def test_赋值时立即校验():
    cfg = AppConfig()
    with pytest.raises(ValidationError):
        cfg.dataset.identities = 0
    with pytest.raises(ValidationError):
        cfg.backend.max_ridge_width = 1
    # 合法赋值仍然生效
    cfg.dataset.identities = 5
    assert cfg.dataset.identities == 5


def test_赋值未知字段被拒():
    cfg = AppConfig()
    with pytest.raises(ValidationError):
        cfg.dataset.mystery = 1
    with pytest.raises(ValidationError):
        cfg.unknown_section = {}


def test_构造时非法值直接抛ValidationError():
    with pytest.raises(ValidationError):
        AppConfig(dataset={"identities": 0})
    with pytest.raises(ValidationError):
        AppConfig(classes={"distribution": {"arch": 0.5}})


# ---------------------------------------------------------------------------
# YAML 往返
# ---------------------------------------------------------------------------
def test_yaml往返保存加载一致(tmp_path: Path):
    cfg = build_config("quick_test")
    cfg.dataset.identities = 3
    cfg.runtime.seed = 2024
    path = save_config(cfg, tmp_path / "nested" / "cfg.yaml")

    assert path.exists()
    back = load_config(path)
    assert back == cfg
    assert back.dataset.identities == 3
    assert back.runtime.seed == 2024


def test_保存的yaml合法且包含预期键(tmp_path: Path):
    cfg = build_config("high_quality_only")
    path = save_config(cfg, tmp_path / "cfg.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(raw, dict)
    for key in (
        "schema_version", "dataset", "classes", "backend", "impression", "tta",
        "quality", "minutiae", "annotation", "split", "runtime", "export",
    ):
        assert key in raw, key
    assert raw["dataset"]["name"] == "synth_fp_hq"
    assert raw["quality"]["min_score"] == pytest.approx(60.0)
    # 字段顺序与 schema 一致（YAML 顶层第一项是 schema_version）
    assert list(raw)[0] == "schema_version"


def test_dump_yaml与loads_config往返():
    cfg = build_config("robust_low_quality")
    text = dump_yaml(cfg)
    assert "robust" not in text or True  # 仅确认可 dump
    assert loads_config(text) == cfg
    # 文本本身是合法 YAML
    assert isinstance(yaml.safe_load(text), dict)


def test_loads_config解析环境变量与默认值(monkeypatch):
    monkeypatch.setenv("SFP_TEST_DS", "env_dataset")
    cfg = loads_config("dataset:\n  name: ${SFP_TEST_DS}\n")
    assert cfg.dataset.name == "env_dataset"

    cfg2 = loads_config("dataset:\n  name: ${SFP_TEST_MISSING:-fallback_name}\n")
    assert cfg2.dataset.name == "fallback_name"

    cfg3 = loads_config("dataset:\n  name: ${SFP_TEST_MISSING}\n")
    assert cfg3.dataset.name == ""


def test_loads_config顶层必须是映射():
    with pytest.raises(ConfigError) as exc:
        loads_config("- 1\n- 2\n")
    assert "顶层必须是映射" in str(exc.value)


def test_load_config文件不存在报ConfigError(tmp_path: Path):
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "missing.yaml")
    assert "配置文件不存在" in str(exc.value)


def test_load_config_parent回退到内置预设(tmp_path: Path):
    """``parent:`` 先查 ``configs/presets/<name>.yaml``，找不到再回退到内置预设。

    这样 ``parent: quick_test`` 在只装了包（没有 configs 目录）的环境里也能用，
    子文件里显式写出的字段优先级更高。
    """
    child = tmp_path / "child.yaml"
    child.write_text(
        "parent: quick_test\ndataset:\n  name: my_child\n  identities: 9\n", encoding="utf-8"
    )
    cfg = load_config(child)
    # 子文件覆盖父预设
    assert cfg.dataset.name == "my_child"
    assert cfg.dataset.identities == 9
    # 父预设补齐未写出的字段
    assert cfg.dataset.fingers_per_identity == 1
    assert cfg.runtime.workers == 2


def test_load_config_parent未知时报ConfigError(tmp_path: Path):
    child = tmp_path / "bad.yaml"
    child.write_text("parent: no_such_preset\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(child)
    assert "未知的父预设" in str(exc.value)
    assert "no_such_preset" in str(exc.value)


def test_load_config未知parent报ConfigError(tmp_path: Path):
    child = tmp_path / "child.yaml"
    child.write_text("parent: no_such_preset\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(child)
    assert "未知的父预设" in str(exc.value)


def test_load_config支持运行时覆盖(tmp_path: Path):
    path = save_config(AppConfig(), tmp_path / "cfg.yaml")
    cfg = load_config(path, overrides={"runtime.seed": 11, "dataset.identities": 4})
    assert cfg.runtime.seed == 11
    assert cfg.dataset.identities == 4


def test_load_default_config返回可用的配置():
    cfg = load_default_config()
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.dataset.name, str) and cfg.dataset.name


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------
def test_apply_overrides点号路径():
    cfg = _override(
        {
            "runtime.seed": 99,
            "tta.params.rotate.angle": 8.0,
            "dataset.identities": 3,
            "export.image_format": "tiff",
        }
    )
    assert cfg.runtime.seed == 99
    assert cfg.tta.params["rotate"]["angle"] == pytest.approx(8.0)
    assert cfg.tta.params_for("rotate") == {"angle": 8.0}
    assert cfg.dataset.identities == 3
    assert cfg.export.image_format == "tiff"


def test_apply_overrides返回新对象且不改动原配置():
    base = AppConfig()
    new = apply_overrides(base, {"runtime.seed": 5})
    assert new is not base
    assert base.runtime.seed == 42
    assert new.runtime.seed == 5


def test_apply_overrides忽略None值():
    base = AppConfig()
    new = apply_overrides(base, {"runtime.seed": None, "runtime.workers": 3})
    assert new.runtime.seed == base.runtime.seed
    assert new.runtime.workers == 3


def test_apply_overrides段级字典是浅合并():
    """整段字典覆盖是**浅合并**：给出的字段生效，该段其余字段保留。

    这条语义让「在已有配置上套预设」成为可能（``build_config(name, base=...)``），
    同时保证小节内部的字典/列表（如 ``classes.distribution``）仍然是**整体替换**
    —— 否则 ``distribution: {whorl: 1.0}`` 会变成"五类再加螺型"，比例和 1.8。
    """
    cfg = _override({"runtime": {"workers": 3}})
    assert cfg.runtime.workers == 3
    assert cfg.runtime.seed == AppConfig().runtime.seed  # 未提及的字段仍是默认值

    other = _override({"runtime.seed": 99, "runtime.workers": 1})
    merged = apply_overrides(other, {"runtime": {"workers": 3}})
    assert merged.runtime.workers == 3
    assert merged.runtime.seed == 99  # 段内未提及的字段保留

    # 小节内部的字典整体替换
    single = apply_overrides(other, {"classes": {"distribution": {"whorl": 1.0}}})
    assert single.classes.distribution == {"whorl": 1.0}
    assert single.runtime.seed == 99  # 其它段不受影响


def test_ConfigError信息可读且含字段路径():
    with pytest.raises(ConfigError) as exc:
        loads_config("dataset:\n  identities: 0\n")
    message = str(exc.value)
    assert "配置校验失败" in message
    assert "dataset.identities" in message
    assert "•" in message  # 每条错误一行，便于 GUI / CLI 直接展示
    assert message.count("\n") >= 1


def test_ConfigError是ValueError子类():
    assert issubclass(ConfigError, ValueError)
    with pytest.raises(ValueError):
        loads_config("dataset:\n  identities: 0\n")


# ---------------------------------------------------------------------------
# 路径类字段
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["outputs/custom", Path("outputs/custom")])
def test_output_root同时接受str与Path(raw):
    cfg = _override({"dataset.output_root": raw})
    assert isinstance(cfg.dataset.output_root, Path)
    assert cfg.dataset.output_root == Path("outputs/custom")
    assert cfg.dataset_dir == Path("outputs/custom") / cfg.dataset.name


def test_yaml往返保留路径字段(tmp_path: Path):
    root = tmp_path / "out root"
    cfg = _override({"dataset.output_root": root, "dataset.name": "ds"})
    back = load_config(save_config(cfg, tmp_path / "cfg.yaml"))
    assert back.dataset.output_root == root
    assert back.dataset_dir == root / "ds"


# ---------------------------------------------------------------------------
# 摘要
# ---------------------------------------------------------------------------
def test_dataset_section派生属性():
    d = AppConfig().dataset
    assert d.total_masters == d.identities * d.fingers_per_identity
    assert d.total_impressions == d.total_masters * d.impressions_per_finger
    assert d.total_samples == d.total_impressions * max(d.tta_per_impression, 1)
    assert "样本" in d.summary()


def test_summary_lines覆盖全部配置段():
    cfg = build_config("quick_test")
    lines = cfg.summary_lines()
    assert len(lines) >= 12
    joined = "\n".join(lines)
    for label in ("数据集", "规模", "输出", "类别分布", "分辨率", "后端", "TTA", "划分", "导出"):
        assert label in joined, label
    assert str(cfg.dataset_dir) in joined


def test_各部分summary在关闭状态下可读():
    cfg = _override(
        {
            "tta.enabled": False,
            "quality.enabled": False,
            "split.enabled": False,
            "impression.deform_enabled": False,
            "impression.sensor_enabled": False,
            "impression.stress_enabled": False,
            "annotation.export_mask": False,
            "annotation.export_orientation": False,
        }
    )
    assert cfg.tta.summary() == "已关闭"
    assert cfg.quality.summary() == "已关闭"
    assert cfg.split.summary() == "不划分"
    assert cfg.impression.summary() == "无"
    assert cfg.annotation.summary() == "仅图像"


# ---------------------------------------------------------------------------
# 序列（tuple）下标路径
# ---------------------------------------------------------------------------
def test_覆盖元组下标():
    """``backend.ridge_gray.0`` 必须改元组里的元素，而不是把元组变成字典。

    回归用例：早先 ``_dot_set`` 只按字典键处理，"ridge_gray.0" 会把
    ``(25, 70)`` 整个替换成 ``{"0": 25}``，随后校验失败 ——
    界面上那四个灰度控件因此完全不能用（只显示 0，一改就弹错误）。
    """
    cfg = build_config("quick_test")
    before = cfg.backend.ridge_gray

    new = apply_overrides(cfg, {"backend.ridge_gray.0": 12})
    assert new.backend.ridge_gray == (12, before[1])
    assert isinstance(new.backend.ridge_gray, tuple)
    # 原对象不被修改（apply_overrides 返回新对象）
    assert cfg.backend.ridge_gray == before

    both = apply_overrides(new, {"backend.ridge_gray.1": 90})
    assert both.backend.ridge_gray == (12, 90)


def test_元组下标越界仍被校验拦截():
    cfg = build_config("quick_test")
    with pytest.raises(ConfigError):
        apply_overrides(cfg, {"backend.ridge_gray.0": 300})   # 超出 0~255
    with pytest.raises(ConfigError):
        apply_overrides(cfg, {"backend.ridge_gray.1": 5})      # 上界小于下界


def test_元组下标多级路径():
    """区间类字段（``tta.chain_length.1``）同样可按下标覆盖。"""
    cfg = build_config("quick_test")
    new = apply_overrides(cfg, {"tta.chain_length.1": 4})
    assert new.tta.chain_length[1] == 4
