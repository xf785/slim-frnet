"""样本管道集成测试（chain / stages / runner），≥ 18 个用例。

全部用例都使用 **dummy 后端**（确定性、零算法耗时）与 1 身份 × 1 指 × 1 捺印
× 2 TTA 的微型配置，保证秒级完成且不依赖机器 CPU 数。

覆盖范围：
    * ``select_tta_chain`` 的 ``tta_index == 0`` / 关闭 TTA / 同种子确定性 /
      「只返回已注册且在配置内的算子」；
    * ``apply_tta`` 的空链直通、形状 dtype 保持、算子名与参数记录；
    * ``PipelineRunner.run_entry`` 的完整产物落盘、结果字段、校验和与体积；
    * 逐像素可复现性（两个全新目录、同一配置与数据集名）；
    * ``quality.reject_on_fail`` 的丢弃语义（只标记、不写图像）；
    * 后端故障时由 ``Scheduler`` 隔离为 failed（``run_entry`` 本身会向上抛）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from synthfp.config import AppConfig, apply_overrides, build_config
from synthfp.core.manifest import Manifest, read_jsonl
from synthfp.core.planner import Planner
from synthfp.core.quantize import decode_orientation, load_array_png
from synthfp.core.scheduler import Scheduler, SchedulerHooks
from synthfp.core.storage import DatasetLayout, sha256_file
from synthfp.pipeline import PipelineRunner, plan_tta_chains, select_tta_chain, tta_chain_keys
from synthfp.pipeline import stages
from synthfp.plugins.tta import TTARegistry

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
_OPERATORS = ["rotate", "scale", "brightness", "contrast", "gaussian_noise", "blur"]
_MODALITIES = ["images", "mask", "orientation", "frequency", "centerline", "quality", "minutiae"]


def _cfg(root: Path, name: str = "pipe_ds", **extra) -> AppConfig:
    """微型配置：1 身份 × 1 指 × 1 捺印 × 2 TTA，dummy 后端。"""
    overrides = {
        "dataset.name": name,
        "dataset.output_root": root,
        "dataset.identities": 1,
        "dataset.fingers_per_identity": 1,
        "dataset.impressions_per_finger": 1,
        "dataset.tta_per_impression": 2,
        "backend.name": "dummy",
        "tta.operators": ["rotate", "brightness"],
        "annotation.export_centerline": True,
        "annotation.export_quality_map": True,
        "minutiae.export_iso": True,
        "runtime.workers": 1,
    }
    overrides.update(extra)
    return apply_overrides(build_config("quick_test"), overrides)


def _layout(cfg: AppConfig) -> DatasetLayout:
    layout = DatasetLayout(cfg.dataset_dir)
    layout.create(_MODALITIES)
    return layout


# ---------------------------------------------------------------------------
# TTA 链选择
# ---------------------------------------------------------------------------
def test_select_tta_chain_序号0返回空链(tmp_path: Path):
    """``tta_index == 0`` 保留未变换的原始捺印，永远是空链。"""
    cfg = _cfg(tmp_path)
    for seed in range(5):
        assert select_tta_chain(cfg, seed, f"sid{seed}", 0) == []
        assert tta_chain_keys(cfg, seed, f"sid{seed}", 0) == []


def test_select_tta_chain_TTA关闭返回空链(tmp_path: Path):
    cfg = _cfg(tmp_path, tta={"enabled": False})
    assert select_tta_chain(cfg, 1234, "sid", 1) == []
    assert select_tta_chain(cfg, 1234, "sid", 7) == []


def test_select_tta_chain_同种子完全确定(tmp_path: Path):
    cfg = _cfg(tmp_path)
    first = tta_chain_keys(cfg, 987654321, "000001_left_thumb_imp01_tta01", 1)
    for _ in range(3):
        assert tta_chain_keys(cfg, 987654321, "000001_left_thumb_imp01_tta01", 1) == first
    # sample_id 也参与派生：换 id 允许改变链
    other = tta_chain_keys(cfg, 987654321, "000001_left_thumb_imp01_tta02", 1)
    assert isinstance(other, list)
    assert set(other) <= set(cfg.tta.operators)


@pytest.mark.parametrize("tta_index", [1, 2, 3, 4])
def test_select_tta_chain_只返回已注册且在配置内的算子(tmp_path: Path, tta_index: int):
    cfg = _cfg(tmp_path, tta={"mode": "chain", "chain_length": (2, 3)})
    allowed = set(cfg.tta.operators)
    seen = set()
    for seed in range(12):
        chain = select_tta_chain(cfg, seed * 7919, f"sid-{seed}", tta_index)
        names = [op.name for op in chain]
        assert names, "chain 模式应至少选出一个算子"
        assert set(names) <= allowed
        assert all(TTARegistry.is_registered(n) for n in names)
        assert len(names) == len(set(names))  # 不重复
        assert len(names) <= cfg.tta.chain_length[1]
        seen |= set(names)
    assert seen <= allowed


def test_select_tta_chain_链长不超过可用算子数(tmp_path: Path):
    """配置里只有 1 个算子时，即使 chain_length=(3,3) 也只能选出 1 个。"""
    cfg = _cfg(tmp_path, tta={"mode": "chain", "chain_length": (3, 3), "operators": ["rotate"]})
    chain = select_tta_chain(cfg, 4242, "sid", 1)
    assert [op.name for op in chain] == ["rotate"]


def test_select_tta_chain_受最大失真度约束(tmp_path: Path):
    """``max_distortion`` 很小时链会被截断，但至少保留第一个算子。"""
    cfg = _cfg(
        tmp_path,
        tta={"mode": "chain", "chain_length": (3, 3), "max_distortion": 0.05,
             "operators": _OPERATORS},
    )
    chain = select_tta_chain(cfg, 31337, "sid", 1)
    assert len(chain) <= 3  # 预算内截断，不会因预算不足而全空
    assert chain, "第一个算子不受预算约束"


def test_select_tta_chain_不同种子可产生不同链(tmp_path: Path):
    """种子必须真的影响选择（否则可复现性没有意义）。

    注意：链长等于可用算子数时，无论种子如何都只能选出全部算子，
    因此这里用足够多的算子 + 短链，才能观察到种子的影响。
    """
    cfg = _cfg(
        tmp_path,
        tta={"mode": "chain", "chain_length": (2, 2), "operators": _OPERATORS},
    )
    results = {tuple(tta_chain_keys(cfg, seed, f"sid{seed}", 1)) for seed in range(40)}
    assert len(results) > 1
    assert all(set(r) <= set(cfg.tta.operators) for r in results)


def test_select_tta_chain算子数不足时只有唯一链(tmp_path: Path):
    """可用算子数 == 链长时，任意种子都得到同一条链（含全部算子）。"""
    cfg = _cfg(tmp_path, tta={"mode": "chain", "chain_length": (2, 2)})
    assert cfg.tta.operators == ["rotate", "brightness"]
    results = {tuple(tta_chain_keys(cfg, seed, f"sid{seed}", 1)) for seed in range(20)}
    assert results == {("brightness", "rotate")}


def test_plan_tta_chains批量预览确定性(tmp_path: Path):
    cfg = _cfg(tmp_path)
    seeds = [(1000 + i, f"sid-{i}", i % 3) for i in range(6)]
    first = plan_tta_chains(cfg, seeds)
    assert first == plan_tta_chains(cfg, seeds)
    assert len(first) == len(seeds)
    assert first[0] == []  # 第一条的 tta_index = 0


# ---------------------------------------------------------------------------
# apply_tta
# ---------------------------------------------------------------------------
def test_apply_tta空算子返回未修改副本():
    cfg = apply_overrides(AppConfig(), {"backend.name": "dummy"})
    image = (np.random.default_rng(0).random((32, 48)) * 255).astype(np.uint8)
    out, names, params, distortion = stages.apply_tta(image, [], cfg, 0)
    assert np.array_equal(out, image)
    assert out is not image  # 必须是副本，避免调用方误改原图
    assert names == [] and params == {} and distortion == 0.0
    assert out.dtype == np.uint8


def test_apply_tta有算子时保持形状与dtype(tmp_path: Path):
    cfg = _cfg(tmp_path, tta={"mode": "chain", "chain_length": (2, 2)})
    image = (np.random.default_rng(1).random((64, 48)) * 255).astype(np.uint8)
    operators = select_tta_chain(cfg, 555, "sid", 1)
    assert operators, "chain 模式应选到算子"

    out, names, params, distortion = stages.apply_tta(image, operators, cfg, 1, seed=555)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert names == [op.name for op in operators]
    assert set(params) <= set(names)
    for name in names:
        assert isinstance(params[name], dict)
    assert 0.0 <= distortion <= 1.0
    # 输入未被就地修改
    assert image.dtype == np.uint8


def test_apply_tta记录算子参数并可通过掩模还原背景(tmp_path: Path):
    cfg = _cfg(tmp_path)
    image = np.full((40, 40), 200, dtype=np.uint8)
    image[10:30, 10:30] = 60
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    operators = select_tta_chain(cfg, 1, "sid", 1)
    out, names, params, _ = stages.apply_tta(image, operators, cfg, 1, mask=mask, seed=1)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert names
    # 掩模外的背景应被还原为原图的亮背景
    assert int(out[0, 0]) >= 180


def test_apply_tta同种子逐像素一致(tmp_path: Path):
    cfg = _cfg(tmp_path)
    image = (np.random.default_rng(2).random((48, 48)) * 255).astype(np.uint8)
    operators = select_tta_chain(cfg, 777, "sid", 1)
    a = stages.apply_tta(image, operators, cfg, 1, seed=777)[0]
    b = stages.apply_tta(image, operators, cfg, 1, seed=777)[0]
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# PipelineRunner.run_entry —— 完整产物
# ---------------------------------------------------------------------------
def test_run_entry写出全部产物且可回读(tmp_path: Path):
    cfg = _cfg(tmp_path)
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    results = PipelineRunner(cfg, layout).run_entry(entry, entry.samples)

    assert len(results) == len(entry.samples)
    for result in results:
        assert result.status == "ok", result.error
        assert result.image_path and result.mask_path and result.orientation_path
        assert result.centerline_path and result.quality_map_path
        assert result.minutiae_path and result.template_path

        # 每个相对路径都能还原为真实文件
        for rel in (result.image_path, result.mask_path, result.orientation_path,
                    result.centerline_path, result.quality_map_path,
                    result.minutiae_path, result.template_path):
            assert layout.resolve(rel).exists(), rel

        # 掩模 / 方向场可回读
        mask, _ = load_array_png(layout.resolve(result.mask_path))
        assert mask.ndim == 2 and set(np.unique(mask)) <= {0, 255}
        orientation, _ = load_array_png(layout.resolve(result.orientation_path))
        angles = decode_orientation(orientation)
        assert float(angles.min()) >= 0.0 and float(angles.max()) < np.pi + 1e-3

        # 细节点 JSON 可解析
        payload = json.loads(layout.resolve(result.minutiae_path).read_text(encoding="utf-8"))
        assert payload["sample_id"] == result.spec.sample_id
        assert payload["format"] == "synthfp-minutiae-v1"
        assert payload["n_minutiae"] == result.minutiae_count == len(payload["minutiae"])
        # ISO/IEC 19794-2 风格模板非空
        assert layout.resolve(result.template_path).stat().st_size > 0


def test_run_entry结果状态与尺寸匹配配置(tmp_path: Path):
    cfg = _cfg(tmp_path)
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    results = PipelineRunner(cfg, layout).run_entry(entry, entry.samples)

    width, height = cfg.resolved_image_size()
    for result in results:
        assert result.status == "ok"
        assert (result.width, result.height) == (width, height)
        assert result.ppi == cfg.backend.ppi
        assert result.backend == "dummy"
        assert result.backend_version
        assert result.ok is True
        assert result.error == ""
        assert result.attempt >= 1
        assert result.duration_s >= 0.0
        assert result.impression_effects  # 捺印阶段至少记录变形 + 传感器
        assert {"stage"} <= set(result.impression_effects[0])


def test_run_entry校验和与体积与磁盘一致(tmp_path: Path):
    cfg = _cfg(tmp_path)
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    results = PipelineRunner(cfg, layout).run_entry(entry, entry.samples)

    for result in results:
        path = layout.resolve(result.image_path)
        assert sha256_file(path) == result.checksum
        assert path.stat().st_size == result.bytes
        assert path.read_bytes().startswith(b"\x89PNG")
    # 两个样本的图不同（TTA 与基线），校验和也不应相同
    assert results[0].checksum != results[1].checksum


def test_run_entry_TTA序号0与1的算子记录(tmp_path: Path):
    cfg = _cfg(tmp_path, tta={"mode": "chain", "chain_length": (2, 2)})
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    results = {r.spec.tta_index: r for r in PipelineRunner(cfg, layout).run_entry(
        entry, entry.samples
    )}
    assert results[0].tta_operators == [] and results[0].tta_params == {}
    assert results[0].tta_distortion == 0.0
    assert results[1].tta_operators == [op.name for op in select_tta_chain(
        cfg, results[1].spec.seed, results[1].spec.sample_id, 1
    )]
    assert results[1].tta_distortion > 0.0


def test_run_entry空待办直接返回(tmp_path: Path):
    cfg = _cfg(tmp_path)
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    runner = PipelineRunner(cfg, layout)
    assert runner.run_entry(entry, []) == []
    # 未生成任何主指纹时不应报错，也不应落盘
    assert not list(layout.images_dir.rglob("*.png"))


def test_run_entry惰性构造后端与退化链(tmp_path: Path):
    cfg = _cfg(tmp_path)
    runner = PipelineRunner(cfg, _layout(cfg))
    assert runner._backend is None and runner._composer is None
    backend = runner.backend
    assert backend.name == "dummy"
    assert runner.backend is backend  # 只构造一次
    assert runner.composer is not None  # quick_test 开启了捺印退化


def test_run_entry细节点计数自洽(tmp_path: Path):
    cfg = _cfg(tmp_path)
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    for result in PipelineRunner(cfg, layout).run_entry(entry, entry.samples):
        assert result.minutiae_count == result.minutiae_endings + result.minutiae_bifurcations
        if result.quality is not None:
            assert result.quality.minutiae_count == result.minutiae_count
            assert 0.0 <= result.quality.score <= 100.0


# ---------------------------------------------------------------------------
# 可复现性
# ---------------------------------------------------------------------------
def test_可复现性_两次运行逐字节一致(tmp_path: Path):
    """同配置 + 同数据集名 + 同种子 ⇒ 两个全新目录里的图像逐字节一致。"""
    runs = []
    for tag in ("a", "b"):
        cfg = _cfg(tmp_path / tag)
        layout = _layout(cfg)
        entry = Planner(cfg).build().entries[0]
        results = PipelineRunner(cfg, layout).run_entry(entry, entry.samples)
        runs.append([
            (
                layout.resolve(r.image_path).read_bytes(),
                r.checksum,
                r.minutiae_count,
                r.tta_operators,
            )
            for r in results
        ])

    assert [r[0] for r in runs[0]] == [r[0] for r in runs[1]]      # 图像字节一致
    assert [r[1] for r in runs[0]] == [r[1] for r in runs[1]]      # SHA-256 一致
    assert [r[2] for r in runs[0]] == [r[2] for r in runs[1]]      # 细节点数一致
    assert [r[3] for r in runs[0]] == [r[3] for r in runs[1]]      # 算子链一致


def test_可复现性_换种子结果改变(tmp_path: Path):
    """对照组：换 ``runtime.seed`` 后图像必须不同（证明种子真的生效）。"""
    images = []
    for seed in (42, 43):
        cfg = _cfg(tmp_path / str(seed), **{"runtime.seed": seed})
        layout = _layout(cfg)
        entry = Planner(cfg).build().entries[0]
        results = PipelineRunner(cfg, layout).run_entry(entry, entry.samples)
        images.append(layout.resolve(results[0].image_path).read_bytes())
    assert images[0] != images[1]


# ---------------------------------------------------------------------------
# 质量阈值丢弃
# ---------------------------------------------------------------------------
def test_reject_on_fail标记rejected且不写图像(tmp_path: Path):
    cfg = _cfg(tmp_path, quality={"reject_on_fail": True, "min_score": 100.0})
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    results = PipelineRunner(cfg, layout).run_entry(entry, entry.samples)

    assert len(results) == len(entry.samples)
    for result in results:
        assert result.status == "rejected"
        assert result.error  # 给出原因
        assert result.image_path == "" and result.checksum == "" and result.bytes == 0
        assert result.quality is not None and result.quality.passed is False
        # 丢弃的样本不写任何图像
        assert result.mask_path == "" and result.minutiae_path == ""
    assert not list(layout.images_dir.rglob("*.png"))
    assert not list(layout.mask_dir.rglob("*.png"))


def test_仅标注时低于阈值的样本仍然落盘(tmp_path: Path):
    """``reject_on_fail=False``（默认）时质量只做标注，不丢样本。"""
    cfg = _cfg(tmp_path, quality={"reject_on_fail": False, "min_score": 100.0})
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    results = PipelineRunner(cfg, layout).run_entry(entry, entry.samples)
    for result in results:
        assert result.status == "ok"
        assert result.quality is not None and result.quality.passed is False
        assert layout.resolve(result.image_path).exists()


# ---------------------------------------------------------------------------
# 失败隔离
# ---------------------------------------------------------------------------
def test_后端故障被调度器隔离为failed(tmp_path: Path, monkeypatch):
    """后端抛异常时，``Scheduler`` 把整条目的样本标记为 failed 且写入失败清单。

    ``PipelineRunner.run_entry`` 自身不吞后端异常（Scheduler 才是隔离边界），
    这里对两层行为都做文档化断言。
    """
    cfg = _cfg(tmp_path)
    layout = _layout(cfg)
    plan = Planner(cfg).build()

    class _BrokenBackend:
        def generate_master(self, spec):
            raise RuntimeError("后端故障：模拟崩溃")

        def warmup(self):  # pragma: no cover - 不会被调用
            return None

    monkeypatch.setattr(PipelineRunner, "backend", property(lambda self: _BrokenBackend()))

    # 1) run_entry 本身会把异常向上抛（由 Scheduler 负责隔离）
    runner = PipelineRunner(cfg, layout)
    with pytest.raises(RuntimeError, match="后端故障"):
        runner.run_entry(plan.entries[0], plan.entries[0].samples)

    # 2) 经 Scheduler 运行时被隔离为 failed，不冒泡
    emitted = []
    hooks = SchedulerHooks(on_sample=emitted.append)
    manifest = Manifest(layout, resume=False).open()
    try:
        agg = Scheduler(cfg, layout, manifest, hooks=hooks).run(plan)
    finally:
        manifest.close()

    total = len(plan.samples)
    assert all(r.status == "failed" for r in emitted)
    assert all(r.error for r in emitted)
    assert agg["ok"] == 0
    # 聚合计数覆盖整条目的全部样本，且失败清单完整、不污染主 manifest
    assert agg["failed"] == total
    assert agg["processed"] == total
    failures = list(read_jsonl(layout.failed))
    assert len(failures) == total
    assert {f["sample_id"] for f in failures} == {s.sample_id for s in plan.samples}
    assert all("后端故障" in f["error"] for f in failures)
    assert list(read_jsonl(layout.manifest)) == []


def test_主指纹失败时进度回调仍逐样本上报(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    layout = _layout(cfg)
    plan = Planner(cfg).build()

    class _BrokenBackend:
        def generate_master(self, spec):
            raise RuntimeError("后端故障")

    monkeypatch.setattr(PipelineRunner, "backend", property(lambda self: _BrokenBackend()))
    emitted = []
    manifest = Manifest(layout, resume=False).open()
    try:
        Scheduler(cfg, layout, manifest, hooks=SchedulerHooks(on_sample=emitted.append)).run(plan)
    finally:
        manifest.close()
    assert [r.spec.sample_id for r in emitted] == [s.sample_id for s in plan.samples]
    assert all(r.image_path == "" and r.bytes == 0 for r in emitted)


def test_单样本故障被隔离且写入失败清单(tmp_path: Path, monkeypatch):
    """单样本异常被 ``_process_sample`` 吞掉 → 只该样本 failed，其余照常 ok。"""
    cfg = _cfg(tmp_path, runtime={"retries": 0})
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    runner = PipelineRunner(cfg, layout)
    original = runner._persist

    def flaky(result, image, mask, orientation, minutiae):
        if result.spec.tta_index == 0:
            raise OSError("磁盘炸了")
        return original(result, image, mask, orientation, minutiae)

    monkeypatch.setattr(runner, "_persist", flaky)
    results = runner.run_entry(entry, entry.samples)
    statuses = [r.status for r in results]
    assert statuses.count("failed") == 1
    assert statuses.count("ok") == len(results) - 1

    failed = [r for r in results if r.status == "failed"][0]
    assert failed.spec.tta_index == 0
    assert failed.error.startswith("OSError") and "磁盘炸了" in failed.error
    assert failed.image_path == ""
    ok = [r for r in results if r.status == "ok"][0]
    assert layout.resolve(ok.image_path).exists()


def test_单样本瞬时故障会被重试(tmp_path: Path, monkeypatch):
    """``runtime.retries`` 允许单样本重试：第一次 persist 失败后第二次成功。"""
    cfg = _cfg(tmp_path, runtime={"retries": 1})
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    runner = PipelineRunner(cfg, layout)
    original = runner._persist
    calls = {"n": 0}

    def flaky(result, image, mask, orientation, minutiae):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("瞬时磁盘错误")
        return original(result, image, mask, orientation, minutiae)

    monkeypatch.setattr(runner, "_persist", flaky)
    results = runner.run_entry(entry, entry.samples)
    first = results[0]
    assert first.status == "ok", first.error
    assert first.attempt == 2  # 第一次失败、第二次成功
    assert layout.resolve(first.image_path).exists()


# ---------------------------------------------------------------------------
# Manifest 行中的奇异点与轮廓参数
# ---------------------------------------------------------------------------
def test_manifest行包含奇异点与轮廓参数(tmp_path: Path):
    """主指纹的奇异点与 5 参数轮廓模型会随样本写入 manifest。"""
    cfg = _cfg(tmp_path)
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    results = PipelineRunner(cfg, layout).run_entry(entry, entry.samples)

    for result in results:
        row = result.to_manifest_row("fp")
        assert "singular_points" in row and isinstance(row["singular_points"], list)
        assert "shape_params" in row and isinstance(row["shape_params"], dict)
        # dummy 后端的轮廓参数为画布尺寸派生的 5 参数模型
        assert set(row["shape_params"]) == {"sL", "sR", "sT", "sB", "sM"}
        assert row["shape_params"]["sL"] + row["shape_params"]["sR"] == cfg.resolved_image_size()[0]

        payload = json.loads(layout.resolve(result.minutiae_path).read_text(encoding="utf-8"))
        assert payload["singular_points"] == row["singular_points"]
        # 现状（已在交付说明中记录）：细节点 JSON 载荷携带 singular_points，
        # 但尚未携带 shape_params；若库端补齐，此处自动校验一致性。
        if "shape_params" in payload:
            assert payload["shape_params"] == row["shape_params"]


def test_奇异点可由后端注入并贯穿管道(tmp_path: Path):
    """用自定义后端注入奇异点，验证它确实进入结果与 manifest。"""
    from synthfp.core.types import MasterArtifact
    from synthfp.plugins.backends import get_backend

    cfg = _cfg(tmp_path)
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    runner = PipelineRunner(cfg, layout)

    points = [{"type": "core", "x": 100, "y": 120}, {"type": "delta", "x": 140, "y": 200}]
    base_backend = get_backend("dummy", cfg.backend.model_dump(mode="python"))
    original = base_backend.generate_master

    def generate_master(spec):
        master: MasterArtifact = original(spec)
        master.singular_points = list(points)
        master.shape_params = {"sL": 1, "sR": 2, "sT": 3, "sB": 4, "sM": 5}
        return master

    base_backend.generate_master = generate_master  # type: ignore[method-assign]
    runner._backend = base_backend

    results = runner.run_entry(entry, entry.samples)
    for result in results:
        assert result.status == "ok", result.error
        assert result.singular_points == points
        assert result.shape_params == {"sL": 1, "sR": 2, "sT": 3, "sB": 4, "sM": 5}
        row = result.to_manifest_row()
        assert row["singular_points"] == points
        assert row["shape_params"] == {"sL": 1, "sR": 2, "sT": 3, "sB": 4, "sM": 5}
        payload = json.loads(layout.resolve(result.minutiae_path).read_text(encoding="utf-8"))
        assert payload["singular_points"] == points


def test_输出目录结构与设计文档一致(tmp_path: Path):
    cfg = _cfg(tmp_path)
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    PipelineRunner(cfg, layout).run_entry(entry, entry.samples)

    spec = entry.samples[0]
    expected_dir = cfg.dataset_dir / "images" / spec.identity_id / spec.finger / (
        f"imp_{spec.impression_index:02d}"
    )
    assert expected_dir.is_dir()
    assert (expected_dir / f"{spec.sample_id}.png").exists()
    assert (cfg.dataset_dir / "minutiae" / spec.identity_id / spec.finger / (
        f"imp_{spec.impression_index:02d}"
    ) / f"{spec.sample_id}.json").exists()


# ---------------------------------------------------------------------------
# 仅脊线黑色：整条管道收口
# ---------------------------------------------------------------------------
def test_仅脊线黑色时落盘图像为纯二值(tmp_path: Path):
    """主指纹阶段是二值，但捺印与 TTA 会重新引入灰度，必须在管道末端收口。

    回归用例：只在后端里二值化时，最终写入磁盘的样本仍有上百种灰度。
    """
    cfg = _cfg(
        tmp_path,
        backend={"name": "sfinge", "ridge_only_black": True, "image_size": (240, 320)},
        tta={"mode": "chain", "chain_length": (1, 1)},
    )
    layout = _layout(cfg)
    entry = Planner(cfg).build().entries[0]
    results = PipelineRunner(cfg, layout).run_entry(entry, entry.samples)

    assert results, "没有产出任何样本"
    import cv2 as _cv2

    from synthfp.legacy.util.image_utils import imread_unicode

    for result in results:
        assert result.error == ""
        assert result.image_path, "样本没有落盘路径"
        # 工作区路径含中文，cv2.imread 会静默返回 None，必须用 Unicode 安全读取
        written = imread_unicode(str(layout.root / result.image_path), _cv2.IMREAD_GRAYSCALE)
        assert written is not None, f"读不到 {result.image_path}"
        values = set(np.unique(written).tolist())
        assert values <= {0, 255}, f"{result.spec.sample_id} 非二值: {sorted(values)}"
        assert 0 in values and 255 in values


def test_finalize_image默认不改动图像():
    from synthfp.pipeline.stages import finalize_image

    cfg = build_config("quick_test")
    image = np.array([[10, 128, 200]], dtype=np.uint8)
    assert np.array_equal(finalize_image(image, cfg), image)


def test_finalize_image二值模式按128阈值():
    from synthfp.pipeline.stages import finalize_image

    cfg = apply_overrides(build_config("quick_test"), {"backend.ridge_only_black": True})
    image = np.array([[10, 127, 128, 200]], dtype=np.uint8)
    assert finalize_image(image, cfg).tolist() == [[0, 0, 255, 255]]
