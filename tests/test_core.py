"""核心域集成测试（seeds / planner / storage / manifest / quantize / types），≥ 30 个用例。

覆盖范围：
    * ``stable_seed`` 的确定性、输入敏感性与「非内置 hash」性质；
    * ``SeedTree`` 的层级派生与根种子敏感性；
    * ``expand_class_assignment`` 的最大余数配额与确定性；
    * ``assign_splits`` 的确定性、全覆盖、零权重与退化输入；
    * ``Planner`` 的规模、ID 稳定性、``by_identity`` 防泄漏、``plan.jsonl`` 逐字节一致；
    * ``DatasetLayout`` 的命名约定、目录骨架与 ``resolve`` 往返；
    * ``Manifest`` 的追加 / 断点续传 / 失败隔离 / 划分 CSV / 损坏行容错；
    * ``quantize`` 的量化往返精度与常量数组不除零；
    * ``sha256_file`` / ``atomic_write_bytes`` 与 ``RunStats`` 的边界行为。

全部用例只写 ``tmp_path``，不依赖当前工作目录与机器 CPU 数。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from synthfp.config import AppConfig, apply_overrides, build_config
from synthfp.core.manifest import Manifest, read_jsonl
from synthfp.core.planner import Plan, Planner, assign_splits, expand_class_assignment
from synthfp.core.quantize import (
    decode_float_map,
    decode_orientation,
    encode_float_map,
    encode_orientation,
)
from synthfp.core.seeds import SeedTree, as_sequence, derive, mix, stable_seed
from synthfp.core.storage import (
    DatasetLayout,
    atomic_write_bytes,
    environment_snapshot,
    sha256_bytes,
    sha256_file,
)
from synthfp.core.types import Minutia, QualityReport, RunStats, SampleResult, SampleSpec

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
_DIST = {"arch": 0.05, "left_loop": 0.35, "right_loop": 0.35, "whorl": 0.20, "twin_loop": 0.05}


def _cfg(tmp: Path, name: str = "core_ds", **extra) -> AppConfig:
    """构造一个小而完整的配置（dummy 后端，1 身份 1 指 1 捺印 2 TTA）。"""
    overrides = {
        "dataset.name": name,
        "dataset.output_root": tmp,
        "dataset.identities": 2,
        "dataset.fingers_per_identity": 2,
        "dataset.impressions_per_finger": 2,
        "dataset.tta_per_impression": 2,
        "backend.name": "dummy",
        "runtime.workers": 1,
    }
    overrides.update(extra)
    return apply_overrides(build_config("quick_test"), overrides)


# ---------------------------------------------------------------------------
# 种子派生
# ---------------------------------------------------------------------------
def test_stable_seed确定性():
    """同一组输入永远得到同一个种子（可复现性基石）。"""
    assert stable_seed("ds", 42, "000001", "left_thumb") == stable_seed(
        "ds", 42, "000001", "left_thumb"
    )
    assert stable_seed("x") == stable_seed("x")
    assert stable_seed() == stable_seed()


def test_stable_seed对参数变化敏感():
    """任一参数变化都会改变种子；顺序不同也算不同输入。"""
    base = stable_seed("ds", 1, "left_thumb")
    assert stable_seed("ds", 2, "left_thumb") != base
    assert stable_seed("ds", 1, "right_thumb") != base
    assert stable_seed("ds", "left_thumb", 1) != base
    assert stable_seed("ds", 1) != base


def test_stable_seed与类型无关():
    """参数先做 ``str()`` 强制转换，因此 1 与 "1" 等价。"""
    assert stable_seed(1, 2) == stable_seed("1", "2")
    assert stable_seed(None) == stable_seed("None")


def test_stable_seed落在64位无符号范围():
    for parts in [(), ("a",), (1, 2, 3), ("x" * 100,)]:
        seed = stable_seed(*parts)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**64


def test_stable_seed不是内置hash():
    """内置 ``hash()`` 逐进程加盐，``stable_seed`` 必须与其不同且跨进程稳定。"""
    value = stable_seed("salt-check")
    assert value != hash("salt-check")
    assert stable_seed("salt-check") == value
    # 同一次调用内重复取用完全一致（不依赖 PYTHONHASHSEED）
    assert {stable_seed("salt-check") for _ in range(5)} == {value}


def test_derive带上父种子():
    assert derive(5, "a", "b") == stable_seed(5, "a", "b")
    assert derive(5, "a") != derive(6, "a")


def test_as_sequence与mix():
    assert as_sequence(7) == (7,)
    assert as_sequence(2**64 + 3) == (3,)
    assert mix([1, 2, 3]) == mix([1, 2, 3])
    assert mix([1, 2, 3]) != mix([3, 2, 1])


def test_seed_tree同根同数据集完全一致():
    a = SeedTree(42, "ds")
    b = SeedTree(42, "ds")
    assert a.dataset_seed() == b.dataset_seed()
    assert a.identity_seed("000001") == b.identity_seed("000001")
    assert a.finger_seed("000001", "left_thumb") == b.finger_seed("000001", "left_thumb")
    assert a.impression_seed("000001", "left_thumb", 1) == b.impression_seed(
        "000001", "left_thumb", 1
    )
    assert a.tta_seed("000001", "left_thumb", 1, 2) == b.tta_seed("000001", "left_thumb", 1, 2)
    assert a.node("stage", "x", 1) == b.node("stage", "x", 1)


def test_seed_tree各层级互不相同():
    tree = SeedTree(42, "ds")
    seeds = {
        tree.dataset_seed(),
        tree.identity_seed("000001"),
        tree.identity_seed("000002"),
        tree.finger_seed("000001", "left_thumb"),
        tree.finger_seed("000001", "left_index"),
        tree.impression_seed("000001", "left_thumb", 1),
        tree.impression_seed("000001", "left_thumb", 2),
        tree.tta_seed("000001", "left_thumb", 1, 0),
        tree.tta_seed("000001", "left_thumb", 1, 1),
    }
    assert len(seeds) == 9


def test_seed_tree换根后整棵树改变():
    a = SeedTree(42, "ds")
    b = SeedTree(43, "ds")
    c = SeedTree(42, "other_ds")
    assert a.root == 42 and b.root == 43
    assert a.dataset_seed() != b.dataset_seed()
    assert a.dataset_seed() != c.dataset_seed()
    assert a.tta_seed("000001", "left_thumb", 1, 1) != b.tta_seed("000001", "left_thumb", 1, 1)
    assert a.tta_seed("000001", "left_thumb", 1, 1) != c.tta_seed("000001", "left_thumb", 1, 1)


# ---------------------------------------------------------------------------
# 类别分配
# ---------------------------------------------------------------------------
def test_expand_class_assignment长度精确():
    for count in (1, 2, 3, 7, 50, 999):
        labels = expand_class_assignment(count, _DIST, 1)
        assert len(labels) == count
        assert set(labels) <= set(_DIST)


def test_expand_class_assignment确定性():
    a = expand_class_assignment(200, _DIST, 12345)
    b = expand_class_assignment(200, _DIST, 12345)
    c = expand_class_assignment(200, _DIST, 999)
    assert a == b
    assert a != c  # 换种子后顺序被打散


def test_expand_class_assignment大规模遵循最大余数法():
    """1000 个槽位按 5/35/35/20/5 分配，各类误差不超过 ±1。"""
    labels = expand_class_assignment(1000, _DIST, 7)
    counts = Counter(labels)
    expected = {"arch": 50, "left_loop": 350, "right_loop": 350, "whorl": 200, "twin_loop": 50}
    assert set(counts) == set(expected)
    for name, want in expected.items():
        assert abs(counts[name] - want) <= 1, (name, counts[name], want)


def test_expand_class_assignment非正数返回空():
    assert expand_class_assignment(0, _DIST, 1) == []
    assert expand_class_assignment(-5, _DIST, 1) == []


def test_expand_class_assignment单类别():
    labels = expand_class_assignment(12, {"whorl": 1.0}, 3)
    assert labels == ["whorl"] * 12


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------
def test_assign_splits确定性且全覆盖():
    units = [f"{i:06d}" for i in range(1, 21)]
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = assign_splits(units, ratios, 5)
    b = assign_splits(units, ratios, 5)
    assert a == b
    assert sorted(a) == sorted(units)
    assert set(a.values()) <= set(ratios)
    # 三类都分到了样本，且 train 最多
    counts = Counter(a.values())
    assert set(counts) == {"train", "val", "test"}
    assert counts["train"] == max(counts.values())
    assert sum(counts.values()) == len(units)


def test_assign_splits与输入顺序无关():
    units = [f"{i:06d}" for i in range(1, 11)]
    assert assign_splits(units, {"train": 1.0}, 1) == assign_splits(
        list(reversed(units)), {"train": 1.0}, 1
    )


def test_assign_splits支持零权重集合():
    units = ["a", "b", "c"]
    result = assign_splits(units, {"train": 1.0, "val": 0.0, "test": 0.0}, 3)
    assert set(result.values()) == {"train"}
    assert set(result) == set(units)


def test_assign_splits全部权重为零时退化为train():
    result = assign_splits(["a", "b"], {"train": 0.0, "val": 0.0}, 3)
    assert result == {"a": "train", "b": "train"}


def test_assign_splits单元少于划分数量不崩溃():
    """单元数 < 划分数量时只保证「不崩溃 + 全部落到合法集合」。"""
    result = assign_splits(["only"], {"train": 0.7, "val": 0.15, "test": 0.15}, 3)
    assert result == {"only": "train"}
    result2 = assign_splits(["a", "b"], {"train": 0.7, "val": 0.15, "test": 0.15}, 3)
    assert set(result2) == {"a", "b"}
    assert set(result2.values()) <= {"train", "val", "test"}
    # 配额按四舍五入集中到 train（2 个单元不足以铺满 3 个集合）
    assert Counter(result2.values())["train"] == 2


def test_assign_splits空输入返回空():
    assert assign_splits([], {"train": 1.0}, 3) == {}


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
def test_planner总样本数等于配置乘积(tmp_path: Path):
    cfg = _cfg(tmp_path)
    plan = Planner(cfg).build()
    d = cfg.dataset
    assert len(plan) == d.identities * d.fingers_per_identity * d.impressions_per_finger * max(
        d.tta_per_impression, 1
    )
    assert len(plan.entries) == d.total_masters
    assert len(plan.impressions) == d.total_impressions
    assert plan.summary()["samples"] == len(plan)
    assert plan.summary()["dataset"] == cfg.dataset.name
    assert plan.summary()["seed"] == cfg.runtime.seed


def test_planner_ID唯一且两次构建稳定(tmp_path: Path):
    cfg = _cfg(tmp_path)
    p1 = Planner(cfg).build()
    p2 = Planner(cfg).build()
    ids1 = [s.sample_id for s in p1.iter_samples()]
    ids2 = [s.sample_id for s in p2.iter_samples()]
    assert ids1 == ids2
    assert len(set(ids1)) == len(ids1)
    assert [e.master.master_id for e in p1.entries] == [e.master.master_id for e in p2.entries]
    assert [e.master.seed for e in p1.entries] == [e.master.seed for e in p2.entries]
    assert [s.seed for s in p1.iter_samples()] == [s.seed for s in p2.iter_samples()]
    assert len({e.master.master_id for e in p1.entries}) == len(p1.entries)


def test_planner按身份划分不泄漏(tmp_path: Path):
    cfg = _cfg(tmp_path, dataset={"identities": 8, "fingers_per_identity": 2})
    assert cfg.split.strategy == "by_identity"
    plan = Planner(cfg).build()

    identity_splits: dict[str, set] = {}
    for entry in plan.entries:
        identity_splits.setdefault(entry.master.identity_id, set()).add(entry.master.split)
    assert all(len(s) == 1 for s in identity_splits.values()), identity_splits
    # 样本的 split 必须与所属主指纹一致
    for entry, spec in ((e, s) for e in plan.entries for s in e.samples):
        assert spec.split == entry.master.split
    assert set(plan.split_histogram()) <= {"train", "val", "test"}


def test_planner关闭划分时全部落在train(tmp_path: Path):
    cfg = _cfg(tmp_path, split={"enabled": False})
    plan = Planner(cfg).build()
    assert {e.master.split for e in plan.entries} == {"train"}
    assert plan.split_histogram() == {"train": len(plan)}


def test_planner_to_jsonl两次构建逐字节一致(tmp_path: Path):
    cfg = _cfg(tmp_path)
    text1 = Planner(cfg).build().to_jsonl()
    text2 = Planner(cfg).build().to_jsonl()
    assert text1 == text2
    lines = [ln for ln in text1.splitlines() if ln]
    assert len(lines) == cfg.dataset.total_masters
    payload = json.loads(lines[0])
    assert set(payload) == {"master", "impressions", "samples"}


def test_planner_sample_id与相对路径格式(tmp_path: Path):
    cfg = _cfg(tmp_path, dataset={"identities": 1, "fingers_per_identity": 1,
                                  "impressions_per_finger": 2, "tta_per_impression": 3})
    plan = Planner(cfg).build()
    layout = DatasetLayout(cfg.dataset_dir)
    for spec in plan.iter_samples():
        assert spec.sample_id == (
            f"{spec.master_id}_imp{spec.impression_index:02d}_tta{spec.tta_index:02d}"
        )
        assert spec.master_id == f"{spec.identity_id}_{spec.finger}"
        assert spec.relative_path == (
            f"{spec.identity_id}/{spec.finger}/imp_{spec.impression_index:02d}/"
            f"{spec.sample_id}.png"
        )
        assert layout.relative_dir(spec.identity_id, spec.finger, spec.impression_index) in (
            spec.relative_path
        )
    # 相对路径唯一
    paths = [s.relative_path for s in plan.iter_samples()]
    assert len(set(paths)) == len(paths)


def test_planner_TTA关闭时只有基线样本(tmp_path: Path):
    cfg = _cfg(tmp_path, tta={"enabled": False}, dataset={"tta_per_impression": 5})
    plan = Planner(cfg).build()
    samples = plan.samples
    assert len(samples) == cfg.dataset.total_impressions
    assert {s.tta_index for s in samples} == {0}
    assert all(s.is_base for s in samples)


def test_planner开启TTA时基线不是is_base(tmp_path: Path):
    cfg = _cfg(tmp_path)
    plan = Planner(cfg).build()
    assert not any(s.is_base for s in plan.samples)
    assert {s.tta_index for s in plan.samples} == set(range(cfg.dataset.tta_per_impression))


def test_planner捺印压力与旋转落在配置区间内(tmp_path: Path):
    cfg = _cfg(tmp_path)
    plan = Planner(cfg).build()
    lo, hi = cfg.impression.pressure_range
    rlo, rhi = cfg.impression.rotation_range
    for imp in plan.impressions:
        assert lo <= imp.pressure <= hi
        assert rlo <= imp.rotation <= rhi
        assert imp.impression_id == f"imp_{imp.index:02d}"


def test_planner_pending_samples按已完成集合过滤(tmp_path: Path):
    cfg = _cfg(tmp_path)
    planner = Planner(cfg)
    plan = planner.build()
    done = {plan.samples[0].sample_id, plan.samples[1].sample_id}
    pending = planner.pending_samples(done)
    assert len(pending) == len(plan.samples) - 2
    assert not (done & {s.sample_id for s in pending})
    assert len(planner.pending_samples(set())) == len(plan.samples)


def test_plan对象是dataclass且可写入文件(tmp_path: Path):
    cfg = _cfg(tmp_path)
    plan = Planner(cfg).build()
    assert isinstance(plan, Plan)
    out = plan.write(tmp_path / "plan.jsonl")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == plan.to_jsonl()
    assert len(plan) == len(plan.samples)


# ---------------------------------------------------------------------------
# DatasetLayout / 存储工具
# ---------------------------------------------------------------------------
def test_dataset_layout命名约定(tmp_path: Path):
    layout = DatasetLayout(tmp_path / "ds")
    assert layout.relative_dir("000001", "left_thumb", 3) == "000001/left_thumb/imp_03"
    rel = layout.sample_relative_path("000001", "left_thumb", 3, "sid")
    assert rel == "000001/left_thumb/imp_03/sid.png"
    assert layout.sample_relative_path("000001", "left_thumb", 3, "sid", ".json") == (
        "000001/left_thumb/imp_03/sid.json"
    )
    assert layout.path_for("images", rel) == layout.images_dir / rel


def test_dataset_layout关键路径属性(tmp_path: Path):
    root = tmp_path / "ds"
    layout = DatasetLayout(root)
    assert layout.readme == root / "README.md"
    assert layout.manifest == root / "dataset_manifest.jsonl"
    assert layout.plan == root / "plan.jsonl"
    assert layout.stats == root / "stats.json"
    assert layout.config_resolved == root / "config.resolved.yaml"
    assert layout.environment == root / "environment.json"
    assert layout.done == root / "logs" / "done.jsonl"
    assert layout.failed == root / "logs" / "failed.jsonl"
    assert layout.checksums == root / "checksums" / "sha256sums.txt"
    assert layout.splits_dir == root / "splits"
    assert layout.images_dir == root / "images"
    assert layout.mask_dir == root / "mask"
    assert layout.minutiae_dir == root / "minutiae"


def test_dataset_layout_create建立目录骨架(tmp_path: Path):
    layout = DatasetLayout(tmp_path / "ds")
    layout.create()
    for path in (layout.root, layout.splits_dir, layout.root / "checksums", layout.root / "logs"):
        assert path.is_dir(), path
    layout.create(["images", "mask", "orientation", "centerline", "quality", "minutiae"])
    for path in (layout.images_dir, layout.mask_dir, layout.orientation_dir,
                 layout.centerline_dir, layout.quality_dir, layout.minutiae_dir):
        assert path.is_dir(), path


def test_dataset_layout_resolve往返(tmp_path: Path):
    layout = DatasetLayout(tmp_path / "ds")
    rel = layout.sample_relative_path("000001", "left_thumb", 1, "sid")
    resolved = layout.resolve(rel)
    assert resolved == layout.root / rel
    assert resolved.relative_to(layout.root).as_posix() == rel


def test_sha256_file与hashlib一致(tmp_path: Path):
    target = tmp_path / "blob.bin"
    target.write_bytes(b"synthfp" * 1000)
    assert sha256_file(target) == hashlib.sha256(target.read_bytes()).hexdigest()
    assert sha256_bytes(target.read_bytes()) == sha256_file(target)
    assert len(sha256_file(target)) == 64


def test_atomic_write_bytes不留tmp(tmp_path: Path):
    target = tmp_path / "nested" / "a.bin"
    atomic_write_bytes(target, b"v1")
    assert target.read_bytes() == b"v1"
    atomic_write_bytes(target, b"v2-longer")  # 覆盖写
    assert target.read_bytes() == b"v2-longer"
    assert list(tmp_path.rglob("*.tmp")) == []
    assert sorted(p.name for p in (tmp_path / "nested").iterdir()) == ["a.bin"]


def test_environment_snapshot包含关键字段(tmp_path: Path):
    snap = environment_snapshot("fp123")
    for key in ("timestamp", "python", "platform", "machine", "processor_count",
                "numpy_version", "cv2_version", "synthfp_version", "config_fingerprint"):
        assert key in snap, key
    assert snap["config_fingerprint"] == "fp123"
    assert isinstance(snap["processor_count"], int) and snap["processor_count"] >= 1


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def test_manifest追加后能回读(tmp_path: Path):
    layout = DatasetLayout(tmp_path / "ds")
    layout.create()
    rows = [{"sample_id": f"s{i}", "status": "ok", "nfiq2": i} for i in range(3)]
    with Manifest(layout, resume=False) as manifest:
        for row in rows:
            manifest.append(row)
        assert manifest.written == 3
    assert list(read_jsonl(layout.manifest)) == rows


def test_manifest_done_ids与is_done(tmp_path: Path):
    layout = DatasetLayout(tmp_path / "ds")
    layout.create()
    manifest = Manifest(layout, resume=False)
    assert manifest.done_ids == set()
    with manifest:
        manifest.append({"sample_id": "a"})
        assert manifest.done_ids == {"a"}
        assert manifest.is_done("a") is True
        assert manifest.is_done("b") is False
        manifest.append({"sample_id": "b"})
    assert manifest.done_ids == {"a", "b"}
    # 新实例（resume=True）从 done.jsonl 恢复
    resumed = Manifest(layout, resume=True)
    assert resumed.done_ids == {"a", "b"}
    assert [r["sample_id"] for r in read_jsonl(layout.done)] == ["a", "b"]
    # resume=False 时不读取历史
    assert Manifest(layout, resume=False).done_ids == set()


def test_manifest_append_failure写failed而不污染manifest(tmp_path: Path):
    layout = DatasetLayout(tmp_path / "ds")
    layout.create()
    with Manifest(layout, resume=False) as manifest:
        manifest.append({"sample_id": "ok1"})
        manifest.append_failure("bad1", "boom", attempt=2, extra={"stage": "tta"})
        assert manifest.failed_ids == {"bad1"}
        assert manifest.written == 1
    assert [r["sample_id"] for r in read_jsonl(layout.manifest)] == ["ok1"]
    failures = list(read_jsonl(layout.failed))
    assert len(failures) == 1
    assert failures[0]["sample_id"] == "bad1"
    assert failures[0]["error"] == "boom"
    assert failures[0]["attempt"] == 2
    assert failures[0]["stage"] == "tta"
    # 失败记录只进 failed 集合；done 集合不会被 append_failure 反向清理
    manifest2 = Manifest(layout, resume=False)
    with manifest2:
        manifest2.append({"sample_id": "x"})
        manifest2.append_failure("x", "later failed")
        assert manifest2.failed_ids == {"x"}
        # 现行语义：图像已成功落盘，因此 done 仍保留该样本（resume 时视为已完成）
        assert "x" in manifest2.done_ids
        manifest2.append({"sample_id": "x"})  # 重跑成功后从失败集合移出
        assert manifest2.failed_ids == set()


def test_manifest未打开时写入报错(tmp_path: Path):
    layout = DatasetLayout(tmp_path / "ds")
    with pytest.raises(RuntimeError):
        Manifest(layout, resume=False).append({"sample_id": "a"})
    with pytest.raises(RuntimeError):
        Manifest(layout, resume=False).append_failure("a", "e")


def test_manifest_open幂等且existing_rows可读(tmp_path: Path):
    layout = DatasetLayout(tmp_path / "ds")
    layout.create()
    with Manifest(layout, resume=False) as manifest:
        manifest.append({"sample_id": "a", "split": "train"})
    manifest = Manifest(layout, resume=True)
    assert manifest.existing_rows() == [{"sample_id": "a", "split": "train"}]
    with manifest:
        manifest.append({"sample_id": "b", "split": "val"})
    assert [r["sample_id"] for r in manifest.existing_rows()] == ["a", "b"]


def test_manifest_export_splits写入每类一个CSV(tmp_path: Path):
    layout = DatasetLayout(tmp_path / "ds")
    layout.create()
    manifest = Manifest(layout, resume=False)
    rows = [
        {"sample_id": "b", "split": "train", "identity_id": "000001", "finger": "left_thumb",
         "image_path": "images/b.png", "nfiq2": 50.0, "minutiae_count": 30, "sha256": "ab"},
        {"sample_id": "a", "split": "train", "identity_id": "000001", "finger": "left_thumb",
         "image_path": "images/a.png", "nfiq2": 60.0, "minutiae_count": 40, "sha256": "cd"},
        {"sample_id": "c", "split": "val", "identity_id": "000002", "finger": "left_index",
         "image_path": "images/c.png", "nfiq2": 70.0, "minutiae_count": 50, "sha256": "ef"},
    ]
    counts = manifest.export_splits(rows)
    assert counts == {"train": 2, "val": 1}
    assert sorted(p.name for p in layout.splits_dir.iterdir()) == ["train.csv", "val.csv"]

    text = (layout.splits_dir / "train.csv").read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert lines[0] == (
        "sample_id,identity_id,finger,impression_index,tta_index,class_label,split,"
        "image_path,minutiae_path,nfiq2,minutiae_count,sha256"
    )
    assert lines[1].startswith("a,000001,left_thumb")  # 按 sample_id 排序
    assert lines[2].startswith("b,")
    assert len(lines) == 3


def test_read_jsonl跳过损坏行且容忍缺失文件(tmp_path: Path):
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"ok": 1}\nnot-json\n\n   \n{"ok": 2}\n{"ok": 3}\n', encoding="utf-8"
    )
    rows = list(read_jsonl(path))
    assert [r["ok"] for r in rows] == [1, 2, 3]
    assert list(read_jsonl(tmp_path / "missing.jsonl")) == []


def test_read_jsonl只跳过非法JSON(tmp_path: Path):
    """跳过非法 JSON，**也**跳过合法的非对象行（数组/标量）。

    下游一律按 ``dict`` 使用 ``read_jsonl`` 的结果，放数组过去只会在远处炸出
    ``AttributeError``，所以非对象行同样被丢弃。
    """
    target = tmp_path / "rows.jsonl"
    target.write_text('[1, 2]\n{"ok": 1}\n{broken\n"solo"\n42\n', encoding="utf-8")
    assert list(read_jsonl(target)) == [{"ok": 1}]
    assert list(read_jsonl(tmp_path / "no_such_file.jsonl")) == []


# ---------------------------------------------------------------------------
# 量化编解码
# ---------------------------------------------------------------------------
def test_encode_decode_orientation往返精度():
    angles = np.linspace(0.0, float(np.pi), 512, endpoint=False, dtype=np.float32)
    encoded = encode_orientation(angles)
    assert encoded.dtype == np.uint8
    assert encoded.min() >= 0 and encoded.max() <= 255

    decoded = decode_orientation(encoded)
    assert decoded.dtype == np.float32
    # 环绕误差（π 与 0 等价）
    diff = np.abs(((decoded - angles + np.pi / 2) % np.pi) - np.pi / 2)
    assert float(diff.max()) < 0.013  # 量化步长约 0.7°，远小于估计精度
    assert float(diff.mean()) < 0.005


def test_encode_orientation处理越界与负角():
    values = np.array([[-1.0, 0.0], [float(np.pi), float(2 * np.pi)]], dtype=np.float32)
    encoded = encode_orientation(values)
    assert encoded.min() >= 0 and encoded.max() <= 255
    # 取模后 -1.0 与 π-1.0 等价
    assert encoded[0, 0] == encode_orientation(np.array([np.pi - 1.0], dtype=np.float32))[0]


def test_encode_decode_float_map往返():
    rng = np.random.default_rng(0)
    data = (rng.random((48, 64)).astype(np.float32) * 4.0) + 1.0
    encoded, meta = encode_float_map(data)
    assert encoded.dtype == np.uint8 and encoded.shape == data.shape
    assert set(meta) == {"vmin", "vmax"}
    assert meta["vmax"] > meta["vmin"]

    restored = decode_float_map(encoded, meta)
    assert restored.dtype == np.float32
    # 量化步长 = (vmax - vmin) / 255；默认走 0.5%~99.5% 分位，
    # 落在分位之外的样本会被裁剪，因此误差上界放宽到 2 个步长。
    step = (meta["vmax"] - meta["vmin"]) / 255.0
    assert float(np.abs(restored - data).max()) <= step * 2


def test_encode_decode_float_map显式范围():
    data = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    encoded, meta = encode_float_map(data, vmin=0.0, vmax=1.0)
    assert meta == {"vmin": 0.0, "vmax": 1.0}
    assert encoded.min() == 0 and encoded.max() == 255
    restored = decode_float_map(encoded, meta)
    assert float(np.abs(restored - data).max()) <= 1.0 / 255.0


def test_encode_float_map常量数组不除零():
    const = np.full((8, 8), 3.5, dtype=np.float32)
    encoded, meta = encode_float_map(const)
    assert np.isfinite(encoded).all()
    assert meta["vmax"] > meta["vmin"]  # hi 被抬高了 1e-6，避免除零
    restored = decode_float_map(encoded, meta)
    assert float(np.abs(restored - const).max()) < 1e-3
    zeros = np.zeros((4, 4), dtype=np.float32)
    encoded0, meta0 = encode_float_map(zeros)
    assert np.isfinite(encoded0).all() and meta0["vmax"] > meta0["vmin"]


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------
def test_runstats零耗时rate与eta为零():
    stats = RunStats()
    assert stats.rate == 0.0
    assert stats.eta_s == 0.0
    assert stats.processed == 0
    payload = stats.to_dict()
    assert payload["rate"] == 0.0
    assert payload["eta_s"] == 0.0
    assert payload["processed"] == 0
    assert payload["status"] == "idle"


def test_runstats正常计算rate与eta():
    stats = RunStats(total=10, ok=4, rejected=1, failed=1, elapsed_s=2.0)
    assert stats.processed == 6
    assert stats.rate == pytest.approx(3.0)
    assert stats.eta_s == pytest.approx(4.0 / 3.0)
    assert stats.to_dict()["rate"] == pytest.approx(3.0)
    assert stats.to_dict()["eta_s"] == pytest.approx(1.3, abs=0.05)


def test_minutia字段与类型编码():
    ending = Minutia(x=3, y=4, angle=1.23456, kind="ending", quality=0.98765)
    assert ending.type_code == 0
    assert ending.to_dict() == {"x": 3, "y": 4, "angle": 1.2346, "type": "ending", "quality": 0.988}
    bifurcation = Minutia(x=1, y=2, kind="bifurcation")
    assert bifurcation.type_code == 1
    assert bifurcation.to_dict()["type"] == "bifurcation"


def test_sample_result_manifest行字段(tmp_path: Path):
    spec = SampleSpec(
        sample_id="000001_left_thumb_imp01_tta00",
        master_id="000001_left_thumb",
        impression_id="imp_01",
        identity_id="000001",
        finger="left_thumb",
        fp_class="left_loop",
        impression_index=1,
        tta_index=0,
        seed=123,
        relative_path="000001/left_thumb/imp_01/x.png",
        split="train",
        is_base=True,
    )
    result = SampleResult(
        spec=spec,
        status="ok",
        image_path="images/x.png",
        width=320,
        height=480,
        mask_path="mask/x.png",
        orientation_path="orientation/x.png",
        centerline_path="centerline/x.png",
        quality_map_path="quality/x.png",
        minutiae_path="minutiae/x.json",
        template_path="minutiae/x.iso",
        checksum="deadbeef",
        bytes=11,
        quality=QualityReport(score=77.5, level=2),
        minutiae_count=4,
        minutiae_endings=3,
        minutiae_bifurcations=1,
        tta_operators=["rotate"],
        tta_params={"rotate": {"angle": 3.0}},
        tta_distortion=0.25,
        singular_points=[{"type": "core", "x": 10, "y": 20}],
        shape_params={"sL": 100},
        backend="dummy",
        backend_version="2.0",
        ppi=500,
    )
    row = result.to_manifest_row("fp-abc")
    assert row["sample_id"] == spec.sample_id
    assert row["tta_id"] == "tta_00"
    assert row["tta_index"] == 0
    assert row["is_base"] is True
    assert row["class_label"] == "left_loop"
    assert row["split"] == "train"
    assert row["nfiq2"] == pytest.approx(77.5)
    assert row["quality_level"] == 2
    assert row["minutiae_count"] == 4 and row["minutiae_endings"] == 3
    assert row["tta_operator"] == "rotate"
    assert row["sha256"] == "deadbeef" and row["bytes"] == 11
    assert row["singular_points"] == [{"type": "core", "x": 10, "y": 20}]
    assert row["shape_params"] == {"sL": 100}
    assert row["config_fingerprint"] == "fp-abc"
    assert row["mask_path"] == "mask/x.png" and row["template_path"] == "minutiae/x.iso"
    assert "error" not in row
    assert json.loads(result.to_json())["sample_id"] == spec.sample_id
    assert result.ok is True
    assert SampleResult(spec=spec, status="failed", error="boom").to_manifest_row()["error"] == "boom"
