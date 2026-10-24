"""应用服务层集成测试（GenerationService / export / stats / PreviewService），≥ 25 个用例。

全部用例使用 **dummy 后端** + 1 身份 × 1 指 × 1 捺印 × 2 TTA 的微型配置，
所有输出都写进 ``tmp_path``（模块级 fixture 生成一次数据集，供多个只读断言复用），
因此与机器 CPU 数、当前工作目录完全无关。

覆盖范围：
    * ``config_fingerprint`` 的稳定性与敏感性；
    * ``dry_run`` 摘要与计划一致且不落盘；
    * ``GenerationService.start(blocking=True)`` 的全流程、回调、统计与产物清单；
    * 断点续传（第二次运行全部跳过且不新增 manifest 行）；
    * ``stop`` / ``pause`` / ``resume`` 的状态语义；
    * ``export_dataset`` 幂等（重复导出不改动图像）与 ``compute_dataset_stats`` 自洽；
    * ``format_stats_text`` 可读文本；
    * ``PreviewService.render`` 的形状、算子链与逐像素确定性。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from synthfp.app import (
    GenerationCallbacks,
    GenerationService,
    PreviewService,
    ProgressEvent,
    compute_dataset_stats,
    config_fingerprint,
    export_dataset,
    format_stats_text,
    summarize_numbers,
)
from synthfp.config import apply_overrides, build_config, loads_config
from synthfp.core.manifest import read_jsonl
from synthfp.core.storage import DatasetLayout
from synthfp.core.types import RunStats

# ---------------------------------------------------------------------------
# 工具与 fixture
# ---------------------------------------------------------------------------
def _cfg(root: Path, name: str = "app_ds", **extra):
    """微型配置：1 身份 × 1 指 × 1 捺印 × 2 TTA，dummy 后端，单工作单元。"""
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


def _strip_timestamp(data: bytes) -> list[str]:
    """去掉 README 中的「生成时间」行，用于比对两次导出的派生内容。"""
    return [line for line in data.decode("utf-8").splitlines() if "生成时间" not in line]


class _Recorder:
    """把所有回调事件收集起来，便于逐项断言。"""

    def __init__(self) -> None:
        self.phases: list[str] = []
        self.logs: list[tuple[str, str]] = []
        self.samples: list = []
        self.progress: list[ProgressEvent] = []
        self.finished: list[RunStats] = []
        self.plans: list[dict] = []

    def callbacks(self) -> GenerationCallbacks:
        return GenerationCallbacks(
            on_phase=self.phases.append,
            on_plan=self.plans.append,
            on_progress=self.progress.append,
            on_sample=self.samples.append,
            on_log=lambda level, message: self.logs.append((level, message)),
            on_finished=self.finished.append,
        )


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> SimpleNamespace:
    """完整跑一次生成（含导出），供多个只读断言复用。"""
    root = tmp_path_factory.mktemp("app_generated")
    cfg = _cfg(root)
    recorder = _Recorder()
    service = GenerationService(cfg, recorder.callbacks())
    service.start(blocking=True)
    return SimpleNamespace(cfg=cfg, service=service, recorder=recorder, root=service.layout.root)


# ---------------------------------------------------------------------------
# config_fingerprint
# ---------------------------------------------------------------------------
def test_config_fingerprint稳定且为短哈希(tmp_path: Path):
    cfg = _cfg(tmp_path)
    first = config_fingerprint(cfg)
    assert first == config_fingerprint(cfg)
    assert first == config_fingerprint(apply_overrides(cfg, {}))
    assert len(first) == 16
    assert all(c in "0123456789abcdef" for c in first)


def test_config_fingerprint随配置变化(tmp_path: Path):
    cfg = _cfg(tmp_path)
    base = config_fingerprint(cfg)
    variants = [
        apply_overrides(cfg, {"runtime.seed": cfg.runtime.seed + 1}),
        apply_overrides(cfg, {"dataset.identities": 2}),
        apply_overrides(cfg, {"tta.operators": ["rotate"]}),
        apply_overrides(cfg, {"dataset.name": "other"}),
    ]
    fingerprints = {config_fingerprint(v) for v in variants}
    assert len(fingerprints) == len(variants)
    assert base not in fingerprints


def test_config_fingerprint对字段顺序不敏感(tmp_path: Path):
    """指纹基于排序后的 JSON，因此重建同样内容得到同一指纹。"""
    cfg = _cfg(tmp_path)
    rebuilt = loads_config(json.dumps(cfg.model_dump(mode="json")))
    assert config_fingerprint(rebuilt) == config_fingerprint(cfg)


# ---------------------------------------------------------------------------
# dry_run
# ---------------------------------------------------------------------------
def test_dry_run摘要与计划一致(tmp_path: Path):
    cfg = _cfg(tmp_path)
    service = GenerationService(cfg, export_after_run=False)
    summary = service.dry_run()
    plan = service.plan

    assert plan is not None
    assert summary["dataset"] == cfg.dataset.name
    assert summary["seed"] == cfg.runtime.seed
    assert summary["masters"] == len(plan.entries) == cfg.dataset.total_masters
    assert summary["impressions"] == len(plan.impressions) == cfg.dataset.total_impressions
    assert summary["samples"] == len(plan) == cfg.estimate()["samples"]
    assert summary["fingerprints"] == {
        "masters": summary["masters"],
        "impressions": summary["impressions"],
        "samples": summary["samples"],
    }
    assert summary["classes"] == plan.class_histogram()
    assert summary["splits"] == plan.split_histogram()
    assert summary["config_fingerprint"] == service.fingerprint


def test_dry_run不写任何文件(tmp_path: Path):
    cfg = _cfg(tmp_path)
    service = GenerationService(cfg, export_after_run=False)
    service.dry_run()
    assert not service.layout.root.exists()


# ---------------------------------------------------------------------------
# 完整生成
# ---------------------------------------------------------------------------
def test_start回调全部触发(generated: SimpleNamespace):
    recorder = generated.recorder
    assert recorder.phases[:2] == ["准备", "生成"]
    assert "导出" in recorder.phases
    assert len(recorder.samples) == generated.service.stats.total
    assert recorder.progress and all(isinstance(e, ProgressEvent) for e in recorder.progress)
    assert recorder.progress[-1].percent == pytest.approx(100.0)
    assert [s.status for s in recorder.finished] == ["finished"]
    assert recorder.plans and recorder.plans[0]["samples"] == generated.service.stats.total
    assert any(level == "INFO" for level, _ in recorder.logs)
    assert all(isinstance(message, str) and message for _, message in recorder.logs)


def test_start统计与计划一致(generated: SimpleNamespace):
    cfg, stats = generated.cfg, generated.service.stats
    assert stats.status == "finished"
    assert stats.total == cfg.estimate()["samples"] == 2
    assert stats.ok == stats.total
    assert stats.rejected == 0 and stats.failed == 0 and stats.skipped == 0
    assert stats.done == stats.ok + stats.rejected + stats.failed
    assert stats.output_dir == str(generated.root)
    assert stats.finished_at >= stats.started_at > 0
    assert stats.elapsed_s >= 0.0
    assert stats.message == ""


def test_start写出核心元数据文件(generated: SimpleNamespace):
    root = generated.root
    for name in ("dataset_manifest.jsonl", "config.resolved.yaml", "environment.json", "plan.jsonl"):
        assert (root / name).is_file(), name

    resolved = loads_config((root / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved.dataset.name == generated.cfg.dataset.name
    assert resolved.backend.name == "dummy"

    env = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    assert env["config_fingerprint"] == generated.service.fingerprint
    assert env["run"]["dataset"] == generated.cfg.dataset.name
    assert env["run"]["seed"] == generated.cfg.runtime.seed
    assert env["run"]["estimate"] == generated.cfg.estimate()
    assert env["run"]["summary"] == generated.cfg.summary_lines()

    plan_lines = [ln for ln in (root / "plan.jsonl").read_text(encoding="utf-8").splitlines() if ln]
    assert len(plan_lines) == generated.cfg.dataset.total_masters


def test_start导出产物齐全(generated: SimpleNamespace):
    root = generated.root
    assert (root / "README.md").is_file()
    assert (root / "stats.json").is_file()
    assert (root / "checksums" / "sha256sums.txt").is_file()
    csvs = sorted(p.name for p in (root / "splits").iterdir())
    assert csvs and all(name.endswith(".csv") for name in csvs)


def test_start_manifest行与计划一一对应(generated: SimpleNamespace):
    rows = list(read_jsonl(generated.root / "dataset_manifest.jsonl"))
    assert len(rows) == generated.service.stats.total
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["backend"] == "dummy" for row in rows)
    assert all(row["image_path"].startswith("images/") for row in rows)
    assert all(row["config_fingerprint"] == generated.service.fingerprint for row in rows)
    sample_ids = {row["sample_id"] for row in rows}
    assert len(sample_ids) == len(rows)
    for row in rows:
        assert generated.service.layout.resolve(row["image_path"]).is_file()
        assert generated.service.layout.resolve(row["minutiae_path"]).is_file()


def test_start_README与统计内容完整(generated: SimpleNamespace):
    readme = (generated.root / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(f"# {generated.cfg.dataset.name}")
    assert "合成数据声明" in readme
    assert generated.service.fingerprint in readme
    assert "dataset_manifest.jsonl" in readme

    stats = json.loads((generated.root / "stats.json").read_text(encoding="utf-8"))
    assert stats["dataset"] == generated.cfg.dataset.name
    assert stats["totals"]["ok"] == generated.service.stats.total
    assert stats["config_fingerprint"] == generated.service.fingerprint


# ---------------------------------------------------------------------------
# 断点续传
# ---------------------------------------------------------------------------
def test_resume第二次运行全部跳过(generated: SimpleNamespace):
    before = list(read_jsonl(generated.root / "dataset_manifest.jsonl"))
    service = GenerationService(generated.cfg, export_after_run=False)
    service.start(blocking=True)

    assert service.stats.status == "finished"
    assert service.stats.skipped == service.stats.total == len(before)
    assert service.stats.ok == 0 and service.stats.failed == 0
    assert list(read_jsonl(generated.root / "dataset_manifest.jsonl")) == before
    assert generated.service.stats.ok == len(before)  # 数据集本身未被改动


def test_关闭resume后重新生成(generated: SimpleNamespace, tmp_path: Path):
    """``runtime.resume=False`` 时忽略已有记录，样本全部重新生成。"""
    cfg = apply_overrides(generated.cfg, {"runtime.resume": False, "dataset.name": "no_resume"})
    cfg.dataset.output_root = tmp_path
    service = GenerationService(cfg, export_after_run=False)
    service.start(blocking=True)
    assert service.stats.ok == service.stats.total
    assert service.stats.skipped == 0


# ---------------------------------------------------------------------------
# 生命周期控制
# ---------------------------------------------------------------------------
def test_stop在start之前可用(tmp_path: Path):
    service = GenerationService(_cfg(tmp_path), export_after_run=False)
    assert service.stats.status == "idle"
    assert service.is_running is False
    service.stop()  # 未启动也应安全
    assert service.stats.status == "stopped"
    assert service._stop.is_set()
    # start 会清除停止位，因此随后仍能正常跑完
    service.start(blocking=True)
    assert service._stop.is_set() is False
    assert service.stats.status == "finished"


def test_pause与resume切换状态字段(tmp_path: Path):
    service = GenerationService(_cfg(tmp_path), export_after_run=False)
    assert service.is_paused is False
    service.pause()
    assert service.is_paused is True
    assert service.stats.status == "paused"
    service.resume()
    assert service.is_paused is False
    assert service.stats.status == "running"
    service.stop()
    assert service.is_paused is False  # stop 会顺带清掉暂停位
    assert service.stats.status == "stopped"


def test_pause与resume记录日志(tmp_path: Path):
    recorder = _Recorder()
    service = GenerationService(_cfg(tmp_path), recorder.callbacks(), export_after_run=False)
    service.pause()
    service.resume()
    levels = [level for level, _ in recorder.logs]
    assert "WARNING" in levels and "INFO" in levels
    assert len(recorder.logs) == 2


def test_后台线程模式可等待(tmp_path: Path):
    service = GenerationService(_cfg(tmp_path), export_after_run=False)
    service.start(blocking=False)
    assert service.wait(120) is True
    assert service.is_running is False
    assert service.stats.status == "finished"
    assert service.stats.ok == service.stats.total


def test_中断后状态为stopped且已完成样本保留(tmp_path: Path):
    """运行中请求停止：在跑的条目收尾，统计状态为 stopped（已落盘样本保留）。"""
    holder: dict = {}

    def on_sample(result) -> None:  # 第一个样本回调里就请求停止
        holder["service"].stop()

    service = GenerationService(
        _cfg(tmp_path), GenerationCallbacks(on_sample=on_sample), export_after_run=False
    )
    holder["service"] = service
    service.start(blocking=True)

    assert service.stats.status == "stopped"
    assert service.stats.total == 2
    assert service.stats.ok == 2  # 当前条目跑完，样本完整保留
    rows = list(read_jsonl(service.layout.manifest))
    assert len(rows) == service.stats.ok
    # 计划与元数据仍被写出，便于续跑
    assert (service.layout.root / "plan.jsonl").exists()


# ---------------------------------------------------------------------------
# 导出幂等
# ---------------------------------------------------------------------------
def test_export_dataset幂等(generated: SimpleNamespace):
    cfg, layout = generated.cfg, DatasetLayout(generated.root)
    images = sorted(p for p in layout.images_dir.rglob("*.png"))
    assert images

    first = export_dataset(cfg, layout, fingerprint=generated.service.fingerprint)
    snapshot = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in images}
    readme_before = layout.readme.read_bytes()
    checksums_before = layout.checksums.read_text(encoding="utf-8")

    second = export_dataset(cfg, layout, fingerprint=generated.service.fingerprint)

    # 两次导出的摘要完全一致，且校验和条目数与产物数吻合
    assert first["samples"] == second["samples"] == generated.service.stats.total
    assert first["artifacts"] == second["artifacts"] > 0
    assert first["checksums"] == second["checksums"] == first["artifacts"]
    assert first["splits"] == second["splits"]
    assert first["readme"] == second["readme"] == str(layout.readme)

    # 图像完全未被触碰
    for path, (data, mtime) in snapshot.items():
        assert path.read_bytes() == data
        assert path.stat().st_mtime_ns == mtime
    assert layout.checksums.read_text(encoding="utf-8") == checksums_before
    # README 的派生内容一致，只有生成时间戳会变
    assert _strip_timestamp(layout.readme.read_bytes()) == _strip_timestamp(readme_before)


def test_export_dataset校验和覆盖全部产物(generated: SimpleNamespace):
    layout = DatasetLayout(generated.root)
    lines = [ln for ln in layout.checksums.read_text(encoding="utf-8").splitlines() if ln]
    assert lines
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert len(digest) == 64
        assert layout.resolve(rel).exists(), rel


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def test_compute_dataset_stats键完整(generated: SimpleNamespace):
    stats = compute_dataset_stats(generated.cfg, DatasetLayout(generated.root))
    for key in ("dataset", "generated_at", "config_fingerprint", "seed", "totals",
                "class_distribution", "split_distribution", "finger_distribution",
                "quality", "minutiae", "tta_operators", "tta_distortion",
                "impression_effects", "throughput", "storage", "identity_balance"):
        assert key in stats, key
    for key in ("count", "mean", "median", "min", "max", "stdev"):
        assert key in stats["quality"], key
        assert key in stats["minutiae"], key


def test_compute_dataset_stats与manifest自洽(generated: SimpleNamespace):
    layout = DatasetLayout(generated.root)
    rows = list(read_jsonl(layout.manifest))
    stats = compute_dataset_stats(generated.cfg, layout, rows=rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    assert stats["dataset"] == generated.cfg.dataset.name
    assert stats["seed"] == generated.cfg.runtime.seed
    assert stats["totals"]["manifest_rows"] == len(rows)
    assert stats["totals"]["ok"] == len(ok_rows)
    assert stats["totals"]["rejected"] == 0 and stats["totals"]["failed"] == 0
    assert stats["totals"]["identities"] == len({r["identity_id"] for r in ok_rows})
    assert stats["totals"]["masters"] == generated.cfg.dataset.total_masters
    assert stats["totals"]["impressions"] == generated.cfg.dataset.total_impressions

    class_counts = {k: v["count"] for k, v in stats["class_distribution"].items()}
    assert sum(class_counts.values()) == len(ok_rows)
    assert set(class_counts) <= set(generated.cfg.classes.distribution)
    assert sum(stats["split_distribution"].values()) == len(ok_rows)
    assert sum(stats["finger_distribution"].values()) == len(ok_rows)
    assert stats["identity_balance"]["max_samples"] == len(ok_rows)

    # 质量与细节点统计来自 manifest 行
    expected_scores = [float(r["nfiq2"]) for r in ok_rows if r.get("nfiq2") is not None]
    assert stats["quality"]["count"] == len(expected_scores)
    assert stats["quality"]["mean"] == pytest.approx(round(sum(expected_scores) / len(
        expected_scores
    ), 3))
    expected_minutiae = [float(r["minutiae_count"]) for r in ok_rows
                         if r.get("minutiae_count") is not None]
    assert stats["minutiae"]["count"] == len(expected_minutiae)
    assert stats["minutiae"]["max"] == max(expected_minutiae)


def test_compute_dataset_stats可写入stats_json(generated: SimpleNamespace, tmp_path: Path):
    cfg = apply_overrides(generated.cfg, {"dataset.name": "stats_ds"})
    cfg.dataset.output_root = tmp_path
    layout = DatasetLayout(cfg.dataset_dir)
    layout.create()
    payload = compute_dataset_stats(cfg, layout, rows=[], write=True)
    assert layout.stats.is_file()
    assert json.loads(layout.stats.read_text(encoding="utf-8")) == payload
    assert payload["totals"]["ok"] == 0
    assert payload["class_distribution"] == {}
    assert payload["quality"]["count"] == 0
    assert payload["identity_balance"] == {"min_samples": 0, "max_samples": 0, "mean_samples": 0.0}


def test_format_stats_text含数据集名与样本数(generated: SimpleNamespace):
    stats = compute_dataset_stats(generated.cfg, DatasetLayout(generated.root))
    text = format_stats_text(stats)
    assert stats["dataset"] in text
    assert f"样本: {stats['totals']['ok']} 成功" in text
    assert f"身份 {stats['totals']['identities']}" in text
    assert "类别分布:" in text
    assert "吞吐:" in text and "存储:" in text
    # 文本可读：至少十几行有效内容，且不出现 None / nan 这类占位
    lines = text.splitlines()
    assert len(lines) >= 15
    assert sum(1 for line in lines if line.strip()) >= 15
    assert "None" not in text and "nan" not in text


def test_summarize_numbers边界():
    assert summarize_numbers([]) == {"count": 0, "mean": 0.0, "median": 0.0,
                                     "min": 0.0, "max": 0.0, "stdev": 0.0}
    assert summarize_numbers([5.0]) == {"count": 1, "mean": 5.0, "median": 5.0,
                                        "min": 5.0, "max": 5.0, "stdev": 0.0}
    stats = summarize_numbers([1, 2, 3, 4])
    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["min"] == 1.0 and stats["max"] == 4.0
    assert stats["stdev"] == pytest.approx(1.118, abs=0.001)


def test_progress_event百分比():
    assert ProgressEvent(stats=RunStats(total=4, ok=1)).percent == pytest.approx(25.0)
    assert ProgressEvent(stats=RunStats()).percent == 0.0
    event = ProgressEvent(stats=RunStats(total=2, ok=2), result=None)
    assert event.percent == pytest.approx(100.0)
    assert event.result is None


# ---------------------------------------------------------------------------
# 预览服务
# ---------------------------------------------------------------------------
def test_preview_render无错误且形状正确(tmp_path: Path):
    cfg = _cfg(tmp_path)
    result = PreviewService(cfg).render(
        identity_id="000001", finger="left_thumb", fp_class="left_loop", seed=1234, tta_index=0
    )
    assert result.error == ""
    width, height = cfg.resolved_image_size()
    assert result.image.shape == (height, width)
    assert result.image.dtype == np.uint8
    assert result.mask.shape == (height, width)
    assert set(np.unique(result.mask)) <= {0, 255}
    assert result.mask.any()
    assert result.orientation.shape == (height, width)
    assert float(result.orientation.max()) < np.pi + 1e-3
    assert result.master is not None and result.master.fp_class == "left_loop"
    assert result.quality is not None and 0.0 <= result.quality.score <= 100.0
    assert result.minutiae and all(hasattr(m, "x") for m in result.minutiae)
    assert result.effects and result.duration_s >= 0.0


def test_preview_tta0无算子_tta1有算子(tmp_path: Path):
    cfg = _cfg(tmp_path, tta={"mode": "chain", "chain_length": (2, 2)})
    service = PreviewService(cfg)
    base = service.render("000001", "left_thumb", fp_class="whorl", seed=99, tta_index=0)
    assert base.error == "" and base.tta_operators == [] and base.tta_params == {}

    modified = service.render("000001", "left_thumb", fp_class="whorl", seed=99, tta_index=1)
    assert modified.error == ""
    assert modified.tta_operators
    assert set(modified.tta_operators) <= set(cfg.tta.operators)
    assert set(modified.tta_params) <= set(modified.tta_operators)
    assert not np.array_equal(base.image, modified.image)


def test_preview_同种子两次渲染逐像素一致(tmp_path: Path):
    service = PreviewService(_cfg(tmp_path))
    first = service.render("000002", "left_index", fp_class="whorl", seed=777, tta_index=0)
    second = service.render("000002", "left_index", fp_class="whorl", seed=777, tta_index=0)
    assert first.error == "" and second.error == ""
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)
    assert [m.to_dict() for m in first.minutiae] == [m.to_dict() for m in second.minutiae]

    other = service.render("000002", "left_index", fp_class="whorl", seed=778, tta_index=0)
    assert not np.array_equal(first.image, other.image)


def test_preview_不落盘(tmp_path: Path):
    cfg = _cfg(tmp_path)
    PreviewService(cfg).render("000001", "left_thumb", fp_class="arch", seed=5, tta_index=1)
    assert not cfg.dataset_dir.exists()


def test_preview_关闭标注时不计算质量与细节点(tmp_path: Path):
    service = PreviewService(_cfg(tmp_path))
    result = service.render(
        "000001", "left_thumb", fp_class="arch", seed=11, tta_index=0, with_annotation=False
    )
    assert result.error == ""
    assert result.orientation is None
    assert result.quality is None
    assert result.minutiae == []
    assert result.image is not None and result.mask is not None


def test_preview_tta_grid逐个算子(tmp_path: Path):
    cfg = _cfg(tmp_path)
    grid = PreviewService(cfg).render_tta_grid(
        ["rotate", "brightness", "not_registered"], identity_id="000001", finger="left_thumb",
        fp_class="left_loop", base_seed=2468,
    )
    assert [name for name, _ in grid][0] == "原始"
    assert len(grid) == 3  # 未注册算子被跳过
    for _, image in grid:
        assert image.dtype == np.uint8
        assert image.shape == grid[0][1].shape


def test_preview_yaml_text可往返(tmp_path: Path):
    cfg = _cfg(tmp_path)
    text = PreviewService(cfg).yaml_text()
    assert loads_config(text) == cfg
