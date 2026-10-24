"""统计报告 —— 设计文档 F10「生成统计报告」。

从 ``dataset_manifest.jsonl`` 直接计算，不重新读图，
因此在几万样本规模下也是秒级完成。
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config.schema import CLASS_LABELS_ZH, AppConfig
from ..core.manifest import read_jsonl
from ..core.storage import DatasetLayout, atomic_write_text

__all__ = ["compute_dataset_stats", "summarize_numbers", "format_stats_text"]


def summarize_numbers(values: Sequence[float]) -> Dict[str, float]:
    """一组数值的常用统计量（空集返回全 0）。"""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "count": len(vals),
        "mean": round(statistics.fmean(vals), 3),
        "median": round(statistics.median(vals), 3),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
        "stdev": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
    }


def _histogram(values: Sequence[float], edges: Sequence[float]) -> Dict[str, int]:
    """左闭右开分箱（最后一箱右闭）。"""
    out: Dict[str, int] = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        n = sum(1 for v in values if (lo <= v < hi) or (i == len(edges) - 2 and v == hi))
        out[f"{lo:g}-{hi:g}"] = n
    return out


def compute_dataset_stats(
    config: AppConfig,
    layout: DatasetLayout,
    rows: Optional[List[Dict[str, Any]]] = None,
    write: bool = False,
) -> Dict[str, Any]:
    """计算数据集统计报告。

    Args:
        config: 配置。
        layout: 数据集布局。
        rows: 预读的 manifest 行；为 ``None`` 时自动读取。
        write: 是否写入 ``stats.json``。

    Returns:
        统计字典（同时可直接 ``json.dump``）。
    """
    if rows is None:
        rows = [r for r in read_jsonl(layout.manifest)]

    ok_rows = [r for r in rows if r.get("status") in (None, "ok")]
    rejected = sum(1 for r in rows if r.get("status") == "rejected")
    failed = _count_lines(layout.failed)

    classes: Counter = Counter()
    splits: Counter = Counter()
    fingers: Counter = Counter()
    identities = set()
    operators: Counter = Counter()
    levels: Counter = Counter()
    per_impression: Counter = Counter()

    quality_scores: List[float] = []
    clarity: List[float] = []
    contrast: List[float] = []
    certainty: List[float] = []
    usable: List[float] = []
    periods: List[float] = []
    minutiae_counts: List[float] = []
    end_counts: List[float] = []
    bif_counts: List[float] = []
    distortions: List[float] = []
    durations: List[float] = []
    sizes: List[float] = []
    identity_samples: Counter = Counter()

    for row in ok_rows:
        cls = str(row.get("class_label", "?"))
        classes[cls] += 1
        splits[str(row.get("split", "?"))] += 1
        fingers[str(row.get("finger", "?"))] += 1
        identities.add(str(row.get("identity_id", "")))
        identity_samples[str(row.get("identity_id", ""))] += 1

        for name in row.get("tta_operators") or []:
            operators[str(name)] += 1
        if row.get("impression_effects"):
            per_impression["有捺印效果"] += 1

        detail = row.get("quality_detail") or {}
        if row.get("nfiq2") is not None:
            quality_scores.append(float(row["nfiq2"]))
        if detail:
            if detail.get("clarity") is not None:
                clarity.append(float(detail["clarity"]))
            if detail.get("contrast") is not None:
                contrast.append(float(detail["contrast"]))
            if detail.get("orientation_certainty") is not None:
                certainty.append(float(detail["orientation_certainty"]))
            if detail.get("usable_area") is not None:
                usable.append(float(detail["usable_area"]))
            if detail.get("ridge_period") is not None:
                periods.append(float(detail["ridge_period"]))
        if row.get("quality_level") is not None:
            levels[int(row["quality_level"])] += 1

        if row.get("minutiae_count") is not None:
            minutiae_counts.append(float(row["minutiae_count"]))
        if row.get("minutiae_endings") is not None:
            end_counts.append(float(row["minutiae_endings"]))
        if row.get("minutiae_bifurcations") is not None:
            bif_counts.append(float(row["minutiae_bifurcations"]))
        if row.get("tta_distortion") is not None:
            distortions.append(float(row["tta_distortion"]))
        if row.get("duration_s") is not None:
            durations.append(float(row["duration_s"]))
        if row.get("bytes"):
            sizes.append(float(row["bytes"]))

    total = len(ok_rows)
    stats: Dict[str, Any] = {
        "dataset": config.dataset.name,
        "generated_at": _now(),
        "config_fingerprint": _read_fingerprint(layout),
        "seed": config.runtime.seed,
        "totals": {
            "manifest_rows": len(rows),
            "ok": total,
            "rejected": rejected,
            "failed": failed,
            "identities": len(identities),
            "masters": config.dataset.total_masters,
            "impressions": config.dataset.total_impressions,
        },
        "class_distribution": {
            k: {"count": v, "ratio": round(v / total, 4) if total else 0.0,
                "label": CLASS_LABELS_ZH.get(k, k)}
            for k, v in sorted(classes.items())
        },
        "split_distribution": dict(sorted(splits.items())),
        "finger_distribution": dict(sorted(fingers.items())),
        "quality": {
            **summarize_numbers(quality_scores),
            "levels": {int(k): int(v) for k, v in sorted(levels.items())},
            "histogram": _histogram(quality_scores, [0, 20, 35, 50, 65, 80, 100]) if quality_scores else {},
            "clarity": summarize_numbers(clarity),
            "contrast": summarize_numbers(contrast),
            "orientation_certainty": summarize_numbers(certainty),
            "usable_area": summarize_numbers(usable),
            "ridge_period": summarize_numbers(periods),
        },
        "minutiae": {
            **summarize_numbers(minutiae_counts),
            "endings": summarize_numbers(end_counts),
            "bifurcations": summarize_numbers(bif_counts),
            "histogram": _histogram(minutiae_counts, [0, 20, 40, 60, 80, 120, 200])
            if minutiae_counts
            else {},
        },
        "tta_operators": dict(operators.most_common()),
        "tta_distortion": summarize_numbers(distortions),
        "impression_effects": dict(per_impression),
        "throughput": {
            "sample_duration_s": summarize_numbers(durations),
            "total_duration_s": round(sum(durations), 2),
            "samples_per_second": round(len(durations) / sum(durations), 3) if sum(durations) > 0 else 0.0,
        },
        "storage": {
            "image_bytes": summarize_numbers(sizes),
            "image_total_mb": round(sum(sizes) / (1024 * 1024), 3),
            "dataset_total_mb": round(_dir_size(layout.root) / (1024 * 1024), 3),
        },
        "identity_balance": {
            "min_samples": min(identity_samples.values()) if identity_samples else 0,
            "max_samples": max(identity_samples.values()) if identity_samples else 0,
            "mean_samples": round(statistics.fmean(identity_samples.values()), 2)
            if identity_samples
            else 0.0,
        },
    }

    if write:
        atomic_write_text(layout.stats, json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


# ---------------------------------------------------------------------------
# 文本呈现（CLI / GUI 共用）
# ---------------------------------------------------------------------------
def format_stats_text(stats: Dict[str, Any]) -> str:
    """把统计字典渲染成人类可读的多行文本。"""
    lines: List[str] = []
    totals = stats.get("totals", {})
    lines.append(f"数据集: {stats.get('dataset', '?')}   (seed={stats.get('seed', '?')})")
    if stats.get("config_fingerprint"):
        lines.append(f"配置指纹: {stats['config_fingerprint']}")
    lines.append(
        f"样本: {totals.get('ok', 0)} 成功 / {totals.get('rejected', 0)} 超阈值丢弃 / "
        f"{totals.get('failed', 0)} 失败"
    )
    lines.append(f"身份 {totals.get('identities', 0)} | 主指纹 {totals.get('masters', 0)} | "
                 f"捺印 {totals.get('impressions', 0)}")

    lines.append("")
    lines.append("类别分布:")
    for key, item in stats.get("class_distribution", {}).items():
        lines.append(f"  {item.get('label', key):20s} {item.get('count', 0):6d}  {item.get('ratio', 0):6.1%}")

    if stats.get("split_distribution"):
        lines.append("")
        lines.append("划分: " + "  ".join(f"{k}={v}" for k, v in stats["split_distribution"].items()))

    quality = stats.get("quality", {})
    if quality.get("count"):
        lines.append("")
        lines.append(f"质量分 (0-100): 均值 {quality['mean']:.1f} | 中位数 {quality['median']:.1f} | "
                     f"范围 {quality['min']:.1f}~{quality['max']:.1f} | 标准差 {quality['stdev']:.1f}")
        levels = quality.get("levels", {})
        if levels:
            lines.append("  等级分布: " + "  ".join(f"L{k}:{v}" for k, v in sorted(levels.items())))
        lines.append("  分项均值: 清晰度 %.3f | 对比度 %.3f | 方向一致性 %.3f | 有效面积 %.3f"
                     % (quality.get("clarity", {}).get("mean", 0.0),
                        quality.get("contrast", {}).get("mean", 0.0),
                        quality.get("orientation_certainty", {}).get("mean", 0.0),
                        quality.get("usable_area", {}).get("mean", 0.0)))

    minutiae = stats.get("minutiae", {})
    if minutiae.get("count"):
        lines.append("")
        lines.append(f"细节点: 均值 {minutiae['mean']:.1f} | 中位数 {minutiae['median']:.0f} | "
                     f"范围 {minutiae['min']:.0f}~{minutiae['max']:.0f}")
        lines.append(f"  末梢 {minutiae.get('endings', {}).get('mean', 0):.1f} | "
                     f"分叉 {minutiae.get('bifurcations', {}).get('mean', 0):.1f}")

    operators = stats.get("tta_operators", {})
    if operators:
        lines.append("")
        lines.append("TTA 算子使用频次:")
        for name, count in sorted(operators.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:22s} {count}")

    throughput = stats.get("throughput", {})
    if throughput:
        lines.append("")
        lines.append(f"吞吐: {throughput.get('samples_per_second', 0):.2f} 样本/秒 | "
                     f"累计 {throughput.get('total_duration_s', 0):.1f}s")

    storage = stats.get("storage", {})
    if storage:
        lines.append(f"存储: 图像 {storage.get('image_total_mb', 0):.1f} MB | "
                     f"数据集总计 {storage.get('dataset_total_mb', 0):.1f} MB")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 内部
# ---------------------------------------------------------------------------
def _now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _read_fingerprint(layout: DatasetLayout) -> str:
    if not layout.environment.exists():
        return ""
    try:
        env = json.loads(layout.environment.read_text(encoding="utf-8"))
        return str(env.get("config_fingerprint", ""))
    except Exception:  # pragma: no cover
        return ""


def _dir_size(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:  # pragma: no cover
                continue
    return total
