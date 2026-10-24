"""数据集导出 —— 设计文档 F10 / 4.8。

生成结束后统一产出：

* ``README.md``         数据集说明（规模、类别分布、字段释义、合成声明）
* ``environment.json``  环境快照（已在生成前写入，这里补齐全量统计）
* ``stats.json``        统计报告
* ``splits/*.csv``      划分清单
* ``checksums/sha256sums.txt``  全量文件校验和

导出是**幂等**的：重复执行只会重写这些派生产物，不改动任何图像。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import __version__
from ..config.schema import CLASS_LABELS_ZH, AppConfig
from ..core.manifest import Manifest, read_jsonl
from ..core.storage import DatasetLayout, atomic_write_text

__all__ = ["export_dataset", "write_readme", "write_checksums", "collect_artifacts"]


# ---------------------------------------------------------------------------
# 收集产物
# ---------------------------------------------------------------------------
def collect_artifacts(layout: DatasetLayout) -> tuple[List[Dict[str, Any]], List[str]]:
    """读取 manifest，返回 ``(行列表, 所有产物相对路径)``。"""
    rows = [r for r in read_jsonl(layout.manifest) if r.get("status") in (None, "ok", "rejected")]
    paths: List[str] = []
    for row in rows:
        if row.get("status") == "rejected":
            continue
        for key in (
            "image_path", "mask_path", "orientation_path", "centerline_path",
            "quality_map_path", "minutiae_path", "template_path",
        ):
            value = row.get(key)
            if value:
                paths.append(str(value))
    return rows, paths


def write_checksums(layout: DatasetLayout, relatives: Optional[List[str]] = None) -> int:
    """写 ``checksums/sha256sums.txt``。"""
    if relatives is None:
        _, relatives = collect_artifacts(layout)
    return layout.write_checksums(relatives)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
def write_readme(
    config: AppConfig,
    layout: DatasetLayout,
    rows: List[Dict[str, Any]],
    stats: Optional[Dict[str, Any]] = None,
    fingerprint: str = "",
) -> Path:
    """生成数据集 README（含合成数据声明）。"""
    cfg = config
    n = len(rows)
    classes: Dict[str, int] = {}
    splits: Dict[str, int] = {}
    identities = set()
    for row in rows:
        classes[str(row.get("class_label", "?"))] = classes.get(str(row.get("class_label", "?")), 0) + 1
        splits[str(row.get("split", "?"))] = splits.get(str(row.get("split", "?")), 0) + 1
        identities.add(str(row.get("identity_id", "")))

    lines: List[str] = []
    lines.append(f"# {cfg.dataset.name}")
    lines.append("")
    lines.append("> 由 **synthfp v" + __version__ + "** 自动生成的合成指纹数据集")
    lines.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    if fingerprint:
        lines.append(f"> 配置指纹：`{fingerprint}`（同配置 + 同种子 ⇒ 同数据）")
    lines.append("")

    lines.append("## ⚠️ 合成数据声明")
    lines.append("")
    lines.append("本数据集**完全由算法合成**，不含任何真实人类指纹数据，")
    lines.append("不涉及任何个人生物特征信息。生成算法复刻自 SFinGe 方法学")
    lines.append("（Cappelli et al., ICPR 2000）。可用于算法研究、模型预训练、")
    lines.append("数据增强与教学；**不得**用于冒充真实身份或规避生物识别系统。")
    lines.append("")

    lines.append("## 规模")
    lines.append("")
    lines.append("| 项目 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 身份数 | {len(identities)} |")
    lines.append(f"| 主指纹数（手指数） | {cfg.dataset.total_masters} |")
    lines.append(f"| 捺印数 | {cfg.dataset.total_impressions} |")
    lines.append(f"| **样本总数** | **{n}** |")
    lines.append(f"| 每手指捺印数 | {cfg.dataset.impressions_per_finger} |")
    lines.append(f"| 每捺印 TTA 数 | {cfg.dataset.tta_per_impression} |")
    w, h = cfg.resolved_image_size()
    lines.append(f"| 图像尺寸 | {w} × {h} px |")
    lines.append(f"| 分辨率标注 | {cfg.backend.ppi} ppi |")
    lines.append(f"| 随机种子 | {cfg.runtime.seed} |")
    lines.append("")

    lines.append("## 类别分布（Henry 分类）")
    lines.append("")
    lines.append("| 类别 | 目标比例 | 实际样本数 | 实际占比 |")
    lines.append("|------|---------|-----------|---------|")
    for key, weight in cfg.classes.distribution.items():
        count = classes.get(key, 0)
        ratio = count / n if n else 0.0
        label = CLASS_LABELS_ZH.get(key, key)
        lines.append(f"| {label} | {weight:.0%} | {count} | {ratio:.1%} |")
    lines.append("")

    if splits:
        lines.append("## 划分")
        lines.append("")
        lines.append(f"划分策略：`{cfg.split.strategy}`" + ("（按身份划分，无身份泄漏）" if cfg.split.strategy == "by_identity" else ""))
        lines.append("")
        lines.append("| 集合 | 样本数 |")
        lines.append("|------|-------|")
        for key in ("train", "val", "test"):
            if key in splits:
                lines.append(f"| {key} | {splits[key]} |")
        lines.append("")

    lines.append("## 目录结构")
    lines.append("")
    lines.append("```")
    lines.append(f"{cfg.dataset.name}/")
    lines.append("├── dataset_manifest.jsonl   # 每行一个样本的完整记录（唯一元数据来源）")
    lines.append("├── plan.jsonl               # 生成计划（可复现依据）")
    lines.append("├── config.resolved.yaml     # 实际生效的完整配置")
    lines.append("├── environment.json         # 环境快照（版本 / 平台 / 依赖）")
    lines.append("├── stats.json               # 统计报告")
    lines.append("├── splits/                  # train/val/test.csv")
    lines.append("├── checksums/sha256sums.txt # 全量文件校验和")
    lines.append("├── images/<身份>/<手指>/imp_NN/<样本>.png   # 指纹图像")
    lines.append("├── mask/       ...          # 手指区域掩模 {0,255}")
    lines.append("├── orientation/...          # 方向场（uint8 量化，×π/255 还原为弧度）")
    lines.append("├── frequency/  ...          # 脊线频率图（uint8 + 伴随 _meta.json 记录量纲）")
    lines.append("├── minutiae/   ...          # 细节点 JSON（+ 可选 ISO/IEC 19794-2 模板）")
    lines.append("└── logs/                    # generation.log / done.jsonl / failed.jsonl")
    lines.append("```")
    lines.append("")

    lines.append("## Manifest 字段")
    lines.append("")
    lines.append("| 字段 | 含义 |")
    lines.append("|------|------|")
    for field_name, desc in (
        ("sample_id", "样本唯一 ID：`<身份>_<手指>_imp<NN>_tta<NN>`"),
        ("identity_id", "身份 ID（同一身份的多手指共享）"),
        ("finger", "手指名（ISO/IEC 19794-4 命名）"),
        ("master_id", "主指纹 ID `<身份>_<手指>`"),
        ("impression_id / impression_index", "捺印标识与序号"),
        ("tta_id / tta_index", "TTA 标识与序号；`tta_00` 为未变换的原始捺印"),
        ("class_label", "Henry 类别：arch / left_loop / right_loop / whorl / twin_loop"),
        ("split", "所属划分：train / val / test"),
        ("image_path", "图像相对路径"),
        ("width / height / ppi", "图像尺寸与分辨率标注"),
        ("nfiq2", "自研 NFIQ2 风格质量分（0-100，越高越好）"),
        ("quality_detail", "质量分项（清晰度 / 对比度 / 方向一致性 / 有效面积 / 脊线周期）"),
        ("minutiae_count", "细节点总数（endings + bifurcations）"),
        ("tta_operator / tta_operators / tta_params", "施加的 TTA 算子及实际参数"),
        ("tta_distortion", "累计失真度（0-1，受 `tta.max_distortion` 约束）"),
        ("impression_effects", "捺印阶段效果记录（退化链 / 变形 / 传感器）"),
        ("seed", "该样本的稳定随机种子"),
        ("backend / backend_version", "生成后端及版本"),
        ("sha256 / bytes", "图像校验和与字节数"),
    ):
        lines.append(f"| `{field_name}` | {desc} |")
    lines.append("")

    lines.append("## 复现方法")
    lines.append("")
    lines.append("```bash")
    lines.append(f"synthfp generate --config {cfg.dataset.name}.yaml")
    lines.append("# 或")
    lines.append(f"synthfp generate --preset default --set runtime.seed={cfg.runtime.seed}")
    lines.append("```")
    lines.append("")
    lines.append("复现四要素：**配置文件** + `config.resolved.yaml`、**种子** `runtime.seed`、")
    lines.append("**版本** `environment.json`、**校验和** `checksums/sha256sums.txt`。")
    lines.append("四者一致即保证逐像素可复现（浮点运算在相同平台上确定）。")
    lines.append("")

    if stats:
        lines.append("## 关键统计")
        lines.append("")
        quality = stats.get("quality", {})
        if quality.get("count"):
            lines.append(f"- 质量分：均值 {quality.get('mean', 0):.1f}，中位数 {quality.get('median', 0):.1f}，"
                         f"范围 {quality.get('min', 0):.1f} ~ {quality.get('max', 0):.1f}")
            levels = quality.get("levels", {})
            if levels:
                lines.append("- 质量等级（1 最好，5 最差）：" +
                             "，".join(f"L{k} {v}" for k, v in sorted(levels.items())))
        minutiae = stats.get("minutiae", {})
        if minutiae.get("count"):
            lines.append(f"- 细节点数：均值 {minutiae.get('mean', 0):.1f}，"
                         f"中位数 {minutiae.get('median', 0):.0f}，"
                         f"范围 {minutiae.get('min', 0)} ~ {minutiae.get('max', 0)}")
        operators = stats.get("tta_operators", {})
        if operators:
            top = sorted(operators.items(), key=lambda kv: -kv[1])[:8]
            lines.append("- TTA 算子使用频次 Top 8：" + "，".join(f"{k} {v}" for k, v in top))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*本文件由 synthfp v{__version__} 自动生成，请勿手工编辑。*")
    lines.append("")

    atomic_write_text(layout.readme, "\n".join(lines))
    return layout.readme


# ---------------------------------------------------------------------------
# 顶层导出
# ---------------------------------------------------------------------------
def export_dataset(
    config: AppConfig,
    layout: DatasetLayout,
    fingerprint: str = "",
    include_checksums: Optional[bool] = None,
) -> Dict[str, Any]:
    """执行完整导出流程，返回导出摘要。"""
    from .stats import compute_dataset_stats

    cfg = config
    rows, artifacts = collect_artifacts(layout)
    summary: Dict[str, Any] = {"samples": len(rows), "artifacts": len(artifacts)}

    # 1. 统计报告
    stats: Optional[Dict[str, Any]] = None
    if cfg.export.write_stats:
        stats = compute_dataset_stats(cfg, layout, rows=rows, write=True)
        summary["stats"] = str(layout.stats)

    # 2. 划分 CSV
    if cfg.split.enabled and cfg.split.write_csv:
        manifest = Manifest(layout, resume=False)
        counts = manifest.export_splits(rows)
        summary["splits"] = counts

    # 3. 校验和
    if include_checksums if include_checksums is not None else cfg.export.write_sha256:
        summary["checksums"] = write_checksums(layout, artifacts)

    # 4. README
    if cfg.export.write_readme:
        write_readme(cfg, layout, rows, stats, fingerprint)
        summary["readme"] = str(layout.readme)

    # 5. 环境快照补齐导出信息（保留生成前写入的原内容）
    if cfg.export.write_environment and layout.environment.exists():
        try:
            env = json.loads(layout.environment.read_text(encoding="utf-8"))
            env["export"] = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "samples": len(rows),
                "artifacts": len(artifacts),
            }
            atomic_write_text(layout.environment, json.dumps(env, ensure_ascii=False, indent=2))
        except Exception:  # pragma: no cover
            pass

    return summary
