"""Planner —— 设计文档 4.2。

职责：
1. 把「身份 × 手指 × 捺印 × TTA」展开成完整的样本计划；
2. 按身份划分 train/val/test，防止同一身份跨集合造成泄漏；
3. 为每个主指纹 / 捺印 / 样本分配稳定 ID、稳定种子与相对路径；
4. 输出 ``plan.jsonl``，与 ``config.resolved.yaml`` + 种子一起构成可复现依据。

计划的生成**必须完全确定**：同一份配置在任何机器上跑出的
``plan.jsonl``（除时间戳外）逐字节一致。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..config.schema import FINGER_NAMES, AppConfig
from .seeds import SeedTree, derive
from .storage import DatasetLayout, atomic_write_text
from .types import ImpressionSpec, MasterSpec, SampleSpec

logger = logging.getLogger(__name__)

__all__ = ["PlanEntry", "Plan", "Planner", "expand_class_assignment", "assign_splits"]


# ---------------------------------------------------------------------------
# 类别分配
# ---------------------------------------------------------------------------
def expand_class_assignment(
    count: int, distribution: Dict[str, float], seed: int
) -> List[str]:
    """按比例精确分配 ``count`` 个类别标签，并用确定性洗牌打散顺序。

    先按最大余数法算整数配额（保证总数精确等于 ``count``），再重复洗牌拼接，
    这样小批量也能得到接近目标比例的标签序列。

    Args:
        count: 标签总数。
        distribution: ``{类别: 比例}``，比例之和为 1。
        seed: 洗牌种子。

    Returns:
        长度 ``count`` 的类别列表。
    """
    if count <= 0:
        return []
    classes = list(distribution)
    weights = np.asarray([distribution[c] for c in classes], dtype=np.float64)
    weights = weights / weights.sum()

    # 最大余数法
    exact = weights * count
    base = np.floor(exact).astype(int)
    remainder = count - int(base.sum())
    if remainder > 0:
        order = np.argsort(-(exact - base))
        for i in range(remainder):
            base[order[i % len(order)]] += 1

    rng = np.random.default_rng(seed)
    out: List[str] = []
    remaining = list(base)
    while len(out) < count:
        perm = list(rng.permutation(len(classes)))
        for idx in perm:
            if remaining[idx] > 0:
                out.append(classes[int(idx)])
                remaining[idx] -= 1
                if len(out) >= count:
                    break
    return out


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------
def assign_splits(
    unit_ids: Sequence[str], ratios: Dict[str, float], seed: int
) -> Dict[str, str]:
    """把划分单元（默认是身份）分配到 train/val/test。

    采用交错分配而非连续切块：先确定性洗牌，再按比例交替取，
    这样即使身份数很少，各类别在三个集合中的分布也尽量均匀。

    Args:
        unit_ids: 划分单元 ID 列表（会被排序以保证确定性）。
        ratios: ``{split: 比例}``。
        seed: 洗牌种子。

    Returns:
        ``{unit_id: split}``。
    """
    units = sorted(unit_ids)
    if not units:
        return {}
    active = [(k, float(v)) for k, v in ratios.items() if v > 0]
    if not active:
        return {u: "train" for u in units}

    rng = np.random.default_rng(seed)
    shuffled = [units[i] for i in rng.permutation(len(units))]

    weights = np.asarray([w for _, w in active], dtype=np.float64)
    weights = weights / weights.sum()
    n = len(shuffled)
    exact = weights * n
    base = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - base))[: n - int(base.sum())]:
        base[i] += 1
    # 保证每个非零比例至少分到 1 个（当样本数足够时）
    if n >= len(active):
        for i in range(len(active)):
            if base[i] == 0:
                donor = int(np.argmax(base))
                if base[donor] > 1:
                    base[donor] -= 1
                    base[i] = 1

    # 交错填充：轮转各 split，实现均匀铺开
    assignment: Dict[str, str] = {}
    cursor = 0
    quotas = {active[i][0]: int(base[i]) for i in range(len(active))}
    names = [name for name, _ in active]
    while cursor < n:
        progressed = False
        for name in names:
            if quotas.get(name, 0) > 0 and cursor < n:
                assignment[shuffled[cursor]] = name
                quotas[name] -= 1
                cursor += 1
                progressed = True
        if not progressed:
            # 配额用尽但仍有剩余（浮点误差），全部给第一个集合
            while cursor < n:
                assignment[shuffled[cursor]] = names[0]
                cursor += 1
            break
    return assignment


# ---------------------------------------------------------------------------
# 计划条目
# ---------------------------------------------------------------------------
@dataclass
class PlanEntry:
    """一个主指纹（一个手指一遍完整合成）及其派生的捺印与样本。"""

    master: MasterSpec
    impressions: List[ImpressionSpec] = field(default_factory=list)
    samples: List[SampleSpec] = field(default_factory=list)

    @property
    def master_id(self) -> str:
        return self.master.master_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "master": self.master.to_dict(),
            "impressions": [i.to_dict() for i in self.impressions],
            "samples": [s.to_dict() for s in self.samples],
        }


@dataclass
class Plan:
    """完整生成计划。"""

    entries: List[PlanEntry]
    seed: int
    dataset_name: str

    @property
    def masters(self) -> List[MasterSpec]:
        return [e.master for e in self.entries]

    @property
    def samples(self) -> List[SampleSpec]:
        out: List[SampleSpec] = []
        for e in self.entries:
            out.extend(e.samples)
        return out

    @property
    def impressions(self) -> List[ImpressionSpec]:
        out: List[ImpressionSpec] = []
        for e in self.entries:
            out.extend(e.impressions)
        return out

    def __len__(self) -> int:
        return sum(len(e.samples) for e in self.entries)

    def iter_samples(self) -> Iterator[SampleSpec]:
        for entry in self.entries:
            yield from entry.samples

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), ensure_ascii=False) for e in self.entries) + "\n"

    def write(self, path: Path) -> Path:
        atomic_write_text(Path(path), self.to_jsonl())
        return Path(path)

    def class_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for entry in self.entries:
            hist[entry.master.fp_class] = hist.get(entry.master.fp_class, 0) + len(entry.samples)
        return hist

    def split_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for s in self.iter_samples():
            hist[s.split] = hist.get(s.split, 0) + 1
        return hist

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "seed": self.seed,
            "masters": len(self.entries),
            "impressions": len(self.impressions),
            "samples": len(self),
            "classes": self.class_histogram(),
            "splits": self.split_histogram(),
        }


# ---------------------------------------------------------------------------
# 计划器
# ---------------------------------------------------------------------------
class Planner:
    """把配置展开为样本计划。"""

    def __init__(self, config: AppConfig, layout: Optional[DatasetLayout] = None) -> None:
        self.config = config
        self.layout = layout or DatasetLayout(config.dataset_dir)
        self.tree = SeedTree(config.runtime.seed, config.dataset.name)

    # -- 主入口 -----------------------------------------------------------
    def build(self) -> Plan:
        """构造完整计划。"""
        cfg = self.config
        n_fingers = min(cfg.dataset.fingers_per_identity, len(FINGER_NAMES))

        # 1. 类别分配（稳定顺序：身份 -> 手指）
        slots: List[Tuple[str, str, int]] = []
        for i in range(1, cfg.dataset.identities + 1):
            identity_id = f"{i:06d}"
            for f_idx in range(n_fingers):
                slots.append((identity_id, FINGER_NAMES[f_idx], f_idx))
        classes = expand_class_assignment(
            len(slots), cfg.classes.distribution, derive(cfg.runtime.seed, "classes", cfg.dataset.name)
        )

        # 2. 划分（master_id 作为主指纹级单元，再由策略决定聚合粒度）
        split_of = self._make_split_map(slots)

        # 3. 展开
        entries: List[PlanEntry] = []
        for (identity_id, finger, f_idx), fp_class in zip(slots, classes, strict=True):
            master_id = f"{identity_id}_{finger}"
            master = MasterSpec(
                master_id=master_id,
                identity_id=identity_id,
                finger=finger,
                finger_index=f_idx,
                fp_class=fp_class,
                image_size=cfg.resolved_image_size(),
                seed=self.tree.finger_seed(identity_id, finger),
                split=split_of.get(master_id, "train"),
            )
            entry = PlanEntry(master=master)
            self._expand_impressions(entry)
            entries.append(entry)
        return Plan(entries=entries, seed=cfg.runtime.seed, dataset_name=cfg.dataset.name)

    # -- 内部 -------------------------------------------------------------
    def _make_split_map(self, slots: Sequence[Tuple[str, str, int]]) -> Dict[str, str]:
        """生成 ``{master_id: split}``。

        三种策略的区别只在「划分单元」的粒度：

        * ``by_identity``：同一身份的所有手指进同一个集合（**防身份泄漏，默认**）；
        * ``by_finger``：以手指为单元，同一身份的不同手指可能落在不同集合；
        * ``random``：以主指纹为单元随机划分。
        """
        cfg = self.config
        master_ids = [f"{sid}_{finger}" for sid, finger, _ in slots]
        if not cfg.split.enabled:
            return {mid: "train" for mid in master_ids}

        strategy = cfg.split.strategy
        if strategy == "by_identity":
            units = sorted({sid for sid, _, _ in slots})
            key_of = lambda mid: mid.split("_")[0]  # noqa: E731
            # 身份级划分时「单元数 < 集合数」没有意义：
            # 2 个身份无法既进 train 又进 val/test，硬塞会造出空的 val/test
            # 却让 train 显得"正常"。此时退化为全部 train 并显式告知。
            active = [name for name, weight in cfg.split.ratios.items() if weight > 0]
            if len(units) < len(active):
                logger.warning(
                    "身份数 %d 少于划分数 %d，无法按身份划分；本次全部归入 train。"
                    "（要得到 val/test，请把 identities 调到至少 %d）",
                    len(units),
                    len(active),
                    len(active),
                )
                return {mid: "train" for mid in master_ids}
        elif strategy == "by_finger":
            units = sorted(master_ids)
            key_of = lambda mid: mid  # noqa: E731
        else:  # random
            units = sorted(master_ids)
            key_of = lambda mid: mid  # noqa: E731

        unit_split = assign_splits(
            units, cfg.split.ratios, derive(cfg.runtime.seed, "split", cfg.dataset.name)
        )
        return {mid: unit_split.get(key_of(mid), "train") for mid in master_ids}

    def _expand_impressions(self, entry: PlanEntry) -> None:
        """展开一个主指纹的捺印与样本。"""
        cfg = self.config
        master = entry.master
        n_imp = cfg.dataset.impressions_per_finger
        n_tta = max(cfg.dataset.tta_per_impression, 1) if cfg.tta.enabled else 1

        for imp_idx in range(1, n_imp + 1):
            imp_seed = self.tree.impression_seed(master.identity_id, master.finger, imp_idx)
            rng = np.random.default_rng(imp_seed)
            impression = ImpressionSpec(
                impression_id=f"imp_{imp_idx:02d}",
                master_id=master.master_id,
                index=imp_idx,
                seed=imp_seed,
                pressure=float(rng.uniform(*cfg.impression.pressure_range)),
                rotation=float(rng.uniform(*cfg.impression.rotation_range)),
            )
            entry.impressions.append(impression)

            for tta_idx in range(n_tta):
                sample_id = f"{master.master_id}_imp{imp_idx:02d}_tta{tta_idx:02d}"
                rel = self.layout.sample_relative_path(
                    master.identity_id, master.finger, imp_idx, sample_id
                )
                entry.samples.append(
                    SampleSpec(
                        sample_id=sample_id,
                        master_id=master.master_id,
                        impression_id=impression.impression_id,
                        identity_id=master.identity_id,
                        finger=master.finger,
                        fp_class=master.fp_class,
                        impression_index=imp_idx,
                        tta_index=tta_idx,
                        seed=self.tree.tta_seed(
                            master.identity_id, master.finger, imp_idx, tta_idx
                        ),
                        relative_path=rel,
                        split=master.split,
                        is_base=(tta_idx == 0 and not cfg.tta.enabled),
                    )
                )

    # -- 断点续传辅助 -----------------------------------------------------
    def pending_samples(self, done_ids: set) -> List[SampleSpec]:
        """过滤出尚未完成的样本（断点续传）。"""
        return [s for s in self.build().iter_samples() if s.sample_id not in done_ids]
