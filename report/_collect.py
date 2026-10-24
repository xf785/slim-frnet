"""为大创研究报告生成实验数据（可重复执行）。

产物：report/data/stats.json —— 报告里的所有数字都来自这里，不手改。

测量项：
  E1 端到端生成吞吐（72 样本，8 线程 / 单线程）
  E2 单张主指纹耗时（10 次，冷/热分开）
  E3 质量分与细节点统计
  E4 脊线周期与占空比（与真实 500 dpi 指纹对照）
  E5 重复运行逐字节一致性（可复现性）
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthfp.config import apply_overrides, build_config  # noqa: E402
from synthfp.core.planner import Planner  # noqa: E402
from synthfp.core.storage import DatasetLayout  # noqa: E402
from synthfp.core.types import MasterSpec  # noqa: E402
from synthfp.pipeline.runner import PipelineRunner  # noqa: E402
from synthfp.plugins.backends import get_backend  # noqa: E402

OUT = ROOT / "report" / "data" / "stats.json"
EXP_ROOT = ROOT / "outputs" / "_report_exp"

COMMON = {
    "dataset.output_root": EXP_ROOT,
    "dataset.identities": 6,
    "dataset.fingers_per_identity": 2,
    "dataset.impressions_per_finger": 2,
    "dataset.tta_per_impression": 3,
    "runtime.seed": 2026,
    "runtime.workers": 8,
    "annotation.export_centerline": True,
    "annotation.export_quality_map": True,
    "minutiae.export_iso": True,
}


def make_cfg(name: str, **extra):
    over = dict(COMMON)
    over["dataset.name"] = name
    over.update(extra)
    return apply_overrides(build_config("default"), over)


def run_dataset(cfg) -> tuple[float, int, Path]:
    """跑一遍完整数据集（走应用服务层，含 manifest / 导出），返回 (净耗时秒, 样本数, 目录)。"""
    from synthfp.app.services import GenerationCallbacks, GenerationService

    svc = GenerationService(cfg, GenerationCallbacks(), export_after_run=True)
    t0 = time.perf_counter()
    svc.start(blocking=True)
    elapsed = time.perf_counter() - t0
    return elapsed, int(svc.stats.total), DatasetLayout(cfg.dataset_dir).root


def master_latency(n: int = 10) -> dict:
    cfg = build_config("default")
    backend = get_backend("sfinge", cfg.backend.model_dump())

    def render(seed: int) -> float:
        spec = MasterSpec(
            master_id=f"m{seed}", identity_id="000001", finger="left_thumb",
            finger_index=0, fp_class="left_loop", image_size=(320, 480), seed=seed,
        )
        t = time.perf_counter()
        backend.generate_master(spec)
        return time.perf_counter() - t

    cold = render(1)
    warm = [render(s) for s in range(2, 2 + n)]
    return {
        "cold_s": round(cold, 3),
        "median_s": round(statistics.median(warm), 3),
        "min_s": round(min(warm), 3),
        "max_s": round(max(warm), 3),
        "n": len(warm),
    }


def autocorr_period(row: np.ndarray, mask_row: np.ndarray) -> float:
    """用自相关估计脊线周期（主指标）。"""
    idx = np.nonzero(mask_row > 0)[0]
    if idx.size < 60:
        return 0.0
    seg = row[idx[0]: idx[-1]].astype(np.float64)
    seg -= seg.mean()
    if np.allclose(seg, 0):
        return 0.0
    ac = np.correlate(seg, seg, mode="full")[seg.size - 1:]
    ac /= ac[0] + 1e-9
    for lag in range(4, min(30, ac.size - 1)):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.15:
            return float(lag)
    return 0.0


def crossing_period(row: np.ndarray, mask_row: np.ndarray) -> float:
    """用过零点计数估计脊线周期。

    比自相关稳健：自相关在有噪声或局部对比度变化时会锁到 2 倍谐波
    （实测出现 23 px 的伪周期），而过零点直接数"一脊一谷"的个数。
    """
    idx = np.nonzero(mask_row > 0)[0]
    if idx.size < 60:
        return 0.0
    # 只取掩模跨度的中间 60%：手指边缘对比度低，脊线断掉会把周期估大
    lo_i = idx[0] + int(idx.size * 0.2)
    hi_i = idx[-1] - int(idx.size * 0.2)
    seg = row[lo_i:hi_i].astype(np.float64)
    # 5 点滑动平均压掉传感器噪声，再去均值 → 过零点即脊/谷边界
    kernel = np.ones(5) / 5.0
    smooth = np.convolve(seg, kernel, mode="same")[3:-3]
    smooth = smooth - smooth.mean()
    if smooth.std() < 3.0:
        return 0.0
    signs = np.signbit(smooth)
    crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
    cycles = crossings / 2.0
    if cycles < 2:
        return 0.0
    return float(smooth.size / cycles)


def ridge_stats(root: Path, rows: list) -> dict:
    from synthfp.legacy.util.image_utils import imread_unicode

    periods, periods_cross, duties = [], [], []
    for row in [r for r in rows if int(r["tta_index"]) == 0][:12]:
        img = imread_unicode(str(root / row["image_path"]), 0)
        msk = imread_unicode(str(root / row["mask_path"]), 0)
        if img is None or msk is None:
            continue
        h = img.shape[0]
        for y in range(h // 3, 2 * h // 3, 6):
            p = crossing_period(img[y], msk[y])
            if p > 0:
                periods.append(p)
            vals = img[y][msk[y] > 0]
            if vals.size < 100:
                continue
            lo, hi = np.percentile(vals, [10, 90])
            if hi - lo < 15:
                continue
            duties.append(float((vals < (lo + hi) / 2).mean()))
    p = np.array(periods) if periods else np.array([0.0])
    # 自相关偶尔会锁到 2 倍周期（得到 20+ 的离群值），因此报告分位数并单列区间命中率
    in_range = float(np.mean((p >= 8) & (p <= 12))) if periods else 0.0
    return {
        "period_median_px": round(float(np.median(p)), 2),
        "period_mean_px": round(float(np.mean(p)), 2),
        "period_p25_px": round(float(np.percentile(p, 25)), 2),
        "period_p75_px": round(float(np.percentile(p, 75)), 2),
        "period_p10_px": round(float(np.percentile(p, 10)), 2),
        "period_p90_px": round(float(np.percentile(p, 90)), 2),
        "period_in_8_12_ratio": round(in_range, 4),
        "duty_mean": round(float(np.mean(duties)), 4) if duties else 0.0,
        "cross_median_px": round(float(np.median(periods_cross)), 2) if periods_cross else 0.0,
        "rows_sampled": len(periods),
        "periods_raw": [round(float(x), 1) for x in p.tolist()],
    }


def main() -> int:
    import shutil

    shutil.rmtree(EXP_ROOT, ignore_errors=True)
    stats: dict = {"hardware": {}, "experiments": {}}

    # --- 硬件 ---
    import os
    import platform

    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            stats["hardware"]["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        stats["hardware"]["cpu"] = platform.processor() or "unknown"
    stats["hardware"].update(
        {
            "logical_cores": os.cpu_count(),
            "os": platform.platform(),
            "python": platform.python_version(),
        }
    )
    import cv2
    import pydantic
    import yaml

    stats["hardware"].update(
        {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pydantic": pydantic.VERSION,
            "pyyaml": yaml.__version__,
        }
    )

    # --- E2 主指纹耗时 ---
    print("[E2] 主指纹耗时 ...", flush=True)
    stats["experiments"]["master_latency"] = master_latency()

    # --- E1 端到端（8 线程）---
    print("[E1] 72 样本端到端（8 线程）...", flush=True)
    cfg8 = make_cfg("exp_r8", **{"runtime.workers": 8})
    base = ROOT
    t_start = time.perf_counter()
    elapsed8, n8, root8 = run_dataset(cfg8)
    wall8 = time.perf_counter() - t_start
    print(f"    {n8} 样本 {elapsed8:.2f}s", flush=True)

    # 读 manifest 做统计
    rows = [
        json.loads(line)
        for line in (root8 / "dataset_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qualities = np.array([float(r["nfiq2"]) for r in rows])
    levels = [str(r["quality_level"]) for r in rows]
    minutiae = np.array([int(r["minutiae_count"]) for r in rows])
    classes = {}
    for r in rows:
        classes[str(r["class_label"])] = classes.get(str(r["class_label"]), 0) + 1
    size_bytes = sum(f.stat().st_size for f in root8.rglob("*") if f.is_file())

    stats["experiments"]["end_to_end"] = {
        "samples": n8,
        "threads": 8,
        "elapsed_s": round(elapsed8, 2),
        "wall_s": round(wall8, 2),
        "samples_per_s": round(n8 / elapsed8, 2),
        "ms_per_sample": round(elapsed8 / n8 * 1000, 1),
    }

    # --- E1b 单线程 ---
    print("[E1b] 72 样本端到端（单线程）...", flush=True)
    cfg1 = make_cfg("exp_r1", **{"runtime.workers": 1})
    elapsed1, n1, root1 = run_dataset(cfg1)
    print(f"    {n1} 样本 {elapsed1:.2f}s", flush=True)
    stats["experiments"]["end_to_end_single"] = {
        "samples": n1,
        "threads": 1,
        "elapsed_s": round(elapsed1, 2),
        "samples_per_s": round(n1 / elapsed1, 2),
        "speedup": round(elapsed1 / elapsed8, 2),
    }

    # --- E3 质量 / 细节点 / 类别 ---
    stats["experiments"]["quality"] = {
        "mean": round(float(qualities.mean()), 2),
        "median": round(float(np.median(qualities)), 2),
        "std": round(float(qualities.std()), 2),
        "min": round(float(qualities.min()), 2),
        "max": round(float(qualities.max()), 2),
        "levels": {k: levels.count(k) for k in sorted(set(levels))},
        "minutiae_mean": round(float(minutiae.mean()), 1),
        "minutiae_median": round(float(np.median(minutiae)), 1),
        "minutiae_min": int(minutiae.min()),
        "minutiae_max": int(minutiae.max()),
        "class_counts": classes,
        "dataset_mb": round(size_bytes / 1024 / 1024, 2),
        "per_sample_kb": round(size_bytes / len(rows) / 1024, 1),
        "hist_quality": np.histogram(qualities, bins=10, range=(0, 100))[0].astype(int).tolist(),
        "raw_quality": [round(float(q), 1) for q in qualities],
    }

    # --- E4 脊线几何 ---
    print("[E4] 脊线周期/占空比 ...", flush=True)
    stats["experiments"]["ridge"] = ridge_stats(root8, rows)

    # --- E5 可复现性：同名 + 同种子 + 不同输出目录，再跑一次比较校验和 ---
    # 注意：dataset.name 参与种子派生（dataset → identity → finger → impression → tta），
    # 所以"改名重跑"等价于换了一份数据，必须保持同名才是在测可复现性。
    print("[E5] 可复现性（同名同种子，换输出目录重跑）...", flush=True)
    cfg_rep = make_cfg("exp_r8", **{"runtime.workers": 8, "dataset.output_root": EXP_ROOT / "rep"})
    run_dataset(cfg_rep)
    rep_root = EXP_ROOT / "rep" / "exp_r8"
    sums_a = (root8 / "checksums" / "sha256sums.txt").read_text(encoding="utf-8").splitlines()
    sums_b = (rep_root / "checksums" / "sha256sums.txt").read_text(encoding="utf-8").splitlines()
    stats["experiments"]["reproducibility"] = {
        "identical_lines": sums_a == sums_b,
        "files_compared": len(sums_a),
        "different": sum(1 for a, b in zip(sums_a, sums_b) if a != b) + abs(len(sums_a) - len(sums_b)),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUT}")
    print(json.dumps(stats["experiments"]["quality"], ensure_ascii=False)[:200], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
