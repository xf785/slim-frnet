"""生成研究报告的插图（矢量 PDF），数据全部来自 report/data/stats.json。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "report" / "data" / "stats.json").read_text(encoding="utf-8"))
FIGS = ROOT / "report" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# 中文字体（Windows 自带）
plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

C_MAIN = "#2f6fb5"
C_ACCENT = "#c0504d"
C_GRAY = "#8c8c8c"

q = DATA["experiments"]["quality"]
perf = DATA["experiments"]["end_to_end"]
perf1 = DATA["experiments"]["end_to_end_single"]
ridge = json.loads((ROOT / 'report' / 'data' / 'ridge.json').read_text(encoding='utf-8'))
master = DATA["experiments"]["master_latency"]


# ----------------------------------------------------------------------
def fig_quality() -> None:
    """质量分布 + 等级构成。"""
    scores = np.array(q["raw_quality"])
    levels = q["levels"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))

    ax1.hist(scores, bins=np.arange(0, 101, 5), color=C_MAIN, edgecolor="white", linewidth=0.6)
    ax1.axvline(q["mean"], color=C_ACCENT, linestyle="--", linewidth=1.2,
                label=f"均值 {q['mean']:.1f}")
    ax1.axvline(q["median"], color=C_GRAY, linestyle=":", linewidth=1.2,
                label=f"中位数 {q['median']:.1f}")
    ax1.set_xlabel("质量分（0–100）")
    ax1.set_ylabel("样本数")
    ax1.set_title(f"质量分分布（$n={len(scores)}$）")
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_xlim(0, 100)

    names = ["L1", "L2", "L3", "L4", "L5"]
    vals = [levels.get(str(i), 0) for i in range(1, 6)]
    # 与 NFIQ 2 同向：L1 最好、L5 最差
    bars = ax2.bar(names, vals, color=[C_MAIN, "#8fb3d9", C_GRAY, "#d98b89", C_ACCENT])
    ax2.set_xlabel("质量等级（L1 最好 → L5 最差，与 NFIQ 2 同向）")
    ax2.set_ylabel("样本数")
    ax2.set_title("质量等级构成")
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.6, str(v),
                 ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGS / "fig_quality.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
def fig_perf() -> None:
    """性能：主指纹延迟 + 端到端吞吐。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))

    # 左：单张主指纹耗时
    labels = ["首次\n（含核库构建）", "常态中位", "常态最快", "常态最慢"]
    vals = [master["cold_s"], master["median_s"], master["min_s"], master["max_s"]]
    bars = ax1.bar(labels, vals, color=[C_GRAY, C_MAIN, "#8fb3d9", "#d98b89"], width=0.62)
    ax1.set_ylabel("耗时 / s")
    ax1.set_title(f"单张主指纹渲染（{master['n']} 次，320×480）")
    ax1.set_ylim(0, max(vals) * 1.25)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}s",
                 ha="center", va="bottom", fontsize=7.5)

    # 右：吞吐
    labels2 = [f"单线程\n({perf1['samples']} 样本)", f"8 线程\n({perf['samples']} 样本)"]
    vals2 = [perf1["samples_per_s"], perf["samples_per_s"]]
    bars2 = ax2.bar(labels2, vals2, color=[C_GRAY, C_MAIN], width=0.5)
    ax2.set_ylabel("吞吐 / (样本·s$^{-1}$)")
    ax2.set_title("端到端生成吞吐")
    ax2.set_ylim(0, max(vals2) * 1.3)
    for bar, v, e in zip(bars2, vals2, [perf1["elapsed_s"], perf["elapsed_s"]]):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.08,
                 f"{v:.2f}\n({e:.1f}s)", ha="center", va="bottom", fontsize=7.5)
    ax2.annotate(f"加速比 {perf1['speedup']:.2f}×", xy=(1, vals2[1]), xytext=(0.05, max(vals2)),
                 fontsize=8, color=C_ACCENT)

    fig.tight_layout()
    fig.savefig(FIGS / "fig_perf.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
def fig_ridge() -> None:
    """脊线几何：两种估计方法对照真实 500 dpi 参考区间。"""
    fig, ax = plt.subplots(figsize=(5.2, 2.4))

    ax.axhspan(8, 12, color="#8fb3d9", alpha=0.35, label="真实 500 dpi 参考区间 (8–12 px)")

    xs = [0, 1]
    meds = [ridge["autocorr_median_px"], ridge["crossing_median_px"]]
    los = [ridge["autocorr_p25_px"], ridge["crossing_p25_px"]]
    his = [ridge["autocorr_p75_px"], ridge["crossing_p75_px"]]
    ax.errorbar(
        xs, meds,
        yerr=[[m - l for m, l in zip(meds, los)], [h - m for m, h in zip(meds, his)]],
        fmt="o", color=C_ACCENT, capsize=5, markersize=6,
        label="本系统实测（中位 / IQR，$n=%d$ 条扫描线）" % ridge["scanlines"],
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(["自相关估计", "过零点估计"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(4, max(his) + 3)
    ax.set_ylabel("脊线周期 / px")
    ax.set_title("脊线周期对照")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIGS / "fig_ridge.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
def main() -> int:
    fig_quality()
    fig_perf()
    fig_ridge()
    for name in ("fig_quality", "fig_perf", "fig_ridge"):
        p = FIGS / f"{name}.pdf"
        print(f"{p.name:16s} {p.stat().st_size / 1024:6.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
