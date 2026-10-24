"""独立测量脊线几何（周期 / 占空比），结果写入 report/data/ridge.json。

两种估计方法互为交叉验证：
  * 自相关：对整条扫描线做归一化自相关，取第一个显著峰的位置；
  * 过零点：轻度平滑后数符号变化次数，周期 = 跨度 / 周期数。

只统计掩模跨度中间 60% 的区段——手指边缘对比度低，脊线会断，
两种方法在那里都会把周期估大。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthfp.legacy.util.image_utils import imread_unicode  # noqa: E402

MANIFEST = ROOT / "outputs" / "_report_exp" / "exp_r8"
OUT = ROOT / "report" / "data" / "ridge.json"


def central_segment(row: np.ndarray, mask_row: np.ndarray, frac: float = 0.6):
    idx = np.nonzero(mask_row > 0)[0]
    if idx.size < 60:
        return None
    span = idx[-1] - idx[0]
    pad = int(span * (1 - frac) / 2)
    lo, hi = idx[0] + pad, idx[-1] - pad
    if hi - lo < 40:
        return None
    return row[lo:hi].astype(np.float64)


def autocorr_period(seg: np.ndarray) -> float:
    s = seg - seg.mean()
    if np.allclose(s, 0):
        return 0.0
    ac = np.correlate(s, s, mode="full")[s.size - 1:]
    ac = ac / (ac[0] + 1e-9)
    for lag in range(4, min(30, ac.size - 1)):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.15:
            return float(lag)
    return 0.0


def crossing_period(seg: np.ndarray) -> float:
    if seg.std() < 3.0:
        return 0.0
    kernel = np.ones(5) / 5.0
    smooth = np.convolve(seg, kernel, mode="valid")
    smooth = smooth - smooth.mean()
    signs = np.signbit(smooth)
    crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
    cycles = crossings / 2.0
    if cycles < 2:
        return 0.0
    return float(smooth.size / cycles)


def main() -> int:
    rows = [
        json.loads(line)
        for line in (MANIFEST / "dataset_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base = [r for r in rows if int(r["tta_index"]) == 0]

    ac_periods, xing_periods, duties = [], [], []
    for row in base:
        img = imread_unicode(str(MANIFEST / row["image_path"]), 0)
        msk = imread_unicode(str(MANIFEST / row["mask_path"]), 0)
        if img is None or msk is None:
            continue
        h = img.shape[0]
        for y in range(h // 3, 2 * h // 3, 4):
            seg = central_segment(img[y], msk[y])
            if seg is None:
                continue
            pa = autocorr_period(seg)
            if pa > 0:
                ac_periods.append(pa)
            px = crossing_period(seg)
            if px > 0:
                xing_periods.append(px)
            lo, hi = np.percentile(seg, [10, 90])
            if hi - lo > 15:
                duties.append(float((seg < (lo + hi) / 2).mean()))

    ac = np.array(ac_periods)
    xg = np.array(xing_periods)
    out = {
        "samples_used": len(base),
        "scanlines": int(ac.size),
        "autocorr_median_px": round(float(np.median(ac)), 2),
        "autocorr_p25_px": round(float(np.percentile(ac, 25)), 2),
        "autocorr_p75_px": round(float(np.percentile(ac, 75)), 2),
        "autocorr_in_8_12_ratio": round(float(np.mean((ac >= 8) & (ac <= 12))), 3),
        "crossing_median_px": round(float(np.median(xg)), 2),
        "crossing_p25_px": round(float(np.percentile(xg, 25)), 2),
        "crossing_p75_px": round(float(np.percentile(xg, 75)), 2),
        "crossing_in_8_12_ratio": round(float(np.mean((xg >= 8) & (xg <= 12))), 3),
        "duty_mean": round(float(np.mean(duties)), 3),
        "autocorr_raw": [round(float(v), 1) for v in ac.tolist()],
        "crossing_raw": [round(float(v), 1) for v in xg.tolist()],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in out.items():
        if not k.endswith("_raw"):
            print(f"  {k:26s} {v}")
    print(f"\n已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
