"""生成可视化对比图：主指纹 × 捺印 × TTA，以及标注叠加。

用法::

    python tools/make_preview_sheet.py --dataset outputs/e2e --identity 000001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from synthfp.legacy.util.image_utils import imread_unicode, imwrite_unicode  # noqa: E402

TILE = (150, 225)


def load(root: Path, rel: str):
    if not rel:
        return None
    path = root / rel
    return imread_unicode(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None


def label_tile(image: np.ndarray, text: str) -> np.ndarray:
    tile = cv2.resize(image, TILE, interpolation=cv2.INTER_AREA)
    if tile.ndim == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(tile, (0, 0), (TILE[0], 18), (255, 255, 255), -1)
    cv2.putText(tile, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)
    return tile


def blank() -> np.ndarray:
    tile = np.full((TILE[1], TILE[0], 3), 255, dtype=np.uint8)
    return tile


def overlay_minutiae(image: np.ndarray, payload: dict) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for m in payload.get("minutiae", []):
        x, y = int(m["x"]), int(m["y"])
        color = (255, 144, 30) if m.get("type") == "ending" else (0, 90, 255)
        cv2.circle(canvas, (x, y), 3, color, -1, cv2.LINE_AA)
        angle = float(m.get("angle", 0.0))
        cv2.line(
            canvas,
            (x, y),
            (int(x + 8 * np.cos(angle)), int(y + 8 * np.sin(angle))),
            color,
            1,
            cv2.LINE_AA,
        )
    for sp in payload.get("singular_points", []):
        x, y = int(sp["x"]), int(sp["y"])
        if sp.get("type") == "core":
            cv2.circle(canvas, (x, y), 7, (0, 0, 220), 2, cv2.LINE_AA)
        else:
            pts = np.array([[x, y - 6], [x + 6, y + 5], [x - 6, y + 5]], np.int32)
            cv2.polylines(canvas, [pts], True, (0, 170, 0), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--identity", default="000001")
    parser.add_argument(
        "--finger",
        default=None,
        help="只取这根手指（默认自动选该身份下的第一根），避免把不同手指混在一张图里",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.dataset
    rows = [
        json.loads(line)
        for line in (root / "dataset_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mine = [r for r in rows if str(r.get("identity_id")) == args.identity]
    if not mine:
        print(f"数据集里没有身份 {args.identity}")
        return 1

    finger = args.finger or sorted({str(r.get("finger")) for r in mine})[0]
    mine = [r for r in mine if str(r.get("finger")) == finger]
    if not mine:
        print(f"身份 {args.identity} 没有手指 {finger}")
        return 1
    print(f"身份 {args.identity} / 手指 {finger}：{len(mine)} 个样本")

    impressions = sorted({int(r["impression_index"]) for r in mine})
    sheet_rows = []
    for imp in impressions:
        group = sorted(
            [r for r in mine if int(r["impression_index"]) == imp],
            key=lambda r: int(r["tta_index"]),
        )
        tiles = []
        for row in group:
            image = load(root, row.get("image_path"))
            if image is None:
                tiles.append(blank())
                continue
            ops = "、".join(row.get("tta_operators") or []) or "原始"
            tiles.append(label_tile(image, f"i{imp}t{row['tta_index']} {ops[:14]}"))
        while len(tiles) < 4:
            tiles.append(blank())
        sheet_rows.append(np.hstack(tiles[:4]))

    # 第二块：标注叠加（取第一个 TTA != 0 的样本）
    annotated = [r for r in mine if int(r["tta_index"]) == 1] or mine
    ann_tiles = []
    for row in annotated[:4]:
        image = load(root, row.get("image_path"))
        payload = {}
        if row.get("minutiae_path"):
            path = root / row["minutiae_path"]
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
        if image is None:
            ann_tiles.append(blank())
            continue
        marked = overlay_minutiae(image, payload)
        n = len(payload.get("minutiae", []))
        ann_tiles.append(label_tile(marked, f"{row['sample_id'][-11:]} n={n}"))
    while len(ann_tiles) < 4:
        ann_tiles.append(blank())
    sheet_rows.append(np.hstack(ann_tiles[:4]))

    sheet = np.vstack(sheet_rows)
    out = args.out or root / "_preview_sheet.png"
    imwrite_unicode(str(out), sheet)
    print(f"已写入 {out}  {sheet.shape[1]}×{sheet.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
