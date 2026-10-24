"""存储层 —— 设计文档 4.8。

负责：目录结构、原子写入、校验和、路径命名。

**所有元数据都写入统一 Manifest（``dataset_manifest.jsonl``），
任何模块不得私自另建元数据文件**（设计文档 0.2「单一数据源」）。

输出目录结构::

    output/<dataset>/
      README.md
      config.resolved.yaml
      environment.json
      dataset_manifest.jsonl        # 每行一个样本
      plan.jsonl                    # 生成计划（可复现依据）
      stats.json                    # 统计报告
      splits/{train,val,test}.csv
      images/<identity>/<finger>/<impression>/<sample>.png
      mask/...  orientation/...  frequency/...  centerline/...
      minutiae/.../<sample>.json    (+ .iso 模板)
      checksums/sha256sums.txt
      logs/generation.log  logs/failed.jsonl
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .quantize import (
    decode_orientation,
    encode_orientation,
    save_array_png,
)

__all__ = ["DatasetLayout", "atomic_write_bytes", "sha256_file", "sha256_bytes", "environment_snapshot"]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """流式计算文件 SHA-256。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """原子写入（先写临时文件再 rename），避免中断产生半截文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path)
    tmp.write_bytes(data)
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(Path(path), text.encode("utf-8"))


def _temp_path(target: Path) -> Path:
    """生成同目录下的临时路径。

    注意：临时路径必须以 ``.tmp`` 结尾且**不能**保留 ``.png`` 之类的后缀在末尾，
    否则 OpenCV 的 ``imencode`` 会按扩展名找编码器而失败；
    因此这里用 ``<stem>.tmp``（同目录，保证 ``replace`` 是原子 rename）。
    """
    return target.with_name(f"{target.stem}.tmp")


def environment_snapshot(config_fingerprint: str = "") -> Dict[str, Any]:
    """环境快照（``environment.json``）—— 可复现四要素之一。"""
    snap: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch": time.time(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_count": _cpu_count(),
        "executable": sys.executable,
    }
    for mod in ("numpy", "cv2", "pydantic", "yaml"):
        try:
            m = __import__(mod)
            snap[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:  # pragma: no cover
            snap[f"{mod}_version"] = "missing"
    try:
        from .. import __version__ as pkg_version

        snap["synthfp_version"] = pkg_version
    except Exception:  # pragma: no cover
        snap["synthfp_version"] = "unknown"
    if config_fingerprint:
        snap["config_fingerprint"] = config_fingerprint
    return snap


def _cpu_count() -> int:
    try:
        import os

        return os.cpu_count() or 1
    except Exception:  # pragma: no cover
        return 1


# ---------------------------------------------------------------------------
# 目录布局
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetLayout:
    """数据集输出目录布局（唯一路径来源）。"""

    root: Path

    # -- 顶层 -------------------------------------------------------------
    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    @property
    def config_resolved(self) -> Path:
        return self.root / "config.resolved.yaml"

    @property
    def environment(self) -> Path:
        return self.root / "environment.json"

    @property
    def manifest(self) -> Path:
        return self.root / "dataset_manifest.jsonl"

    @property
    def plan(self) -> Path:
        return self.root / "plan.jsonl"

    @property
    def stats(self) -> Path:
        return self.root / "stats.json"

    @property
    def done(self) -> Path:
        return self.root / "logs" / "done.jsonl"

    @property
    def failed(self) -> Path:
        return self.root / "logs" / "failed.jsonl"

    @property
    def log(self) -> Path:
        return self.root / "logs" / "generation.log"

    @property
    def checksums(self) -> Path:
        return self.root / "checksums" / "sha256sums.txt"

    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    # -- 产物目录 ---------------------------------------------------------
    def _modality_dir(self, modality: str) -> Path:
        return self.root / modality

    @property
    def images_dir(self) -> Path:
        return self._modality_dir("images")

    @property
    def mask_dir(self) -> Path:
        return self._modality_dir("mask")

    @property
    def orientation_dir(self) -> Path:
        return self._modality_dir("orientation")

    @property
    def frequency_dir(self) -> Path:
        return self._modality_dir("frequency")

    @property
    def centerline_dir(self) -> Path:
        return self._modality_dir("centerline")

    @property
    def quality_dir(self) -> Path:
        return self._modality_dir("quality")

    @property
    def minutiae_dir(self) -> Path:
        return self._modality_dir("minutiae")

    # -- 命名 -------------------------------------------------------------
    @staticmethod
    def relative_dir(identity_id: str, finger: str, impression_index: int) -> str:
        """样本所在的相对目录（即 ``images/`` 之后的部分）。"""
        return f"{identity_id}/{finger}/imp_{impression_index:02d}"

    def sample_relative_path(
        self, identity_id: str, finger: str, impression_index: int, sample_id: str,
        suffix: str = ".png",
    ) -> str:
        return f"{self.relative_dir(identity_id, finger, impression_index)}/{sample_id}{suffix}"

    def path_for(self, modality: str, relative: str) -> Path:
        return self._modality_dir(modality) / relative

    def resolve(self, relative: str) -> Path:
        """把 manifest 里的相对路径还原为绝对路径。"""
        return self.root / relative

    # -- 创建 -------------------------------------------------------------
    def create(self, modalities: Iterable[str] = ()) -> None:
        """建立目录骨架。"""
        for p in (
            self.root,
            self.splits_dir,
            self.root / "checksums",
            self.root / "logs",
        ):
            p.mkdir(parents=True, exist_ok=True)
        for m in modalities:
            self._modality_dir(m).mkdir(parents=True, exist_ok=True)

    # -- 写产物 -----------------------------------------------------------
    def save_image(
        self, relative: str, image: np.ndarray, compression: int = 3, fmt: str = "png"
    ) -> str:
        """保存灰度图，返回相对路径（``images/...``，正斜杠）。"""
        from ..legacy.util.image_utils import imwrite_unicode

        rel = Path(relative).with_suffix(f".{fmt}")
        target = self.images_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression)] if fmt == "png" else None
        if not imwrite_unicode(str(_temp_path(target)), image, params, ext=f".{fmt}"):
            raise OSError(f"写入图像失败: {target}")
        _temp_path(target).replace(target)
        return str(Path("images") / rel).replace("\\", "/")

    def save_mask(self, relative: str, mask: np.ndarray, compression: int = 3) -> str:
        from ..legacy.util.image_utils import imwrite_unicode

        rel = Path(relative).with_suffix(".png")
        target = self.mask_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite_unicode(
            str(_temp_path(target)),
            mask,
            [cv2.IMWRITE_PNG_COMPRESSION, int(compression)],
            ext=".png",
        ):
            raise OSError(f"写入掩模失败: {target}")
        _temp_path(target).replace(target)
        return str(Path("mask") / rel).replace("\\", "/")

    def save_orientation(self, relative: str, orientation: np.ndarray) -> str:
        rel = Path(relative).with_suffix(".png")
        target = self.orientation_dir / rel
        save_array_png(
            target,
            encode_orientation(orientation),
            meta={"kind": "orientation", "encoding": "uint8", "scale": "pi/255"},
        )
        return str(Path("orientation") / rel).replace("\\", "/")

    def save_float_map(self, modality: str, relative: str, data: np.ndarray) -> str:
        from .quantize import encode_float_map

        rel = Path(relative).with_suffix(".png")
        target = self._modality_dir(modality) / rel
        enc, meta = encode_float_map(data)
        save_array_png(target, enc, meta={"kind": modality, **meta})
        return str(Path(modality) / rel).replace("\\", "/")

    def save_json(self, modality: str, relative: str, payload: Any) -> str:
        rel = Path(relative).with_suffix(".json")
        target = self._modality_dir(modality) / rel
        atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=1))
        return str(Path(modality) / rel).replace("\\", "/")

    def load_orientation(self, relative: str) -> np.ndarray:
        from .quantize import load_array_png

        arr, _ = load_array_png(self.resolve(relative))
        return decode_orientation(arr)

    # -- 校验和 -----------------------------------------------------------
    def write_checksums(self, relatives: Sequence[str]) -> int:
        """对给定相对路径列表计算 SHA-256 并写 ``checksums/sha256sums.txt``。"""
        lines: List[str] = []
        for rel in sorted(relatives):
            p = self.resolve(rel)
            if p.exists():
                lines.append(f"{sha256_file(p)}  {rel}")
        atomic_write_text(self.checksums, "\n".join(lines) + ("\n" if lines else ""))
        return len(lines)

    def summary_tree(self, max_entries: int = 40) -> List[Tuple[str, int, int]]:
        """返回 ``[(相对路径, 文件数, 总字节), ...]``，供 GUI 展示数据集结构。"""
        out: List[Tuple[str, int, int]] = []
        if not self.root.exists():
            return out
        for child in sorted(self.root.iterdir()):
            if child.is_dir():
                files = [f for f in child.rglob("*") if f.is_file()]
                out.append((child.name + "/", len(files), sum(f.stat().st_size for f in files)))
            elif child.is_file():
                out.append((child.name, 1, child.stat().st_size))
            if len(out) >= max_entries:
                break
        return out
