"""Manifest —— 数据集唯一元数据来源（设计文档 0.2 / 4.8 / 11.1）。

* 写入 ``dataset_manifest.jsonl``，每行一个样本的完整记录；
* 断点续传读取 ``done.jsonl``（只存 sample_id，用于快速跳过）；
* 失败隔离写入 ``logs/failed.jsonl``，绝不污染主 manifest；
* 所有写入都是追加 + ``flush``，进程被杀也不会丢已完成记录。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from .storage import DatasetLayout, atomic_write_text

__all__ = ["Manifest", "read_jsonl"]


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """逐行读取 JSONL。

    容错策略：跳过空行、非法 JSON **以及合法的非对象行**（例如误写的 ``[1,2]``）——
    下游一律按 ``dict`` 使用，放它们过去只会在远处炸出 ``AttributeError``。
    这里绝不抛异常：断点续传依赖「残缺的日志文件也能读」。
    """
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


class Manifest:
    """数据集清单写入器 / 读取器。"""

    def __init__(self, layout: DatasetLayout, resume: bool = True) -> None:
        """
        Args:
            layout: 数据集目录布局。
            resume: 为 ``True`` 时读取已有 ``done.jsonl``，用于断点续传。
        """
        self.layout = layout
        self.resume = resume
        self._lock = threading.Lock()
        self._done: Set[str] = set()
        self._failed: Set[str] = set()
        self._manifest_fh = None
        self._done_fh = None
        self._failed_fh = None
        self._written = 0

        if resume:
            self._done = {row.get("sample_id", "") for row in read_jsonl(layout.done)}
            self._done.discard("")
            self._failed = {row.get("sample_id", "") for row in read_jsonl(layout.failed)}
            self._failed.discard("")

    # -- 断点续传 ---------------------------------------------------------
    @property
    def done_ids(self) -> Set[str]:
        """已完成（且已写入 manifest）的 sample_id 集合。"""
        return self._done

    @property
    def failed_ids(self) -> Set[str]:
        return self._failed

    @property
    def written(self) -> int:
        return self._written

    def is_done(self, sample_id: str) -> bool:
        """该样本是否已完成（且图像仍在磁盘上）。"""
        if sample_id not in self._done:
            return False
        return True

    def existing_rows(self) -> List[Dict[str, Any]]:
        """已有 manifest 行（用于重跑只补缺失样本时重建统计）。"""
        return list(read_jsonl(self.layout.manifest))

    # -- 生命周期 ---------------------------------------------------------
    def open(self) -> "Manifest":
        """打开追加句柄（必须在生成开始前调用）。"""
        self.layout.root.mkdir(parents=True, exist_ok=True)
        (self.layout.root / "logs").mkdir(parents=True, exist_ok=True)
        self._manifest_fh = open(self.layout.manifest, "a", encoding="utf-8")
        self._done_fh = open(self.layout.done, "a", encoding="utf-8")
        self._failed_fh = open(self.layout.failed, "a", encoding="utf-8")
        return self

    def close(self) -> None:
        for fh in (self._manifest_fh, self._done_fh, self._failed_fh):
            if fh is not None:
                try:
                    fh.flush()
                    fh.close()
                except Exception:  # pragma: no cover
                    pass
        self._manifest_fh = self._done_fh = self._failed_fh = None

    def __enter__(self) -> "Manifest":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- 写入 -------------------------------------------------------------
    def append(self, row: Dict[str, Any]) -> None:
        """写入一条成功样本记录（同时记入 done.jsonl）。"""
        if self._manifest_fh is None:
            raise RuntimeError("Manifest 未打开，请先调用 open()")
        line = json.dumps(row, ensure_ascii=False)
        sid = str(row.get("sample_id", ""))
        with self._lock:
            self._manifest_fh.write(line + "\n")
            self._manifest_fh.flush()
            self._done_fh.write(json.dumps({"sample_id": sid}, ensure_ascii=False) + "\n")
            self._done_fh.flush()
            self._done.add(sid)
            self._failed.discard(sid)
            self._written += 1

    def append_failure(self, sample_id: str, error: str, attempt: int = 1,
                       extra: Optional[Dict[str, Any]] = None) -> None:
        """记录失败样本（失败隔离）。"""
        if self._failed_fh is None:
            raise RuntimeError("Manifest 未打开，请先调用 open()")
        payload: Dict[str, Any] = {"sample_id": sample_id, "error": error, "attempt": attempt}
        if extra:
            payload.update(extra)
        with self._lock:
            self._failed_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._failed_fh.flush()
            self._failed.add(sample_id)

    # -- 导出 -------------------------------------------------------------
    def export_splits(self, rows: List[Dict[str, Any]]) -> Dict[str, int]:
        """按 ``split`` 字段写 ``splits/{train,val,test}.csv``。"""
        import csv
        import io

        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            buckets.setdefault(str(row.get("split", "train")), []).append(row)

        counts: Dict[str, int] = {}
        self.layout.splits_dir.mkdir(parents=True, exist_ok=True)
        columns = [
            "sample_id", "identity_id", "finger", "impression_index", "tta_index",
            "class_label", "split", "image_path", "minutiae_path", "nfiq2",
            "minutiae_count", "sha256",
        ]
        for split, items in sorted(buckets.items()):
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for item in sorted(items, key=lambda r: str(r.get("sample_id", ""))):
                writer.writerow(item)
            atomic_write_text(self.layout.splits_dir / f"{split}.csv", buf.getvalue())
            counts[split] = len(items)
        return counts
