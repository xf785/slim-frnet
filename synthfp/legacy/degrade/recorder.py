"""退化参数记录器 — 记录完整退化链，保证可复现性"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class DegradationRecorder:
    """
    退化参数记录器

    记录格式 (JSON):
    {
      "sample_id": "...",
      "timestamp": "2026-07-06T10:30:00",
      "degradation_chain": [
        {"operator": "downsample", "params": {"scale": 2}},
        {"operator": "poisson_noise", "params": {"lam": 15.2}},
      ]
    }
    """

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        self._records: Dict[str, List[Dict]] = {}

    def record(
        self, sample_id: str, chain: List[Dict[str, Any]]
    ) -> None:
        """记录一个样本的退化链"""
        self._records[sample_id] = {
            "sample_id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "degradation_chain": chain,
        }

    def save(self, sample_id: str, output_path: Optional[Path] = None) -> None:
        """保存单个样本的记录"""
        if output_path is None and self.save_dir is None:
            return

        path = output_path or (self.save_dir / f"{sample_id}_degradation.json")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._records.get(sample_id, {}),
                f,
                indent=2,
                ensure_ascii=False,
            )

    def save_all(self) -> None:
        """保存所有记录的退化参数"""
        if self.save_dir is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)

        path = self.save_dir / "degradation_records.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for record in self._records.values():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self, sample_id: str) -> Optional[Dict]:
        """加载单个样本的退化记录"""
        if sample_id in self._records:
            return self._records[sample_id]
        if self.save_dir is not None:
            path = self.save_dir / f"{sample_id}_degradation.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self._records[sample_id] = json.load(f)
                return self._records[sample_id]
        return None

    def get_params(self, sample_id: str) -> Optional[List[Dict]]:
        """获取某样本的退化参数链"""
        record = self._records.get(sample_id)
        if record is None:
            record = self.load(sample_id)
        if record is None:
            return None
        return record.get("degradation_chain", [])
