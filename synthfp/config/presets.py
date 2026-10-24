"""预设配置 —— 设计文档 F12（配置保存/加载/预设）。

预设以「对默认配置的覆盖片段」形式存在，而不是独立 YAML 副本，
这样 schema 升级时预设自动跟随，不会腐化。
``presets/<name>.yaml`` 若存在则作为附加覆盖读取。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .loader import _deep_merge, apply_overrides, load_config
from .schema import AppConfig

__all__ = ["Preset", "PRESETS", "preset_names", "get_preset", "build_config", "preset_path", "presets_dir"]

#: 项目内预设 YAML 目录（可被用户覆盖）
PRESETS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "presets"


@dataclass(frozen=True)
class Preset:
    """一个预设：名称 + 中文说明 + 覆盖片段 + 标签。"""

    key: str
    label: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    tags: tuple = ()

    def build(self, base: Optional[AppConfig] = None) -> AppConfig:
        """基于 ``base``（默认配置）构造出该预设的完整配置。"""
        cfg = base.model_copy(deep=True) if base is not None else AppConfig()
        # 先应用 YAML 附加覆盖（若存在）
        extra = _load_preset_yaml(self.key)
        merged = dict(self.overrides)
        if extra:
            merged = _deep_merge(merged, extra)
        return apply_overrides(cfg, merged)


def _load_preset_yaml(name: str) -> Dict[str, Any]:
    path = PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 内置预设
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Preset] = {
    "default": Preset(
        key="default",
        label="标准数据集（推荐）",
        description="10 身份 × 2 指 × 2 捺印 × 3 TTA = 120 样本，自然类别分布，含质量与细节点。",
        overrides={},
        tags=("通用",),
    ),
    "quick_test": Preset(
        key="quick_test",
        label="快速自检",
        description="2 身份的小规模配置，用于验证环境与参数，几十秒内跑完。",
        overrides={
            "dataset": {
                "name": "quick_test",
                "identities": 2,
                "fingers_per_identity": 1,
                "impressions_per_finger": 1,
                "tta_per_impression": 2,
            },
            "backend": {"resolution": "finger_500dpi"},
            "runtime": {"workers": 2},
        },
        tags=("调试",),
    ),
    "large_research": Preset(
        key="large_research",
        label="科研大规模集",
        description="500 身份 × 4 指 × 3 捺印 × 4 TTA = 24000 样本，含划分与校验和。",
        overrides={
            "dataset": {
                "name": "synth_fp_large",
                "identities": 500,
                "fingers_per_identity": 4,
                "impressions_per_finger": 3,
                "tta_per_impression": 4,
            },
            "runtime": {"workers": 8, "resume": True},
            "split": {"enabled": True, "write_csv": True},
            "export": {"write_sha256": True, "write_stats": True},
        },
        tags=("科研", "大批量"),
    ),
    "high_quality_only": Preset(
        key="high_quality_only",
        label="高质量筛选",
        description="质量阈值 60 分，低于阈值的样本直接丢弃，适合训练清晰度敏感的任务。",
        overrides={
            "dataset": {"name": "synth_fp_hq"},
            "quality": {"enabled": True, "min_score": 60.0, "reject_on_fail": True},
            "impression": {"stress_enabled": False},
            "tta": {"max_distortion": 0.4, "operators": ["rotate", "scale", "brightness", "contrast"]},
        },
        tags=("质量",),
    ),
    "robust_low_quality": Preset(
        key="robust_low_quality",
        label="低质量鲁棒性",
        description="重度捺印退化 + 全部噪声/遮挡 TTA，用于鲁棒性与拒识研究。",
        overrides={
            "dataset": {"name": "synth_fp_robust"},
            "impression": {
                "stress_enabled": True,
                "stress_num_operators": [2, 4],
                "sensor_strength": 1.6,
            },
            "tta": {
                "mode": "chain",
                "chain_length": [2, 3],
                "max_distortion": 0.9,
                "operators": [
                    "rotate", "elastic", "blur", "motion_blur", "gaussian_noise",
                    "jpeg", "occlusion_rect", "scratch",
                ],
            },
            "quality": {"enabled": True, "min_score": 0.0, "reject_on_fail": False},
        },
        tags=("鲁棒性",),
    ),
    "class_balanced": Preset(
        key="class_balanced",
        label="类别均衡",
        description="五类指纹各占 20%，用于分类任务的类别平衡训练集。",
        overrides={
            "dataset": {"name": "synth_fp_balanced", "identities": 50, "fingers_per_identity": 2},
            "classes": {
                "distribution": {
                    "arch": 0.2,
                    "left_loop": 0.2,
                    "right_loop": 0.2,
                    "whorl": 0.2,
                    "twin_loop": 0.2,
                }
            },
            "split": {"enabled": True, "strategy": "by_identity"},
        },
        tags=("分类",),
    ),
    "whorl_only": Preset(
        key="whorl_only",
        label="仅螺型",
        description="只生成 Whorl（螺型）指纹，用于单类别深度研究。",
        overrides={
            "dataset": {"name": "synth_fp_whorl"},
            "classes": {"distribution": {"whorl": 1.0}},
        },
        tags=("单类别",),
    ),
    "arch_only": Preset(
        key="arch_only",
        label="仅弓型",
        description="只生成 Arch（弓型）指纹（无奇异点，正弦方向场模型）。",
        overrides={
            "dataset": {"name": "synth_fp_arch"},
            "classes": {"distribution": {"arch": 1.0}},
        },
        tags=("单类别",),
    ),
    "loop_only": Preset(
        key="loop_only",
        label="仅箕型",
        description="只生成左/右箕型指纹，最常见的指纹形态。",
        overrides={
            "dataset": {"name": "synth_fp_loop"},
            "classes": {"distribution": {"left_loop": 0.5, "right_loop": 0.5}},
        },
        tags=("单类别",),
    ),
    "no_tta": Preset(
        key="no_tta",
        label="纯捺印（无 TTA）",
        description="只生成原始捺印图像，不做任何 TTA 变换，用于获得干净基底。",
        overrides={
            "dataset": {"name": "synth_fp_plain", "tta_per_impression": 1},
            "tta": {"enabled": False},
        },
        tags=("基底",),
    ),
}


def preset_names() -> List[str]:
    return list(PRESETS)


def get_preset(name: str) -> Optional[Preset]:
    return PRESETS.get(name)


def build_config(name: str = "default", base: Optional[AppConfig] = None) -> AppConfig:
    """按预设名构造配置对象。"""
    preset = PRESETS.get(name)
    if preset is None:
        raise KeyError(f"未知预设 '{name}'；可用: {list(PRESETS)}")
    return preset.build(base)


def preset_path(name: str) -> Optional[Path]:
    p = PRESETS_DIR / f"{name}.yaml"
    return p if p.exists() else None


def presets_dir() -> Path:
    return PRESETS_DIR


def load_default_config() -> AppConfig:
    """加载 ``configs/default.yaml``；不存在时回退到内置默认值。"""
    path = PRESETS_DIR.parent / "default.yaml"
    if path.exists():
        try:
            return load_config(path)
        except Exception:
            pass
    return AppConfig()
