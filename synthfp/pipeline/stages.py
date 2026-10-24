"""管道阶段 —— 每个阶段独立可测（设计文档 4.4）。

管道顺序（与设计文档一致，但把主指纹生成提到最前以便复用）::

    Backend.generate_master()        # 每个手指一次
        -> Impression.apply()        # 每个捺印一次（退化链 -> 按压变形 -> 传感器）
        -> TTA.apply()               # 每个样本一次
        -> Minutiae.extract()
        -> Quality.evaluate()
        -> Storage.save() + Manifest

每个阶段的开关都在配置里，关闭即跳过。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config.schema import AppConfig
from ..core.types import MasterArtifact, Minutia, QualityReport

logger = logging.getLogger(__name__)

__all__ = [
    "apply_impression",
    "apply_tta",
    "extract_minutiae",
    "evaluate_quality",
    "estimate_orientation",
    "build_stress_composer",
    "minutiae_payload",
    "STRESS_PRESETS",
]


# ---------------------------------------------------------------------------
# 捺印阶段的辅助
# ---------------------------------------------------------------------------
def _background_level(image: np.ndarray) -> float:
    """估计采集背景灰度（指纹图背景为亮色，取 98 分位）。"""
    try:
        return float(np.percentile(image, 98.0))
    except Exception:  # pragma: no cover
        return 255.0


def _edge_feather(foreground: np.ndarray, pad: int = 2, sigma: float = 1.2) -> np.ndarray:
    """前景布尔掩模 → 带羽化的 ``[0,1]`` 权重图（避免合成处出现硬边）。"""
    f = foreground.astype(np.uint8)
    if pad > 0:
        k = np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8)
        f = cv2.dilate(f, k, iterations=1)
    weight = cv2.GaussianBlur(f.astype(np.float32), (0, 0), sigma)
    return np.clip(weight, 0.0, 1.0)


def _restore_background(
    original: np.ndarray,
    warped: np.ndarray,
    valid: np.ndarray,
    pad: int = 1,
    sigma: float = 1.0,
) -> np.ndarray:
    """把 ``warped`` 中有效区域之外的部分还原为 ``original``。

    ``cv2.remap`` 的 ``BORDER_REPLICATE`` 会把边界像素向外拖出一条亮带；
    这里用「有效源区域」掩模按羽化权重把它换回原图，既保留手指边缘的
    抗锯齿过渡，又不会留下拖影。
    """
    if valid.shape != warped.shape[:2]:
        valid = cv2.resize(
            valid.astype(np.uint8), (warped.shape[1], warped.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    weight = _edge_feather(valid, pad=pad, sigma=sigma)
    base = original.astype(np.float32)
    if warped.ndim == 3 and warped.shape[2] > 1:
        weight = weight[..., None]
        if base.ndim == 2:
            base = base[..., None]
    blended = warped.astype(np.float32) * weight + base * (1.0 - weight)
    return np.clip(blended, 0, 255).astype(original.dtype)


def _shift_valid_region(valid: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """计算「平移后仍在图像内」的源像素集合（用于判断 remap 输出是否可信）。"""
    h, w = valid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    tx = np.rint(xx + dx).astype(np.int32)
    ty = np.rint(yy + dy).astype(np.int32)
    inside = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    out = np.zeros((h, w), dtype=bool)
    np.logical_and(valid, inside, out=out)
    return out


def _calibrated_pressure(imp, image: np.ndarray, rng: np.random.Generator) -> float:
    """把配置的「按压力度」换算为插件层的压力参数。

    插件层的双曲正切模型在 ``pressure=1`` 时中心位移约为手指尺寸的 1/3，
    量级过大（会把指纹裹成一个球）。真实捺印的按压形变通常只有几个像素，
    因此这里把力度按 ``influence_ratio`` 折算：
    ``pressure_eff = ratio * (shorter_side / 320) * p``；
    320px 短边、p=0.4、比例 0.20 ⇒ 等效 0.08，中心位移约 8 px，符合实际。
    """
    lo, hi = imp.pressure_range
    p = float(rng.uniform(lo, hi))
    shorter = float(min(image.shape[:2])) if image.ndim >= 2 else 320.0
    ratio = float(getattr(imp, "influence_ratio", 0.20))
    return float(np.clip(p * ratio * (shorter / 320.0), 0.0, 0.5))


# ---------------------------------------------------------------------------
# 捺印阶段：退化链 -> 按压变形 -> 传感器
# ---------------------------------------------------------------------------
#: 原项目退化算子在捺印场景下的推荐参数（保守档，避免过度损坏）
STRESS_PRESETS: Dict[str, Dict[str, Any]] = {
    "ridge_breaks": {"density": [0.004, 0.022], "break_width": [1, 3], "break_types": ["point", "line"]},
    "dirt_stain": {"num_stains": [1, 5], "size_range": [12, 55], "opacity": [0.12, 0.40],
                   "darkness": [0.05, 0.25]},
    "ink_smudge": {"intensity": [0.08, 0.30]},
    "blemish_spot": {"num_spots": [1, 4], "size_range": [6, 28], "severity": [0.15, 0.50],
                     "types": ["dark", "light", "blur"]},
    "ridge_edge_roughness": {"roughness": [0.04, 0.20]},
    "ghosting": {"num_ghosts": [1, 2], "offset_px": [10, 40], "opacity": [0.06, 0.20],
                 "angle_deg": [-12, 12], "blur_sigma": [0.5, 2.0]},
    "partial_missing": {"area_ratio": [0.04, 0.18], "num_regions": [1, 2],
                        "positions": ["edge"], "fill_types": ["background"]},
    "sensor_noise": {"fpn_std": [0.004, 0.018], "dead_pixel_ratio": [0.0, 0.0008],
                     "row_noise_std": [0.0, 0.010], "thermal_std": [0.004, 0.014]},
    "defocus_blur": {"radius": [2, 8], "aperture_blades": 0},
    "motion_blur": {"kernel_sizes": [5, 7, 9]},
    "edge_feathering": {"feather_width": [8, 40], "intensity": [0.15, 0.60],
                        "asymmetric_prob": 0.5, "direction_bias": [0, 360]},
}


def build_stress_composer(config: AppConfig):
    """按配置装配捺印退化链组合器（复用原项目 19 个退化算子）。

    Returns:
        ``DegradationComposer`` 实例；未启用时返回 ``None``。
    """
    imp = config.impression
    if not imp.stress_enabled:
        return None
    operators: Dict[str, Dict[str, Any]] = {}
    for name, enabled in imp.stress_operators.items():
        if not enabled:
            continue
        if name not in STRESS_PRESETS:
            logger.debug("忽略未知捺印退化算子: %s", name)
            continue
        operators[name] = {"enabled": True, **STRESS_PRESETS[name]}
    if not operators:
        return None

    from ..legacy.degrade.composer import DegradationComposer

    cfg = {"num_operators": tuple(imp.stress_num_operators), "operators": operators}
    return DegradationComposer(cfg, np.random.default_rng(0))


def apply_impression(
    master: MasterArtifact,
    image: np.ndarray,
    rng: np.random.Generator,
    config: AppConfig,
    composer=None,
    sample_key: str = "",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """把一个主指纹渲染成一次捺印。

    顺序：随机退化链（干湿/污渍/断线）→ 按压变形 → 传感器特性。
    退化链放在最前，因为它是「手指状态」，随后的几何变形与传感器响应作用其上。

    Args:
        master: 主指纹产物（提供初始掩模）。
        image: 当前捺印图像（uint8，来自主指纹或上一次迭代）。
        rng: 随机源。
        config: 配置。
        composer: 退化链组合器（``None`` 表示跳过）。
        sample_key: 用于退化链派生稳定种子的标识。

    Returns:
        ``(image, effects)``，``effects`` 为可写入 manifest 的效果记录列表。
    """
    imp = config.impression
    effects: List[Dict[str, Any]] = []
    out = image.copy()

    # --- 1. 随机退化链 ---
    if composer is not None:
        try:
            f = out.astype(np.float32) / 255.0
            degraded, param_log = composer.apply_chain(f, sample_id=sample_key)
            out = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
            for item in param_log:
                effects.append({"stage": "stress", **item})
        except Exception as exc:  # 退化链失败不应中断整条样本
            logger.warning("捺印退化链失败，已跳过: %s", exc)
            effects.append({"stage": "stress", "skipped": str(exc)})

    # --- 2. 按压变形 ---
    if imp.deform_enabled:
        try:
            from ..plugins.impressions.deform import deform_image, impression_transform

            h, w = out.shape[:2]
            shift = (
                float(rng.uniform(-imp.translation_px, imp.translation_px)),
                float(rng.uniform(-imp.translation_px, imp.translation_px)),
            )
            rotation = float(rng.uniform(*imp.rotation_range))
            pressure = _calibrated_pressure(imp, out, rng)
            fg = master.mask > 0
            if fg.shape != (h, w):  # pragma: no cover
                fg = cv2.resize(fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

            dx, dy = impression_transform(
                h,
                w,
                pressure=pressure,
                rotation=rotation,
                shift=shift,
                rng=rng,
                mask=master.mask,
                elastic_sigma=float(imp.elastic_sigma),
                elastic_alpha=float(imp.elastic_alpha),
            )
            warped = deform_image(out, dx, dy)
            # 位移后"没有有效内容"的区域要还原成**采集背景**，而不是变形前的图像。
            #
            # 还原成原图会在手指边缘留下一道细弧线（旧图案与位移后图案的交界），
            # 看上去像划痕；还原成背景才是物理上正确的（那部分根本没接触传感器）。
            out = _restore_background(
                _background_canvas(out), warped, _shift_valid_region(fg, dx, dy)
            )

            effects.append(
                {
                    "stage": "deform",
                    "rotation_deg": round(rotation, 2),
                    "shift_px": [round(shift[0], 2), round(shift[1], 2)],
                    "pressure": round(pressure, 4),
                    "max_shift_px": round(float(max(np.abs(dx).max(), np.abs(dy).max())), 2),
                }
            )
        except Exception as exc:
            logger.warning("按压变形失败，已跳过: %s", exc)
            effects.append({"stage": "deform", "skipped": str(exc)})

    # --- 3. 传感器特性 ---
    if imp.sensor_enabled and imp.sensor_type != "none":
        try:
            from ..plugins.impressions.sensor import get_sensor

            sensor = get_sensor(imp.sensor_type)
            noisy = sensor.apply(out, rng, strength=float(imp.sensor_strength))
            # 传感器噪声是**全画幅**的，但采集背景应当是干净的白：只保留手指
            # 区域内的噪声结果，区域外还原为采集背景色。否则手指周围会出现
            # 一圈灰噪，细节点检测还会在噪声里报出大量伪点。
            out = _restore_background(
                _background_canvas(out), noisy, master.mask > 0, pad=2, sigma=1.5
            )
            effects.append(
                {"stage": "sensor", "type": imp.sensor_type, "strength": imp.sensor_strength}
            )
        except Exception as exc:
            logger.warning("传感器模拟失败，已跳过: %s", exc)
            effects.append({"stage": "sensor", "skipped": str(exc)})

    return out, effects


def _background_canvas(image: np.ndarray) -> np.ndarray:
    """构造"纯采集背景"图：与输入同形状，整幅填充背景灰度。

    背景灰度取整幅的 98 分位（指纹图背景是亮色），并向上取到至少 250，
    避免把手指边缘的亮像素误当背景而拉低整体亮度。
    """
    level = float(np.percentile(image, 98.0))
    level = max(level, 250.0)
    return np.full(image.shape, min(level, 255.0), dtype=image.dtype)


# ---------------------------------------------------------------------------
# TTA 阶段
# ---------------------------------------------------------------------------
def apply_tta(
    image: np.ndarray,
    operators: Sequence[Any],
    config: AppConfig,
    tta_index: int = 0,
    mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Any], float]:
    """按算子链对图像施加 TTA。

    Args:
        image: 捺印图像（uint8）。
        operators: 已实例化的 TTA 算子序列（可为空 = 原样输出）。
        config: 配置。
        tta_index: 样本的 TTA 序号（仅用于日志与默认种子派生）。
        mask: 捺印掩模。给出时变换后的背景会被还原为原背景，
            避免几何算子把背景像素拖进画面（采集背景始终是干净的白）。
        seed: **样本种子**。TTA 算子内部的随机参数由它派生，
            因此同一 ``sample_id`` 在任何机器上都会得到逐像素相同的结果。

    Returns:
        ``(image, operator_names, params, distortion)``。
    """
    if not operators:
        return image.copy(), [], {}, 0.0

    src = image.astype(np.float32) / 255.0
    names: List[str] = []
    params: Dict[str, Any] = {}
    rng = np.random.default_rng(_tta_rng_seed(config, tta_index, operators, seed))

    out = src
    for op in operators:
        try:
            out = op.apply(out, rng)
            names.append(op.name)
            params[op.name] = dict(getattr(op, "last_params", {}) or {})
        except Exception as exc:
            logger.warning("TTA 算子 %s 失败，已跳过: %s", getattr(op, "name", "?"), exc)

    out = np.clip(out, 0.0, 1.0)
    img_u8 = (out * 255.0).astype(np.uint8)

    if mask is not None:
        # 区域之外一律还原为**纯采集背景**，而不是原图。
        #
        # 若还原成原图，几何算子（旋转 / 缩放 / 平移）会把"未变换的手指"残留在
        # 变换后手指之外，形成一块倾斜的矩形鬼影；亮度/对比度算子还会把背景
        # 一起压灰。用背景画布合成后，背景始终是干净的白，掩模也保持有意义。
        img_u8 = _restore_background(
            _background_canvas(image), img_u8, _dilate_bool(mask > 0, 3), pad=0, sigma=1.5
        )

    distortion = _chain_distortion(operators, config)
    return img_u8, names, params, distortion


def finalize_image(image: np.ndarray, config: AppConfig) -> np.ndarray:
    """管道收口：按配置把最终样本图定型。

    目前只有一件事 —— ``backend.ridge_only_black`` 打开时把图像二值化。
    主指纹阶段已经输出二值图，但捺印（传感器噪声 / 纹理）与 TTA（模糊、对比度）
    会重新引入灰度，所以必须在**最后**再收一次，保证落盘产物是真正的二值图。
    """
    if getattr(config.backend, "ridge_only_black", False):
        return np.where(np.asarray(image) < 128, 0, 255).astype(np.uint8)
    return image


def _dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    """布尔掩模膨胀 ``radius`` 像素（用于覆盖边缘抗锯齿像素）。"""
    if radius <= 0:
        return mask
    k = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k, iterations=1) > 0


def _tta_rng_seed(
    config: AppConfig, tta_index: int, operators: Sequence[Any] = (), seed: Optional[int] = None
) -> int:
    """TTA 算子内部随机参数的种子。

    优先使用**样本种子**（manifest 中可查，保证跨机可复现）；
    未提供时退回配置种子 + TTA 序号派生。
    """
    from ..core.seeds import derive

    names = "|".join(str(getattr(op, "name", "?")) for op in operators)
    if seed is not None:
        return derive(int(seed), "tta-apply", names)
    return derive(config.runtime.seed, "tta-apply", tta_index, names)


def _chain_distortion(operators: Sequence[Any], config: AppConfig) -> float:
    """累计失真度（用于配置的 ``max_distortion`` 上限约束）。"""
    total = sum(float(getattr(op, "distortion", 0.2)) for op in operators)
    cap = max(float(config.tta.max_distortion), 1e-6)
    return float(min(total / cap, 1.0)) if operators else 0.0


# ---------------------------------------------------------------------------
# 细节点 与 质量
# ---------------------------------------------------------------------------
def estimate_orientation(image: np.ndarray) -> np.ndarray:
    """对最终样本图估计方向场（弧度 ``[0, π)``）。"""
    from ..plugins.quality.metrics import estimate_orientation as _est

    gray = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    orientation, _ = _est(gray)
    return orientation.astype(np.float32)


def extract_minutiae(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    orientation: Optional[np.ndarray],
    config: AppConfig,
) -> List[Minutia]:
    """提取细节点（关闭时返回空列表）。"""
    if not config.minutiae.enabled:
        return []
    from ..plugins.minutiae.extractor import MinutiaeExtractor

    extractor = MinutiaeExtractor(
        min_distance=float(config.minutiae.min_distance),
        border_margin=int(config.minutiae.border_margin),
        max_count=int(config.minutiae.max_count),
        compute_angle=bool(config.minutiae.compute_angle),
    )
    return list(extractor.extract(image, mask, orientation))


def evaluate_quality(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    orientation: Optional[np.ndarray],
    config: AppConfig,
) -> Optional[QualityReport]:
    """质量评估（关闭时返回 ``None``）。

    ``count_minutiae=False``：管道自己会提取一遍细节点并回填
    ``QualityReport.minutiae_count``，因此不必在评分内部重复提取。
    """
    if not config.quality.enabled:
        return None
    from ..plugins.quality.evaluator import QualityEvaluator

    evaluator = QualityEvaluator(
        min_score=float(config.quality.min_score),
        reject_on_fail=bool(config.quality.reject_on_fail),
        min_usable_area=float(config.quality.min_usable_area),
        count_minutiae=False,
    )
    return evaluator.evaluate(image, mask, orientation)


def minutiae_payload(
    spec_id: str,
    minutiae: Sequence[Minutia],
    width: int,
    height: int,
    ppi: int,
    fp_class: str,
    quality: Optional[QualityReport] = None,
    singular_points: Optional[Sequence[Dict[str, Any]]] = None,
    shape_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """细节点 JSON 载荷。

    除细节点本身，还带上奇异点与轮廓参数 —— 这样单张样本的 JSON 就自包含，
    下游做可视化或做几何研究时不必回头查 manifest。
    """
    return {
        "sample_id": spec_id,
        "width": int(width),
        "height": int(height),
        "ppi": int(ppi),
        "class_label": fp_class,
        "n_minutiae": len(minutiae),
        "n_endings": sum(1 for m in minutiae if m.kind == "ending"),
        "n_bifurcations": sum(1 for m in minutiae if m.kind == "bifurcation"),
        "minutiae": [m.to_dict() for m in minutiae],
        "singular_points": list(singular_points or []),
        "shape_params": dict(shape_params or {}),
        "quality": quality.to_dict() if quality else None,
        "format": "synthfp-minutiae-v1",
    }


def dump_minutiae_json(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
