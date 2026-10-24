"""SFinGe 后端 —— 把原项目验证过的合成内核接入 v2.0 插件接口。

合成链（全部来自 ``synthfp.legacy.render``）::

    手指轮廓 5 参数模型          FingerShapeModel
        -> 奇异点 Core/Delta     SingularPointGenerator
        -> 方向场 Zero-Pole      OrientationFieldModel / arch_model
        -> 频率图 Perlin 多层     FrequencyMapGenerator
        -> 脊线 Gabor 迭代        RidgeGenerator

本模块是**纯组合层**：不含任何新算法，只负责把配置映射为内核参数、
把内核输出打包成 :class:`MasterArtifact`。这样原始的算法文件保持零改动，
便于与本项目的历史版本做逐像素对比（可复现性验证）。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from ...core.types import MasterArtifact, MasterSpec
from ...legacy.render.fingerprint_class import FingerprintClass
from ...legacy.render.frequency_map import FrequencyMapGenerator
from ...legacy.render.orientation_field import OrientationFieldModel
from ...legacy.render.ridge_generator import RidgeGenerator, build_gabor_bank
from ...legacy.render.shape_model import FingerShapeModel, FingerShapeParams
from ...legacy.render.singular_points import SingularPointGenerator
from .base import BackendRegistry, FingerprintBackend

logger = logging.getLogger(__name__)

__all__ = ["SFinGeBackend"]

#: 字符串类别 -> 枚举
_CLASS_MAP: Dict[str, FingerprintClass] = {
    "arch": FingerprintClass.ARCH,
    "left_loop": FingerprintClass.LEFT_LOOP,
    "right_loop": FingerprintClass.RIGHT_LOOP,
    "whorl": FingerprintClass.WHORL,
    "twin_loop": FingerprintClass.TWIN_LOOP,
}


def _superellipse_mask(width: int, height: int, n: float = 3.4) -> np.ndarray:
    """超椭圆手指轮廓掩模（``{0,255}`` uint8）。

    ``|x/a|^n + |y/b|^n <= 1``，``n=2`` 是椭圆，``n`` 越大越接近矩形。
    取 ``n≈3.4`` 得到「两侧微鼓、四角圆润」的指腹轮廓：既不像椭圆那样
    在窄画布上过度收窄，也不像矩形那样有生硬直边。

    Args:
        width / height: 轮廓包围盒（像素）。
        n: 超椭圆指数。
    """
    h, w = max(int(height), 8), max(int(width), 8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    a, b = max(w / 2.0, 1.0), max(h / 2.0, 1.0)
    value = np.abs((xx - cx) / a) ** n + np.abs((yy - cy) / b) ** n
    return np.where(value <= 1.0, np.uint8(255), np.uint8(0))


@BackendRegistry.register
class SFinGeBackend(FingerprintBackend):
    """SFinGe 合成内核后端（虚拟指纹全合成，无需任何真实指纹输入）。"""

    name = "sfinge"
    version = "2.0"
    description = "SFinGe 复刻内核：轮廓→奇异点→零极点方向场→Perlin频率图→Gabor迭代脊线"

    # -- 主入口 -----------------------------------------------------------
    def warmup(self) -> None:
        """预构建 Gabor 核库（多进程 worker 首次调用时省一轮构建）。"""
        try:
            lo, hi = self._freq_range()
            build_gabor_bank(lo, hi, 36, 20, int(self.config.get("gabor_filter_size", 10)))
        except Exception:  # pragma: no cover - 预热失败不影响生成
            logger.debug("Gabor 核库预热失败", exc_info=True)

    def generate_master(self, spec: MasterSpec) -> MasterArtifact:
        """渲染主指纹。"""
        rng = np.random.default_rng(int(spec.seed))
        width, height = int(spec.image_size[0]), int(spec.image_size[1])
        fp_class = spec.fp_class if spec.fp_class in _CLASS_MAP else "left_loop"

        # --- 1. 手指轮廓 5 参数模型 ---
        # 注意：FingerShapeModel.generate(image_size=...) 会把小图**放大**填满画布，
        # 因此这里按缩小后的尺寸生成，再居中贴回目标画布，才能得到留白与边距。
        place = self._placement(width, height)
        raw_shape = self._sample_shape(place[0], place[1], rng)
        mask = self._render_shape(raw_shape, place, (width, height))

        # --- 2. 奇异点 ---
        sp_gen = SingularPointGenerator(rng)
        jitter = float(self.config.get("singular_jitter", 0.13))
        singular_points = sp_gen.generate(_CLASS_MAP[fp_class], (width, height), jitter=jitter)

        # --- 3. 方向场 ---
        orientation = self._orientation(singular_points, fp_class, width, height, rng)

        # --- 4. 频率图 ---
        lo_freq, hi_freq = self._freq_range()
        base_freq = (lo_freq + hi_freq) / 2.0
        spread = float(self.config.get("freq_spread", 0.25))
        freq_var = abs(hi_freq - lo_freq) / 2.0 * spread
        freq_gen = FrequencyMapGenerator(base_frequency=base_freq, freq_variation=freq_var)
        frequency = freq_gen.generate(
            height, width, [(sp.x, sp.y) for sp in singular_points], rng
        )

        # --- 5. 脊线合成 ---
        ridge_gen = (
            RidgeGenerator(
                min_freq=lo_freq,
                max_freq=hi_freq,
                gabor_filter_size=int(self.config.get("gabor_filter_size", 10)),
                convergence_eps=float(self.config.get("convergence_eps", 1e-3)),
                max_iterations=int(self.config.get("max_iterations", 80)),
                seed_density=float(self.config.get("seed_density", 1e-3)),
                n_frequencies=int(self.config.get("gabor_n_frequencies", 5)),
                coarse_scale=float(self.config.get("coarse_scale", 0.7)),
            )
            .set_ridge_color(*self.config.get("ridge_gray", (25, 70)))
            .set_valley_color(*self.config.get("valley_gray", (210, 248)))
            .set_texture_strength(float(self.config.get("texture_strength", 10.0)))
            .set_noise_sigma(float(self.config.get("noise_sigma", 1.5)))
            .set_mask_feather_width(int(self.config.get("mask_feather", 12)))
            .set_ridge_only_black(bool(self.config.get("ridge_only_black", False)))
        )
        image = ridge_gen.generate(orientation, frequency, mask, rng)
        image[~(mask > 0)] = 255

        return MasterArtifact(
            master_id=spec.master_id,
            fp_class=fp_class,
            image=image,
            mask=mask,
            orientation=orientation.astype(np.float32),
            frequency=frequency.astype(np.float32),
            singular_points=[
                {"x": round(float(sp.x), 2), "y": round(float(sp.y), 2), "type": sp.sp_type}
                for sp in singular_points
            ],
            shape_params={
                "sL": raw_shape.sL,
                "sR": raw_shape.sR,
                "sT": raw_shape.sT,
                "sB": raw_shape.sB,
                "sM": raw_shape.sM,
            },
            seed=int(spec.seed),
            backend=self.name,
            backend_version=self.version,
        )

    # -- 内核参数映射 -----------------------------------------------------
    def _freq_range(self) -> Tuple[float, float]:
        """由脊宽范围换算频率范围（``freq = 1 / width``）。"""
        min_w = max(int(self.config.get("min_ridge_width", 5)), 2)
        max_w = max(int(self.config.get("max_ridge_width", 15)), min_w + 1)
        return 1.0 / max_w, 1.0 / min_w

    def _sample_shape(
        self, canvas_w: int, canvas_h: int, rng: np.random.Generator
    ) -> FingerShapeParams:
        """按（缩小后的）画布比例采样 5 参数轮廓，单位为该画布上的像素。

        比例化表达（而不是固定像素）保证换分辨率时手指形态不变：

        * ``finger_width_frac`` / ``finger_height_frac``：手指占画布的比例；
        * ``aspect_bias``：左右半宽分配（拇指略不对称）；
        * ``top_cap_frac`` / ``bottom_cap_frac``：上下弧高相对**手指宽度**的比例。

        SFinGe 的 5 参数模型中，中段是矩形、上下是半椭圆帽，
        因此 ``sT``/``sB`` 越大手指越"胖"；取 0.24~0.38 倍宽度接近真实指形。
        """
        w_frac = rng.uniform(*self.config.get("finger_width_frac", (0.56, 0.68)))
        h_frac = rng.uniform(*self.config.get("finger_height_frac", (0.82, 0.90)))
        bias = float(rng.uniform(*self.config.get("aspect_bias", (0.46, 0.54))))
        top_frac = float(rng.uniform(*self.config.get("top_cap_frac", (0.24, 0.38))))
        bot_frac = float(rng.uniform(*self.config.get("bottom_cap_frac", (0.18, 0.30))))

        finger_w = max(int(round(canvas_w * w_frac)), 24)
        finger_h = max(int(round(canvas_h * h_frac)), 48)

        sL = max(int(round(finger_w * bias)), 10)
        sR = max(finger_w - sL, 10)
        sT = max(int(round(finger_w * top_frac)), 6)
        sB = max(int(round(finger_w * bot_frac)), 6)
        # 中段吃掉剩余高度；三者之和正好等于目标高度
        sM = max(finger_h - sT - sB, 16)

        return FingerShapeParams(sL=sL, sR=sR, sT=sT, sB=sB, sM=sM)

    def _fill_ratio(self) -> float:
        return float(min(max(self.config.get("fill_ratio", 0.88), 0.4), 1.0))

    def _placement(self, width: int, height: int) -> tuple:
        """缩小画布尺寸 (w, h)。

        ``FingerShapeModel.generate`` 会把给定画布填满，因此先把画布缩到
        ``fill_ratio`` 倍生成轮廓，再居中贴回原尺寸画布，即可得到稳定留白。
        """
        fill = self._fill_ratio()
        return max(int(round(width * fill)), 32), max(int(round(height * fill)), 48)

    def _render_shape(
        self, shape: FingerShapeParams, place: tuple, target: tuple
    ) -> np.ndarray:
        """在缩小画布上生成轮廓，再居中贴回目标尺寸画布。

        ``shape_style`` 控制轮廓模型：

        * ``"round"``（默认）：超椭圆轮廓。SFinGe 的 5 参数模型是「矩形 + 上下半椭圆帽」，
          中段两侧是平行的直边，贴到方形画布上会像一块圆角砖。真实捺印的手指
          轮廓整体呈椭圆，因此这里用超椭圆在同样的包围盒内画一个连续闭合曲线，
          与 5 参数模型的包围盒完全一致，只是把直边收成弧边。
        * ``"box"``：严格使用原版 5 参数模型（便于与历史版本逐像素对照）。
        """
        style = str(self.config.get("shape_style", "round")).lower()
        if style == "box":
            small = FingerShapeModel({}).generate(shape, image_size=place)
            small = (small > 127).astype(np.uint8) * 255
        else:
            small = _superellipse_mask(shape.sL + shape.sR, shape.sT + shape.sB + shape.sM)

        full_w, full_h = int(target[0]), int(target[1])
        canvas = np.zeros((full_h, full_w), dtype=np.uint8)
        oy = max((full_h - small.shape[0]) // 2, 0)
        ox = max((full_w - small.shape[1]) // 2, 0)
        canvas[oy : oy + small.shape[0], ox : ox + small.shape[1]] = small
        return canvas

    def _orientation(
        self,
        singular_points: List,
        fp_class: str,
        width: int,
        height: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """方向场：无奇异点的类别走正弦模型，其余走 Zero-Pole 模型。

        弓型（arch）没有 Core/Delta，零极点模型会退化成均匀场（直条纹），
        因此按设计文档 4.2 使用正弦模型。
        """
        if fp_class == "arch" or not singular_points:
            beta = float(rng.uniform(0.3, 0.7))
            freq = float(rng.uniform(1.5, 2.5))
            return OrientationFieldModel.arch_model(height, width, beta=beta, freq=freq)

        cores = [(sp.x, sp.y) for sp in singular_points if sp.sp_type == "core"]
        deltas = [(sp.x, sp.y) for sp in singular_points if sp.sp_type == "delta"]
        bg = float(rng.uniform(-0.12, 0.12))
        return OrientationFieldModel(cores=cores, deltas=deltas, bg_angle=bg).compute(height, width)
