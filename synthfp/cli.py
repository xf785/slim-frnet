"""CLI —— 设计文档 11.3 / F11。

仅使用标准库 ``argparse``（不引入 click），保持依赖最小。

::

    synthfp generate --preset quick_test
    synthfp generate --config my.yaml --set dataset.identities=50 --set runtime.workers=8
    synthfp plan     --preset default
    synthfp stats    --dataset outputs/synth_fp_v1
    synthfp validate --dataset outputs/synth_fp_v1
    synthfp presets
    synthfp show     --preset high_quality_only
    synthfp preview  --preset default --finger left_thumb --tta 1 --save out.png
    synthfp selfcheck
    synthfp gui
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .config import (
    PRESETS,
    AppConfig,
    ConfigError,
    apply_overrides,
    build_config,
    dump_yaml,
    load_config,
    load_default_config,
    preset_names,
)
from .core.storage import DatasetLayout

__all__ = ["main", "build_parser", "configure_stdio"]

_SET_HELP = "覆盖任意配置项，点号路径，可重复，例如 --set runtime.seed=7"


# ---------------------------------------------------------------------------
# 控制台编码
# ---------------------------------------------------------------------------
def configure_stdio() -> None:
    """让中文与符号（✓ ✗ ✗ 等）在任何 Windows 控制台里都能正常输出。

    Windows 的 ``cmd.exe`` 默认用本地代码页（简体中文为 GBK/936），
    此时 ``print("✓")`` 会抛 ``UnicodeEncodeError`` 直接崩掉 ——
    这在双击 ``start_cli.bat`` 的场景里非常容易遇到。

    做法分两步，任何一步失败都不影响程序继续运行：

    1. 把控制台输入/输出代码页切到 UTF-8（65001）；
    2. 把 ``stdout``/``stderr`` 重新配置为 UTF-8 且 ``errors="replace"``，
       这样即使终端不支持某些字符，也只会显示占位符而不会抛异常。
    """
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:  # pragma: no cover - 非控制台 / 受限环境
            pass

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # pragma: no cover - 被重定向成非文本流
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthfp",
        description="轻量级 SFinGe 合成指纹数据集合成软件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  synthfp generate --preset quick_test\n"
            "  synthfp generate --preset default --set dataset.identities=100\n"
            "  synthfp stats --dataset outputs/synth_fp_v1\n"
            "  synthfp gui\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"synthfp {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<命令>")

    def add_config_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--config", type=Path, default=None, help="YAML 配置文件路径")
        p.add_argument(
            "--preset", default=None, choices=preset_names(), help="使用内置预设"
        )
        p.add_argument("--set", dest="overrides", action="append", default=[], help=_SET_HELP)

    # -- generate --------------------------------------------------------
    p_gen = sub.add_parser("generate", help="批量生成数据集")
    add_config_args(p_gen)
    p_gen.add_argument("-o", "--output-root", type=Path, default=None, help="输出根目录")
    p_gen.add_argument("-n", "--identities", type=int, default=None, help="身份数")
    p_gen.add_argument("--fingers", type=int, default=None, help="每身份手指数")
    p_gen.add_argument("--impressions", type=int, default=None, help="每手指捺印数")
    p_gen.add_argument("--tta", type=int, default=None, help="每捺印 TTA 数")
    p_gen.add_argument("--seed", type=int, default=None, help="随机种子")
    p_gen.add_argument("-j", "--workers", type=int, default=None, help="并行工作单元数")
    p_gen.add_argument("--no-resume", action="store_true", help="禁用断点续传，从头生成")
    p_gen.add_argument("--no-export", action="store_true", help="生成后不执行导出阶段")
    p_gen.add_argument("--dry-run", action="store_true", help="只打印计划，不生成")
    p_gen.add_argument("-q", "--quiet", action="store_true", help="不打印进度日志")

    # -- plan ------------------------------------------------------------
    p_plan = sub.add_parser("plan", help="预览生成计划（不写任何文件）")
    add_config_args(p_plan)
    p_plan.add_argument("-o", "--output-root", type=Path, default=None)
    p_plan.add_argument("--json", action="store_true", help="以 JSON 输出")

    # -- stats -----------------------------------------------------------
    p_stats = sub.add_parser("stats", help="统计已有数据集")
    p_stats.add_argument("--dataset", type=Path, required=True, help="数据集目录（含 manifest）")
    p_stats.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_stats.add_argument("--write", action="store_true", help="回写 stats.json")

    # -- validate --------------------------------------------------------
    p_val = sub.add_parser("validate", help="校验数据集完整性")
    p_val.add_argument("--dataset", type=Path, required=True)
    p_val.add_argument("--check-hash", action="store_true", help="逐文件重算 SHA-256（较慢）")
    p_val.add_argument("--json", action="store_true")

    # -- presets ---------------------------------------------------------
    sub.add_parser("presets", help="列出内置预设")

    # -- show ------------------------------------------------------------
    p_show = sub.add_parser("show", help="打印配置内容（含预设解析结果）")
    add_config_args(p_show)
    p_show.add_argument("--resolved", action="store_true", help="输出完整 YAML")

    # -- preview ---------------------------------------------------------
    p_prev = sub.add_parser("preview", help="渲染单张样本预览（默认不落盘）")
    add_config_args(p_prev)
    p_prev.add_argument("--identity", default="000001", help="身份 ID")
    p_prev.add_argument("--finger", default="left_thumb", help="手指名")
    p_prev.add_argument("--class", dest="fp_class", default=None, help="指纹类别")
    p_prev.add_argument("--seed", type=int, default=None)
    p_prev.add_argument("--impression", type=int, default=1, help="捺印序号")
    p_prev.add_argument("--tta", type=int, default=0, help="TTA 序号（0 = 原始捺印）")
    p_prev.add_argument("--save", type=Path, default=None, help="保存预览图到该路径")

    # -- selfcheck -------------------------------------------------------
    sub.add_parser("selfcheck", help="自检：渲染一张样本并校验各阶段产物")

    # -- gui -------------------------------------------------------------
    p_gui = sub.add_parser("gui", help="启动图形界面（默认命令）")
    p_gui.add_argument("-c", "--config", type=Path, default=None, help="启动时加载的配置")
    p_gui.add_argument("--preset", default=None, choices=preset_names())

    # -- export ----------------------------------------------------------
    p_exp = sub.add_parser("export", help="对已有数据集重新执行导出（幂等）")
    p_exp.add_argument("--dataset", type=Path, required=True)

    return parser


# ---------------------------------------------------------------------------
# 配置装配
# ---------------------------------------------------------------------------
def _parse_overrides(pairs: Sequence[str]) -> Dict[str, Any]:
    """把 ``["a.b=1", "c=x"]`` 解析为 ``{"a.b": 1, "c": "x"}``（值做 YAML 解析）。"""
    import yaml

    out: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ConfigError(f"--set 需要 key=value 形式，收到: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"--set 的键不能为空: {item}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        out[key] = value
    return out


def _resolve_config(args: argparse.Namespace, extra_overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
    """按 ``--config`` / ``--preset`` / ``--set`` 装配配置。"""
    overrides: Dict[str, Any] = dict(_parse_overrides(getattr(args, "overrides", []) or []))
    if extra_overrides:
        overrides.update({k: v for k, v in extra_overrides.items() if v is not None})

    config_path = getattr(args, "config", None)
    preset = getattr(args, "preset", None)

    if config_path:
        cfg = load_config(config_path)
        if preset:
            cfg = build_config(preset, base=cfg)
    elif preset:
        cfg = build_config(preset)
    else:
        cfg = load_default_config()

    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------
def _cmd_generate(args: argparse.Namespace) -> int:
    from .app import GenerationCallbacks, GenerationService, format_stats_text
    from .app.stats import compute_dataset_stats

    extra = {
        "dataset.output_root": args.output_root,
        "dataset.identities": args.identities,
        "dataset.fingers_per_identity": args.fingers,
        "dataset.impressions_per_finger": args.impressions,
        "dataset.tta_per_impression": args.tta,
        "runtime.seed": args.seed,
        "runtime.workers": args.workers,
        "runtime.resume": False if args.no_resume else None,
    }
    cfg = _resolve_config(args, extra)

    if args.dry_run:
        svc = GenerationService(cfg, export_after_run=False)
        summary = svc.dry_run()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("\n配置摘要:")
        for line in cfg.summary_lines():
            print("  " + line)
        return 0

    quiet = bool(args.quiet)
    _last = {"pct": -5}

    def on_progress(event) -> None:
        if quiet:
            return
        pct = int(event.percent)
        if pct >= _last["pct"] + 5 or pct >= 100:
            _last["pct"] = pct
            s = event.stats
            sys.stdout.write(
                f"\r  [{pct:3d}%] {s.processed}/{s.total}  "
                f"成功 {s.ok}  丢弃 {s.rejected}  失败 {s.failed}  "
                f"{s.rate:.2f} 样本/秒  ETA {s.eta_s:5.0f}s   "
            )
            sys.stdout.flush()

    def on_log(level: str, message: str) -> None:
        if quiet and level == "INFO":
            return
        sys.stdout.write("\r" + " " * 100 + "\r")
        print(f"[{level}] {message}")

    callbacks = GenerationCallbacks(
        on_log=on_log,
        on_progress=on_progress,
        on_phase=lambda name: on_log("INFO", f"阶段：{name}"),
    )
    svc = GenerationService(cfg, callbacks, export_after_run=not args.no_export)

    print(f"synthfp v{__version__}  {cfg.dataset.summary()}")
    for line in cfg.summary_lines():
        print("  " + line)
    print()

    try:
        svc.start(blocking=True)
    except KeyboardInterrupt:
        svc.stop()
        svc.wait(10)
        print("\n已中断；已完成的样本保留在输出目录，可用同样命令续跑。")
        return 130

    print()
    stats = compute_dataset_stats(cfg, svc.layout)
    print(format_stats_text(stats))
    print()
    print(f"输出目录: {svc.layout.root}")
    return 0 if svc.stats.failed == 0 else 1


def _cmd_plan(args: argparse.Namespace) -> int:
    from .app import GenerationService

    cfg = _resolve_config(args, {"dataset.output_root": args.output_root})
    summary = GenerationService(cfg, export_after_run=False).dry_run()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"数据集: {summary['dataset']}   seed={summary['seed']}")
        print(
            f"主指纹 {summary['masters']} | 捺印 {summary['impressions']} | "
            f"样本 {summary['samples']}"
        )
        print("类别分布: " + json.dumps(summary["classes"], ensure_ascii=False))
        print("划分: " + json.dumps(summary["splits"], ensure_ascii=False))
        print(f"配置指纹: {summary['config_fingerprint']}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from .app import compute_dataset_stats, format_stats_text

    layout = DatasetLayout(args.dataset)
    if not layout.manifest.exists():
        print(f"错误：{layout.manifest} 不存在，这不是一个 synthfp 数据集目录", file=sys.stderr)
        return 2
    cfg = _load_config_from_dataset(layout)
    stats = compute_dataset_stats(cfg, layout, write=args.write)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print(format_stats_text(stats))
        if args.write:
            print(f"\n已写入 {layout.stats}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from .core.manifest import read_jsonl
    from .core.storage import sha256_file

    layout = DatasetLayout(args.dataset)
    if not layout.manifest.exists():
        print(f"错误：{layout.manifest} 不存在", file=sys.stderr)
        return 2

    rows = list(read_jsonl(layout.manifest))
    problems: List[str] = []
    ok = 0
    missing_required = {"sample_id", "identity_id", "image_path", "seed", "class_label", "split"}
    seen: Dict[str, int] = {}

    for i, row in enumerate(rows, 1):
        sid = str(row.get("sample_id", f"<row {i}>"))
        if sid in seen:
            problems.append(f"重复 sample_id: {sid}")
        seen[sid] = i
        absent = missing_required - set(row)
        if absent:
            problems.append(f"{sid}: 缺少字段 {sorted(absent)}")
        if row.get("status") not in (None, "ok", "rejected"):
            continue
        img = row.get("image_path")
        if img:
            path = layout.resolve(str(img))
            if not path.exists():
                problems.append(f"{sid}: 图像缺失 {img}")
                continue
            if args.check_hash and row.get("sha256"):
                actual = sha256_file(path)
                if actual != row["sha256"]:
                    problems.append(f"{sid}: SHA-256 不符（记录 {row['sha256'][:12]}… 实际 {actual[:12]}…）")
                    continue
        ok += 1

    result = {
        "dataset": layout.root.name,
        "rows": len(rows),
        "ok": ok,
        "problems": len(problems),
        "details": problems[:50],
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"数据集 {result['dataset']}: {len(rows)} 行，{ok} 条可校验通过")
        if problems:
            print(f"\n发现 {len(problems)} 个问题：")
            for p in problems[:50]:
                print("  • " + p)
            if len(problems) > 50:
                print(f"  … 其余 {len(problems) - 50} 条已省略")
        else:
            print("校验通过：清单完整、图像齐全" + ("、校验和一致" if args.check_hash else ""))
    return 1 if problems else 0


def _cmd_presets(_: argparse.Namespace) -> int:
    print(f"{'名称':<20}{'标题':<18}说明")
    print("-" * 100)
    for name in preset_names():
        preset = PRESETS[name]
        tags = f" [{'/'.join(preset.tags)}]" if preset.tags else ""
        print(f"{name:<20}{preset.label:<18}{preset.description}{tags}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    if args.resolved:
        print(dump_yaml(cfg), end="")
    else:
        for line in cfg.summary_lines():
            print(line)
        print()
        print("规模估算: " + json.dumps(cfg.estimate(), ensure_ascii=False))
        print("提示：加 --resolved 查看完整 YAML，或用 --set key=value 覆盖任意字段。")
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    from .app import PreviewService
    from .legacy.util.image_utils import imwrite_unicode

    cfg = _resolve_config(args)
    svc = PreviewService(cfg)
    res = svc.render(
        identity_id=args.identity,
        finger=args.finger,
        fp_class=args.fp_class,
        seed=args.seed,
        tta_index=args.tta,
        impression_index=args.impression,
    )
    if res.error:
        print(f"渲染失败: {res.error}", file=sys.stderr)
        return 1

    print(f"用时 {res.duration_s:.2f}s")
    print(f"类别: {cfg.classes.distribution and res.master.fp_class}")
    print(f"尺寸: {res.image.shape[1]}×{res.image.shape[0]}  掩模前景 {float((res.mask > 0).mean()):.1%}")
    print(f"捺印效果: {[e.get('stage') for e in res.effects]}")
    print(f"TTA 算子: {res.tta_operators or '（原始捺印）'}")
    if res.tta_params:
        print(f"TTA 参数: {json.dumps(res.tta_params, ensure_ascii=False)}")
    if res.quality is not None:
        print(
            f"质量: {res.quality.score:.1f}/100 (L{res.quality.level})  "
            f"清晰度 {res.quality.clarity:.3f}  对比度 {res.quality.contrast:.3f}  "
            f"方向一致性 {res.quality.orientation_certainty:.3f}"
        )
    print(f"细节点: {len(res.minutiae)} 个")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite_unicode(str(args.save), res.image, ext=args.save.suffix or ".png"):
            print(f"写入失败: {args.save}", file=sys.stderr)
            return 1
        print(f"已保存: {args.save}")
    return 0


def _cmd_selfcheck(_: argparse.Namespace) -> int:
    """自检：不需要任何外部数据，几十秒内验证整条链路。"""
    import numpy as np

    from .app import PreviewService
    from .pipeline import chain as chain_mod
    from .plugins.backends import list_backends
    from .plugins.tta import OPERATORS

    cfg = build_config("quick_test")
    checks: List[tuple] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # 1. 依赖
    try:
        import cv2
        import numpy
        import pydantic
        import yaml  # noqa: F401

        check(
            "依赖",
            True,
            f"numpy {numpy.__version__} / opencv {cv2.__version__} / pydantic {pydantic.VERSION}",
        )
    except Exception as exc:  # pragma: no cover
        check("依赖", False, str(exc))
        _print_checks(checks)
        return 1

    check("后端注册", len(list_backends()) >= 2, ", ".join(list_backends()))
    check("TTA 算子", len(OPERATORS) >= 15, f"{len(OPERATORS)} 个")
    check("预设", len(preset_names()) >= 5, f"{len(preset_names())} 个")

    # 2. 渲染 + 各阶段
    svc = PreviewService(cfg)
    res = svc.render("000001", "left_thumb", fp_class="left_loop", seed=1234, tta_index=0)
    check("后端渲染", res.error == "", res.error or f"{res.duration_s:.2f}s")
    if res.error:
        _print_checks(checks)
        return 1
    img = res.image
    check("图像形状", img.ndim == 2, str(img.shape))
    check("图像 dtype", img.dtype == np.uint8, str(img.dtype))
    check("掩模", res.mask is not None and set(np.unique(res.mask)) <= {0, 255},
          f"前景 {float((res.mask > 0).mean()):.1%}")
    check("方向场", res.orientation is not None and float(np.nanmax(res.orientation)) < np.pi + 1e-3,
          "值域 [0, π)")
    check("质量评估", res.quality is not None and 0 <= res.quality.score <= 100,
          f"{res.quality.score:.1f}/100 L{res.quality.level}" if res.quality else "未启用")
    check("细节点提取", len(res.minutiae) > 0, f"{len(res.minutiae)} 个")

    res_tta = svc.render("000001", "left_thumb", fp_class="left_loop", seed=1234, tta_index=1)
    check("TTA 链路", res_tta.error == "", ", ".join(res_tta.tta_operators) or "（空链）")

    # 3. 复现性：同种子两次渲染必须逐像素一致
    a = svc.render("000002", "left_index", fp_class="whorl", seed=777, tta_index=0,
                   with_annotation=False)
    b = svc.render("000002", "left_index", fp_class="whorl", seed=777, tta_index=0,
                   with_annotation=False)
    same = bool(np.array_equal(a.image, b.image))
    check("可复现性", same, "同种子两次渲染逐像素一致" if same else "两次结果不同！")

    # 4. 种子树稳定性
    from .core.seeds import SeedTree

    t1 = SeedTree(42, "ds")
    t2 = SeedTree(42, "ds")
    stable = t1.tta_seed("000001", "left_thumb", 1, 2) == t2.tta_seed("000001", "left_thumb", 1, 2)
    check("种子派生", stable, str(t1.tta_seed("000001", "left_thumb", 1, 2))[:16] + "…")

    # 5. 计划确定性
    # 用 3 个身份：quick_test 只有 2 个身份，按身份划分需要 3 个集合，
    # 否则 planner 会打印"身份数不足"的告警，干扰自检输出。
    from .config import apply_overrides
    from .core.planner import Planner

    plan_cfg = apply_overrides(cfg, {"dataset.identities": 3})
    p1 = Planner(plan_cfg).build().to_jsonl()
    p2 = Planner(plan_cfg).build().to_jsonl()
    check("计划确定性", p1 == p2, f"{len(p1.splitlines())} 行计划一致")

    # 6. TTA 链确定性
    seeds = [(1234, "sid", i) for i in range(4)]
    check(
        "TTA 链确定性",
        chain_mod.plan_tta_chains(cfg, seeds) == chain_mod.plan_tta_chains(cfg, seeds),
        str(chain_mod.plan_tta_chains(cfg, seeds[:2])),
    )

    # 7. GUI 可用性（PySide6 或 PyQt6，任一即可）
    try:
        from .gui.qt import QT_BINDING, QT_VERSION, is_available

        if is_available():
            check("GUI 依赖", True, f"{QT_BINDING} {QT_VERSION}".strip())
        else:
            check("GUI 依赖", False, "未安装 PySide6 / PyQt6（CLI 仍完全可用）")
    except Exception as exc:  # pragma: no cover
        check("GUI 依赖", False, f"Qt 兼容层加载失败: {exc}")

    return _print_checks(checks)


def _display_width(text: str) -> int:
    """字符串在等宽终端里的显示宽度（CJK 与全角符号占 2 列）。"""
    import unicodedata

    width = 0
    for ch in text:
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def _pad(text: str, width: int) -> str:
    """按**显示宽度**左对齐补空格（``str.ljust`` 只数字符，会错位）。"""
    return text + " " * max(width - _display_width(text), 0)


def _print_checks(checks: List[tuple]) -> int:
    failed = 0
    print(f"synthfp v{__version__} 自检")
    print("=" * 72)
    for name, ok, detail in checks:
        mark = "✓" if ok else "✗"
        if not ok:
            failed += 1
        print(f"  {mark} {_pad(name, 16)} {detail}")
    print("=" * 72)
    print(f"{len(checks) - failed}/{len(checks)} 项通过" + ("" if failed == 0 else f"，{failed} 项失败"))
    return 0 if failed == 0 else 1


def _cmd_export(args: argparse.Namespace) -> int:
    from .app import export_dataset

    layout = DatasetLayout(args.dataset)
    if not layout.manifest.exists():
        print(f"错误：{layout.manifest} 不存在", file=sys.stderr)
        return 2
    cfg = _load_config_from_dataset(layout)
    summary = export_dataset(cfg, layout)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    cfg = None
    if getattr(args, "config", None) or getattr(args, "preset", None):
        cfg = _resolve_config(args)
    try:
        from .gui.app import run_gui
    except ImportError as exc:  # pragma: no cover
        print(
            "无法启动 GUI：缺少 PySide6。\n"
            f"  详情: {exc}\n"
            "  安装: pip install PySide6\n"
            "  提示: 不装 GUI 也可以使用 CLI（synthfp generate / stats / validate）",
            file=sys.stderr,
        )
        return 1
    return run_gui(cfg)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _load_config_from_dataset(layout: DatasetLayout) -> AppConfig:
    """从数据集里的 ``config.resolved.yaml`` 还原配置；缺失时回退默认。"""
    if layout.config_resolved.exists():
        try:
            return load_config(layout.config_resolved)
        except ConfigError:
            pass
    cfg = AppConfig()
    cfg.dataset.name = layout.root.name
    cfg.dataset.output_root = layout.root.parent
    return cfg


_DISPATCH = {
    "generate": _cmd_generate,
    "plan": _cmd_plan,
    "stats": _cmd_stats,
    "validate": _cmd_validate,
    "presets": _cmd_presets,
    "show": _cmd_show,
    "preview": _cmd_preview,
    "selfcheck": _cmd_selfcheck,
    "export": _cmd_export,
    "gui": _cmd_gui,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（无子命令时默认启动 GUI）。"""
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "gui"
    handler = _DISPATCH.get(command)
    if handler is None:  # pragma: no cover
        parser.print_help()
        return 2
    if command == "gui" and args.command is None:
        args.config = None
        args.preset = None
    try:
        return int(handler(args))
    except ConfigError as exc:
        print(f"配置错误：\n{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\n已取消", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
