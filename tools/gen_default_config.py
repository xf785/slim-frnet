"""开发工具：从 Pydantic schema 生成 ``configs/default.yaml``。

用法::

    python tools/gen_default_config.py          # 重新生成
    python tools/gen_default_config.py --check  # 只校验是否漂移（CI 用，不写文件）

这样做的好处是配置文件**不可能与 schema 漂移**：
默认值只有一个来源（``synthfp/config/schema.py``），
``configs/default.yaml`` 只是它的一份带注释快照。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from synthfp.config import AppConfig, dump_yaml  # noqa: E402

HEADER = """\
# ============================================================
#  synthfp v2.0 — 默认配置（由 tools/gen_default_config.py 从 schema 生成）
# ============================================================
#  本文件是「默认值快照」，字段与 synthfp/config/schema.py 一一对应。
#  修改默认值请改 schema，然后重新生成；也可以直接改本文件用于单次运行。
#
#  快速上手:
#    synthfp show                      # 看当前生效的配置摘要
#    synthfp show --resolved           # 看完整展开后的 YAML
#    synthfp generate -c configs/default.yaml
#    synthfp gui                       # 图形界面（所有参数都能在这里调）
#
#  覆盖任意字段（点号路径，可重复）:
#    synthfp generate -c configs/default.yaml \\
#        --set dataset.identities=100 --set runtime.workers=8 --set tta.mode=chain
#
#  复现四要素: 配置 + 种子(runtime.seed) + 版本(environment.json) + 校验和
# ============================================================

"""


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 schema 生成默认配置")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查 configs/default.yaml 是否与 schema 一致，不写文件（CI 用）",
    )
    args = parser.parse_args(argv)

    target = _ROOT / "configs" / "default.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    body = dump_yaml(AppConfig())
    expected = HEADER + body

    if args.check:
        if not target.is_file():
            print(f"✗ {target} 不存在", file=sys.stderr)
            return 1
        if target.read_text(encoding="utf-8") != expected:
            print(
                "✗ configs/default.yaml 与 schema 已漂移，请运行：\n"
                "    python tools/gen_default_config.py",
                file=sys.stderr,
            )
            return 1
        print("✓ configs/default.yaml 与 schema 一致")
        return 0

    target.write_text(expected, encoding="utf-8")
    print(f"已写入 {target}  ({len(body.splitlines())} 行)")

    # 校验生成的配置能被重新读回且与原配置等价
    from synthfp.config import load_config

    reloaded = load_config(target)
    if reloaded.model_dump() != AppConfig().model_dump():
        print("✗ 生成的配置与 schema 默认值不一致！", file=sys.stderr)
        return 1
    print("✓ 往返校验通过（load_config 结果与 schema 默认值一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
