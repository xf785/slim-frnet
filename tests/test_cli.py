"""CLI 集成测试（``synthfp.cli.main``），≥ 12 个用例。

直接调用 ``synthfp.cli.main([...])``（不启动子进程、不依赖当前工作目录），
所有输出根都落在 ``tmp_path``；生成类用例通过 ``--set backend.name=dummy``
切换到占位后端，保证秒级完成。

覆盖范围：
    * ``presets`` / ``show`` / ``show --resolved`` / ``plan --json`` / ``selfcheck``；
    * ``generate`` → ``stats`` → ``validate --check-hash`` 的端到端链路；
    * ``export`` 子命令对已有数据集幂等；
    * 错误路径：图像缺失（退出 1）、非数据集目录（退出 2）、非法 ``--set``（退出 2）；
    * 解析器行为：无子命令时 ``command is None``（默认 GUI 由解析器层保证）。
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest
import yaml

from synthfp.cli import build_parser, main
from synthfp.config import PRESETS, build_config, loads_config, preset_names

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _invoke(argv: list[str]) -> tuple[int, str, str]:
    """调用 CLI 并捕获 stdout / stderr（供模块级 fixture 使用）。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def generated_dataset(tmp_path_factory) -> Path:
    """用 CLI 生成一个最小数据集（1 身份 × 1 指 × 1 捺印 × 2 TTA，dummy 后端）。"""
    root = tmp_path_factory.mktemp("cli_out")
    code, out, err = _invoke([
        "generate",
        "--preset", "quick_test",
        "--set", "backend.name=dummy",
        "--set", "dataset.name=cli_ds",
        "--set", f"dataset.output_root={root}",
        "--set", "dataset.identities=1",
        "--set", "dataset.tta_per_impression=2",
        "--set", "runtime.workers=1",
        "-q",
    ])
    assert code == 0, err or out
    dataset = root / "cli_ds"
    assert dataset.is_dir()
    return dataset


# ---------------------------------------------------------------------------
# 解析器行为
# ---------------------------------------------------------------------------
def test_无子命令时command为None():
    """缺省命令是 gui，但解析器本身只产出 ``command is None``（不启动 GUI）。"""
    args = build_parser().parse_args([])
    assert args.command is None
    assert build_parser().parse_args(["presets"]).command == "presets"


def test_未知子命令由argparse拒绝():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["no_such_command"])
    assert exc.value.code == 2


def test_preset参数只接受内置预设名():
    parser = build_parser()
    for name in preset_names():
        assert parser.parse_args(["show", "--preset", name]).preset == name
    with pytest.raises(SystemExit):
        parser.parse_args(["show", "--preset", "not_a_preset"])


# ---------------------------------------------------------------------------
# presets / show / plan
# ---------------------------------------------------------------------------
def test_presets退出0并列出全部预设(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    for name in PRESETS:
        assert name in out, name
    for preset in PRESETS.values():
        assert preset.label in out
    assert "说明" in out


def test_show打印数据集名(capsys):
    assert main(["show", "--preset", "quick_test"]) == 0
    out = capsys.readouterr().out
    assert "数据集" in out
    assert build_config("quick_test").dataset.name in out
    assert "规模估算" in out
    assert '"samples": 4' in out.replace("规模估算: ", "")


def test_show_resolved输出可解析YAML并往返(capsys):
    assert main(["show", "--preset", "quick_test", "--resolved"]) == 0
    text = capsys.readouterr().out
    raw = yaml.safe_load(text)
    assert isinstance(raw, dict)
    assert raw["dataset"]["name"] == "quick_test"
    cfg = loads_config(text)
    assert cfg.dataset.identities == 2
    assert cfg == build_config("quick_test")


def test_show_apply_overrides(capsys):
    assert main([
        "show", "--preset", "quick_test",
        "--set", "dataset.identities=3", "--set", "runtime.seed=7",
    ]) == 0
    out = capsys.readouterr().out
    assert "3 身份" in out
    assert "seed=7" in out


def test_plan_json样本数等于配置总数(capsys):
    cfg = build_config("quick_test")
    assert main(["plan", "--preset", "quick_test", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["samples"] == cfg.dataset.total_samples == 4
    assert payload["masters"] == cfg.dataset.total_masters
    assert payload["impressions"] == cfg.dataset.total_impressions
    assert payload["dataset"] == cfg.dataset.name
    assert payload["config_fingerprint"]


def test_plan文本输出(capsys):
    assert main(["plan", "--preset", "no_tta", "--set", "dataset.identities=3"]) == 0
    out = capsys.readouterr().out
    # 默认每身份 2 指 ⇒ 3 身份 = 6 个主指纹；no_tta 预设下 TTA 关闭
    assert "数据集: synth_fp_plain" in out
    assert "主指纹 6 | 捺印 12 | 样本 12" in out
    assert "类别分布:" in out and "划分:" in out


# ---------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------
def test_selfcheck退出码与检查项一致(capsys):
    """自检必须通过除「GUI 依赖」之外的全部检查项。

    本机可能未安装 PySide6（CLI 仍然可用），此时自检返回 1；
    因此这里断言：失败项只允许是 GUI 依赖，且退出码与之对应。
    """
    code = main(["selfcheck"])
    out = capsys.readouterr().out
    assert "synthfp" in out and "自检" in out
    assert "项通过" in out

    failed = [line for line in out.splitlines() if "✗" in line]
    assert all("GUI 依赖" in line for line in failed), failed
    assert code == (1 if failed else 0)
    # 关键检查项必须出现在输出中
    for name in ("依赖", "后端注册", "TTA 算子", "预设", "后端渲染", "掩模", "方向场",
                 "质量评估", "细节点提取", "TTA 链路", "可复现性", "种子派生",
                 "计划确定性", "TTA 链确定性"):
        assert name in out, name


# ---------------------------------------------------------------------------
# generate → stats → validate
# ---------------------------------------------------------------------------
def test_generate退出0并产出数据集(generated_dataset: Path):
    for name in ("dataset_manifest.jsonl", "config.resolved.yaml", "environment.json",
                 "plan.jsonl", "README.md", "stats.json"):
        assert (generated_dataset / name).is_file(), name
    assert (generated_dataset / "checksums" / "sha256sums.txt").is_file()
    rows = [ln for ln in (generated_dataset / "dataset_manifest.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() if ln]
    assert len(rows) == 2
    assert json.loads(rows[0])["status"] == "ok"


def test_stats在生成的数据集上退出0(generated_dataset: Path, capsys):
    assert main(["stats", "--dataset", str(generated_dataset)]) == 0
    out = capsys.readouterr().out
    assert "数据集: cli_ds" in out
    assert "样本: 2 成功" in out
    assert "类别分布:" in out


def test_stats_json输出可解析(generated_dataset: Path, capsys):
    assert main(["stats", "--dataset", str(generated_dataset), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "cli_ds"
    assert payload["totals"]["ok"] == 2
    assert payload["totals"]["identities"] == 1
    assert payload["config_fingerprint"]


def test_stats_write回写stats_json(generated_dataset: Path, capsys):
    stats_path = generated_dataset / "stats.json"
    before = stats_path.read_text(encoding="utf-8")
    assert main(["stats", "--dataset", str(generated_dataset), "--write"]) == 0
    out = capsys.readouterr().out
    assert str(stats_path) in out
    after = json.loads(stats_path.read_text(encoding="utf-8"))
    assert after["totals"]["ok"] == 2
    assert isinstance(before, str) and before


def test_validate_check_hash退出0(generated_dataset: Path, capsys):
    assert main(["validate", "--dataset", str(generated_dataset), "--check-hash"]) == 0
    out = capsys.readouterr().out
    assert "2 行" in out and "2 条可校验通过" in out
    assert "校验通过" in out and "校验和一致" in out


def test_validate_json无问题(generated_dataset: Path, capsys):
    assert main(["validate", "--dataset", str(generated_dataset), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 2 and payload["ok"] == 2 and payload["problems"] == 0
    assert payload["dataset"] == "cli_ds"


def test_export子命令复用已有数据集(generated_dataset: Path, capsys):
    assert main(["export", "--dataset", str(generated_dataset)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["samples"] == 2
    assert payload["checksums"] == payload["artifacts"] > 0


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------
def test_validate图像缺失退出1(tmp_path: Path, capsys):
    dataset = tmp_path / "broken_ds"
    dataset.mkdir()
    (dataset / "dataset_manifest.jsonl").write_text(
        json.dumps({
            "sample_id": "000001_left_thumb_imp01_tta00",
            "identity_id": "000001",
            "image_path": "images/missing.png",
            "seed": 1,
            "class_label": "arch",
            "split": "train",
            "status": "ok",
        }) + "\n",
        encoding="utf-8",
    )
    assert main(["validate", "--dataset", str(dataset), "--check-hash"]) == 1
    out = capsys.readouterr().out
    assert "图像缺失" in out
    assert "发现 1 个问题" in out


def test_validate重复sample_id退出1(tmp_path: Path, capsys):
    dataset = tmp_path / "dup_ds"
    (dataset / "images").mkdir(parents=True)
    (dataset / "images" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    row = {
        "sample_id": "dup",
        "identity_id": "000001",
        "image_path": "images/a.png",
        "seed": 1,
        "class_label": "arch",
        "split": "train",
        "status": "ok",
    }
    (dataset / "dataset_manifest.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )
    assert main(["validate", "--dataset", str(dataset)]) == 1
    assert "重复 sample_id" in capsys.readouterr().out


def test_stats非数据集目录退出2(tmp_path: Path, capsys):
    target = tmp_path / "not_a_dataset"
    target.mkdir()
    assert main(["stats", "--dataset", str(target)]) == 2
    captured = capsys.readouterr()
    assert "不是一个 synthfp 数据集目录" in captured.err
    assert captured.out == ""


def test_validate非数据集目录退出2(tmp_path: Path, capsys):
    assert main(["validate", "--dataset", str(tmp_path / "nothing")]) == 2
    captured = capsys.readouterr()
    assert "不存在" in captured.err
    assert captured.out == ""


def test_非法set值退出2并打印配置错误(tmp_path: Path, capsys):
    assert main([
        "generate", "--preset", "quick_test",
        "--set", "dataset.identities=0",
        "--set", f"dataset.output_root={tmp_path}",
        "-q",
    ]) == 2
    captured = capsys.readouterr()
    assert "配置错误" in captured.err
    assert "dataset.identities" in captured.err
    assert not (tmp_path / "quick_test").exists()


def test_非法set语法退出2(tmp_path: Path, capsys):
    assert main([
        "generate", "--preset", "quick_test",
        "--set", "missing_equals_sign",
        "--set", f"dataset.output_root={tmp_path}",
        "-q",
    ]) == 2
    assert "key=value" in capsys.readouterr().err


def test_generate_dry_run不生成文件(tmp_path: Path, capsys):
    assert main([
        "generate", "--preset", "quick_test", "--dry-run",
        "--set", "backend.name=dummy",
        "--set", f"dataset.output_root={tmp_path}",
    ]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out.split("\n配置摘要:")[0])
    assert payload["samples"] == 4
    assert "配置摘要:" in out
    assert not (tmp_path / "quick_test").exists()


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------
def test_preview渲染并保存(tmp_path: Path, capsys):
    target = tmp_path / "preview.png"
    assert main([
        "preview", "--preset", "quick_test", "--set", "backend.name=dummy",
        "--finger", "left_thumb", "--class", "left_loop", "--seed", "1234",
        "--tta", "1", "--save", str(target),
    ]) == 0
    out = capsys.readouterr().out
    assert "尺寸:" in out
    assert "TTA 算子:" in out
    assert "细节点:" in out
    assert target.is_file() and target.stat().st_size > 0
    assert target.read_bytes().startswith(b"\x89PNG")


# ---------------------------------------------------------------------------
# 控制台输出格式（GBK 控制台兼容、CJK 宽度对齐）
# ---------------------------------------------------------------------------
def test_display_width把CJK算作两列():
    from synthfp.cli import _display_width

    assert _display_width("abc") == 3
    assert _display_width("依赖") == 4
    assert _display_width("GUI 依赖") == 3 + 1 + 4
    assert _display_width("") == 0


def test_pad按显示宽度对齐():
    from synthfp.cli import _display_width, _pad

    padded = _pad("依赖", 10)
    assert _display_width(padded) == 10
    # 已经够宽时不再补空格
    assert _pad("a" * 20, 10) == "a" * 20


def test_配置stdio后打印Unicode不抛异常(capsys):
    """``configure_stdio`` 必须可在任意环境下安全调用（GBK 控制台不再崩）。"""
    from synthfp.cli import configure_stdio

    configure_stdio()
    configure_stdio()  # 幂等
    print("✓ 依赖 ✗ 失败 — 中文")
    out = capsys.readouterr().out
    assert "依赖" in out
