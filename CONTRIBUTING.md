# 参与贡献

感谢你愿意改进 SLIM-FRNet。这个项目刻意保持「小而完整」：**4 个核心依赖 + 一个 Qt 绑定**，
任何新增依赖都需要充足理由。

## 开发环境

```bash
git clone https://github.com/<you>/slim-frnet.git
cd slim-frnet
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install pytest ruff

python -m synthfp selfcheck          # 17 项自检
python -m pytest tests -q            # 全量测试
ruff check synthfp tests tools       # 静态检查
```

无显示器的环境（CI / 服务器）跑测试时要设 `QT_QPA_PLATFORM=offscreen`。

## 提交前请确认

1. `ruff check synthfp tests tools` 无告警；
2. `python -m pytest tests -q` 全绿（当前基线 900+ 用例）；
3. 改了配置字段后重新生成默认配置，否则会与 schema 漂移：

   ```bash
   python tools/gen_default_config.py
   ```

4. 改了界面后重拍截图，确认布局没有塌陷：

   ```bash
   python tools/shoot_gui.py --full-page --dataset outputs/your_dataset
   ```

## 代码约定

| 约定 | 说明 |
|------|------|
| **分层方向** | `gui → app → {pipeline, core} → plugins → legacy`，禁止反向 import |
| **Qt 隔离** | `core` / `pipeline` / `app` / `plugins` / `legacy` / `config` 一律不得 import Qt；所有 Qt 访问必须经过 `synthfp/gui/qt.py`（双绑定兼容层） |
| **类型注解** | 每个模块 `from __future__ import annotations`，公开函数写全注解 |
| **文档字符串** | 中文，说明「为什么这么做」而不只是「做了什么」 |
| **确定性** | 任何随机都必须来自种子树（`synthfp/core/seeds.py`），**禁止**使用 `random` 模块或 `hash()` |
| **测试命名** | 测试函数用中文描述行为，例如 `test_脊线周期接近真实500dpi指纹` |

## 加一个 TTA 算子

1. 在 `synthfp/plugins/tta/operators.py` 里加类，用 `@TTARegistry.register(...)` 装饰；
2. 在 `synthfp/config/schema.py` 的 `TTA_CATALOG` / `TTA_PARAM_SPECS` 登记
   （GUI 会自动出现勾选框与参数控件）；
3. 在 `tests/test_tta.py` 补用例：确定性、值域、不改输入、参数可复现。

不需要改 GUI、管道、配置系统的任何代码。接口契约见 [CONTRACTS.md](CONTRACTS.md)。

## 加一个生成后端

实现 `synthfp/plugins/backends/base.py` 的 `FingerprintBackend`
（只需 `generate_master(spec) -> MasterArtifact`），用 `@BackendRegistry.register` 注册即可。

## 报告问题

请附上：操作系统与 Python 版本、`python -m synthfp selfcheck` 输出、复现用的完整命令
（含 `--set` 覆盖项），以及如果与图像有关的话，一张最小复现的样例图。

## 许可

提交贡献即表示你同意以 [MIT License](LICENSE) 授权你的贡献。
