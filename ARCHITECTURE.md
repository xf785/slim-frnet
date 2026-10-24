# synthfp v2.0 架构说明

本文档面向后续维护者，说明**为什么这样分层**、**每层能做什么不能做什么**、
以及**新增功能应该改哪里**。设计依据见 [docs/DESIGN.md](docs/DESIGN.md)。

---

## 1. 分层与依赖方向

```
┌──────────────────────────────────────────────────────────────┐
│  gui/            PySide6 界面（一级入口，8 页面）              │
│  依赖：app, config, core.types, core.quantize（只读）          │
├──────────────────────────────────────────────────────────────┤
│  app/            应用服务层                                    │
│  GenerationService / PreviewService / exporter / stats        │
│  纯 Python，无 Qt 依赖 → CLI 与 GUI 共用                       │
├──────────────────────────────────────────────────────────────┤
│  pipeline/       样本管道编排（chain / runner / stages）        │
│  core/           计划、调度、清单、存储、种子、领域类型          │
│  无 Qt 依赖 → 服务器 / CI 可跑                                 │
├──────────────────────────────────────────────────────────────┤
│  plugins/        插件层（全部可替换）                           │
│  backends │ impressions │ tta │ quality │ minutiae            │
├──────────────────────────────────────────────────────────────┤
│  legacy/         原项目已验证的生成内核（最小改造，保留资产）    │
│  render（合成）/ degrade（19 退化算子）/ util（图像 IO）        │
└──────────────────────────────────────────────────────────────┘
```

**硬约束**：箭头只能向下。`core` / `pipeline` / `plugins` / `legacy` 里
**不允许出现 `import PySide6`**；GUI 里**不允许出现生成算法**。
这条约束由目录结构与 code review 共同保证，好处是核心逻辑可以脱离 GUI 单测与批处理。

---

## 2. 数据流（一次生成）

```
AppConfig (Pydantic)
   │
   ├─ Planner.build() ──────────► Plan{ PlanEntry[ MasterSpec + ImpressionSpec[] + SampleSpec[] ] }
   │                                  │  按 by_identity 划分，防身份泄漏
   │                                  │  所有 ID / 种子 / 相对路径在此确定
   │                                  ▼
   │                              plan.jsonl（可复现依据）
   │
   ├─ Scheduler.run(plan)
   │     │  任务粒度 = 1 个 PlanEntry（= 1 个手指的一遍完整合成）
   │     │  线程池 / 串行；断点续传；失败重试；失败隔离
   │     ▼
   │  PipelineRunner.run_entry()
   │     │
   │     ├─ Backend.generate_master(spec)        ← 每个手指只算一次
   │     │
   │     └─ for each ImpressionSpec:
   │           stages.apply_impression()         退化链 → 按压变形 → 传感器
   │           └─ for each SampleSpec:
   │                 chain.select_tta_chain()    ← 样本种子派生，可复现
   │                 stages.apply_tta()
   │                 stages.estimate_orientation()
   │                 stages.extract_minutiae()
   │                 stages.evaluate_quality()
   │                 PipelineRunner._persist()   ← 原子写 + 立即算 sha256
   │                 ▼
   │              SampleResult ──► Manifest.append(row)  +  logs/done.jsonl
   │
   └─ GenerationService._finish() ──► export_dataset() + compute_dataset_stats()
                                        README / stats.json / splits / checksums
```

---

## 3. 各层职责与禁忌

### 3.1 `config/` 配置系统

| 文件 | 职责 |
|------|------|
| `schema.py` | **唯一**的参数定义（Pydantic）。所有默认值、范围、区间校验都在这里 |
| `loader.py` | YAML ↔ `AppConfig`；点号路径覆盖；`${ENV}` 替换；友好错误信息 |
| `presets.py` | 预设 = 对默认配置的**覆盖片段**（不是独立 YAML 副本） |

**禁忌**：不要在别处硬编码参数默认值。`configs/default.yaml` 由
`tools/gen_default_config.py` 从 schema 生成，保证不会与代码漂移。

**区间字段**（`(下界, 上界)` 二元组）不能用 Pydantic 的 `gt`/`le` —— 那会直接抛
`TypeError`。统一用 `schema._check_interval()` 手工校验。

### 3.2 `core/` 核心域层

| 文件 | 关键设计 |
|------|---------|
| `seeds.py` | SHA-256 层级种子树，**不依赖 `hash()`**（后者有随机盐，跨进程不一致） |
| `types.py` | 全部领域数据结构。**字段名一旦确定就是契约**，manifest 直接由此序列化 |
| `planner.py` | 计划的**完全确定性**：同一配置在任何机器上产出逐字节相同的 `plan.jsonl` |
| `scheduler.py` | 任务粒度是「主指纹条目」；线程局部持有 `PipelineRunner`（runner 内有不可共享资源） |
| `manifest.py` | 唯一元数据来源；追加写 + flush；断点续传只看 `done.jsonl` |
| `storage.py` | 目录布局的**唯一**定义；所有写入都是「临时文件 + rename」原子操作 |
| `quantize.py` | 方向场→uint8 PNG（体积约 float32 `.npy` 的 1%），伴随 `_meta.json` 记录量纲 |

**踩过的坑**：
- 原子写的临时文件**不能**叫 `xxx.png.part` —— OpenCV 的 `imencode` 按扩展名找编码器，
  `.part` 会直接失败。统一用 `xxx.tmp` 并把真实格式显式传给 `imwrite_unicode(ext=...)`。
- Windows 下中文路径必须走 `legacy/util/image_utils.py` 的 `imread_unicode` / `imwrite_unicode`。

### 3.3 `pipeline/` 管道

| 文件 | 职责 |
|------|------|
| `chain.py` | TTA 算子链的可复现选择（`tta_index == 0` 保留原始捺印） |
| `stages.py` | 各阶段的**纯函数**：捺印、TTA、质量、细节点、载荷组装 |
| `runner.py` | 编排：主指纹复用、掩模对齐、重试、落盘 |

**约定**：所有随机源都由**样本种子**派生，绝不使用全局 `np.random.*`。
`stages.apply_tta(..., seed=sample_seed)` 把种子贯穿到算子内部，
因此 manifest 里的 `seed` 字段足以复算出任意一个样本。

### 3.4 `plugins/` 插件层

五个可替换维度，每个都有自己的注册表与契约（详见 [docs/CONTRACTS.md](docs/CONTRACTS.md)）：

| 维度 | 接口 | 内置实现 |
|------|------|---------|
| 后端 | `FingerprintBackend.generate_master(spec) -> MasterArtifact` | `SFinGeBackend`、`DummyBackend` |
| 捺印 | `deform.py` 纯函数 + `SensorProfile.apply()` | 4 种传感器 |
| TTA | `TTAOperator.sample_params/_apply` + `TTARegistry` | 21 个算子 / 6 类 |
| 质量 | `QualityEvaluator.evaluate() -> QualityReport` | `synthfp-nfiq2-lite` |
| 细节点 | `MinutiaeExtractor.extract() -> list[Minutia]` | `synthfp-cn`（CN 法） |

### 3.5 `app/` 应用服务层

- `GenerationService`：**回调式**进度上报（`GenerationCallbacks`），
  因此它不知道 Qt 的存在；GUI 只是众多消费者之一（CLI 也用同一套）。
- `PreviewService`：单张渲染，**完全不落盘**，供 GUI 调参与 TTA 效果对比。
- `exporter` / `stats`：幂等派生品（重复执行只重写 README/stats/checksums，不动图像）。

### 3.6 `gui/` 图形界面

| 文件 | 职责 |
|------|------|
| `qt.py` | **Qt 绑定兼容层**：优先 PySide6，回退 PyQt6；统一 `Signal`/`Slot`/`Property` |
| `app.py` | 入口：QApplication、主题、全局异常钩子 |
| `theme.py` | 暗/亮主题的 QSS + QPalette（不依赖第三方主题包） |
| `widgets.py` | 共享控件：`ImageView`（缩放/平移/叠加）、`StatCard`、`ThumbnailGrid`、`LogPanel` |
| `config_store.py` | 配置状态中心：所有页面的唯一数据源，`changed` 信号广播 |
| `qt_bridge.py` | **唯一的**「服务层回调 → Qt 信号」适配点 |
| `pages/base.py` | `BasePage`：声明式「配置 ↔ 控件」双向绑定（自动校验 + 失败回滚） |
| `pages/*.py` | 8 个页面 |
| `main_window.py` | 三区布局、菜单、工具栏、状态栏、快捷键 |

**为什么需要 Qt 兼容层**：设计文档 3.5 选定 PySide6，但部署环境常常已经装了
PyQt6。让 GUI 同时接受两者可以省掉一次 80 MB 安装，也让打包更灵活。
差异其实很小（枚举用法一致，只有信号/槽命名不同），因此把所有差异收敛到
`qt.py` 一个文件，其余 GUI 代码只写 `from ..qt import Qt, Signal, QWidget`。
有一条单元测试守着「除 `qt.py` 外任何 GUI 文件都不得直接 import Qt 绑定」。

**线程模型**：`GenerationService.start(blocking=True)` 跑在 `QThread` 里；
回调发生在工作线程；`Signal` 跨线程是队列投递，槽函数一定在主线程执行，
因此槽里可以直接操作控件。

**两条踩过的 Qt 坑**（都已在代码里处理并有测试覆盖）：

1. **动态改 `objectName` 后 QSS 不生效** —— 必须
   `style().unpolish(w); style().polish(w)`；`BasePage._restyle()` 已封装。
2. **已析构的控件仍在接收配置广播** —— Qt 只在 Python 包装对象被 GC 时
   自动断开连接，时机不确定；一旦控件先被析构、连接还在，刷新就撞
   `RuntimeError: wrapped C/C++ object ... has been deleted`（表现为
   "配置一改整个程序就崩"）。`BasePage` 因此用**模块级注册表**强持有
   `(store, slot)`，并挂 `destroyed` 钩子，控件析构时立即主动断开
   （注册表以 `id(page)` 为键，不会阻止页面回收）。

**另有四条只有"真的把界面画出来"才会暴露的问题**（已全部修掉并记在这里）：

| 现象 | 根因 | 修法 |
|------|------|------|
| 打开页面输入框全空、数值停在最小值、表格没有行 | `BasePage` 只连了 `changed` 信号，**从未做过首屏回填** | 构造函数末尾调用一次 `self.refresh()` |
| TTA 页的算子清单整块消失、预览页三栏被压成一条缝 | 页面外层已经是 `QScrollArea`，内层又套一层滚动区，两者互争尺寸 | 页面类设 `scrollable = False`，直接用 `self` 当内容容器 |
| 未勾选的复选框只剩一行文字，看不出"没勾" | QSS 只给了 `::indicator` 宽高、没给图片，Qt 就干脆不画底框 | 显式给 indicator 画边框/底色，选中态填强调色 |
| 关窗口弹"未保存修改"；自动化里直接卡死 | 首屏回填时控件回写配置，把 store 标成脏 | `_updating` 保护覆盖到 `on_refresh()`；`update_flat` 值没变就不标脏；`MainWindow` 构造完 `mark_clean()` |

**命名冲突的坑**：`BasePage` 把页面布局存在 `self._layout`。子页面
**不能**再定义名为 `_layout` 的方法 —— 属性会覆盖方法，调用时得到
`TypeError: 'QVBoxLayout' object is not callable`（`ExportPage` 踩过，
已改名 `_dataset_layout()`）。

**验证方式**：`tools/shoot_gui.py` 会用 `QT_QPA_PLATFORM=offscreen`
把 8 个页面逐个渲染成 PNG。没有显示器也能肉眼检查布局是否塌陷 ——
上面四条问题全是靠它发现的。脚本会自动把 `QT_QPA_FONTDIR` 指向系统
字体目录，否则离屏 Qt 找不到字体，所有文字会画成豆腐块。

### 脊线密度：一个只有物理量能兜住的 bug

`RidgeGenerator` 的 Gabor 迭代跑在 `coarse_scale`（默认 0.7）缩小后的网格上，
之后再把图案放大回原尺寸。**频率是"每像素"量**：同一根手指在缩小网格上
只占 `cw` 个像素，脊线周期也变成 `S×scale`，所以频率必须变成
`f / scale = f × (w / cw)`。

代码里曾写成 `f × (cw / w)`（乘反了），脊线周期被放大 `1/scale² ≈ 2` 倍，
实测周期从应有的 ~10 px 变成 ~22 px —— 肉眼看就是"指纹比真的稀疏"，
但**没有任何功能测试会失败**。

现在 `tests/test_backend_ridge.py` 用自相关直接测周期，并对照真实
500 dpi 指纹（8~12 px）断言，因此这类"看起来不对"的问题也能被自动拦住。
同类的量还有：脊线占空比（应约 50%）、谷线灰度（应约 200 而不是 180）。

调密度改 `backend.min_ridge_width` / `max_ridge_width`（单位是**一个周期的像素数**）
与 `ridge_gray` / `valley_gray` 即可，不必动代码。

`backend.ridge_only_black` 是外观上的第二条分支：跳过纹理、底噪、分位拉伸
与边缘羽化，直接输出 `{0, 255}` 的二值图。注意二值化前要先轻微高斯平滑
再以 0.5 为界 —— 图案是"粗网格 + 双三次放大"来的，边界带斜坡，
直接 `> 0` 会让脊线明显变胖、谷线只剩细缝。

---

## 4. 扩展指南

### 新增一个 TTA 算子

1. 在 `plugins/tta/operators.py` 加一个类，用 `@TTARegistry.register(name, label, category, distortion, param_spec)` 装饰；
2. 在 `config/schema.py` 的 `TTA_CATALOG` 与 `TTA_PARAM_SPECS` 里登记（GUI 会自动出现勾选框与参数滑块）；
3. 在 `tests/test_tta.py` 补一个用例（确定性 + 值域 + 不修改输入）。

不需要改 GUI、管道、配置系统的任何代码。

### 新增一个后端

```python
from synthfp.plugins.backends.base import BackendRegistry, FingerprintBackend
from synthfp.core.types import MasterArtifact, MasterSpec

@BackendRegistry.register
class MyBackend(FingerprintBackend):
    name = "mybackend"
    version = "1.0"
    description = "自洽模型后端"

    def generate_master(self, spec: MasterSpec) -> MasterArtifact:
        ...
```

GUI 的「后端配置」页会自动列出它；配置里把 `backend.name` 改成 `mybackend` 即可。

### 新增一个质量评估器

实现 `evaluate(image, mask, orientation) -> QualityReport`（`core/types.py` 的类型），
在 `pipeline/stages.py` 的 `evaluate_quality()` 里按配置分支选择。
`QualityReport` 的字段是 manifest 契约，扩展请**新增字段**而不是改名。

### 新增一个 GUI 页面

1. 继承 `synthfp/gui/pages/base.py` 的 `BasePage`，实现 `build()`；
2. 用 `bind_spin/bind_double/bind_check/bind_combo/bind_line/bind_interval` 绑定配置项；
3. 在 `pages/__init__.py` 的 `PAGE_CLASSES` 里加进元组（导航与 Ctrl+N 快捷键自动生效）。

---

## 5. 可复现性的实现细节

```python
# synthfp/core/seeds.py
SeedTree(root_seed, dataset_name)
    .dataset_seed()                          # sha256(root, "dataset", name)
    .identity_seed(identity_id)              # 逐层派生
    .finger_seed(identity_id, finger)
    .impression_seed(identity_id, finger, i)
    .tta_seed(identity_id, finger, i, t)     # → 写入 manifest 的 seed 字段
```

下游所有随机性都从这个 `tta_seed` 出发：

| 随机来源 | 种子 |
|---------|------|
| TTA 算子链选择 | `derive(sample_seed, "tta-chain", sample_id, tta_index)` |
| TTA 算子内部参数 | `derive(sample_seed, "tta-apply", operator_names)` |
| 指纹类别分配 | `derive(runtime.seed, "classes", dataset_name)` |
| 划分 | `derive(runtime.seed, "split", dataset_name)` |
| 捺印效果 | `impression_seed`（同一捺印的所有 TTA 共享同样的按压/传感器状态） |

**浮点说明**：numpy/OpenCV 的浮点运算在同一平台上确定；
跨不同 CPU 架构（如 x86 vs ARM）理论上可能出现末位差异，
因此 `environment.json` 记录了平台与版本，校验和用于验证**同平台复现**。

### 5.1 掩模与背景：三处必须"同源"的地方

生成链里凡是"作用在像素上"的操作，都必须对掩模做**同样的**操作，
否则标注与像素会错位：

| 阶段 | 做法 |
|------|------|
| 按压变形 | 位移场经 `_shift_valid_region` 求出"变换后仍在画面内"的源像素集合，用它把 `BORDER_REPLICATE` 拖出来的边缘亮带换回原图 |
| 传感器 | 噪声是**全画幅**的，但只保留手指区域内的结果，区域外还原为纯背景 —— 否则手指周围会出现一圈灰噪，细节点检测会在噪声里报出成百上千个伪点 |
| TTA | **同一批已实例化的算子对象**、**同一个随机源**（`np.random.default_rng(spec.seed)`）依次作用于掩模，保证与图像严格同源 |

另外，掩模本身**不**用"灰度阈值"临时估出来：传感器噪声与皮肤纹理会
在背景留下零星暗点，阈值法会把它们当成手指。正确做法是变换主指纹掩模
（几何上精确），再做形态学清理（闭→开→取最大连通域→闭）。
这条链路由 `PipelineRunner._transform_mask()` + `_clean_mask()` 实现。

### 5.2 脊线灰度映射：顺序决定对比度

`RidgeGenerator._to_grayscale()` 的三步顺序很关键：

1. 图案模糊后线性映射到 `[ridge_mid, valley_mid]`；
2. 叠加多尺度皮肤纹理与高斯底噪；
3. **最后**才在手指区域内做 1%–99% 分位拉伸。

如果第 3 步放到第 2 步之前，噪声会把已经拉开的动态范围再次压扁 ——
实测脊线对比度会从约 160 个灰度级掉到约 35 个，指纹看起来一片灰
（这正是重构过程中被实测发现并修掉的问题）。

---

## 6. 与原项目的关系

| 原项目资产 | v2.0 处理 |
|-----------|----------|
| `synthesis/*`（轮廓/奇异点/方向场/频率图/脊线） | **保留算法**，迁到 `legacy/render/`，仅做工程优化（核缓存、向量化、移除内嵌细化） |
| `degradation/*`（19 个退化算子） | **保留**，迁到 `legacy/degrade/`，作为「捺印阶段」的随机扰动链复用 |
| `utils/image_utils.py` | **保留**，迁到 `legacy/util/`，并新增 `ext` 参数以支持原子写 |
| `config/schema.py` | **重写**（原 schema 与设计文档字段不符，且缺少区间校验） |
| `gui/main_window.py`（79 KB 单文件、崩溃日志里多起 `NameError`） | **完全重写**为 8 页面 + 共享控件 + 配置状态中心 |
| 元数据写入（分散） | **重写**为统一 Manifest（单一数据源） |
| 生成主流程（脚本式，嵌在 GUI 里） | **重写**为 Planner + Scheduler + Pipeline |
| README 中描述但**并不存在**的模块（ingestion / annotation / patching / export / pipeline / cli / 部分 degradation 算子） | 按设计文档**重新实现**为 `plugins/minutiae`、`plugins/quality`、`app/exporter`、`pipeline`、`cli` |

原项目作为**基线快照保留在本地**（本机路径 `_archive/legacy-baseline/`，`git 历史完整、未改动`，
不随仓库发布），便于需要时与 v2.0 做逐像素对照
（`backend.shape_style="box"` + 相同种子）。
