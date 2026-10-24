# 轻量级 SFinGe 指纹数据集合成软件设计文档（重构版）

**文档版本**：v2.0  
**日期**：2026-10-09  
**状态**：重构设计基线  
**重构原则**：保留可用资产，重写架构与 GUI，标准化数据接口  
**核心变更**：以 GUI 为一级入口，后端可插拔，TTA 与标准化作为独立的管道阶段

---

## 0. 重构背景与设计原则

### 0.1 为什么重构而不是新建

你已有一个可运行的项目，直接推倒重来会浪费已有资产。重构的目标是：

1. **保留**：已经验证过的 SFinGe 调用逻辑、生成算法、TTA 算子、元数据字段。
2. **重构**：把"脚本集合"升级为"分层架构 + GUI 应用"。
3. **补齐**：GUI、任务调度、标准化 schema、可复现机制、断点续传。
4. **解耦**：让后端、TTA、质量评估、存储四个维度互相独立。

### 0.2 设计原则

| 原则 | 说明 |
|------|------|
| GUI 优先 | 所有功能都能在 GUI 上完成，CLI 作为高级用户的补充 |
| 后端可插拔 | SFinGe 只是第一个后端，未来可替换为 Turing 自洽模型 |
| 管道化 | 生成 → TTA → 质量 → 细节点 → 存储，每段独立可测 |
| 单一数据源 | 所有元数据来自统一 Manifest，禁止各模块私写文件 |
| 可复现 | 配置 + 种子 + 版本 + 校验和，四要素缺一不可 |
| 最小依赖 | GUI 不引入重型框架，优先 PySide6 或 Tkinter |
| 渐进迁移 | 老代码通过 Adapter 包装接入，不强行一次改完 |

### 0.3 重构范围界定

| 类别 | 处理方式 |
|------|----------|
| 核心生成逻辑 | 保留，包装为 Backend Adapter |
| TTA 算子 | 保留，统一签名与参数 schema |
| 元数据写入 | 重写，统一走 Manifest 模块 |
| 调度逻辑 | 重写，引入任务队列与断点续传 |
| CLI | 保留，作为 GUI 的下层调用接口 |
| GUI | 新建 |
| 配置系统 | 重写，统一 YAML + Pydantic |
| 测试 | 补充，重点覆盖重构后的接口 |

---

## 1. 目标与需求

### 1.1 产品目标

一个带图形界面的桌面应用，让研究者无需写代码即可：

- 配置并生成批量合成指纹数据集；
- 每个主指纹生成多个捺印与多个 TTA 变换样本；
- 实时查看生成进度、预览图像、查看统计；
- 一键导出标准化科研数据集（含清单、划分、校验和）。

### 1.2 功能需求

| 编号 | 需求 | 优先级 |
|------|------|--------|
| F1 | GUI 配置身份数、手指数、捺印数、TTA 数 | P0 |
| F2 | GUI 控制指纹类别比例 | P0 |
| F3 | GUI 选择后端（SFinGe / 其他）与参数 | P0 |
| F4 | GUI 勾选 TTA 算子并设置范围 | P0 |
| F5 | 实时进度条、日志面板、图像预览 | P0 |
| F6 | 断点续传、失败重试 | P0 |
| F7 | 质量评估（NFIQ2）与阈值过滤 | P1 |
| F8 | 细节点提取与 ISO 模板导出 | P1 |
| F9 | 数据集划分（train/val/test，按身份） | P0 |
| F10 | 一键导出、打包、生成统计报告 | P0 |
| F11 | CLI 与 Python API | P1 |
| F12 | 配置保存/加载/预设 | P0 |

### 1.3 非功能需求

| 类别 | 要求 |
|------|------|
| 性能 | 8 核 CPU 下批量生成吞吐可配，单张 512×512 生成 < 2s（取决于后端）|
| 响应性 | GUI 主线程不阻塞，生成任务运行在子进程/线程池 |
| 可复现 | 同配置同种子生成结果校验和一致 |
| 可靠性 | 原子写入、断点续传、失败隔离 |
| 可扩展 | 后端、TTA、质量评估均为插件式 |
| 合规 | 合成数据声明，遵守 SFinGe 许可 |

---

## 2. 总体架构

```
┌────────────────────────────────────────────────────────────────┐
│                       GUI 应用层 (PySide6)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ 配置面板  │ │ 进度面板  │ │ 预览面板  │ │ 统计面板  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└───────────────────────────┬────────────────────────────────────┘
                            │ 调用
┌───────────────────────────▼────────────────────────────────────┐
│                    应用服务层 (Application)                     │
│  • GenerationService  • ValidationService  • ExportService     │
│  • ConfigService      • PreviewService                      │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│                       核心领域层 (Core)                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐  │
│  │  Planner    │ │  Scheduler  │ │  Pipeline   │ │ Manifest │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘  │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│                        插件层 (Plugins)                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐  │
│  │  Backend    │ │  TTA        │ │  Quality    │ │ Minutiae │  │
│  │  SFinGe     │ │  几何/光度   │ │  NFIQ2      │ │ 提取器    │  │
│  │  Turing     │ │  噪声/遮挡   │ │  自定义      │ │ ISO模板  │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘  │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│                       存储层 (Storage)                          │
│  • 图像存储  • Manifest  • 划分文件  • 校验和  • 日志           │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. GUI 设计（核心新增）

### 3.1 界面布局

采用经典三区布局：

```
┌─────────────────────────────────────────────────────────────┐
│  菜单栏: 文件 | 配置 | 生成 | 工具 | 帮助                       │
├─────────────────────────────────────────────────────────────┤
│  工具栏: [新建] [打开配置] [保存配置] [开始生成] [暂停] [停止]   │
├──────────────┬──────────────────────────┬───────────────────┤
│              │                          │                   │
│  左侧配置区   │      中央工作区           │   右侧预览/统计    │
│              │                          │                   │
│  • 数据集     │  • 任务列表/进度表        │  • 图像预览        │
│  • 后端       │  • 日志输出              │  • 缩略图网格      │
│  • TTA       │  • 实时统计图表           │  • 质量分布        │
│  • 质量       │                          │  • 细节点可视化    │
│  • 输出       │                          │                   │
│              │                          │                   │
├──────────────┴──────────────────────────┴───────────────────┤
│  状态栏: 已生成 1234/5000 | 失败 3 | 耗时 00:12:34 | 吞吐 1.2/s │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 页面结构

采用侧边栏导航 + 堆叠页面（QStackedWidget）：

| 页面 | 功能 |
|------|------|
| **1. 数据集配置** | 名称、输出目录、身份数、手指数、捺印数、TTA 数、类别比例 |
| **2. 后端配置** | 选择 SFinGe / Turing / 其他，设置可执行路径与参数 |
| **3. TTA 配置** | 勾选算子，滑块设置范围，实时预览效果 |
| **4. 质量配置** | NFIQ2 开关、阈值、细节点提取开关 |
| **5. 生成监控** | 进度条、任务表、日志、失败重试 |
| **6. 结果预览** | 缩略图网格、单图放大、细节点叠加显示 |
| **7. 统计报告** | 类别比例、质量分布、细节点数量分布、划分统计 |
| **8. 导出** | 打包、生成 README、导出 Manifest |

### 3.3 GUI 与核心的解耦

GUI **不直接调用**任何生成逻辑，只调用 `Application` 服务层：

```python
class GenerationService(QObject):
    progress = Signal(int, int)       # 已完成, 总数
    image_ready = Signal(str)          # 图像路径
    error = Signal(str, str)           # 任务ID, 错误信息
    finished = Signal(dict)            # 统计结果

    def start(self, config: DatasetConfig): ...
    def pause(self): ...
    def resume(self): ...
    def stop(self): ...
```

生成任务运行在 `QThread` 或 `multiprocessing` 中，通过信号与主线程通信，保证 GUI 不卡顿。

### 3.4 关键交互

| 交互 | 实现 |
|------|------|
| 参数修改后实时校验 | Pydantic 校验，错误红框提示 |
| TTA 算子预览 | 选一张样例图，实时应用算子并显示 |
| 生成中暂停/恢复 | 调度器支持暂停标志，任务边界处检查 |
| 断点续传 | 重启后读取 `done.jsonl`，跳过已完成任务 |
| 失败任务重试 | 任务表中失败项可单独重试 |
| 图像预览 | 点击缩略图，右侧大图显示，可切换细节点叠加 |

### 3.5 GUI 技术选型

| 选项 | 理由 |
|------|------|
| **PySide6** | 官方 Qt 绑定，LGPL，控件丰富，适合科研桌面应用 |
| 备选 Tkinter | 轻量但控件简陋，不推荐用于复杂布局 |
| 备选 PyQt6 | 功能相同但 GPL，商用需注意 |

推荐 **PySide6 + qdarktheme**（暗色主题）+ **pyqtgraph**（实时图表）。

---

## 4. 核心模块设计

### 4.1 配置系统

```python
class DatasetConfig(BaseModel):
    name: str
    output_dir: Path
    seed: int = 42
    identities: int
    fingers_per_identity: int
    impressions_per_finger: int
    tta_per_impression: int
    class_distribution: Dict[str, float]
    backend: BackendConfig
    tta: TTAConfig
    quality: QualityConfig
    split: SplitConfig
```

支持：
- YAML 加载/保存；
- 预设（preset）管理；
- GUI 与 CLI 共用同一 schema；
- 生成后写入 `config.resolved.yaml`。

### 4.2 数据集规划器（Planner）

职责：
1. 生成所有样本计划（身份 → 手指 → 捺印 → TTA）；
2. 按身份划分 train/val/test，防止身份泄漏；
3. 为每个样本分配稳定 ID 与路径；
4. 输出 `plan.parquet` 或 `plan.jsonl`。

### 4.3 任务调度器（Scheduler）

- 基于 `concurrent.futures.ProcessPoolExecutor`；
- 任务粒度：一个 TTA 样本；
- 支持优先级、重试（默认 3 次）、超时；
- 断点续传：`done.jsonl` 记录已完成样本；
- 失败隔离：`failed.jsonl` 记录失败样本与原因；
- 与 GUI 通过信号通信。

### 4.4 管道（Pipeline）

每个样本经过统一管道：

```
Backend.generate()
    → TTA.apply()
    → Quality.evaluate()
    → Minutiae.extract()
    → Storage.save()
    → Manifest.append()
```

每段可独立开关与替换，管道定义在配置中。

### 4.5 后端插件（Backend）

统一接口：

```python
class FingerprintBackend(Protocol):
    name: str
    version: str

    def generate_master(self, spec: MasterSpec) -> np.ndarray: ...
    def generate_impressions(self, master, n: int, seed: int) -> List[np.ndarray]: ...
```

内置实现：
- `SFinGeBackend`：CLI 封装，解析输出图像；
- `TuringBackend`：未来接入自洽模型；
- `DummyBackend`：测试用，生成占位图。

### 4.6 TTA 引擎

算子注册表：

```python
@register_tta("rotate")
class RotateTTA(TTAOperator):
    angle: Tuple[float, float]

    def apply(self, image, rng): ...
```

算子分类：

| 类别 | 算子 |
|------|------|
| 几何 | rotate, translate, scale, shear, elastic, perspective |
| 光度 | brightness, contrast, gamma, blur, motion_blur |
| 噪声 | gaussian, poisson, salt_pepper |
| 压缩 | jpeg |
| 遮挡 | rectangle, ellipse, scratch |
| 传感器风格 | optical, capacitive, thermal |

要求：
- 每个 TTA 样本记录算子名与参数；
- 随机种子由样本 ID 派生，保证可复现；
- 支持链式组合，限制最大失真度。

### 4.7 质量与细节点

- **质量**：NFIQ2（CLI 或库），输出分数；
- **细节点**：可插拔提取器（VeriFinger / OpenAFIS / 自研）；
- **输出**：
  - `nfiq2` 分数；
  - 细节点列表（x, y, angle, type, quality）；
  - ISO/IEC 19794-2 模板（可选）。

### 4.8 Manifest 与存储

目录结构：

```
output/
  synth_fp_v1/
    README.md
    config.resolved.yaml
    environment.json
    dataset_manifest.jsonl
    splits/
      train.csv
      val.csv
      test.csv
    images/
      000001/left_thumb/imp_01_tta_01.png
    minutiae/
      000001/left_thumb/imp_01.json
    metadata/
      identities.csv
      impressions.csv
      tta.csv
    checksums/
      sha256sums.txt
    logs/
      generation.log
      failed.jsonl
      done.jsonl
```

Manifest 每行一条样本记录，字段见附录。

---

## 5. 生成流程

1. GUI 加载/编辑配置 → 校验通过；
2. Planner 生成计划 → 写入 `plan.jsonl`；
3. Scheduler 分发任务 → 子进程执行管道；
4. 每个样本：后端生成 → TTA → 质量 → 细节点 → 存储 → Manifest；
5. 进度通过信号回传 GUI；
6. 完成后校验 → 生成统计报告 → 导出。

---

## 6. 重构实施方案

### 6.1 现状盘点（需你补充）

| 现有模块 | 状态 | 重构动作 |
|----------|------|----------|
| SFinGe 调用脚本 | 可用 | 包装为 `SFinGeBackend` |
| TTA 算子 | 可用 | 统一签名，注册到 TTA Registry |
| 元数据写入 | 部分 | 重写为 Manifest 模块 |
| 生成主流程 | 脚本式 | 重写为 Pipeline + Scheduler |
| 配置 | 硬编码/散落 | 重写为 Pydantic + YAML |
| GUI | 无 | 新建 |

### 6.2 迁移步骤

| 阶段 | 内容 | 产出 |
|------|------|------|
| **M0 冻结** | 冻结现有代码，打 tag，写现状文档 | 基线快照 |
| **M1 接口层** | 定义 Backend / TTA / Quality 接口，用 Adapter 包装老代码 | 老代码可被新框架调用 |
| **M2 核心层** | 实现 Planner / Scheduler / Pipeline / Manifest | 无 GUI 可跑通端到端 |
| **M3 CLI** | 提供 `synthfp generate/validate/stats` | 命令行可用 |
| **M4 GUI 骨架** | 主窗口、配置页、进度页 | 能配置并启动生成 |
| **M5 GUI 完整** | 预览、统计、导出、断点续传 | 全功能 GUI |
| **M6 测试与文档** | 单元/集成/性能测试，用户手册 | 可交付 |

### 6.3 兼容性保障

- 老配置通过 `legacy_config_adapter.py` 转换；
- 老输出目录通过 `migrate_dataset.py` 升级到新结构；
- 保留旧 CLI 入口 `old_cli.py`，内部转发到新服务。

---

## 7. 测试与验证

| 类型 | 内容 |
|------|------|
| 单元测试 | 配置校验、TTA 算子、Manifest 写入 |
| 集成测试 | 10 身份端到端生成 |
| GUI 测试 | pytest-qt 模拟点击与信号 |
| 性能测试 | 1000 身份吞吐、内存占用 |
| 统计验证 | 细节点分布、NFIQ2 分布、类别比例 |
| 复现测试 | 同种子两次生成，校验和一致 |
| 匹配验证 | 同身份高分、异身份低分（商用 matcher）|

---

## 8. 部署与运维

- Python 3.10+；
- 依赖：PySide6, numpy, opencv-python, pydantic, pyyaml, tqdm, scikit-image, pyqtgraph；
- 可选：NFIQ2, VeriFinger, OpenAFIS；
- 打包：PyInstaller 生成单文件可执行程序；
- 容器：Dockerfile 用于无 GUI 的批处理模式。

---

## 9. 路线图

| 里程碑 | 内容 | 周期 |
|--------|------|------|
| M1 | 接口层 + Adapter | 1 周 |
| M2 | 核心层（Planner/Scheduler/Pipeline/Manifest）| 2 周 |
| M3 | CLI | 1 周 |
| M4 | GUI 骨架 | 1.5 周 |
| M5 | GUI 完整 | 2 周 |
| M6 | 测试 + 文档 + 打包 | 1.5 周 |
| **合计** | | **约 9 周** |

---

## 10. 风险与对策

| 风险 | 对策 |
|------|------|
| SFinGe 许可不明确 | 确认许可；设计可替换后端 |
| GUI 阻塞 | 生成任务子进程化，信号通信 |
| 重构引入回归 | 保留旧 CLI，Adapter 包装，集成测试 |
| 生成速度慢 | 多进程、缓存主指纹、批处理 |
| TTA 过度失真 | 参数上限 + 人工抽检 |
| 元数据不一致 | 单一 Manifest + Schema 校验 |
| 复现失败 | 种子 + 版本 + 校验和 |

---

## 11. 附录

### 11.1 Manifest 字段

```json
{
  "sample_id": "000001_left_thumb_imp01_tta01",
  "identity_id": "000001",
  "finger": "left_thumb",
  "impression_id": "imp_01",
  "tta_id": "tta_01",
  "class_label": "whorl",
  "image_path": "images/000001/left_thumb/imp_01_tta_01.png",
  "width": 512, "height": 512, "ppi": 500,
  "nfiq2": 52.3,
  "minutiae_count": 78,
  "tta_operator": "rotate",
  "tta_params": {"angle": 7.5},
  "seed": 12345,
  "backend": "sfinge", "backend_version": "4.0",
  "sha256": "..."
}
```

### 11.2 GUI 信号示例

```python
class GenerationService(QObject):
    progress = Signal(int, int)
    image_ready = Signal(str)
    error = Signal(str, str)
    finished = Signal(dict)
```

### 11.3 CLI 示例

```bash
synthfp generate --config configs/default.yaml
synthfp validate --dataset output/synth_fp_v1
synthfp stats    --dataset output/synth_fp_v1
synthfp gui
```

---

**结论**：本设计以 GUI 为一级入口，以可插拔后端与管道化为核心，兼容现有项目资产。建议按 M1→M6 渐进重构，每阶段可独立验证，最终交付一个可复现、标准化、带图形界面的指纹数据集合成工具。