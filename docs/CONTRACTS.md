# synthfp 模块接口契约（v2.0）

本文件是各模块并行开发时的**唯一接口约定**。所有实现必须严格遵守，
因为 GUI / 核心层 / 服务层会直接按此契约调用。

## 0. 通用约定

- 语言：Python 3.10+（实际运行 3.13），纯类型注解。
- 依赖：**只允许** `numpy`、`opencv-python`(`cv2`)、`pydantic`、`PyYAML`、`PySide6`(仅 GUI)。
  禁止引入 scipy / scikit-image / matplotlib / pandas / click / tqdm。
- 图像约定：
  - 后端输出图像为 `uint8`，形状 `(H, W)`（灰度），背景 = 255。
  - 滤镜类算子输入输出统一为 `float32`，范围 `[0, 1]`（`pipeline` 负责转换）。
  - 掩模为 `uint8`，形状 `(H, W)`，取值 `{0, 255}`。
  - 方向场为 `float32`，形状 `(H, W)`，取值 `[0, π)`（弧度）。
- 随机性：所有函数通过 `rng: np.random.Generator` 参数接收随机源，
  **禁止**在函数内部调用 `np.random.*` 全局函数或 `random.*`。
- 禁止修改传入的数组（必须 `copy()` 或生成新数组）。
- 中文 docstring（Google 风格），关键算法注明参考文献或出处。
- 数值边界：所有随机采样结果必须 clip 到声明范围内，不允许产生 NaN/Inf。

## 1. `synthfp/plugins/tta/` —— TTA 引擎

### 1.1 `base.py`

```python
class TTAOperator(abc.ABC):
    name: str                     # 类属性：算子标识（=注册名）
    label: str                    # 类属性：中文名
    category: str                 # 类属性：几何|光度|噪声|压缩|遮挡|传感器
    distortion: float             # 类属性：典型失真度 0~1（用于累计失真截断）
    param_spec: dict[str, tuple]  # 类属性：{参数名: (中文名, 最小值, 最大值, 步长, 单位)}

    def __init__(self, params: dict | None = None) -> None: ...
        # self.params：用户覆盖的固定值（如 {"angle": 8.0}）；
        # 未覆盖的参数从 param_spec 的 [min,max] 内随机采样。

    @abc.abstractmethod
    def sample_params(self, rng: np.random.Generator) -> dict: ...
        # 注意：若 self.params 中已给出某参数，必须直接采用该值（不再随机）。

    @abc.abstractmethod
    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...
        # 等价于 sample_params + _apply，并把结果记录到 self.last_params

    @abc.abstractmethod
    def _apply(self, image: np.ndarray, params: dict) -> np.ndarray: ...

    last_params: dict             # 最近一次 apply 使用的完整参数（供 manifest 记录）

    def describe(self) -> str: ...    # "旋转(角度=7.5°)" 之类的人类可读描述
```

### 1.2 `registry.py`

```python
class TTARegistry:
    @classmethod
    def register(cls, name: str, label: str, category: str, distortion: float,
                 param_spec: dict | None = None) -> Callable[[type], type]: ...
        # 装饰器：注册算子类，并写入类属性 name/label/category/distortion/param_spec
    @classmethod
    def create(cls, name: str, params: dict | None = None) -> TTAOperator: ...
    @classmethod
    def get(cls, name: str) -> type[TTAOperator]: ...
    @classmethod
    def all_names(cls) -> list[str]: ...
    @classmethod
    def by_category(cls) -> dict[str, list[str]]: ...
    @classmethod
    def is_registered(cls, name: str) -> bool: ...
```

未知算子名必须抛 `KeyError`，错误信息里列出可用算子。

### 1.3 `operators.py` —— 必须实现全部 21 个算子

| 注册名 | 中文 | 分类 | 失真度 | 参数（名, 范围, 单位） |
|---|---|---|---|---|
| `rotate` | 旋转 | 几何 | 0.25 | angle: 0~180 °（实际随机为 ±angle） |
| `translate` | 平移 | 几何 | 0.15 | shift: 0~0.5 ×尺寸（随机 ±） |
| `scale` | 缩放 | 几何 | 0.20 | factor: 0~0.6 ×（随机 ±） |
| `shear` | 错切 | 几何 | 0.30 | amount: 0~0.6 ×（随机 ±） |
| `elastic` | 弹性形变 | 几何 | 0.45 | alpha: 0~40 px, sigma: 4~120 px |
| `perspective` | 透视 | 几何 | 0.40 | amount: 0~0.25 × |
| `brightness` | 亮度 | 光度 | 0.20 | delta: 0~0.6 ×（随机 ±） |
| `contrast` | 对比度 | 光度 | 0.25 | factor: 0~0.8 ×（随机 ±） |
| `gamma` | Gamma | 光度 | 0.25 | gamma: 0.2~3.0 × |
| `blur` | 高斯模糊 | 光度 | 0.35 | sigma: 0~4.0 px |
| `motion_blur` | 运动模糊 | 光度 | 0.40 | length: 1~25 px |
| `gaussian_noise` | 高斯噪声 | 噪声 | 0.30 | sigma: 0~0.15 × |
| `poisson` | 泊松噪声 | 噪声 | 0.30 | scale: 10~400 ×（越小噪声越强） |
| `salt_pepper` | 椒盐噪声 | 噪声 | 0.30 | amount: 0~0.15 × |
| `jpeg` | JPEG 压缩 | 压缩 | 0.35 | quality: 5~100（随机取 [quality,100]，quality 为下界） |
| `occlusion_rect` | 矩形遮挡 | 遮挡 | 0.50 | count: 1~8 块, area: 0.01~0.3 × |
| `occlusion_ellipse` | 椭圆遮挡 | 遮挡 | 0.50 | count: 1~8 个, area: 0.01~0.3 × |
| `scratch` | 划痕遮挡 | 遮挡 | 0.40 | count: 1~12 条, width: 1~12 px |
| `sensor_optical` | 光学传感器 | 传感器 | 0.30 | strength: 0~3.0 × |
| `sensor_capacitive` | 电容传感器 | 传感器 | 0.30 | strength: 0~3.0 × |
| `sensor_thermal` | 热感传感器 | 传感器 | 0.30 | strength: 0~3.0 × |

几何算子实现要求：使用 `cv2.warpAffine` / `cv2.warpPerspective` / `cv2.remap`，
边界填充用 `cv2.BORDER_REPLICATE`（指纹背景为白，避免出现黑边）。
旋转/平移/缩放后仍保持原图尺寸。

遮挡算子：遮挡区域填充为该图的背景值（`image` 的高分位灰度，默认 1.0），
边缘可轻度羽化，不要填 0（会造出黑块）。

传感器算子：
- optical：径向暗角（vignette）+ 低对比度 + 轻微散斑
- capacitive：高频噪声 + 少量椒盐 + 轻微行噪声
- thermal：低频漂移 + 行方向条带 + 非线性响应

### 1.4 `__init__.py`
导出 `TTAOperator`、`TTARegistry`、`create_operators`、`OPERATORS`(注册名列表)，
并在导入时确保 `operators.py` 已加载注册。

```python
def create_operators(names: Sequence[str], params: dict[str, dict] | None = None) -> list[TTAOperator]:
    """按名称批量实例化算子（保持 names 顺序）。unknown 名称抛 KeyError。"""
```

### 1.5 测试 `tests/test_tta.py`（≥ 30 个用例）
覆盖：注册表完整性（21 个算子）、每个算子 `sample_params` 在范围内、
`apply` 不改变形状/dtype/范围、不修改输入、确定性（同种子同结果）、
几何算子保持尺寸、遮挡算子不产生纯黑块、`create_operators` 顺序与报错。

## 2. `synthfp/plugins/quality/` —— 质量评估

### 2.1 `metrics.py`

```python
def estimate_orientation(
    image: np.ndarray,        # float32 [0,1] 或 uint8，自动识别
    block: int = 16,
    smooth_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """梯度法方向场估计。

    Returns:
        (orientation, certainty)
        orientation: float32 (H,W)，弧度 ∈ [0, π)
        certainty:   float32 (H,W)，∈ [0,1]，梯度一致性
    """

def orientation_certainty(orientation: np.ndarray, certainty: np.ndarray,
                          mask: np.ndarray | None = None) -> float:
    """掩模内 certainty 的均值 ∈ [0,1]。"""

def ridge_clarity(image: np.ndarray, orientation: np.ndarray,
                  mask: np.ndarray | None = None,
                  period: float = 10.0) -> float:
    """Gabor 响应强度度量脊线清晰度 ∈ [0,1]。"""

def local_contrast(image: np.ndarray, mask: np.ndarray | None = None,
                   block: int = 16) -> float:
    """局部标准差均值（归一化到 [0,1]）。"""

def usable_area(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """有效（非饱和、非背景）面积占比 ∈ [0,1]。"""

def estimate_ridge_period(image: np.ndarray, orientation: np.ndarray,
                          mask: np.ndarray | None = None) -> float:
    """沿方向场正交方向做投影，估计平均脊线周期（像素）。默认回退 10.0。"""
```

所有函数在 `mask` 为空或图像全同色时必须**安全返回 0.0**，不得抛异常或产生 NaN。

### 2.2 `evaluator.py`

```python
class QualityEvaluator:
    """自研 NFIQ2 风格综合质量评分器（无需外部依赖）。"""
    def __init__(self, min_score: float = 0.0, reject_on_fail: bool = False) -> None: ...
    def evaluate(self, image: np.ndarray, mask: np.ndarray | None = None,
                 orientation: np.ndarray | None = None) -> QualityReport: ...
    @property
    def name(self) -> str: ...   # "synthfp-nfiq2-lite"
```

`QualityReport` 定义在 `synthfp/core/types.py`（已存在，勿改字段名）：
`score(0-100), level(1..5), clarity, contrast, orientation_certainty,
usable_area, ridge_period, minutiae_count, passed`。

**评分规则（必须实现）**
1. 五个分项各归一化到 `[0,1]`：`clarity`、`contrast`、`orientation_certainty`、`usable_area`、`ridge_period_score`
   （`ridge_period_score`：周期在 6~16 px 内为 1，偏离则线性衰减到 0）。
2. `score = 100 * (0.35*clarity + 0.20*contrast + 0.25*orientation_certainty + 0.10*usable_area + 0.10*ridge_period_score)`
3. `level` 由 score 分箱：≥80→1，≥65→2，≥50→3，≥35→4，否则 5。
4. `passed = (score >= min_score) and (usable_area >= min_usable_area)`；
   当 `reject_on_fail=True` 时由 pipeline 依据 `passed` 丢弃样本。
5. `min_score` 与 `min_usable_area` 都通过构造函数传入（默认 0）。

### 2.3 测试 `tests/test_quality.py`（≥ 15 个用例）
- 白噪声图 → 低分；理想正弦条纹图 → 高分（≥ 55）。
- 纯色图 / 空掩模 / 全零图 → 不崩溃、无 NaN。
- 确定性：同一输入两次评分完全一致。
- `level` 分箱边界正确。

## 3. `synthfp/plugins/minutiae/` —— 细节点提取

### 3.1 `extractor.py`

```python
class MinutiaeExtractor:
    """基于骨架化 + Crossing Number 的细节点提取器（含伪细节点过滤）。"""
    def __init__(self, min_distance: float = 8.0, border_margin: int = 6,
                 max_count: int = 200, compute_angle: bool = True) -> None: ...
    def extract(self, image: np.ndarray, mask: np.ndarray | None = None,
                orientation: np.ndarray | None = None) -> list[Minutia]: ...
    @property
    def name(self) -> str: ...  # "synthfp-cn"
```

`Minutia` 定义在 `synthfp/core/types.py`（字段：`x, y, angle, kind, quality`；
`kind ∈ {"ending","bifurcation"}`；`angle` 为弧度 ∈ [0, 2π)）。

**算法要求（必须实现）**
1. 二值化：`uint8` → float → 局部阈值/Otsu（`cv2.threshold(..., THRESH_OTSU)`），
   脊线为前景。（注意背景为白：需要先取反，使脊线为 1。）
2. 细化：**必须用向量化的 Zhang-Suen**（`synthfp/plugins/minutiae/skeleton.py`,
   `def zhang_suen(binary: np.ndarray) -> np.ndarray`，纯 numpy 查表实现，
   每次迭代用 8 邻域数组运算，禁止 Python 逐像素循环）。
   同时提供 `def skeleton_minutiae(skeleton: np.ndarray) -> list[tuple[int,int,str]]`。
3. Crossing Number：`CN == 1` → ending，`CN == 3` → bifurcation。
4. 伪细节点过滤：
   - 距 `border_margin` 内的一律丢弃；
   - 掩模外丢弃；
   - 相互距离 < `min_distance` 的合并（保留 quality 更高的一个）；
   - 剔除长度小于 10 px 的骨架短枝产生的细节点。
5. `quality`：由局部对比度/方向一致性给出 `[0,1]`：
   若传入了 `orientation`，用 `cn` 邻域方向一致性；否则用局部对比度。
6. `compute_angle=True` 时，`angle` 取局部方向场（若未传入 orientation 则
   用 `synthfp.plugins.quality.metrics.estimate_orientation` 现场估计）。
7. 结果按 `(y, x)` 排序保证确定性；超过 `max_count` 时按 quality 降序截断后重排。

### 3.2 `iso_template.py`

```python
def to_iso_template(minutiae: Sequence[Minutia], width: int, height: int,
                    ppi: int = 500) -> bytes:
    """ISO/IEC 19794-2:2005 简约格式（格式标识 0）字节流。

    记录：头部 24 字节（格式标识 'FMR\\0' + 版本 + 记录长度 + 宽高 + 分辨率 +
    细节点数），之后每个细节点 6 字节（type| x 高 2bit, x, y, 方向单字节 0~255,
    质量单字节）。
    """
```

### 3.3 测试 `tests/test_minutiae.py`（≥ 15 个用例）
- `zhang_suen` 对细横线/十字/方块的期望骨架。
- 合成“一条线 + 一个断点”的图 → 检出正确数量的 ending。
- 交叉图 → 检出 bifurcation。
- 边界/掩模外细节点被丢弃；`min_distance` 合并生效；确定性；ISO 模板长度 = 24 + 6n。

## 4. `synthfp/plugins/impressions/` —— 捺印效果

### 4.1 `deform.py`

```python
def pressure_field(height: int, width: int, pressure: float, rng: np.random.Generator,
                   mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """按压力度生成位移场。

    模型：双曲正切径向模型（中心受压缩、边缘向外扩展）：
        r_norm = 归一化到 [0,1] 的到手指中心的距离
        scale(r) = 1 + p * (tanh(k*(r_norm - 0.5)) * 2)   # 中心 <1, 边缘 >1
    再叠加低频平滑随机场（弹性抖动）。

    Returns:
        (dx, dy): float32 (H, W)，单位为像素（目标像素 -> 源像素的偏移）
    """

def deform_image(image: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                 border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """用 cv2.remap 施加位移场，保持尺寸与 dtype。"""

def impression_transform(height: int, width: int, pressure: float, rotation: float,
                         shift: tuple[float, float], rng: np.random.Generator,
                         mask: np.ndarray | None = None,
                         elastic_sigma: float = 28.0, elastic_alpha: float = 9.0
                         ) -> tuple[np.ndarray, np.ndarray]:
    """组合变换：旋转 + 平移（仿射）+ 压力位移场。

    Returns:
        (dx, dy) 总位移场（像素），可直接用于 deform_image 与 map_coordinates。
    """

def map_coordinates(dx: np.ndarray, dy: np.ndarray,
                    points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """把源图坐标经位移场映射到目标图坐标（用于标注同步变换）。

    对每个点 p，在位移场中双线性插值得到 (ddx, ddy)，返回 (x+ddx, y+ddy)。
    越界点被裁剪到图像范围内。
    """
```

### 4.2 `sensor.py`

```python
class SensorProfile(abc.ABC):
    name: str; label: str
    def apply(self, image: np.ndarray, rng: np.random.Generator,
              strength: float = 1.0) -> np.ndarray: ...

class OpticalSensor(SensorProfile): ...      # 暗角 + 低对比 + 散斑
class CapacitiveSensor(SensorProfile): ...   # 高频噪声 + 椒盐 + 行噪声
class ThermalSensor(SensorProfile): ...      # 低频漂移 + 条带 + 非线性响应
class NoneSensor(SensorProfile): ...         # 直通

SENSORS: dict[str, type[SensorProfile]]   # {"optical":..., "capacitive":..., "thermal":..., "none":...}
def get_sensor(name: str) -> SensorProfile: ...   # 未知名回退 NoneSensor
```

输入输出统一 `uint8 (H,W)`；`strength=0` 时必须是**逐像素完全一致**的直通。

### 4.3 测试 `tests/test_impressions.py`（≥ 12 个用例）
- 位移场形状/dtype/有限性；`strength=0` 直通；压力越大中心位移越大（单调性）。
- `deform_image` 保持尺寸与 dtype；`map_coordinates` 对零位移场是恒等映射（±0.51px）。
- 每个传感器输出在 `[0,255]`、不修改输入、确定性、`strength=0` 直通。

## 5. 交付要求

- 代码可被 `python -c "import ..."` 直接导入（不要有副作用/打印/写文件）。
- 每个子模块提供 `__init__.py` 导出上述公开符号。
- 自己写测试并**实际运行**：`python -m pytest tests/test_xxx.py -q` 必须全绿。
- 完成后报告：新增文件清单、公开 API、测试运行结果（通过数）、已知限制。
