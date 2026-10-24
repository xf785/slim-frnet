"""页面基类 —— 提供「配置 ↔ 控件」双向绑定的脚手架。

每个页面继承 :class:`BasePage`，用声明式的方式登记控件绑定，
基类负责：

* 控件 → 配置（用户操作后写回，自动 Pydantic 校验，失败则回滚控件值并提示）
* 配置 → 控件（配置被其它页面/文件改变时刷新控件）
* 抑制递归（刷新控件时不再触发写回）
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ...config import AppConfig
from ..config_store import ConfigStore
from ..qt import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    Qt,
    QVBoxLayout,
    QWidget,
)

__all__ = ["BasePage", "FormBuilder"]


# ---------------------------------------------------------------------------
# 配置 → 页面 的连接注册表
# ---------------------------------------------------------------------------
#: ``{id(page): (store, bound_slot)}`` —— 强持有槽函数，避免 GC 时机不确定；
#: 键用 ``id()`` 而非页面对象本身，因此不会阻止页面被回收。
_LIVE_CONNECTIONS: Dict[int, Tuple[Any, Any]] = {}


def _on_page_destroyed(page: Any) -> None:
    """页面底层控件析构时立刻断开配置广播。

    这是 ``QObject.destroyed`` 的槽，由 C++ 侧调用；此时只允许访问注册表，
    **不允许**再碰任何 Qt 子对象（它们可能已经先一步析构）。
    """
    entry = _LIVE_CONNECTIONS.pop(id(page), None)
    if entry is None:
        return
    store, slot = entry
    try:
        store.changed.disconnect(slot)
    except (RuntimeError, TypeError):  # 已经断开 / 对象已析构
        pass


class BasePage(QWidget):
    """所有配置页面的基类。"""

    #: 页面标题（导航栏显示）
    title: str = "页面"
    #: 导航图标（emoji，避免额外资源文件）
    icon: str = "•"
    #: 是否给页面套一层外层滚动区。
    #:
    #: 配置类页面（内容很长、纵向堆叠）需要它；而监控/预览这类
    #: **自己就管理滚动**的页面必须关掉 —— 嵌套两层 QScrollArea 会互相争尺寸，
    #: 把内层内容压成 0 高度（实测会让树表、缩略图整块消失）。
    scrollable: bool = True

    def __init__(self, store: ConfigStore, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.store = store
        self._updating = False
        self._bindings: List[Tuple[str, QWidget, Callable, Callable]] = []
        self._groups: List[QGroupBox] = []

        if self.scrollable:
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            self._scroll: Optional[QScrollArea] = QScrollArea()
            self._scroll.setWidgetResizable(True)
            self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            self._content = QWidget()
            self._scroll.setWidget(self._content)
            outer.addWidget(self._scroll)
            self._layout = QVBoxLayout(self._content)
        else:
            # 直接把 `self` 当内容容器：内层自带的滚动控件才有完整空间。
            # 注意不能先建一个 QVBoxLayout(self) 再建第二个。
            self._scroll = None
            self._content = self
            self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 12, 14, 12)
        self._layout.setSpacing(10)

        self.build()
        # 建立「配置 → 页面」的连接。
        #
        # 这里刻意不依赖 Qt 的自动断开语义：它只在页面的 Python 包装对象被回收时
        # 生效，而 GC 时机不确定；一旦控件已被析构、连接却还在，刷新就会撞上
        # ``RuntimeError: wrapped C/C++ object ... has been deleted``
        # （表现为"配置一改整个程序就崩"）。
        #
        # 做法：用模块级注册表**强持有** (store, slot)，并挂 ``destroyed`` 钩子，
        # 控件一析构就立刻主动断开。注册表用 id() 作键，因此不会因为持有页面引用
        # 而阻止页面被回收。
        self.store.changed.connect(self.refresh)
        _LIVE_CONNECTIONS[id(self)] = (self.store, self.refresh)
        self.destroyed.connect(_on_page_destroyed)

        # 首屏必须自己做一次「配置 → 控件」，否则页面会以控件默认值显示
        # （输入框空白、数值是 QSpinBox 的最小值、表格没有行），
        # 和当前配置完全脱节 —— 之前的 bug 就是这样藏住的。
        self.refresh()

    def build(self) -> None:
        """构建页面控件（子类必须实现）。"""
        raise NotImplementedError

    def refresh(self) -> None:
        """配置变化时刷新控件（按登记的绑定 + 子类的 ``on_refresh``）。

        ``_updating`` 必须覆盖 ``on_refresh()`` 全程：子类在那里会直接
        ``setValue`` / ``setChecked`` 重建控件（例如类别比例表），
        这些调用同样会触发写回。若不加保护，刷新本身就会改动配置、
        把 store 标成"有未保存修改"，随后关窗口时弹出模态对话框
        （自动化环境里表现为卡死）。
        """
        self._updating = True
        try:
            for path, widget, _getter, setter in self._bindings:
                if not _alive(widget):
                    continue
                try:
                    value = _dig(self.store.config, path)
                except (AttributeError, KeyError):
                    continue
                setter(widget, value)
            self.on_refresh()
        except RuntimeError:
            # 子类持有的某个控件已被销毁（页面正在析构）。Qt 的弱引用连接
            # 依赖 Python GC，而 GC 时机不确定，因此这里再兜一层。
            return
        finally:
            self._updating = False

    def on_refresh(self) -> None:
        """子类的额外刷新逻辑（不涉及绑定的控件）。"""
        return None

    # -- 布局辅助 ---------------------------------------------------------
    def add_widget(self, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        return widget

    def add_stretch(self) -> None:
        self._layout.addStretch(1)

    def group(self, title: str) -> Tuple[QGroupBox, QFormLayout]:
        """新建一个分组框并返回 ``(group, form_layout)``。"""
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setContentsMargins(12, 14, 12, 12)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self._groups.append(box)
        self._layout.addWidget(box)
        return box, form

    def hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("Hint")
        label.setWordWrap(True)
        self._layout.addWidget(label)
        return label

    def button_row(self, *buttons: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        for button in buttons:
            row.addWidget(button)
        row.addStretch(1)
        self._layout.addLayout(row)
        return row

    # -- 绑定辅助 ---------------------------------------------------------
    def bind_spin(
        self,
        path: str,
        spin: QSpinBox,
        *,
        on_change: Optional[Callable[[int], None]] = None,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
        suffix: str = "",
    ) -> QSpinBox:
        """把 ``QSpinBox`` 绑定到配置的整数项。"""
        if minimum is not None:
            spin.setRange(minimum, maximum if maximum is not None else 2**31 - 1)
        if suffix:
            spin.setSuffix(suffix)
        spin.valueChanged.connect(
            lambda value, p=path, cb=on_change: self._push(p, int(value), cb)
        )
        self._bindings.append(
            (path, spin, _getter_int, lambda w, v: w.setValue(int(v)))
        )
        return spin

    def bind_double(
        self,
        path: str,
        spin: QDoubleSpinBox,
        *,
        on_change: Optional[Callable[[float], None]] = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        decimals: Optional[int] = None,
        step: Optional[float] = None,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        """把 ``QDoubleSpinBox`` 绑定到配置的浮点项。

        ``decimals`` / ``step`` 默认是 ``None``，表示**沿用调用方已经设好的值**。
        早先这里默认写成 ``decimals=2`` 并强制 ``setDecimals``，会把调用方
        精心设好的 5 位小数（如 ``convergence_eps=0.001``）覆写成 ``0.00`` ——
        界面看起来像 0，用户根本没法调。
        """
        if minimum is not None:
            spin.setRange(minimum, maximum if maximum is not None else 1e9)
        if decimals is not None:
            spin.setDecimals(max(int(decimals), 0))
        if step is not None:
            spin.setSingleStep(step)
        if suffix:
            spin.setSuffix(suffix)
        spin.valueChanged.connect(
            lambda value, p=path, cb=on_change: self._push(p, float(value), cb)
        )
        self._bindings.append((path, spin, _getter_float, lambda w, v: w.setValue(float(v))))
        return spin

    def bind_check(
        self,
        path: str,
        check: QCheckBox,
        *,
        on_change: Optional[Callable[[bool], None]] = None,
    ) -> QCheckBox:
        """把 ``QCheckBox`` 绑定到配置的布尔项。"""
        check.toggled.connect(lambda value, p=path, cb=on_change: self._push(p, bool(value), cb))
        self._bindings.append((path, check, _getter_bool, lambda w, v: w.setChecked(bool(v))))
        return check

    def bind_combo(
        self,
        path: str,
        combo: QComboBox,
        *,
        values: Sequence[str],
        labels: Optional[Sequence[str]] = None,
        on_change: Optional[Callable[[str], None]] = None,
    ) -> QComboBox:
        """把 ``QComboBox`` 绑定到配置的枚举项。"""
        combo.clear()
        for value in values:
            combo.addItem((labels[list(values).index(value)] if labels else value), value)
        combo.currentIndexChanged.connect(
            lambda _idx, p=path, c=combo, cb=on_change: self._push(p, c.currentData(), cb)
        )
        self._bindings.append(
            (
                path,
                combo,
                _getter_str,
                lambda w, v: _select_data(w, str(v)),
            )
        )
        return combo

    def bind_line(
        self,
        path: str,
        line: QLineEdit,
        *,
        on_change: Optional[Callable[[str], None]] = None,
        apply_on_edit: bool = True,
    ) -> QLineEdit:
        """把 ``QLineEdit`` 绑定到配置的字符串/路径项。"""
        if apply_on_edit:
            line.editingFinished.connect(
                lambda p=path, w=line, cb=on_change: self._push(p, w.text(), cb)
            )
        else:
            line.textEdited.connect(lambda text, p=path, cb=on_change: self._push(p, text, cb))
        self._bindings.append((path, line, _getter_str, lambda w, v: w.setText(str(v))))
        return line

    def bind_interval(
        self,
        path: str,
        low: QDoubleSpinBox,
        high: QDoubleSpinBox,
        *,
        decimals: int = 2,
        step: float = 0.05,
        suffix: str = "",
        on_change: Optional[Callable[[Tuple[float, float]], None]] = None,
    ) -> None:
        """把两个 ``QDoubleSpinBox`` 绑定为配置里的 ``(下界, 上界)`` 区间。"""

        def push(*_args: Any) -> None:
            value = (round(low.value(), 6), round(high.value(), 6))
            self._push(path, value, on_change)

        for spin in (low, high):
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            if suffix:
                spin.setSuffix(suffix)
            spin.valueChanged.connect(push)

        def setter(_w: QWidget, value: Any) -> None:
            if isinstance(value, (tuple, list)) and len(value) == 2:
                low.setValue(float(value[0]))
                high.setValue(float(value[1]))

        self._bindings.append((path, low, _getter_float, setter))

    # -- 写回 -------------------------------------------------------------
    def _push(self, path: str, value: Any, callback: Optional[Callable[[Any], None]] = None) -> None:
        if self._updating:
            return
        if self.store.update_flat({path: value}):
            if callback is not None:
                callback(value)
        else:
            self.refresh()

    # -- 便捷显示 ---------------------------------------------------------
    def add_estimate_card(self) -> QLabel:
        label = QLabel()
        label.setObjectName("Card")
        label.setWordWrap(True)
        # 富文本 + 两行中文，不给最小高度会被 QSS 的 border/padding 裁掉
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setMinimumHeight(62)
        self._layout.addWidget(label)
        self._estimate_label = label
        return label

    def update_estimate(self) -> None:
        if not hasattr(self, "_estimate_label"):
            return
        cfg = self.store.config
        est = cfg.estimate()
        w, h = cfg.resolved_image_size()
        self._estimate_label.setText(
            f"<b>规模估算</b>　主指纹 <b>{est['masters']}</b>　捺印 <b>{est['impressions']}</b>　"
            f"样本 <b style='font-size:16px'>{est['samples']}</b><br>"
            f"<span style='color:#9a9aa6'>输出 {cfg.dataset_dir}　·　"
            f"图像 {w}×{h} @ {cfg.backend.ppi} ppi</span>"
        )


class FormBuilder:
    """把一个 ``QFormLayout`` 包装成流式 API，减少样板代码。"""

    def __init__(self, form: QFormLayout, page: BasePage) -> None:
        self.form = form
        self.page = page

    def spin(self, label: str, path: str, minimum: int, maximum: int, *, suffix: str = "",
             value: int = 0, on_change=None) -> QSpinBox:
        widget = QSpinBox()
        self.page.bind_spin(path, widget, on_change=on_change, minimum=minimum, maximum=maximum,
                            suffix=suffix)
        widget.setValue(value)
        self.form.addRow(label, widget)
        return widget

    def double(self, label: str, path: str, minimum: float, maximum: float, *,
               decimals: int = 2, step: float = 0.1, suffix: str = "",
               value: float = 0.0, on_change=None) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        self.page.bind_double(path, widget, on_change=on_change, minimum=minimum, maximum=maximum,
                              decimals=decimals, step=step, suffix=suffix)
        widget.setValue(value)
        self.form.addRow(label, widget)
        return widget

    def check(self, label: str, path: str, *, checked: bool = False, on_change=None) -> QCheckBox:
        widget = QCheckBox()
        self.page.bind_check(path, widget, on_change=on_change)
        widget.setChecked(checked)
        self.form.addRow(label, widget)
        return widget

    def combo(self, label: str, path: str, values: Sequence[str],
              labels: Optional[Sequence[str]] = None, *, on_change=None) -> QComboBox:
        widget = QComboBox()
        self.page.bind_combo(path, widget, values=values, labels=labels, on_change=on_change)
        self.form.addRow(label, widget)
        return widget

    def line(self, label: str, path: str, *, text: str = "", on_change=None) -> QLineEdit:
        widget = QLineEdit()
        self.page.bind_line(path, widget, on_change=on_change)
        widget.setText(text)
        self.form.addRow(label, widget)
        return widget

    def row(self, label: str, widget: QWidget) -> QWidget:
        self.form.addRow(label, widget)
        return widget


# ---------------------------------------------------------------------------
# 取值 / 设值助手
# ---------------------------------------------------------------------------
def _dig(config: AppConfig, path: str) -> Any:
    """按点号路径取值；数字段表示序列下标（``backend.ridge_gray.0``）。"""
    node: Any = config
    for part in path.split("."):
        if isinstance(node, dict):
            node = node[part]
        elif isinstance(node, (list, tuple)):
            node = node[int(part)]
        else:
            node = getattr(node, part)
    return node


def _alive(widget: QWidget) -> bool:
    """判断底层 C++ 对象是否仍然存活。

    Qt 的 Python 包装对象可能比 C++ 对象活得久（``deleteLater`` 之后），
    此时访问任何属性都会抛 ``RuntimeError``。刷新循环里必须先做这个检查。
    """
    try:
        widget.objectName()
        return True
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# 取值 / 设值助手
# ---------------------------------------------------------------------------
def _getter_int(widget: QWidget) -> int:
    return int(widget.value())  # type: ignore[attr-defined]


def _getter_float(widget: QWidget) -> float:
    return float(widget.value())  # type: ignore[attr-defined]


def _getter_bool(widget: QWidget) -> bool:
    return bool(widget.isChecked())  # type: ignore[attr-defined]


def _getter_str(widget: QWidget) -> str:
    if isinstance(widget, QComboBox):
        return str(widget.currentData())
    return str(widget.text())  # type: ignore[attr-defined]


def _select_data(combo: QComboBox, value: str) -> None:
    """按 ``userData`` 选中下拉项；找不到时退回按文本匹配。"""
    index = combo.findData(value)
    if index < 0:
        index = combo.findText(value)
    if index >= 0:
        combo.setCurrentIndex(index)


