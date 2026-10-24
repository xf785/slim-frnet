"""可复用控件 —— 图像查看器、统计卡片、日志面板、缩略图网格。

这些控件被多个页面共用，因此统一放在这里，避免各页面重复实现。
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .qt import (
    QColor,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QImage,
    QLabel,
    QPainter,
    QPen,
    QPixmap,
    QPlainTextEdit,
    QPointF,
    QSizePolicy,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)

__all__ = [
    "Card",
    "StatCard",
    "ImageView",
    "LogPanel",
    "ThumbnailGrid",
    "FormRow",
    "SectionTitle",
    "numpy_to_qimage",
    "numpy_to_pixmap",
]

_LOG_COLORS = {
    "DEBUG": "#8a8a96",
    "INFO": "#dcdce2",
    "WARNING": "#ffb020",
    "ERROR": "#ff453a",
    "OK": "#34c759",
}


# ---------------------------------------------------------------------------
# 图像转换
# ---------------------------------------------------------------------------
def numpy_to_qimage(array: np.ndarray) -> QImage:
    """``numpy`` 灰度/RGB/RGBA 数组 → ``QImage``（内部持有数据副本）。"""
    arr = np.ascontiguousarray(array)
    if arr.ndim == 2:
        arr = np.ascontiguousarray(arr.astype(np.uint8))
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = np.ascontiguousarray(arr.astype(np.uint8))
        h, w, _ = arr.shape
        return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = np.ascontiguousarray(arr.astype(np.uint8))
        h, w, _ = arr.shape
        return QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
    raise ValueError(f"不支持的图像形状: {arr.shape}")


def numpy_to_pixmap(array: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(numpy_to_qimage(array))


# ---------------------------------------------------------------------------
# 卡片
# ---------------------------------------------------------------------------
class Card(QFrame):
    """带标题的卡片容器。"""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 12)
        self._layout.setSpacing(8)
        if title:
            label = QLabel(title)
            label.setObjectName("Accent")
            self._layout.addWidget(label)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)


class StatCard(QFrame):
    """单个统计数字卡片：大号数值 + 说明。"""

    def __init__(self, title: str, value: str = "—", hint: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumWidth(110)
        box = QVBoxLayout(self)
        box.setContentsMargins(12, 8, 12, 8)
        box.setSpacing(2)

        self._title = QLabel(title)
        self._title.setObjectName("Hint")
        self._value = QLabel(value)
        self._value.setObjectName("StatValue")
        self._hint = QLabel(hint)
        self._hint.setObjectName("Hint")
        self._hint.setWordWrap(True)

        box.addWidget(self._title)
        box.addWidget(self._value)
        box.addWidget(self._hint)

    def set_value(self, value: str, hint: str = "") -> None:
        self._value.setText(value)
        if hint:
            self._hint.setText(hint)


class SectionTitle(QLabel):
    """小节标题。"""

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("Accent")
        font = self.font()
        font.setPointSizeF(font.pointSizeF() + 1)
        self.setFont(font)


class FormRow(QWidget):
    """一行「标签 + 控件 + 可选说明」。"""

    def __init__(self, label: str, widget: QWidget, hint: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        box = QHBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)
        self.label = QLabel(label)
        self.label.setMinimumWidth(112)
        self.widget = widget
        box.addWidget(self.label)
        box.addWidget(widget, 1)
        if hint:
            self.hint = QLabel(hint)
            self.hint.setObjectName("Hint")
            box.addWidget(self.hint)
        else:
            self.hint = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 图像视图
# ---------------------------------------------------------------------------
class ImageView(QGraphicsView):
    """支持滚轮缩放 / 拖拽平移的图像查看器，可叠加细节点与方向场。"""

    pixel_hovered = Signal(int, int, int)  # x, y, gray

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self._zoom = 1.0
        self._image: Optional[np.ndarray] = None
        self._fitted = False
        self._empty_text = "（无图像）"
        self._show_minutiae = True
        self._show_orientation = False
        self._show_singular = True
        self._overlay_scale = 1.0
        self._minutiae: List = []
        self._orientation: Optional[np.ndarray] = None

    # -- 数据 -------------------------------------------------------------
    def set_image(
        self,
        image: Optional[np.ndarray],
        minutiae: Optional[Sequence] = None,
        orientation: Optional[np.ndarray] = None,
        singular_points: Optional[Sequence] = None,
        reset_zoom: bool = True,
    ) -> None:
        """设置底图与叠加层。"""
        self._image = image
        self._minutiae = list(minutiae or [])
        self._orientation = orientation
        self._singular = list(singular_points or [])
        self._redraw()
        if reset_zoom or not self._fitted:
            self.fit()

    def clear_image(self, text: str = "（无图像）") -> None:
        self._image = None
        self._minutiae = []
        self._orientation = None
        self._singular = []
        self._empty_text = text
        self._scene.clear()
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self._scene.setSceneRect(0, 0, 1, 1)

    def set_overlays(self, show_minutiae: Optional[bool] = None,
                     show_orientation: Optional[bool] = None,
                     show_singular: Optional[bool] = None,
                     scale: Optional[float] = None) -> None:
        if show_minutiae is not None:
            self._show_minutiae = show_minutiae
        if show_orientation is not None:
            self._show_orientation = show_orientation
        if show_singular is not None:
            self._show_singular = show_singular
        if scale is not None:
            self._overlay_scale = scale
        self._redraw()

    # -- 缩放 -------------------------------------------------------------
    def fit(self) -> None:
        if self._image is None:
            return
        self.resetTransform()
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self._fitted = True

    def set_zoom(self, level: float) -> None:
        if self._image is None:
            return
        level = max(0.05, min(level, 30.0))
        factor = level / max(self._zoom, 1e-6)
        self.scale(factor, factor)
        self._zoom = level

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / 1.25)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt 约定
        if self._image is None:
            return
        delta = event.angleDelta().y()
        self.set_zoom(self._zoom * (1.18 if delta > 0 else 1 / 1.18))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        super().mouseMoveEvent(event)
        if self._image is None:
            return
        point = self.mapToScene(event.position().toPoint())
        x, y = int(point.x()), int(point.y())
        h, w = self._image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.pixel_hovered.emit(x, y, int(self._image[y, x]))

    # -- 绘制 -------------------------------------------------------------
    def _redraw(self) -> None:
        if self._image is None:
            return
        pixmap = numpy_to_pixmap(self._image)
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(self._item.boundingRect())

        painter = QPainter(pixmap)
        try:
            if self._show_orientation and self._orientation is not None:
                self._paint_orientation(painter, pixmap.width(), pixmap.height())
            if self._show_minutiae and self._minutiae:
                self._paint_minutiae(painter)
            if self._show_singular and getattr(self, "_singular", None):
                self._paint_singular(painter)
        finally:
            painter.end()
        self._item.setPixmap(pixmap)

    def _paint_orientation(self, painter: QPainter, width: int, height: int) -> None:
        ori = self._orientation
        if ori is None:
            return
        step = max(int(min(width, height) / 26), 6)
        pen = QPen(QColor(255, 60, 60, 170))
        pen.setWidth(1)
        painter.setPen(pen)
        length = step * 0.42
        for y in range(step // 2, height, step):
            for x in range(step // 2, width, step):
                theta = float(ori[y, x])
                dx, dy = length * np.cos(theta), length * np.sin(theta)
                painter.drawLine(
                    QPointF(x - dx, y - dy), QPointF(x + dx, y + dy)
                )

    def _paint_minutiae(self, painter: QPainter) -> None:
        radius = 3.0 * self._overlay_scale
        for minutia in self._minutiae:
            x = float(getattr(minutia, "x", 0))
            y = float(getattr(minutia, "y", 0))
            kind = str(getattr(minutia, "kind", "ending"))
            color = QColor(30, 144, 255) if kind == "ending" else QColor(255, 90, 0)
            painter.setPen(QPen(color, 1.4))
            painter.setBrush(color)
            if kind == "ending":
                painter.drawEllipse(QPointF(x, y), radius, radius)
            else:
                painter.drawRect(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))
            angle = float(getattr(minutia, "angle", 0.0))
            if angle:
                painter.setPen(QPen(color, 1.2))
                painter.drawLine(
                    QPointF(x, y),
                    QPointF(x + radius * 2.4 * np.cos(angle), y + radius * 2.4 * np.sin(angle)),
                )

    def _paint_singular(self, painter: QPainter) -> None:
        size = 7.0 * self._overlay_scale
        for point in getattr(self, "_singular", []):
            x, y = float(point.get("x", 0)), float(point.get("y", 0))
            kind = str(point.get("type", "core"))
            if kind == "core":
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(x, y), size, size)
                painter.setBrush(QColor(255, 0, 0))
                painter.drawEllipse(QPointF(x, y), 2.0, 2.0)
            else:
                painter.setPen(QPen(QColor(0, 170, 0), 2))
                painter.setBrush(QColor(0, 200, 0, 120))
                painter.drawPolygon(
                    [
                        QPointF(x, y - size),
                        QPointF(x + size, y + size * 0.8),
                        QPointF(x - size, y + size * 0.8),
                    ]
                )
            painter.setBrush(Qt.BrushStyle.NoBrush)


# ---------------------------------------------------------------------------
# 缩略图网格
# ---------------------------------------------------------------------------
class ThumbnailGrid(QWidget):
    """缩略图网格（点击选中某张后发出信号）。"""

    selected = Signal(int, object)  # index, payload

    def __init__(self, columns: int = 4, thumb_size: Tuple[int, int] = (110, 150),
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._columns = columns
        self._thumb = thumb_size
        self._items: List[Tuple[np.ndarray, object]] = []
        self._labels: List[QLabel] = []
        self._grid = QGridLayout(self)
        self._grid.setSpacing(6)
        self._grid.setContentsMargins(2, 2, 2, 2)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

    def clear(self) -> None:
        while self._grid.count():
            child = self._grid.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()
        self._items.clear()
        self._labels.clear()

    def set_items(self, items: Sequence[Tuple[np.ndarray, object]]) -> None:
        """``items`` 为 ``[(图像, 附加数据)]``。"""
        self.clear()
        self._items = list(items)
        for index, (image, payload) in enumerate(self._items):
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFixedSize(*self._thumb)
            label.setCursor(Qt.CursorShape.PointingHandCursor)
            label.setToolTip(str(payload))
            pixmap = numpy_to_pixmap(image).scaled(
                *self._thumb,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label.setPixmap(pixmap)
            label.setStyleSheet("QLabel { border: 1px solid #3d3d46; border-radius: 4px; }")
            label.mousePressEvent = self._make_handler(index)  # type: ignore[assignment]
            self._labels.append(label)
            self._grid.addWidget(label, index // self._columns, index % self._columns)

    def _make_handler(self, index: int):
        def handler(_event) -> None:
            for i, label in enumerate(self._labels):
                label.setStyleSheet(
                    "QLabel { border: 2px solid #0a84ff; border-radius: 4px; }"
                    if i == index
                    else "QLabel { border: 1px solid #3d3d46; border-radius: 4px; }"
                )
            if index < len(self._items):
                self.selected.emit(index, self._items[index][1])

        return handler


# ---------------------------------------------------------------------------
# 日志面板
# ---------------------------------------------------------------------------
class LogPanel(QWidget):
    """彩色日志面板（按级别过滤 + 上限行数）。"""

    def __init__(self, max_lines: int = 4000, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(max_lines)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = self.view.font()
        font.setFamilies(["Cascadia Mono", "Consolas", "Menlo", "monospace"])
        font.setPointSizeF(font.pointSizeF() - 0.5)
        self.view.setFont(font)
        box.addWidget(self.view)
        self._counts: Dict[str, int] = {}

    def append(self, level: str, message: str) -> None:
        level = level.upper()
        self._counts[level] = self._counts.get(level, 0) + 1
        color = _LOG_COLORS.get(level, "#dcdce2")
        text = html.escape(str(message)).replace("\n", "<br>")
        self.view.appendHtml(
            f'<span style="color:{color}"><b>[{level[:4]}]</b> {text}</span>'
        )
        self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum())

    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def clear(self) -> None:
        self.view.clear()
        self._counts.clear()

    def to_text(self) -> str:
        return self.view.toPlainText()
