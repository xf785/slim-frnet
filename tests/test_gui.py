"""GUI 冒烟测试 —— 无需显示器（``QT_QPA_PLATFORM=offscreen``）。

覆盖目标：**每一个页面都能构造、能刷新、能在配置变化后存活**。
原项目最大的问题就是 GUI 一启动就 `NameError`（见其 ``crash.log``），
所以这里刻意把「构造全部 8 个页面 + 主窗口 + 触发一轮配置广播 + 切换全部页面」
做成必须通过的冒烟测试。

Qt 缺失时整个模块会被跳过（CLI 与核心层仍然可用）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from synthfp.config import build_config  # noqa: E402
from synthfp.gui.config_store import ConfigStore  # noqa: E402
from synthfp.gui.pages import PAGE_CLASSES  # noqa: E402
from synthfp.gui.qt import QT_BINDING, QApplication, is_available  # noqa: E402

pytestmark = pytest.mark.skipif(
    not is_available(), reason="GUI 冒烟测试需要 PySide6 或 PyQt6"
)


@pytest.fixture(scope="module")
def qapp():
    """整个模块共用一个 QApplication（Qt 不允许同一进程有多个）。"""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def store(qapp):
    return ConfigStore(build_config("quick_test"))


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
def test_all_pages_construct(qapp, store):
    """8 个页面都要能构造出来，且声明了标题与图标。"""
    assert len(PAGE_CLASSES) == 8, "设计文档要求 8 个页面"

    pages = []
    for page_cls in PAGE_CLASSES:
        page = page_cls(store)
        assert page.title, f"{page_cls.__name__} 缺少 title"
        assert page.icon, f"{page_cls.__name__} 缺少 icon"
        pages.append(page)
    assert len(pages) == 8

    titles = [page.title for page in pages]
    assert len(set(titles)) == 8, f"页面标题重复: {titles}"


def test_pages_survive_config_broadcast(qapp, store):
    """配置被其它页面改动后，所有页面都必须能安全刷新（不抛异常）。

    这里刻意**保留页面引用**：一旦 Python 对象被回收，Qt 会析构底层控件，
    此时若信号连接还在就会 ``RuntimeError``。``BasePage`` 用
    ``connect(slot, receiver)`` 的弱引用语义保证了那种情况也不会炸，
    本测试同时覆盖「页面存活」这条正常路径。
    """
    pages = [cls(store) for cls in PAGE_CLASSES]
    with store.suspend():
        assert store.update(
            dataset={"identities": 3, "fingers_per_identity": 1, "tta_per_impression": 2}
        )
    for page in pages:
        page.refresh()
    assert len(pages) == 8


def test_destroyed_pages_do_not_break_config_broadcast(qapp, store):
    """页面被销毁后配置广播仍必须安全（防止访问已析构的 C++ 对象）。"""
    import gc

    pages = [cls(store) for cls in PAGE_CLASSES]
    for page in pages:
        page.deleteLater()
    del pages
    gc.collect()
    # 触发广播：不应抛 RuntimeError
    assert store.update_flat({"dataset.identities": 5}) is True


def test_config_store_rejects_invalid_and_keeps_previous(qapp, store):
    """非法值不得落地，且配置对象必须保持上一步的合法状态。"""
    before = store.config.dataset.identities
    errors = []
    store.error.connect(errors.append)

    assert store.update_flat({"dataset.identities": 0}) is False
    assert errors, "非法值应当触发 error 信号"
    assert store.config.dataset.identities == before


def test_config_store_suspend_batches_one_signal(qapp, store):
    """``suspend()`` 期间不应触发 changed，退出时只触发一次。"""
    hits = []
    store.changed.connect(lambda: hits.append(1))
    with store.suspend():
        store.update_flat({"dataset.identities": 4})
        store.update_flat({"dataset.fingers_per_identity": 2})
        assert hits == [], "suspend 期间不应广播"
    assert len(hits) == 1, "退出 suspend 时应恰好广播一次"


def test_config_store_yaml_roundtrip(qapp, store):
    """YAML 文本 → 配置 → YAML 文本必须稳定。"""
    text = store.to_yaml()
    assert store.load_yaml_text(text) is True
    assert store.to_yaml() == text

    assert store.load_yaml_text("dataset:\n  identities: 0\n") is False


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
def test_main_window_constructs_and_navigates(qapp, store):
    from synthfp.gui.main_window import MainWindow

    window = MainWindow(build_config("quick_test"))
    try:
        assert window.stack.count() == 8
        assert window.nav.count() == 8
        for row in range(8):
            window.nav.setCurrentRow(row)
            assert window.stack.currentIndex() == row
        # 状态栏应当反映配置
        assert "quick_test" in window.status_dataset.text()
    finally:
        window.close()
        window.deleteLater()


def test_main_window_goto_by_class_name(qapp):
    from synthfp.gui.main_window import MainWindow

    window = MainWindow(build_config("quick_test"))
    try:
        window._goto("ExportPage")
        assert window.stack.currentIndex() == 7
        window._goto("DatasetPage")
        assert window.stack.currentIndex() == 0
        window._goto("不存在的页面")  # 不应抛异常
    finally:
        window.close()
        window.deleteLater()


def test_main_window_preset_application(qapp):
    from synthfp.gui.main_window import MainWindow

    window = MainWindow(build_config("quick_test"))
    try:
        window._apply_preset("class_balanced")
        assert window.store.config.dataset.name == "synth_fp_balanced"
        assert all(w == pytest.approx(0.2) for w in
                   window.store.config.classes.distribution.values())
        # 切换预设后所有页面仍能刷新
        for page in window.pages:
            page.refresh()
    finally:
        window.close()
        window.deleteLater()


# ---------------------------------------------------------------------------
# 生成控制器（不真正跑生成，只验证状态机）
# ---------------------------------------------------------------------------
def test_generation_controller_state_machine(qapp, store):
    from synthfp.gui.qt_bridge import GenerationController

    controller = GenerationController(store.config)
    assert controller.is_running is False
    assert controller.is_paused is False
    # 未启动时暂停/恢复/停止都是安全的空操作
    controller.pause()
    controller.resume()
    controller.stop()
    assert controller.is_running is False
    # stats 在未运行时也要可用
    assert controller.stats.total == 0
    assert controller.stats.rate == 0.0


def test_gui_driven_generation_end_to_end(qapp, tmp_path):
    """从 GUI 控制器真正跑一遍生成：信号齐全、产物落盘、可再续跑。

    这是 GUI 与后台线程、信号队列、服务层、管道、导出之间的**整链路**验证，
    离线冒烟测试（只构造函数）覆盖不到它。
    """
    import time
    from pathlib import Path

    from synthfp.config import apply_overrides, build_config
    from synthfp.gui.qt_bridge import GenerationController

    cfg = apply_overrides(
        build_config("quick_test"),
        {
            "dataset.name": "gui_e2e",
            "dataset.output_root": tmp_path,
            "dataset.identities": 1,
            "dataset.fingers_per_identity": 1,
            "dataset.impressions_per_finger": 1,
            "dataset.tta_per_impression": 2,
            "backend.name": "dummy",
            "runtime.workers": 1,
        },
    )
    store = ConfigStore(cfg)
    controller = GenerationController(store.config)

    seen = {"samples": [], "phases": [], "logs": [], "finished": None}
    controller.sample_done.connect(seen["samples"].append)
    controller.phase.connect(seen["phases"].append)
    controller.log.connect(lambda level, msg: seen["logs"].append((level, msg)))
    controller.finished.connect(lambda stats: seen.__setitem__("finished", stats))
    controller.planned.connect(lambda summary: seen.__setitem__("planned", summary))

    assert controller.start(export_after_run=True) is True

    deadline = time.time() + 180
    while controller.is_running and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    assert controller.wait(20000), "生成线程未在超时内结束"
    qapp.processEvents()

    # --- 信号 ---
    assert seen["finished"] is not None, "finished 信号没有到达主线程"
    assert seen["samples"], "没有收到任何 sample_done 信号"
    assert seen["phases"], "没有收到 phase 信号"
    assert any(level == "INFO" for level, _ in seen["logs"]), "日志信号为空"

    stats = seen["finished"]
    assert stats.status == "finished"
    assert stats.ok == 2 and stats.failed == 0
    assert stats.total == 2

    # --- 产物 ---
    root = Path(cfg.dataset_dir)
    assert (root / "dataset_manifest.jsonl").exists()
    assert (root / "config.resolved.yaml").exists()
    assert (root / "plan.jsonl").exists()
    assert (root / "README.md").exists()
    assert (root / "stats.json").exists()
    assert list((root / "splits").glob("*.csv"))
    assert (root / "checksums" / "sha256sums.txt").exists()
    images = list((root / "images").rglob("*.png"))
    assert len(images) == 2, f"应写出 2 张图，实际 {len(images)}"


# ---------------------------------------------------------------------------
# 控件
# ---------------------------------------------------------------------------
def test_image_view_handles_empty_and_data(qapp):
    import numpy as np

    from synthfp.gui.widgets import ImageView

    view = ImageView()
    view.clear_image("空")
    view.set_image(None)
    view.zoom_in()  # 无图像时不应抛异常
    view.fit()

    image = (np.random.default_rng(0).random((60, 40)) * 255).astype(np.uint8)
    view.set_image(image)
    view.set_zoom(2.0)
    view.zoom_out()
    view.fit()


def test_thumbnail_grid_selection(qapp):
    import numpy as np

    from synthfp.gui.widgets import ThumbnailGrid

    grid = ThumbnailGrid(columns=2, thumb_size=(60, 80))
    images = [
        ((np.random.default_rng(i).random((80, 60)) * 255).astype(np.uint8), f"item{i}")
        for i in range(5)
    ]
    picked = []
    grid.selected.connect(lambda index, payload: picked.append(payload))
    grid.set_items(images)
    assert grid._grid.count() == 5
    grid.clear()
    assert grid._grid.count() == 0


def test_log_panel_levels_and_export(qapp):
    from synthfp.gui.widgets import LogPanel

    panel = LogPanel(max_lines=50)
    panel.append("INFO", "hello")
    panel.append("ERROR", "boom")
    panel.append("WARNING", "careful")
    assert panel.counts()["INFO"] == 1
    assert "boom" in panel.to_text()
    panel.clear()
    assert panel.to_text() == ""


def test_numpy_to_qimage_shapes(qapp):
    import numpy as np

    from synthfp.gui.widgets import numpy_to_qimage

    assert numpy_to_qimage(np.zeros((10, 20), dtype=np.uint8)).width() == 20
    assert numpy_to_qimage(np.zeros((10, 20, 3), dtype=np.uint8)).height() == 10
    assert numpy_to_qimage(np.zeros((10, 20, 4), dtype=np.uint8)).width() == 20
    with pytest.raises(ValueError):
        numpy_to_qimage(np.zeros((10,), dtype=np.uint8))


def test_theme_applies_and_switches(qapp):
    from synthfp.gui.theme import apply_theme, theme_names

    assert set(theme_names()) == {"dark", "light"}
    for name in theme_names():
        apply_theme(qapp, name)
        assert qapp.styleSheet(), f"{name} 主题应当生成 QSS"
    apply_theme(qapp, "dark")


# ---------------------------------------------------------------------------
# 静态约束：GUI 层不得包含生成算法；核心层不得依赖 Qt
# ---------------------------------------------------------------------------
def test_core_layers_do_not_import_qt():
    """``core`` / ``pipeline`` / ``plugins`` / ``legacy`` / ``app`` 必须能在无 Qt 环境运行。

    只检查**真正的 import 语句**（docstring 里提到 Qt 是说明性的，不算依赖）。
    """
    import re

    root = Path(__file__).resolve().parent.parent / "synthfp"
    pattern = re.compile(r"^\s*(?:from|import)\s+(PySide6|PyQt6|PyQt5)\b", re.M)
    offenders = []
    for package in ("core", "pipeline", "plugins", "legacy", "app", "config"):
        for path in (root / package).rglob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(root)))
    assert not offenders, f"核心层不应依赖 Qt: {offenders}"


def test_gui_only_imports_qt_through_compat_layer():
    """GUI 里除 ``qt.py`` 外，任何文件都不得直接 import Qt 绑定。"""
    import re

    root = Path(__file__).resolve().parent.parent / "synthfp" / "gui"
    pattern = re.compile(r"^\s*(?:from|import)\s+(PySide6|PyQt6|PyQt5)\b", re.M)
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.name != "qt.py" and pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"这些文件绕过了 Qt 兼容层: {offenders}"


def test_no_forbidden_dependencies():
    """整个包只允许 numpy / cv2 / pydantic / yaml / PySide6。"""
    forbidden = ("scipy", "skimage", "matplotlib", "pandas", "click", "tqdm")
    root = Path(__file__).resolve().parent.parent / "synthfp"
    offenders = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            for pattern in (f"import {name}", f"from {name}"):
                if pattern in text:
                    offenders.append(f"{path.relative_to(root)}: {pattern}")
    assert not offenders, f"出现了被刻意排除的依赖: {offenders}"


# ---------------------------------------------------------------------------
# 后端配置页：外观与二值开关
# ---------------------------------------------------------------------------
def test_外观灰度控件显示配置值并受二值开关控制(qapp):
    """四个灰度框必须回填真实值；勾选二值后置灰。

    回归用例：``backend.ridge_gray.0`` 这类元组下标路径以前既不能读也不能写 ——
    控件全显示 0，改动会校验失败并弹回。
    """
    from synthfp.config import apply_overrides, build_config
    from synthfp.gui.config_store import ConfigStore
    from synthfp.gui.pages.backend_page import BackendPage

    cfg = apply_overrides(build_config("quick_test"), {"backend.ridge_gray": (30, 80)})
    store = ConfigStore(cfg)
    page = BackendPage(store)

    values = [w.value() for w in page._color_widgets]
    assert values == [30, 80, cfg.backend.valley_gray[0], cfg.backend.valley_gray[1]]
    assert all(w.isEnabled() for w in page._color_widgets)

    # 勾选二值 → 灰度框置灰（二值模式下它们不起作用）
    page.ridge_only_black.setChecked(True)
    qapp.processEvents()
    assert store.config.backend.ridge_only_black is True
    assert not any(w.isEnabled() for w in page._color_widgets)

    page.ridge_only_black.setChecked(False)
    qapp.processEvents()
    assert store.config.backend.ridge_only_black is False
    assert all(w.isEnabled() for w in page._color_widgets)


def test_改灰度下界会写回配置(qapp):
    from synthfp.config import build_config
    from synthfp.gui.config_store import ConfigStore
    from synthfp.gui.pages.backend_page import BackendPage

    store = ConfigStore(build_config("quick_test"))
    page = BackendPage(store)
    page._color_widgets[0].setValue(15)
    qapp.processEvents()
    assert store.config.backend.ridge_gray[0] == 15


def test_窗口标题带SLIM_FRNet前缀(qapp):
    from synthfp.config import build_config
    from synthfp.gui.main_window import APP_TITLE, MainWindow

    window = MainWindow(build_config("quick_test"))
    assert window.windowTitle().startswith("SLIM-FRNet")
    assert "SLIM-FRNet" in APP_TITLE
    window.close()
