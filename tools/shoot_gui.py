"""把 GUI 的 8 个页面渲染成 PNG（离屏，无需显示器）。

用途有两个：

* **交付证据**：在没有图形会话的环境（CI、远程 shell）里证明界面能正常构建与绘制；
* **回归对比**：改动界面后重新生成，肉眼比对是否有布局塌陷。

用法::

    python tools/shoot_gui.py                     # 输出到 outputs/gui-shots/
    python tools/shoot_gui.py --out docs/shots    # 自定义目录
    python tools/shoot_gui.py --keep-config       # 用默认配置而不是 quick_test

依赖 PySide6 或 PyQt6（与主程序一致）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# 必须在导入 Qt 之前设置，否则没有显示器时会直接失败
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# 离屏平台的 Qt 找不到自带字体目录时，所有文字都会画成"豆腐块"。
# 指向系统的 Fonts 目录即可正常出字（只影响本脚本的截图，不影响真实窗口）。
if sys.platform == "win32":
    _fontdir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    if _fontdir.is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", str(_fontdir))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=_ROOT / "outputs" / "gui-shots")
    parser.add_argument("--size", default="1440x900", help="窗口尺寸，例如 1440x900")
    parser.add_argument("--theme", default="dark", choices=["dark", "light"])
    parser.add_argument("--keep-config", action="store_true", help="用默认配置而非 quick_test")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="把「结果预览」「统计报告」指向这个已有数据集，截图才有内容",
    )
    parser.add_argument(
        "--full-page",
        action="store_true",
        help="按整页高度截图（会滚动到视野外的分组也能拍到），而不是只拍窗口可见区域",
    )
    args = parser.parse_args()

    from synthfp.config import build_config, load_default_config
    from synthfp.gui.main_window import MainWindow
    from synthfp.gui.qt import QT_BINDING, QApplication
    from synthfp.gui.theme import apply_theme

    width, height = (int(x) for x in args.size.lower().split("x"))

    app = QApplication.instance() or QApplication([])
    apply_theme(app, args.theme)

    config = load_default_config() if args.keep_config else build_config("quick_test")
    window = MainWindow(config)
    window.resize(width, height)
    window.show()
    app.processEvents()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 让「结果预览」「统计报告」「导出」指向真实数据集，否则这几页截出来是空的
    if args.dataset is not None:
        from synthfp.gui.pages import ExportPage, PreviewPage, StatsPage

        dataset = args.dataset
        for page in window.pages:
            if isinstance(page, PreviewPage):
                page.dir_edit.setText(str(dataset))
                page._scan_dataset()
                page.refresh()
            elif isinstance(page, StatsPage):
                page.dir_edit.setText(str(dataset))
                page.reload()
            elif isinstance(page, ExportPage):
                page.dir_edit.setText(str(dataset))
                page.refresh_view()
        app.processEvents()
        print(f"已载入数据集: {dataset}")

    print(f"Qt 绑定: {QT_BINDING}    窗口: {width}×{height}    主题: {args.theme}")
    written = []
    for index, page in enumerate(window.pages):
        window.nav.setCurrentRow(index)
        app.processEvents()
        # 再跑一次事件循环，让布局与样式表完成计算
        app.processEvents()

        content = getattr(page, "_content", None)
        saved_size = page.size()
        full_page = args.full_page and content is not None and content is not page
        if full_page:
            # 整页截图：把滚动内容撑到完整高度后再抓，视野外的分组也能拍到。
            # 直接 grab() 视野外的子控件会得到空白（Qt 从没绘制过它），
            # 所以必须先把页面撑开、让布局重新算一遍。
            page.resize(saved_size.width(), max(content.sizeHint().height(), saved_size.height()))
            page._layout.activate()
            app.processEvents()
            source = content
        else:
            source = window

        target = out_dir / f"{index + 1:02d}_{page.title}.png"
        pixmap = source.grab()
        ok = pixmap.save(str(target), "PNG")
        if full_page:
            page.resize(saved_size)
            app.processEvents()
        if not ok:
            print(f"✗ 写入失败: {target}")
            return 1
        written.append(target)
        print(f"  ✓ {target.name}  {pixmap.width()}×{pixmap.height()}")

    window.close()
    print(f"\n共 {len(written)} 张，输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
