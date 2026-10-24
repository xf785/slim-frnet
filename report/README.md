# 结项研究报告（LaTeX）

本目录是《SLIM-FRNet：轻量级可复现合成指纹数据集生成平台的设计、实现与评测》
一文的 LaTeX 源与实验数据。渲染结果：[main.pdf](main.pdf)（14 页）。

## 编译

需要 TeX Live 2022+（含 `ctex`、`gbt7714`、`tikz`）与系统中文字体。

```bash
cd report
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex          # 条目/引用稳定
```

或一条命令：`latexmk -xelatex main.tex`

## 目录结构

| 文件 | 说明 |
|------|------|
| `main.tex` | 正文（摘要、7 节、附录 A/B/C） |
| `refs.bib` | 参考文献（17 条，GB/T 7714 数字制） |
| `figures/*.pdf` | 由 `_figures.py` 生成的矢量插图 |
| `data/stats.json` | 性能、质量、细节点、可复现性等实验数据 |
| `data/ridge.json` | 脊线周期与占空比（两种估计方法） |
| `_collect.py` | 运行正式实验并写出 `data/stats.json` |
| `_ridge.py` | 测量脊线几何，写出 `data/ridge.json` |
| `_figures.py` | 由上述 JSON 生成插图 |

## 复现实验数据

```bash
# 1) 跑正式实验（约 40 秒，生成 3 个 72 样本数据集）
python report/_collect.py
# 2) 测量脊线几何（约 20 秒）
python report/_ridge.py
# 3) 重画插图
python report/_figures.py
```

`_collect.py` 产出的数据集在 `outputs/_report_exp/`（已被 `.gitignore` 忽略），
文中所有数字都直接来自这两个 JSON，未手工填写。

## 注意

- 数据集**名称参与种子派生**（dataset → identity → finger → impression → tta），
  因此"改名重跑"会得到完全不同的数据。验证可复现性时必须保持同名、只改输出目录。
- 插图使用系统字体（微软雅黑 / SimHei）；换机器若中文缺字，请改
  `_figures.py` 顶部的 `font.sans-serif` 列表。
