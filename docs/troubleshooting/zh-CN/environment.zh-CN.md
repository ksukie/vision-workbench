# 环境问题排查

[索引](./README.zh-CN.md) | [English](../en/environment.md)

本页覆盖 Python、conda、editable install、命令行入口、Tkinter 启动、路径编码和基础依赖安装问题。

## 命令找不到

常见表现：

- `vision-workbench` 不是可识别命令。
- `image-classification-workbench` 或 `yolo26-detection-workbench` 找不到。

先检查：

```bash
conda activate vision-workbench
python -m pip install -e .
python -m pip show vision-workbench
```

如果入口命令仍然找不到，可以先直接运行模块：

```bash
python -m cv_basics.window.app
```

## 导入失败

常见表现：

- `ModuleNotFoundError`
- 从仓库运行 GUI 时无法导入本项目包。

检查：

```bash
cd path/to/vision-workbench
python -c "import cv_basics, vision_workbench; print('ok')"
python -m pip install -e .
```

除非脚本明确支持，否则不要从 `src/` 子目录内部启动项目。

## Tkinter 启动问题

常见表现：

- GUI 不打开。
- `_tkinter` 导入失败。

请使用带 Tkinter 的 Python。Windows 上 conda Python 通常自带 Tkinter。若当前 Python 是精简构建，建议重新创建环境：

```bash
conda create -n vision-workbench python=3.11 -y
conda activate vision-workbench
python -m pip install -e .
```

## 路径和编码

排查文件问题时优先使用绝对路径。如果路径里有中文、空格或特殊符号，先把项目或数据复制到较短的纯英文路径，例如 `C:\work\vision-workbench`，确认问题是否与路径解析有关。

## 基础依赖安装失败

执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果镜像源临时不可用，可改用 PyPI 官方源重试：

```bash
python -m pip install -e . --index-url https://pypi.org/simple
```
