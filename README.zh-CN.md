# Vision Workbench

<p align="center">
  <img src="./docs/assets/readme/vision_workbench_hero.png" alt="Vision Workbench 项目总览" width="100%">
</p>

<p align="center">
  <a href="./README.md">English</a>
  ·
  <a href="./docs/adding_custom_features_README.zh-CN.md">二次开发指南</a>
  ·
  <a href="./docs/legal/release_policy.zh-CN.md">发布策略</a>
  ·
  <a href="./THIRD_PARTY_NOTICES.md">第三方引用说明</a>
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-AGPL--3.0-0f766e">
  <img alt="Python" src="https://img.shields.io/badge/python-3.8%2B-2563eb">
  <img alt="GUI" src="https://img.shields.io/badge/GUI-Tkinter-f59e0b">
  <img alt="Status" src="https://img.shields.io/badge/status-learning%20workbench-16a34a">
</p>

Vision Workbench 是一个本地计算机视觉学习工作台。项目把传统图像处理、全景重构、相机诊断、图像分类、YOLO26 目标检测、YOLO26 分割和 YOLO26 训练放在同一个新手友好的工程里。

它不是单个脚本集合，而是一个完整学习链路：对外 Python API、桌面 GUI、模型与数据集目录、自动化测试、打包配置、开源许可文件和第三方引用说明都已经放在工程中。

## 项目生态

<p align="center">
  <img src="./docs/assets/readme/ecosystem.svg" alt="Vision Workbench 生态链示意图" width="100%">
</p>

## 功能模块

| 模块            | 功能定位                                                           | 文档                                                                  |
| --------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| 基础 CV         | OpenCV 基础图像处理、色彩空间、通道分离、直方图、形态学与几何变换  | [README](./docs/modules/zh-CN/cv_basics_README.zh-CN.md)               |
| 全景重构        | 左右图像重构、SIFT 匹配、人工点选、辅助点选和全景输出              | [README](./docs/modules/zh-CN/panorama_reconstruction_README.zh-CN.md) |
| 相机诊断        | 摄像头检测、读取模式测试、实时预览、FPS、截图与录屏                | [README](./docs/modules/zh-CN/camera_diagnostics_README.zh-CN.md)      |
| 图像分类        | ResNet18、MobileNetV3 Small 预测、预训练权重、数据集校验和基础训练 | [README](./docs/modules/zh-CN/image_classification_README.zh-CN.md)    |
| YOLO26 目标检测 | YOLO26 检测模型加载、摄像头实时推理、截图和录屏                    | [README](./docs/modules/zh-CN/yolo26_detection_README.zh-CN.md)        |
| YOLO26 分割     | YOLO26 实例分割和语义分割，支持图片和摄像头输入                    | [README](./docs/modules/zh-CN/yolo26_segmentation_README.zh-CN.md)     |
| YOLO26 训练     | 检测、实例分割、语义分割训练入口与数据集校验                       | [README](./docs/modules/zh-CN/yolo26_training_README.zh-CN.md)         |

## 快速开始

基础环境，适用于基础 CV、全景重构和相机诊断：

```bash
conda create -n vision-workbench python=3.11 -y
conda activate vision-workbench
cd path/to/vision-workbench
pip install -e .
vision-workbench
```

图像分类：

```bash
python scripts/install_dependencies.py classification
image-classification-workbench
```

YOLO26 相关功能：

```bash
python scripts/install_dependencies.py yolo26
yolo26-detection-workbench
```

从 `vision-workbench` 主界面打开的可选 GUI 模块会作为独立子进程运行。关闭子窗口只释放该模块自己的资源，包括摄像头句柄或 CUDA 显存，不会关闭主界面；关闭主界面时会结束它启动过的子模块进程。

## 问题排查

如果安装命令、GUI 弹窗、摄像头流程、模型加载、数据集校验或训练运行失败，请先查看 [问题排查指南](./docs/troubleshooting/zh-CN/README.zh-CN.md)。用户可见的错误提示也会附带对应的排查文档路径。

## 发布包说明

本仓库只保存源码、文档、测试、配置和许可文件。`dist/` 目录属于本地构建产物，不建议提交到源码仓库。

正式版本建议通过 [GitHub Releases](https://github.com/ksukie/vision-workbench/releases) 发布。只想安装基础打包版本的用户，可以在 Release 页面下载 `.whl` 文件，然后本地安装：

```bash
pip install vision_workbench-0.1.0-py3-none-any.whl
vision-workbench
```

wheel 会安装 Python 包和命令行入口，但它不是完整的离线运行环境：深度学习相关功能仍然需要额外安装对应依赖组，大型模型权重也会单独分发。

如果用户希望基于源码自己构建 wheel，可以执行：

```bash
conda activate vision-workbench
pip install build
python -m build
```

如果隔离构建环境无法访问 PyPI，可以改用当前环境构建：

```bash
python -m build --no-isolation
```

## 项目结构

```text
VisionWorkbench/
  src/                         源码目录
  docs/                        项目文档
  docs/assets/readme/          README 图片和图标
  models/                      模型权重目录
  datasets/                    数据集目录
  runs/                        训练输出目录
  third_party/yolo26_source/   内置 YOLO26 源码
  tests/                       自动化测试
```

## 依赖策略

基础依赖保持轻量，只包含 NumPy、OpenCV 和 Pillow。

深度学习能力按需安装：

- 图像分类：`python scripts/install_dependencies.py classification`
- YOLO26 检测、分割与训练：`python scripts/install_dependencies.py yolo26`

安装脚本会检测 NVIDIA GPU，有 NVIDIA GPU 时安装 CUDA 12.6 Torch wheel，否则选择 CPU 或平台默认 Torch 构建。直接执行 `pip install -r requirements-*.txt` 无法自行检测 GPU，因此新环境优先使用安装脚本。基础包仍然默认使用清华 PyPI 源。

如果手动使用 `requirements-*.txt` 安装深度学习依赖，安装后建议再执行一次检测：

```bash
python scripts/install_dependencies.py doctor
```

它会检查 Torch 是否匹配当前机器，并在需要重装 Torch 时先询问用户。

这样可以保证新手先用基础功能跑通项目，再根据需要启用更重的深度学习功能。

## 开源许可

Vision Workbench 是面向学习和研究的开源项目，采用 AGPL-3.0 许可发布。详见 [LICENSE](./LICENSE)。

本项目在 `third_party/yolo26_source/` 内置了 Ultralytics YOLO26 源码。Vision Workbench 不是 Ultralytics 官方项目。YOLO26 源码和 YOLO26 模型权重仍然遵守 Ultralytics 原始许可条款。详见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。
