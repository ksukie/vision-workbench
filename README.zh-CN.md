# Vision Workbench

<p align="center">
  <img src="./docs/assets/readme/vision_workbench_hero.png" alt="Vision Workbench 项目总览" width="100%">
</p>

<p align="center">
  <a href="./README.md">English</a>
  ·
  <a href="./docs/二次开发指南.md">二次开发指南</a>
  ·
  <a href="./docs/legal/发布策略.md">发布策略</a>
  ·
  <a href="./SECURITY.md">安全策略</a>
  ·
  <a href="./CHANGELOG.md">更新日志</a>
  ·
  <a href="./THIRD_PARTY_NOTICES.md">第三方引用说明</a>
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-AGPL--3.0-0f766e">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-2563eb">
  <img alt="GUI" src="https://img.shields.io/badge/GUI-PySide6-0a84ff">
  <img alt="Status" src="https://img.shields.io/badge/status-learning%20workbench-16a34a">
</p>

Vision Workbench 是一个本地计算机视觉学习工作台，当前以统一 PySide6 / Qt 桌面界面为准。项目把传统图像处理、全景重构、相机诊断、图像分类、YOLO26 目标检测、YOLO26 分割和 YOLO26 训练统一放在 `vision-workbench` 主入口中。

它不是零散脚本集合，而是一条完整学习链路：对外 Python API、Qt 桌面主界面、模型与数据集目录、自动化测试、打包配置、开源许可证和第三方引用说明都已经放在工程里。

## 项目生态

<p align="center">
  <img src="./docs/assets/readme/ecosystem.svg" alt="Vision Workbench 生态链示意图" width="100%">
</p>

## 功能模块

| 模块 | 功能定位 | 文档 |
| --- | --- | --- |
| 基础 CV | OpenCV 基础图像处理、色彩空间、通道分离、直方图、形态学与几何变换 | [README](./docs/modules/zh-CN/基础CV.md) |
| 全景重构 | 左右图像拼接、SIFT 匹配、人工点选、辅助点选和全景输出 | [README](./docs/modules/zh-CN/全景重构.md) |
| 相机诊断 | 摄像头检测、读取模式测试、实时预览、FPS、截图与录屏 | [README](./docs/modules/zh-CN/相机诊断.md) |
| 图像分类 | ResNet18、MobileNetV3 Small 预测、预训练权重、数据集校验和基础训练 | [README](./docs/modules/zh-CN/图像分类.md) |
| YOLO26 目标检测 | YOLO26 模型加载、图片检测、摄像头实时推理、截图和下载 | [README](./docs/modules/zh-CN/YOLO26目标检测.md) |
| YOLO26 分割 | YOLO26 实例分割和语义分割，支持图片输入 | [README](./docs/modules/zh-CN/YOLO26分割.md) |
| YOLO26 训练 | 检测、实例分割、语义分割训练入口与数据集校验 | [README](./docs/modules/zh-CN/YOLO26训练.md) |

## 快速开始

环境要求：

- Python 3.10 或更新版本，默认推荐 Python 3.11。
- Conda 或其它隔离 Python 环境。
- 本地项目源码目录。

创建基础环境，使用 editable 模式安装项目，然后启动桌面应用：

```bash
conda create -n vision-workbench python=3.11 -y
conda activate vision-workbench
cd path/to/vision-workbench
pip install -e .
vision-workbench
```

首次启动会打开统一 Qt 桌面主界面。左侧导航直接进入基础 CV、全景重构、相机诊断、YOLO 检测、YOLO 分割、模型训练和图像分类。

图像分类预测或训练需要先安装分类依赖组：

```bash
python scripts/install_dependencies.py classification
vision-workbench
```

YOLO26 检测、分割和训练需要先安装 YOLO26 依赖组：

```bash
python scripts/install_dependencies.py yolo26
vision-workbench
```

可选依赖安装完成后执行依赖诊断：

```bash
python scripts/install_dependencies.py doctor
```

基础安装保持轻量，深度学习能力单独启用，便于先打开桌面界面并跑通基础功能。

## 问题排查

安装命令失败、GUI 弹窗报错、摄像头流程异常、模型加载失败、数据集校验失败和训练失败统一从 [问题排查指南](./docs/troubleshooting/zh-CN/问题排查指南.md) 开始定位。用户可见错误提示也会附带对应的排查文档路径。

异常退出后残留的 GUI、摄像头或训练进程可用 `python scripts/cleanup_runtime.py` 列出。确认 dry-run 列表后，使用 `python scripts/cleanup_runtime.py --kill` 清理匹配的项目进程。

## 安全与版本记录

疑似安全漏洞通过 [SECURITY.md](./SECURITY.md) 中的私密渠道报告。公开 issue 中不发布利用细节或敏感环境信息。

版本变动维护在 [CHANGELOG.md](./CHANGELOG.md)。

## 发布包说明

本仓库只保存源码、文档、测试、配置和许可文件。`dist/` 等本地构建产物不提交到源码仓库。

正式版本建议通过 [GitHub Releases](https://github.com/ksukie/vision-workbench/releases) 发布。基础打包版本可通过 wheel 文件本地安装：

```bash
pip install vision_workbench-0.2.0-py3-none-any.whl
vision-workbench
```

wheel 会安装 Python 包和入口命令，但它不是完整离线运行环境：深度学习相关功能仍需要额外安装对应依赖组，大型模型权重也可能单独分发。

从源码构建：

```bash
conda activate vision-workbench
pip install build
python -m build
```

隔离构建环境无法访问 PyPI 时，改用当前环境构建：

```bash
python -m build --no-isolation
```

## 项目结构

```text
VisionWorkbench/
  src/                         源码目录
  src/vision_workbench/desktop/ 统一 PySide6 桌面主界面和页面
  docs/                        项目文档
  docs/assets/readme/          README 图片和图标
  models/                      模型权重目录
  datasets/                    数据集目录
  runs/                        训练输出目录
  third_party/yolo26_source/   内置 YOLO26 源码
  tests/                       自动化测试
```

## 依赖策略

基础依赖覆盖 Qt 桌面主界面和传统 CV 功能：PySide6、NumPy、OpenCV 和 Pillow。

深度学习能力按需安装：

- 图像分类：`python scripts/install_dependencies.py classification`
- YOLO26 检测、分割与训练：`python scripts/install_dependencies.py yolo26`

安装脚本会检测 NVIDIA GPU；有 NVIDIA GPU 时安装 CUDA 12.6 Torch wheel，否则选择 CPU 或平台默认 Torch 构建。直接执行 `pip install -r requirements-*.txt` 无法自动检测 GPU，因此新环境优先使用安装脚本。基础包仍默认使用清华 PyPI 源。

手动使用 `requirements-*.txt` 安装深度学习依赖后执行：

```bash
python scripts/install_dependencies.py doctor
```

这一路径保证基础功能先完成验证，再启用更重的深度学习功能。

## 架构说明

各功能包把业务逻辑放在 `api/`、`application/`、`domain/`、`infrastructure/` 和 `processing/` 中。用户直接使用的桌面体验位于 `src/vision_workbench/desktop/`。旧 `*/window/` Tkinter 模块暂时保留为兼容/参考代码，不再作为公开 GUI 入口。

## 开源许可

Vision Workbench 是面向学习和研究的开源项目，采用 AGPL-3.0 许可发布。详见 [LICENSE](./LICENSE)。

本项目在 `third_party/yolo26_source/` 内置了 Ultralytics YOLO26 源码。Vision Workbench 不是 Ultralytics 官方项目。YOLO26 源码和 YOLO26 模型权重仍然遵守 Ultralytics 原始许可条款。详见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。

## 致谢

2026.07.01 - 感谢 [@antique798](https://github.com/antique798) 协助测试 Vision Workbench 并反馈问题。
