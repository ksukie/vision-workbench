# YOLO26 目标检测 README

[English](../en/yolo26_detection_README.md) | [返回总 README](../../../README.zh-CN.md) | [二次开发入口](../../adding_custom_features_README.zh-CN.md#yolo26-detection-add)

## 概述

YOLO26 目标检测模块用于加载 YOLO26 检测模型，并通过摄像头进行实时目标检测。模块提供摄像头选择、模型选择、推理参数设置、截图和录屏功能。

## 安装说明

请先按根目录 [快速开始](../../../README.zh-CN.md#快速开始) 统一配置项目环境，然后只需安装一次 YOLO26 依赖组：

```bash
python scripts/install_dependencies.py yolo26
```

如果手动使用 `requirements-yolo26.txt` 安装依赖，安装后建议执行 `python scripts/install_dependencies.py doctor` 检查 Torch 构建是否匹配当前机器。

## 启动方式

```bash
yolo26-detection-workbench
```

源码方式：

```bash
python -m yolo26_detection.window.app
```

## 模型目录

```text
third_party/yolo26_source/   YOLO26 源码
models/yolo26_models/        YOLO26 检测模型
```

支持模型文件：

```text
yolo26n.pt
yolo26s.pt
yolo26m.pt
yolo26l.pt
yolo26x.pt
```

自定义 `.pt` 模型可放入 `models/yolo26_models/`，也可通过界面选择。

## 操作流程

1. 点击 `Refresh Cameras` 扫描摄像头。
2. 在 `Camera` 中选择设备。
3. 在 `Model` 中选择检测模型。
4. 如需自定义模型，点击 `Browse PT`。
5. 设置 `Device`、`Image size`、`Conf` 和 `IoU`。
6. 点击 `Open` 打开摄像头。
7. 点击 `Start Detection` 开始检测。
8. 点击 `Screenshot` 保存当前帧。
9. 点击 `Start Recording` 和 `Stop Recording` 录制检测视频。

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `Device` | 推理设备，支持 `auto`、`cpu`、`cuda`、`mps` 等 |
| `Image size` | 模型输入尺寸 |
| `Conf` | 置信度阈值 |
| `IoU` | NMS IoU 阈值 |

## Python API

```python
from yolo26_detection.api import list_models, load_model, detect_objects

models = list_models()
detector = load_model(models[0].path)
result = detect_objects(detector, image)
```

## 目录结构

```text
src/yolo26_detection/api/             对外 API
src/yolo26_detection/application/     检测业务流程
src/yolo26_detection/configuration/   默认参数和模型目录
src/yolo26_detection/domain/          数据模型
src/yolo26_detection/infrastructure/  模型扫描、摄像头、YOLO26 后端
src/yolo26_detection/processing/      检测处理器
src/yolo26_detection/window/          Tkinter GUI
```

## 二次开发

新增检测模型、推理参数、类别过滤或后处理逻辑见：[添加自定义功能 README - YOLO26 目标检测扩展](../../adding_custom_features_README.zh-CN.md#yolo26-detection-add)。
