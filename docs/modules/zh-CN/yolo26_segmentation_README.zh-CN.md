# YOLO26 分割 README

[English](../en/yolo26_segmentation_README.md) | [返回总 README](../../../README.zh-CN.md) | [二次开发入口](../../adding_custom_features_README.zh-CN.md#yolo26-segmentation-add)

## 概述

YOLO26 分割模块提供实例分割和语义分割功能，支持图片输入和摄像头输入。模块用于显示 mask 可视化结果，并支持保存当前分割结果图。

## 安装说明

请先按根目录 [快速开始](../../../README.zh-CN.md#快速开始) 统一配置项目环境，然后只需安装一次 YOLO26 依赖组：

```bash
python scripts/install_dependencies.py yolo26
```

如果手动使用 `requirements-yolo26.txt` 安装依赖，安装后建议执行 `python scripts/install_dependencies.py doctor` 检查 Torch 构建是否匹配当前机器。

## 启动方式

```bash
yolo26-segmentation-workbench
```

源码方式：

```bash
python -m yolo26_segmentation.window.app
```

## 模型目录

```text
third_party/yolo26_source/                 YOLO26 源码
models/yolo26_segmentation_models/         YOLO26 分割模型
models/yolo26_segmentation_models/custom/  自定义分割模型
```

模型命名：

```text
*-seg.pt  实例分割模型
*-sem.pt  语义分割模型
```

## 任务类型

| 任务         | 说明                       |
| ------------ | -------------------------- |
| `segment`  | 实例分割，输出对象级 mask  |
| `semantic` | 语义分割，输出像素类别区域 |

## 操作流程

1. 在 `Task` 中选择 `segment` 或 `semantic`。
2. 在 `Model` 中选择匹配任务的模型。
3. 如需自定义模型，点击 `Browse PT`。
4. 点击 `Open Image` 选择图片，或选择摄像头。
5. 点击 `Run Once` 执行单次分割。
6. 点击 `Start Live` 执行实时分割。
7. 点击 `Save Result` 保存当前分割结果。
8. 点击 `Stop` 停止实时处理。

## Python API

```python
from yolo26_segmentation.api import list_models, segment_image

models = list_models(task="segment")
result = segment_image("input.jpg", model_path=models[0].path)
```

## 目录结构

```text
src/yolo26_segmentation/api/             对外 API
src/yolo26_segmentation/application/     分割业务流程
src/yolo26_segmentation/configuration/   默认参数和模型目录
src/yolo26_segmentation/domain/          数据模型
src/yolo26_segmentation/infrastructure/  模型扫描、图片读写、YOLO26 后端
src/yolo26_segmentation/processing/      分割处理器
src/yolo26_segmentation/window/          Tkinter GUI
```

## 二次开发

新增分割模型、mask 保存策略、可视化方式或批量处理流程见：[添加自定义功能 README - YOLO26 分割扩展](../../adding_custom_features_README.zh-CN.md#yolo26-segmentation-add)。
