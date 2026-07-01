# YOLO26 训练 README

[English](../en/yolo26_training_README.md) | [返回总 README](../../../README.zh-CN.md) | [二次开发入口](../../adding_custom_features_README.zh-CN.md#yolo26-training-add)

## 概述

YOLO26 训练模块提供检测、实例分割和语义分割三类任务的基础训练入口。模块在启动训练前执行数据集格式校验，校验失败时终止训练并返回错误信息。

## 安装说明

请先按根目录 [快速开始](../../../README.zh-CN.md#快速开始) 统一配置项目环境，然后只需安装一次 YOLO26 依赖组：

```bash
python scripts/install_dependencies.py yolo26
```

如果手动使用 `requirements-yolo26.txt` 安装依赖，安装后建议执行 `python scripts/install_dependencies.py doctor` 检查 Torch 构建是否匹配当前机器。GPU 训练需要 CUDA 版 PyTorch；CPU 可运行训练流程，但训练速度较慢。

## 启动方式

GUI：

```bash
yolo26-training-workbench
```

命令行：

```bash
yolo26-train --task detect --data path/to/dataset/data.yaml --model models/yolo26_models/yolo26n.pt
```

基础脚本：

```bash
python -m yolo26_training.train
```

## 任务类型

| 任务 | 标签格式 |
| --- | --- |
| `detect` | `class x_center y_center width height` |
| `segment` | `class x1 y1 x2 y2 x3 y3 ...` |
| `semantic` | PNG/TIF mask，或 polygon 标签 |

数据集需提供 `data.yaml`，并保证图片路径、类别数量和类别名称与实际数据一致。

## GUI 操作流程

1. 在 `Task` 中选择 `detect`、`segment` 或 `semantic`。
2. 选择数据集 `data.yaml`。
3. 选择初始模型。
4. 设置 `Epochs`、`Image size`、`Batch size`、`Device` 等参数。
5. 执行数据集校验。
6. 校验通过后开始训练。
7. 在日志区域查看训练输出。
8. 在 `runs/yolo26_training/` 查看训练结果。

## 命令行训练

```bash
yolo26-train --task detect --data path/to/dataset/data.yaml --model path/to/model.pt --epochs 100 --imgsz 640 --batch 16 --device auto
```

仅校验数据集：

```bash
yolo26-train --task detect --data path/to/data.yaml --model path/to/model.pt --dry-run
```

## 基础脚本

可修改以下文件顶部变量：

```text
src/yolo26_training/train.py
```

运行：

```bash
python -m yolo26_training.train
```

## 输出目录

```text
runs/yolo26_training/
```

训练输出内容由 YOLO26 源码生成，通常包含权重文件、配置文件、日志和指标图。

## Python API

```python
from yolo26_training.api import validate_dataset, list_models

report = validate_dataset("path/to/dataset/data.yaml", task="detect")
models = list_models(task="detect")
```

## 目录结构

```text
src/yolo26_training/api/             对外 API
src/yolo26_training/application/     训练业务流程
src/yolo26_training/configuration/   默认参数和路径
src/yolo26_training/domain/          数据模型
src/yolo26_training/infrastructure/  数据集校验、模型扫描、YOLO26 训练后端
src/yolo26_training/runner.py        命令行入口
src/yolo26_training/train.py         基础训练脚本
src/yolo26_training/window/          Tkinter GUI
```

## 二次开发

新增训练任务、参数、数据集校验规则或日志保存策略见：[添加自定义功能 README - YOLO26 训练扩展](../../adding_custom_features_README.zh-CN.md#yolo26-training-add)。
