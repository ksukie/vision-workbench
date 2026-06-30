# 图像分类 README

[English](../en/image_classification_README.md) | [返回总 README](../../../README.zh-CN.md) | [二次开发入口](../../adding_custom_features_README.zh-CN.md#image-classification-add)

## 概述

图像分类模块用于判断整张图像所属类别。模块提供 ResNet18 与 MobileNetV3 Small 两个 backbone，支持官方预训练权重预测、离线权重导入、自定义分类数据集训练和 Top-K 预测结果展示。

## 功能范围

| 功能 | 说明 |
| --- | --- |
| 预训练预测 | 使用 ResNet18 或 MobileNetV3 Small 的 ImageNet 预训练权重进行单图预测 |
| 权重管理 | 检测、下载、离线导入预训练权重 |
| 数据集检查 | 校验 `train/val/class_name` 文件夹分类数据集 |
| 数据集划分 | 将原始类别文件夹划分为 `train/val` |
| 模型训练 | 基于预训练权重进行迁移学习 |
| 结果展示 | GUI 展示 Top-K 类别、置信度和推理耗时 |

## 模型

| 模型 | 定位 |
| --- | --- |
| `resnet18` | 经典 CNN backbone，适用于教学和基线实验 |
| `mobilenet_v3_small` | 轻量分类 backbone，适用于低资源环境和快速演示 |

## 环境要求

```bash
conda create -n vision-classification python=3.11 -y
conda activate vision-classification
cd path/to/vision-workbench
pip install -e .
python scripts/install_dependencies.py classification
```

## 启动方式

```bash
image-classification-workbench
```

主界面入口：

```bash
vision-workbench
```

源码方式：

```bash
python .\src\image_classification\window\app.py
```

## 数据集格式

分类数据集使用文件夹名作为类别名：

```text
dataset/
  train/
    cat/
      001.jpg
    dog/
      001.jpg
  val/
    cat/
      101.jpg
    dog/
      101.jpg
```

校验内容：

- `train/` 与 `val/` 是否存在。
- 训练集和验证集类别是否一致。
- 每个类别是否包含图片。
- 图片文件是否可读取。
- 每个类别训练样本数量是否过少。

## 预训练权重

预训练权重目录：

```text
models/image_classification_models/pretrained/
```

当前支持文件：

```text
resnet18-f37072fd.pth
mobilenet_v3_small-047dcff4.pth
```

GUI 权重管理按钮：

| 按钮 | 功能 |
| --- | --- |
| `Check Weights` | 检查当前 backbone 的本地权重状态 |
| `Download Pretrained Weights` | 下载当前 backbone 的 TorchVision 官方权重 |
| `Import Local Weights` | 导入离线 `.pth` 或 `.pt` 权重文件 |

预测和训练优先使用项目目录中的本地权重。若本地权重不存在，TorchVision 会按默认机制尝试下载。

TorchVision 默认缓存位置：

```text
Windows: C:\Users\用户名\.cache\torch\hub\checkpoints
Linux/macOS: ~/.cache/torch/hub/checkpoints
```

## 操作流程：预测

1. 打开 `Predict` 页签。
2. 点击 `Open Image` 选择图片。
3. 在 `Model` 中选择 backbone。
4. 点击 `Check Weights` 查看权重状态。
5. 根据需要执行下载或离线导入。
6. 点击 `Predict Pretrained` 执行预训练预测。
7. 查看 Top-K 预测结果。

## 操作流程：训练

1. 打开 `Train` 页签。
2. 选择分类数据集根目录。
3. 点击 `Validate` 校验数据集。
4. 选择 backbone、设备、训练轮数、图像尺寸和 batch size。
5. 根据训练需求选择是否使用预训练权重和是否冻结 backbone。
6. 点击 `Start Training` 开始训练。
7. 在 `runs/image_classification/` 查看训练输出。

## 命令行训练

```bash
image-classification-train ^
  --data C:\path\to\classification_dataset ^
  --model resnet18 ^
  --epochs 5 ^
  --imgsz 224 ^
  --batch 16 ^
  --device auto
```

仅执行数据集校验：

```bash
image-classification-train --data C:\path\to\classification_dataset --model mobilenet_v3_small --dry-run
```

## Python API

```python
from image_classification.api import (
    validate_classification_dataset,
    predict_with_pretrained,
)

report = validate_classification_dataset("C:/path/to/dataset")
result = predict_with_pretrained("resnet18", "C:/path/to/image.jpg", topk=5)
```

常用函数：

```text
supported_models()
validate_classification_dataset(dataset_dir)
split_classification_dataset(input_dir, output_dir, train_ratio=0.8)
pretrained_weight_status(model_name=None)
download_pretrained_weight(model_name)
import_pretrained_weight(model_name, source_path)
list_classification_models()
train_classifier(config)
load_pretrained_classifier(model_name)
load_classifier(model_path)
predict_with_pretrained(model_name, image_path, topk=5)
predict_image(model_path, image_path, topk=5)
```

## 目录结构

```text
src/image_classification/api/             对外 API
src/image_classification/application/     分类业务流程
src/image_classification/configuration/   路径和默认参数
src/image_classification/domain/          数据模型
src/image_classification/infrastructure/  数据集检查、划分、模型扫描、权重管理
src/image_classification/processing/      TorchVision 训练和预测后端
src/image_classification/window/          Tkinter GUI
```

## 二次开发

新增 backbone、预训练权重规则、数据增强或导出逻辑见：[添加自定义功能 README - 图像分类扩展](../../adding_custom_features_README.zh-CN.md#image-classification-add)。
