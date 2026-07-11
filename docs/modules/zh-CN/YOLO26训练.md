# YOLO26 训练 README

[返回 README](../../../README.zh-CN.md) | [English](../en/yolo26_training_README.md) | [二次开发指南](../../二次开发指南.md#yolo26-训练扩展)

## 概览

YOLO26 训练模块提供检测、实例分割和语义分割的数据集校验与基础训练入口。用户界面以统一 Vision Workbench 桌面里的原生 Qt 页面为准，同时保留 `yolo26-train` 命令行训练入口。

## 安装

按需安装 YOLO26 依赖：

```bash
python scripts/install_dependencies.py yolo26
```

如果手动使用 `requirements-yolo26.txt` 安装依赖，建议再执行 `python scripts/install_dependencies.py doctor` 检查 Torch 构建。GPU 训练需要 CUDA 版 PyTorch；CPU 可以运行但速度较慢。

## 启动

Qt GUI：

```bash
vision-workbench
```

在左侧导航打开 **模型训练**，选择 `detect`、`segment` 或 `semantic`，选择 `data.yaml` 和预训练权重文件，校验数据集后开始训练。

没有现成数据时可点击“创建示例数据”，页面会根据当前任务生成可通过校验的最小数据集。建议开始前执行“检查训练环境”，应用推荐批量后先用少量轮次试跑。训练日志实时显示在页面中，任务可停止；已由上游写出的 checkpoint 会保留。

命令行：

```bash
yolo26-train --task detect --data path/to/dataset/data.yaml --model models/yolo26_models/yolo26n.pt
```

## 任务类型

| 任务 | 标签格式 |
| --- | --- |
| `detect` | `class x_center y_center width height` |
| `segment` | `class x1 y1 x2 y2 x3 y3 ...` |
| `semantic` | PNG/TIF mask 或 polygon 标签 |

训练模型列表会按任务过滤：检测权重、`-seg` 权重和 `-sem` 权重不会互相混用；不完整的 `.pt` 文件会被跳过。

## 命令行训练

```bash
yolo26-train --task detect --data path/to/dataset/data.yaml --model path/to/model.pt --epochs 100 --imgsz 640 --batch 16 --device auto
```

仅校验：

```bash
yolo26-train --task detect --data path/to/data.yaml --model path/to/model.pt --dry-run
```

## 输出目录

```text
runs/yolo26_training/
```

YOLO26 训练输出通常包含权重、配置、日志和指标图。

常见权重路径：

```text
runs/yolo26_training/<run_name>/weights/best.pt
runs/yolo26_training/<run_name>/weights/last.pt
```

通常优先使用 `best.pt`。如果要让训练后的模型出现在检测或分割页面下拉框中，请复制并重命名到对应目录，然后点击页面里的“查找/刷新模型”按钮：

| 训练任务 | 推荐放置位置 | 命名建议 |
| --- | --- | --- |
| `detect` | `models/yolo26_models/custom/` | `my-det.pt` |
| `segment` | `models/yolo26_segmentation_models/custom/` | `my-seg.pt` |
| `semantic` | `models/yolo26_segmentation_models/custom/` | `my-sem.pt` |

更完整的训练后模型加载、下拉框扫描和 Python 接口说明见 [训练后模型加载与接口说明](./训练后模型加载与接口说明.md)。

## Python API

```python
from yolo26_training.api import validate_dataset, list_models

report = validate_dataset("path/to/dataset/data.yaml", task="detect")
models = list_models(task="detect")
```

## 源码结构

```text
src/yolo26_training/api/             对外 API
src/yolo26_training/application/     训练工作流
src/yolo26_training/configuration/   默认参数和路径
src/yolo26_training/domain/          数据模型
src/yolo26_training/infrastructure/  数据集校验、模型发现、YOLO 后端
src/yolo26_training/runner.py        命令行训练入口
src/yolo26_training/train.py         基础训练脚本
src/vision_workbench/desktop/        统一 Qt 界面
src/yolo26_training/window/          旧 Tkinter 兼容/参考代码
```

## 二次开发

新增训练任务、参数、数据集校验规则或日志策略时，修改 service/infrastructure 层并在 Qt 页面暴露控件。详见 [二次开发指南 - YOLO26 训练扩展](../../二次开发指南.md#yolo26-训练扩展)。
