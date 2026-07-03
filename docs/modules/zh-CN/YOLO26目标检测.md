# YOLO26 目标检测 README

[返回 README](../../../README.zh-CN.md) | [English](../en/yolo26_detection_README.md) | [二次开发指南](../../二次开发指南.md#yolo26-目标检测扩展)

## 概览

YOLO26 目标检测模块支持官方或本地 `.pt` 权重加载、安全下载缺失官方权重、单图检测和摄像头实时检测。用户界面以统一 Vision Workbench 桌面里的原生 Qt 页面为准。

## 安装

按需安装 YOLO26 依赖：

```bash
python scripts/install_dependencies.py yolo26
```

如果手动使用 `requirements-yolo26.txt` 安装依赖，建议再执行 `python scripts/install_dependencies.py doctor` 检查 Torch 构建。

## 启动

```bash
vision-workbench
```

在左侧导航打开 **YOLO 检测**。模型下拉框展开时显示完整路径，收起后只显示文件名。不完整或损坏的模型文件不会被视为可用权重。

## 自定义权重位置

检测页面会扫描：

```text
models/yolo26_models/
models/yolo26_models/custom/
~/.vision_workbench/models/yolo26_models/
```

训练完成后，把 `runs/yolo26_training/<run_name>/weights/best.pt` 复制为类似 `models/yolo26_models/custom/my-det.pt`，再点击检测页面的刷新按钮即可出现在模型下拉框中。

## Python API

```python
from yolo26_detection.api import detect_image

result = detect_image("image.jpg", model_path="models/yolo26_models/yolo26n.pt")
```

## 源码结构

```text
src/yolo26_detection/api/             对外 API
src/yolo26_detection/application/     检测工作流
src/yolo26_detection/configuration/   默认参数和模型路径
src/yolo26_detection/domain/          数据模型
src/yolo26_detection/infrastructure/  相机、模型和 YOLO 适配
src/yolo26_detection/processing/      检测器封装
src/vision_workbench/desktop/         统一 Qt 界面
src/yolo26_detection/window/          旧 Tkinter 兼容/参考代码
```

## 二次开发

修改模型发现、摄像头推理、结果绘制、下载或运行参数时，先更新 application/infrastructure 层，再在 Qt 页面暴露控件。详见 [二次开发指南 - YOLO26 目标检测扩展](../../二次开发指南.md#yolo26-目标检测扩展)。
