# YOLO26 分割 README

[返回 README](../../../README.zh-CN.md) | [English](../en/yolo26_segmentation_README.md) | [二次开发指南](../../二次开发指南.md#yolo26-分割扩展)

## 概览

YOLO26 分割模块支持图片实例分割和语义分割。它会安全下载官方权重、校验本地 `.pt` 文件，并按任务区分 `-seg` 与 `-sem` 模型。用户界面以统一 Vision Workbench 桌面里的原生 Qt 页面为准。

## 安装

按需安装 YOLO26 依赖：

```bash
python scripts/install_dependencies.py yolo26
```

## 启动

```bash
vision-workbench
```

在左侧导航打开 **YOLO 分割**，选择任务和模型，加载图片，运行分割并保存结果。

## Python API

```python
from yolo26_segmentation.api import segment_image

result = segment_image("image.jpg", model_path="models/yolo26_segmentation_models/yolo26n-seg.pt")
```

## 源码结构

```text
src/yolo26_segmentation/api/             对外 API
src/yolo26_segmentation/application/     分割工作流
src/yolo26_segmentation/configuration/   默认参数和模型路径
src/yolo26_segmentation/domain/          数据模型
src/yolo26_segmentation/infrastructure/  图像、模型和 YOLO 适配
src/yolo26_segmentation/processing/      分割器封装
src/vision_workbench/desktop/            统一 Qt 界面
src/yolo26_segmentation/window/          旧 Tkinter 兼容/参考代码
```

## 二次开发

修改任务模型规则、渲染、保存、下载或运行参数时，先更新 service/infrastructure 层，再在 Qt 页面暴露控件。详见 [二次开发指南 - YOLO26 分割扩展](../../二次开发指南.md#yolo26-分割扩展)。
