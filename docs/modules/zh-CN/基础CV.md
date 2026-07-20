# 基础 CV README

[返回 README](../../../README.md) | [English](../en/cv_basics_README.md) | [二次开发指南](../../二次开发指南.md#基础-cv-扩展)

## 概览

基础 CV 模块提供单张图像的 OpenCV 处理能力，覆盖滤波、边缘检测、色彩空间、通道分离、直方图、形态学和几何变换。用户界面以统一 Vision Workbench 桌面里的原生 Qt 页面为准。

## 启动

```bash
vision-workbench
```

在左侧导航打开 **基础 CV**，选择图片、选择效果、调整可用参数，然后应用并保存结果。

## Python API

```python
from cv_basics.api import apply_effect, load_image, save_image

image = load_image("input.png")
result = apply_effect(image, "grayscale")
save_image(result, "output.png")
```

## 源码结构

```text
src/cv_basics/api/             对外 API
src/cv_basics/application/     工作流编排
src/cv_basics/configuration/   默认配置
src/cv_basics/domain/          数据模型
src/cv_basics/infrastructure/  图像 I/O 适配
src/cv_basics/processing/      OpenCV 操作
src/vision_workbench/desktop/  统一 Qt 界面
src/cv_basics/window/          旧 Tkinter 兼容/参考代码
```

## 二次开发

新增效果、参数或界面控件时，优先修改 processing/API 层，再在 `src/vision_workbench/desktop/pages/` 中补 Qt 控件。详见 [二次开发指南 - 基础 CV 扩展](../../二次开发指南.md#基础-cv-扩展)。
