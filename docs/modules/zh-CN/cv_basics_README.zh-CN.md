# 基础 CV README

[English](../en/cv_basics_README.md) | [返回总 README](../../../README.zh-CN.md) | [二次开发入口](../../adding_custom_features_README.zh-CN.md#cv-basics-add)

## 概述

基础 CV 模块提供单张图像的传统计算机视觉处理功能，覆盖基础滤波、边缘检测、色彩空间、通道分离、直方图、形态学和几何变换。模块提供 Tkinter GUI 与 Python API。

## 功能范围

| 分类 | 功能 |
| --- | --- |
| 图像输入输出 | 打开、显示、保存 PNG/JPG/JPEG/BMP/TIF/TIFF |
| 基础处理 | 灰度化、模糊、边缘检测、二值化、卡通化 |
| 色彩空间 | RGB Space、HSV Space、Gray |
| 通道分离 | Red、Green、Blue、Hue、Saturation、Value |
| 直方图 | 灰度直方图、RGB 三通道直方图 |
| 形态学 | 腐蚀、膨胀、开运算、闭运算 |
| 几何变换 | 旋转、缩放、中心裁剪、透视变换 |
| 图像信息 | 尺寸、通道数、数据类型、像素范围 |

## 环境要求

```bash
conda create -n vision-workbench python=3.11 -y
conda activate vision-workbench
cd C:\Users\asus\Desktop\Package-Wheel\VisionWorkbench
pip install -e .
```

安装 wheel 时使用：

```bash
pip install C:\path\to\your_package.whl
```

## 启动方式

```bash
vision-workbench
```

源码方式：

```bash
python .\src\cv_basics\window\app.py
```

## 操作流程

1. 点击 `Open` 选择图片。
2. 在 `Effect` 中选择处理效果。
3. 根据当前效果调整参数滑块。
4. 点击 `Apply Effect` 生成处理结果。
5. 点击 `Save Result` 保存结果图像。
6. 点击 `Reset` 恢复原图结果。

参数区统一显示多组滑块。各效果仅使用自身所需参数。

## Python API

```python
from cv_basics.api import load_image, detect_edges, save_image

image = load_image("input.jpg")
result = detect_edges(image, low=80, high=160)
save_image(result, "output.png")
```

常用函数：

```text
load_image(path)
save_image(image, path)
to_grayscale(image)
apply_blur(image, ksize=9)
detect_edges(image, low=80, high=160)
threshold_image(image, threshold=127)
cartoonize(image)
show_rgb_space(image)
show_hsv_space(image)
extract_red_channel(image)
extract_green_channel(image)
extract_blue_channel(image)
extract_hue_channel(image)
extract_saturation_channel(image)
extract_value_channel(image)
gray_histogram(image)
rgb_histogram(image)
erode_image(image, kernel_size=5, iterations=1)
dilate_image(image, kernel_size=5, iterations=1)
morph_open_image(image, kernel_size=5, iterations=1)
morph_close_image(image, kernel_size=5, iterations=1)
rotate_image(image, angle=30)
scale_image(image, percent=120)
center_crop_image(image, percent=70)
perspective_warp(image, shift=12)
get_image_info(image)
list_effects()
apply_effect(image, effect_name, params)
```

## 目录结构

```text
src/cv_basics/api/             对外 API
src/cv_basics/application/     图像处理业务流程
src/cv_basics/configuration/   默认参数和配置读取
src/cv_basics/domain/          数据模型
src/cv_basics/infrastructure/  图像读写和格式转换
src/cv_basics/processing/      图像处理算法
src/cv_basics/window/          Tkinter GUI
```

## 二次开发

新增图像处理功能、参数或 GUI 控件见：[添加自定义功能 README - 基础 CV 扩展](../../adding_custom_features_README.zh-CN.md#cv-basics-add)。
