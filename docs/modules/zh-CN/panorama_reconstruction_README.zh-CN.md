# 全景重构 README

[English](../en/panorama_reconstruction_README.md) | [返回总 README](../../../README.zh-CN.md) | [二次开发入口](../../adding_custom_features_README.zh-CN.md#panorama-add)

## 概述

全景重构模块用于将具有重叠区域的左右图像重构为全景图。模块包含自动特征重构、人工点选和人工点选辅助重构流程。

## 功能范围

- 左图、右图加载。
- 内置 buildings 示例图加载。
- 左右图对应点人工标记。
- 点对 JSON 保存与读取。
- 3 点仿射恢复。
- 4 点及以上透视 / 单应性恢复。
- 人工点选辅助匹配。
- TPS 薄板样条非刚性恢复。
- 全景图与调试结果保存。

## 环境要求

```bash
conda create -n vision-workbench python=3.11 -y
conda activate vision-workbench
cd C:\Users\asus\Desktop\Package-Wheel\VisionWorkbench
pip install -e .
```

## 启动方式

```bash
panorama-reconstruction-workbench
```

源码方式：

```bash
python .\src\panorama_reconstruction\window\app.py
```

## 输入要求

左右图像应来自同一场景，并具有稳定重叠区域。推荐在重叠区域选择角点、边缘交叉点或纹理明显的位置作为对应点。

## 操作流程

1. 点击 `Open Left` 选择左图。
2. 点击 `Open Right` 选择右图。
3. 可点击 `Load Sample Pair` 加载内置示例。
4. 在左图选择特征点。
5. 在右图选择对应点。
6. 至少添加 3 组对应点。
7. 在 `Mode` 中选择 `手动` 或 `手动+辅助`。
8. 点击 `Reconstruct` 生成全景图。
9. 点击 `Save Panorama` 保存最终全景图。
10. 点击 `Save All Outputs` 保存调试结果。

## 模式说明

| 模式 | 说明 |
| --- | --- |
| 手动 | 完全使用人工标记点。3 点执行仿射恢复，4 点及以上执行透视 / 单应性恢复。 |
| 手动+辅助 | 以人工点为种子生成局部辅助匹配，并使用 TPS 做非刚性恢复。 |

## Python API

```python
from panorama_reconstruction.api import (
    load_image,
    reconstruct_manual_panorama,
    save_reconstruction_outputs,
)

left = load_image("left.png")
right = load_image("right.png")
point_pairs = [
    ((120, 80), (34, 82)),
    ((360, 90), (274, 88)),
    ((355, 260), (270, 258)),
    ((130, 250), (44, 252)),
]

result = reconstruct_manual_panorama(left, right, point_pairs)
save_reconstruction_outputs(result, "panorama_output")
```

## 目录结构

```text
src/panorama_reconstruction/api/             对外 API
src/panorama_reconstruction/application/     重构业务流程
src/panorama_reconstruction/configuration/   默认参数
src/panorama_reconstruction/domain/          数据模型
src/panorama_reconstruction/infrastructure/  图像读写
src/panorama_reconstruction/processing/      重构算法
src/panorama_reconstruction/assets/          示例图片
src/panorama_reconstruction/window/          Tkinter GUI
```

## 二次开发

新增拼接算法、配准策略、点选策略或融合策略见：[添加自定义功能 README - 全景重构扩展](../../adding_custom_features_README.zh-CN.md#panorama-add)。
