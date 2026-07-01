# 相机诊断 README

[English](../en/camera_diagnostics_README.md) | [返回总 README](../../../README.zh-CN.md) | [二次开发入口](../../adding_custom_features_README.zh-CN.md#camera-add)

## 概述

相机诊断模块用于检测本机摄像头、探测读取模式、实时预览画面、显示 FPS，并提供截图与录屏功能。模块根据操作系统选择 OpenCV 摄像头后端。

## 功能范围

| 功能 | 说明 |
| --- | --- |
| 系统识别 | Windows、Linux、macOS |
| 摄像头扫描 | 扫描可打开的摄像头编号 |
| 读取模式探测 | 尝试不同后端、分辨率和帧率 |
| 实时预览 | 显示摄像头画面 |
| FPS 显示 | FPS 文本显示在图像区域外 |
| 截图 | 保存当前帧 |
| 录屏 | 保存实时视频 |
| 异常提示 | 摄像头占用、读取失败、保存失败等错误提示 |

后端路线：

```text
Windows: DSHOW / MSMF / ANY
Linux:   V4L2 / ANY
macOS:   AVFOUNDATION / ANY
```

## 安装说明

请先按根目录 [快速开始](../../../README.zh-CN.md#快速开始) 统一配置项目环境。本模块只依赖基础安装，不需要额外依赖组。

## 启动方式

```bash
camera-diagnostics-workbench
```

源码方式：

```bash
python -m camera_diagnostics.window.app
```

## 操作流程

1. 点击 `Refresh Cameras` 扫描摄像头。
2. 在 `Camera` 中选择设备。
3. 点击 `Probe Modes` 探测读取模式。
4. 在 `Read mode` 中选择读取方案。
5. 点击 `Open` 开始预览。
6. 点击 `Screenshot` 保存截图。
7. 点击 `Start Recording` 开始录屏。
8. 点击 `Stop Recording` 停止录屏。
9. 关闭窗口释放摄像头资源。

## Python API

```python
from camera_diagnostics.api import discover_cameras, probe_profiles

cameras = discover_cameras()
profiles = probe_profiles(cameras[0].index)
```

常用函数：

```text
detect_platform()
discover_cameras()
probe_profiles(camera_index)
create_camera_diagnostics_service()
```

## 目录结构

```text
src/camera_diagnostics/api/             对外 API
src/camera_diagnostics/application/     摄像头业务流程
src/camera_diagnostics/configuration/   默认参数
src/camera_diagnostics/domain/          数据模型
src/camera_diagnostics/infrastructure/  平台判断、相机读取、截图录屏
src/camera_diagnostics/window/          Tkinter GUI
```

## 二次开发

新增相机后端、读取模式、曝光设置或录制策略见：[添加自定义功能 README - 相机诊断扩展](../../adding_custom_features_README.zh-CN.md#camera-add)。
