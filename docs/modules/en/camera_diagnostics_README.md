# Camera Diagnostics README

[Back to README](../../../README.en.md) | [中文文档](../zh-CN/相机诊断.md) | [Extension Guide](../../adding_custom_features_README.md#camera-diagnostics-extensions)

## Overview

Camera Diagnostics discovers local cameras, probes capture modes, opens live previews, reports FPS, and supports screenshots or recording. The user-facing workflow is the native Qt page inside the unified Vision Workbench desktop.

## Launch

```bash
vision-workbench
```

Open **Camera Diagnostics** from the left navigation, refresh cameras, choose a device/profile, then open the camera and capture outputs.

## Python API

```python
from camera_diagnostics.api import discover_cameras

cameras = discover_cameras()
```

## Source Layout

```text
src/camera_diagnostics/api/             Public API
src/camera_diagnostics/application/     Camera workflows
src/camera_diagnostics/configuration/   Defaults
src/camera_diagnostics/domain/          Data models
src/camera_diagnostics/infrastructure/  OpenCV/platform adapters
src/vision_workbench/desktop/           Unified Qt UI
src/camera_diagnostics/window/          Legacy Tkinter compatibility/reference code
```

## Secondary Development

For camera backends, profile probing, screenshots, recording, or resource-handling changes, update the service and infrastructure layers first, then expose UI controls in the Qt desktop page. See [Adding Custom Features - Camera Diagnostics Extensions](../../adding_custom_features_README.md#camera-diagnostics-extensions).
