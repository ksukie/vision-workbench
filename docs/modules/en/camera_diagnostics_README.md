# Camera Diagnostics README

[Back to README](../../../README.md) | [中文文档](../zh-CN/camera_diagnostics_README.zh-CN.md) | [Extension Guide](../../adding_custom_features_README.md#camera-add)

## Overview

The Camera Diagnostics module detects local cameras, probes read modes, displays live frames and FPS, and supports screenshots and video recording. It selects OpenCV camera backends based on the operating system.

## Feature Scope

| Feature | Description |
| --- | --- |
| Platform detection | Windows, Linux, macOS |
| Camera scan | Detect available camera indices |
| Read-mode probing | Try backends, resolutions, and frame rates |
| Live preview | Display camera frames |
| FPS display | Display FPS outside the image region |
| Screenshot | Save the current frame |
| Recording | Save video output |
| Error messages | Camera occupied, read failure, save failure, and related errors |

Backend routes:

```text
Windows: DSHOW / MSMF / ANY
Linux:   V4L2 / ANY
macOS:   AVFOUNDATION / ANY
```

## Setup

Use the shared project environment from the root [Quick Start](../../../README.md#quick-start). This module has no extra optional dependencies beyond the base install.

## Launch

```bash
camera-diagnostics-workbench
```

Source entry:

```bash
python -m camera_diagnostics.window.app
```

## Workflow

1. Select `Refresh Cameras`.
2. Select a device from `Camera`.
3. Select `Probe Modes`.
4. Select a read mode from `Read mode`.
5. Select `Open` to start preview.
6. Select `Screenshot` to save a frame.
7. Select `Start Recording` to start recording.
8. Select `Stop Recording` to stop recording.
9. Close the window to release camera resources.

## Python API

```python
from camera_diagnostics.api import discover_cameras, probe_profiles

cameras = discover_cameras()
profiles = probe_profiles(cameras[0].index)
```

Common functions:

```text
detect_platform()
discover_cameras()
probe_profiles(camera_index)
create_camera_diagnostics_service()
```

## Source Layout

```text
src/camera_diagnostics/api/             Public API
src/camera_diagnostics/application/     Camera workflows
src/camera_diagnostics/configuration/   Defaults
src/camera_diagnostics/domain/          Data models
src/camera_diagnostics/infrastructure/  Platform detection, camera I/O, screenshots, recording
src/camera_diagnostics/window/          Tkinter GUI
```

## Secondary Development

For camera backends, read modes, exposure settings, or recording strategies, see [Adding Custom Features - Camera Diagnostics Extensions](../../adding_custom_features_README.md#camera-add).
