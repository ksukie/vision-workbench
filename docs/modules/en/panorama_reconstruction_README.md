# Panorama Reconstruction README

[Back to README](../../../README.en.md) | [中文文档](../zh-CN/全景重构.md) | [Extension Guide](../../adding_custom_features_README.md#panorama-reconstruction-extensions)

## Overview

Panorama Reconstruction stitches a left reference image and a right image to stitch with automatic SIFT matching, manual point selection, and assisted manual workflows. The user-facing workflow is the native Qt page inside the unified Vision Workbench desktop.

## Launch

```bash
vision-workbench
```

Open **Panorama Reconstruction** from the left navigation, load the left reference image and the right image to stitch, choose automatic or manual mode, then reconstruct and save outputs. The algorithm maps the right image into the left reference coordinate system.

## Python API

```python
from panorama_reconstruction.api import reconstruct_panorama_from_paths

result = reconstruct_panorama_from_paths("left_reference.png", "right_to_stitch.png")
```

## Source Layout

```text
src/panorama_reconstruction/api/             Public API
src/panorama_reconstruction/application/     Reconstruction workflows
src/panorama_reconstruction/configuration/   Defaults
src/panorama_reconstruction/domain/          Data models
src/panorama_reconstruction/infrastructure/  Image I/O adapters
src/panorama_reconstruction/processing/      SIFT and manual reconstructors
src/vision_workbench/desktop/                Unified Qt UI
src/panorama_reconstruction/window/          Legacy Tkinter compatibility/reference code
```

## Secondary Development

For stitching algorithms, point selection, blending, or output policies, update the processing/application layers and expose controls through the Qt desktop page. See [Adding Custom Features - Panorama Reconstruction Extensions](../../adding_custom_features_README.md#panorama-reconstruction-extensions).
