# CV Basics README

[Back to README](../../../README.en.md) | [中文文档](../zh-CN/基础CV.md) | [Extension Guide](../../adding_custom_features_README.md#cv-basics-extensions)

## Overview

CV Basics provides single-image OpenCV operations for filtering, edge detection, color spaces, channel splitting, histograms, morphology, and geometric transforms. The user-facing workflow is the native Qt page inside the unified Vision Workbench desktop.

## Launch

```bash
vision-workbench
```

Open **CV Basics** from the left navigation, choose an image, select an effect, tune available parameters, then apply and save the result.

## Python API

```python
from cv_basics.api import apply_effect, load_image, save_image

image = load_image("input.png")
result = apply_effect(image, "grayscale")
save_image(result, "output.png")
```

## Source Layout

```text
src/cv_basics/api/             Public API
src/cv_basics/application/     Workflow orchestration
src/cv_basics/configuration/   Defaults
src/cv_basics/domain/          Data models
src/cv_basics/infrastructure/  Image I/O adapters
src/cv_basics/processing/      OpenCV operations
src/vision_workbench/desktop/  Unified Qt UI
src/cv_basics/window/          Legacy Tkinter compatibility/reference code
```

## Secondary Development

For new effects, parameters, or UI controls, update the processing/API layers first and add Qt controls under `src/vision_workbench/desktop/pages/`. See [Adding Custom Features - CV Basics Extensions](../../adding_custom_features_README.md#cv-basics-extensions).
