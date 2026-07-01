# CV Basics README

[Back to README](../../../README.md) | [中文文档](../zh-CN/cv_basics_README.zh-CN.md) | [Extension Guide](../../adding_custom_features_README.md#cv-basics-add)

## Overview

The CV Basics module provides traditional single-image computer-vision operations. It covers filtering, edge detection, color spaces, channel splitting, histograms, morphology, and geometric transforms. The module includes both a Tkinter GUI and Python APIs.

## Feature Scope

| Category | Features |
| --- | --- |
| Image I/O | Open, display, and save PNG/JPG/JPEG/BMP/TIF/TIFF images |
| Basic processing | Grayscale, blur, edge detection, thresholding, cartoonization |
| Color spaces | RGB Space, HSV Space, Gray |
| Channel splitting | Red, Green, Blue, Hue, Saturation, Value |
| Histograms | Grayscale histogram, RGB histogram |
| Morphology | Erosion, dilation, opening, closing |
| Geometric transforms | Rotation, scaling, center crop, perspective warp |
| Image metadata | Size, channel count, dtype, value range |

## Setup

Use the shared project environment from the root [Quick Start](../../../README.md#quick-start). This module has no extra optional dependencies beyond the base install.

For packaged releases, see [Release Packages](../../../README.md#release-packages).

## Launch

```bash
vision-workbench
```

Source entry:

```bash
python -m cv_basics.window.app
```

## Workflow

1. Select `Open` and choose an image.
2. Select an effect from `Effect`.
3. Adjust the relevant parameter sliders.
4. Select `Apply Effect`.
5. Select `Save Result` to save the processed image.
6. Select `Reset` to restore the result to the original image.

The parameter panel displays shared controls. Each effect uses only the parameters relevant to that operation.

## Python API

```python
from cv_basics.api import load_image, detect_edges, save_image

image = load_image("input.jpg")
result = detect_edges(image, low=80, high=160)
save_image(result, "output.png")
```

Common functions:

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

## Source Layout

```text
src/cv_basics/api/             Public API
src/cv_basics/application/     Image-processing workflows
src/cv_basics/configuration/   Defaults and configuration loading
src/cv_basics/domain/          Data models
src/cv_basics/infrastructure/  Image I/O and format conversion
src/cv_basics/processing/      Image-processing operations
src/cv_basics/window/          Tkinter GUI
```

## Secondary Development

For new image operations, parameters, or GUI controls, see [Adding Custom Features - CV Basics Extensions](../../adding_custom_features_README.md#cv-basics-add).
