# Panorama Reconstruction README

[Back to README](../../../README.md) | [中文文档](../zh-CN/panorama_reconstruction_README.zh-CN.md) | [Extension Guide](../../adding_custom_features_README.md#panorama-add)

## Overview

The Panorama Reconstruction module reconstructs a panorama from left and right images with overlapping regions. It includes automatic feature reconstruction, manual control points, and assisted manual reconstruction.

## Feature Scope

- Load left and right images.
- Load the built-in buildings sample pair.
- Select matching points manually.
- Save and load point-pair JSON files.
- Perform affine reconstruction with 3 points.
- Perform perspective / homography reconstruction with 4 or more points.
- Generate assisted local matches.
- Apply TPS non-rigid reconstruction.
- Save panorama and debug outputs.

## Setup

Use the shared project environment from the root [Quick Start](../../../README.md#quick-start). This module has no extra optional dependencies beyond the base install.

## Launch

```bash
panorama-reconstruction-workbench
```

Source entry:

```bash
python -m panorama_reconstruction.window.app
```

## Input Requirements

The left and right images should come from the same scene and contain a stable overlapping region. Corners, edge intersections, and textured points are preferred as control points.

## Workflow

1. Select `Open Left`.
2. Select `Open Right`.
3. Optionally select `Load Sample Pair`.
4. Select one feature point in the left image.
5. Select the corresponding point in the right image.
6. Add at least 3 point pairs.
7. Select `手动` or `手动+辅助` in `Mode`.
8. Select `Reconstruct`.
9. Select `Save Panorama` to save the final panorama.
10. Select `Save All Outputs` to save debug outputs.

## Modes

| Mode | Description |
| --- | --- |
| 手动 | Uses only manually selected points. 3 points run affine reconstruction; 4 or more points run perspective / homography reconstruction. |
| 手动+辅助 | Uses manual points as seeds, generates assisted local matches, and applies TPS non-rigid reconstruction. |

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

## Source Layout

```text
src/panorama_reconstruction/api/             Public API
src/panorama_reconstruction/application/     Reconstruction workflows
src/panorama_reconstruction/configuration/   Defaults
src/panorama_reconstruction/domain/          Data models
src/panorama_reconstruction/infrastructure/  Image I/O
src/panorama_reconstruction/processing/      Reconstruction algorithms
src/panorama_reconstruction/assets/          Sample images
src/panorama_reconstruction/window/          Tkinter GUI
```

## Secondary Development

For stitching algorithms, registration strategies, point-selection workflows, or blending logic, see [Adding Custom Features - Panorama Reconstruction Extensions](../../adding_custom_features_README.md#panorama-add).
