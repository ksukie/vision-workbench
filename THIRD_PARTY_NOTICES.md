# Third-Party Notices

This document lists third-party software and model assets used by Vision Workbench.
Each third-party component remains under its own license.

## Ultralytics YOLO26

- Component: Ultralytics YOLO / YOLO26 source code
- Local path: `third_party/yolo26_source/`
- Upstream repository: https://github.com/ultralytics/ultralytics
- Vendored commit: `10d17c168cb6ff891d8c763c06c83dee53d4c75e`
- License: AGPL-3.0
- Local license file: `third_party/yolo26_source/LICENSE`
- Citation file: `third_party/yolo26_source/CITATION.cff`

Vision Workbench is not an official Ultralytics project. The vendored YOLO26
source and YOLO26 model weights are attributed to Ultralytics and must be used
according to the Ultralytics license terms.

## YOLO26 Model Weights

- Local paths:
  - `models/yolo26_models/`
  - `models/yolo26_segmentation_models/`
- Source: Ultralytics release assets
- Configured download base URL: `https://github.com/ultralytics/assets/releases/download/v8.4.0`
- License: follow the Ultralytics YOLO license terms

The repository release policy allows model files up to 100 MB to be committed.
Files above 100 MB should not be committed to Git. They should be downloaded
locally by the user or distributed through release assets.

## TorchVision Pretrained Classification Weights

- Components: ResNet18 and MobileNetV3 Small pretrained weights
- Local path: `models/image_classification_models/pretrained/`
- Upstream project: https://github.com/pytorch/vision
- Documentation: https://pytorch.org/vision/stable/models.html

These weights are used for educational image-classification demonstrations.
Users may also download them through the Image Classification GUI or import
local weight files.

## Runtime Dependencies

Vision Workbench depends on the following Python packages:

- NumPy: https://numpy.org/
- OpenCV: https://opencv.org/
- Pillow: https://python-pillow.org/
- PyTorch: https://pytorch.org/
- TorchVision: https://pytorch.org/vision/

See each project for its current license terms and citation requirements.
