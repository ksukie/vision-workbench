# Vision Workbench

<p align="center">
  <img src="./docs/assets/readme/vision_workbench_hero.png" alt="Vision Workbench overview" width="100%">
</p>

<p align="center">
  <a href="./README.zh-CN.md">中文文档</a>
  ·
  <a href="./docs/adding_custom_features_README.md">Extension Guide</a>
  ·
  <a href="./docs/legal/release_policy.md">Release Policy</a>
  ·
  <a href="./THIRD_PARTY_NOTICES.md">Third-Party Notices</a>
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-AGPL--3.0-0f766e">
  <img alt="Python" src="https://img.shields.io/badge/python-3.8%2B-2563eb">
  <img alt="GUI" src="https://img.shields.io/badge/GUI-Tkinter-f59e0b">
  <img alt="Status" src="https://img.shields.io/badge/status-learning%20workbench-16a34a">
</p>

Vision Workbench is a local computer-vision learning workbench. It connects traditional image processing, panorama reconstruction, camera diagnostics, image classification, YOLO26 object detection, YOLO26 segmentation, and YOLO26 training into one beginner-friendly project.

The project is designed as a complete learning pipeline: public Python APIs, desktop GUI windows, model and dataset folders, tests, packaging metadata, open-source license files, and third-party notices.

## Ecosystem

<p align="center">
  <img src="./docs/assets/readme/ecosystem.svg" alt="Vision Workbench ecosystem" width="100%">
</p>

## Modules

| Module | Scope | Documentation |
| --- | --- | --- |
| <img src="./docs/assets/readme/icons/cv.svg" width="28" alt=""> CV Basics | OpenCV image processing, color spaces, channel splitting, histograms, morphology, and geometric transforms | [README](./docs/modules/en/cv_basics_README.md) |
| <img src="./docs/assets/readme/icons/panorama.svg" width="28" alt=""> Panorama Reconstruction | Left-right image reconstruction, SIFT matching, manual control points, assisted points, and panorama output | [README](./docs/modules/en/panorama_reconstruction_README.md) |
| <img src="./docs/assets/readme/icons/camera.svg" width="28" alt=""> Camera Diagnostics | Camera discovery, read-mode probing, live preview, FPS, screenshots, and recording | [README](./docs/modules/en/camera_diagnostics_README.md) |
| <img src="./docs/assets/readme/icons/classification.svg" width="28" alt=""> Image Classification | ResNet18 and MobileNetV3 Small prediction, pretrained weights, dataset validation, and basic training | [README](./docs/modules/en/image_classification_README.md) |
| <img src="./docs/assets/readme/icons/detection.svg" width="28" alt=""> YOLO26 Detection | YOLO26 detection model loading, camera inference, screenshots, and recording | [README](./docs/modules/en/yolo26_detection_README.md) |
| <img src="./docs/assets/readme/icons/segmentation.svg" width="28" alt=""> YOLO26 Segmentation | YOLO26 instance segmentation and semantic segmentation for images and camera input | [README](./docs/modules/en/yolo26_segmentation_README.md) |
| <img src="./docs/assets/readme/icons/training.svg" width="28" alt=""> YOLO26 Training | Detection, instance segmentation, and semantic segmentation training with dataset validation | [README](./docs/modules/en/yolo26_training_README.md) |

## Quick Start

Base environment for traditional CV, panorama reconstruction, and camera diagnostics:

```bash
conda create -n vision-workbench python=3.11 -y
conda activate vision-workbench
cd path/to/vision-workbench
pip install -e .
vision-workbench
```

Image classification:

```bash
python scripts/install_dependencies.py classification
image-classification-workbench
```

YOLO26 workflows:

```bash
python scripts/install_dependencies.py yolo26
yolo26-detection-workbench
```

## Release Packages

This repository keeps source code, documentation, tests, configuration, and license files. Generated build outputs such as `dist/` are not committed to the source repository.

Official version packages should be published through [GitHub Releases](https://github.com/ksukie/vision-workbench/releases). Users who only want to install a packaged version can download the `.whl` file from a release page and install it locally:

```bash
pip install vision_workbench-0.1.0-py3-none-any.whl
vision-workbench
```

Users who want to build their own package from source can run:

```bash
conda activate vision-workbench
pip install build
python -m build
```

If the isolated build environment cannot access PyPI, use the current environment instead:

```bash
python -m build --no-isolation
```

## Project Layout

```text
VisionWorkbench/
  src/                         Source packages
  docs/                        Documentation
  docs/assets/readme/          README images and icons
  models/                      Model weight directory
  datasets/                    Dataset directory
  runs/                        Training output directory
  third_party/yolo26_source/   Vendored YOLO26 source
  tests/                       Automated tests
```

## Dependency Strategy

Base dependencies are intentionally small: NumPy, OpenCV, and Pillow.

Deep-learning features are optional:

- Image classification: `python scripts/install_dependencies.py classification`
- YOLO26 detection, segmentation, and training: `python scripts/install_dependencies.py yolo26`

The helper detects NVIDIA GPUs and installs CUDA 12.6 Torch wheels when available. Direct requirements installs keep the same CUDA Torch pins, while base packages continue to use the Tsinghua PyPI mirror.

If you install deep-learning dependencies manually with `requirements-*.txt`, run the doctor afterward:

```bash
python scripts/install_dependencies.py doctor
```

It checks whether Torch matches the current machine and asks before reinstalling Torch.

This keeps the entry-level installation lightweight while still allowing users to enable heavier workflows when needed.



## License

Vision Workbench is released as an open-source learning project under AGPL-3.0. See [LICENSE](./LICENSE).

This project includes vendored Ultralytics YOLO26 source code under `third_party/yolo26_source/`. Vision Workbench is not an official Ultralytics project. YOLO26 source code and model weights remain subject to the Ultralytics license terms. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
