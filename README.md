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
  <a href="./SECURITY.md">Security</a>
  ·
  <a href="./CHANGELOG.md">Changelog</a>
  ·
  <a href="./THIRD_PARTY_NOTICES.md">Third-Party Notices</a>
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-AGPL--3.0-0f766e">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-2563eb">
  <img alt="GUI" src="https://img.shields.io/badge/GUI-PySide6-0a84ff">
  <img alt="Status" src="https://img.shields.io/badge/status-learning%20workbench-16a34a">
</p>

Vision Workbench is a local computer-vision learning workbench with one unified PySide6 desktop UI. It brings together traditional image processing, panorama reconstruction, camera diagnostics, image classification, YOLO26 detection, YOLO26 segmentation, and YOLO26 training behind the `vision-workbench` launcher.

The project is organized as a complete learning pipeline: public Python APIs, a Qt desktop shell, model and dataset folders, automated tests, packaging metadata, open-source license files, and third-party notices.

## Ecosystem

<p align="center">
  <img src="./docs/assets/readme/ecosystem.svg" alt="Vision Workbench ecosystem" width="100%">
</p>

## Modules

| Module | Scope | Documentation |
| --- | --- | --- |
| <img src="./docs/assets/readme/icons/cv.svg" width="28" alt=""> CV Basics | OpenCV image processing, color spaces, channel splitting, histograms, morphology, and geometric transforms | [README](./docs/modules/en/cv_basics_README.md) |
| <img src="./docs/assets/readme/icons/panorama.svg" width="28" alt=""> Panorama Reconstruction | Left-right image stitching, SIFT matching, manual control points, assisted points, and panorama output | [README](./docs/modules/en/panorama_reconstruction_README.md) |
| <img src="./docs/assets/readme/icons/camera.svg" width="28" alt=""> Camera Diagnostics | Camera discovery, read-mode probing, live preview, FPS, screenshots, and recording | [README](./docs/modules/en/camera_diagnostics_README.md) |
| <img src="./docs/assets/readme/icons/classification.svg" width="28" alt=""> Image Classification | ResNet18 and MobileNetV3 Small prediction, pretrained weights, dataset validation, and basic training | [README](./docs/modules/en/image_classification_README.md) |
| <img src="./docs/assets/readme/icons/detection.svg" width="28" alt=""> YOLO26 Detection | YOLO26 model loading, image detection, live camera inference, screenshots, and downloads | [README](./docs/modules/en/yolo26_detection_README.md) |
| <img src="./docs/assets/readme/icons/segmentation.svg" width="28" alt=""> YOLO26 Segmentation | YOLO26 instance and semantic segmentation for images | [README](./docs/modules/en/yolo26_segmentation_README.md) |
| <img src="./docs/assets/readme/icons/training.svg" width="28" alt=""> YOLO26 Training | Detection, instance segmentation, and semantic segmentation training with dataset validation | [README](./docs/modules/en/yolo26_training_README.md) |

## Quick Start

Prerequisites:

- Python 3.10 or newer; Python 3.11 is the tested default.
- Conda or another isolated Python environment.
- A local checkout of this repository.

Create the base environment, install the package in editable mode, then start the desktop application:

```bash
conda create -n vision-workbench python=3.11 -y
conda activate vision-workbench
cd path/to/vision-workbench
pip install -e .
vision-workbench
```

The first launch opens the unified Qt desktop shell. The left navigation provides direct access to CV Basics, Panorama Reconstruction, Camera Diagnostics, YOLO Detection, YOLO Segmentation, YOLO Training, and Image Classification.

Install the image-classification dependency group before using classification prediction or training:

```bash
python scripts/install_dependencies.py classification
vision-workbench
```

Install the YOLO26 dependency group before using detection, segmentation, or YOLO training:

```bash
python scripts/install_dependencies.py yolo26
vision-workbench
```

Run the dependency doctor after optional dependency installation:

```bash
python scripts/install_dependencies.py doctor
```

The base install keeps the first run lightweight. Optional deep-learning workflows are enabled explicitly so the desktop can open before large Torch and model dependencies are installed.

## Troubleshooting

For install command failures, GUI popups, camera workflow errors, model loading errors, dataset validation errors, and training failures, start with the [Troubleshooting Guide](./docs/troubleshooting/en/README.md). User-visible errors also print the matching troubleshooting document path.

After an abnormal exit, stale GUI, camera, or training processes can be listed with `python scripts/cleanup_runtime.py`. Review the dry-run list, then terminate matched project processes with `python scripts/cleanup_runtime.py --kill`.

## Security And Changes

Report suspected vulnerabilities through the private channel in [SECURITY.md](./SECURITY.md). Do not publish exploit details or sensitive environment data in public issues.

Version history is maintained in [CHANGELOG.md](./CHANGELOG.md).

## Release Packages

This repository keeps source code, documentation, tests, configuration, and license files. Generated build outputs such as `dist/` are not committed.

Official packages should be published through [GitHub Releases](https://github.com/ksukie/vision-workbench/releases). A packaged base application can be installed from the wheel file:

```bash
pip install vision_workbench-0.2.0-py3-none-any.whl
vision-workbench
```

The wheel installs the Python package and entry points. It is not a complete offline runtime: optional deep-learning workflows still need their dependency group, and large model weights may be distributed separately.

To build from source:

```bash
conda activate vision-workbench
pip install build
python -m build
```

When the isolated build environment cannot access PyPI, use the current environment instead:

```bash
python -m build --no-isolation
```

## Project Layout

```text
VisionWorkbench/
  src/                         Source packages
  src/vision_workbench/desktop/ Unified PySide6 desktop shell and pages
  docs/                        Documentation
  docs/assets/readme/          README images and icons
  models/                      Model weight directory
  datasets/                    Dataset directory
  runs/                        Training output directory
  third_party/yolo26_source/   Vendored YOLO26 source
  tests/                       Automated tests
```

## Dependency Strategy

Base dependencies cover the Qt desktop shell and traditional CV features: PySide6, NumPy, OpenCV, and Pillow.

Deep-learning features are optional:

- Image classification: `python scripts/install_dependencies.py classification`
- YOLO26 detection, segmentation, and training: `python scripts/install_dependencies.py yolo26`

The helper detects NVIDIA GPUs and installs CUDA 12.6 Torch wheels when available; otherwise it chooses the CPU or platform-default Torch build. Plain `pip install -r requirements-*.txt` cannot detect GPU availability by itself, so use the helper for new environments. Base packages continue to use the Tsinghua PyPI mirror.

After manual installation with `requirements-*.txt`, run the doctor:

```bash
python scripts/install_dependencies.py doctor
```

This keeps the entry-level installation lightweight while still allowing users to enable heavier workflows when needed.

## Architecture

Feature packages keep the domain logic in `api/`, `application/`, `domain/`, `infrastructure/`, and `processing/`. The user-facing desktop experience lives in `src/vision_workbench/desktop/`. Legacy `*/window/` Tkinter modules remain in the source tree only as compatibility/reference code and are not the public GUI entry points.

## License

Vision Workbench is released as an open-source learning project under AGPL-3.0. See [LICENSE](./LICENSE).

This project includes vendored Ultralytics YOLO26 source code under `third_party/yolo26_source/`. Vision Workbench is not an official Ultralytics project. YOLO26 source code and model weights remain subject to the Ultralytics license terms. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

## Acknowledgments

2026.07.01 - Thanks to [@antique798](https://github.com/antique798) for helping test Vision Workbench and providing feedback.
