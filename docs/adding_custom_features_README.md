# Adding Custom Features

[Back to README](../README.md) | [中文文档](./adding_custom_features_README.zh-CN.md)

This document describes the extension points, dependency rules, and wheel packaging workflow for Vision Workbench.

## Development Environment

```bash
conda create -n vision-dev python=3.11 -y
conda activate vision-dev
cd path/to/vision-workbench
pip install -e .[test]
```

Install image-classification dependencies when using the classification module:

```bash
python scripts/install_dependencies.py classification
```

Install YOLO26 dependencies when using YOLO26 modules:

```bash
python scripts/install_dependencies.py yolo26
```

Run tests:

```bash
pytest
```

## Layering Model

Modules follow a consistent layered structure:

```text
api/             Public API functions
application/     Application services and workflow orchestration
configuration/   Paths, parameters, and defaults
domain/          Data models
infrastructure/  File, camera, model, and third-party adapters
processing/      Core algorithms or model processing
window/          Tkinter GUI
```

Recommended call flow:

```text
window -> application -> processing / infrastructure -> application -> window
```

Algorithm and model logic should be placed in `processing/`. External adapters should be placed in `infrastructure/`. Workflow orchestration should be placed in `application/`. User-facing Python functions should be exposed through `api/`.

## Dependency Rules

Base dependencies:

```text
requirements.txt
pyproject.toml
```

Image-classification dependencies:

```text
requirements-classification.txt
```

YOLO26 dependencies:

```text
requirements-yolo26.txt
```

Deep-learning dependencies used only by image classification or YOLO26 modules should not be added to the base dependency set.
Use `scripts/install_dependencies.py` for runtime installs because it can choose CUDA or CPU Torch based on the machine.
If users install `requirements-*.txt` manually, ask them to run `python scripts/install_dependencies.py doctor` afterward to verify and optionally repair the Torch build.

<a id="cv-basics-add"></a>
## CV Basics Extensions

Scope: new image-processing effects, channel analysis, histograms, morphology operations, or geometric transforms.

Primary files:

```text
src/cv_basics/domain/models.py
src/cv_basics/processing/operations.py
src/cv_basics/api/facade.py
src/cv_basics/api/__init__.py
src/cv_basics/window/app.py
```

Extension steps:

1. Add an effect name to `EffectName`.
2. Add parameters to `ProcessingParams` when required.
3. Implement the operation in `operations.py`.
4. Register the operation in `build_default_registry()`.
5. Add a public function in `api/facade.py`.
6. Export the function in `api/__init__.py`.
7. Update `window/app.py` when GUI controls are required.
8. Update `configuration/settings.py` when JSON configuration support is required.

<a id="panorama-add"></a>
## Panorama Reconstruction Extensions

Scope: stitching algorithms, registration strategies, point-selection workflows, blending, or cropping.

Primary files:

```text
src/panorama_reconstruction/processing/sift_reconstructor.py
src/panorama_reconstruction/processing/manual_reconstructor.py
src/panorama_reconstruction/application/reconstruction_service.py
src/panorama_reconstruction/api/facade.py
src/panorama_reconstruction/window/app.py
```

Place algorithms in `processing/`, image loading and saving workflows in `application/`, public functions in `api/`, and user interaction in `window/`.

<a id="camera-add"></a>
## Camera Diagnostics Extensions

Scope: camera backends, read modes, exposure settings, resolution strategies, screenshots, or recording workflows.

Primary files:

```text
src/camera_diagnostics/infrastructure/platform_detector.py
src/camera_diagnostics/infrastructure/camera_repository.py
src/camera_diagnostics/application/camera_service.py
src/camera_diagnostics/api/facade.py
src/camera_diagnostics/window/app.py
```

Camera errors should be handled in the adapter layer and returned to the GUI as clear messages.

<a id="image-classification-add"></a>
## Image Classification Extensions

Scope: classification backbones, pretrained weight rules, augmentation, metrics, batch prediction, or export formats.

Primary files:

```text
src/image_classification/configuration/settings.py
src/image_classification/infrastructure/pretrained_weights.py
src/image_classification/infrastructure/dataset_validator.py
src/image_classification/infrastructure/dataset_splitter.py
src/image_classification/infrastructure/model_repository.py
src/image_classification/processing/classifier_backend.py
src/image_classification/application/classification_service.py
src/image_classification/api/facade.py
src/image_classification/window/app.py
```

Backbone options are maintained in `configuration/settings.py`. Pretrained weight detection, download, and offline import are maintained in `pretrained_weights.py`. Training and prediction logic is maintained in `classifier_backend.py`.

<a id="yolo26-detection-add"></a>
## YOLO26 Detection Extensions

Scope: detection models, model directories, inference parameters, class filtering, result saving, or post-processing.

Primary files:

```text
src/yolo26_detection/configuration/settings.py
src/yolo26_detection/infrastructure/model_registry.py
src/yolo26_detection/infrastructure/detector_backend.py
src/yolo26_detection/application/detection_service.py
src/yolo26_detection/api/facade.py
src/yolo26_detection/window/app.py
```

YOLO26 third-party source is stored in:

```text
third_party/yolo26_source/
```

Business modules should call third-party source through backend adapters.

<a id="yolo26-segmentation-add"></a>
## YOLO26 Segmentation Extensions

Scope: instance segmentation models, semantic segmentation models, mask saving, visualization, class filtering, or batch segmentation.

Primary files:

```text
src/yolo26_segmentation/configuration/settings.py
src/yolo26_segmentation/infrastructure/model_registry.py
src/yolo26_segmentation/infrastructure/segmentation_backend.py
src/yolo26_segmentation/application/segmentation_service.py
src/yolo26_segmentation/api/facade.py
src/yolo26_segmentation/window/app.py
```

Recommended model naming:

```text
*-seg.pt  Instance segmentation
*-sem.pt  Semantic segmentation
```

<a id="yolo26-training-add"></a>
## YOLO26 Training Extensions

Scope: training parameters, dataset validation rules, training tasks, log saving, or model management.

Primary files:

```text
src/yolo26_training/train.py
src/yolo26_training/runner.py
src/yolo26_training/infrastructure/dataset_validator.py
src/yolo26_training/infrastructure/training_backend.py
src/yolo26_training/application/training_service.py
src/yolo26_training/api/facade.py
src/yolo26_training/window/app.py
```

Dataset validation must run before training. Invalid datasets should stop the training flow and return actionable messages.

<a id="build-wheel"></a>
## Wheel Packaging

Pre-package checklist:

- GUI entry points start successfully.
- Newly added features run successfully.
- Automated tests pass.
- New dependencies are declared in the proper dependency file.
- `pyproject.toml` version is updated when required.
- Cache and temporary files are removed.
- Release assets pass the model file size policy.

Check release assets:

```bash
python scripts/check_release_assets.py
```

Install build tools:

```bash
pip install build
```

Build:

```bash
python -m build
```

Output directory:

```text
dist/
```

Validate in a clean environment:

```bash
conda create -n vision-user python=3.11 -y
conda activate vision-user
pip install dist/your_package.whl
vision-workbench
```

Large model weights should not be bundled into the default wheel. They should be distributed as project assets or offline resource packages.

For the current project policy, see [Release Policy](./legal/release_policy.md).
