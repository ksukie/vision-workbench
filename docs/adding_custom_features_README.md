# Adding Custom Features

[Back to README](../README.md) | [Documentation](./README.md) | [中文文档](./二次开发指南.md)

This contributor guide describes where to add new Vision Workbench capabilities. Runtime users should start from the root [Quick Start](../README.md#quick-start).

## Development Environment

```bash
conda create -n vision-dev python=3.11 -y
conda activate vision-dev
cd path/to/vision-workbench
pip install -e .[test]
pytest
```

Install optional runtime groups only when working on those modules:

```bash
python scripts/install_dependencies.py classification
python scripts/install_dependencies.py yolo26
```

## Layering Model

Feature packages use the following directories according to their responsibilities. Apart from the main `api/` and `application/` boundaries, directories are optional and should not be added only to make every package look identical:

```text
api/             Public Python API for callers
application/     Use-case and workflow orchestration
configuration/   Paths, parameters, and defaults; optional
domain/          Domain data models; optional
infrastructure/  File, camera, model, and third-party adapters; optional
processing/      Core algorithms or model processing; optional
ports/           Protocols and dependency boundaries; optional
```

The `window/` directories preserve legacy Tkinter compatibility/reference code and are not part of the standard structure for new feature work. The supported GUI and all new interface work belong to the Qt desktop application.

The public desktop UI lives in:

```text
src/vision_workbench/desktop/
src/vision_workbench/desktop/pages/
```

Recommended call flow:

```text
desktop page -> application -> processing / infrastructure -> application -> desktop page
```

Keep algorithm/model logic in `processing/`, external adapters in `infrastructure/`, workflows in `application/`, and public Python functions in `api/`. Omit a directory when the package has no corresponding responsibility; do not create empty layers for visual uniformity. New user-facing GUI work should target the Qt desktop pages. Do not add new public GUI entry points under `window/` unless required by an explicit compatibility commitment. See the [Legacy GUI Policy](./legacy-gui-policy.md).

## Dependency Rules

Base dependencies live in `requirements.txt` and `pyproject.toml`. Heavy dependencies should stay optional:

- Image classification: `requirements-classification.txt`
- YOLO26 workflows: `requirements-yolo26.txt`

Use `scripts/install_dependencies.py` for runtime installs because it can choose CUDA or CPU Torch based on the machine. If users install `requirements-*.txt` manually, ask them to run `python scripts/install_dependencies.py doctor` afterward.

## CV Basics Extensions

Scope: new image-processing effects, channel analysis, histograms, morphology operations, or geometric transforms.

Primary files:

```text
src/cv_basics/domain/models.py
src/cv_basics/processing/operations.py
src/cv_basics/api/facade.py
src/cv_basics/api/__init__.py
src/vision_workbench/desktop/pages/cv_basics_page.py
```

Add effect names and parameters in the domain layer, implement operations in `processing/`, expose API functions, then add Qt controls when needed.

## Panorama Reconstruction Extensions

Scope: stitching algorithms, registration strategies, point-selection workflows, blending, or cropping.

Primary files:

```text
src/panorama_reconstruction/processing/
src/panorama_reconstruction/application/reconstruction_service.py
src/panorama_reconstruction/api/facade.py
src/vision_workbench/desktop/pages/panorama_page.py
```

Put algorithms in `processing/`, file/workflow orchestration in `application/`, public functions in `api/`, and UI controls in the Qt page.

Note: `left` and `right` in panorama reconstruction are semantic roles, not filenames. `left` is the reference image, `right` is the image to stitch, and the algorithm should warp `right` into the `left` coordinate system. Some historical sample assets may have filenames that do not match the spatial semantics; relying on filenames can produce correct output with confusing UI semantics.

## Camera Diagnostics Extensions

Scope: camera backends, read modes, exposure/settings, screenshot, recording, and resource ownership.

Primary files:

```text
src/camera_diagnostics/infrastructure/
src/camera_diagnostics/application/camera_service.py
src/camera_diagnostics/api/facade.py
src/vision_workbench/desktop/pages/camera_page.py
```

Capture platform-specific errors in infrastructure and return clear messages through the service layer.

## Image Classification Extensions

Scope: new backbones, pretrained weights, augmentation, metrics, batch prediction, checkpoint loading, or export.

Primary files:

```text
src/image_classification/configuration/settings.py
src/image_classification/infrastructure/
src/image_classification/processing/classifier_backend.py
src/image_classification/application/classification_service.py
src/image_classification/api/facade.py
src/vision_workbench/desktop/pages/classification_page.py
```

Keep weight validation and download logic in infrastructure. Keep training/prediction orchestration in the service layer. Add Qt controls only after the API/service contract is stable.

## YOLO26 Detection Extensions

Scope: detection models, downloads, camera inference, result rendering, screenshots, or runtime settings.

Primary files:

```text
src/yolo26_detection/infrastructure/
src/yolo26_detection/application/detection_service.py
src/yolo26_detection/api/facade.py
src/vision_workbench/desktop/pages/yolo_detection_page.py
```

Model discovery should validate `.pt` files before showing them as usable. Keep camera resource coordination in the desktop layer and camera/model adapters in infrastructure.

## YOLO26 Segmentation Extensions

Scope: instance/semantic segmentation tasks, model routing, result rendering, downloads, or saving policies.

Primary files:

```text
src/yolo26_segmentation/infrastructure/
src/yolo26_segmentation/application/segmentation_service.py
src/yolo26_segmentation/api/facade.py
src/vision_workbench/desktop/pages/yolo_segmentation_page.py
```

Keep `-seg` and `-sem` model rules task-aware, and validate local `.pt` files before exposing them as usable models.

## YOLO26 Training Extensions

Scope: training tasks, parameters, dataset validation, pretrained weight rules, logs, and output management.

Primary files:

```text
src/yolo26_training/infrastructure/
src/yolo26_training/application/training_service.py
src/yolo26_training/runner.py
src/yolo26_training/api/facade.py
src/vision_workbench/desktop/pages/yolo_training_page.py
```

Keep dataset/model validation in infrastructure or the CLI runner. Keep command construction in the service layer. The Qt page should collect parameters, show logs, and launch the runner process.

## YOLO26 Model Discovery Notes

Project-local models, per-user models, bundled defaults, remote manifests, and local manifest caches should be merged through the registry/repository layer before reaching services and APIs. Qt dropdowns should read only the local merged result; startup refresh or the user-facing "refresh model list" action may update the remote manifest cache. The reason is that network access is unstable, and binding dropdown population directly to network calls can freeze the UI, break offline use, or create hard-to-reproduce failures.

Task filtering is a development invariant: detection models should not match `-seg` or `-sem` suffixes, instance segmentation should prefer `-seg`, and semantic segmentation should prefer `-sem`. User-facing instructions for copying trained `best.pt` files belong in the YOLO26 training module docs; this guide owns discovery order, validation policy, cache policy, and tests.

When adding a model source or task, update `configuration/settings.py`, the `infrastructure` registry/repository, the `application` refresh interface, `api/facade.py`, Qt refresh behavior, and matching tests together. Missing one layer commonly appears as "the file exists but the dropdown does not show it", which is expensive to debug.

## Qt Copy Notes

Copy that involves file picking and model selection must name the source clearly. Use "select local model file" or "select local weight file" for disk pickers, "select model" for existing dropdown choices, and "refresh model list" for rescanning or cache updates. Button copy is part of workflow semantics: vague labels make users confuse local file selection, dropdown selection, and remote/local cache refresh.

If UI copy carries workflow meaning, such as "left reference image / right image to stitch" in panorama reconstruction or "local" in model picker buttons, lock it in Qt UI tests so later refactors do not make the workflow ambiguous again.

## Documentation Ownership Notes

The root README should contain installation, launch, and the true quick start. Module READMEs should contain the user workflow for that module. Trained-model loading, dropdown discovery, and Python API notes belong under the YOLO26 training module. Troubleshooting docs should capture symptoms and fixes. This contributor guide should record structural constraints, root causes, and extension notes.

This split reduces duplicated guidance and conflicting updates. Structural constraints and coding requirements that the project must maintain belong in the official contributor or architecture documentation. Temporary investigations, detailed debugging records, and a maintainer's personal development notes may remain in a local knowledge base and are not public user instructions. Keep user actions in the relevant module documentation.

## Packaging Checklist

Before preparing a release or commit:

```bash
python -m compileall -q src
python scripts/check_markdown_links.py
python -m pytest -q
python scripts/cleanup_runtime.py
git status --short
```

Only `vision-workbench` is the public GUI entry point. Keep old `*/window/` code out of user-facing docs unless documenting legacy compatibility.
