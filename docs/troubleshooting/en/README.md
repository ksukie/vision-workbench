# Troubleshooting Guide

[Project README](../../../README.en.md) | [中文](../zh-CN/问题排查指南.md)

Use this guide when an install command, GUI popup, camera workflow, model load, dataset validation, or training run fails. Most user-visible errors in Vision Workbench include one of these document paths.

## Choose by Symptom

| If you see | Start here |
| --- | --- |
| Stale GUI processes, camera handles, training jobs, memory, or GPU resources after an abnormal exit | [Emergency cleanup](./emergency-cleanup.md) |
| `vision-workbench` not found, Python import errors, PySide6 startup issues, path or encoding problems | [Environment](./environment.md) |
| Torch, torchvision, CUDA, `nvidia-smi`, or `doctor` problems | [Deep-learning dependencies](./deep-learning-dependencies.md) |
| Missing `.pt`/`.pth` files, failed model downloads, incomplete downloads, checkpoint errors | [Models and weights](./models-and-weights.md) |
| Cannot open/save images, JSON point-pair files, checkpoints, or output files | [Data and files](./data-and-files.md) |
| Camera scan/open/read failures, screenshots, recording, device permissions | [Camera and video](./camera-and-video.md) |
| Classification datasets, YOLO `data.yaml`, labels, masks, training aborted | [Datasets and training](./datasets-and-training.md) |
| CV effects, panorama reconstruction, detection/segmentation runtime errors | [Module runtime errors](./module-runtime-errors.md) |
| Version mismatch, `pip show` is stale, update checks, one-click updates, wheel builds, or release assets | [Packaging and release](./packaging-and-release.md) |

## Fast Checks

Run these from the repository root unless a document says otherwise:

```bash
python --version
python -m pip --version
python -m pytest
python scripts/install_dependencies.py doctor
```

When reporting a problem, include the full popup text or terminal output, the command you ran, the document path printed by the error, and whether you installed the base, classification, or YOLO26 dependency group.
