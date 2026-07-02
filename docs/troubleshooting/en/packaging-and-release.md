# Packaging and Release Troubleshooting

[Index](./README.md) | [中文](../zh-CN/打包与发布.md)

This page covers wheel builds, `dist/`, release assets, and model files larger than 100 MB.

## Build Fails

Use:

```bash
python -m pip install build
python -m build
```

If isolated build dependencies cannot be downloaded, reuse the current environment:

```bash
python -m build --no-isolation
```

## dist Directory

`dist/` contains local build outputs. It is not intended to be committed to the source repository.

## Release Asset Check Failed

The release checker blocks large model files in Git:

```bash
python scripts/check_release_assets.py
```

Model files larger than 100 MB should be moved to local download instructions, Git LFS, or GitHub Release assets before publishing.

## Wheel Is Not a Complete Runtime

The base wheel installs Python packages and entry points. Deep-learning workflows still require the classification or YOLO26 dependency group, and large model weights may be distributed separately.
