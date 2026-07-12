# Packaging and Release Troubleshooting

[Index](./README.md) | [Release Policy](../../legal/release_policy.md) | [中文](../zh-CN/打包与发布.md)

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

The base wheel installs the first-party Python packages, required resources, and entry points, but it is not a complete copy of the Git repository. Tests, development scripts, the complete vendored source tree, large model weights, and optional deep-learning dependencies may be excluded.

Use the matching Git tag or source archive when complete project source is required. For models, follow the relevant module documentation and use the built-in download action, a local model directory, or GitHub Release Assets.

A Python sdist is a source distribution for building the base Python package and should not be treated as the complete Git repository source. See the [Release Policy](../../legal/release_policy.md) for the official artifact boundaries.
