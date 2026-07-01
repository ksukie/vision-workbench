# Environment Troubleshooting

[Index](./README.md) | [中文](../zh-CN/environment.zh-CN.md)

This page covers Python, conda, editable installs, command entry points, Tkinter startup, paths, encodings, and base dependency installation.

## Command Not Found

Symptoms:

- `vision-workbench` is not recognized.
- `image-classification-workbench` or `yolo26-detection-workbench` is not recognized.

Checks:

```bash
conda activate vision-workbench
python -m pip install -e .
python -m pip show vision-workbench
```

If the entry point still is not found, run the module directly from the source tree:

```bash
python -m cv_basics.window.app
```

## Import Errors

Symptoms:

- `ModuleNotFoundError`
- A GUI starts from the repository but cannot import a local package.

Checks:

```bash
cd path/to/vision-workbench
python -c "import cv_basics, vision_workbench; print('ok')"
python -m pip install -e .
```

Avoid running scripts from inside `src/` unless the script explicitly supports that mode.

## Tkinter Startup Problems

Symptoms:

- The GUI does not open.
- `_tkinter` import errors.

Use a Python distribution that includes Tkinter. Conda Python on Windows usually includes it. If you use a minimal Python build, install Tk/Tcl support or create a fresh conda environment:

```bash
conda create -n vision-workbench python=3.11 -y
conda activate vision-workbench
python -m pip install -e .
```

## Paths and Encodings

Prefer absolute paths when diagnosing file problems. If a path contains non-ASCII characters, first retry from a shorter ASCII-only path such as `C:\work\vision-workbench` to separate path issues from model or dataset issues.

## Base Dependency Install Fails

Run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

If the mirror is unavailable, retry with your normal PyPI configuration:

```bash
python -m pip install -e . --index-url https://pypi.org/simple
```
