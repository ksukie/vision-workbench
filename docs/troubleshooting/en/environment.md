# Environment Troubleshooting

[Index](./README.md) | [中文](../zh-CN/环境问题.md)

This page covers Python, conda, editable installs, the unified Qt entry point, PySide6 startup, paths, encodings, and base dependency installation.

## Command Not Found

Symptoms:

- `vision-workbench` is not recognized.
- The old module GUI commands such as `yolo26-detection-workbench` are not available.

Checks:

```bash
conda activate vision-workbench
python -m pip install -e .
python -m pip show vision-workbench
```

Vision Workbench exposes one public GUI command:

```bash
vision-workbench
```

When the entry point is still unavailable, run the module directly from the source tree:

```bash
python -m vision_workbench.desktop.app
```

## Import Errors

Symptoms:

- `ModuleNotFoundError`
- The GUI starts from the repository but cannot import a local package.

Checks:

```bash
cd path/to/vision-workbench
python -c "import cv_basics, vision_workbench; print('ok')"
python -m pip install -e .
```

Avoid running scripts from inside `src/` unless the script explicitly supports that mode.

## PySide6 Startup Problems

Symptoms:

- The GUI does not open.
- `PySide6` import errors.
- Qt platform plugin errors.
- The GUI command returns immediately with no visible window.

Use Python 3.10 or newer and install the base dependencies:

```bash
conda create -n vision-workbench python=3.11 -y
conda activate vision-workbench
python -m pip install -e .
```

If `PySide6` imports fail with a DLL load error in an Anaconda environment, avoid mixing a user-site PySide6 install with a base conda Python. Recreate the environment and install the project inside that active environment, or install PySide6 from conda-forge before running `vision-workbench`.

If `vision-workbench` returns immediately and no window appears, run the module form to see console errors:

```bash
python -m vision_workbench.desktop.app
```

The GUI launcher also writes startup failures to `%TEMP%\vision-workbench-startup.log`. Check that file when Windows hides the GUI-script traceback.

## Legacy Tkinter Modules

The old `*/window/` Tkinter modules remain in the source tree only as compatibility/reference code. They are not the public GUI entry points. If you manually run one of them and hit `_tkinter` errors, use a Python distribution with Tk/Tcl support, or switch back to the Qt entry point:

```bash
vision-workbench
```

## Paths and Encodings

Prefer absolute paths when diagnosing file problems. Paths with non-ASCII characters should first be retried from a shorter ASCII-only path such as `C:\work\vision-workbench` to separate path issues from model or dataset issues.

## Base Dependency Install Fails

Run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

When the mirror is unavailable, retry with the normal PyPI configuration:

```bash
python -m pip install -e . --index-url https://pypi.org/simple
```
