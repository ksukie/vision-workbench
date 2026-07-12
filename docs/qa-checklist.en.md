# Pre-release QA Checklist

[中文](./qa-checklist.md) | [Documentation](./README.md) | [Back to README](../README.md)

Automated tests cover core logic, but cameras, GPU drivers, font rendering, and assistive technologies require validation on physical systems. Record at least one manual result for every official release. For each failed item, include the operating system, Python, Qt, Torch, and driver versions together with reproducible steps.

## Desktop and Accessibility

- Launch `vision-workbench` once on Windows, Ubuntu, and macOS, then open every page and confirm that the application remains stable.
- Complete navigation using only the keyboard: `Alt+1` through `Alt+7`, Tab/Shift+Tab, Space/Enter, `Ctrl+O`, `Ctrl+S`, and `Ctrl+Return`.
- Confirm that focus is visible, labels are associated with inputs, and the minimize, maximize, and close controls are keyboard accessible.
- Use at least one supported screen reader: Windows Narrator, macOS VoiceOver, or Linux Orca. Confirm that it reads the main navigation, training parameters, and button names.
- Check the 1040×680 minimum window and common widescreen sizes at 100%, 125%, 150%, and 200% scaling. Text must remain visible and essential controls must remain reachable.

## Data and Training

- Create the classification, detection, instance-segmentation, and semantic-segmentation sample datasets and confirm that validation succeeds.
- Complete at least one classification epoch and confirm updates for loss, validation accuracy, best accuracy, and the `best.pt` path.
- Stop one classification training run and one YOLO training run. Confirm that the interface becomes available again and no training process remains.
- Run the environment check on CPU, an available NVIDIA CUDA device, and Apple MPS where applicable. Review the recommended batch size and memory or disk warnings.
- Use a corrupt image, oversized image, missing label, invalid class index, and invalid run name. Confirm that each input is rejected with an understandable message.

## Models and Downloads

- Download one official YOLO26 weight and confirm that size and SHA-256 validation pass. Modify the cached file and confirm that the application rejects it.
- Verify that offline operation, an interrupted download, an untrusted redirect, and insufficient disk space do not leave a partial model marked as usable.
- Test custom models only with trusted checkpoints. Confirm that the interface and documentation continue to identify PyTorch model files as executable serialization formats rather than passive media files.

## Cameras and Outputs

- Discover, open, and close a camera. Test exclusive camera ownership between the camera diagnostics page and YOLO live detection.
- Save detection, segmentation, screenshot, and panorama results. Confirm that the files can be opened again and that the cleanup script can identify project processes after an abnormal exit.

## Automated Baseline

```bash
python -m compileall -q src tests scripts
python -m pytest -q
python scripts/check_markdown_links.py
```

CI should pass on Windows, Ubuntu, and macOS with Python 3.10 and 3.12. The security workflow should complete `pip-audit` and generate a CycloneDX SBOM. Review the audit results manually before release rather than relying only on a successful workflow status.
