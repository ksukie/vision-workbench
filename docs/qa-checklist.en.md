# Pre-release QA Checklist

[中文](./qa-checklist.md) | [Documentation](./README.md) | [Back to README](../README.md)

Automated tests cover core logic, but cameras, GPU drivers, font rendering, and assistive technologies require validation on physical systems. Record at least one manual result for every official release. For each failed item, include the operating system, Python, Qt, Torch, and driver versions together with reproducible steps.

## Desktop and Accessibility

- Launch `vision-workbench` once on Windows, Ubuntu, and macOS, then open every page and confirm that the application remains stable.
- Complete navigation using only the keyboard: `Alt+1` through `Alt+8`, Tab/Shift+Tab, Space/Enter, `Ctrl+O`, `Ctrl+S`, and `Ctrl+Return`.
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

## Version and Updates

- Confirm that editable source, an installed wheel, and the Windows base EXE each show the version bound to the code actually running. In editable mode, deliberately stale `pip show` metadata must not override the live source identity; after refreshing the editable registration, both values must match.
- Check updates while online, offline, rate-limited, and behind an interrupted connection. A failed query must never be reported as up to date.
- Confirm that an available update remains disabled when its compatible asset or SHA-256 digest is missing.
- When a Python wheel dependency-contract fingerprint is missing or changed, confirm that one-click update remains unavailable and directs the user to manual installation; the self-contained EXE should still follow its EXE update path.
- Verify one valid update end to end: download, size and SHA-256 checks, Qt exit, external installation, restart, and the new version shown on the version page.
- Corrupt a staged asset or give a wheel mismatched internal metadata and confirm that installation does not start. On Windows, test a cross-volume cache update, force an EXE self-test or replacement failure, and confirm that the previous executable remains usable.
- Install the final wheel into a clean environment and run `python -m vision_workbench.self_test --expected-version <version> --expected-mode wheel --qt`. Run the frozen EXE with `--vision-workbench-self-test --expected-version <version> --qt` before publishing it.

## Automated Baseline

```bash
python -m compileall -q src tests scripts
python scripts/check_version_contract.py
python -m pytest -q
python scripts/check_markdown_links.py
```

CI should pass on Windows, Ubuntu, and macOS with Python 3.10 and 3.12. The security workflow should complete `pip-audit` and generate a CycloneDX SBOM. Review the audit results manually before release rather than relying only on a successful workflow status.
