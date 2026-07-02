# Emergency Cleanup

[Index](./README.md) | [中文](../zh-CN/应急清理.md)

Use this page when an abnormal Vision Workbench exit leaves GUI processes, camera handles, training jobs, memory, or GPU resources allocated. The cleanup script terminates stale project-related processes so the operating system or device driver can release cameras, file handles, memory, and GPU memory.

## Safe Workflow

Run commands from the repository root.

First list matched processes:

```bash
python scripts/cleanup_runtime.py
```

Review the dry-run list. If every listed process is a Vision Workbench process, terminate them:

```bash
python scripts/cleanup_runtime.py --kill
```

Use force mode only if normal termination does not release the device:

```bash
python scripts/cleanup_runtime.py --kill --force
```

## What It Matches

The script targets processes whose command line contains one of these signals:

- The public Vision Workbench GUI entry point: `vision-workbench`.
- Known manual module entries such as `python -m yolo26_detection.window.app`, retained for legacy compatibility.
- Training CLI entries such as `python -m yolo26_training.runner`.
- The current repository path.

It does not intentionally match unrelated Python processes. Still, review the dry-run list before using `--kill`.

## If Nothing Is Found

When the script reports no Vision Workbench runtime processes but the camera or GPU still looks busy, the resource may be held by another application, a driver, or the operating system. Check Task Manager, close camera apps, or reboot if the driver does not release the device.
