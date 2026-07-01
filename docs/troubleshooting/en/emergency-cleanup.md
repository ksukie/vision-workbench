# Emergency Cleanup

[Index](./README.md) | [中文](../zh-CN/emergency-cleanup.zh-CN.md)

Use this page when an abnormal Vision Workbench exit leaves GUI processes, camera handles, training jobs, memory, or GPU resources allocated. Run the emergency cleanup script before restarting the project, or after standard troubleshooting fails, to restore a clean runtime environment. The script terminates stale project-related processes so the operating system or device driver can release cameras, file handles, memory, and GPU memory.

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

The script only targets processes whose command line contains one of these signals:

- A Vision Workbench entry point, such as `vision-workbench` or `yolo26-detection-workbench`.
- A known Vision Workbench module entry, such as `python -m yolo26_detection.window.app`.
- The current repository path.

It does not intentionally match unrelated Python processes. Still, review the dry-run list before using `--kill`.

## If Nothing Is Found

If the script says no Vision Workbench runtime processes were found, but the camera or GPU still looks busy, the resource may be held by another application, a driver, or the operating system. Check Task Manager, close camera apps, or reboot if the driver does not release the device.
