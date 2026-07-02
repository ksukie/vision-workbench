"""Clean up leftover Vision Workbench runtime processes.

By default this script only lists matching processes. Pass --kill to terminate
them. Matching is intentionally scoped to Vision Workbench entry points,
modules, or this repository path to avoid killing unrelated Python jobs.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PID = os.getpid()

ENTRY_POINTS = (
    "vision-workbench",
)

MODULE_MARKERS = (
    "cv_basics.window.app",
    "panorama_reconstruction.window.app",
    "camera_diagnostics.window.app",
    "yolo26_detection.window.app",
    "yolo26_training.window.app",
    "yolo26_segmentation.window.app",
    "image_classification.window.app",
    "yolo26_training.runner",
    "image_classification.runner",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List or terminate leftover Vision Workbench runtime processes.",
    )
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Terminate matching processes. Without this flag the script only lists matches.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force-kill after graceful termination. On Windows this uses taskkill /F.",
    )
    parser.add_argument(
        "--include-current-tree",
        action="store_true",
        help="Also match this terminal's current process tree. Usually not needed.",
    )
    args = parser.parse_args(argv)

    processes = find_processes(include_current_tree=args.include_current_tree)
    if not processes:
        print("No Vision Workbench runtime processes found.")
        return 0

    print("Matched Vision Workbench runtime processes:")
    for pid, command in processes:
        print(f"- PID {pid}: {command}")

    if not args.kill:
        print("\nDry run only. Re-run with --kill to terminate these processes.")
        return 0

    for pid, _command in processes:
        terminate_process_tree(pid, force=args.force)
    print("\nCleanup requested.")
    return 0


def find_processes(include_current_tree: bool) -> List[Tuple[int, str]]:
    rows = windows_processes() if os.name == "nt" else posix_processes()
    matches = []
    current_tree = current_process_tree()
    for pid, command in rows:
        if pid == CURRENT_PID:
            continue
        if not include_current_tree and pid in current_tree:
            continue
        if is_vision_workbench_process(command):
            matches.append((pid, command))
    return sorted(matches)


def is_vision_workbench_process(command: str) -> bool:
    normalized = command.replace("\\", "/").lower()
    root_text = str(ROOT).replace("\\", "/").lower()
    if root_text in normalized:
        return True
    if any(entry in normalized for entry in ENTRY_POINTS):
        return True
    if any(module.lower() in normalized for module in MODULE_MARKERS):
        return True
    return False


def windows_processes() -> List[Tuple[int, str]]:
    code = (
        "$ErrorActionPreference = 'SilentlyContinue'; "
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    import json

    data = json.loads(result.stdout)
    if isinstance(data, dict):
        data = [data]
    rows = []
    for item in data:
        try:
            pid = int(item.get("ProcessId"))
        except (TypeError, ValueError):
            continue
        command = str(item.get("CommandLine") or "")
        if command:
            rows.append((pid, command))
    return rows


def posix_processes() -> List[Tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            rows.append((int(pid_text), command.strip()))
        except ValueError:
            continue
    return rows


def current_process_tree() -> set[int]:
    if os.name != "nt":
        return {CURRENT_PID, os.getppid()}
    rows = windows_parent_rows()
    children = {}
    for pid, parent_pid in rows:
        children.setdefault(parent_pid, set()).add(pid)
    tree = {CURRENT_PID}
    frontier = [CURRENT_PID]
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, ()):
            if child not in tree:
                tree.add(child)
                frontier.append(child)
    tree.add(os.getppid())
    return tree


def windows_parent_rows() -> List[Tuple[int, int]]:
    code = (
        "$ErrorActionPreference = 'SilentlyContinue'; "
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId | "
        "ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    import json

    data = json.loads(result.stdout)
    if isinstance(data, dict):
        data = [data]
    rows = []
    for item in data:
        try:
            rows.append((int(item.get("ProcessId")), int(item.get("ParentProcessId"))))
        except (TypeError, ValueError):
            continue
    return rows


def terminate_process_tree(pid: int, force: bool) -> None:
    if os.name == "nt":
        command = ["taskkill", "/T", "/PID", str(pid)]
        if force:
            command.insert(1, "/F")
        subprocess.run(command, check=False)
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if force:
        time.sleep(1.0)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
