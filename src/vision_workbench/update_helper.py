"""Out-of-process update application and restart helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .update_installer import UpdatePreparationError, validate_python_wheel
from .versioning import REPOSITORY_URL, current_version_info, stable_version_tuple


WAIT_TIMEOUT_SECONDS = 120
SELF_TEST_TIMEOUT_SECONDS = 60


class UpdateApplyError(RuntimeError):
    """Raised when a prepared update cannot be applied safely."""


def _is_windows() -> bool:
    """Return whether the update helper is running on Windows."""

    return os.name == "nt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a prepared Vision Workbench update.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    args = parser.parse_args(argv)

    plan_path = Path(args.plan).resolve()
    log_path = plan_path.parent / "update.log"
    try:
        plan = _read_plan(plan_path)
        _wait_for_process_exit(args.parent_pid)
        _validate_running_installation(plan)
        _verify_plan_asset(plan, plan_path)
        if plan["mode"] == "single-file":
            _apply_single_file(plan, log_path)
        else:
            _apply_python_wheel(plan, log_path)
    except Exception as exc:
        _append_log(log_path, f"ERROR: {type(exc).__name__}: {exc}")
        _report_failure(exc, log_path)
        return 1
    return 0


def _read_plan(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpdateApplyError("无法读取更新计划。") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise UpdateApplyError("不支持的更新计划格式。")
    if payload.get("mode") not in {"editable", "wheel", "single-file"}:
        raise UpdateApplyError("更新计划包含未知安装模式。")
    _verify_plan_asset(payload, path)
    _validate_plan(payload, path)
    return payload


def _verify_plan_asset(payload: dict[str, object], plan_path: Path) -> None:
    asset_path = Path(_required_text(payload, "asset_path")).resolve()
    if asset_path.parent != plan_path.parent:
        raise UpdateApplyError("更新资产不在受控缓存目录中。")
    expected_hash = _required_text(payload, "asset_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise UpdateApplyError("更新计划中的 SHA-256 无效。")
    if _file_sha256(asset_path) != expected_hash:
        raise UpdateApplyError("应用更新前的 SHA-256 复核失败。")
    if payload.get("asset_kind") == "python-wheel":
        try:
            validate_python_wheel(
                asset_path,
                _required_text(payload, "target_version"),
                _required_text(payload, "target_dependency_contract_sha256"),
            )
        except UpdatePreparationError as exc:
            raise UpdateApplyError(str(exc)) from exc


def _validate_plan(payload: dict[str, object], plan_path: Path) -> None:
    mode = _required_text(payload, "mode")
    current_version = _required_text(payload, "current_version")
    target_version = _required_text(payload, "target_version")
    try:
        current_tuple = stable_version_tuple(current_version)
        target_tuple = stable_version_tuple(target_version)
    except ValueError as exc:
        raise UpdateApplyError(str(exc)) from exc
    if target_tuple <= current_tuple:
        raise UpdateApplyError("更新计划的目标版本必须高于当前版本。")

    expected_kind = "windows-x64-exe" if mode == "single-file" else "python-wheel"
    if payload.get("asset_kind") != expected_kind:
        raise UpdateApplyError("更新计划的安装模式与资产类型不一致。")
    expected_name = (
        "Vision-Workbench-win-x64.exe"
        if mode == "single-file"
        else f"vision_workbench-{target_version}-py3-none-any.whl"
    )
    asset_path = Path(_required_text(payload, "asset_path")).resolve()
    if asset_path.name != expected_name or asset_path.parent != plan_path.parent:
        raise UpdateApplyError("更新计划的资产名称或位置无效。")
    if _required_text(payload, "release_url") != (
        f"{REPOSITORY_URL}/releases/tag/v{target_version}"
    ):
        raise UpdateApplyError("更新计划的发布页面与目标版本不一致。")

    application = Path(_required_text(payload, "application_executable")).resolve()
    if not application.is_file():
        raise UpdateApplyError("更新计划中的应用程序不存在。")
    if mode == "single-file":
        if application.suffix.lower() != ".exe":
            raise UpdateApplyError("单文件更新目标不是 Windows EXE。")
        return

    python = Path(_required_text(payload, "python_executable")).resolve()
    if not python.is_file():
        raise UpdateApplyError("更新计划中的 Python 解释器不存在。")
    if python != Path(sys.executable).resolve():
        raise UpdateApplyError("更新计划中的 Python 解释器不是当前更新助手环境。")
    if application.parent != python.parent:
        raise UpdateApplyError("更新计划中的应用启动器不属于当前 Python 环境。")
    current_contract = _required_text(payload, "current_dependency_contract_sha256")
    target_contract = _required_text(payload, "target_dependency_contract_sha256")
    if (
        re.fullmatch(r"[0-9a-f]{64}", current_contract) is None
        or re.fullmatch(r"[0-9a-f]{64}", target_contract) is None
        or current_contract != target_contract
    ):
        raise UpdateApplyError("Python 更新计划的运行依赖契约不兼容。")
    source_root = payload.get("source_root")
    if mode == "editable":
        if not isinstance(source_root, str) or not (Path(source_root).resolve() / "pyproject.toml").is_file():
            raise UpdateApplyError("editable 更新计划缺少有效源码目录。")
    elif source_root is not None:
        raise UpdateApplyError("wheel 更新计划不应包含 editable 源码目录。")


def _validate_running_installation(plan: dict[str, object]) -> None:
    current_version_info.cache_clear()
    info = current_version_info()
    if info.install_mode != plan["mode"] or info.version != plan["current_version"]:
        raise UpdateApplyError("等待退出期间当前安装身份已变化，已取消更新。")
    if info.install_mode != "single-file":
        expected_contract = _required_text(plan, "current_dependency_contract_sha256")
        if info.dependency_contract_sha256 != expected_contract:
            raise UpdateApplyError("等待退出期间运行依赖契约已变化，已取消更新。")
    if info.install_mode == "editable":
        planned_root = Path(_required_text(plan, "source_root")).resolve()
        if info.source_root is None or info.source_root.resolve() != planned_root:
            raise UpdateApplyError("等待退出期间 editable 源码目录已变化，已取消更新。")


def _apply_python_wheel(plan: dict[str, object], log_path: Path) -> None:
    if plan.get("asset_kind") != "python-wheel":
        raise UpdateApplyError("Python 安装模式没有匹配的 wheel 资产。")
    python = Path(_required_text(plan, "python_executable"))
    application_python = Path(_required_text(plan, "application_executable"))
    wheel = Path(_required_text(plan, "asset_path"))
    target_version = _required_text(plan, "target_version")
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--force-reinstall",
        str(wheel),
    ]
    _append_log(log_path, f"Installing {wheel.name} with {python}")
    return_code = _run_logged(command, log_path)
    if return_code != 0:
        restore_code = _restore_editable(plan, python, log_path)
        if restore_code in {None, 0}:
            _restart_python_application(application_python, log_path)
        if restore_code not in {None, 0}:
            raise UpdateApplyError(
                f"pip 安装失败（退出码 {return_code}），恢复 editable 注册也失败（退出码 {restore_code}）。"
            )
        raise UpdateApplyError(f"pip 安装失败，退出码 {return_code}。")

    probe = (
        "import importlib.metadata as m; "
        "from vision_workbench.versioning import current_version_info; "
        "i=current_version_info(); "
        "raise SystemExit(0 if (m.version('vision-workbench')==i.version=="
        "__import__('sys').argv[1] and i.install_mode=='wheel') else 1)"
    )
    probe_code = _run_logged([str(python), "-c", probe, target_version], log_path)
    if probe_code != 0:
        restore_code = _restore_editable(plan, python, log_path)
        if restore_code == 0:
            _restart_python_application(application_python, log_path)
        raise UpdateApplyError("pip 返回成功，但安装后的版本身份复核失败。")
    _append_log(log_path, "Install succeeded; restarting the application.")
    _restart_python_application(application_python, log_path)


def _restore_editable(plan: dict[str, object], python: Path, log_path: Path) -> int | None:
    source_root = plan.get("source_root")
    if not isinstance(source_root, str) or not source_root:
        return None
    _append_log(log_path, "Restoring the previous editable source registration.")
    return _run_logged(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            "--editable",
            source_root,
        ],
        log_path,
    )


def _apply_single_file(plan: dict[str, object], log_path: Path) -> None:
    if not _is_windows() or plan.get("asset_kind") != "windows-x64-exe":
        raise UpdateApplyError("单文件更新仅支持 Windows EXE。")
    current = Path(_required_text(plan, "application_executable")).resolve()
    downloaded = Path(_required_text(plan, "asset_path")).resolve()
    expected_hash = _required_text(plan, "asset_sha256")
    target_version = _required_text(plan, "target_version")
    backup = current.with_name(f"{current.stem}.previous{current.suffix}")
    staged = current.with_name(f".{current.stem}.update{current.suffix}")
    failed = current.with_name(f".{current.name}.failed")
    _append_log(log_path, f"Replacing {current} with {downloaded}")
    staged.unlink(missing_ok=True)
    failed.unlink(missing_ok=True)
    try:
        shutil.copy2(downloaded, staged)
        if _file_sha256(staged) != expected_hash:
            raise UpdateApplyError("同目录暂存 EXE 的 SHA-256 复核失败。")
        _self_test_executable(staged, target_version, log_path)
        backup.unlink(missing_ok=True)
        _replace_file_with_backup(current, staged, backup)
    except OSError as exc:
        raise UpdateApplyError(f"替换单文件 EXE 失败：{exc}") from exc
    finally:
        staged.unlink(missing_ok=True)
    try:
        subprocess.Popen([str(current)], close_fds=True)
    except OSError as exc:
        try:
            _replace_file_with_backup(current, backup, failed)
            failed.unlink(missing_ok=True)
        except OSError as rollback_exc:
            raise UpdateApplyError(
                f"新版本无法启动，自动回滚也失败；旧版本备份位于 {backup}：{rollback_exc}"
            ) from rollback_exc
        try:
            subprocess.Popen([str(current)], close_fds=True)
        except OSError as restart_exc:
            raise UpdateApplyError(f"已回滚旧版本，但无法自动重新启动：{restart_exc}") from restart_exc
        raise UpdateApplyError(f"新版本无法启动，已回滚：{exc}") from exc
    _append_log(log_path, f"Started {current}; previous executable kept at {backup}")


def _self_test_executable(executable: Path, expected_version: str, log_path: Path) -> None:
    command = [
        str(executable),
        "--vision-workbench-self-test",
        "--expected-version",
        expected_version,
        "--qt",
        "--report",
        str(log_path),
    ]
    _append_log(log_path, f"Self-testing {executable.name} before replacement.")
    try:
        with log_path.open("a", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=SELF_TEST_TIMEOUT_SECONDS,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UpdateApplyError(f"新版本 EXE 自检无法完成：{exc}") from exc
    if completed.returncode != 0:
        raise UpdateApplyError(f"新版本 EXE 自检失败，退出码 {completed.returncode}。")


def _replace_file_with_backup(current: Path, replacement: Path, backup: Path) -> None:
    """Atomically replace a Windows file while preserving the previous file."""

    import ctypes
    import ctypes.wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    replace_file = kernel32.ReplaceFileW
    replace_file.argtypes = (
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    replace_file.restype = ctypes.wintypes.BOOL
    if not replace_file(str(current), str(replacement), str(backup), 0, None, None):
        raise ctypes.WinError(ctypes.get_last_error())


def _wait_for_process_exit(pid: int) -> None:
    deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
    while _process_exists(pid):
        if time.monotonic() >= deadline:
            raise UpdateApplyError("等待主程序退出超时。")
        time.sleep(0.2)


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if _is_windows():
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)
        if not process:
            return False
        ctypes.windll.kernel32.CloseHandle(process)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_logged(command: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            env=_update_subprocess_environment(),
        )
    return completed.returncode


def _restart_python_application(python: Path, log_path: Path) -> None:
    try:
        subprocess.Popen(
            [str(python), "-m", "vision_workbench.desktop.app"],
            close_fds=True,
            env=_update_subprocess_environment(),
        )
    except OSError as exc:
        _append_log(log_path, f"Restart failed: {exc}")
        raise UpdateApplyError(f"更新完成但重新启动失败：{exc}") from exc


def _update_subprocess_environment() -> dict[str, str]:
    blocked = {"PIP_PREFIX", "PIP_TARGET", "PIP_USER", "PYTHONHOME", "PYTHONPATH"}
    environment = {key: value for key, value in os.environ.items() if key.upper() not in blocked}
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def _report_failure(exc: Exception, log_path: Path) -> None:
    message = f"Vision Workbench 更新失败。\n\n{exc}\n\n详细日志：{log_path}"
    if _is_windows():
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "Vision Workbench 更新失败", 0x10)
            return
        except Exception:
            pass
    try:
        print(message, file=sys.stderr)
    except Exception:
        pass


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise UpdateApplyError(f"更新计划缺少 {key}。")
    return value.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise UpdateApplyError(f"无法读取更新资产：{exc}") from exc
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
