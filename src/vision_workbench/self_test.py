"""Installation identity and optional Qt construction smoke test."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import TextIO

from .versioning import DISTRIBUTION_NAME, current_version_info, stable_version_tuple


def _emit(message: str, stream: TextIO | None) -> None:
    """Write a self-test message when the runtime has a console stream."""

    if stream is None:
        return
    try:
        print(message, file=stream)
    except (AttributeError, OSError, ValueError):
        pass


def _append_report(path: Path | None, messages: list[str]) -> None:
    """Persist diagnostics for a windowed executable that has no console."""

    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as stream:
            for message in messages:
                stream.write(f"{message}\n")
    except OSError:
        pass


def installation_errors(
    *,
    expected_version: str | None = None,
    expected_mode: str | None = None,
    check_qt: bool = False,
) -> list[str]:
    """Return failures without mutating the installation or contacting a network."""

    errors = []
    info = current_version_info()
    if expected_version is not None:
        stable_version_tuple(expected_version)
        if info.version != expected_version:
            errors.append(f"runtime version {info.version} != expected {expected_version}")
    if expected_mode is not None and info.install_mode != expected_mode:
        errors.append(f"install mode {info.install_mode} != expected {expected_mode}")

    if info.install_mode == "wheel":
        installed_version = metadata.version(DISTRIBUTION_NAME)
        if installed_version != info.version:
            errors.append(f"installed metadata {installed_version} != runtime {info.version}")

    if check_qt:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication

            from .desktop.main_window import MainWindow

            app = QApplication.instance() or QApplication([])
            window = MainWindow()
            version_page = window.pages.get("version")
            if version_page is None:
                errors.append("Qt shell does not contain the version page")
            elif version_page.version_info.version != info.version:
                errors.append("Qt version page does not match the runtime version")
            elif version_page.version_info.install_mode != info.install_mode:
                errors.append("Qt version page does not match the runtime installation mode")
            window.close()
            app.processEvents()
        except Exception as exc:
            errors.append(f"Qt construction failed: {type(exc).__name__}: {exc}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version")
    parser.add_argument("--expected-mode", choices=("editable", "wheel", "single-file"))
    parser.add_argument("--qt", action="store_true", help="also construct and close the main Qt window")
    parser.add_argument("--report", type=Path, help="append self-test diagnostics to this file")
    args = parser.parse_args(argv)

    try:
        errors = installation_errors(
            expected_version=args.expected_version,
            expected_mode=args.expected_mode,
            check_qt=args.qt,
        )
    except Exception as exc:
        errors = [f"{type(exc).__name__}: {exc}"]
    if errors:
        messages = [f"ERROR: {error}" for error in errors]
        for message in messages:
            _emit(message, sys.stderr)
        _append_report(args.report, messages)
        return 1
    info = current_version_info()
    message = f"Vision Workbench {info.version} ({info.install_mode}) self-test OK"
    _emit(message, sys.stdout)
    _append_report(args.report, [message])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
