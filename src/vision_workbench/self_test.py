"""Installation identity and optional Qt construction smoke test."""

from __future__ import annotations

import argparse
import os
from importlib import metadata

from .versioning import DISTRIBUTION_NAME, current_version_info, stable_version_tuple


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
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    info = current_version_info()
    print(f"Vision Workbench {info.version} ({info.install_mode}) self-test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
