# Legacy GUI Policy

[中文](./旧版界面维护策略.md) | [Documentation](./README.md) | [Back to README](../README.md)

The supported Vision Workbench GUI is the PySide6 / Qt desktop application launched with `vision-workbench`. New and maintained user-interface work belongs under `src/vision_workbench/desktop/` and its `pages/` directory.

## Purpose of the Legacy Windows

The `window/` directories inside feature packages preserve earlier Tkinter interfaces for compatibility with existing calls and for reference when reviewing historical implementations. They are not the recommended user entry points and do not represent the same supported feature set as the Qt application.

## Maintenance Boundary

- Add new interface features and interaction changes only to the unified Qt desktop application.
- Continue to expose module capabilities through `api/`, `application/`, `processing/`, or `infrastructure/`; do not implement a capability only in a legacy window.
- Do not add new public entry points under `window/` unless required for an explicit compatibility commitment.
- User documentation should not recommend launching legacy windows directly. Troubleshooting pages may mention them when explaining historical entry points.
- Record the affected entry points and supported replacement in `CHANGELOG.md` before removing legacy compatibility.

## Support Statement

Automated tests and release QA target the unified Qt desktop application. Legacy windows receive limited compatibility maintenance and are not expected to gain new capabilities or remain behaviorally identical to the Qt pages.
