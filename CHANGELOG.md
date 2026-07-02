# Changelog

All notable changes to Vision Workbench are documented here.

## [0.2.0] - 2026-07-02

### Added

- Unified PySide6 desktop shell exposed through the `vision-workbench` GUI entry point.
- Native Qt pages for CV Basics, Panorama Reconstruction, Camera Diagnostics, YOLO26 Detection, YOLO26 Segmentation, YOLO26 Training, and Image Classification.
- Safe model download helpers that write to `.pt.part`, validate completed archives, and promote only complete model files.
- Model integrity checks for detection, segmentation, and YOLO training workflows.
- Task-aware YOLO training model discovery for detection, instance segmentation, and semantic segmentation.
- Chinese document filenames under `docs/modules/zh-CN/` and `docs/troubleshooting/zh-CN/`.
- Security policy and changelog documents.

### Changed

- Public GUI packaging now exposes only `vision-workbench`; training CLIs remain available as `yolo26-train` and `image-classification-train`.
- Project documentation now presents the Qt desktop workflow as the primary user path.
- README and module documents now point new users to the unified desktop entry point and optional dependency groups.
- Troubleshooting documents now emphasize PySide6 startup, model cache integrity, and emergency runtime cleanup.

### Fixed

- Incomplete or corrupt `.pt` files no longer appear as usable custom models.
- YOLO segment model lists exclude semantic weights, and semantic model lists exclude instance-segmentation weights.
- Stale `.pt.part` files are ignored by Git and do not mask missing model downloads.

### Packaging

- Built `vision_workbench-0.2.0-py3-none-any.whl`.
- Built `vision-workbench-0.2.0.tar.gz`.
- Existing `0.1.0` artifacts remain in `dist/`.

## [0.1.0] - 2026-06-30

### Added

- Initial Vision Workbench package with traditional CV, panorama reconstruction, camera diagnostics, image classification, YOLO26 detection, YOLO26 segmentation, and YOLO26 training modules.
- Early Tkinter module windows and module-specific GUI script entry points.
- Initial English and Chinese documentation, license files, third-party notices, and release packaging notes.

---

# 更新日志

Vision Workbench 的重要版本变动记录如下。

## [0.2.0] - 2026-07-02

### 新增

- 统一 PySide6 桌面主界面，对外 GUI 入口为 `vision-workbench`。
- 基础 CV、全景重构、相机诊断、YOLO26 目标检测、YOLO26 分割、YOLO26 训练和图像分类的原生 Qt 页面。
- 安全模型下载辅助函数：先写入 `.pt.part`，完成后校验归档，再提升为正式模型文件。
- 检测、分割和 YOLO 训练流程的模型完整性校验。
- YOLO 训练模型发现按目标检测、实例分割、语义分割任务过滤。
- `docs/modules/zh-CN/` 和 `docs/troubleshooting/zh-CN/` 下的中文文档文件名。
- 安全策略文档和更新日志。

### 变更

- 公开 GUI 打包入口只保留 `vision-workbench`；训练 CLI 保留 `yolo26-train` 和 `image-classification-train`。
- 项目文档以 Qt 桌面工作流作为主要使用路径。
- README 和模块文档统一指向桌面主入口和可选依赖组。
- 排查文档重点覆盖 PySide6 启动、模型缓存完整性和运行时应急清理。

### 修复

- 不完整或损坏的 `.pt` 文件不再作为可用自定义模型出现。
- YOLO 实例分割模型列表排除语义权重，语义分割模型列表排除实例分割权重。
- `.pt.part` 残留文件被 Git 忽略，也不会遮蔽未完成的模型下载状态。

### 打包

- 已构建 `vision_workbench-0.2.0-py3-none-any.whl`。
- 已构建 `vision-workbench-0.2.0.tar.gz`。
- `dist/` 中保留既有 `0.1.0` 构建产物。

## [0.1.0] - 2026-06-30

### 新增

- 初始 Vision Workbench 包，包含传统 CV、全景重构、相机诊断、图像分类、YOLO26 目标检测、YOLO26 分割和 YOLO26 训练模块。
- 早期 Tkinter 模块窗口和模块级 GUI 脚本入口。
- 初始中英文文档、许可文件、第三方引用说明和发布打包说明。
