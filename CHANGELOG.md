# Changelog

All notable changes to Vision Workbench are documented here.

## [0.3.0] - 2026-07-13

### Security

- Restricted PyTorch and Ultralytics checkpoint loading is now enabled by default.
- Model downloads enforce HTTPS/local-file URLs, bounded sizes, optional manifest SHA-256 and size metadata, and archive expansion limits.
- Bundled YOLO26 v8.4.0 assets have pinned sizes and full SHA-256 values; download hosts, redirects, image inputs, dataset counts, and metadata sizes are bounded.
- Training run names are constrained to one safe path component so outputs cannot escape the configured run directory.
- Conda and virtual-environment launches ignore conflicting user-site packages unless explicitly opted back in.
- CI installs the base dependency graph from a hash-locked file, audits dependencies, and produces a CycloneDX SBOM.

### Added

- Cached model manifests can be refreshed through the detection, segmentation, and training APIs.
- Detection now exposes a single-image path API, and training can register the best weight from a completed run.
- Training pages provide deterministic sample datasets and environment diagnostics.
- Desktop workflows now provide a bundled sample image where they accept an input image; panorama reconstruction and training keep their existing sample controls.

### Changed

- YOLO and classification training pages now include beginner parameter guidance, accessible label associations, and safer disabled states before a dataset is selected.
- Classification prediction and training are separate tabs; classification training now runs in a stoppable process with epoch loss and accuracy metrics.
- Both training pages can create deterministic sample datasets, inspect the Torch/accelerator/disk environment, and apply a recommended batch size.
- Global navigation, title-bar controls, form labels, shortcuts, and focus styles have improved keyboard and assistive-technology metadata.
- Base dependencies are version-bounded, and CI covers Python 3.10 and 3.12 on Windows, macOS, and Linux.

### Compatibility

- Existing workflows that use untrusted model URLs, oversized inputs, unsafe run names, or user-site packages in an isolated environment may now be rejected by the default safety policy.

### Tests

- Added coverage for model hash, URL and size validation, safe run names, restricted loading defaults, and user-site isolation.
- Added sample-dataset, input-limit, trusted-host, manifest-pinning, runtime-diagnostic, training-command, and accessibility coverage.

### Packaging

- Built `vision_workbench-0.3.0-py3-none-any.whl`.
- Built `vision_workbench-0.3.0.tar.gz`.
- Added a base-only single-file Windows EXE build; deep-learning navigation directs users to the supported full-source install route.

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

## [0.3.0] - 2026-07-13

### 安全

- 默认启用受限的 PyTorch 与 Ultralytics checkpoint 加载。
- 模型下载限制为 HTTPS/本地文件 URL，并增加体积上限、可选清单 SHA-256/大小校验和归档解压上限。
- 内置 YOLO26 v8.4.0 权重固定文件大小和完整 SHA-256，并限制下载主机、重定向、图片输入、数据集数量与元数据大小。
- 训练运行名被限制为单个安全路径组件，输出不能越过配置的运行目录。
- Conda 和虚拟环境启动时默认忽略冲突的用户级包，并保留显式兼容开关。
- CI 使用带哈希的依赖锁文件安装基础环境，执行依赖审计并生成 CycloneDX SBOM。

### 新增

- 可通过检测、分割和训练 API 刷新本地缓存的模型清单。
- 目标检测新增按单张图像路径推理的 API；训练完成后可注册最佳权重。
- 训练页面新增确定性示例数据集与环境诊断。
- 所有接收输入图像的桌面流程均可加载内置示例图；全景重构和训练保留既有示例入口。

### 变更

- YOLO 与分类训练页面增加新手参数提示、可访问标签关联，并在未选择数据集时禁用训练操作。
- 分类预测和训练改为独立标签页；分类训练改用可停止的独立进程，并显示每轮损失与准确率。
- 两个训练页面均可创建确定性示例数据、检查 Torch/加速器/磁盘环境并应用推荐批量。
- 全局导航、标题栏控件、表单标签、快捷键与焦点样式增加键盘和辅助技术元数据。
- 基础依赖增加版本边界；CI 覆盖 Windows、macOS、Linux 上的 Python 3.10 与 3.12。

### 兼容性

- 默认安全策略现在可能拒绝不受信任的模型 URL、超限输入、不安全的运行名，以及隔离环境中的用户级包。

### 测试

- 增加模型哈希、URL、体积、安全运行名、受限加载默认值和用户包隔离测试。
- 增加示例数据、输入上限、可信主机、清单固定、环境诊断、训练命令和可访问性测试。

### 打包

- 已构建 `vision_workbench-0.3.0-py3-none-any.whl`。
- 已构建 `vision_workbench-0.3.0.tar.gz`。
- 新增仅含基础功能的单文件 Windows EXE 构建；深度学习入口会引导用户使用受支持的完整源码安装路线。

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
