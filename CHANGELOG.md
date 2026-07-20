# Changelog

All notable changes to Vision Workbench are documented here.

## [1.0.0] - 2026-07-19

### Added

- Added a first-class Qt version-information page with the running version, update date, installation mode, official repository link, non-blocking update checks, and an explicit one-click update action.
- Added runtime-aware version resolution for editable source, installed wheels, and frozen Windows executables.
- Added strict GitHub release metadata validation, bounded downloads, SHA-256 verification, and an out-of-process update helper with Windows EXE backup and rollback behavior.
- Added a stable update-manifest generator and CI version-contract gate for the 1.0 release line.
- Added a tagged release-draft workflow and a dependency-contract fingerprint that blocks unsafe no-dependency wheel upgrades.
- Added installed-wheel and frozen-EXE self-tests to the release pipeline, including internal wheel identity validation before Python updates and a Qt construction check before EXE replacement.
- Included the runtime dependency declarations in the source distribution so an unpacked sdist can resolve the same 1.0 identity contract.
- Standardized the Windows release asset as `Vision-Workbench-win-x64.exe`, preventing an in-place update from retaining an obsolete version in its filename.

### Changed

- Promoted Vision Workbench to the 1.0.0 stable release contract and removed hard-coded component version strings.
- Editable installations now report the checked-out source version; accepting an official update switches the environment to the verified release wheel without modifying the source checkout.
- Runtime identity now fails closed on mismatched bundled metadata, frozen builds ignore unrelated editable registrations, and local `datasets/` content is protected from accidental commits.
- Version-page actions now use stable responsive breakpoints with content-width safeguards, so Linux, macOS, and Windows reflow consistently without compressing button labels.
- Cross-platform and release CI now provisions the required Linux Qt runtime and treats unavailable Qt UI tests as a failure instead of silently skipping them.
- The Windows EXE builder now takes required release metadata and sample assets directly from the tagged source tree, eliminating dependence on an unrelated local editable installation.

## [0.4.0] - 2026-07-17

### Fixed

- Fixed a critical Windows high-DPI hit-testing bug that could classify a large area on the right side of a normal window as the resize border, preventing Qt controls there from receiving clicks.
- Kept native edge resizing, title-bar dragging, and title-bar controls working consistently by using Win32 native pixels for non-client hit testing and converting client positions to Qt device-independent coordinates.
- Release asset checks now ignore Git-ignored local model files, so a deliberately excluded local weight cannot block an otherwise publishable release.

### Tests

- Added Windows regression coverage for 125% display scaling, including an interior point previously misclassified as `HTRIGHT` and a real right-edge resize point.
- Added regression coverage for Git-ignored release assets.
- Verified the complete test suite and the affected controls in a real 125% DPI Windows window.

### Packaging

- Bumped Python package, source-download, citation, documentation, and Windows base-EXE metadata to `0.4.0`.

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

## [1.0.0] - 2026-07-19

### 新增

- 新增与其它工作流同级的 Qt“版本信息”页面，展示当前版本、更新时间、运行方式和官方仓库，并支持后台检查更新和用户确认后的一键更新。
- 新增按 editable 源码、wheel 安装和 Windows 单文件 EXE 分别解析的运行版本模型。
- 新增 GitHub Release 元数据严格校验、下载大小限制、SHA-256 复核以及进程外更新助手；Windows EXE 更新保留旧文件并在替换失败时回滚。
- 新增稳定更新清单生成器和 CI 版本契约门禁，作为 1.0 正式版本规范。
- 新增 tag 驱动的发布草稿工作流和运行依赖契约指纹，阻止不安全的无依赖 wheel 升级。
- 发布流程新增已安装 wheel 与冻结 EXE 自检；Python 更新前校验 wheel 内部身份，EXE 替换前执行 Qt 构造检查。
- sdist 现包含运行依赖声明，解压后的源码发布包也能解析与 1.0 一致的身份契约。
- Windows 正式资产统一使用 `Vision-Workbench-win-x64.exe`，避免原地更新后文件名仍残留旧版本号。

### 变更

- Vision Workbench 进入 1.0.0 正式版本契约，移除组件内分散的硬编码版本号。
- editable 安装现在展示当前源码版本；用户确认安装正式更新后，环境切换为经校验的正式 wheel，不修改源码仓库。
- 运行身份在内置元数据不一致时会直接失败；冻结构建忽略无关 editable 注册，并防止本地 `datasets/` 内容被误提交。
- 版本页面操作按钮采用稳定的响应式断点并保留内容宽度保护，使 Linux、macOS 与 Windows 在不压缩按钮文字的前提下保持一致换行。
- 跨平台与发布 CI 会安装 Linux Qt 所需运行库；Qt UI 测试不可用时直接失败，不再静默跳过。
- Windows EXE 构建器直接从带标签的源码树加入发布元数据和示例资源，不再依赖构建机器中无关的本地 editable 安装。

## [0.4.0] - 2026-07-17

### 修复

- 修复 Windows 高 DPI 下的重大命中测试缺陷：普通窗口右侧大片区域可能被误判为缩放边框，导致其中的 Qt 控件无法接收点击。
- 非客户区命中测试改用 Win32 原生像素，并将客户区位置转换为 Qt 设备无关坐标，从而同时保留窗口边缘缩放、标题栏拖动和标题栏按钮行为。
- 发布资产检查现在会跳过被 Git 忽略的本地模型，避免明确排除的本地权重阻塞原本可发布的版本。

### 测试

- 增加 Windows 125% 显示缩放回归测试，覆盖此前被误判为 `HTRIGHT` 的内部位置和真正的右边缘缩放位置。
- 增加被 Git 忽略发布资产的回归测试。
- 在真实 Windows 125% DPI 窗口中验证完整测试套件和受影响控件。

### 打包

- Python 包、源码下载、引用信息、文档和 Windows 基础 EXE 元数据统一升级为 `0.4.0`。

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
