# Security Policy

## Reporting a Vulnerability

Report suspected vulnerabilities through GitHub private vulnerability reporting:

https://github.com/ksukie/vision-workbench/security/advisories/new

Use public GitHub Issues for ordinary bugs, documentation problems, installation failures, model download failures, and feature requests. Do not include exploit details, private datasets, credentials, tokens, or sensitive environment information in public issues.

## Supported Versions

| Version | Security support |
| --- | --- |
| 0.2.x | Supported |
| 0.1.x | Best-effort maintenance |

## Scope

Security review covers the Vision Workbench Python packages, Qt desktop entry point, model download and validation helpers, training launchers, packaging metadata, and repository documentation. Third-party components, including PySide6, OpenCV, Torch, TorchVision, Ultralytics YOLO26 source code, CUDA packages, and downloaded model weights, remain governed by their upstream security processes.

## Model Trust Boundary

PyTorch model files are executable serialization formats, not passive media files. Vision Workbench enables restricted
PyTorch and Ultralytics loading by default, pins the size and full SHA-256 of bundled YOLO26 release assets, restricts
model download hosts and redirects, and applies download, archive, image, and dataset metadata limits. A cached manifest
cannot replace a pinned official asset unless its hash matches. These controls reduce risk but do not make an unknown checkpoint safe. Only load model files from
the bundled sources, official upstream releases, or a party you trust.

The environment variables `ULTRALYTICS_SAFE_LOAD=true` and `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1` are set automatically.
Do not disable them to open an untrusted or merely "scanned" checkpoint.

## Response Process

Reports are triaged by severity, reproducibility, affected versions, and exposure surface. Valid reports receive a fix plan before public disclosure. Releases that contain security fixes are documented in [CHANGELOG.md](./CHANGELOG.md).

## 安全策略

### 漏洞报告渠道

疑似安全漏洞通过 GitHub 私密安全通告提交：

https://github.com/ksukie/vision-workbench/security/advisories/new

普通缺陷、文档问题、安装失败、模型下载失败和功能建议使用公开 GitHub Issues。公开 issue 中不包含利用细节、私有数据集、凭据、令牌或敏感环境信息。

### 支持版本

| 版本 | 安全维护状态 |
| --- | --- |
| 0.2.x | 支持 |
| 0.1.x | 尽力维护 |

### 范围

安全维护范围包括 Vision Workbench Python 包、Qt 桌面入口、模型下载与校验辅助函数、训练启动器、打包元数据和仓库文档。PySide6、OpenCV、Torch、TorchVision、Ultralytics YOLO26 源码、CUDA 包和下载的模型权重遵循各自上游项目的安全流程。

### 模型信任边界

PyTorch 模型文件属于可执行序列化格式，不是普通媒体文件。Vision Workbench 默认启用受限的 PyTorch 与
Ultralytics 加载模式，在清单提供哈希时进行校验，并限制下载和解压体积。这些措施只能降低风险，不能把
未知 checkpoint 变成可信文件。请只加载仓库内置模型、上游官方发布模型或来自可信人员的模型。

内置 YOLO26 官方权重固定了文件大小和完整 SHA-256，并限制下载主机与重定向。缓存清单只有在哈希与内置值
一致时才能替换官方资源地址；图片、数据集元数据和模型归档也有资源上限。

程序会自动设置 `ULTRALYTICS_SAFE_LOAD=true` 和 `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`。不要为了打开来源不明或
仅经过扫描的 checkpoint 而关闭它们。

### 处理流程

漏洞报告按照严重程度、可复现性、影响版本和暴露面进行分级。确认有效的问题会先形成修复计划，再进行公开披露。包含安全修复的版本会记录在 [CHANGELOG.md](./CHANGELOG.md)。
