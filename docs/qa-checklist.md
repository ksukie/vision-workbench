# 发布前 QA 检查清单

[English](./qa-checklist.en.md) | [文档中心](./README.md) | [返回项目 README](../README.zh-CN.md)

自动化测试覆盖核心逻辑，但摄像头、GPU 驱动、字体渲染和辅助技术必须在真实设备上验证。每个正式版本至少记录一次下面的人工结果；失败项应附操作系统、Python、Qt、Torch、驱动版本和复现步骤。

## 桌面与可访问性

- Windows、Ubuntu 和 macOS 各启动一次 `vision-workbench`，逐页切换且无崩溃。
- 仅使用键盘完成导航：`Alt+1` 至 `Alt+8`、Tab/Shift+Tab、Space/Enter、`Ctrl+O`、`Ctrl+S`、`Ctrl+Return`。
- 检查所有焦点位置可见，标签与输入控件关联，窗口最小化、最大化、关闭按钮可被键盘访问。
- Windows Narrator、macOS VoiceOver 或 Linux Orca 至少选择一种，读取主导航、训练参数和按钮名称。
- 在 100%、125%、150%、200% 缩放下检查 1040×680 最小窗口和常用宽屏尺寸，无文本遮挡或关键按钮不可达。

## 数据与训练

- 分别创建分类、检测、实例分割和语义分割示例数据，确认校验通过。
- 分类训练至少完成 1 轮，确认损失、验证准确率、最佳准确率和 `best.pt` 路径更新。
- 分类和 YOLO 训练各执行一次“停止训练”，确认界面恢复可操作且没有遗留训练进程。
- 在 CPU、可用的 NVIDIA CUDA 设备和 Apple MPS 上分别运行环境检查；核对推荐批量与显存/磁盘提示。
- 使用损坏图片、超大图片、缺失标签、错误类别编号和非法运行名，确认被阻止且错误可理解。

## 模型与下载

- 下载一个官方 YOLO26 权重，确认大小和 SHA-256 校验通过；篡改缓存后确认模型被拒绝。
- 验证断网、下载中断、重定向到非信任主机和磁盘空间不足时不会留下可用的半成品模型。
- 仅使用可信 checkpoint 测试自定义模型；确认界面和文档仍明确提示 PyTorch 模型不是被动媒体文件。

## 摄像头与输出

- 枚举、打开、关闭摄像头，测试基础相机页和 YOLO 实时检测之间的独占切换。
- 保存检测、分割、截图和全景结果，确认文件可重新打开且异常退出后可由清理脚本识别残留进程。

## 版本与更新

- 分别确认 editable 源码、wheel 安装和 Windows 基础 EXE 展示的是实际运行代码所绑定的版本。editable 模式下应故意保留一次旧的 `pip show` 元数据，确认它不会覆盖当前源码身份；刷新 editable 注册后，两者必须一致。
- 在联网、离线、限流和连接中断条件下检查更新；查询失败绝不能显示为“已是最新版本”。
- 缺少兼容资产或 SHA-256 时，确认一键更新保持不可用。
- Python wheel 的运行依赖契约指纹缺失或变化时，确认一键更新保持不可用并引导手动安装；自包含 EXE 仍可按 EXE 流程更新。
- 完成一次有效更新全流程：下载、大小与 SHA-256 校验、退出 Qt、进程外安装、重启，并在版本页面看到新版本。
- 篡改暂存资产或让 wheel 内部元数据与文件名不一致，确认安装不会开始；在 Windows 上测试跨盘缓存更新、模拟 EXE 自检或替换失败并确认旧程序仍可使用。
- 将最终 wheel 安装到干净环境并执行 `python -m vision_workbench.self_test --expected-version <version> --expected-mode wheel --qt`；公开 Windows EXE 前执行 `--vision-workbench-self-test --expected-version <version> --qt`。

## 自动化基线

```bash
python -W error -m compileall -q src tests scripts
python scripts/check_version_contract.py
python -m pytest -q
python scripts/check_markdown_links.py
```

CI 应在 Windows、Ubuntu、macOS 与 Python 3.10/3.12 上通过。CI 和发布门禁中的 Qt 探测或 UI 测试不得因运行库缺失而静默跳过；无法构造 Qt 主界面必须直接判定失败。安全工作流应完成 `pip-audit` 并生成 CycloneDX SBOM；发布前人工确认审计结果，不应仅依赖工作流绿色状态。
