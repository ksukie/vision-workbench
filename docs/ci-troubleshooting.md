# Vision Workbench CI 故障记录与发布排查手册

[返回文档中心](./README.md) | [发布策略](./legal/发布策略.md) | [QA 检查清单](./qa-checklist.md)

本文记录截至 2026-07-20 在 Vision Workbench 持续集成和正式发布流程中实际遇到的问题、根因、修复方式与防复发规则。以后出现新的 CI 故障时，应在问题解决后补充到本文，而不是只保留在临时聊天或 Actions 日志中。

## 流水线边界

| 流水线 | 触发方式 | 主要职责 |
| --- | --- | --- |
| `CI` | 分支 push、Pull Request | 在 Windows、Ubuntu、macOS 与 Python 3.10/3.12 上验证 editable 源码、Qt 主界面、版本契约、完整测试和 Markdown 链接 |
| `Security` | 分支 push、Pull Request | 依赖审计、安全检查与软件物料清单 |
| `Prepare release draft` | 推送 `v*.*.*` 标签 | 从精确标签构建 wheel、sdist 和 Windows EXE，执行安装后自检，生成更新清单并创建未公开 Release 草稿 |

分支 CI 全绿只说明分支提交通过了常规检查，不代表任意标签都可以发布。正式发布还要求源码版本、附注标签、内置身份、产物名称和更新清单完全一致。

## 已发生问题

### 1. GitHub Actions 环境表达式使用位置不正确

- 现象：工作流在运行测试前就出现环境配置问题。
- 原因：曾在 job 级 `env` 中用 `runner.os` 动态生成 `QT_QPA_PLATFORM`；该上下文在该位置不可作为可靠的跨平台条件来源，非 Linux runner 还会得到无意义的空值。
- 修复：移除这条 job 级动态环境变量；Qt 自检和测试代码在需要时显式使用 `offscreen`。
- 防复发：操作系统条件放在 step 的 `if: runner.os == 'Linux'` 中；不要用空字符串模拟“未设置”的环境变量。
- 对应提交：`9d32fde`。

### 2. Python 3.10 安装阶段难以定位失败

- 现象：Python 3.10 矩阵在合并的依赖安装步骤中失败，日志无法快速区分锁定依赖、pytest 还是 editable 安装失败。
- 修复：将基础锁定依赖、pytest 和项目 editable 安装拆成独立步骤，使 Python 3.10/3.12 使用同一顺序并能准确定位失败阶段。
- 防复发：安装步骤保持单一职责；基础依赖继续使用 `requirements-base.lock` 与 `--require-hashes`。
- 对应提交：`e9a4830`。仓库本地历史不包含当时托管日志，因此不对更深层的外部安装错误作推测。

### 3. 测试修改全局 `os.name`，污染非 Windows 平台

- 现象：Windows 逻辑测试在 Ubuntu 或 macOS 上影响 `pathlib` 的路径类型选择，导致与被测更新逻辑无关的跨平台异常。
- 原因：测试直接把共享 `os` 模块的 `name` 改成 `nt`。同一进程中的其它标准库代码也会观察到这个修改。
- 修复：在更新助手中增加 `_is_windows()` 边界，测试只替换该项目函数，不再修改 Python 全局平台状态。
- 防复发：平台分支通过项目内的小函数或注入接口隔离；不要 monkeypatch `os.name`、`sys.platform` 等会影响标准库全局行为的值，除非测试进程完全隔离。
- 对应提交：`0d49d27`。

### 4. Python 启动器名称被测试写死

- 现象：Windows 测试通过，而 Ubuntu 和 macOS 的 Python 3.10/3.12 任务失败；日志中的首个参数可能是 `python3.10` 或 `python3.12`，测试却固定期待 `python`。
- 典型日志：`.../bin/python3.12 != .../bin/python`。
- 原因：可执行文件名称与路径由 runner 和 Python 安装方式决定，不能跨平台写死。
- 修复：测试替换 `_console_python_executable()` 并断言更新助手使用该返回值，验证语义而不是 runner 的文件命名习惯。
- 防复发：命令测试应比较已解析的解释器路径；不要自行拼接 `bin/python`、`Scripts/python.exe` 或版本后缀。
- 对应提交：`35b3cac`。

### 5. 正式标签不是附注标签，或 checkout 丢失标签对象

- 现象：发布门禁报告 `release tag v1.0.0 must be an annotated tag object`。
- 原因一：使用 `git tag v1.0.0` 创建了 lightweight tag，而发布契约要求带发布说明和独立 tag 对象的 annotated tag。
- 原因二：即使远端是附注标签，发布 workflow 若只检出解析后的提交而未显式保留 `${{ github.ref }}`，标签身份检查也可能失去正确上下文。
- 修复：所有发布 job 使用 `ref: ${{ github.ref }}` 和 `fetch-depth: 0`；发布标签统一通过 `git tag -a` 创建。
- 防复发：推送前执行 `git cat-file -t vX.Y.Z`，输出必须是 `tag`，不能是 `commit`。
- 对应提交：`ef66738`。

### 6. Linux 缺少 Qt/EGL 系统运行库

- 现象：PySide6 在 Ubuntu 导入或构造 Qt 界面时失败，wheel 冒烟测试出现 `ImportError: libEGL.so.1: cannot open shared object file`。
- 原因：pip 安装的 PySide6 仍依赖 Linux runner 提供 EGL 共享库；只安装 Python 包并不足以构造 Qt 应用。
- 修复：常规 CI 和 Python 发布任务在安装 Python 依赖前执行 `apt-get install --no-install-recommends --yes libegl1`。
- 防复发：不要删除 Linux Qt runtime 步骤；升级 runner 或 PySide6 后仍需在 Ubuntu 上执行真实 Qt 构造自检。
- 对应提交：`537ccb5`。

### 7. Qt 测试被跳过后产生“假绿”

- 现象：PySide6 或 QtWidgets 不可用时，测试模块通过 `pytest.skip` 跳过，CI 可能显示成功，但实际没有验证桌面界面。
- 原因：本地开发环境允许缺少可选 Qt 依赖时跳过的策略不适用于正式 CI。
- 修复：CI 与发布 workflow 设置 `VISION_WORKBENCH_REQUIRE_QT_TESTS=1`；在此模式下缺少 Qt 会直接失败。流水线还会执行 `vision_workbench.self_test --qt` 构造主窗口。
- 防复发：正式 CI 中 Qt 测试数量异常减少也应视为故障，不能仅看退出码。
- 对应提交：`537ccb5`、`ac0d1c6`。

### 8. Windows 路径示例触发 Python `SyntaxWarning`

- 现象：编译日志出现 `SyntaxWarning: invalid escape sequence '\i'`。
- 原因：普通 Python docstring 中写入了带反斜杠的 Windows 命令，形成无效转义序列。
- 修复：文档字符串中的命令改用正斜杠；CI 编译命令升级为 `python -W error -m compileall`，让此类警告立即失败。
- 防复发：Python 字符串中的 Windows 路径使用原始字符串、双反斜杠或正斜杠；不要忽略编译警告。
- 对应提交：`537ccb5`。

### 9. Qt 响应式布局断言受平台样式影响

- 现象：相同的窄窗口测试在 Windows、macOS 通过，但 Ubuntu 得到 `_action_columns == 2`，测试期待 `1`；日志还可能包含 `propagateSizeHints()` 或跨窗口 tab order 提示。
- 原因：按钮最小尺寸、字体度量、滚动条宽度和 offscreen Qt 插件行为随平台变化，仅根据 `minimumSizeHint()` 判断断点不稳定；控件重新布局后还需要重新建立 tab 顺序。
- 修复：加入固定的最小内容宽度断点，并与真实尺寸提示取较大值；每次重排后重建 tab order；测试强制执行布局更新。
- 防复发：跨平台 UI 测试验证可观察的布局规则和无压缩边界，不依赖某个平台偶然计算出的像素值。
- 对应提交：`ac0d1c6`。

### 10. Windows windowed EXE 失败时没有可见诊断

- 现象：Windows EXE 自检只有退出码 `1`，Actions 看不到真实异常。
- 原因：PyInstaller `--windowed` 构建没有可用控制台，写入 `stdout`/`stderr` 的诊断不会出现在日志中。
- 修复：自检支持 `--report` 文件；workflow 在失败时读取该文件并输出到 Actions 日志。
- 防复发：所有无控制台产物的自检必须提供显式报告路径，失败处理先输出报告再抛出错误。
- 对应提交：`537ccb5`。

### 11. 干净 runner 构建的 EXE 缺少内置发布信息

- 现象：冻结 EXE 自检报告 `RuntimeError: 缺少内置发布信息 release_info.json`。
- 原因：PyInstaller 的 `--collect-data vision_workbench` 会从构建环境中已安装的包查找数据，而发布 runner 构建的是标签源码；它可能收集不到数据，或误用另一个已安装版本的数据。
- 修复：构建脚本从当前标签源码树显式 `--add-data` 加入 `release_info.json`、主程序资源和全景示例资源；构建前验证这些路径存在，自检再验证内置资源。
- 防复发：正式产物不能依赖构建机器上恰好存在的 editable/旧 wheel；所有发布身份和资源都必须来自当前 checkout。
- 对应提交：`a2b41e9`。

### 12. 标签版本与源码版本不一致

- 现象：发布任务在 `Validate tagged source` 报告 `HEAD tag v1.0.1 != v1.0.0`，后续构建和草稿任务全部跳过。
- 原因：先把 `v1.0.1` 标签推到了仍声明 `1.0.0` 的提交。Git 标签不会自动修改 `pyproject.toml`，也不会自动生成新的内置发布信息。
- 正确修复：先在普通分支提交中将版本、日期、CHANGELOG、CITATION、README 产物示例、内置发布信息和当前版本测试同步为 `1.0.1`；分支 CI 全绿后，再将未公开且失败的错误标签替换到该提交。
- 防复发：严格执行“准备版本提交 → 分支 CI 全绿 → 创建附注标签 → 只推送标签”的顺序。失败的旧 run 不能通过 `Re-run jobs` 修复，因为重跑仍使用旧标签提交。

## 本地排查顺序

项目约定使用固定解释器，且不修改该环境：

```powershell
$python = 'D:\conda\envs\vision-workbench\python.exe'
& $python -W error -m compileall -q src tests scripts
& $python scripts/check_version_contract.py
& $python scripts/check_release_assets.py
& $python -m pytest -q
& $python scripts/check_markdown_links.py
git diff --check
```

判断失败阶段时按下面顺序处理：

1. `Validate tagged source` 失败：先查版本、日期、标签和工作区，不要先改打包器。
2. Qt 导入或主窗口构造失败：查系统库、`QT_QPA_PLATFORM` 和 Qt 是否被跳过。
3. 单一操作系统测试失败：查路径、解释器名称、平台 monkeypatch、字体和样式度量。
4. wheel 自检失败：查安装后的 distribution metadata、包内 `release_info.json` 和依赖契约。
5. EXE 自检失败：先读取报告文件，再查内置身份、资源路径和冻结运行模式。
6. `release-draft` 被跳过：先修复它依赖的 Python 或 Windows job；不要手工拼装不完整 Release。

## 发布前强制核对

在创建标签前执行：

```powershell
git status --short
Select-String -LiteralPath pyproject.toml -Pattern '^version\s*='
& 'D:\conda\envs\vision-workbench\python.exe' scripts/check_version_contract.py
```

`git status --short` 必须为空。确认版本提交已经推送且分支 CI 全绿后，再创建附注标签：

```powershell
git tag -a v1.0.1 -m "Vision Workbench 1.0.1"
git cat-file -t v1.0.1
git show --no-patch --format='%H %D' v1.0.1^{}
git push origin v1.0.1
```

`git cat-file` 必须输出 `tag`。标签 workflow 完成后只会生成未公开草稿；核对 wheel、sdist、`Vision-Workbench-win-x64.exe`、`update-manifest.json`、哈希、大小和内置版本后，才可手动公开。

## 错误标签恢复规则

只有同时满足以下条件，才允许删除并重建同名标签：

- 对应 GitHub Release 尚未公开；
- release draft 尚未创建或已经明确删除；
- 标签 workflow 在发布资产生成前失败；
- 团队确认没有用户依赖该标签。

满足条件时，先完成并推送正确的版本提交，再执行：

```powershell
git push origin :refs/tags/v1.0.1
git tag -d v1.0.1
git tag -a v1.0.1 -m "Vision Workbench 1.0.1"
git cat-file -t v1.0.1
git push origin v1.0.1
```

如果同名 Release 已经公开或资产已被用户获取，不得移动标签；应修复源码并发布更高的补丁版本，例如 `v1.0.2`。

## 禁止事项

- 不要在版本提交之前创建发布标签。
- 不要用 lightweight tag 代替 annotated tag。
- 不要重新运行一个指向错误提交的失败 tag workflow，并期待它读取 `main` 的新代码。
- 不要为了全绿而删除平台矩阵、关闭 Qt 强制检查或把失败测试改成 skip。
- 不要把 runner 特有的 Python 路径、字体像素或共享库位置写死在跨平台测试中。
- 不要提交 `dist/`，也不要将本地旧产物上传到新版本 Release。
- 不要让 EXE 或更新清单从构建环境中的旧安装读取版本身份。

