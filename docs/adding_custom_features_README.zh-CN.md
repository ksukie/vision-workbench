# 添加自定义功能 README

[English](./adding_custom_features_README.md) | [返回总 README](../README.zh-CN.md)

本文档说明 Vision Workbench 的二次开发方式、模块扩展位置、依赖管理规则和 wheel 打包流程。

这是贡献者文档。普通运行用户只需要按根目录 [快速开始](../README.zh-CN.md#快速开始) 统一配置一次环境；各模块 README 只回指这个入口，不重复要求用户重新建环境。

## 开发环境

只有在修改代码、运行测试或准备打包时才需要使用本节：

```bash
conda create -n vision-dev python=3.11 -y
conda activate vision-dev
cd path/to/vision-workbench
pip install -e .[test]
```

图像分类功能需要额外安装：

```bash
python scripts/install_dependencies.py classification
```

YOLO26 相关功能需要额外安装：

```bash
python scripts/install_dependencies.py yolo26
```

执行测试：

```bash
pytest
```

## 模块分层

各功能模块采用统一分层结构：

```text
api/             对外 API
application/     应用服务与业务流程
configuration/   路径、参数与默认配置
domain/          数据模型
infrastructure/  文件、相机、模型、第三方库等外部资源适配
processing/      核心算法或模型处理逻辑
window/          Tkinter 图形界面
```

推荐调用链：

```text
window -> application -> processing / infrastructure -> application -> window
```

扩展功能时，应优先将算法或模型逻辑放入 `processing/`，将外部资源适配放入 `infrastructure/`，将流程编排放入 `application/`，将用户可调用函数放入 `api/`。

## 依赖管理规则

基础依赖写入：

```text
requirements.txt
pyproject.toml
```

图像分类依赖写入：

```text
requirements-classification.txt
```

YOLO26 依赖写入：

```text
requirements-yolo26.txt
```

基础依赖应保持轻量。仅图像分类或 YOLO26 使用的深度学习依赖不应写入基础依赖文件。
运行安装时优先使用 `scripts/install_dependencies.py`，它可以根据机器自动选择 CUDA 或 CPU Torch。
如果用户手动安装 `requirements-*.txt`，建议安装后执行 `python scripts/install_dependencies.py doctor`，用于检查并按需修复 Torch 构建。

<a id="cv-basics-add"></a>
## 基础 CV 扩展

适用范围：新增图像处理效果、通道分析、直方图、形态学操作或几何变换。

主要文件：

```text
src/cv_basics/domain/models.py
src/cv_basics/processing/operations.py
src/cv_basics/api/facade.py
src/cv_basics/api/__init__.py
src/cv_basics/window/app.py
```

扩展步骤：

1. 在 `EffectName` 中定义效果名称。
2. 在 `ProcessingParams` 中添加所需参数。
3. 在 `operations.py` 中实现 operation 类。
4. 在 `build_default_registry()` 中注册 operation。
5. 在 `api/facade.py` 中添加公开函数。
6. 在 `api/__init__.py` 中导出函数。
7. 如需界面参数，修改 `window/app.py`。
8. 如需 JSON 配置，修改 `configuration/settings.py`。

<a id="panorama-add"></a>
## 全景重构扩展

适用范围：新增拼接算法、配准策略、点选策略、融合方法或裁剪方法。

主要文件：

```text
src/panorama_reconstruction/processing/sift_reconstructor.py
src/panorama_reconstruction/processing/manual_reconstructor.py
src/panorama_reconstruction/application/reconstruction_service.py
src/panorama_reconstruction/api/facade.py
src/panorama_reconstruction/window/app.py
```

扩展原则：

- 算法实现放入 `processing/`。
- 读图、保存和流程编排放入 `application/`。
- 对外函数放入 `api/`。
- 用户交互和参数控件放入 `window/`。

<a id="camera-add"></a>
## 相机诊断扩展

适用范围：新增摄像头后端、读取模式、曝光设置、分辨率策略、截图或录屏策略。

主要文件：

```text
src/camera_diagnostics/infrastructure/platform_detector.py
src/camera_diagnostics/infrastructure/camera_repository.py
src/camera_diagnostics/application/camera_service.py
src/camera_diagnostics/api/facade.py
src/camera_diagnostics/window/app.py
```

相机相关异常应在底层适配层捕获，并通过明确错误信息返回到界面。

<a id="image-classification-add"></a>
## 图像分类扩展

适用范围：新增分类 backbone、预训练权重管理规则、数据增强、分类指标、批量预测或模型导出。

主要文件：

```text
src/image_classification/configuration/settings.py
src/image_classification/infrastructure/pretrained_weights.py
src/image_classification/infrastructure/dataset_validator.py
src/image_classification/infrastructure/dataset_splitter.py
src/image_classification/infrastructure/model_repository.py
src/image_classification/processing/classifier_backend.py
src/image_classification/application/classification_service.py
src/image_classification/api/facade.py
src/image_classification/window/app.py
```

扩展说明：

- backbone 列表在 `configuration/settings.py` 中维护。
- 预训练权重检测、下载和离线导入在 `pretrained_weights.py` 中维护。
- 训练和预测逻辑在 `classifier_backend.py` 中维护。
- 分类依赖写入 `requirements-classification.txt`。

<a id="yolo26-detection-add"></a>
## YOLO26 目标检测扩展

适用范围：新增检测模型、模型目录、推理参数、类别过滤、结果保存或后处理逻辑。

主要文件：

```text
src/yolo26_detection/configuration/settings.py
src/yolo26_detection/infrastructure/model_registry.py
src/yolo26_detection/infrastructure/detector_backend.py
src/yolo26_detection/application/detection_service.py
src/yolo26_detection/api/facade.py
src/yolo26_detection/window/app.py
```

YOLO26 第三方源码保存在：

```text
third_party/yolo26_source/
```

业务代码通过后端适配层调用第三方源码，不应将第三方源码复制到业务模块中。

<a id="yolo26-segmentation-add"></a>
## YOLO26 分割扩展

适用范围：新增实例分割模型、语义分割模型、mask 保存、透明叠加、类别过滤或批量分割。

主要文件：

```text
src/yolo26_segmentation/configuration/settings.py
src/yolo26_segmentation/infrastructure/model_registry.py
src/yolo26_segmentation/infrastructure/segmentation_backend.py
src/yolo26_segmentation/application/segmentation_service.py
src/yolo26_segmentation/api/facade.py
src/yolo26_segmentation/window/app.py
```

模型命名建议：

```text
*-seg.pt  实例分割模型
*-sem.pt  语义分割模型
```

<a id="yolo26-training-add"></a>
## YOLO26 训练扩展

适用范围：新增训练参数、数据集检查规则、训练任务、日志保存或模型管理策略。

主要文件：

```text
src/yolo26_training/train.py
src/yolo26_training/runner.py
src/yolo26_training/infrastructure/dataset_validator.py
src/yolo26_training/infrastructure/training_backend.py
src/yolo26_training/application/training_service.py
src/yolo26_training/api/facade.py
src/yolo26_training/window/app.py
```

训练前必须执行数据集校验。数据集不符合要求时，应返回错误信息并终止训练。

<a id="build-wheel"></a>
## wheel 打包

打包前检查：

- GUI 入口可启动。
- 新增功能可运行。
- 自动化测试通过。
- 新增依赖已写入对应依赖文件。
- `pyproject.toml` 版本号已按需更新。
- 不需要交付的缓存文件已清理。
- 发布资源符合模型文件体积策略。

检查发布资源：

```bash
python scripts/check_release_assets.py
```

安装打包工具：

```bash
pip install build
```

执行打包：

```bash
python -m build
```

输出目录：

```text
dist/
```

新环境安装验证：

```bash
conda create -n vision-user python=3.11 -y
conda activate vision-user
pip install dist/your_package.whl
vision-workbench
```

这个新环境验证只用于确认 wheel 的包入口是否可用。它不代表 wheel 是完整离线运行环境；可选深度学习依赖组和大型模型权重仍然需要单独安装或分发。

wheel 包不应默认内置大型模型权重。模型权重建议作为项目资源或离线资源包单独交付。

当前项目发布策略见：[发布策略](./legal/release_policy.zh-CN.md)。
