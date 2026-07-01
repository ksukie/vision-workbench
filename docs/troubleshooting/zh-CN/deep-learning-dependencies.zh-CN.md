# 深度学习依赖排查

[索引](./README.zh-CN.md) | [English](../en/deep-learning-dependencies.md)

本页覆盖 Torch、torchvision、CUDA/CPU wheel、NVIDIA 检测和 YOLO26 导入问题。

## 推荐安装命令

仅图像分类：

```bash
python scripts/install_dependencies.py classification
```

YOLO26 检测、分割和训练：

```bash
python scripts/install_dependencies.py yolo26
```

检查当前 Torch 环境：

```bash
python scripts/install_dependencies.py doctor
```

## Torch 或 TorchVision 导入失败

执行：

```bash
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__)"
```

如果导入失败，执行：

```bash
python scripts/install_dependencies.py doctor
```

当 doctor 提示当前构建和机器不匹配时，按提示允许重装。

## CUDA 不可用

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

如果 `nvidia-smi` 不存在或看不到 GPU，先更新 NVIDIA 驱动。如果 `nvidia-smi` 正常但 Torch 返回 `False`，重新安装 CUDA 版：

```bash
python scripts/install_dependencies.py yolo26 --torch cuda
```

平台不支持 CUDA wheel 时，显式使用 CPU：

```bash
python scripts/install_dependencies.py yolo26 --torch cpu
```

## YOLO26 导入失败

常见表现：

- `No module named ultralytics`
- YOLO26 训练脚本提示无法导入运行环境。

请在仓库根目录安装 YOLO26 依赖组：

```bash
python scripts/install_dependencies.py yolo26
```

## 安装源失败

安装脚本会先尝试 PyTorch 官方 wheel 源，再尝试镜像源。网络失败可能只是临时问题。可以换稳定网络重试，或根据 `doctor` 输出手动安装对应 Torch 构建。
