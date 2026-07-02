# Deep-learning Dependency Troubleshooting

[Index](./README.md) | [中文](../zh-CN/深度学习依赖.md)

This page covers Torch, torchvision, CUDA/CPU wheels, NVIDIA detection, and YOLO26 import problems.

## Recommended Install Commands

Classification only:

```bash
python scripts/install_dependencies.py classification
```

YOLO26 detection, segmentation, and training:

```bash
python scripts/install_dependencies.py yolo26
```

Check the current Torch environment:

```bash
python scripts/install_dependencies.py doctor
```

## Torch or TorchVision Import Error

Run:

```bash
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__)"
```

If import fails, run the doctor:

```bash
python scripts/install_dependencies.py doctor
```

Accept the reinstall prompt if the expected build does not match the current machine.

## CUDA Is Not Available

Checks:

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `nvidia-smi` is missing or cannot see a GPU, update the NVIDIA driver first. If `nvidia-smi` works but Torch reports `False`, reinstall with:

```bash
python scripts/install_dependencies.py yolo26 --torch cuda
```

Use CPU explicitly when CUDA wheels are not supported on the platform:

```bash
python scripts/install_dependencies.py yolo26 --torch cpu
```

## YOLO26 Import Error

Symptoms:

- `No module named ultralytics`
- The YOLO26 training script says it cannot import the runtime.

Install the YOLO26 dependency group from the repository root:

```bash
python scripts/install_dependencies.py yolo26
```

## Install Source Fails

The helper tries the official PyTorch wheel index first, then a mirror for tagged builds. Network failures can be temporary. Retry on a stable network, or manually install the Torch build recommended by the doctor output.
