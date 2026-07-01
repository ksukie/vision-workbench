# Datasets and Training Troubleshooting

[Index](./README.md) | [中文](../zh-CN/datasets-and-training.zh-CN.md)

This page covers classification datasets, YOLO `data.yaml`, label and mask formats, validation failures, training startup, and missing output weights.

## Classification Dataset Invalid

Expected layout:

```text
dataset/
  train/
    class_a/
    class_b/
  val/
    class_a/
    class_b/
```

The `train/` and `val/` class folders must match. Each class folder must contain at least one supported image.

## YOLO data.yaml Problems

Expected fields:

```yaml
path: path/to/dataset
train: images/train
val: images/val
names: [class_a, class_b]
```

The project supports local YOLO-style datasets. Auto-download datasets and COCO JSON annotations are not supported by the basic trainer.

## Missing Labels

For detection and instance segmentation, labels should mirror image paths:

```text
images/train/001.jpg
labels/train/001.txt
```

Detection labels use:

```text
class x_center y_center width height
```

Segmentation labels use:

```text
class x1 y1 x2 y2 x3 y3 ...
```

All coordinates must be normalized to `0..1`.

## Semantic Masks

For semantic training with `masks_dir`, masks should mirror image paths and use lossless formats such as PNG or TIF.

## Training Failed to Start

Check:

- The selected model file exists.
- Epochs, image size, batch, and worker values are positive.
- The output directory under `runs/` is writable.
- Deep-learning dependencies are installed.

## best.pt Not Found

If training finishes but `best.pt` is missing, check the training log for upstream YOLO errors. The run may have stopped before producing weights.
