# Models and Weights Troubleshooting

[Index](./README.md) | [中文](../zh-CN/models-and-weights.zh-CN.md)

This page covers YOLO26 `.pt` files, classification pretrained weights, local imports, downloads, and checkpoints.

## Model File Not Found

Symptoms:

- `Model file not found`
- `Model file does not exist`
- `Missing model`

Check that the path points to an existing `.pt` or `.pth` file:

```bash
dir models
```

For YOLO26 detection models, use `models/yolo26_models/`. For YOLO26 segmentation models, use `models/yolo26_segmentation_models/`. You can also browse a local `.pt` file from the GUI.

## Official Model Download Failed

Check network access and write permission to the `models/` directory. If downloading from the GUI fails, manually place the model file in the expected folder and click Refresh Models.

## Classification Pretrained Weights

The classification GUI stores TorchVision pretrained weights under `models/image_classification_models/pretrained/`. Use:

- Check Weights
- Download Pretrained Weights
- Import Local Weights

If import fails, confirm the selected file is a file, not a directory, and that its extension is `.pth` or `.pt`.

## Checkpoint Prediction Fails

Classification checkpoints created by this project include `class_names`. If a checkpoint says `Checkpoint does not contain class_names`, use a model saved by the classification training workflow or retrain with this project.

## Unsupported Model Name

Use the GUI combobox values or these supported classification names:

- `resnet18`
- `mobilenet_v3_small`
