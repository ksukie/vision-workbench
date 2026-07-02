# Module Runtime Error Troubleshooting

[Index](./README.md) | [中文](../zh-CN/模块运行错误.md)

This page covers runtime errors inside CV effects, panorama reconstruction, YOLO detection, YOLO segmentation, and classification prediction.

## CV Basics Processing Failed

Try resetting the result and applying the effect again. Custom operations must return a valid grayscale or color NumPy image.

## Unsupported Effect

Use the GUI effect combobox or `service.list_effects()` to choose a registered effect. Custom effects must be registered before use.

## Panorama Reconstruction Failed

Common causes:

- Fewer than 3 manual point pairs.
- Point pairs are poorly distributed.
- SIFT cannot find enough features.
- Homography or affine estimation fails.

Use clear overlapping image pairs. Place manual points across the shared area, not all in one corner.

## YOLO Detection Runtime Failed

Check that a model is loaded, the camera is open, and image size is positive. If failures happen only on CUDA, retry with `Device=cpu` to separate model/data problems from GPU runtime problems.

## YOLO Segmentation Failed

Check that the selected task matches the model type, the model file exists, and image size is a valid integer. If live segmentation fails, run once on a still image first.

## Classification Prediction Failed

Check that the image opens correctly, pretrained weights or checkpoint files exist, and the selected device is available.
