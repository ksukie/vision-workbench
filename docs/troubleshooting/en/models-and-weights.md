# Models and Weights Troubleshooting

[Index](./README.md) | [中文](../zh-CN/模型与权重.md)

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

## Remote Model Manifest

The YOLO26 pages read the local manifest cache plus bundled defaults when filling model dropdowns. Set `VISION_WORKBENCH_MODEL_MANIFEST_URL` to a JSON manifest URL if you maintain a remote catalog. Clicking Refresh Models updates the cache at `~/.vision_workbench/cache/model_manifest.json` and then repopulates the dropdown; without network access, the cached or bundled model list is still used.

Minimal manifest shape:

```json
{
  "schema_version": 1,
  "models": [
    {"family": "yolo26", "task": "detect", "name": "yolo26n.pt", "url": "https://example.com/yolo26n.pt"},
    {"family": "yolo26", "tasks": ["segment"], "name": "yolo26n-seg.pt", "url": "https://example.com/yolo26n-seg.pt"}
  ]
}
```

## Incomplete or Corrupt `.pt` Files

Vision Workbench downloads model files to a temporary `.pt.part` path first, validates the finished archive, and only then promotes it to `.pt`. Closing the UI or process during download leaves a partial file that is ignored on the next startup, so the model remains available for re-download.

Stale or corrupt `.pt` files should be removed individually and downloaded again from the Qt model page. Temporary files matching `models/**/*.pt.part` are intentionally ignored by Git.

## Classification Pretrained Weights

The classification GUI stores TorchVision pretrained weights under `models/image_classification_models/pretrained/`. Use:

- Check Weights
- Download Pretrained Weights
- Import Local Weights

If import fails, confirm the selected file is a file, not a directory, and that its extension is `.pth` or `.pt`.

## Checkpoint Prediction Fails

Classification checkpoints created by this project include `class_names`. The message `Checkpoint does not contain class_names` indicates that the selected file was not saved by the classification training workflow and should be replaced by a compatible checkpoint.

## Unsupported Model Name

Use the GUI combobox values or these supported classification names:

- `resnet18`
- `mobilenet_v3_small`
