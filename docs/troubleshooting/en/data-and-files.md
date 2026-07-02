# Data and File Troubleshooting

[Index](./README.md) | [中文](../zh-CN/数据与文件.md)

This page covers image open/save errors, decoding/encoding errors, point-pair JSON files, checkpoints, and output paths.

## Image Open Failed

Symptoms:

- `Open failed`
- `Cannot decode image file`
- `Loaded file is not a supported image`

Try another common image format such as PNG or JPEG. Confirm the file is not corrupted and is readable by the current user.

```bash
python -c "from PIL import Image; Image.open(r'path\to\image.png').verify(); print('ok')"
```

## Save Failed

Symptoms:

- `Cannot encode image`
- `Save failed`

Check that the output directory exists and is writable. Use `.png` first when diagnosing save issues.

## Point-pair JSON Fails to Load

Panorama point-pair files must contain a JSON list, and each item must contain `left` and `right` point values. Manually edited files should be re-saved from the GUI to restore the expected structure.

## Path Problems

Use absolute paths while diagnosing. Paths containing non-ASCII characters or special punctuation should be retried from a short ASCII-only directory to rule out path parsing problems.

## Checkpoint Files

Use `.pt` or `.pth` checkpoint files. A checkpoint selected for prediction must match the workflow that produced it, especially for classification class names.
