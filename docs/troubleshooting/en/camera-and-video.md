# Camera and Video Troubleshooting

[Index](./README.md) | [中文](../zh-CN/camera-and-video.zh-CN.md)

This page covers camera discovery, open failures, frame read failures, screenshots, recordings, device permissions, and device contention.

## Camera Scan Failed or No Camera Found

Checks:

- Confirm the camera is connected.
- Close other apps that may own the camera.
- Check OS privacy settings and camera permissions.
- Try Refresh Cameras again.

On Windows, test the camera in the system Camera app. If it fails there, fix the OS or driver issue first.

## Open Failed

Symptoms:

- `Cannot open Camera`
- `Camera open failed`

Try a different camera route from the combobox. USB cameras may expose multiple backends; one may work while another fails.

## Read Failed

Symptoms:

- `Camera frame read failed`
- Preview opens, then stops.

Lower the requested resolution or close apps that use the camera. For YOLO workflows, stop detection first, then reopen the camera.

## Screenshot or Recording Failed

Check that the output folder is writable. Start with `.png` screenshots and `.mp4` recordings. If MP4 fails on a machine, retry AVI to separate codec problems from camera problems.

## Slow or Unstable Preview

YOLO detection and segmentation can be heavy on CPU. Use a smaller image size, CPU-friendly model, or stop live inference before recording.
