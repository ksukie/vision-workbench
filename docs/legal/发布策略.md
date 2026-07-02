# 发布策略

[English](./release_policy.md) | [返回总 README](../../README.zh-CN.md)

Vision Workbench 是一个基于 AGPL-3.0 发布的开源学习项目。本文档说明公开发布时建议包含和不建议包含的内容。

## 源码

`src/`、`tests/`、`docs/` 和项目元数据文件可以作为公开发布内容。

`third_party/yolo26_source/` 下的 YOLO26 源码作为学习和本地集成用途内置。该部分仍然遵守 Ultralytics 的 AGPL-3.0 许可和署名要求。

## 模型文件

模型文件可以公开，但单个文件建议不超过 100 MB。超过 100 MB 的文件不建议提交到 Git 仓库。

大型文件建议通过以下方式提供：

- GUI 下载按钮
- 用户本地自行下载
- GitHub Release Assets
- 外部模型资源包
- 仓库维护者明确启用 Git LFS 后使用 Git LFS

公开发布前建议运行：

```bash
python scripts/check_release_assets.py
```

## 构建产物

不要提交 `dist/` 目录下生成的 wheel 或源码压缩包。只有准备发布时再重新构建。
发布 wheel 应保持轻量：可选深度学习依赖组和大型模型权重应通过文档说明或 Release Assets 单独分发。

## 许可文件

每次公开发布建议包含：

- `LICENSE`
- `NOTICE`
- `THIRD_PARTY_NOTICES.md`
- `CITATION.cff`
- 第三方源码目录中保留原始许可文件
