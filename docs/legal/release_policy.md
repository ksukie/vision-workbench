# Release Policy

[中文文档](./release_policy.zh-CN.md) | [Back to README](../../README.md)

Vision Workbench is an open-source learning project released under AGPL-3.0.
This policy explains what should be included in a public release.

## Source Code

The source code under `src/`, `tests/`, `docs/`, and the project metadata files
is intended to be published.

The vendored YOLO26 source under `third_party/yolo26_source/` is included for
learning and local integration. It remains subject to the Ultralytics AGPL-3.0
license and attribution requirements.

## Model Files

Model files may be published only when each individual file is at or below
100 MB. Files above 100 MB should not be committed to Git.

Recommended distribution methods for large files:

- GUI download buttons
- local user download
- GitHub Release Assets
- external model package mirrors
- Git LFS, if the repository owner explicitly enables it

Run the release asset check before publishing:

```bash
python scripts/check_release_assets.py
```

## Build Artifacts

Do not commit generated wheel or source distribution files from `dist/`.
Build them only when preparing a release.

## License Files

Every public release should include:

- `LICENSE`
- `NOTICE`
- `THIRD_PARTY_NOTICES.md`
- `CITATION.cff`
- third-party license files kept with vendored third-party source
