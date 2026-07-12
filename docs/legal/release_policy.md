# Release Policy

[中文文档](./发布策略.md) | [Documentation](../README.md) | [Back to README](../../README.md)

Vision Workbench is an open-source learning project released under AGPL-3.0.
This policy defines the official boundaries between Git source, Python packages, and model release assets.

## Release Types

The project uses the following release types. They serve different purposes and are not expected to contain identical directory trees.

| Release type | Purpose | Content boundary |
| --- | --- | --- |
| Git repository, version tag, or matching source archive | Obtain and review the complete project source | Includes first-party source, tests, official documentation, helper scripts, project metadata, license files, model assets allowed by repository policy, and vendored third-party source kept in the repository |
| Python sdist | Build the base Python package | Includes the source, metadata, and applicable license files required to build the Python package; it is not the complete Git repository source |
| Python wheel | Install the base application and Python entry points | Includes first-party Python packages, required package resources, entry points, and applicable license files; it does not need to include tests, development scripts, the complete vendored repository, or large models |
| GitHub Release Assets | Distribute files that do not belong in Git or the wheel | May include large models, optional offline resources, and prebuilt Python packages |

Use the matching Git tag or source archive when complete project source is required. Do not treat a wheel or Python sdist as a complete copy of the repository.

## Source Code

A complete source release includes `src/`, `tests/`, `docs/`, `scripts/`, model assets allowed by repository policy, project metadata, and the applicable license files.

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

Do not commit generated wheel or source distribution files from `dist/`. Build them only when preparing a release and from the matching versioned source state.

Release wheels should stay lightweight. Optional deep-learning dependencies, large model weights, tests, development scripts, and the complete vendored repository may be excluded from the wheel. Document required additions in the README, module documentation, or Release Assets.

## License Files

A complete source release should preserve:

- `LICENSE`
- `NOTICE`
- `THIRD_PARTY_NOTICES.md`
- `CITATION.cff`
- third-party license files kept with vendored third-party source

Wheels and sdists should include the license and notice files applicable to the content they actually distribute. An artifact that excludes a third-party component should not be described as containing that component's complete source.
