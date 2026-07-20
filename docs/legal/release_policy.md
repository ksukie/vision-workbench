# Release Policy

[中文文档](./发布策略.md) | [Documentation](../README.md) | [Back to README](../../README.en.md)

Vision Workbench is an open-source learning project released under AGPL-3.0.
This policy defines the official boundaries between Git source, Python packages, and model release assets.

## Release Types

The project uses the following release types. They serve different purposes and are not expected to contain identical directory trees.

| Release type | Purpose | Content boundary |
| --- | --- | --- |
| Git repository, version tag, or matching source archive | Obtain and review the complete project source | Includes first-party source, tests, official documentation, helper scripts, project metadata, license files, model assets allowed by repository policy, and vendored third-party source kept in the repository |
| Python sdist | Build the base Python package | Includes the source, metadata, runtime dependency declarations, and applicable license files required to build and identify the Python package; it is not the complete Git repository source |
| Python wheel | Install the base application and Python entry points | Includes first-party Python packages, required package resources, entry points, and applicable license files; it does not need to include tests, development scripts, the complete vendored repository, or large models |
| GitHub Release Assets | Distribute files that do not belong in Git or the wheel | Includes the formal Python packages, Windows base EXE, and update manifest; may also include large models and optional offline resources |

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

## Version and Update Contract

Vision Workbench 1.0.0 and later releases must include both the exact-version
Python wheel and the stable-name `Vision-Workbench-win-x64.exe` expected by
the update manifest. The EXE's bundled identity must match the release. Generate
`update-manifest.json` only after final artifacts exist, and publish that
manifest with the same GitHub Release. The manifest version, tag, repository,
asset names, byte sizes, and SHA-256 digests must match the immutable release
artifacts.

The canonical user-facing repository URL is
`https://github.com/ksukie/Vision-WorkBench`. Version 1.0.0 embedded the
lowercase GitHub path alias in its trusted update contract, so
`release_info.json` and generated update manifests must retain
`https://github.com/ksukie/vision-workbench`. Runtime UI and project metadata
normalize that compatibility identity to the canonical URL. Removing the
alias from release artifacts would prevent existing 1.0.0 installations from
validating later official updates.

The desktop update client accepts stable releases only. It must not install an
asset without an exact compatible name, bounded positive size, and SHA-256
digest, and it must perform installation outside the running Qt process. A
Python wheel must also contain matching package metadata and bundled release
identity. A Windows EXE must pass its version and Qt construction self-test
before replacement. Run the version contract check for every build, and use
strict release mode only from the clean release tag:

Python wheel updates intentionally install with `--no-deps`. The manifest
therefore carries a SHA-256 fingerprint of the runtime dependency contract,
including base and optional runtime declarations. One-click installation is
enabled only when that fingerprint matches the running release; a
dependency-contract change requires the documented manual installation path.
The self-contained Windows EXE does not depend on the Python environment and
is exempt from this wheel-only compatibility gate.

```bash
python scripts/check_version_contract.py
python scripts/check_version_contract.py --release
python scripts/generate_update_manifest.py --published-at <UTC_TIMESTAMP>
```

The manifest generator derives the full commit SHA from the clean exact tag;
it does not accept a manually supplied commit identity.

Pushing the exact stable tag starts the `Prepare release draft` workflow. It
reruns the tagged-source gates, builds the Python distributions and Windows
EXE on their native runners, smoke-tests the installed wheel and frozen EXE,
generates and parses the manifest, and creates an unpublished GitHub Release
draft. Review the artifacts and confirm the full cross-platform CI result
before manually publishing that draft.

Use this release order; do not create the tag before the branch commit has
passed review and CI:

1. Update the version, release date, changelog, documentation, dependency
   contract, and tests in one reviewed commit.
2. Run the non-release contract, full tests, documentation links, and local
   package build; push the branch commit and wait for the complete CI matrix.
3. Create the exact annotated tag `v<major>.<minor>.<patch>` on that unchanged
   commit and push only that tag.
4. Let the tag workflow create the draft. Verify both required assets and
   their embedded identities, the manifest hashes and sizes, artifact
   self-tests, and CI status before publishing the draft manually.

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
