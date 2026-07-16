from scripts import check_release_assets


def test_release_asset_check_skips_git_ignored_models(tmp_path, monkeypatch) -> None:
    model = tmp_path / "ignored.pt"
    model.write_bytes(b"oversized")
    monkeypatch.setattr(check_release_assets, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(check_release_assets, "MAX_MODEL_BYTES", 1)
    monkeypatch.setattr(check_release_assets, "is_git_ignored", lambda _path: True)

    assert check_release_assets.main() == 0
