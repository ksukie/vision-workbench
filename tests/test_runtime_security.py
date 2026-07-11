from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import vision_workbench.runtime_security as runtime_security


def test_validate_run_name_accepts_plain_unicode_name() -> None:
    assert runtime_security.validate_run_name("缺陷检测_v1") == "缺陷检测_v1"


@pytest.mark.parametrize(
    "value",
    ("../escape", r"..\escape", "C:/absolute", "bad/name", "NUL", "trailing."),
)
def test_validate_run_name_rejects_paths_and_reserved_names(value: str) -> None:
    with pytest.raises(ValueError):
        runtime_security.validate_run_name(value)


def test_confined_child_path_stays_under_root(tmp_path: Path) -> None:
    result = runtime_security.confined_child_path(tmp_path / "runs", "experiment-1")

    assert result == (tmp_path / "runs" / "experiment-1").resolve()


def test_configure_restricted_model_loading_sets_safe_defaults(monkeypatch) -> None:
    monkeypatch.delenv(runtime_security.SAFE_LOAD_ENV, raising=False)
    monkeypatch.delenv(runtime_security.TORCH_SAFE_LOAD_ENV, raising=False)

    runtime_security.configure_restricted_model_loading()

    assert os.environ[runtime_security.SAFE_LOAD_ENV] == "true"
    assert os.environ[runtime_security.TORCH_SAFE_LOAD_ENV] == "1"


def test_configure_isolated_environment_removes_user_site(monkeypatch, tmp_path: Path) -> None:
    user_site = tmp_path / "user-site"
    env_site = tmp_path / "env-site"
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "conda"))
    monkeypatch.delenv(runtime_security.ALLOW_USER_SITE_ENV, raising=False)
    monkeypatch.setattr(runtime_security.site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(sys, "path", [str(user_site), str(env_site)])

    removed = runtime_security.configure_isolated_python_environment()

    assert removed == (str(user_site),)
    assert sys.path == [str(env_site)]
    assert os.environ["PYTHONNOUSERSITE"] == "1"
