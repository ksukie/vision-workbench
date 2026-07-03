from pathlib import Path

from scripts.check_markdown_links import find_broken_links


def test_markdown_links_are_valid() -> None:
    assert find_broken_links(Path.cwd()) == []


def test_markdown_link_checker_reports_missing_local_file(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("[missing](docs/missing.md)\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()

    broken = find_broken_links(tmp_path)

    assert len(broken) == 1
    assert broken[0].target == "docs/missing.md"
