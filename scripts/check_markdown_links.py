"""Check local links in Markdown documentation."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import unquote, urlparse


SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "datasets",
    "dist",
    "models",
    "runs",
    "third_party",
}

MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
HTML_LINK_RE = re.compile(r"""(?:href|src)=["']([^"']+)["']""", re.IGNORECASE)


@dataclass(frozen=True)
class BrokenLink:
    file: Path
    line: int
    target: str
    resolved: Path

    def format(self, root: Path) -> str:
        try:
            rel_file = self.file.relative_to(root)
        except ValueError:
            rel_file = self.file
        try:
            rel_resolved = self.resolved.relative_to(root)
        except ValueError:
            rel_resolved = self.resolved
        return f"{rel_file}:{self.line}: {self.target} -> {rel_resolved}"


def find_broken_links(root: Path) -> list[BrokenLink]:
    root = root.resolve()
    broken: list[BrokenLink] = []
    for markdown_path in iter_markdown_files(root):
        text = markdown_path.read_text(encoding="utf-8")
        stripped = strip_fenced_code(text)
        line_starts = _line_starts(stripped)
        for target, offset in iter_link_targets(stripped):
            local_target = normalize_local_target(target)
            if local_target is None:
                continue
            resolved = (markdown_path.parent / local_target).resolve()
            if not resolved.exists():
                broken.append(
                    BrokenLink(
                        file=markdown_path,
                        line=_line_number(line_starts, offset),
                        target=target,
                        resolved=resolved,
                    )
                )
    return broken


def iter_markdown_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*.md"))
    docs_dir = root / "docs"
    if not docs_dir.exists():
        return
    for path in sorted(docs_dir.rglob("*.md")):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in SKIP_DIRS for part in relative_parts):
            continue
        yield path


def strip_fenced_code(text: str) -> str:
    lines = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            lines.append("\n" if line.endswith("\n") else "")
            continue
        lines.append("\n" if in_fence and line.endswith("\n") else ("" if in_fence else line))
    return "".join(lines)


def iter_link_targets(text: str) -> Iterable[tuple[str, int]]:
    for regex in (MARKDOWN_LINK_RE, HTML_LINK_RE):
        for match in regex.finditer(text):
            yield match.group(1).strip(), match.start(1)


def normalize_local_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if not target or target.startswith("#"):
        return None
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split()[0]

    parsed = urlparse(target)
    if parsed.scheme or target.startswith("//"):
        return None
    path_text = target.split("#", 1)[0].split("?", 1)[0]
    if not path_text:
        return None
    return unquote(path_text)


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _line_number(line_starts: Sequence[int], offset: int) -> int:
    line = 1
    for index, start in enumerate(line_starts, start=1):
        if start > offset:
            break
        line = index
    return line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local Markdown links.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    broken = find_broken_links(root)
    if broken:
        print("Broken local Markdown links:")
        for item in broken:
            print(f"- {item.format(root)}")
        return 1
    print("Markdown local links OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
