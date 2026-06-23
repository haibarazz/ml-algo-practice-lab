#!/usr/bin/env python3
"""Check local Markdown links under docs/."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def iter_markdown_files() -> list[Path]:
    return sorted(
        path
        for path in DOCS.rglob("*.md")
        if "node_modules" not in path.parts and ".vitepress" not in path.parts
    )


def should_skip(raw_link: str) -> bool:
    link = raw_link.strip()
    return (
        not link
        or link.startswith("#")
        or link.startswith("http://")
        or link.startswith("https://")
        or link.startswith("mailto:")
        or link.startswith("data:")
    )


def candidates_for(source: Path, raw_link: str) -> list[Path]:
    link = raw_link.strip().split("#", 1)[0].split("?", 1)[0]
    if should_skip(link):
        return []

    if link.startswith("/"):
        target = DOCS / link.lstrip("/")
        public_target = DOCS / "public" / link.lstrip("/")
    else:
        target = (source.parent / link).resolve()
        public_target = None

    candidates = [target]
    if public_target is not None:
        candidates.append(public_target)
    if target.suffix == "":
        candidates.extend([target.with_suffix(".md"), target / "README.md", target / "index.md"])
    elif target.suffix == ".html":
        candidates.append(target.with_suffix(".md"))

    return candidates


def main() -> int:
    missing: list[tuple[Path, str]] = []

    for path in iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        for link in LINK_RE.findall(text):
            candidates = candidates_for(path, link)
            if candidates and not any(candidate.exists() for candidate in candidates):
                missing.append((path.relative_to(ROOT), link))

    if missing:
        print("Missing local links:")
        for path, link in missing:
            print(f"- {path.as_posix()}: {link}")
        print(f"\nTotal missing: {len(missing)}")
        return 1

    print(f"All local docs links resolve across {len(iter_markdown_files())} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
