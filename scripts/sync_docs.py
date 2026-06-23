#!/usr/bin/env python3
"""Sync source Markdown pages into the VitePress docs tree."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO = os.environ.get("DOCS_REPO", "haibarazz/ml-algo-practice-lab")
BRANCH = os.environ.get("DOCS_BRANCH", "main")

SOURCE_ROOTS = [
    ROOT / "ml_dl_handwriting",
    ROOT / "projects" / "minimind",
]

GENERATED_ROOTS = [
    DOCS / "ml_dl_handwriting",
    DOCS / "projects" / "minimind",
]


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return "source" in rel.parts or ".ipynb_checkpoints" in rel.parts


def notebook_for_markdown(path: Path) -> Path | None:
    if path.name != f"{path.parent.name}.md":
        return None
    notebook = path.with_suffix(".ipynb")
    return notebook if notebook.exists() else None


def cloud_block(notebook: Path) -> str:
    rel = notebook.relative_to(ROOT).as_posix()
    colab = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{rel}"
    return f"""
::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::
""".strip()


def inject_cloud_block(text: str, notebook: Path | None) -> str:
    if notebook is None or "Open In Colab" in text:
        return text

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("# "):
            return "\n".join(lines[: index + 1] + ["", cloud_block(notebook), ""] + lines[index + 1 :])
    return f"{cloud_block(notebook)}\n\n{text}"


def fold_answer_section(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    in_answer = False
    in_code = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code

        is_answer_heading = (
            not in_code
            and line.startswith("## ")
            and ("参考答案" in line or "答案与解析" in line or "官方解析" in line)
        )
        is_next_h2 = not in_code and in_answer and line.startswith("## ") and not is_answer_heading

        if is_answer_heading and not in_answer:
            output.extend(["::: details 点击查看参考答案与解析", "", line])
            in_answer = True
            continue

        if is_next_h2:
            output.extend(["", ":::", "", line])
            in_answer = False
            continue

        output.append(line)

    if in_answer:
        output.extend(["", ":::"])

    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def remove_source_clues(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    skip = False

    for line in lines:
        if line.startswith("## 题源线索"):
            skip = True
            continue
        if skip and line.startswith("## "):
            skip = False
            while output and output[-1] == "":
                output.pop()
            output.append("")
        if not skip:
            output.append(line)

    return "\n".join(output).strip() + "\n"


def notes_body(notes_path: Path) -> str | None:
    if not notes_path.exists():
        return None

    lines = notes_path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and lines[0] == "":
        lines = lines[1:]

    demoted: list[str] = []
    for line in lines:
        if line.startswith("## "):
            demoted.append("#" + line)
        else:
            demoted.append(line)

    body = "\n".join(demoted).strip()
    return body if body else None


def inline_notes_reference(text: str, source_path: Path) -> str:
    heading = "## 工程要点 / 面试追问"
    if heading not in text or "见 `notes.md`" not in text:
        return text

    body = notes_body(source_path.parent / "notes.md")
    if body is None:
        return text

    before = text.split(heading, 1)[0].rstrip()
    return f"{before}\n\n{heading}\n\n{body}\n"


def transform_markdown(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = remove_source_clues(text)
    text = inline_notes_reference(text, path)
    text = inject_cloud_block(text, notebook_for_markdown(path))
    text = fold_answer_section(text)
    return text


def copy_markdown_files() -> int:
    count = 0
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*.md")):
            if is_excluded(path):
                continue

            rel = path.relative_to(ROOT)
            dest = DOCS / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            transformed = transform_markdown(path)
            dest.write_text(transformed, encoding="utf-8")
            count += 1

            if path.name == "README.md":
                dest.with_name("index.md").write_text(transformed, encoding="utf-8")
                count += 1

    return count


def reset_generated_roots() -> None:
    for path in GENERATED_ROOTS:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    reset_generated_roots()
    count = copy_markdown_files()
    print(f"Synced {count} Markdown files into docs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
