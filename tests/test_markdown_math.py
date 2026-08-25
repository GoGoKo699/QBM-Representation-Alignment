from __future__ import annotations

import re
from pathlib import Path


UNSUPPORTED_DELIMITERS = (r"\[", r"\]", r"\(", r"\)")
UNSUPPORTED_MACROS = (r"\operatorname",)


def _outside_fenced_code(text: str) -> str:
    """Return Markdown text with fenced code blocks removed."""
    kept: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue
        if not in_fence:
            kept.append(line)
    assert not in_fence, "unterminated fenced code block"
    return "\n".join(kept)


def _unescaped_dollar_count(text: str) -> int:
    return len(re.findall(r"(?<!\\)\$", text))


ROOT = Path(__file__).resolve().parents[1]


def test_public_markdown_uses_github_math_syntax() -> None:
    markdown_files = sorted(ROOT.rglob("*.md"))
    assert markdown_files

    for path in markdown_files:
        relative = path.relative_to(ROOT)
        # These files are cryptographically frozen protocol evidence.  Their
        # maintained, GitHub-rendered counterparts live one directory above.
        if "frozen_source" in relative.parts:
            continue

        text = path.read_text(encoding="utf-8")
        visible = _outside_fenced_code(text)

        for delimiter in UNSUPPORTED_DELIMITERS:
            assert delimiter not in visible, f"{relative}: use GitHub $/$$ math delimiters, not {delimiter}"
        for macro in UNSUPPORTED_MACROS:
            assert macro not in visible, f"{relative}: use a portable MathJax form instead of {macro}"

        assert visible.count("$$") % 2 == 0, f"{relative}: unbalanced display-math delimiters"
        inline_only = visible.replace("$$", "")
        assert _unescaped_dollar_count(inline_only) % 2 == 0, f"{relative}: unbalanced inline-math delimiters"

        disallowed_controls = [
            character
            for character in text
            if ord(character) < 32 and character not in {"\n", "\t", "\r"}
        ]
        assert not disallowed_controls, f"{relative}: contains control characters"
