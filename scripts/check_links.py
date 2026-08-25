#!/usr/bin/env python3
"""Check repository-local Markdown links."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)|!\[[^\]]*\]\(([^)]+)\)")


def main() -> int:
    missing: list[tuple[Path, str]] = []
    checked = 0
    for path in ROOT.rglob("*.md"):
        if "frozen_source" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in LINK.finditer(text):
            target = (match.group(1) or match.group(2)).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            if ' "' in target:
                target = target.split(' "', 1)[0]
            candidate = (path.parent / unquote(target)).resolve()
            checked += 1
            try:
                candidate.relative_to(ROOT.resolve())
            except ValueError:
                missing.append((path.relative_to(ROOT), target))
                continue
            if not candidate.exists():
                missing.append((path.relative_to(ROOT), target))
    if missing:
        for path, target in missing:
            print(f"BROKEN {path}: {target}")
        return 1
    print(f"Local Markdown links passed: {checked} checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
