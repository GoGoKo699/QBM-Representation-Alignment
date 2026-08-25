#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -W error::SyntaxWarning -m compileall -f -q src experiments studies scripts tests
python -m pytest -q
python scripts/check_links.py
python scripts/validate_repository.py

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir "$tmp" >/dev/null
wheel_count="$(find "$tmp" -maxdepth 1 -type f -name '*.whl' | wc -l)"
if [[ "$wheel_count" -ne 1 ]]; then
  echo "Expected exactly one wheel, found $wheel_count" >&2
  exit 1
fi

echo "Packaged release check passed."
echo "Run 'make validate-all' for complete scientific recomputation."
