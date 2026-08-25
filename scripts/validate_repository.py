#!/usr/bin/env python3
"""Validate release metadata, layout, links, and packaged scientific records."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NAME = "qbm-representation-alignment"
EXPECTED_VERSION = "1.1.0"
EXPECTED_URL = "https://github.com/GoGoKo699/QBM-Representation-Alignment"
EXPECTED_LICENSE = "BSD-3-Clause"
VALIDATION_RECORDS = {
    "confirmatory": ROOT / "results" / "confirmatory" / "validation.json",
    "boundary_geometry": ROOT / "results" / "boundary_geometry" / "validation.json",
    "finite_sample_geometry": ROOT / "results" / "finite_sample_geometry" / "validation.json",
    "partial_alignment_geometry": ROOT / "results" / "partial_alignment_geometry" / "validation.json",
    "temperature_tree_geometry": ROOT / "results" / "temperature_tree_geometry" / "validation.json",
}
REQUIRED_FILES = (
    ".gitattributes",
    ".github/workflows/tests.yml",
    ".gitignore",
    "CITATION.cff",
    "CITATION.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "docs/evidence_map.md",
    "docs/research_context.md",
    "docs/statistical_analysis.md",
    "llms.txt",
    "pyproject.toml",
    "scripts/release_check.sh",
    "src/qbm_alignment/__init__.py",
    "studies/temperature_tree_geometry/README.md",
    "studies/temperature_tree_geometry/protocol.md",
    "studies/temperature_tree_geometry/scripts/validate_study.py",
    "results/temperature_tree_geometry/validation.json",
)
TEXT_SUFFIXES = {".cff", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
EXCLUDED_RUNTIME_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", "build", "dist"}


def _repository_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        directories = relative.parts[:-1]
        if any(part in EXCLUDED_RUNTIME_DIRS or part.endswith(".egg-info") for part in directories):
            continue
        yield path


def _run_link_checker() -> int:
    path = ROOT / "scripts" / "check_links.py"
    spec = importlib.util.spec_from_file_location("qbm_repository_link_checker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


def _yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*[\"']?([^\n\"']+)[\"']?\s*$", text)
    if match is None:
        raise AssertionError(f"missing {key!r} in CITATION.cff")
    return match.group(1).strip()


def _validate_release_metadata() -> dict[str, object]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["name"] == EXPECTED_NAME
    assert project["version"] == EXPECTED_VERSION
    assert project["urls"]["Repository"] == EXPECTED_URL
    assert project["urls"]["Homepage"] == EXPECTED_URL

    init_text = (ROOT / "src" / "qbm_alignment" / "__init__.py").read_text(encoding="utf-8")
    version_match = re.search(r'__version__\s*=\s*[\"\']([^\"\']+)', init_text)
    assert version_match is not None and version_match.group(1) == EXPECTED_VERSION

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert _yaml_scalar(citation, "version") == EXPECTED_VERSION
    assert _yaml_scalar(citation, "repository-code") == EXPECTED_URL
    assert _yaml_scalar(citation, "url") == EXPECTED_URL
    assert _yaml_scalar(citation, "license") == EXPECTED_LICENSE

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("BSD 3-Clause License\n")
    assert "Copyright (c) 2026, Ruge Lin" in license_text

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert EXPECTED_URL in readme
    assert "BSD 3-Clause License" in readme

    return {
        "package_name": EXPECTED_NAME,
        "version": EXPECTED_VERSION,
        "repository_url": EXPECTED_URL,
        "license": EXPECTED_LICENSE,
    }


def _validate_layout() -> dict[str, object]:
    missing = [relative for relative in REQUIRED_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise AssertionError(f"missing required files: {missing}")

    zero_byte_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in _repository_files()
        if path.stat().st_size == 0
    )
    if zero_byte_files:
        raise AssertionError(f"zero-byte files present: {zero_byte_files}")

    oversized_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in _repository_files()
        if path.stat().st_size > 100 * 1024 * 1024
    )
    if oversized_files:
        raise AssertionError(f"files exceed GitHub's 100 MiB limit: {oversized_files}")

    absolute_path_hits: list[str] = []
    for path in _repository_files():
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        container_prefix = "/" + "mnt/data/"
        work_marker = "qbm_" + "public_work"
        if container_prefix in text or work_marker in text:
            absolute_path_hits.append(path.relative_to(ROOT).as_posix())
    if absolute_path_hits:
        raise AssertionError(f"container-local paths present: {absolute_path_hits}")

    return {
        "file_count": sum(1 for _path in _repository_files()),
        "zero_byte_files": 0,
        "oversized_files": 0,
        "container_local_paths": 0,
    }


def main() -> int:
    records: dict[str, object] = {}
    for name, path in VALIDATION_RECORDS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise AssertionError(f"validation record did not pass: {name}")
        records[name] = payload

    metadata = _validate_release_metadata()
    layout = _validate_layout()

    if _run_link_checker() != 0:
        raise AssertionError("local documentation link check failed")

    report = {
        "status": "PASS",
        "mode": "release metadata, layout, links, and packaged scientific records",
        "release_metadata": metadata,
        "layout": layout,
        "validation_records": {
            name: path.relative_to(ROOT).as_posix()
            for name, path in VALIDATION_RECORDS.items()
        },
        "records": records,
    }
    output = ROOT / "results" / "repository_validation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Repository validation passed. Wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
