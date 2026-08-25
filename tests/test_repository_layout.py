from __future__ import annotations

import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "1.1.0"
EXPECTED_URL = "https://github.com/GoGoKo699/QBM-Representation-Alignment"


def test_canonical_public_files_exist():
    required = [
        ".gitattributes",
        ".github/workflows/tests.yml",
        "README.md",
        "CITATION.cff",
        "CITATION.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "llms.txt",
        "docs/evidence_map.md",
        "docs/scientific_claims.md",
        "docs/research_context.md",
        "docs/statistical_analysis.md",
        "docs/theory.md",
        "docs/preparation.md",
        "docs/reproducibility.md",
        "docs/data_formats.md",
        "docs/limitations.md",
        "figures/success_by_representation.png",
        "figures/primary_effects.png",
        "figures/preparation_resources.png",
        "results/confirmatory/validation.json",
        "results/temperature_tree_geometry/validation.json",
        "results/repository_validation.json",
        "studies/temperature_tree_geometry/README.md",
        "studies/temperature_tree_geometry/protocol.md",
        "studies/temperature_tree_geometry/scripts/validate_study.py",
        "scripts/release_check.sh",
    ]
    assert not [relative for relative in required if not (ROOT / relative).is_file()]


def test_release_metadata_is_consistent():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == EXPECTED_VERSION
    assert project["urls"]["Repository"] == EXPECTED_URL

    init_text = (ROOT / "src" / "qbm_alignment" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*[\"\']([^\"\']+)', init_text)
    assert match is not None and match.group(1) == EXPECTED_VERSION

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f"version: {EXPECTED_VERSION}" in citation
    assert f'repository-code: "{EXPECTED_URL}"' in citation
    assert "license: BSD-3-Clause" in citation


def test_license_is_bsd_three_clause():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("BSD 3-Clause License\n")
    assert "Copyright (c) 2026, Ruge Lin" in text


def test_citation_is_easy_to_find_and_versioned():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.md").read_text(encoding="utf-8")
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "## How to cite" in readme
    assert "Cite this repository" in readme
    assert f"Version {EXPECTED_VERSION}" in citation
    assert "@software" in citation
    assert "v1.0.0" in citation
    assert "full commit SHA" in citation
    assert "see CITATION.md" in cff


def test_deep_entry_points_link_back_to_citation():
    readmes = [
        "experiments/sparse_ising_confirmation/README.md",
        "studies/boundary_geometry/README.md",
        "studies/finite_sample_geometry/README.md",
        "studies/partial_alignment_geometry/README.md",
        "studies/temperature_tree_geometry/README.md",
    ]
    for relative in readmes:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "## Citation" in text, relative
        assert "CITATION.md" in text, relative


def test_primary_experiment_terms_are_unambiguous():
    text = (ROOT / "experiments/sparse_ising_confirmation/README.md").read_text(encoding="utf-8")
    assert "not an external replication" in text
    assert "`problem_tree`" in text
    assert "ground-state probability" in text
    assert "planted-ground-state probability" not in text


def test_packaged_repository_validation_is_current():
    payload = json.loads((ROOT / "results" / "repository_validation.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["release_metadata"]["version"] == EXPECTED_VERSION
    assert "temperature_tree_geometry" in payload["validation_records"]
    assert payload["records"]["temperature_tree_geometry"]["status"] == "PASS"
