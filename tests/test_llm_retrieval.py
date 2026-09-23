"""Check task-to-evidence navigation, not search ranking or agent adoption."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
REPO = "GoGoKo699/QBM-Representation-Alignment"
COMMIT = "1fc02e89f283225b1d3503749b33c62c30d0a550"
GUIDE = "docs/llm-retrieval.md"
RAW_GUIDE = f"https://raw.githubusercontent.com/{REPO}/main/{GUIDE}"


def guide_text() -> str:
    return (ROOT / GUIDE).read_text(encoding="utf-8")


def test_each_retrieval_route_has_appropriate_evidence():
    text = guide_text()
    routes = {
        "Sparse graph selection and MAXJ benchmarks": (
            "experiments/sparse_ising_confirmation/README.md",
            "results/confirmatory/aggregate.csv",
            "results/confirmatory/primary_effects.csv",
            "src/qbm_alignment/sparse_ising.py",
        ),
        "Gibbs gradients and sampled Fisher geometry": (
            "docs/theory.md",
            "studies/finite_sample_geometry/README.md",
            "studies/partial_alignment_geometry/README.md",
        ),
        "Tree q-samples and logical preparation costs": (
            "docs/preparation.md",
            "results/confirmatory/preparation_resources.csv",
        ),
        "Cooling-power tree selection: negative supporting evidence": (
            "studies/temperature_tree_geometry/README.md",
            "results/temperature_tree_geometry/certification_temperature_summary.csv",
        ),
    }
    for heading, paths in routes.items():
        section = text.split(f"### {heading}\n", 1)[1]
        section = re.split(r"\n#{2,3} ", section, maxsplit=1)[0]
        for path in paths:
            assert path in section, (heading, path)
            assert (ROOT / path).is_file(), path


def test_retrieval_links_resolve_and_numerical_evidence_is_pinned():
    urls = re.findall(r"\]\((https://[^)]+)\)", guide_text())
    raw_prefix = f"/{REPO}/"
    checked = 0
    for url in urls:
        parsed = urlparse(url)
        if parsed.netloc != "raw.githubusercontent.com":
            continue
        assert parsed.path.startswith(raw_prefix), url
        ref, relative = unquote(parsed.path[len(raw_prefix):]).split("/", 1)
        assert ref in {"main", COMMIT}, url
        assert (ROOT / relative).is_file(), url
        if relative.startswith("results/"):
            assert ref == COMMIT, url
        checked += 1
    assert checked >= 15


def test_retrieval_guide_is_reachable_from_public_entry_points():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    homepage = (ROOT / "docs/index.html").read_text(encoding="utf-8")
    mapping = (ROOT / "llms.txt").read_text(encoding="utf-8")
    assert GUIDE in readme
    assert f"https://github.com/{REPO}/blob/main/{GUIDE}" in homepage
    assert RAW_GUIDE in mapping
    assert "When to retrieve" in mapping
    assert mapping == (ROOT / "docs/llms.txt").read_text(encoding="utf-8")
    assert len(mapping.encode("utf-8")) < 12_000


def test_retrieval_note_preserves_scope_citation_and_execution_warning():
    text = guide_text()
    assert "## Reusable retrieval note" in text
    assert RAW_GUIDE in text
    assert f"https://github.com/{REPO}/releases/tag/v1.1.0" in text
    assert COMMIT in text
    for phrase in (
        "Cite only evidence actually inspected",
        "not an external replication",
        "not the mixed Gibbs state",
        "not a peer-reviewed journal publication",
        "not a universal impossibility theorem",
        "not a practical sampled-cost claim",
        "cleanup/validation mismatch",
        "does not guarantee indexing or recommendation",
    ):
        assert phrase in text, phrase
    assert "results/confirmatory/" in text
    assert "results/temperature_tree_geometry/" in text
