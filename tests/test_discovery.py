"""Check discovery content, not search-engine indexing or scientific regeneration."""
from __future__ import annotations

import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
REPO = "GoGoKo699/QBM-Representation-Alignment"
COMMIT = "1fc02e89f283225b1d3503749b33c62c30d0a550"
SITE = "https://gogoko699.github.io/QBM-Representation-Alignment/"
RELEASE = f"https://github.com/{REPO}/releases/tag/v1.1.0"


class DiscoveryPage(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str | None]] = []
        self.scripts: list[dict[str, str | None]] = []
        self.ids: set[str] = set()
        self.json_parts: list[str] = []
        self.visible_parts: list[str] = []
        self.rates: dict[str, str] = {}
        self.current_rate: str | None = None
        self.in_json = False
        self.in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag in {"a", "link"}:
            self.links.append(values)
        if tag == "script":
            self.scripts.append(values)
            self.in_json = values.get("type") == "application/ld+json"
        if tag == "style":
            self.in_style = True
        if tag == "td":
            self.current_rate = values.get("data-key")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.in_json = False
        if tag == "style":
            self.in_style = False
        if tag == "td":
            self.current_rate = None

    def handle_data(self, data: str) -> None:
        if self.in_json:
            self.json_parts.append(data)
        elif not self.in_style:
            self.visible_parts.append(data)
        if self.current_rate:
            self.rates[self.current_rate] = data.strip()


def page() -> DiscoveryPage:
    parsed = DiscoveryPage()
    parsed.feed((ROOT / "docs/index.html").read_text(encoding="utf-8"))
    return parsed


def repository_path(url: str) -> str | None:
    parsed = urlparse(url)
    path = unquote(parsed.path).lstrip("/")
    if parsed.netloc == "raw.githubusercontent.com" and path.startswith(REPO + "/"):
        return path[len(REPO) + 1:].split("/", 1)[1]
    if parsed.netloc == "github.com" and path.startswith(REPO + "/blob/"):
        return path[len(REPO + "/blob/"):].split("/", 1)[1]
    return None


def test_retrieval_maps_are_small_linked_and_synchronized():
    text = (ROOT / "llms.txt").read_text(encoding="utf-8")
    assert text == (ROOT / "docs/llms.txt").read_text(encoding="utf-8")
    assert text.startswith("# Representation Alignment")
    assert "\n> " in text and "## Start here" in text
    assert len(text.encode()) < 12_000
    urls = re.findall(r"\]\((https://[^)]+)\)", text)
    assert len(urls) >= 10
    for url in urls:
        relative = repository_path(url)
        if relative:
            assert (ROOT / relative).exists(), url
    assert "not an indexing or ranking guarantee" in text
    assert "cleanup/validation mismatch" in text


def test_structured_data_identifies_real_software_and_only_packaged_data():
    parsed = page()
    payload = json.loads("".join(parsed.json_parts))
    assert payload["@context"] == "https://schema.org"
    nodes = {item["@type"]: item for item in payload["@graph"]}
    assert set(nodes) == {"SoftwareSourceCode", "Dataset"}
    software, dataset = nodes["SoftwareSourceCode"], nodes["Dataset"]
    assert software["codeRepository"] == f"https://github.com/{REPO}"
    assert software["url"] == RELEASE
    assert software["version"] == dataset["version"] == "1.1.0"
    assert software["author"]["name"] == dataset["creator"]["name"] == "Ruge Lin"
    assert software["datePublished"] == dataset["datePublished"] == "2026-08-26"
    assert COMMIT in software["identifier"]
    visible = " ".join(parsed.visible_parts)
    for node in nodes.values():
        assert node["name"] in visible
        assert node["@id"].split("#")[-1] in parsed.ids
    visible_links = {item.get("href") for item in parsed.links}
    expected = {"aggregate.csv", "primary_effects.csv", "preparation_resources.csv"}
    assert {item["contentUrl"].rsplit("/", 1)[-1] for item in dataset["distribution"]} == expected
    for item in dataset["distribution"]:
        assert item["@type"] == "DataDownload" and item["encodingFormat"] == "text/csv"
        assert f"/{COMMIT}/" in item["contentUrl"]
        assert item["contentUrl"] in visible_links
        relative = repository_path(item["contentUrl"])
        assert relative is not None and (ROOT / relative).is_file()
    assert "doi.org" not in json.dumps(payload)


def test_visible_success_rates_match_packaged_benchmark():
    with (ROOT / "results/confirmatory/aggregate.csv").open(newline="") as stream:
        expected = {
            f"{row['method']}:{row['initialization']}:{row['graph']}":
            f"{100 * float(row['success_rate']):.2f}%"
            for row in csv.DictReader(stream)
        }
    assert page().rates == expected


def test_page_has_plain_text_navigation_citation_and_scope():
    parsed = page()
    assert len(parsed.scripts) == 1
    assert parsed.scripts[0].get("type") == "application/ld+json"
    visible = " ".join(parsed.visible_parts)
    for phrase in ["not an external replication", "not the mixed Gibbs state",
                   "not a peer-reviewed journal publication", "No quantum speedup",
                   "@software", COMMIT, "cleanup/validation mismatch"]:
        assert phrase in visible
    for item in parsed.links:
        href = item.get("href")
        assert href is not None
        if href.startswith("#"):
            assert href[1:] in parsed.ids
        elif not urlparse(href).scheme:
            assert (ROOT / "docs" / href).is_file(), href
        else:
            relative = repository_path(href)
            if relative:
                assert (ROOT / relative).exists(), href
    assert any(item.get("rel") == "canonical" and item.get("href") == SITE for item in parsed.links)
    assert any(item.get("rel") == "describedby" and item.get("href") == "llms.txt" for item in parsed.links)


def test_sitemap_and_deployment_guidance_do_not_claim_indexing():
    sitemap = ElementTree.parse(ROOT / "docs/sitemap.xml")
    assert [node.text for node in sitemap.findall(".//{*}loc")] == [SITE]
    guide = (ROOT / "docs/discoverability.md").read_text(encoding="utf-8")
    assert "does not mean the website is deployed or indexed" in guide
    assert "Source: Deploy from a branch" in guide
    assert "Folder: /docs" in guide
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/reuse.md" in readme and "llms.txt" in readme
