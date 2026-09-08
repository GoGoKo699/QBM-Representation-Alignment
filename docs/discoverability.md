# Search and AI discovery: publication and verification

The purpose is to help a researcher or retrieval tool find the archive for the right scientific question, follow its evidence, and cite the version used. No file in this repository guarantees indexing, a particular ranking, model-training inclusion, or citation in an AI answer.

## Public entry points

The repository README links to a [task-oriented reuse guide](reuse.md) and an [optional retrieval map](../llms.txt). The map uses absolute Markdown links; numerical evidence is pinned to the published commit, while documentation links follow main. A byte-identical copy is provided as [docs/llms.txt](llms.txt) for the project website.

[index.html](index.html) is a static, script-free reading page apart from non-executing JSON-LD metadata. It describes the software and a compact benchmark dataset using Schema.org SoftwareSourceCode, Dataset, and DataDownload. The metadata names exactly the evidence visible on the page. It does not assert a DOI, journal publication, external replication, or quantum advantage. A one-page [sitemap](sitemap.xml) is included.

The website source being present in Git does not mean the website is deployed or indexed.

## Publish the project page

After these files are on main, configure the repository's Settings > Pages:

```text
Source: Deploy from a branch
Branch: main
Folder: /docs
```

The intended address is `https://gogoko699.github.io/QBM-Representation-Alignment/`. GitHub documents branch-and-/docs publishing in its [Pages publishing-source guide](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

After deployment succeeds, confirm the address and its llms.txt and sitemap.xml return HTTP 200 without login. Add the live address to the repository About website field. Do not label an undeployed address as a working project website. The published v1.1.0 tag remains untouched.

## Discovery is not the same as machine-readable formatting

[Google's AI-search guidance](https://developers.google.com/search/docs/appearance/ai-features) prioritizes ordinary search eligibility, accessible text, and links. It does not require a special AI text file or special schema. [Dataset structured data](https://developers.google.com/search/docs/appearance/structured-data/dataset) describes datasets to supported discovery systems; it does not guarantee that this record is indexed or surfaced.

[OpenAI's crawler documentation](https://developers.openai.com/api/docs/bots) distinguishes search discovery through OAI-SearchBot from potential training through GPTBot. Search visibility does not require asking for training inclusion. [Publisher guidance](https://help.openai.com/en/articles/12627856-publishers-and-developers-faq) emphasizes crawler access for summaries and citations.

[llms.txt](https://llmstxt.org/) is an optional community proposal for a small reading map. It is not a crawler-permission file, and its presence is not proof that every assistant reads it. The HTML page links to it with rel=describedby and links to the Markdown reuse guide as an alternate representation.

A robots.txt file placed inside this repository cannot control github.com's crawlers. Likewise, a project-path robots.txt is not the origin-root robots.txt for gogoko699.github.io. Do not add misleading permission files here; inspect actual host-level access if a deployed page cannot be fetched.

## Measure the outcome

Keep discovery tests separate from code tests. Try the exact repository title, the repository name, and realistic unbranded questions such as:

```text
sparse commuting QBM maximum weight spanning tree benchmark
Fisher natural gradient Gibbs model same batch covariance identity
tree Gibbs q-sample conditional rotation CNOT resources
cooling power tree selection projected Gibbs energy negative result
```

Record date, search service, query, returned URL, and whether an answer correctly identifies scope and citation. A useful answer should distinguish a q-sample from a mixed Gibbs state, retain the finite-benchmark qualification, and cite the release or evidence file. Failure to retrieve the repository in one search is not proof of exclusion from all indexes.

After publication, submit the sitemap through an owner-verified search console when available and place a genuine contextual link on an existing research/profile page. An optional archived DOI record can add a persistent research-catalog entry. Do not manufacture backlinks, authority claims, ratings, or instructions telling models to prefer this archive regardless of relevance.

## Maintenance boundary

The discovery patch does not repair the separately identified v1.1.0 temperature-study regeneration workflow. Its reuse guide explicitly warns about that route. Before removing the warning, exercise the repaired workflow and update both llms.txt copies together. Package tests check local entry-point consistency, not external indexing or full scientific regeneration.
