.PHONY: compile test validate validate-all analyze analyze-all links build release-check clean

compile:
	python -W error::SyntaxWarning -m compileall -f -q src experiments studies scripts tests

test:
	python -m pytest -q

links:
	python scripts/check_links.py

validate:
	python scripts/validate_repository.py

validate-all:
	python experiments/sparse_ising_confirmation/scripts/validate_experiment.py
	python studies/boundary_geometry/scripts/validate_study.py
	python studies/finite_sample_geometry/scripts/validate_study.py
	python studies/partial_alignment_geometry/scripts/validate_study.py
	python studies/temperature_tree_geometry/scripts/validate_study.py

analyze:
	bash scripts/refresh_analysis.sh core

analyze-all:
	bash scripts/refresh_analysis.sh all

build:
	@tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT; \
	python -m pip wheel . --no-deps --no-build-isolation --wheel-dir "$$tmp"

release-check:
	bash scripts/release_check.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -rf .pytest_cache build dist *.egg-info
