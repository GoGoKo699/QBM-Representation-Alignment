# Contributing

Bug reports, reproducibility problems, documentation improvements, and focused scientific corrections are welcome through GitHub issues or pull requests.

## Before opening a pull request

1. Install the repository with `python -m pip install -e ".[test]"`.
2. Run `make release-check`.
3. Keep numerical changes separate from documentation-only changes.
4. Add or update tests for implementation changes.
5. Explain any change to a scientific result, threshold, instance set, graph rule, or optimizer in the pull-request description.

## Scientific records

The independent sparse-Ising confirmation is a frozen scientific record. Changes that create a new analysis, optimizer, graph rule, threshold, or data selection must be stored as a new study rather than silently replacing the packaged confirmatory results.

Please cite the repository using `CITATION.cff` when reusing code, data, figures, or derived results.
