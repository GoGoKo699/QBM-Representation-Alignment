#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: bash scripts/refresh_analysis.sh [core|boundary|finite|partial|all]

  core      regenerate the primary confirmatory tables and figures (default)
  boundary  regenerate the boundary-geometry supporting study
  finite    regenerate the finite-sample supporting study
  partial   regenerate the partial-alignment supporting study
  all       regenerate the primary experiment and all supporting studies
EOF
}

run_core() {
  python experiments/sparse_ising_confirmation/scripts/analyze_results.py
  python experiments/sparse_ising_confirmation/scripts/make_figures.py
}

run_boundary() {
  python studies/boundary_geometry/scripts/analyze_results.py
  python studies/boundary_geometry/scripts/make_figures.py
}

run_finite() {
  python studies/finite_sample_geometry/scripts/analyze_results.py
  python studies/finite_sample_geometry/scripts/make_figures.py
}

run_partial() {
  python studies/partial_alignment_geometry/scripts/analyze_results.py
  python studies/partial_alignment_geometry/scripts/make_figures.py
}

selection="${1:-core}"
case "$selection" in
  core)
    run_core
    ;;
  boundary)
    run_boundary
    ;;
  finite)
    run_finite
    ;;
  partial)
    run_partial
    ;;
  all)
    run_core
    run_boundary
    run_finite
    run_partial
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

echo "Regenerated ${selection} tables and figures."
