#!/usr/bin/env bash
# Runs the Evolutionary.jl adapter:
#   run.sh <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#   run.sh --version   # the version of Evolutionary.jl
#   run.sh --build     # installs the pinned packages of Manifest.toml and precompiles them
# JULIA (default ~/opt/julia/bin/julia) and JULIA_DEPOT_PATH (default ~/opt/julia-depot) can be overridden.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA="${JULIA:-$HOME/opt/julia/bin/julia}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/opt/julia-depot}"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "${1:-}" = "--build" ]; then
    exec "$JULIA" --project="$DIR" --threads=1 -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
fi
exec "$JULIA" --project="$DIR" --threads=1 "$DIR/bench.jl" "$@"
