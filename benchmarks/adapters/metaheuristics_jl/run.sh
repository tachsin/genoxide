#!/usr/bin/env bash
# Runs the Metaheuristics.jl adapter:
#   run.sh <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#   run.sh --version   # the version of Metaheuristics.jl
#   run.sh --build     # installs the pinned packages of Manifest.toml and precompiles them
# JULIA (default ~/opt/julia/bin/julia) and JULIA_DEPOT_PATH (default ~/opt/julia-depot) can be overridden.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA="${JULIA:-$HOME/opt/julia/bin/julia}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/opt/julia-depot}"
# one thread (rule 4.3): one Julia thread, one GC mark thread and no concurrent GC sweep thread
# (--gcthreads=1,0, Julia 1.10+), BLAS with one thread
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "${1:-}" = "--build" ]; then
    exec "$JULIA" --project="$DIR" --threads=1 --gcthreads=1,0 -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
fi
exec "$JULIA" --project="$DIR" --threads=1 --gcthreads=1,0 "$DIR/bench.jl" "$@"
