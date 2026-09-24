#!/usr/bin/env bash
# Builds the openGA adapter (Linux / WSL): downloads openGA's single header, pinned to a commit,
# into ~/opt/openga and compiles bench.cpp into ~/bench-targets/openga/ga_bench_openga.
#   bash build.sh
#   ~/bench-targets/openga/ga_bench_openga onemax 100 matched 0 2 200000 60
#   ~/bench-targets/openga/ga_bench_openga --version
set -euo pipefail

# master of 2026-03-22, 16 commits after the last tag v1.0.5 (2020), which it fixes
COMMIT=f9b15e70600e20491504391dec6de5c64eb18913
VERSION="1.0.5+f9b15e7"
SHA256=d359c5b73fb55bdc473f1b1f2b944e37a1ab9db7816b4f1d13069e7220aee3cc

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCLUDE="${OPENGA_HOME:-$HOME/opt/openga}/$COMMIT"
OUT="${OPENGA_TARGET_DIR:-$HOME/bench-targets/openga}"

if [ ! -f "$INCLUDE/openGA.hpp" ]; then
    mkdir -p "$INCLUDE"
    curl -fsSL -o "$INCLUDE/openGA.hpp.part" \
        "https://raw.githubusercontent.com/Arash-codedev/openGA/$COMMIT/src/openGA.hpp"
    echo "$SHA256  $INCLUDE/openGA.hpp.part" | sha256sum -c --quiet -
    mv "$INCLUDE/openGA.hpp.part" "$INCLUDE/openGA.hpp"
fi

mkdir -p "$OUT"
g++ -O3 -march=native -std=c++17 -DNDEBUG -DOPENGA_VERSION="\"$VERSION\"" \
    -I "$INCLUDE" -o "$OUT/ga_bench_openga" "$HERE/bench.cpp"
