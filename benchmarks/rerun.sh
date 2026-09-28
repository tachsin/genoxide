#!/usr/bin/env bash
# The full rerun of the benchmarks, on the machine the published results come from (WSL, pinned
# with pin-wsl.ps1), and its publishing into docs/benchmarks:
#
#   benchmarks/rerun.sh 0.9.0    # genoxide's version for `run.py versions`: a release on crates.io, or path
#
# 1. `run.py setup`: the .venv with the pinned Python libraries;
# 2. `run.py check`: every adapter against the rules;
# 3. `run.py`: every library in every scenario, 10 seeds, timed;
# 4. `run.py versions --genoxide <version>`: its instruction counts into genoxide-versions.json;
# 5. `run.py publish`: results.md, the charts, charts.json and results.json.xz into docs/benchmarks.
#
# It stops at the first step that fails. Everything it prints also goes to
# results/rerun-<timestamp>.log. PYTHON chooses the Python that runs run.py (default python3).
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <genoxide version, e.g. 0.9.0, or path>" >&2
  exit 2
fi
version=$1
python=${PYTHON:-python3}

cd "$(dirname "$0")"
mkdir -p results
log="results/rerun-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$log") 2>&1
echo "rerun of the benchmarks, genoxide $version, logged to benchmarks/$log"

# step <what> <run.py's arguments>
step() {
  echo
  echo "== $(date '+%F %T') $1"
  shift
  "$python" run.py "$@"
}

step "setup" setup
step "check" check
step "the timed run: every library in every scenario, 10 seeds" --seeds 10
step "genoxide $version's instruction counts" versions --genoxide "$version"
step "publish" publish

echo
echo "== $(date '+%F %T') done. Review and commit docs/benchmarks: results.md, results.json.xz, the SVGs,"
echo "charts.json and genoxide-versions.json."
