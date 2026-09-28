"""Benchmarks of evolutionary computation libraries: the matched suite, three problems with one
method each, the same method in every library (docs/benchmarks/rules.md, rule 6).

Usage:
    python run.py setup                      # create .venv and install the Python libraries
    python run.py                            # all scenarios, 10 seeds
    python run.py --quick                    # all scenarios, 3 seeds
    python run.py --max-seconds 10           # one time cap for every scenario, instead of each one's
    python run.py --seeds 5 --scenarios rosenbrock-10-matched
    python run.py --libraries deap pycma
    python run.py check                      # test the adapters against the rules, before a run,
                                             # --jobs scenarios at a time
    python run.py chart                      # redraw the charts of the latest results: the published
                                             # docs/benchmarks/results.json.xz, or a newer run
    python run.py publish                    # the latest run into docs/benchmarks: results.md, the
                                             # charts, charts.json, runs/ and results.json.xz
    python run.py versions --genoxide 0.8.0  # count the CPU instructions of genoxide 0.8.0's runs
                                             # with Callgrind into docs/benchmarks/genoxide-versions.json
                                             # (--genoxide path: this repository's genoxide), --jobs
                                             # at a time; unpinned is fine
    python run.py --libraries genoxide --update
                                             # rerun one library, keep the others' results of the
                                             # published run (or of --update <file>), if a
                                             # reference run still takes the file's time
                                             # (--allow-drift: even if not)
    python run.py --version-label genoxide=0.7.0
                                             # record genoxide as 0.7.0, e.g. before the release
                                             # PR bumps Cargo.toml
    python run.py outdated                   # the pinned, published and latest versions of each
                                             # library (--issue: keep a GitHub issue of the newer
                                             # ones with gh, --dry-run: print what it would do)

Results are written to results/<timestamp>.json (all runs), results/latest.md (table) and
results/charts/ (charts: *.svg, and charts.json, their numbers for the project site's interactive
charts, and runs/<scenario>.json, each method's runs, output and adapter code, for the site's
details of a bar); `python run.py publish` puts them in docs/benchmarks. `python run.py versions` compares
genoxide's versions by the CPU instructions of the same runs, counted with Callgrind (Linux,
Valgrind), in docs/benchmarks/genoxide-versions.json and the genoxide_versions chart.
"""

import argparse
import concurrent.futures
import datetime
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
GENOXIDE_ADAPTER = ROOT / "adapters" / "genoxide"
# genoxide's Python package, built with maturin in release mode and installed into .venv
GENOXIDE_PYTHON = ROOT.parent / "python"
GENOXIDE_PYTHON_BUILD = (
    # rebuilt every time: uv would otherwise keep a wheel built from older Rust sources
    ["uv", "pip", "install", "--quiet", "--python", str(VENV_PYTHON), "--reinstall-package", "genoxide",
     str(GENOXIDE_PYTHON)]
    if shutil.which("uv")
    else [str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "--force-reinstall", "--no-deps",
          str(GENOXIDE_PYTHON)]
)
# the published results: the tables, the charts and the run they're drawn from, which `run.py
# publish` writes compressed (xz), so anyone can redraw them, and `chart` and `--update` read by default
DOCS = ROOT.parent / "docs" / "benchmarks"
PUBLISHED_RESULTS = DOCS / "results.json.xz"
# a run's file in results/: its timestamp
RUN_FILE = re.compile(r"\d{8}-\d{6}\.json")

# Each adapter prints one JSON line per seed of the scenario's method, with the same command line:
#   <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
# "release" is the registry where `run.py outdated` looks for the library's latest release
# (outdated.py).
ADAPTERS = {
    "genoxide": {
        "build": ["cargo", "build", "--release", "--quiet", "--manifest-path", str(GENOXIDE_ADAPTER / "Cargo.toml")],
        "command": [str(GENOXIDE_ADAPTER / "target" / "release" / "ga_bench_genoxide")],
        # the genoxide of this repository: its version and commit
        "version": ("cargo", "genoxide", GENOXIDE_ADAPTER),
        "language": "Rust",
    },
    "genoxide_python": {
        # genoxide's Python package: its Rust algorithms, calling Python fitness functions
        "build": GENOXIDE_PYTHON_BUILD,
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "genoxide_python" / "bench.py")],
        # the package of this repository: its version and commit
        "version": ("python", "genoxide"),
        "language": "Rust via Python",
    },
    "deap": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "deap" / "bench.py")],
        "version": ("python", "deap"),
        "language": "Python",
        "release": ("pypi", "deap"),
    },
    "pygad": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "pygad" / "bench.py")],
        "version": ("python", "pygad"),
        "language": "Python",
        "release": ("pypi", "pygad"),
    },
    "pymoo": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "pymoo" / "bench.py")],
        "version": ("python", "pymoo"),
        "language": "Python",
        "release": ("pypi", "pymoo"),
    },
    "radiate": {
        "build": ["cargo", "build", "--release", "--quiet", "--manifest-path",
                  str(ROOT / "adapters" / "radiate" / "Cargo.toml")],
        "command": [str(ROOT / "adapters" / "radiate" / "target" / "release" / "ga_bench_radiate")],
        "version": ("cargo", "radiate", ROOT / "adapters" / "radiate"),
        "language": "Rust",
        "release": ("crates", "radiate"),
    },
    "pycma": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "pycma" / "bench.py")],
        "version": ("python", "cma"),
        "language": "Python",
        "release": ("pypi", "cma"),
    },
    "scipy": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "scipy" / "bench.py")],
        "version": ("python", "scipy"),
        "language": "Python",
        "release": ("pypi", "scipy"),
    },
    "pygmo": {
        # pagmo's C++ algorithms, calling the Python fitness function
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "pygmo" / "bench.py")],
        "version": ("python", "pygmo"),
        "language": "C++ via Python",
        "release": ("pypi", "pygmo"),
    },
    "jmetal": {
        # the JDK and the pinned jars are downloaded into ~/opt, and compiled into ~/bench-targets
        "build": ["bash", str(ROOT / "adapters" / "jmetal" / "build.sh")],
        "command": ["bash", str(ROOT / "adapters" / "jmetal" / "run.sh")],
        "version": ("command", ["bash", str(ROOT / "adapters" / "jmetal" / "run.sh"), "--version"]),
        "language": "Java",
        "release": ("maven", "org.uma.jmetal", "jmetal-core"),
    },
    "evolutionary_jl": {
        # installs the pinned packages and precompiles, then runs with one thread
        "build": ["bash", str(ROOT / "adapters" / "evolutionary_jl" / "run.sh"), "--build"],
        "command": ["bash", str(ROOT / "adapters" / "evolutionary_jl" / "run.sh")],
        "version": ("command", ["bash", str(ROOT / "adapters" / "evolutionary_jl" / "run.sh"), "--version"]),
        "language": "Julia",
        "release": ("julia", "Evolutionary"),
    },
    "metaheuristics_jl": {
        "build": ["bash", str(ROOT / "adapters" / "metaheuristics_jl" / "run.sh"), "--build"],
        "command": ["bash", str(ROOT / "adapters" / "metaheuristics_jl" / "run.sh")],
        "version": ("command", ["bash", str(ROOT / "adapters" / "metaheuristics_jl" / "run.sh"), "--version"]),
        "language": "Julia",
        "release": ("julia", "Metaheuristics"),
    },
}

# (problem, size, mode, max_evaluations, max_seconds). A run stops at the target, at max_evaluations
# or at max_seconds, its time cap, whichever comes first. The suite is matched: each problem has
# one method, defined in rule 6 of docs/benchmarks/rules.md, and every library that has its own
# implementation of it runs it, set to that definition. The mode is "matched" in every scenario.
SCENARIOS = [
    ("onemax", 1000, "matched", 2_000_000, 60),
    # no target: every run uses this fixed budget (rule 6.3), measured by the time for it and the
    # error at the end
    ("rastrigin", 30, "matched", 300_000, 60),
    ("rosenbrock", 10, "matched", 500_000, 60),
]
BUDGETS = {f"{problem}-{size}-{mode}": budget for problem, size, mode, budget, _ in SCENARIOS}
CAPS = {f"{problem}-{size}-{mode}": cap for problem, size, mode, _, cap in SCENARIOS}
# each scenario's method (rule 6), as the charts name it
METHODS = {"onemax-1000-matched": "GA", "rastrigin-30-matched": "DE/rand/1/bin", "rosenbrock-10-matched": "CMA-ES"}
# the scenarios without a target: a fixed budget, measured by the time for it and the error at the end
FIXED_BUDGET = {"rastrigin-30-matched"}


def scenario_name(problem, size, mode):
    return f"{problem}-{size}-{mode}"


def setup():
    """Create .venv with the pinned Python libraries from requirements.txt."""
    requirements = str(ROOT / "requirements.txt")
    if shutil.which("uv"):
        # uv doesn't need pip or ensurepip, which some system Pythons leave out
        if not VENV_PYTHON.exists():
            subprocess.run(["uv", "venv", str(VENV)], check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(VENV_PYTHON), "-r", requirements], check=True)
        return
    if not VENV_PYTHON.exists():
        print(f"creating {VENV} ...", flush=True)
        venv.create(VENV, with_pip=True)
    subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "-r", requirements], check=True)


def library_version(kind, package, adapter=None, label=None):
    """The version of a library, or `label` in its place; genoxide's has its commit too."""
    if label:
        version = label
    elif kind == "command":
        # a command that prints the version, e.g. an adapter's --version
        return subprocess.run(package, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    elif kind == "python":
        version = subprocess.run(
            [str(VENV_PYTHON), "-c", f"import importlib.metadata as m; print(m.version('{package}'))"],
            capture_output=True, text=True, check=True, cwd=ROOT,
        ).stdout.strip()
    else:
        metadata = json.loads(subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--manifest-path", str(adapter / "Cargo.toml")],
            capture_output=True, text=True, check=True,
        ).stdout)
        version = next(p["version"] for p in metadata["packages"] if p["name"] == package)
    if package == "genoxide":
        # genoxide, in Rust or Python, is this repository's: its commit too
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT,
        ).stdout.strip()
        version = f"{version}+{commit}" if commit else version
    return version


# A solver whose first EARLY_SEEDS runs all hit the time cap without reaching the target runs no
# more seeds (rule 5.3): the others would take the whole cap each, for the same result. Some
# adapters (e.g. PyGAD's) skip those seeds themselves; stop_early applies the same rule to every
# adapter's runs.
EARLY_SEEDS = 3
# a run that took this share of the cap was stopped by it
CAPPED = 0.98
# the fewest runs reaching the target that give an ERT (rule 8.1); with fewer, the results show how
# many reached it
ERT_REACHED = 3


def scenario_caps(max_seconds, runs=()):
    """Each scenario's time cap, from a results file's "max_seconds": a cap per scenario, or one
    number for every scenario in the files from before the caps were per scenario."""
    if isinstance(max_seconds, dict):
        return max_seconds
    names = set(CAPS) | {scenario_name(run.get("problem"), run.get("size"), run.get("mode")) for run in runs}
    return {name: max_seconds for name in names}


def run_cap(run, caps):
    """The time cap of a run's scenario."""
    return caps[scenario_name(run.get("problem"), run.get("size"), run.get("mode"))]


def stop_early(runs, max_seconds):
    """The runs without the seeds after EARLY_SEEDS of a solver whose first EARLY_SEEDS runs all
    hit the time cap."""
    seeds = sorted({run.get("seed") for run in runs if isinstance(run.get("seed"), int)})[:EARLY_SEEDS]
    stopped = set()
    for solver in {run.get("solver") for run in runs}:
        first = [run for run in runs if run.get("solver") == solver and run.get("seed") in seeds]
        if len(first) == EARLY_SEEDS and all(capped(run, max_seconds) for run in first):
            stopped.add(solver)
    return [run for run in runs if run.get("solver") not in stopped or run.get("seed") in seeds]


def capped(run, max_seconds):
    """Whether the time cap stopped a run: it took the cap without reaching the target within the
    cap or using its evaluation budget."""
    budget = BUDGETS.get(scenario_name(run.get("problem"), run.get("size"), run.get("mode")))
    return (run.get("time_s", 0) >= CAPPED * max_seconds and not first_hit(run, max_seconds)
            and (budget is None or run.get("evaluations", 0) < budget))


def run_outcome(run, max_seconds):
    """How a run ended, for the log."""
    if run.get("success"):
        return "target reached" if first_hit(run, max_seconds) else "target reached after the time cap: not reached"
    return "not reached, stopped by the time cap" if capped(run, max_seconds) else "not reached"


def run_adapter(adapter, problem, size, mode, seeds, max_evaluations, max_seconds):
    """The runs of an adapter in a scenario. Each is validated with the checks of `run.py check`
    (check.check_run); a run that fails keeps its failures in "invalid" and is left out of every
    summary."""
    import check
    command = adapter["command"] + [
        problem, str(size), mode, "0", str(seeds - 1), str(max_evaluations), str(float(max_seconds)),
    ]
    runs = []
    # run from this folder, so a library repository checked out next to it can't shadow a package.
    # Each run is logged as the adapter prints it, e.g. for a live view of the progress; stderr goes
    # to a file, so an adapter that writes a lot there can't block.
    with tempfile.TemporaryFile(mode="w+") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, text=True, cwd=ROOT)
        for line in process.stdout:
            if line.strip():
                run = json.loads(line)
                runs.append(run)
                failures = check.check_run(run, problem, size, max_evaluations, max_seconds)
                if failures:
                    run["invalid"] = failures
                print(f"  run: {run.get('library')} {run.get('solver')} seed {run.get('seed')}: "
                      f"{run.get('time_s', 0):.3f} s, "
                      + (f"INVALID: {'; '.join(failures)}" if failures else run_outcome(run, max_seconds)),
                      flush=True)
        if process.wait() != 0:
            stderr.seek(0)
            print(stderr.read(), file=sys.stderr)
            raise SystemExit(f"adapter failed: {' '.join(command)}")
    return stop_early(runs, max_seconds)


def valid(runs):
    """The runs that passed validation: the only ones summarized."""
    return [run for run in runs if not run.get("invalid")]


# genoxide's versions, compared by the CPU instructions of the same runs (rule 10 of
# docs/benchmarks/rules.md). The same adapter source is built against each version of genoxide:
# a release from crates.io, or this repository's genoxide. In every scenario, each of the
# adapter's solvers makes one run with seed VERSIONS_SEED, to its target or its budget, without a
# time cap, alone in a process under Callgrind (GENOXIDE_BENCH_SOLVER names it). The adapter's
# startup, a process that runs no solver, is subtracted. genoxide gives the same run for a seed on
# every platform, and Callgrind's counts don't depend on the machine's load: a count changes only
# with genoxide, the adapter, the Rust toolchain or Valgrind. The history of every version measured
# is committed in VERSIONS_FILE, and the genoxide_versions chart is drawn from it.
VERSIONS_FILE = ROOT.parent / "docs" / "benchmarks" / "genoxide-versions.json"
VERSIONS_SEED = 0
# no time cap: under Callgrind a run takes about 50 times as long, and it must end where it ends
# without Callgrind
VERSIONS_SECONDS = 10_000_000.0
SOLVER_VARIABLE = "GENOXIDE_BENCH_SOLVER"
# a solver name no adapter has: the process runs no solver, for the startup's instructions
NO_SOLVER = "none"
# the copies of the adapter built against genoxide's releases, one folder per version
VERSION_BUILDS = GENOXIDE_ADAPTER / "target" / "versions"
# where the release dates come from
CRATES_API = "https://crates.io/api/v1/crates/genoxide"


def version_key(version):
    """A version's numbers, to sort them: 0.10.0 after 0.9.1; a build of this repository's genoxide
    (0.8.0+abc1234) after its release."""
    base, _, build = version.partition("+")
    return tuple(int(number) for number in re.findall(r"\d+", base)), bool(build)


def build_genoxide(version):
    """(command, version, source) of the genoxide adapter built against `version` of genoxide, a
    release on crates.io, or "path" for this repository's genoxide; None if the adapter doesn't
    compile against it. Each builds a copy of the adapter in VERSION_BUILDS whose Cargo.toml asks
    for `genoxide = "=version"` or this repository's path, with the adapter's Cargo.lock and
    profile: nothing in the repository changes, and every version builds the same way."""
    repository = version == "path"
    if not repository and not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit(f"--genoxide {version}: expected a release, e.g. 0.8.0, or path")
    folder = VERSION_BUILDS / ("repository" if repository else version)
    (folder / "src").mkdir(parents=True, exist_ok=True)
    for source in (GENOXIDE_ADAPTER / "src").iterdir():
        shutil.copy2(source, folder / "src" / source.name)
    # the adapter's locked dependencies, as far as the version allows
    shutil.copy2(GENOXIDE_ADAPTER / "Cargo.lock", folder / "Cargo.lock")
    dependency = f"path = {json.dumps(ROOT.parent.as_posix())}" if repository else f'version = "={version}"'
    manifest, replaced = re.subn(r'(?m)^genoxide = \{ path = "[^"]*"', lambda _: f"genoxide = {{ {dependency}",
                                 (GENOXIDE_ADAPTER / "Cargo.toml").read_text(encoding="utf-8"))
    if replaced != 1:
        raise SystemExit(f"{GENOXIDE_ADAPTER / 'Cargo.toml'}: no `genoxide = {{ path = ... }}` dependency to replace")
    # a crate of its own, outside any workspace
    (folder / "Cargo.toml").write_text(manifest.rstrip() + "\n\n[workspace]\n", encoding="utf-8", newline="\n")
    print(f"building the genoxide adapter against "
          + ("this repository's genoxide" if repository else f"genoxide {version} from crates.io") + " ...", flush=True)
    completed = subprocess.run(["cargo", "build", "--release", "--quiet", "--manifest-path", str(folder / "Cargo.toml")],
                               capture_output=True, text=True)
    if completed.returncode != 0:
        errors = [line for line in completed.stderr.splitlines() if line.startswith("error")]
        print(f"genoxide {version}: the adapter doesn't compile against it, so it's skipped: "
              + ("; ".join(dict.fromkeys(errors[:8])) or completed.stderr.strip()[-500:]), flush=True)
        return None
    command = [str(folder / "target" / "release" / "ga_bench_genoxide")]
    if not repository:
        return command, version, "crates.io"
    # this repository's genoxide: its version and commit, marked when its sources have uncommitted changes
    label = library_version("cargo", "genoxide", folder)
    if "+" not in label:
        # without its commit (git can't read the repository), still apart from the release
        label += "+repository"
    changed = subprocess.run(["git", "status", "--porcelain", "--", "src", "Cargo.toml", "Cargo.lock"],
                             capture_output=True, text=True, cwd=ROOT.parent).stdout.strip()
    return command, f"{label}.modified" if changed and "+" in label else label, "repository"


def version_runs(command, scenario, solver=None, callgrind=False):
    """(runs, instructions) of the adapter `command` in `scenario` with seed VERSIONS_SEED and no
    time cap: every solver, or only `solver` (NO_SOLVER: none); with `callgrind`, under Callgrind,
    the instructions of the whole process, otherwise None."""
    problem, size, mode, budget, _ = scenario
    environment = {name: value for name, value in os.environ.items() if name != SOLVER_VARIABLE}
    if solver:
        environment[SOLVER_VARIABLE] = solver
    prefix = ["valgrind", "--tool=callgrind", "--callgrind-out-file=/dev/null"] if callgrind else []
    full = prefix + command + [problem, str(size), mode, str(VERSIONS_SEED), str(VERSIONS_SEED), str(budget),
                               str(VERSIONS_SECONDS)]
    completed = subprocess.run(full, capture_output=True, text=True, cwd=ROOT, env=environment)
    match = re.search(r"Collected\s*:\s*(\d+)", completed.stderr)
    if completed.returncode != 0 or (callgrind and not match):
        print(completed.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"failed: {SOLVER_VARIABLE}={solver or ''} {' '.join(full)}")
    runs = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    return runs, int(match.group(1)) if callgrind else None


def same_run(a, b):
    """Whether two runs of the same solver and seed made the same search."""
    return all(a.get(key) == b.get(key) for key in ("solver", "evaluations", "generations", "best", "solution"))


def measure_version(command, jobs=None):
    """{scenario: {"startup": instructions, "methods": {solver: {...}}}} of the adapter `command`
    (VERSIONS_SEED, rule 10). Each solver's entry: its instructions without the startup, its
    evaluations, whether it reached the target and its best value. The runs go in parallel, `jobs`
    at a time."""
    import check
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=jobs or os.cpu_count())
    try:
        # every solver without Callgrind first: which solvers run, and the runs Callgrind's must repeat
        native = {scenario_name(*scenario[:3]): pool.submit(version_runs, command, scenario) for scenario in SCENARIOS}
        native = {name: future.result()[0] for name, future in native.items()}
        # the longest runs first
        longest = sorted(((scenario, run["solver"], run["time_s"]) for scenario in SCENARIOS
                        for run in native[scenario_name(*scenario[:3])]), key=lambda job: -job[2])
        counts = {(scenario_name(*scenario[:3]), solver): pool.submit(version_runs, command, scenario, solver, True)
                  for scenario, solver, _ in longest}
        counts |= {(scenario_name(*scenario[:3]), NO_SOLVER): pool.submit(version_runs, command, scenario, NO_SOLVER,
                                                                          True) for scenario in SCENARIOS}
        measured = {}
        for problem, size, mode, budget, _ in SCENARIOS:
            name = scenario_name(problem, size, mode)
            if not native[name]:
                # a scenario this version's adapter doesn't run
                continue
            runs, startup = counts[(name, NO_SOLVER)].result()
            if runs:
                raise SystemExit(f"{name}: the adapter ran {len(runs)} solvers with {SOLVER_VARIABLE}={NO_SOLVER}")
            methods = {}
            for run in native[name]:
                counted, instructions = counts[(name, run["solver"])].result()
                if len(counted) != 1 or not same_run(counted[0], run):
                    raise SystemExit(f"{name}: {run['solver']} didn't make the same run under Callgrind as without it")
                entry = {"instructions": instructions - startup, "evaluations": run["evaluations"],
                         # a fixed budget has no target: null
                         "reached": None if name in FIXED_BUDGET else bool(run["success"]), "best": run["best"]}
                failures = check.check_run(run, problem, size, budget, VERSIONS_SECONDS)
                if failures:
                    # recorded, not left out: the history shows what the version did
                    entry["invalid"] = failures
                    print(f"warning: {name}: {'; '.join(failures)}", flush=True)
                methods[run["solver"]] = entry
                print(f"{name}: {run['solver']}: {format_count(entry['instructions'])} instructions, "
                      f"{format_count(run['evaluations'])} evaluations"
                      + ("" if name in FIXED_BUDGET else ", target reached" if run["success"] else ", target not reached"),
                      flush=True)
            measured[name] = {"startup": startup, "methods": methods}
    finally:
        pool.shutdown(cancel_futures=True)
    return measured


def release_date(version):
    """The day a release of genoxide was published on crates.io, or None if it can't be read."""
    import outdated
    try:
        return json.loads(outdated.fetch(f"{CRATES_API}/{version}"))["version"]["created_at"][:10]
    except Exception as error:  # noqa: BLE001: the date is only informative
        print(f"warning: the release date of genoxide {version}: {error}", flush=True)
        return None


def tool_version(command):
    return subprocess.run(command, capture_output=True, text=True).stdout.strip()


def load_versions(history):
    """The history file of genoxide's versions: {"format": 1, "seed": ..., "versions": [...]}."""
    if history.exists():
        return json.loads(history.read_text(encoding="utf-8"))
    return {"format": 1, "seed": VERSIONS_SEED, "versions": []}


def save_versions(data, history):
    """The history file, a version per block, sorted by version."""
    data["versions"].sort(key=lambda row: version_key(row["version"]))
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def measure_versions(versions, history, jobs, charts, png=False):
    """Measures each of `versions` (rule 10) into the history file, in place of its previous row,
    and redraws the genoxide_versions chart into `charts`. A version the adapter doesn't compile
    against is reported and skipped."""
    data = load_versions(history)
    skipped = []
    for version in versions:
        built = build_genoxide(version)
        if built is None:
            skipped.append(version)
            continue
        command, label, source = built
        print(f"genoxide {label}: counting the instructions of every run with Callgrind ...", flush=True)
        row = {
            "version": label,
            "source": source,
            "released": release_date(label) if source == "crates.io" else None,
            "measured": datetime.date.today().isoformat(),
            "machine": describe_platform(),
            "rustc": tool_version(["rustc", "--version"]),
            "valgrind": tool_version(["valgrind", "--version"]),
            "scenarios": measure_version(command, jobs),
        }
        # one row per version, and at most one of this repository's genoxide, until a newer release
        base = version_key(label)[0]
        data["versions"] = [
            other for other in data["versions"]
            if other["version"] != label
            and not (source == "repository" and other.get("source") == "repository")
            and not (source == "crates.io" and other.get("source") == "repository"
                     and version_key(other["version"])[0] < base)
        ] + [row]
        data["seed"] = VERSIONS_SEED
        save_versions(data, history)
        print(f"genoxide {label}: written to {history}", flush=True)
    if skipped:
        print(f"skipped, the adapter doesn't compile against them: genoxide {', '.join(skipped)}", flush=True)
    draw_versions_of(history, charts, png)


def median(values):
    return statistics.median(values) if values else None


def format_seconds(seconds):
    if seconds is None:
        return "-"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def format_number(value):
    if value is None:
        return "-"
    if isinstance(value, float) and not value.is_integer() and abs(value) < 1000:
        return f"{value:.4g}"
    # from 1000, rounded: 13,027, not 1.303e+04
    return f"{round(value):,}"


def format_count(value):
    """A count, or a median of counts, which can be halfway between two: rounded."""
    return "-" if value is None else f"{round(value):,}"


def gap_to_optimum(run):
    """The distance of a run's best value from the optimum: 0 when it's optimal."""
    return run["size"] - run["best"] if run["problem"] == "onemax" else run["best"]


def first_hit(run, max_seconds):
    """(evaluations, seconds) at the run's first hit of the target, or None. A hit after the time
    cap counts as not reached (rule 3.3). A run from before first_hit was recorded stopped at the
    target: its end."""
    if not run.get("success"):
        return None
    hit = run.get("first_hit")
    evaluations, seconds = (hit["evaluations"], hit["time_s"]) if hit else (run["evaluations"], run["time_s"])
    return (evaluations, seconds) if seconds <= max_seconds else None


def expected_to_target(group, max_seconds):
    """The expected running time (ERT, Hansen et al., COCO) of a solver in evaluations and in seconds:
    what all its runs spent, up to the first hit in the runs that reached the target and in full in
    the others, divided by the number of runs that reached it. (None, None) if fewer than
    ERT_REACHED did."""
    hits = [first_hit(run, max_seconds) for run in group]
    reached = sum(1 for hit in hits if hit)
    if reached < ERT_REACHED:
        return None, None
    evaluations = sum(hit[0] if hit else run["evaluations"] for run, hit in zip(group, hits))
    seconds = sum(hit[1] if hit else run["time_s"] for run, hit in zip(group, hits))
    return evaluations / reached, seconds / reached


def budget_share(runs):
    """The median share of its scenario's evaluation budget that a run used, over `runs`, or None."""
    shares = [run["evaluations"] / budget for run in runs
              if (budget := BUDGETS.get(scenario_name(run["problem"], run["size"], run["mode"])))]
    return median(shares)


def solver_groups(runs, caps, split):
    """The valid runs per (scenario, library, solver), and with `split` per whether the time cap
    stopped them too."""
    groups = {}
    for run in valid(runs):
        key = (scenario_name(run["problem"], run["size"], run["mode"]), run["library"], run["solver"])
        if split:
            key += (capped(run, run_cap(run, caps)),)
        groups.setdefault(key, []).append(run)
    return groups


def capped_runs(group, caps):
    """The runs of a group that the time cap stopped."""
    return [run for run in group if capped(run, run_cap(run, caps))]


def summarize(runs, caps, split=False):
    """The results of the valid runs, per scenario and solver. `caps` has each
    scenario's time cap. With `split`, the runs the time cap stopped are a row of their own, with
    "ended_on_cap" true."""
    def evaluations_per_second(group):
        return sum(run["evaluations"] for run in group) / max(sum(run["time_s"] for run in group), 1e-9)

    # reference for the throughput ratio: DEAP's method in the same scenario, all its runs
    reference = {
        scenario: evaluations_per_second(group)
        for (scenario, library, solver), group in solver_groups(runs, caps, False).items()
        if library == "deap"
    }

    rows = []
    for key, group in solver_groups(runs, caps, split).items():
        scenario, library, solver = key[:3]
        cap = caps[scenario]
        stopped = capped_runs(group, caps)
        rate = evaluations_per_second(group)
        # how far each run's best is from the optimum (rule 8.1): every problem's optimum is 0,
        # except OneMax's, all ones
        gaps = sorted(gap_to_optimum(run) for run in group)
        ert_evaluations, ert_time = expected_to_target(group, cap)
        rows.append({
            "scenario": scenario,
            "library": library,
            "solver": solver,
            **({"ended_on_cap": key[3]} if split else {}),
            "runs": len(group),
            # a first hit after the cap counts as not reached (rule 3.3)
            "reached": sum(1 for run in group if first_hit(run, cap)),
            # runs the time cap stopped, not the budget: limited by speed, not by the search
            "capped": len(stopped),
            "capped_share": budget_share(stopped),
            "ert_time": ert_time,
            "ert_evaluations": ert_evaluations,
            "median_time": median([run["time_s"] for run in group]),
            # a fixed budget's measures (FIXED_BUDGET): the median time of the runs that used the
            # whole budget, and the runs a library ended before it (rules 2.2, 8.4)
            "budget_time": median([run["time_s"] for run in group
                                   if not run.get("ended_by") and run["evaluations"] >= BUDGETS.get(scenario, math.inf)]),
            "ended_early": sum(1 for run in group if run.get("ended_by")),
            "median_evaluations": median([run["evaluations"] for run in group]),
            "median_best": median([run["best"] for run in group]),
            "median_gap": median(gaps),
            "best_gap": gaps[0],
            "worst_gap": gaps[-1],
            "evaluations_per_second": rate,
            "throughput_vs_deap": rate / reference[scenario] if scenario in reference and library != "deap" else None,
        })
    rows.sort(key=lambda row: (row["scenario"], row["library"], row["solver"], row.get("ended_on_cap", False)))
    return rows


def coverage_table(runs, libraries):
    """Which library ran which scenario: a library missing from a scenario can't run it (see
    docs/benchmarks/notes.md)."""
    def where(run):
        return run.get("library"), scenario_name(run.get("problem"), run.get("size"), run.get("mode"))

    ran = {where(run) for run in valid(runs)}
    printed = {where(run) for run in runs}
    scenarios = [scenario_name(*scenario[:3]) for scenario in SCENARIOS]
    names = [name for name in list(ADAPTERS) + sorted(set(libraries) - set(ADAPTERS)) if name in libraries]
    lines = ["| Library | " + " | ".join(scenario_title(scenario) for scenario in scenarios) + " |",
             "|---|" + "---|" * len(scenarios)]
    for name in names:
        # ✗: every run the library printed there was invalid; blank: the scenario awaits the next run
        cells = ["✓" if (name, scenario) in ran else "✗" if (name, scenario) in printed
                 else "" if not any(where_ == scenario for _, where_ in printed) else "–"
                 for scenario in scenarios]
        lines.append(f"| {LIBRARY_NAMES.get(name, name)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_table(rows):
    """The results (rule 8.1): a table of the scenarios with a target, then one of those with a fixed
    budget. The time and evaluations to target are the expected running time (ERT), as in the
    charts; the median evaluations and the distance to the optimum at the end are of all runs."""
    target_rows = [row for row in rows if row["scenario"] not in FIXED_BUDGET]
    budget_rows = [row for row in rows if row["scenario"] in FIXED_BUDGET]
    lines = []
    if target_rows:
        lines += [
            "| Scenario | Library / method | Reached the target | Stopped by the time cap | Expected time to target "
            "| Expected evaluations to target | Median evaluations "
            "| Distance to the optimum at the end: median (best to worst) | Evaluations/s | Throughput vs DEAP |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    for row in target_rows:
        ratio = row["throughput_vs_deap"]
        ratio = "-" if ratio is None else (f"{ratio:.1f}×" if ratio < 10 else f"{ratio:.0f}×")
        # no ERT from fewer than ERT_REACHED runs that reached the target: how many did instead
        too_few = f"{row['reached']}/{row['runs']} reached"
        ended = f", {row['ended_early']} ended early" if row.get("ended_early") else ""
        lines.append(
            f"| {row['scenario']} | {row['library']} / {row['solver']} "
            f"| {row['reached']} of {row['runs']}{ended} "
            f"| {format_capped(row)} "
            f"| {format_seconds(row['ert_time']) if row['ert_time'] is not None else too_few} "
            f"| {format_count(row['ert_evaluations']) if row['ert_evaluations'] is not None else too_few} "
            f"| {format_count(row['median_evaluations'])} "
            f"| {format_gap(row, 'median_gap')} ({format_gap(row, 'best_gap')} to {format_gap(row, 'worst_gap')}) "
            f"| {format_count(row['evaluations_per_second'])} "
            f"| {ratio} |"
        )
    if target_rows and budget_rows:
        lines.append("")
    if budget_rows:
        lines += [
            "| Scenario (fixed budget) | Library / method | Runs | Ended early by the library "
            "| Stopped by the time cap | Median time for the budget | Median evaluations "
            "| Error at the end: median (best to worst) | Evaluations/s |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    for row in budget_rows:
        lines.append(
            f"| {row['scenario']} | {row['library']} / {row['solver']} "
            f"| {row['runs']} "
            f"| {row.get('ended_early', 0)} "
            f"| {format_capped(row)} "
            f"| {format_seconds(row.get('budget_time'))} "
            f"| {format_count(row['median_evaluations'])} "
            f"| {format_gap(row, 'median_gap')} ({format_gap(row, 'best_gap')} to {format_gap(row, 'worst_gap')}) "
            f"| {format_count(row['evaluations_per_second'])} |"
        )
    return "\n".join(lines)


def format_share(share):
    """A share of the budget, rounded down: a run the time cap stopped didn't use all of it."""
    return "<1%" if share < 0.01 else f"{int(share * 100)}%"


def format_capped(row):
    """The runs the time cap stopped, and the median share of the budget they used."""
    share = row.get("capped_share")
    return f"{row.get('capped', 0)}" + (f" (at {format_share(share)} of the budget)"
                                        if row.get("capped") and share is not None else "")


def format_gap(row, key):
    # summaries from before rule 8.1 have no distances
    return format_number(row[key]) if key in row else "-"


def invalid_list(runs):
    """The runs that failed validation, with why: excluded from every table and chart."""
    invalid = [run for run in runs if run.get("invalid")]
    lines = [f"Invalid runs, left out of every table and chart: {len(invalid)}"]
    if invalid:
        lines.append("")
        for run in invalid:
            lines.append(f"- {run.get('library')} / {run.get('solver')}, "
                         f"{scenario_name(run.get('problem'), run.get('size'), run.get('mode'))}, seed {run.get('seed')}: "
                         + "; ".join(run["invalid"]))
    return "\n".join(lines)


LIBRARY_NAMES = {"genoxide": "genoxide", "genoxide_python": "genoxide (Python)", "deap": "DEAP", "pygad": "PyGAD",
                 "pymoo": "pymoo", "radiate": "radiate", "pycma": "pycma", "scipy": "SciPy",
                 "pygmo": "pygmo", "jmetal": "jMetal", "evolutionary_jl": "Evolutionary.jl",
                 "metaheuristics_jl": "Metaheuristics.jl"}
SOLVER_NAMES = {"ga": "GA", "de": "DE", "cma_es": "CMA-ES"}
PROBLEM_NAMES = {"onemax": "OneMax", "rastrigin": "Rastrigin", "rosenbrock": "Rosenbrock"}
# a scenario of the suite that the results file has no runs of, e.g. one added since the run
PENDING = "Awaiting the next run"
# the width of its panel, in bars
PENDING_BARS = 5
# what "matched" means, in the charts' subtitles
MATCHED = "matched: one method per problem, the same in every library, with its own implementation"
GENOXIDE_COLOR = "#ce422b"
# genoxide's Python package: a lighter red
GENOXIDE_PYTHON_COLOR = "#ec8b78"
GENOXIDE_COLORS = {"genoxide": GENOXIDE_COLOR, "genoxide_python": GENOXIDE_PYTHON_COLOR}
# the other libraries, in the order of ADAPTERS: a qualitative palette without red
PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d", "#72b7b2", "#e0b000", "#ff9da6",
           "#79706e", "#1b9e77", "#7570b3", "#a6761d", "#e7298a", "#66a61e", "#666666"]


def library_colors(libraries):
    """A color per library, the same in every chart."""
    colors, others = {}, iter(PALETTE)
    for name in list(ADAPTERS) + sorted(set(libraries) - set(ADAPTERS)):
        colors[name] = GENOXIDE_COLORS[name] if name in GENOXIDE_COLORS else next(others, "#888888")
    return colors


def label(library, solver):
    """A bar's label: the library, since a scenario has one method (its name is in the panel's
    title)."""
    return LIBRARY_NAMES.get(library, library)


def scenario_title(scenario):
    """A scenario's title with its method, e.g. "Rastrigin 30: DE/rand/1/bin"."""
    problem, size, mode = scenario.split("-")
    title = f"{PROBLEM_NAMES.get(problem, problem)} {size}"
    return f"{title}: {METHODS[scenario]}" if scenario in METHODS else f"{title} ({mode})"


def short_number(value):
    """3 significant digits and a suffix, e.g. 1.23k or 45.6M. The rounding can reach the next
    suffix: 999,600 is 1M, not 1e+03k."""
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if float(f"{value:.3g}") >= limit:
            return f"{value / limit:.3g}{suffix}"
    return f"{value:.3g}"


def results_date(results):
    """The date of a results file: stored, or from its timestamp."""
    if results.get("date"):
        return results["date"]
    stamp = results.get("timestamp", "")
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}" if len(stamp) >= 8 else ""


def significant(value, digits=6):
    """A number for charts.json: 6 significant digits, more than any chart shows."""
    return None if value is None else float(f"{value:.{digits}g}")


def draw_charts(results, out_dir, formats=("svg",), history=VERSIONS_FILE):
    """Bar charts of a results file: time and evaluations to target and the distance to the optimum,
    a panel per scenario of SCENARIOS, a bar per library (a scenario without runs in the file is a
    panel awaiting the next run); and, from the `history` file of genoxide's versions, the
    instructions of their runs (draw_versions_chart). Their numbers go to charts.json beside them,
    for the interactive charts of the project site: recorded as each chart draws them, so the file
    and the charts can't disagree."""
    plt = pyplot()
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, LogLocator

    out_dir.mkdir(parents=True, exist_ok=True)
    versions = results.get("versions", {})
    languages = results.get("languages") or {name: ADAPTERS.get(name, {}).get("language", "") for name in versions}
    colors = library_colors(versions)
    genoxide = versions.get("genoxide", "").split("+")[0]
    context = " · ".join(part for part in (
        f"genoxide {genoxide}" if genoxide else "",
        results.get("platform", ""),
        "single-threaded",
        f"{results['seeds']} seeds",
        results_date(results),
    ) if part)
    order = {scenario_name(*scenario[:3]): index for index, scenario in enumerate(SCENARIOS)}
    budgets = BUDGETS
    # the summaries from the runs, so the charts follow this code
    runs = results["runs"]
    caps = scenario_caps(results.get("max_seconds", 60.0), runs)

    # charts.json: the run, every library as the legends show it, and each chart's panels and bars
    data = {
        "format": 1,
        "run": {
            "timestamp": results.get("timestamp", ""),
            "date": results_date(results),
            "platform": results.get("platform", ""),
            "threads": "single-threaded",
            "seeds": results["seeds"],
            "context": context,
        },
        "libraries": [
            {"id": name, "name": LIBRARY_NAMES.get(name, name), "version": versions[name].split("+")[0],
             "language": languages.get(name, ""), "color": colors[name]}
            for name in dict.fromkeys(name for name in list(ADAPTERS) + sorted(versions) if name in versions)
        ],
        "charts": {},
    }
    # each panel's record, by its axes
    records = {}

    # sizes in inches
    width, margin, gap = 12.0, 0.1, 0.5
    panel_height, labels_height, title_height = 1.75, 1.05, 0.42
    bar_capacity = 44  # bars per row of panels

    def save(figure, name):
        for extension in formats:
            figure.savefig(out_dir / f"{name}.{extension}", metadata={"Date": None} if extension == "svg" else None,
                           dpi=200)
        plt.close(figure)

    def chart(name, title, subtitle, panels, libraries, quantity):
        """A figure with a header, a legend of every library (version and language), and axes for
        the panels, `[(key, bars)]`, packed into rows with the same width for every bar. Its record
        in charts.json is `name`'s, its values `quantity`."""
        rows, row = [], []
        for key, count in panels:
            if row and sum(c + 1.5 for _, c in row) + count + 1.5 > bar_capacity:
                rows.append(row)
                row = []
            row.append((key, count))
        if row:
            rows.append(row)
        names = list(dict.fromkeys(library for library in list(ADAPTERS) + sorted(versions) if library in libraries))
        record = data["charts"][name] = {"file": f"{name}.svg", "title": title, "subtitle": subtitle,
                                         "quantity": quantity, "libraries": names, "panels": []}
        columns = 5
        legend_rows = (len(names) + columns - 1) // columns
        # a long subtitle on several lines, the legend below it
        lines = [""]
        for part in subtitle.split(" · "):
            if lines[-1] and len(lines[-1]) + len(part) + 3 > 230:
                lines.append("")
            lines[-1] += (" · " if lines[-1] else "") + part
        subtitle = "\n".join(textwrap.fill(line, 230) for line in lines)
        legend_top = 0.62 + subtitle.count("\n") * 0.14
        header = legend_top + legend_rows * 0.19 + 0.12
        height = header + len(rows) * (title_height + panel_height + labels_height) + 0.05
        figure = plt.figure(figsize=(width, height))
        figure.patch.set_facecolor("white")
        figure.text(margin / width, 1 - 0.12 / height, title, fontsize=12.5, fontweight="bold", va="top")
        figure.text(margin / width, 1 - 0.40 / height, subtitle, fontsize=7.8, color="#555555", va="top",
                    linespacing=1.3)
        handles = [Patch(color=colors[name], label=f"{LIBRARY_NAMES.get(name, name)} "
                                                   f"{versions.get(name, '').split('+')[0]} ({languages.get(name, '')})")
                   for name in names]
        figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(margin / width, 1 - legend_top / height),
                      ncol=columns, frameon=False, fontsize=7.5, handlelength=0.9, handleheight=0.9,
                      columnspacing=1.6, borderaxespad=0.0, labelspacing=0.35)
        axes = {}
        y = height - header
        for row in rows:
            y -= title_height + panel_height
            usable = width - 2 * margin - 0.65 - gap * (len(row) - 1)
            units = sum(count + 1.5 for _, count in row)
            unit = min(usable / units, 0.36)
            x = margin + 0.65
            for key, count in row:
                panel_width = (count + 1.5) * unit
                axes[key] = figure.add_axes((x / width, y / height, panel_width / width, panel_height / height))
                records[axes[key]] = {"key": key, "title": None, "detail": None}
                record["panels"].append(records[axes[key]])
                x += panel_width + gap
            y -= labels_height
        return figure, axes

    def bars(axis, group, value, text, note=None, missing_text=None):
        """Vertical bars of one scenario on a log axis, lowest (best) first; missing values (e.g. no
        ERT) last, as a cross, labelled with `missing_text`. The rows of runs the time cap stopped
        ("ended_on_cap") come after the others, past a dotted line, cross-hatched."""
        def ordered(rows):
            present = sorted((row for row in rows if value(row) is not None), key=value)
            return present + [row for row in rows if value(row) is None]

        apart = [row for row in group if row.get("ended_on_cap")]
        group = ordered([row for row in group if not row.get("ended_on_cap")]) + ordered(apart)
        present = [(position, row, value(row)) for position, row in enumerate(group) if value(row) is not None]
        missing = [(position, row) for position, row in enumerate(group) if value(row) is None]
        values = [v for _, _, v in present]
        positions = list(range(len(group)))
        axis.set_xlim(-0.65, len(group) - 0.35)
        if values:
            axis.set_yscale("log")
            # room above the tallest bar for its label
            axis.set_ylim(min(values) / 2.5, max(values) * 22)
            bars_drawn = axis.bar([position for position, _, _ in present], values, width=0.74,
                                  color=[colors[row["library"]] for _, row, _ in present], linewidth=0)
            for patch, (_, row, _) in zip(bars_drawn, present):
                if row.get("ended_on_cap"):
                    patch.set_hatch("xxxx")
                    patch.set_alpha(0.55)
            for position, row, v in present:
                label_text = text(v) + (note(row) if note else "")
                axis.text(position, v * 1.15, label_text, rotation=90, ha="center", va="bottom", fontsize=6.2,
                          color="#222222")
        if missing:
            bottom, ceiling = axis.get_ylim()
            logarithmic = axis.get_yscale() == "log"
            marker_y = bottom * 1.6 if logarithmic else bottom + (ceiling - bottom) * 0.04
            axis.scatter([position for position, _ in missing], [marker_y] * len(missing), marker="x", s=14,
                         linewidths=1.1, color=[colors[row["library"]] for _, row in missing], zorder=3, clip_on=False)
            if missing_text:
                label_y = marker_y * 1.6 if logarithmic else marker_y + (ceiling - bottom) * 0.03
                for position, row in missing:
                    axis.text(position, label_y, missing_text(row), rotation=90, ha="center", va="bottom",
                              fontsize=6.2, color="#222222")
        if apart and len(apart) < len(group):
            axis.axvline(len(group) - len(apart) - 0.5, color="#9a9a9a", linewidth=0.7, linestyle=(0, (1, 2)),
                         zorder=0)
        # the panel in charts.json: its bars in this order, each with its value and labels as drawn
        record = records[axis]
        record.update({"log": True, "better": "lower"})
        record["bars"] = []
        for row in group:
            v = value(row)
            bar = {"library": row["library"], "solver": row["solver"], "label": label(row["library"], row["solver"]),
                   "value": significant(v)}
            if v is None:
                bar["missing"] = missing_text(row) if missing_text else ""
            else:
                bar["text"] = text(v)
                if note and note(row).strip():
                    bar["note"] = note(row).strip()
            # a fixed budget has no target, so nothing to reach
            bar.update({key: row[key] for key in ("runs", "reached") if key in row
                        and not (key == "reached" and row["scenario"] in FIXED_BUDGET)})
            if row.get("capped"):
                bar["capped"] = row["capped"]
            if row.get("capped_share") is not None:
                bar["capped_share"] = significant(row["capped_share"])
            if row.get("ended_on_cap"):
                bar["ended_on_cap"] = True
            if row.get("ended_early"):
                bar["ended_early"] = row["ended_early"]
            record["bars"].append(bar)
        axis.set_xticks(positions, [label(row["library"], row["solver"]) for row in group], rotation=60,
                        ha="right", rotation_mode="anchor", fontsize=6.5)
        for tick_label, row in zip(axis.get_xticklabels(), group):
            if row["library"] in ("genoxide", "genoxide_python"):
                tick_label.set_fontweight("bold")
        axis.tick_params(axis="x", length=0, pad=1.5)
        axis.tick_params(axis="y", labelsize=6.3, length=2, pad=1.5)
        axis.spines[["top", "right"]].set_visible(False)

    def awaiting(axis, scenario):
        """A panel of a scenario the results file has no runs of: awaiting the next run."""
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(0.5, 0.5, PENDING, transform=axis.transAxes, ha="center", va="center", fontsize=7.2,
                  color="#666666", wrap=True)
        records[axis].update({"log": True, "better": "lower", "bars": [], "pending": True, "note": PENDING})

    def panel_title(axis, scenario, detail):
        # the detail (e.g. the budget) on a second, lighter line, so narrow panels' titles fit
        axis.set_title(scenario_title(scenario), fontsize=8.2, loc="left", fontweight="bold", pad=11)
        axis.text(0, 1.015, detail, transform=axis.transAxes, fontsize=6.6, color="#666666", va="bottom")
        records[axis].update({"title": scenario_title(scenario), "detail": detail, "budget": budgets.get(scenario),
                              "cap": caps.get(scenario),
                              **({"fixed_budget": True} if scenario in FIXED_BUDGET else {})})

    def limits(scenario):
        """A panel's budget and time cap."""
        parts = (["no target"] if scenario in FIXED_BUDGET else []) + (
            [f"{'fixed budget' if scenario in FIXED_BUDGET else 'budget'} {short_number(budgets[scenario])} evaluations"]
            if scenario in budgets else [])
        parts += [f"cap {format_number(caps[scenario])} s"] if scenario in caps else []
        return " · ".join(parts)

    def capped_note(row):
        """How many runs the time cap stopped, and the median share of the budget they used."""
        if not row.get("capped"):
            return []
        share = row.get("capped_share")
        return [f"{row['capped']} capped" + (f" at {format_share(share)}" if share is not None else "")]

    time_ticks = FuncFormatter(lambda value, _position: format_seconds(value).replace(".0 ", " ").replace(".00 ", " "))
    count_ticks = FuncFormatter(lambda value, _position: short_number(value))
    capped_legend = ("c capped at p%: c runs stopped by the scenario's time cap before the budget, after a median p% "
                     "of it, so limited by speed, not by the search")

    # --- time and evaluations to target ----------------------------------------------------------
    rows = summarize(runs, caps)
    # every scenario of the suite, those without runs awaiting the next run
    scenarios = sorted({row["scenario"] for row in rows} | set(order), key=lambda s: (order.get(s, len(order)), s))
    caps = {scenario: caps.get(scenario, CAPS.get(scenario)) for scenario in scenarios}

    def bar_count(group):
        """A panel's width in bars: its bars, or room for the note of a panel awaiting the next run."""
        return len(group) or PENDING_BARS

    def reached(row):
        """How many runs reached the target, when not all did, and how many the time cap stopped."""
        parts = [] if row["reached"] >= row["runs"] else [f"{row['reached']} of {row['runs']}"]
        parts += capped_note(row)
        return "  " + ", ".join(parts) if parts else ""

    def too_few(row):
        """The label of a solver with no ERT: how many runs reached the target."""
        return ", ".join([f"{row['reached']}/{row['runs']} reached"] + capped_note(row))

    ert = ("expected running time (ERT): what all runs spent, up to the first hit in those that reached the target, "
           f"divided by the runs that reached it; at least {ERT_REACHED} must have")
    def budget_time(row):
        """A fixed budget's measure of time: the median time of the runs that used the whole budget."""
        return row["budget_time"]

    def ended_note(row):
        """How many runs the library ended before the budget (rules 2.2, 8.4), and how many the
        time cap stopped: left out of the time for the budget, not of the error at the end."""
        parts = [f"{row['ended_early']} of {row['runs']} ended early"] if row.get("ended_early") else []
        parts += capped_note(row)
        return "  " + ", ".join(parts) if parts else ""

    def no_budget_time(row):
        """The label of a method none of whose runs used the whole budget."""
        return f"none of {row['runs']} used the budget" + ("," + ended_note(row)[1:] if ended_note(row) else "")

    for name, title, value, text, ticks, quantity, fixed in (
        ("time_to_target", "Expected time to target, or median time for a fixed budget (lower is better)",
         lambda row: row["ert_time"], format_seconds, time_ticks, "seconds", True),
        ("evaluations_to_target",
         "Expected fitness evaluations to target (lower is better): search efficiency, whatever the language",
         lambda row: row["ert_evaluations"], short_number, count_ticks, "evaluations", False),
    ):
        # a scenario without a target has no evaluations to target: every run uses its budget
        shown = [scenario for scenario in scenarios if fixed or scenario not in FIXED_BUDGET]
        figure, axes = chart(
            name, title, context + f" · {MATCHED} · {ert} · k of n: reached in k of n runs · {capped_legend} · ×: "
            f"fewer than {ERT_REACHED} runs reached the target, k/n: how many did · "
            + ("a problem without a target: the median time of the runs that used its whole fixed budget, those the library "
                "ended early or the time cap stopped noted beside the bar · " if fixed else
               "a problem without a target isn't shown: every run uses its budget · ")
            + "a missing library can't run the scenario's method: see notes.md",
            [(scenario, bar_count([row for row in rows if row["scenario"] == scenario])) for scenario in shown],
            {row["library"] for row in rows if row["scenario"] in shown}, quantity)
        for scenario in shown:
            axis = axes[scenario]
            group = [row for row in rows if row["scenario"] == scenario]
            if not group:
                awaiting(axis, scenario)
            elif scenario in FIXED_BUDGET:
                bars(axis, group, budget_time, text, note=ended_note, missing_text=no_budget_time)
            else:
                bars(axis, group, value, text, note=reached, missing_text=too_few)
            panel_title(axis, scenario, limits(scenario))
            if axis.get_yscale() == "log":
                axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
                axis.yaxis.set_major_formatter(ticks)
        save(figure, name)

    # --- how close every run got: the distance to the optimum at the end (rule 8.1) -----------------
    # the runs the time cap stopped apart from the others
    gap_rows = summarize(runs, caps, split=True)
    figure, axes = chart(
        "distance_to_optimum",
        "Distance to the optimum at the end of the run, median of the runs (lower is better)",
        context + f" · {MATCHED} · a run ends at the target, its budget or its time cap · for a problem without a "
        "target, the error the fixed budget ends at: the same algorithm, so the libraries should agree within the "
        "seeds' spread · dashed: the target, 0.01 · "
        f"cross-hatched, past the dotted line: the runs stopped by the time cap · {capped_legend}",
        [(scenario, bar_count([row for row in gap_rows if row["scenario"] == scenario])) for scenario in scenarios],
        {row["library"] for row in gap_rows}, "distance")
    for scenario in scenarios:
        axis = axes[scenario]
        # a log axis can't show 0: a distance of 0 is drawn at a tenth of the smallest other one
        group = [row for row in gap_rows if row["scenario"] == scenario]
        if not group:
            awaiting(axis, scenario)
            panel_title(axis, scenario, limits(scenario))
            continue
        positive = [row["median_gap"] for row in group if row["median_gap"] > 0]
        floor = min(positive) / 10 if positive else 1e-3
        bars(axis, group, lambda row, floor=floor: max(row["median_gap"], floor),
             lambda v, floor=floor: "0" if v <= floor else short_number(v) if v >= 1000 else f"{v:.3g}",
             note=lambda row: (ended_note(row) if scenario in FIXED_BUDGET
                               else "  " + ", ".join(capped_note(row)) if row.get("ended_on_cap") else ""))
        panel_title(axis, scenario, limits(scenario))
        if scenario.split("-")[0] == "rosenbrock":
            axis.axhline(0.01, color="#444444", linewidth=0.7, linestyle=(0, (4, 3)), zorder=0)
            records[axis]["target"] = 0.01
        if axis.get_yscale() == "log":
            axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    save(figure, "distance_to_optimum")

    # --- genoxide's versions (rule 10), from their history file ---------------------------------------
    if history is not None and history.exists():
        record = draw_versions_chart(load_versions(history), out_dir, formats)
        if record:
            data["charts"]["genoxide_versions"] = record

    write_charts_json(data, out_dir)
    write_run_details(results, out_dir)


def pyplot():
    """matplotlib's pyplot, in the style of every chart of the harness."""
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Inter if installed (~/.local/share/fonts), DejaVu Sans otherwise; text as paths, so a chart
    # looks the same everywhere, whatever fonts the viewer has
    for path in Path.home().glob(".local/share/fonts/**/Inter-*.ttf"):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans"],
        "font.size": 8,
        "svg.fonttype": "path",
        "svg.hashsalt": "genoxide-benchmarks",
        "axes.edgecolor": "#9a9a9a",
        "axes.linewidth": 0.6,
        "xtick.color": "#333333",
        "ytick.color": "#555555",
        "ytick.major.width": 0.5,
        "ytick.minor.width": 0.3,
        "ytick.minor.size": 1.5,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
    })
    return plt


def write_charts_json(data, out_dir):
    """charts.json: the run and the libraries on the first line, then one chart per line, compact."""
    data = dict(data)
    lines = [f"{json.dumps(name)}:{json.dumps(record, ensure_ascii=False, separators=(',', ':'))}"
             for name, record in data.pop("charts").items()]
    head = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (out_dir / "charts.json").write_text(f'{head[:-1]},"charts":{{\n' + ",\n".join(lines) + "\n}}\n",
                                         encoding="utf-8", newline="\n")


# The run details beside charts.json, for the project site's benchmark page: clicking a bar shows
# that method's runs in its scenario, what they printed, and the adapter's code that set it up.
# One file per scenario, runs/<scenario>.json, so the page reads only the one it shows.
RUN_DETAILS = "runs"
# what a run's record has that its adapter didn't print: run_adapter adds it
ADDED_FIELDS = ("invalid",)
SCENARIO_FILE = re.compile(r"[a-z0-9]+-\d+-[a-z]+\.json")


def git_state(paths):
    """The commit of the repository, and which of `paths` (relative to it) differ from it: their
    lines at the commit aren't the ones read now. (None, set()) without git."""
    repository = ROOT.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                                cwd=repository).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", *paths], capture_output=True, text=True,
                                check=True, cwd=repository).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, set()
    return commit or None, {line[3:].strip().strip('"') for line in status.splitlines() if len(line) > 3}


def run_details(results):
    """The details of each scenario of a results file: the scenario and its settings, and per
    library and method its summary (as the charts' bars), every run (seed, time, evaluations, how
    it ended, and the JSON line its adapter printed) and the blocks of
    its adapter's code that set the method up (method_code.METHOD_CODE), with the commit whose file
    has those lines if the file is unchanged from it. Invalid runs are there too, with why."""
    import method_code

    runs = results["runs"]
    caps = scenario_caps(results.get("max_seconds", 60.0), runs)
    versions = results.get("versions", {})
    languages = results.get("languages") or {name: ADAPTERS.get(name, {}).get("language", "") for name in versions}
    summaries = {(row["scenario"], row["library"], row["solver"]): row
                 for row in summarize(runs, caps)}
    parts = {}
    for row in summarize(runs, caps, split=True):
        parts.setdefault((row["scenario"], row["library"], row["solver"]), []).append(row)
    order = {scenario_name(*scenario[:3]): index for index, scenario in enumerate(SCENARIOS)}
    libraries = {name: index for index, name in enumerate(ADAPTERS)}

    def numbers(row):
        return {key: significant(value) if isinstance(value, float) else value for key, value in row.items()
                if key not in ("scenario", "library", "solver") and value is not None}

    groups = {}
    for run in runs:
        key = (scenario_name(run["problem"], run["size"], run["mode"]), run["library"], run["solver"])
        groups.setdefault(key, []).append(run)
    code = {}
    for scenario, library, solver in groups:
        problem, _, mode = scenario.split("-")
        code[(scenario, library, solver)] = method_code.method_code(library, solver, problem, mode)
    commit, changed = git_state(sorted({block["path"] for blocks in code.values() for block in blocks}))

    details = {}
    for (scenario, library, solver), group in sorted(
            groups.items(), key=lambda item: (order.get(item[0][0], len(order)), item[0][0],
                                              libraries.get(item[0][1], len(libraries)), item[0][1], item[0][2])):
        cap = caps[scenario]
        problem, size, mode = scenario.split("-")
        if scenario not in details:
            targets = [] if scenario in FIXED_BUDGET else [run["target"] for run in group
                                                            if run.get("target") is not None]
            details[scenario] = {
                "format": 1,
                "run": {"timestamp": results.get("timestamp", ""), "date": results_date(results),
                        "platform": results.get("platform", "")},
                "scenario": {
                    "key": scenario, "title": scenario_title(scenario), "problem": problem,
                    "problem_name": PROBLEM_NAMES.get(problem, problem), "size": int(size), "mode": mode,
                    **({"method": METHODS[scenario]} if scenario in METHODS else {}),
                    **({"fixed_budget": True} if scenario in FIXED_BUDGET else {}),
                    "budget": BUDGETS.get(scenario), "cap": cap,
                    **({"target": targets[0]} if targets else {}),
                    "seeds": sorted({run["seed"] for key, runs_ in groups.items() if key[0] == scenario
                                     for run in runs_ if isinstance(run.get("seed"), int)}),
                },
                "methods": [],
            }
        entry = method_code.METHOD_CODE.get(library)
        details[scenario]["methods"].append({
            "library": library, "solver": solver, "label": label(library, solver),
            "name": LIBRARY_NAMES.get(library, library), "method": SOLVER_NAMES.get(solver, solver),
            "version": versions.get(library, ""), "language": languages.get(library, ""),
            "page": f"docs/benchmarks/libraries/{library}.md"
            if (DOCS / "libraries" / f"{library}.md").is_file() else None,
            "adapter": f"benchmarks/adapters/{entry['file']}" if entry else None,
            # as the charts summarize it: every valid run, and apart, with "ended_on_cap", the runs
            # the time cap stopped and the others, when there are both
            "summary": numbers(summaries[(scenario, library, solver)])
            if (scenario, library, solver) in summaries else None,
            "parts": [numbers(row) for row in parts.get((scenario, library, solver), [])]
            if len(parts.get((scenario, library, solver), [])) > 1 else [],
            "runs": [{
                "seed": run.get("seed"),
                "time_s": run.get("time_s"),
                "evaluations": run.get("evaluations"),
                **({"best": run["best"]} if "best" in run else {}),
                "reached": bool(first_hit(run, cap)),
                "first_hit": run.get("first_hit"),
                **({"capped": True} if capped(run, cap) else {}),
                # a run the library ended itself (rules 2.2, 8.4): why
                **({"ended_by": run["ended_by"]} if run.get("ended_by") else {}),
                **({"invalid": run["invalid"]} if run.get("invalid") else {}),
                "output": json.dumps({key: value for key, value in run.items()
                                      if key not in ADDED_FIELDS}, ensure_ascii=False),
            } for run in sorted(group, key=lambda run: (run.get("seed") is None, run.get("seed")))],
            "code": [dict(block, commit=commit if commit and block["path"] not in changed else None)
                     for block in code[(scenario, library, solver)]],
        })
    return details


def write_run_details(results, out_dir):
    """runs/<scenario>.json in `out_dir`, the details of each scenario (run_details): the file's
    first line the scenario, then one method per line, compact. Scenario files of an older run
    that this one doesn't have are removed."""
    folder = out_dir / RUN_DETAILS
    folder.mkdir(parents=True, exist_ok=True)
    details = run_details(results)
    for path in folder.glob("*.json"):
        if SCENARIO_FILE.fullmatch(path.name) and path.stem not in details:
            path.unlink()
    for scenario, data in details.items():
        data = dict(data)
        methods = data.pop("methods")
        head = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        lines = [json.dumps(method, ensure_ascii=False, separators=(",", ":")) for method in methods]
        (folder / f"{scenario}.json").write_text(f'{head[:-1]},"methods":[\n' + ",\n".join(lines) + "\n]}\n",
                                                 encoding="utf-8", newline="\n")


# the methods of genoxide's versions chart, a color each, in the harness's palette
SOLVER_COLORS = {"ga": "#4c78a8", "local_search": "#f58518", "cma_es": "#54a24b", "de": "#b279a2", "es": "#9d755d"}


def draw_versions_chart(history, out_dir, formats=("svg",)):
    """genoxide_versions: a panel per scenario, genoxide's versions on the x axis, a line per method
    through the instructions of its run in each version (rule 10), hollow where it didn't reach the
    target. Returns its record for charts.json, or None without a version."""
    rows = sorted(history.get("versions", []), key=lambda row: version_key(row["version"]))
    if not rows:
        return None
    plt = pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

    versions = [row["version"] for row in rows]
    order = {scenario_name(*scenario[:3]): index for index, scenario in enumerate(SCENARIOS)}
    scenarios = sorted({name for row in rows for name in row["scenarios"]}, key=lambda s: (order.get(s, len(order)), s))

    def methods(scenario):
        return list(dict.fromkeys(solver for row in rows for solver in row["scenarios"].get(scenario, {}).get("methods", {})))

    solvers = list(dict.fromkeys(solver for scenario in scenarios for solver in methods(scenario)))
    others = iter(color for color in PALETTE if color not in SOLVER_COLORS.values())
    colors = {solver: SOLVER_COLORS.get(solver) or next(others, "#888888") for solver in solvers}

    def unique(key):
        return "; ".join(dict.fromkeys(str(row.get(key)) for row in rows if row.get(key)))

    seed = history.get("seed", VERSIONS_SEED)
    title = "genoxide's versions: CPU instructions of the same runs (lower is better)"
    parts = [
        f"each method of the benchmark, one run per scenario with seed {seed}, to its target or its evaluation budget, "
        "without a time cap",
        "counted by Callgrind, the adapter's startup subtracted: exact, whatever the machine's load",
        "filled: the run reached the target, or used the fixed budget of a problem without one; hollow: it didn't reach "
        "the target within the budget",
        unique("rustc"), unique("valgrind"), unique("machine"),
        f"measured {unique('measured')}",
    ]
    subtitle = " · ".join(part for part in parts if part)
    record = {"file": "genoxide_versions.svg", "title": title, "subtitle": subtitle, "quantity": "instructions",
              "libraries": [], "seed": seed,
              "versions": [{"version": row["version"], "source": row.get("source"), "released": row.get("released"),
                            "measured": row.get("measured")} for row in rows],
              "panels": []}

    width, margin, columns = 12.0, 0.1, 3
    left_room, gap, panel_height, title_height, ticks_height = 0.55, 0.32, 1.45, 0.42, 0.42
    panel_width = (width - 2 * margin - columns * left_room - (columns - 1) * gap) / columns
    lines = [""]
    for part in subtitle.split(" · "):
        if lines[-1] and len(lines[-1]) + len(part) + 3 > 200:
            lines.append("")
        lines[-1] += (" · " if lines[-1] else "") + part
    subtitle_text = "\n".join(textwrap.fill(line, 200) for line in lines)
    legend_top = 0.62 + subtitle_text.count("\n") * 0.14
    legend_rows = (len(solvers) + 1 + 5) // 6
    header = legend_top + legend_rows * 0.19 + 0.1
    panel_rows = [scenarios[i:i + columns] for i in range(0, len(scenarios), columns)]
    height = header + len(panel_rows) * (title_height + panel_height + ticks_height) + 0.05
    figure = plt.figure(figsize=(width, height))
    figure.patch.set_facecolor("white")
    figure.text(margin / width, 1 - 0.12 / height, title, fontsize=12.5, fontweight="bold", va="top")
    figure.text(margin / width, 1 - 0.40 / height, subtitle_text, fontsize=7.8, color="#555555", va="top",
                linespacing=1.3)
    handles = [Line2D([], [], color=colors[solver], marker="o", markersize=4, linewidth=1.2,
                      label=SOLVER_NAMES.get(solver, solver)) for solver in solvers]
    if any(entry.get("reached") is False for row in rows for scenario in row["scenarios"].values()
           for entry in scenario.get("methods", {}).values()):
        handles.append(Line2D([], [], color="#666666", marker="o", markersize=4, markerfacecolor="white", linewidth=0,
                              label="target not reached"))
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(margin / width, 1 - legend_top / height),
                  ncol=6, frameon=False, fontsize=7.5, handlelength=1.6, columnspacing=1.6, borderaxespad=0.0,
                  labelspacing=0.35)
    count_ticks = FuncFormatter(lambda value, _position: short_number(value))
    y = height - header
    for panel_row in panel_rows:
        y -= title_height + panel_height
        for column, scenario in enumerate(panel_row):
            x = margin + left_room + column * (panel_width + left_room + gap)
            axis = figure.add_axes((x / width, y / height, panel_width / width, panel_height / height))
            values, bars = [], []
            for solver in methods(scenario):
                points = []
                for index, row in enumerate(rows):
                    entry = row["scenarios"].get(scenario, {}).get("methods", {}).get(solver)
                    if entry is None:
                        continue
                    # filled unless the run missed its target (a fixed budget's "reached" is null)
                    points.append((index, entry["instructions"], entry.get("reached") is not False))
                    bar = {"library": "genoxide", "solver": solver, "method": SOLVER_NAMES.get(solver, solver),
                           "version": row["version"], "label": f"{SOLVER_NAMES.get(solver, solver)} {row['version']}",
                           "color": colors[solver], "value": entry["instructions"],
                           "text": short_number(entry["instructions"]), "evaluations": entry["evaluations"], "runs": 1,
                           **({} if entry.get("reached") is None else {"reached": int(entry["reached"])})}
                    if entry.get("invalid"):
                        bar["invalid"] = True
                    bars.append(bar)
                values += [value for _, value, _ in points]
                axis.plot([p[0] for p in points], [p[1] for p in points], color=colors[solver], linewidth=1.2,
                          zorder=2)
                axis.scatter([p[0] for p in points], [p[1] for p in points], s=16, zorder=3, linewidths=1.0,
                             edgecolors=colors[solver],
                             facecolors=[colors[solver] if reached else "white" for _, _, reached in points])
            log = bool(values) and min(values) > 0 and max(values) / min(values) >= 10
            if log:
                axis.set_yscale("log")
                low, high = min(values), max(values)
                axis.set_ylim(low / 2, high * 2)
                axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            else:
                # from 0: a change of a fraction of a percent stays as small as it is
                axis.set_ylim(0, max(values, default=1) * 1.25)
                axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_formatter(count_ticks)
            axis.set_xlim(-0.5, len(versions) - 0.5)
            axis.set_xticks(range(len(versions)), versions, fontsize=6.5,
                            rotation=45 if len(versions) > 4 else 0, ha="right" if len(versions) > 4 else "center",
                            rotation_mode="anchor")
            axis.tick_params(axis="x", length=2, pad=2)
            axis.tick_params(axis="y", labelsize=6.3, length=2, pad=1.5)
            axis.spines[["top", "right"]].set_visible(False)
            budget = BUDGETS.get(scenario)
            detail = (f"{'no target · fixed budget' if scenario in FIXED_BUDGET else 'budget'} "
                      f"{short_number(budget)} evaluations" if budget else "")
            axis.set_title(scenario_title(scenario), fontsize=8.2, loc="left", fontweight="bold", pad=11)
            axis.text(0, 1.015, detail, transform=axis.transAxes, fontsize=6.6, color="#666666", va="bottom")
            record["panels"].append({"key": scenario, "title": scenario_title(scenario), "detail": detail, "log": log,
                                     "better": "lower", "budget": budget, "bars": bars})
        y -= ticks_height
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        figure.savefig(out_dir / f"genoxide_versions.{extension}",
                       metadata={"Date": None} if extension == "svg" else None, dpi=200)
    plt.close(figure)
    return record


def update_versions_chart(history, out_dir, formats=("svg",)):
    """Redraws the genoxide_versions chart of the history file into `out_dir`, and puts its record
    into the charts.json there, the other charts as they are."""
    record = draw_versions_chart(load_versions(history), out_dir, formats)
    if record is None:
        return
    charts = out_dir / "charts.json"
    if not charts.exists():
        print(f"{out_dir / 'genoxide_versions.svg'} drawn; no charts.json there to add its numbers to "
              "(`run.py chart --charts` writes one)", flush=True)
        return
    data = json.loads(charts.read_text(encoding="utf-8"))
    data["charts"]["genoxide_versions"] = record
    write_charts_json(data, out_dir)
    print(f"{out_dir / 'genoxide_versions.svg'} drawn, and its numbers written to {charts}", flush=True)


# The cores that times are measured on, under WSL: the two favoured P-cores (the highest turbo
# frequency) of the Intel Core Ultra 7 265K the published benchmarks run on. pin-wsl.ps1 pins the
# WSL virtual machine to them; Windows would otherwise move a run between P-cores of different
# frequencies and E-cores. Other machines pass their own with --cores.
PINNED_CORES = (8, 19)


def is_wsl():
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def busy_windows_cores(loops, seconds=4):
    """The Windows logical processors that are busy while `loops` busy loops run in WSL, from
    Windows' own counters (no administrator rights needed)."""
    spin = f"import time\nend = time.time() + {seconds + 3}\nwhile time.time() < end: pass"
    processes = [subprocess.Popen([sys.executable, "-c", spin]) for _ in range(loops)]
    script = (
        f"$samples = Get-Counter '\\Processor(*)\\% Processor Time' -SampleInterval 1 -MaxSamples {seconds}; "
        "$samples.CounterSamples | Where-Object InstanceName -ne '_total' | Group-Object InstanceName | "
        "ForEach-Object { $_.Name + ' ' + [int]($_.Group | Measure-Object CookedValue -Average).Average }"
    )
    try:
        # a second for the loops to start
        time.sleep(1)
        output = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                capture_output=True, text=True, timeout=120).stdout
    finally:
        for process in processes:
            process.wait()
    load = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            load[int(parts[0])] = int(parts[1])
    return sorted(core for core, percent in load.items() if percent > 50)


def check_pinning(cores):
    """Under WSL, that the virtual machine only runs on `cores`: with more busy loops than cores,
    those cores and no others are busy."""
    busy = busy_windows_cores(len(cores) + 2)
    if not busy:
        raise SystemExit("can't read Windows' processor counters from WSL (powershell.exe), to check that WSL "
                         "is pinned; pass --allow-unpinned for a run whose times don't count")
    if not set(busy) <= set(cores):
        raise SystemExit(
            f"WSL isn't pinned to cores {', '.join(map(str, cores))}: busy loops ran on processors "
            f"{', '.join(map(str, busy))}. Run benchmarks/pin-wsl.ps1 in an Administrator PowerShell, or install "
            "its scheduled task once with `pin-wsl.ps1 -Install`; or pass --allow-unpinned for a run whose times "
            "don't count."
        )
    print(f"WSL is pinned to cores {', '.join(map(str, cores))}", flush=True)


def describe_platform(cores=None):
    """The operating system and processor, for the charts, and the cores times were measured on."""
    import platform
    processor = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            processor = next(line.split(":", 1)[1].strip() for line in cpuinfo if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    described = f"{platform.system()}, {processor}".rstrip(", ")
    if cores:
        described += f", WSL pinned to cores {', '.join(map(str, cores))}"
    return described


def draw_charts_of(results_file, out_dir, png=False, history=VERSIONS_FILE):
    """Draws the charts of a results file, and of the history of genoxide's versions, with the
    .venv's matplotlib if this Python has none."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        subprocess.run(
            [str(VENV_PYTHON), str(ROOT / "run.py"), "chart", "--results", str(results_file), "--charts", str(out_dir),
             "--history", str(history)] + (["--png"] if png else []),
            check=True,
        )
        return
    draw_charts(read_results(results_file), out_dir, formats=("svg", "png") if png else ("svg",), history=history)


def draw_versions_of(history, out_dir, png=False):
    """Draws the chart of genoxide's versions into `out_dir` and its charts.json, with the .venv's
    matplotlib if this Python has none."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        subprocess.run([str(VENV_PYTHON), str(ROOT / "run.py"), "versions", "--history", str(history), "--charts",
                        str(out_dir)] + (["--png"] if png else []), check=True)
        return
    update_versions_chart(history, out_dir, formats=("svg", "png") if png else ("svg",))


def markdown_report(report):
    """results/latest.md, which becomes docs/benchmarks/results.md: the coverage, and the table of
    the results."""
    header = [f"# Results {report['timestamp']}", "",
              "The matched suite: three problems, one method each, the same in every library, with its own "
              "implementation ([rule 6](rules.md#6-the-methods)).", "",
              f"Seeds per scenario: {report['seeds']}, wall time cap per run: "
              f"{describe_caps(scenario_caps(report['max_seconds'], report['runs']))}",
              report["platform"], ""]
    header += [f"- {name} {version}" for name, version in report["versions"].items()] + [""]
    # every timed run is validated (check.check_run): the ones that failed are listed, not kept
    header += [invalid_list(report["runs"]), ""]
    header += charts_section(report)
    ran = {scenario_name(run["problem"], run["size"], run["mode"]) for run in report["runs"]}
    pending = [scenario_title(scenario_name(*scenario[:3])) for scenario in SCENARIOS
               if scenario_name(*scenario[:3]) not in ran]
    if pending:
        header += [f"{PENDING}: {', '.join(pending)}. This run has no runs of "
                   + ("it" if len(pending) == 1 else "them") + ".", ""]
    header += ["## Coverage", "",
               "✓ ran, ✗ every run invalid, – can't run the scenario's method: why, and the bugs found in the "
               "libraries, in [notes.md](notes.md).", "", coverage_table(report["runs"], report["versions"]), "",
               "## Results", "",
               "Expected time and evaluations to target: the expected running time (ERT), what all runs spent, up "
               "to the first hit of the target in the runs that reached it, divided by the number of runs that "
               f"reached it; with fewer than {ERT_REACHED}, how many reached it. A first hit after the time cap "
               "counts as not reached. Stopped by the time cap: runs that ended at the cap, not at the target or "
               "the budget, and the median share of the budget they used. A problem without a target (Rastrigin 30, "
               "[rule 6.3](rules.md#6-the-methods)) has a table of its own: the median time of the runs that used "
               "the whole fixed budget, the error at the end of every run, and the runs the library ended early.",
               ""]
    # a blank line between the list and the table, or the table becomes part of the list
    return "\n".join(header) + "\n" + markdown_table(report["summary"]) + "\n"


def charts_section(report):
    """The lines of results.md that link the charts, which `run.py publish` puts beside it."""
    return ["## Charts", "",
            "Interactive, with each bar's numbers and runs: "
            "[tachsin.gr/projects/genoxide/benchmarks](https://tachsin.gr/projects/genoxide/benchmarks).", "",
            "![Expected time to target: a panel per problem, a bar per library](time_to_target.svg)", "",
            "- [Expected evaluations to target](evaluations_to_target.svg)",
            "- [Distance to the optimum at the end](distance_to_optimum.svg)",
            "- [genoxide's versions](genoxide_versions.svg): the CPU instructions of the same runs in each release, "
            "genoxide only ([rule 10](rules.md#10-instruction-counts-genoxides-versions))", ""]


def describe_caps(caps):
    """The time caps of a results file's scenarios, e.g. "60 s"."""
    return ", ".join(f"{format_number(value)} s" for value in sorted(set(caps.values())))


def read_results(path):
    """A results file: a run's JSON in results/, or compressed with xz or gzip, like the published
    one (PUBLISHED_RESULTS)."""
    path = Path(path)
    if path.suffix == ".xz":
        import lzma
        text = lzma.decompress(path.read_bytes()).decode("utf-8")
    elif path.suffix == ".gz":
        import gzip
        text = gzip.decompress(path.read_bytes()).decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    results = json.loads(text)
    # the files from before the timestamp was stored are named after it
    results.setdefault("timestamp", path.name.split(".")[0])
    return results


def write_published_results(results, path=PUBLISHED_RESULTS):
    """The published copy of a results file, compressed with xz."""
    import lzma
    text = json.dumps(results, indent=1, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lzma.compress(text.encode("utf-8"), preset=9 | lzma.PRESET_EXTREME))


def local_runs():
    """The runs' results files in results/, oldest first."""
    return sorted(path for path in (ROOT / "results").glob("*.json") if RUN_FILE.fullmatch(path.name))


def latest_results():
    """The newest results: a run in results/ newer than the published one, or the published one."""
    local = local_runs()
    if PUBLISHED_RESULTS.is_file():
        published = read_results(PUBLISHED_RESULTS)["timestamp"]
        if not local or local[-1].name.split(".")[0] <= published:
            return PUBLISHED_RESULTS
    if not local:
        raise SystemExit("no results yet: run `python run.py` first")
    return local[-1]


def publish(results_file, history=VERSIONS_FILE):
    """Publishes a run into DOCS: its tables (results.md), its charts, charts.json and the run
    details (runs/), and the run itself, compressed (PUBLISHED_RESULTS). The tables are summarized
    from the runs again, as the charts are, so they agree with them whichever version of run.py
    wrote the file."""
    results = read_results(results_file)
    caps = scenario_caps(results.get("max_seconds", 60.0), results["runs"])
    report = dict(results, summary=summarize(results["runs"], caps))
    (DOCS / "results.md").write_text(markdown_report(report), encoding="utf-8", newline="\n")
    write_published_results(results)
    draw_charts_of(results_file, DOCS, history=history)
    print(f"{Path(results_file).name} published in {DOCS}: results.md, the charts, charts.json, "
          f"{RUN_DETAILS}/ and {PUBLISHED_RESULTS.name}", flush=True)


# --update keeps the other libraries' times, so the machine must still measure what it measured
# then: a fixed reference, DEAP's GA in OneMax 1000 with seeds 0 to 2, must take the same median
# time as in the results file, within DRIFT
DRIFT_REFERENCE = ("deap", "ga", ("onemax", 1000, "matched"), 3)
DRIFT = 0.03


def check_drift(previous, caps, allow):
    """Measures the reference and refuses to go on, unless `allow`, if it differs from the results
    file's."""
    library, solver, scenario, seeds = DRIFT_REFERENCE
    name = scenario_name(*scenario)

    def reference(runs):
        return {run["seed"]: run for run in valid(runs)
                if (run.get("library"), run.get("solver")) == (library, solver)
                and scenario_name(run.get("problem"), run.get("size"), run.get("mode")) == name
                and run.get("seed") in range(seeds)}

    def refuse(message):
        if not allow:
            raise SystemExit(f"--update: {message}. Pass --allow-drift to rerun anyway.")
        print(f"warning: {message}; rerunning anyway (--allow-drift)", flush=True)

    before = reference(previous["runs"])
    if len(before) != seeds:
        refuse(f"the results file has no runs of the reference, {library} {solver} in {name} with seeds 0 to "
               f"{seeds - 1}, to check that this machine still measures the same")
        return
    print(f"drift reference: {name}: {library} ({seeds} seeds) ...", flush=True)
    now = reference(run_adapter(ADAPTERS[library], *scenario, seeds, BUDGETS[name], caps[name]))
    if len(now) != seeds:
        refuse(f"the reference, {library} {solver} in {name}, didn't give {seeds} valid runs")
        return
    changed = [seed for seed in range(seeds) if now[seed]["evaluations"] != before[seed]["evaluations"]]
    if changed:
        refuse(f"the reference, {library} {solver} in {name}, doesn't make the same evaluations as in the results "
               f"file (seed {changed[0]}: {now[changed[0]]['evaluations']} instead of "
               f"{before[changed[0]]['evaluations']}): its adapter or the problem changed, so the times can't be "
               "compared. Rerun every library")
        return
    old = median([run["time_s"] for run in before.values()])
    new = median([run["time_s"] for run in now.values()])
    drift = new / old - 1
    print(f"drift reference: median {format_seconds(new)}, {format_seconds(old)} in the results file ({drift:+.1%})",
          flush=True)
    if abs(drift) > DRIFT:
        refuse(f"the reference, {library} {solver} in {name}, takes {abs(drift):.1%} {'more' if drift > 0 else 'less'} "
               f"time than in the results file (median {format_seconds(new)} against {format_seconds(old)}), over "
               f"{DRIFT:.0%}: this machine "
               "doesn't measure the same, so the rerun libraries' times wouldn't be comparable with the kept ones. "
               "Check the load, the pinning and the versions of WSL and Python, or rerun every library")


def build(libraries, labels):
    """Builds the adapters of `libraries` and returns their versions."""
    versions = {}
    for name in libraries:
        adapter = ADAPTERS[name]
        if adapter.get("build"):
            print(f"building {name} adapter ...", flush=True)
            subprocess.run(adapter["build"], check=True)
        versions[name] = library_version(*adapter["version"], label=labels.get(name))
        print(f"{name} {versions[name]}", flush=True)
    return versions


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?",
                        choices=["run", "setup", "check", "chart", "publish", "versions", "outdated"], default="run")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-seconds", type=float,
                        help="wall time cap per run, in every scenario (default: each scenario's own)")
    parser.add_argument("--quick", action="store_true", help="3 seeds")
    parser.add_argument("--scenarios", nargs="*", help="scenario names, e.g. rastrigin-30-matched (default all)")
    parser.add_argument("--libraries", nargs="*", choices=list(ADAPTERS), help="default all")
    parser.add_argument("--results", type=Path,
                        help="results file to chart (default: the latest, the published "
                             f"{PUBLISHED_RESULTS.relative_to(ROOT.parent).as_posix()} or a newer run in results/), "
                             "or to publish (default: the latest run in results/)")
    parser.add_argument("--charts", type=Path,
                        help="folder for the charts (default: results/charts; for versions, docs/benchmarks, where "
                             "the history file is)")
    parser.add_argument("--genoxide", nargs="+", metavar="VERSION",
                        help="with versions, the versions of genoxide to measure: releases on crates.io, e.g. "
                             "0.8.0, or path, this repository's genoxide")
    parser.add_argument("--history", type=Path, default=VERSIONS_FILE,
                        help="the history file of genoxide's versions (default: "
                             f"{VERSIONS_FILE.relative_to(ROOT.parent).as_posix()})")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(),
                        help="Callgrind runs, or check scenarios, at a time (default: the number of cores)")
    parser.add_argument("--png", action="store_true", help="also draw the charts as PNG, e.g. to preview them")
    parser.add_argument("--update", type=Path, nargs="?", const=PUBLISHED_RESULTS, metavar="RESULTS",
                        help="rerun only --libraries, with the seeds and every scenario of this results file "
                             f"(default: the published {PUBLISHED_RESULTS.relative_to(ROOT.parent).as_posix()}), "
                             "and keep its results of the other libraries")
    parser.add_argument("--allow-drift", action="store_true",
                        help="with --update, rerun even if the reference run's time differs from the results "
                             f"file's by more than {DRIFT * 100:.0f}%%")
    parser.add_argument("--version-label", nargs="*", default=[], metavar="LIBRARY=VERSION",
                        help="the version to record for a library instead of the one it reports, e.g. "
                             "genoxide=0.7.0 before the release PR bumps Cargo.toml; genoxide's labels "
                             "genoxide_python too, and genoxide's commit is still added")
    parser.add_argument("--cores", default=",".join(map(str, PINNED_CORES)),
                        help="under WSL, the Windows cores the virtual machine must be pinned to (pin-wsl.ps1) "
                             f"before times are measured; default {','.join(map(str, PINNED_CORES))}")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="measure under WSL without checking the pinning, for runs whose times don't count")
    parser.add_argument("--issue", action="store_true",
                        help="with outdated, keep one open GitHub issue listing the newer releases (gh)")
    parser.add_argument("--dry-run", action="store_true",
                        help="with outdated --issue, print what it would do to the issue instead")
    args = parser.parse_args()

    if args.command == "versions":
        # genoxide's versions (rule 10): no timed run, so no check and no pinning
        charts = args.charts or args.history.parent
        if not args.genoxide:
            # the chart of the history file, as it is
            draw_versions_of(args.history, charts, png=args.png)
            return
        if not shutil.which("valgrind"):
            raise SystemExit("counting the instructions needs Valgrind (Linux)")
        measure_versions(args.genoxide, args.history, args.jobs, charts, png=args.png)
        return
    if args.genoxide:
        raise SystemExit("--genoxide is for `run.py versions`")
    if args.charts is None:
        args.charts = ROOT / "results" / "charts"
    if args.libraries is None:
        args.libraries = list(ADAPTERS)

    labels = {}
    for item in args.version_label:
        name, _, version = item.partition("=")
        if name not in args.libraries or not version:
            raise SystemExit(f"--version-label {item}: expected LIBRARY=VERSION, with a library that runs, "
                             f"one of {' '.join(args.libraries)}")
        labels[name] = version
    if "genoxide" in labels:
        # the Python package has genoxide's version
        labels.setdefault("genoxide_python", labels["genoxide"])

    if args.command == "setup":
        setup()
        return
    if args.command == "outdated":
        import outdated
        outdated.outdated(args.libraries, args.issue, args.dry_run)
        return
    if args.command == "chart":
        results_file = args.results or latest_results()
        draw_charts_of(results_file, args.charts, png=args.png, history=args.history)
        print(f"charts of {results_file.name} in {args.charts}")
        return
    if args.command == "publish":
        runs = local_runs()
        if not args.results and not runs:
            raise SystemExit("no run in results/ to publish: run `python run.py` first, or pass --results")
        publish(args.results or runs[-1], history=args.history)
        return
    if not VENV_PYTHON.exists():
        raise SystemExit("run `python run.py setup` first")
    import check
    if args.command == "check":
        scenarios = [s for s in SCENARIOS if not args.scenarios or scenario_name(*s[:3]) in args.scenarios]
        raise SystemExit(0 if check.check(args.libraries, scenarios, args.jobs) else 1)
    # the rules come first: only adapters that passed `run.py check` as they are now are measured
    unchecked = check.unchecked(args.libraries)
    if unchecked:
        raise SystemExit(f"{', '.join(unchecked)}: the adapter hasn't passed `python run.py check` since it last "
                         "changed. Check it first (docs/benchmarks/rules.md).")
    pinned = None
    if is_wsl() and not args.allow_unpinned:
        pinned = tuple(int(core) for core in args.cores.split(","))
        check_pinning(pinned)

    previous = None
    if args.update:
        if set(args.libraries) == set(ADAPTERS):
            raise SystemExit("--update needs --libraries: the ones to rerun")
        if args.scenarios or args.quick:
            # the rerun libraries' other scenarios would keep their old runs, labeled with the new
            # version, and a library new to the file would seem unable to run them
            raise SystemExit("--update reruns the libraries on every scenario of the results file, so it can't "
                             "be combined with --scenarios or --quick. To add a scenario, rerun every library.")
        previous = read_results(args.update)
    seeds = previous["seeds"] if previous else 3 if args.quick and args.seeds == 10 else args.seeds
    # each scenario's time cap: an update keeps the results file's
    caps = (scenario_caps(previous.get("max_seconds", 60.0), previous["runs"]) if previous
            else {name: args.max_seconds or cap for name, cap in CAPS.items()})
    previous_scenarios = {scenario_name(r["problem"], r["size"], r["mode"]) for r in previous["runs"]} if previous else None
    scenarios = [
        scenario for scenario in SCENARIOS
        if (not args.scenarios or scenario_name(*scenario[:3]) in args.scenarios)
        # an update reruns every scenario of its results file
        and (previous_scenarios is None or scenario_name(*scenario[:3]) in previous_scenarios)
    ]

    versions = build(args.libraries, labels)

    if previous:
        check_drift(previous, caps, args.allow_drift)

    runs = []
    for problem, size, mode, max_evaluations, _ in scenarios:
        for name in args.libraries:
            # an adapter prints nothing for the problems its library can't do
            print(f"{scenario_name(problem, size, mode)}: {name} ({seeds} seeds) ...", flush=True)
            runs += run_adapter(ADAPTERS[name], problem, size, mode, seeds, max_evaluations,
                                caps[scenario_name(problem, size, mode)])

    if previous:
        # the results of the other libraries and scenarios, as they were
        rerun = {scenario_name(*scenario[:3]) for scenario in scenarios}
        runs = [
            run for run in previous["runs"]
            if run["library"] not in args.libraries
            or scenario_name(run["problem"], run["size"], run["mode"]) not in rerun
        ] + runs
        versions = {**previous["versions"], **versions}

    # the time cap of each scenario of the results
    caps = {name: caps[name] for name in dict.fromkeys(scenario_name(r["problem"], r["size"], r["mode"]) for r in runs)}
    rows = summarize(runs, caps)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    platform = describe_platform(pinned)
    if previous and previous.get("platform") not in (None, platform):
        platform = f"{previous['platform']}; {', '.join(args.libraries)} rerun on {platform}"
    languages = {name: ADAPTERS[name]["language"] for name in versions if name in ADAPTERS}
    report = {"date": datetime.date.today().isoformat(), "timestamp": timestamp, "versions": versions,
              "languages": languages, "seeds": seeds, "max_seconds": caps, "platform": platform,
              "runs": runs, "summary": rows}
    results_file = results / f"{timestamp}.json"
    results_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = markdown_report(report)
    (results / "latest.md").write_text(markdown, encoding="utf-8")
    print()
    print(markdown)
    draw_charts_of(results_file, args.charts, history=args.history)
    print(f"{results_file.name}: publish it into docs/benchmarks with `python run.py publish`", flush=True)


if __name__ == "__main__":
    main()
