"""Benchmarks of evolutionary computation libraries on the same problems.

Usage:
    python run.py setup                      # create .venv and install the Python libraries
    python run.py                            # all scenarios, 10 seeds
    python run.py --quick                    # small scenarios, 3 seeds
    python run.py --seeds 5 --scenarios onemax-100-matched nqueens-32-idiomatic
    python run.py --libraries deap genetic_algorithm
    python run.py chart                      # redraw the charts of the latest results
    python run.py --libraries genoxide --update results/<file>.json
                                             # rerun one library, keep the others' results
    python run.py --libraries genoxide pymoo deap --scenarios zdt1-30-matched --update results/<file>.json
                                             # add a scenario to a results file

Results are written to results/<timestamp>.json (all runs), results/latest.md (table) and
results/charts/*.svg (charts). On Linux with Valgrind, a run also measures instructions per
evaluation with Callgrind.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
RUST_ADAPTER = ROOT / "adapters" / "genetic_algorithm"
GENOXIDE_ADAPTER = ROOT / "adapters" / "genoxide"
# the builds of the adapters that aren't Rust or Python, outside the repository
BUILDS = Path(os.environ.get("BENCH_BUILDS", Path.home() / "bench-targets"))

# Each adapter prints one JSON line per solver per seed, with the same command line:
#   <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
ADAPTERS = {
    "genoxide": {
        "build": ["cargo", "build", "--release", "--quiet", "--manifest-path", str(GENOXIDE_ADAPTER / "Cargo.toml")],
        "command": [str(GENOXIDE_ADAPTER / "target" / "release" / "ga_bench_genoxide")],
        # the genoxide of this repository: its version and commit
        "version": ("cargo", "genoxide", GENOXIDE_ADAPTER),
        "language": "Rust",
    },
    "genetic_algorithm": {
        "build": ["cargo", "build", "--release", "--quiet", "--manifest-path", str(RUST_ADAPTER / "Cargo.toml")],
        "command": [str(RUST_ADAPTER / "target" / "release" / "ga_bench_genetic_algorithm")],
        "version": ("cargo", "genetic_algorithm", RUST_ADAPTER),
        "language": "Rust",
    },
    "deap": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "deap" / "bench.py")],
        "version": ("python", "deap"),
        "language": "Python",
    },
    "pygad": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "pygad" / "bench.py")],
        "version": ("python", "pygad"),
        "language": "Python",
    },
    "pymoo": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "pymoo" / "bench.py")],
        "version": ("python", "pymoo"),
        "language": "Python",
    },
    "radiate": {
        "build": ["cargo", "build", "--release", "--quiet", "--manifest-path",
                  str(ROOT / "adapters" / "radiate" / "Cargo.toml")],
        "command": [str(ROOT / "adapters" / "radiate" / "target" / "release" / "ga_bench_radiate")],
        "version": ("cargo", "radiate", ROOT / "adapters" / "radiate"),
        "language": "Rust",
    },
    "moors": {
        "build": ["cargo", "build", "--release", "--quiet", "--manifest-path",
                  str(ROOT / "adapters" / "moors" / "Cargo.toml")],
        "command": [str(ROOT / "adapters" / "moors" / "target" / "release" / "ga_bench_moors")],
        "version": ("cargo", "moors", ROOT / "adapters" / "moors"),
        "language": "Rust",
    },
    "pycma": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "pycma" / "bench.py")],
        "version": ("python", "cma"),
        "language": "Python",
    },
    "nevergrad": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "nevergrad" / "bench.py")],
        "version": ("python", "nevergrad"),
        "language": "Python",
    },
    "scipy": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "scipy" / "bench.py")],
        "version": ("python", "scipy"),
        "language": "Python",
    },
    "pygmo": {
        # pagmo's C++ algorithms, calling the Python fitness function
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "pygmo" / "bench.py")],
        "version": ("python", "pygmo"),
        "language": "C++ via Python",
    },
    "jenetics": {
        # the JDK and the pinned jars are downloaded into ~/opt, and compiled into ~/bench-targets
        "build": ["bash", str(ROOT / "adapters" / "jenetics" / "build.sh")],
        "command": ["bash", str(ROOT / "adapters" / "jenetics" / "run.sh")],
        "version": ("command", ["bash", str(ROOT / "adapters" / "jenetics" / "run.sh"), "--version"]),
        "language": "Java",
        # Callgrind can't follow a JIT-compiled runtime in reasonable time
        "instructions": False,
    },
    "jmetal": {
        # the JDK and the pinned jars are downloaded into ~/opt, and compiled into ~/bench-targets
        "build": ["bash", str(ROOT / "adapters" / "jmetal" / "build.sh")],
        "command": ["bash", str(ROOT / "adapters" / "jmetal" / "run.sh")],
        "version": ("command", ["bash", str(ROOT / "adapters" / "jmetal" / "run.sh"), "--version"]),
        "language": "Java",
        # Callgrind can't follow a JIT-compiled runtime in reasonable time
        "instructions": False,
    },
    "evolutionary_jl": {
        # installs the pinned packages and precompiles, then runs with one thread
        "build": ["bash", str(ROOT / "adapters" / "evolutionary_jl" / "run.sh"), "--build"],
        "command": ["bash", str(ROOT / "adapters" / "evolutionary_jl" / "run.sh")],
        "version": ("command", ["bash", str(ROOT / "adapters" / "evolutionary_jl" / "run.sh"), "--version"]),
        "language": "Julia",
        # Callgrind can't follow a JIT-compiled runtime in reasonable time
        "instructions": False,
    },
    "metaheuristics_jl": {
        "build": ["bash", str(ROOT / "adapters" / "metaheuristics_jl" / "run.sh"), "--build"],
        "command": ["bash", str(ROOT / "adapters" / "metaheuristics_jl" / "run.sh")],
        "version": ("command", ["bash", str(ROOT / "adapters" / "metaheuristics_jl" / "run.sh"), "--version"]),
        "language": "Julia",
        "instructions": False,
    },
    "openga": {
        # the header at a pinned commit, compiled with g++ -O3
        "build": ["bash", str(ROOT / "adapters" / "openga" / "build.sh")],
        "command": [str(BUILDS / "openga" / "ga_bench_openga")],
        "version": ("command", [str(BUILDS / "openga" / "ga_bench_openga"), "--version"]),
        "language": "C++",
    },
}

# (problem, size, mode, max_evaluations). A run stops at the target, at max_evaluations or at
# max_seconds, whichever comes first.
#   matched:   configurations as equal as the libraries allow (framework cost)
#   idiomatic: each library's own recommended configuration (what a user gets)
SCENARIOS = [
    ("onemax", 100, "matched", 200_000),
    ("onemax", 1000, "matched", 2_000_000),
    ("onemax", 100, "idiomatic", 200_000),
    ("nqueens", 32, "idiomatic", 500_000),
    ("nqueens", 64, "idiomatic", 1_000_000),
    ("rastrigin", 10, "idiomatic", 500_000),
    ("rastrigin", 30, "idiomatic", 2_000_000),
    ("rosenbrock", 10, "idiomatic", 500_000),
    ("ackley", 30, "idiomatic", 1_000_000),
    # multi-objective: a budget and no target; the quality is the hypervolume of the final front
    ("zdt1", 30, "matched", 25_000),
    ("zdt2", 30, "matched", 25_000),
    ("zdt3", 30, "matched", 25_000),
    ("dtlz2", 3, "matched", 25_000),
    ("dtlz1", 3, "matched", 40_000),
]
QUICK_SCENARIOS = {"onemax-100-matched", "onemax-100-idiomatic", "nqueens-32-idiomatic", "rastrigin-10-idiomatic",
                   "zdt1-30-matched"}

# the multi-objective problems and the hypervolume's reference point
FRONT_PROBLEMS = {"zdt1": (1.1, 1.1), "zdt2": (1.1, 1.1), "zdt3": (1.1, 1.1), "dtlz2": (1.1, 1.1, 1.1),
                  "dtlz1": (1.1, 1.1, 1.1)}


def is_front(problem):
    return problem in FRONT_PROBLEMS


def hypervolume(points, reference):
    """The exact hypervolume of points to minimize: a sweep in 2 dimensions, slices above."""
    points = [p for p in points if all(x < r for x, r in zip(p, reference))]
    if not points:
        return 0.0
    if len(reference) == 1:
        return reference[0] - min(p[0] for p in points)
    if len(reference) == 2:
        volume, ceiling = 0.0, reference[1]
        for x, y in sorted(points):
            if y < ceiling:
                volume += (reference[0] - x) * (ceiling - y)
                ceiling = y
        return volume
    points = sorted(points, key=lambda p: p[-1])
    volume = 0.0
    for index, point in enumerate(points):
        top = points[index + 1][-1] if index + 1 < len(points) else reference[-1]
        if top > point[-1]:
            volume += (top - point[-1]) * hypervolume([p[:-1] for p in points[:index + 1]], reference[:-1])
    return volume


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


def library_version(kind, package, adapter=None):
    if kind == "command":
        # a command that prints the version, e.g. an adapter's --version
        return subprocess.run(package, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    if kind == "python":
        return subprocess.run(
            [str(VENV_PYTHON), "-c", f"import importlib.metadata as m; print(m.version('{package}'))"],
            capture_output=True, text=True, check=True, cwd=ROOT,
        ).stdout.strip()
    metadata = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--manifest-path", str(adapter / "Cargo.toml")],
        capture_output=True, text=True, check=True,
    ).stdout)
    version = next(p["version"] for p in metadata["packages"] if p["name"] == package)
    if package == "genoxide":
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT,
        ).stdout.strip()
        version = f"{version}+{commit}" if commit else version
    return version


def run_adapter(adapter, problem, size, mode, seeds, max_evaluations, max_seconds):
    command = adapter["command"] + [
        problem, str(size), mode, "0", str(seeds - 1), str(max_evaluations), str(max_seconds),
    ]
    # run from this folder, so a library repository checked out next to it can't shadow a package
    completed = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(f"adapter failed: {' '.join(command)}")
    runs = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    for run in runs:
        if is_front(run["problem"]):
            # the same hypervolume for every library; the front itself isn't kept
            run["hypervolume"] = hypervolume(run.pop("front"), FRONT_PROBLEMS[run["problem"]])
    return runs


# Instructions per evaluation, with Callgrind: each adapter runs with a budget of N and of 2N
# evaluations, and (I(2N) - I(N)) / (E(2N) - E(N)) cancels the startup, imports and setup. The
# matched OneMax configurations make it a comparison of framework cost; no library reaches the
# target of OneMax 1000 within 2N evaluations.
INSTRUCTIONS_SCENARIO = ("onemax", 1000, "matched")
INSTRUCTIONS_EVALUATIONS = 3_000


def count_instructions(adapter, evaluations):
    """(instructions, {solver: evaluations}) of one run under Callgrind."""
    problem, size, mode = INSTRUCTIONS_SCENARIO
    command = ["valgrind", "--tool=callgrind", "--callgrind-out-file=/dev/null"] + adapter["command"] + [
        problem, str(size), mode, "0", "0", str(evaluations), "36000",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    match = re.search(r"Collected\s*:\s*(\d+)", completed.stderr)
    if completed.returncode != 0 or not match:
        print(completed.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"callgrind failed: {' '.join(command)}")
    runs = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    return int(match.group(1)), {run["solver"]: run["evaluations"] for run in runs}


def measure_instructions(libraries):
    """Instructions per evaluation of each library in the instructions scenario."""
    rows = []
    for name in libraries:
        if not ADAPTERS[name].get("instructions", True):
            continue
        print(f"instructions: {name} (callgrind) ...", flush=True)
        low, low_evaluations = count_instructions(ADAPTERS[name], INSTRUCTIONS_EVALUATIONS)
        high, high_evaluations = count_instructions(ADAPTERS[name], 2 * INSTRUCTIONS_EVALUATIONS)
        for solver, evaluations in high_evaluations.items():
            rows.append({
                "library": name,
                "solver": solver,
                "instructions_per_evaluation": (high - low) / (evaluations - low_evaluations[solver]),
            })
    return rows


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
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.4g}"
    return f"{int(value):,}"


def summarize(runs):
    groups = {}
    for run in runs:
        if is_front(run["problem"]):
            continue
        key = (scenario_name(run["problem"], run["size"], run["mode"]), run["library"], run["solver"])
        groups.setdefault(key, []).append(run)

    def evaluations_per_second(group):
        return sum(run["evaluations"] for run in group) / max(sum(run["time_s"] for run in group), 1e-9)

    # reference for the throughput ratio: DEAP's GA in the same scenario
    reference = {
        scenario: evaluations_per_second(group)
        for (scenario, library, solver), group in groups.items()
        if library == "deap" and solver == "ga"
    }

    rows = []
    for (scenario, library, solver), group in groups.items():
        successes = [run for run in group if run["success"]]
        rate = evaluations_per_second(group)
        rows.append({
            "scenario": scenario,
            "library": library,
            "solver": solver,
            "runs": len(group),
            "success_rate": len(successes) / len(group),
            "time_to_target": median([run["time_s"] for run in successes]),
            "median_evaluations": median([run["evaluations"] for run in group]),
            "evaluations_to_target": median([run["evaluations"] for run in successes]),
            "median_best": median([run["best"] for run in group]),
            "evaluations_per_second": rate,
            "throughput_vs_deap": rate / reference[scenario] if scenario in reference and library != "deap" else None,
        })
    rows.sort(key=lambda row: (row["scenario"], row["library"], row["solver"]))
    return rows


def summarize_fronts(runs):
    """Median hypervolume and time of each multi-objective solver."""
    groups = {}
    for run in runs:
        if is_front(run["problem"]):
            key = (scenario_name(run["problem"], run["size"], run["mode"]), run["library"], run["solver"])
            groups.setdefault(key, []).append(run)
    rows = []
    for (scenario, library, solver), group in groups.items():
        volumes = sorted(run["hypervolume"] for run in group)
        rows.append({
            "scenario": scenario,
            "library": library,
            "solver": solver,
            "runs": len(group),
            "median_hypervolume": median(volumes),
            "worst_hypervolume": volumes[0],
            "best_hypervolume": volumes[-1],
            "median_time": median([run["time_s"] for run in group]),
            "median_evaluations": median([run["evaluations"] for run in group]),
            # runs where the library failed (an adapter's "error"): an empty front, hypervolume 0
            "errors": sum(1 for run in group if run.get("error")),
            "evaluations_per_second": sum(run["evaluations"] for run in group)
            / max(sum(run["time_s"] for run in group), 1e-9),
        })
    rows.sort(key=lambda row: (row["scenario"], row["library"], row["solver"]))
    return rows


def front_table(rows):
    lines = [
        "| Scenario | Library / solver | Median hypervolume | Range | Median time | Median evaluations | Evaluations/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['library']} / {row['solver']} "
            f"| {row['median_hypervolume']:.4f} "
            f"| {row['worst_hypervolume']:.4f} to {row['best_hypervolume']:.4f} "
            f"| {format_seconds(row['median_time'])} "
            f"| {format_number(round(row['median_evaluations']))} "
            f"| {format_number(round(row['evaluations_per_second']))} |"
        )
    return "\n".join(lines)


def markdown_table(rows):
    lines = [
        "| Scenario | Library / solver | Success | Median time to target | Median evaluations | Median best | Evaluations/s | Throughput vs DEAP GA |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        ratio = row["throughput_vs_deap"]
        ratio = "-" if ratio is None else (f"{ratio:.1f}×" if ratio < 10 else f"{ratio:.0f}×")
        lines.append(
            f"| {row['scenario']} | {row['library']} / {row['solver']} "
            f"| {row['success_rate'] * 100:.0f}% ({row['runs']}) "
            f"| {format_seconds(row['time_to_target'])} "
            f"| {format_number(row['median_evaluations'])} "
            f"| {format_number(row['median_best'])} "
            f"| {format_number(round(row['evaluations_per_second']))} "
            f"| {ratio} |"
        )
    return "\n".join(lines)


LIBRARY_NAMES = {"genoxide": "genoxide", "genetic_algorithm": "genetic_algorithm", "deap": "DEAP",
                 "pygad": "PyGAD", "pymoo": "pymoo", "radiate": "radiate", "moors": "moors", "pycma": "pycma",
                 "nevergrad": "Nevergrad", "scipy": "SciPy", "pygmo": "pygmo", "openga": "openGA",
                 "jenetics": "Jenetics", "jmetal": "jMetal", "evolutionary_jl": "Evolutionary.jl",
                 "metaheuristics_jl": "Metaheuristics.jl"}
SOLVER_NAMES = {"ga": "GA", "evolve": "GA", "hill_climb": "hill climbing", "local_search": "local search",
                "cma_es": "CMA-ES", "de": "DE", "pso": "PSO", "es": "ES", "nsga2": "NSGA-II", "nsga3": "NSGA-III",
                "spea2": "SPEA2", "sms_emoa": "SMS-EMOA", "moead": "MOEA/D", "sade": "SaDE", "ngopt": "NGOpt",
                "two_points_de": "TwoPointsDE", "ipop_cma_es": "IPOP-CMA-ES"}
PROBLEM_NAMES = {"onemax": "OneMax", "nqueens": "N-Queens", "rastrigin": "Rastrigin", "rosenbrock": "Rosenbrock",
                 "ackley": "Ackley", "zdt1": "ZDT1", "zdt2": "ZDT2", "zdt3": "ZDT3", "dtlz1": "DTLZ1", "dtlz2": "DTLZ2"}
GENOXIDE_COLOR = "#ce422b"
# the other libraries, in the order of ADAPTERS: a qualitative palette without red
PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d", "#72b7b2", "#e0b000", "#ff9da6",
           "#79706e", "#1b9e77", "#7570b3", "#a6761d", "#e7298a", "#66a61e", "#666666"]


def library_colors(libraries):
    """A color per library, the same in every chart."""
    colors, others = {}, iter(PALETTE)
    for name in list(ADAPTERS) + sorted(set(libraries) - set(ADAPTERS)):
        colors[name] = GENOXIDE_COLOR if name == "genoxide" else next(others, "#888888")
    return colors


def label(library, solver):
    return f"{LIBRARY_NAMES.get(library, library)} {SOLVER_NAMES.get(solver, solver)}"


def scenario_title(scenario):
    problem, size, mode = scenario.split("-")
    name = PROBLEM_NAMES.get(problem, problem)
    if problem.startswith("dtlz"):
        return f"{name}, {size} objectives"
    if is_front(problem):
        return f"{name}, {size} variables"
    return f"{name} {size} ({mode})"


def short_number(value):
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if value >= limit:
            return f"{value / limit:.3g}{suffix}"
    return f"{value:.3g}"


def results_date(results):
    """The date of a results file: stored, or from its timestamp."""
    if results.get("date"):
        return results["date"]
    stamp = results.get("timestamp", "")
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}" if len(stamp) >= 8 else ""


def draw_charts(results, out_dir, formats=("svg",)):
    """Vertical bar charts of a results file: time and evaluations to target, cost per evaluation,
    and the hypervolume and time of the multi-objective fronts."""
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

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
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = results.get("versions", {})
    languages = results.get("languages") or {name: ADAPTERS.get(name, {}).get("language", "") for name in versions}
    colors = library_colors(versions)
    genoxide = versions.get("genoxide", "").split("+")[0]
    context = " · ".join(part for part in (
        f"genoxide {genoxide}" if genoxide else "",
        results.get("platform", ""),
        "single-threaded",
        f"median of {results['seeds']} seeds",
        results_date(results),
    ) if part)
    order = {scenario_name(*scenario[:3]): index for index, scenario in enumerate(SCENARIOS)}
    budgets = {scenario_name(*scenario[:3]): scenario[3] for scenario in SCENARIOS}

    # sizes in inches
    width, margin, gap = 12.0, 0.1, 0.5
    panel_height, labels_height, title_height = 1.75, 1.05, 0.42
    bar_capacity = 44  # bars per row of panels

    def save(figure, name):
        for extension in formats:
            figure.savefig(out_dir / f"{name}.{extension}", metadata={"Date": None} if extension == "svg" else None,
                           dpi=200)
        plt.close(figure)

    def chart(title, subtitle, panels, libraries):
        """A figure with a header, a legend of every library (version and language), and axes for
        the panels, `[(key, bars)]`, packed into rows with the same width for every bar."""
        rows, row = [], []
        for key, count in panels:
            if row and sum(c + 1.5 for _, c in row) + count + 1.5 > bar_capacity:
                rows.append(row)
                row = []
            row.append((key, count))
        if row:
            rows.append(row)
        names = list(dict.fromkeys(name for name in list(ADAPTERS) + sorted(versions) if name in libraries))
        columns = 5
        legend_rows = (len(names) + columns - 1) // columns
        header = 0.62 + legend_rows * 0.19 + 0.12
        height = header + len(rows) * (title_height + panel_height + labels_height) + 0.05
        figure = plt.figure(figsize=(width, height))
        figure.patch.set_facecolor("white")
        figure.text(margin / width, 1 - 0.12 / height, title, fontsize=12.5, fontweight="bold", va="top")
        figure.text(margin / width, 1 - 0.40 / height, subtitle, fontsize=7.8, color="#555555", va="top")
        handles = [Patch(color=colors[name], label=f"{LIBRARY_NAMES.get(name, name)} "
                                                   f"{versions.get(name, '').split('+')[0]} ({languages.get(name, '')})")
                   for name in names]
        figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(margin / width, 1 - 0.62 / height),
                      ncol=columns, frameon=False, fontsize=7.5, handlelength=0.9, handleheight=0.9,
                      columnspacing=1.6, borderaxespad=0.0, labelspacing=0.35)
        axes = {}
        y = height - header
        for row in rows:
            y -= title_height + panel_height
            usable = width - 2 * margin - 0.45 - gap * (len(row) - 1)
            units = sum(count + 1.5 for _, count in row)
            unit = min(usable / units, 0.36)
            x = margin + 0.45
            for key, count in row:
                panel_width = (count + 1.5) * unit
                axes[key] = figure.add_axes((x / width, y / height, panel_width / width, panel_height / height))
                x += panel_width + gap
            y -= labels_height
        return figure, axes

    def bars(axis, group, value, text, log=True, better="lower", note=None, zoom=False):
        """Vertical bars of one scenario, best first; missing values (e.g. the target never reached)
        last, as a cross."""
        present = [row for row in group if value(row) is not None]
        missing = [row for row in group if value(row) is None]
        present.sort(key=lambda row: value(row), reverse=better == "higher")
        group = present + missing
        values = [value(row) for row in present]
        positions = list(range(len(group)))
        axis.set_xlim(-0.65, len(group) - 0.35)
        base = 0.0
        if values:
            low, high = min(values), max(values)
            if log:
                axis.set_yscale("log")
                base = low / 2.5
                # room above the tallest bar for its label
                axis.set_ylim(base, high * 22)
            elif zoom:
                spread = max(high - low, 1e-4)
                base = low - spread * 0.3
                axis.set_ylim(base, high + spread * 1.2)
            else:
                axis.set_ylim(0, high * 1.6)
            axis.bar(positions[:len(present)], [v - base for v in values] if zoom else values,
                     bottom=base if zoom else None, width=0.74,
                     color=[colors[row["library"]] for row in present], linewidth=0)
            for position, row, v in zip(positions, present, values):
                label_text = text(v) + (note(row) if note else "")
                top = v * 1.15 if log else v + (axis.get_ylim()[1] - axis.get_ylim()[0]) * 0.015
                axis.text(position, top, label_text, rotation=90, ha="center", va="bottom", fontsize=6.2,
                          color="#222222")
        if missing:
            bottom = axis.get_ylim()[0]
            marker_y = bottom * 1.6 if axis.get_yscale() == "log" else bottom + (axis.get_ylim()[1] - bottom) * 0.04
            axis.scatter(positions[len(present):], [marker_y] * len(missing), marker="x", s=14, linewidths=1.1,
                         color=[colors[row["library"]] for row in missing], zorder=3, clip_on=False)
        axis.set_xticks(positions, [label(row["library"], row["solver"]) for row in group], rotation=60,
                        ha="right", rotation_mode="anchor", fontsize=6.5)
        for tick_label, row in zip(axis.get_xticklabels(), group):
            if row["library"] == "genoxide":
                tick_label.set_fontweight("bold")
        axis.tick_params(axis="x", length=0, pad=1.5)
        axis.tick_params(axis="y", labelsize=6.3, length=2, pad=1.5)
        axis.spines[["top", "right"]].set_visible(False)

    def panel_title(axis, scenario, detail):
        # the detail (e.g. the budget) on a second, lighter line, so narrow panels' titles fit
        axis.set_title(scenario_title(scenario), fontsize=8.2, loc="left", fontweight="bold", pad=11)
        axis.text(0, 1.015, detail, transform=axis.transAxes, fontsize=6.6, color="#666666", va="bottom")

    time_ticks = FuncFormatter(lambda value, _position: format_seconds(value).replace(".0 ", " ").replace(".00 ", " "))
    count_ticks = FuncFormatter(lambda value, _position: short_number(value))

    # --- single objective: time and evaluations to target -------------------------------------
    rows = [row for row in results["summary"] if not is_front(row["scenario"].split("-")[0])]
    scenarios = sorted({row["scenario"] for row in rows}, key=lambda s: (order.get(s, len(order)), s))

    def reached(row):
        return "" if row["success_rate"] >= 1 else f"  {round(row['success_rate'] * row['runs'])}/{row['runs']}"

    for name, title, value, text, ticks in (
        ("time_to_target", "Time to target (lower is better)",
         lambda row: row["time_to_target"], format_seconds, time_ticks),
        ("evaluations_to_target",
         "Fitness evaluations to target (lower is better): search efficiency, whatever the language",
         lambda row: row.get("evaluations_to_target"), short_number, count_ticks),
    ):
        if not scenarios:
            break
        figure, axes = chart(
            title, context + " · × never reached the target · k/n: reached in k of n runs",
            [(scenario, len([row for row in rows if row["scenario"] == scenario])) for scenario in scenarios],
            {row["library"] for row in rows})
        for scenario in scenarios:
            axis = axes[scenario]
            bars(axis, [row for row in rows if row["scenario"] == scenario], value, text, note=reached)
            panel_title(axis, scenario, f"budget {short_number(budgets[scenario])} evaluations" if scenario in budgets else "")
            if axis.get_yscale() == "log":
                axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
                axis.yaxis.set_major_formatter(ticks)
        save(figure, name)

    # --- cost per evaluation ------------------------------------------------------------------------
    instructions = results.get("instructions")
    if instructions:
        problem, size, mode = INSTRUCTIONS_SCENARIO
        figure, axes = chart(
            "CPU instructions per evaluation, framework and fitness function together (lower is better)",
            f"{PROBLEM_NAMES[problem]} {size} ({mode}), counted by Callgrind: exact, whatever the machine's load; "
            "startup and imports excluded · " + context,
            [("instructions", len(instructions))], {row["library"] for row in instructions})
        axis = axes["instructions"]
        bars(axis, instructions, lambda row: row["instructions_per_evaluation"], short_number)
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        axis.yaxis.set_major_formatter(count_ticks)
        save(figure, "instructions")

    # --- multi-objective ------------------------------------------------------------------------------
    front_rows = results.get("front_summary") or []
    front_scenarios = sorted({row["scenario"] for row in front_rows}, key=lambda s: (order.get(s, len(order)), s))
    for name, title, value, text, log, better in (
        ("hypervolume", "Hypervolume of the final front (higher is better; the axes don't start at 0)",
         lambda row: row["median_hypervolume"], lambda v: f"{v:.4f}", False, "higher"),
        ("front_time", "Time for the evaluation budget of a multi-objective run (lower is better)",
         lambda row: row["median_time"], format_seconds, True, "lower"),
    ):
        if not front_rows:
            break
        figure, axes = chart(
            title, context + " · SBX and polynomial mutation, the same settings in every library",
            [(scenario, len([row for row in front_rows if row["scenario"] == scenario])) for scenario in front_scenarios],
            {row["library"] for row in front_rows})
        for scenario in front_scenarios:
            axis = axes[scenario]
            bars(axis, [row for row in front_rows if row["scenario"] == scenario], value, text,
                 log=log, better=better, zoom=not log,
                 note=lambda row: f"  {row['errors']}/{row['runs']} failed" if row.get("errors") else "")
            panel_title(axis, scenario, f"{short_number(budgets[scenario])} evaluations" if scenario in budgets else "")
            if log:
                axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
                axis.yaxis.set_major_formatter(time_ticks)
            else:
                axis.yaxis.set_major_locator(MaxNLocator(4))
        save(figure, name)


def describe_platform():
    """The operating system and processor, for the charts."""
    import platform
    processor = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            processor = next(line.split(":", 1)[1].strip() for line in cpuinfo if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return f"{platform.system()}, {processor}".rstrip(", ")


def draw_charts_of(results_file, out_dir, png=False):
    """Draws the charts of a results file, with the .venv's matplotlib if this Python has none."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        subprocess.run(
            [str(VENV_PYTHON), str(ROOT / "run.py"), "chart", "--results", str(results_file), "--charts", str(out_dir)]
            + (["--png"] if png else []),
            check=True,
        )
        return
    results = json.loads(Path(results_file).read_text(encoding="utf-8"))
    results.setdefault("timestamp", Path(results_file).stem)
    # the summaries from the runs, so the charts follow this code
    results["summary"] = summarize(results["runs"])
    results["front_summary"] = summarize_fronts(results["runs"])
    draw_charts(results, out_dir, formats=("svg", "png") if png else ("svg",))


def latest_results():
    files = sorted((ROOT / "results").glob("*.json"))
    if not files:
        raise SystemExit("no results yet: run `python run.py` first")
    return files[-1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["run", "setup", "chart"], default="run")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=60.0, help="wall time cap per run")
    parser.add_argument("--quick", action="store_true", help="small scenarios, 3 seeds")
    parser.add_argument("--scenarios", nargs="*", help="scenario names, e.g. onemax-100-matched (default all)")
    parser.add_argument("--libraries", nargs="*", default=list(ADAPTERS), choices=list(ADAPTERS))
    parser.add_argument("--results", type=Path, help="results file to chart (default: the latest)")
    parser.add_argument("--charts", type=Path, default=ROOT / "results" / "charts", help="folder for the charts")
    parser.add_argument("--no-instructions", action="store_true", help="skip the Callgrind measurement")
    parser.add_argument("--png", action="store_true", help="also draw the charts as PNG, e.g. to preview them")
    parser.add_argument("--update", type=Path,
                        help="rerun only --libraries, with the seeds and scenarios of this results file, "
                             "and keep its results of the other libraries")
    args = parser.parse_args()

    if args.command == "setup":
        setup()
        return
    if args.command == "chart":
        results_file = args.results or latest_results()
        draw_charts_of(results_file, args.charts, png=args.png)
        print(f"charts of {results_file.name} in {args.charts}")
        return
    if not VENV_PYTHON.exists():
        raise SystemExit("run `python run.py setup` first")

    previous = None
    if args.update:
        if set(args.libraries) == set(ADAPTERS):
            raise SystemExit("--update needs --libraries: the ones to rerun")
        previous = json.loads(args.update.read_text(encoding="utf-8"))
    seeds = previous["seeds"] if previous else 3 if args.quick and args.seeds == 10 else args.seeds
    max_seconds = previous.get("max_seconds", args.max_seconds) if previous else args.max_seconds
    previous_scenarios = {scenario_name(r["problem"], r["size"], r["mode"]) for r in previous["runs"]} if previous else None
    scenarios = [
        scenario for scenario in SCENARIOS
        if (not args.quick or scenario_name(*scenario[:3]) in QUICK_SCENARIOS)
        and (not args.scenarios or scenario_name(*scenario[:3]) in args.scenarios)
        # an update reruns the scenarios of its results file, or the ones named, which can be new
        and (previous_scenarios is None or args.scenarios or scenario_name(*scenario[:3]) in previous_scenarios)
    ]

    versions = {}
    for name in args.libraries:
        adapter = ADAPTERS[name]
        if adapter.get("build"):
            print(f"building {name} adapter ...", flush=True)
            subprocess.run(adapter["build"], check=True)
        versions[name] = library_version(*adapter["version"])
        print(f"{name} {versions[name]}", flush=True)

    runs = []
    for problem, size, mode, max_evaluations in scenarios:
        for name in args.libraries:
            # an adapter prints nothing for the problems its library can't do
            print(f"{scenario_name(problem, size, mode)}: {name} ({seeds} seeds) ...", flush=True)
            runs += run_adapter(ADAPTERS[name], problem, size, mode, seeds, max_evaluations, max_seconds)

    instructions = None
    if shutil.which("valgrind") and not args.no_instructions:
        instructions = measure_instructions(args.libraries)

    if previous:
        # the results of the other libraries and scenarios, as they were
        rerun = {scenario_name(*scenario[:3]) for scenario in scenarios}
        runs = [
            run for run in previous["runs"]
            if run["library"] not in args.libraries
            or scenario_name(run["problem"], run["size"], run["mode"]) not in rerun
        ] + runs
        versions = {**previous["versions"], **versions}
        if instructions is None:
            # not measured again (no Valgrind, or --no-instructions): keep the previous counts
            instructions = previous.get("instructions")
            if instructions:
                print("instruction counts: kept from the previous results", flush=True)
        else:
            kept = [row for row in previous.get("instructions") or [] if row["library"] not in args.libraries]
            instructions = kept + instructions

    rows = summarize(runs)
    front_rows = summarize_fronts(runs)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    platform = describe_platform()
    if previous and previous.get("platform") not in (None, platform):
        platform = f"{previous['platform']}; {', '.join(args.libraries)} rerun on {platform}"
    languages = {name: ADAPTERS[name]["language"] for name in versions if name in ADAPTERS}
    report = {"date": datetime.date.today().isoformat(), "timestamp": timestamp, "versions": versions,
              "languages": languages, "seeds": seeds, "max_seconds": max_seconds, "platform": platform,
              "runs": runs, "summary": rows, "front_summary": front_rows, "instructions": instructions}
    results_file = results / f"{timestamp}.json"
    results_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    header = [f"# Results {timestamp}", "", f"Seeds per scenario: {seeds}, wall time cap per run: {max_seconds} s", platform, ""]
    header += [f"- {name} {version}" for name, version in versions.items()] + [""]
    table = markdown_table(rows)
    if front_rows:
        table += "\n\n## Multi-objective\n\n" + front_table(front_rows)
    # a blank line between the list and the table, or the table becomes part of the list
    (results / "latest.md").write_text("\n".join(header) + "\n" + table + "\n", encoding="utf-8")
    print()
    print(table)
    draw_charts_of(results_file, args.charts)


if __name__ == "__main__":
    main()
