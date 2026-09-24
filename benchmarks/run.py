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

# Each adapter prints one JSON line per solver per seed, with the same command line:
#   <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
ADAPTERS = {
    "genoxide": {
        "build": ["cargo", "build", "--release", "--quiet", "--manifest-path", str(GENOXIDE_ADAPTER / "Cargo.toml")],
        "command": [str(GENOXIDE_ADAPTER / "target" / "release" / "ga_bench_genoxide")],
        # the genoxide of this repository: its version and commit
        "version": ("cargo", "genoxide", GENOXIDE_ADAPTER),
    },
    "genetic_algorithm": {
        "build": ["cargo", "build", "--release", "--quiet", "--manifest-path", str(RUST_ADAPTER / "Cargo.toml")],
        "command": [str(RUST_ADAPTER / "target" / "release" / "ga_bench_genetic_algorithm")],
        "version": ("cargo", "genetic_algorithm", RUST_ADAPTER),
    },
    "deap": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "deap" / "bench.py")],
        "version": ("python", "deap"),
    },
    "pygad": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "pygad" / "bench.py")],
        "version": ("python", "pygad"),
    },
    "pymoo": {
        "command": [str(VENV_PYTHON), str(ROOT / "adapters" / "pymoo" / "bench.py")],
        "version": ("python", "pymoo"),
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
]
QUICK_SCENARIOS = {"onemax-100-matched", "onemax-100-idiomatic", "nqueens-32-idiomatic", "rastrigin-10-idiomatic"}


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
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


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
            "median_best": median([run["best"] for run in group]),
            "evaluations_per_second": rate,
            "throughput_vs_deap": rate / reference[scenario] if scenario in reference and library != "deap" else None,
        })
    rows.sort(key=lambda row: (row["scenario"], row["library"], row["solver"]))
    return rows


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
                 "pygad": "PyGAD", "pymoo": "pymoo"}
SOLVER_NAMES = {"ga": "GA", "evolve": "GA", "hill_climb": "hill climbing", "local_search": "local search",
                "cma_es": "CMA-ES", "de": "DE"}
PROBLEM_NAMES = {"onemax": "OneMax", "nqueens": "N-Queens", "rastrigin": "Rastrigin"}
GENOXIDE_COLOR = "#ce422b"
OTHER_COLOR = "#8a9bb0"


def label(library, solver):
    return f"{LIBRARY_NAMES.get(library, library)} {SOLVER_NAMES.get(solver, solver)}"


def scenario_title(scenario):
    problem, size, mode = scenario.split("-")
    return f"{PROBLEM_NAMES.get(problem, problem)} {size} ({mode})"


def draw_charts(results, out_dir):
    """Bar charts of a results file, as SVG: time to target, throughput and instructions."""
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        # text as text, and the same file for the same results
        "svg.fonttype": "none",
        "svg.hashsalt": "genoxide-benchmarks",
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = results["summary"]
    # in the order of SCENARIOS
    order = {scenario_name(*scenario[:3]): index for index, scenario in enumerate(SCENARIOS)}
    scenarios = sorted({row["scenario"] for row in rows}, key=lambda s: (order.get(s, len(order)), s))
    footnote = f"{results['seeds']} seeds per scenario, single-threaded; {results.get('platform', '')}".rstrip("; ")

    from matplotlib.ticker import FuncFormatter

    def short_number(value, _position=None):
        for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
            if value >= limit:
                return f"{value / limit:.3g}{suffix}"
        return f"{value:.3g}"

    tick_formatters = {
        "time_to_target": FuncFormatter(lambda value, _position: format_seconds(value).replace(".0 ", " ")),
        "throughput": FuncFormatter(lambda value, _position: short_number(value) + "/s"),
    }

    def small_multiples(name, title, value, formatter, missing):
        columns = 2
        lines = (len(scenarios) + columns - 1) // columns
        bars = [len([row for row in rows if row["scenario"] == s]) for s in scenarios]
        # each line of charts is as high as its tallest chart
        tallest = [max(bars[start:start + columns]) for start in range(0, len(bars), columns)]
        figure, axes = plt.subplots(
            lines, columns, figsize=(11, 1.0 + sum(tallest) * 0.3 + lines * 0.8), squeeze=False,
        )
        figure.patch.set_facecolor("white")
        for axis in axes.flat[len(scenarios):]:
            axis.set_visible(False)
        for axis, scenario in zip(axes.flat, scenarios):
            group = [row for row in rows if row["scenario"] == scenario]
            # best first; missing values (e.g. never reached the target) last
            group.sort(key=lambda row: (value(row) is None, value(row) or 0) if name == "time_to_target"
                       else (value(row) is None, -(value(row) or 0)))
            labels = [label(row["library"], row["solver"]) for row in group]
            values = [value(row) for row in group]
            positions = range(len(group))[::-1]
            colors = [GENOXIDE_COLOR if row["library"] == "genoxide" else OTHER_COLOR for row in group]
            present = [v for v in values if v]
            axis.barh(list(positions), [v or 0 for v in values], color=colors, height=0.7)
            axis.set_yticks(list(positions), labels)
            axis.set_xscale("log")
            if present:
                # room on the right for the labels
                axis.set_xlim(min(present) / 3, max(present) * 60)
            for position, row, v in zip(positions, group, values):
                text = formatter(v) if v else missing(row)
                if v and row["success_rate"] < 1 and name == "time_to_target":
                    text += f"  ({row['success_rate'] * row['runs']:.0f}/{row['runs']} reached)"
                x = v if v else (min(present) / 3 if present else 1)
                axis.text(x * 1.1, position, text, va="center", fontsize=8.5,
                          color="#222" if v else "#777")
            axis.set_title(scenario_title(scenario), fontsize=10.5, loc="left")
            axis.tick_params(axis="x", labelsize=8)
            axis.xaxis.set_major_formatter(tick_formatters[name])
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
        figure.text(0.01, 0.005, footnote, fontsize=8, color="#555")
        figure.tight_layout(rect=(0, 0.02, 1, 0.97))
        figure.savefig(out_dir / f"{name}.svg", metadata={"Date": None})
        plt.close(figure)

    small_multiples(
        "time_to_target", "Median time to target (lower is better)",
        lambda row: row["time_to_target"], format_seconds,
        lambda row: "did not reach the target",
    )
    small_multiples(
        "throughput", "Evaluations per second (higher is better)",
        lambda row: row["evaluations_per_second"], lambda v: f"{short_number(v)}/s",
        lambda row: "-",
    )

    instructions = results.get("instructions")
    if instructions:
        instructions = sorted(instructions, key=lambda row: row["instructions_per_evaluation"])
        figure, axis = plt.subplots(figsize=(11, 0.9 + 0.4 * len(instructions)))
        figure.patch.set_facecolor("white")
        positions = range(len(instructions))[::-1]
        values = [row["instructions_per_evaluation"] for row in instructions]
        axis.barh(list(positions), values, height=0.7,
                  color=[GENOXIDE_COLOR if row["library"] == "genoxide" else OTHER_COLOR for row in instructions])
        axis.set_yticks(list(positions), [label(row["library"], row["solver"]) for row in instructions])
        axis.set_xscale("log")
        axis.set_xlim(min(values) / 3, max(values) * 12)
        for position, v in zip(positions, values):
            axis.text(v * 1.1, position, short_number(v), va="center", fontsize=8.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=8)
        axis.xaxis.set_major_formatter(FuncFormatter(short_number))
        problem, size, mode = INSTRUCTIONS_SCENARIO
        axis.set_title(
            f"CPU instructions per evaluation, {PROBLEM_NAMES[problem]} {size} ({mode}), "
            "counted by Callgrind (lower is better)",
            fontsize=12, fontweight="bold", loc="left",
        )
        figure.text(0.01, 0.01, "Framework and fitness function together; startup and imports excluded.",
                    fontsize=8, color="#555")
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        figure.savefig(out_dir / "instructions.svg", metadata={"Date": None})
        plt.close(figure)


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


def draw_charts_of(results_file, out_dir):
    """Draws the charts of a results file, with the .venv's matplotlib if this Python has none."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        subprocess.run(
            [str(VENV_PYTHON), str(ROOT / "run.py"), "chart", "--results", str(results_file), "--charts", str(out_dir)],
            check=True,
        )
        return
    draw_charts(json.loads(Path(results_file).read_text(encoding="utf-8")), out_dir)


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
    parser.add_argument("--update", type=Path,
                        help="rerun only --libraries, with the seeds and scenarios of this results file, "
                             "and keep its results of the other libraries")
    args = parser.parse_args()

    if args.command == "setup":
        setup()
        return
    if args.command == "chart":
        results_file = args.results or latest_results()
        draw_charts_of(results_file, args.charts)
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
        and (previous_scenarios is None or scenario_name(*scenario[:3]) in previous_scenarios)
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    platform = describe_platform()
    if previous and previous.get("platform") not in (None, platform):
        platform = f"{previous['platform']}; {', '.join(args.libraries)} rerun on {platform}"
    report = {"versions": versions, "seeds": seeds, "max_seconds": max_seconds, "platform": platform,
              "runs": runs, "summary": rows, "instructions": instructions}
    results_file = results / f"{timestamp}.json"
    results_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    header = [f"# Results {timestamp}", "", f"Seeds per scenario: {seeds}, wall time cap per run: {max_seconds} s", platform, ""]
    header += [f"- {name} {version}" for name, version in versions.items()] + [""]
    table = markdown_table(rows)
    # a blank line between the list and the table, or the table becomes part of the list
    (results / "latest.md").write_text("\n".join(header) + "\n" + table + "\n", encoding="utf-8")
    print()
    print(table)
    draw_charts_of(results_file, args.charts)


if __name__ == "__main__":
    main()
