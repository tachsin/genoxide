"""Benchmarks of evolutionary computation libraries on the same problems.

Usage:
    python run.py setup                      # create .venv and install the Python libraries
    python run.py                            # all scenarios, 10 seeds
    python run.py --quick                    # small scenarios, 3 seeds
    python run.py --seeds 5 --scenarios onemax-100-matched nqueens-32-idiomatic
    python run.py --libraries deap genetic_algorithm

Results are written to results/<timestamp>.json (all runs) and results/latest.md (table).
"""

import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
RUST_ADAPTER = ROOT / "adapters" / "genetic_algorithm"

# Each adapter prints one JSON line per solver per seed, with the same command line:
#   <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
ADAPTERS = {
    "genetic_algorithm": {
        "build": ["cargo", "build", "--release", "--quiet", "--manifest-path", str(RUST_ADAPTER / "Cargo.toml")],
        "command": [str(RUST_ADAPTER / "target" / "release" / "ga_bench_genetic_algorithm")],
        "version": ("cargo", "genetic_algorithm"),
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
    if not VENV_PYTHON.exists():
        print(f"creating {VENV} ...", flush=True)
        venv.create(VENV, with_pip=True)
    subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)


def library_version(kind, package):
    if kind == "python":
        return subprocess.run(
            [str(VENV_PYTHON), "-c", f"import importlib.metadata as m; print(m.version('{package}'))"],
            capture_output=True, text=True, check=True, cwd=ROOT,
        ).stdout.strip()
    metadata = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--manifest-path", str(RUST_ADAPTER / "Cargo.toml")],
        capture_output=True, text=True, check=True,
    ).stdout)
    return next(p["version"] for p in metadata["packages"] if p["name"] == package)


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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["run", "setup"], default="run")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=60.0, help="wall time cap per run")
    parser.add_argument("--quick", action="store_true", help="small scenarios, 3 seeds")
    parser.add_argument("--scenarios", nargs="*", help="scenario names, e.g. onemax-100-matched (default all)")
    parser.add_argument("--libraries", nargs="*", default=list(ADAPTERS), choices=list(ADAPTERS))
    args = parser.parse_args()

    if args.command == "setup":
        setup()
        return
    if not VENV_PYTHON.exists():
        raise SystemExit("run `python run.py setup` first")

    seeds = 3 if args.quick and args.seeds == 10 else args.seeds
    scenarios = [
        scenario for scenario in SCENARIOS
        if (not args.quick or scenario_name(*scenario[:3]) in QUICK_SCENARIOS)
        and (not args.scenarios or scenario_name(*scenario[:3]) in args.scenarios)
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
            runs += run_adapter(ADAPTERS[name], problem, size, mode, seeds, max_evaluations, args.max_seconds)

    rows = summarize(runs)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / f"{timestamp}.json").write_text(
        json.dumps({"versions": versions, "seeds": seeds, "runs": runs, "summary": rows}, indent=2),
        encoding="utf-8",
    )
    header = [f"# Results {timestamp}", "", f"Seeds per scenario: {seeds}, wall time cap per run: {args.max_seconds} s", ""]
    header += [f"- {name} {version}" for name, version in versions.items()] + [""]
    table = markdown_table(rows)
    (results / "latest.md").write_text("\n".join(header) + table + "\n", encoding="utf-8")
    print()
    print(table)


if __name__ == "__main__":
    main()
