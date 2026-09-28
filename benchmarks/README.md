# Benchmarks

A small, matched suite: three problems, one method each, and every library that has its own implementation of a problem's method runs it, set to the same written definition. genoxide and its Python package run all three.

**Interactive results, a card per problem: [tachsin.gr/projects/genoxide/benchmarks](https://tachsin.gr/projects/genoxide/benchmarks).** The full tables are in [results.md](../docs/benchmarks/results.md).

## Why small and matched

The suite used to run many methods per library and chart each library's fastest one. That compared libraries' picks, not their implementations: a library with a good DE beat one with a slow GA on a problem where DE suits better, and nothing said whether either implementation was right. So for now the suite has three problems, each with one method, defined down to its operators, parameters, selection, bounds and termination in [rule 6](../docs/benchmarks/rules.md#6-the-methods). Every library runs that method, or doesn't run the problem, and its page says how its configuration maps to the definition and where it differs. The charts show a bar per library for the same algorithm.

More problems, and multi-objective ones, come back after these, each with one matched method ([ROADMAP.md](../ROADMAP.md#benchmarks)).

## The suite

| Problem | Genome | Method | Target | Budget |
|---|---|---|---|---|
| OneMax 1000 (maximized) | 1000 bits | **GA**, DEAP's `eaSimple`: 300 individuals, tournaments of 3, two-point crossover of each pair at 0.5, each child mutated at 0.2 with a bit flip at 1/n, generational, no elitism | all ones | 2,000,000 |
| Rastrigin 30, shifted (minimized) | 30 reals in [−5.12, 5.12] | **DE/rand/1/bin**: 100 individuals, F 0.5, CR 0.9, one-to-one replacement, generational, no adaptation or restarts | none: a fixed budget | 300,000 |
| Rosenbrock 10 (minimized) | 10 reals in [−5, 10] | **CMA-ES** with Hansen's defaults: 10 samples, the best 5 with positive log weights, a uniform mean, σ₀ 0.3 of the range, no restarts; pycma is the reference | ≤ 0.01 | 500,000 |

Every run has a time cap of 60 seconds. With these settings DE/rand/1/bin reaches an error of 0.01 on Rastrigin 30 in no library, so that problem has no target: every run uses the fixed budget, and the charts show the time for it and the error at the end. The libraries run the same algorithm, so their errors should agree within the seeds' spread; one far off points to a difference or a bug ([rule 6.3](../docs/benchmarks/rules.md#6-the-methods)). Rastrigin is shifted so that an optimum at the origin doesn't favour operators that drift towards 0 ([rule 1.4](../docs/benchmarks/rules.md#1-the-problems)).

| Library | Language | Version | OneMax 1000: GA | Rastrigin 30: DE | Rosenbrock 10: CMA-ES |
|---|---|---|---|---|---|
| [genoxide](../docs/benchmarks/libraries/genoxide.md) | Rust | this repository | `Ga` | `De` | `Cmaes` |
| [genoxide (Python)](../docs/benchmarks/libraries/genoxide_python.md) | Rust via Python | this repository ([python/](../python)) | `gx.Ga` | `gx.De` | `gx.Cmaes` |
| [DEAP](../docs/benchmarks/libraries/deap.md) | Python | 1.4.4 | `eaSimple` | – | `cma.Strategy` |
| [PyGAD](../docs/benchmarks/libraries/pygad.md) | Python | 3.7.0 | `pygad.GA` | – | – |
| [radiate](../docs/benchmarks/libraries/radiate.md) | Rust | 1.3.2 | `GeneticEngine` | – | – |
| [pycma](../docs/benchmarks/libraries/pycma.md) | Python | 4.5.0 | – | – | `CMAEvolutionStrategy` |
| [SciPy](../docs/benchmarks/libraries/scipy.md) | Python | 1.18.1 | – | `differential_evolution` | – |
| [pygmo](../docs/benchmarks/libraries/pygmo.md) | C++ via Python | 2.19.8 | – | `de` | `cmaes` |
| [pymoo](../docs/benchmarks/libraries/pymoo.md) | Python | 0.6.2 | – | `DE` | `CMAES` |
| [jMetal](../docs/benchmarks/libraries/jmetal.md) | Java | 7.5 | – | `DifferentialEvolution` | `CovarianceMatrixAdaptationEvolutionStrategy` |
| [Evolutionary.jl](../docs/benchmarks/libraries/evolutionary_jl.md) | Julia | 0.12.1 | `GA` | – | `CMAES` |
| [Metaheuristics.jl](../docs/benchmarks/libraries/metaheuristics_jl.md) | Julia | 3.5.0 | – | `DE` | – |

The versions are the adapters' pins; the published results record the ones they measured. Each library's page gives its configuration against each definition, its differences, what it can't run and why, its separate test runs and its bugs. [notes.md](../docs/benchmarks/notes.md) indexes them.

## Methodology

The full protocol is in [rules.md](../docs/benchmarks/rules.md). In short:

- **Problems:** every adapter implements the fitness functions in its library's language. `run.py check` compares them with a Python reference.
- **Methods:** the library's own implementation, set with its documented options to the definition ([rule 6](../docs/benchmarks/rules.md#6-the-methods)). A difference that doesn't change the algorithm, such as the bound handling, is listed; one that does, such as a dithered F or a steady-state DE, leaves the library out of the problem.
- **Stop:** the target (OneMax and Rosenbrock), the evaluation budget or the time cap, whichever comes first. The methods have no convergence criterion and no restarts; a GA attempt that evaluates nothing for 10 generations has stalled and starts again ([rule 2.2](../docs/benchmarks/rules.md#2-the-budget)).
- **Seeds:** 10 per problem. A method whose first 3 seeds all reach the time cap without the target (or, for Rastrigin, the budget) stops there.
- **Time:** measured inside the adapter, around the optimization only. Every library runs single-threaded, one run at a time.
- **Results:** time and evaluations to target are the expected running time (ERT, [rule 8.1](../docs/benchmarks/rules.md#8-reporting)), given when at least 3 runs reached the target; the distance to the optimum at the end shows how close the runs came. Rastrigin, without a target, shows the median time of the runs that used its whole fixed budget and the error at the end of every run; a run the library ends early is marked. Runs stopped by the time cap before their budget are "capped", and the charts show them apart.
- **Validation:** every timed run passes the checks of `run.py check`, or it's left out and listed.
- **Bugs** in the libraries aren't worked around, except where a library's page says so and shows both results.

**What time to target is made of.** Time to target is the number of evaluations times the cost of one evaluation, framework and fitness function together. With a cheap fitness function, as here, the library's own cost dominates. With an expensive one, the number of evaluations does, so the charts show evaluations to target too. Where the libraries run the same algorithm, their evaluations should be close; a large gap points to a difference in the implementation.

**Cores.** Times must be measured on fixed cores. The published results come from WSL on an Intel Core Ultra 7 265K, pinned to its two fastest P-cores, 8 and 19. Under WSL, `run.py` refuses to measure times unless the virtual machine is pinned. Pin it with [pin-wsl.ps1](pin-wsl.ps1) (below), or pin WSL's `vmmemWSL` process to the same cores with [Process Lasso](https://bitsum.com/).

## Running

This needs Linux or WSL, with Python 3.11+ and Rust (cargo). Under WSL, pin the virtual machine first, from an Administrator PowerShell. `-Install` adds a scheduled task that pins it at logon and every minute. On another machine, pass its cores to both the script (`-Cores`) and `run.py` (`--cores`):

```powershell
benchmarks\pin-wsl.ps1 -Install
```

The Java and Julia adapters download their runtimes and packages into `~/opt` on their first build. The Java build goes to `~/bench-targets`. The charts use the [Inter](https://rsms.me/inter/) font if it's in `~/.local/share/fonts`, and DejaVu Sans otherwise.

```sh
cd benchmarks
python run.py setup      # .venv with the Python libraries from requirements.txt (uses uv if installed)
python run.py check      # test the adapters against the rules (a timed run needs it), --jobs at a time
python run.py --quick    # every problem, 3 seeds
python run.py            # every problem, 10 seeds
python run.py --scenarios rosenbrock-10-matched --seeds 20 --libraries pycma genoxide
python run.py chart      # redraw the charts of the latest results (--png also draws PNG previews)
python run.py publish    # the latest run into docs/benchmarks (below)
python run.py versions --genoxide 0.9.1  # count the instructions of genoxide 0.9.1's runs (below)
```

`--allow-unpinned` skips the pinning check, for runs whose times don't count.

Each run writes the raw runs to `results/<timestamp>.json`, tables to `results/latest.md`, and charts to `results/charts/`: `time_to_target.svg` (for Rastrigin, the median time for its fixed budget), `evaluations_to_target.svg` (the problems with a target) and `distance_to_optimum.svg` (for Rastrigin, the error at the end), each a panel per problem and a bar per library, and `charts.json`, the numbers, labels and notes of every chart with the run's date, machine and versions, written from the same summaries as the SVGs. A problem the run has no runs of is drawn as awaiting the next run. The project site's [benchmarks page](https://tachsin.gr/projects/genoxide/benchmarks) draws a card per problem from `charts.json`.

Beside it, `runs/<scenario>.json` holds the run details, one file per problem, so the page reads only the one it shows when a bar is clicked:
- **The scenario:** its problem, size, method, budget, time cap, target (or `fixed_budget`) and seeds.
- **Per library:** its version and language, its summary as the charts show it (and apart, the runs the time cap stopped), and every run. A run has its seed, time, evaluations, best value, whether it reached the target and its first hit, whether the time cap stopped it, `ended_by` if the library ended it, why it's invalid if it is, and `output`, the JSON line its adapter printed.
- **Its code:** the blocks of the adapter that set the method up and run it. Each block has its file, lines and text, at most 80 lines, and the commit whose file has those lines, if the file is unchanged from it. [method_code.py](method_code.py) maps each library's methods to these blocks by the text of their first line, not by line numbers, so the map survives edits and the adapters don't change for it. A method the map doesn't know has no code in the details, and a block it can't find is printed.

`chart` and `publish` write the same files from any results file, raw or published.

**Publishing a run.** `python run.py publish` (`--results <file>` for another run than the latest) writes the latest run of `results/` into [docs/benchmarks/](../docs/benchmarks/): the tables, [results.md](../docs/benchmarks/results.md), summarized from its runs; the SVGs, `charts.json` and the run details, [runs/](../docs/benchmarks/runs/); and the run itself, [results.json.xz](../docs/benchmarks/results.json.xz), compressed with xz, which Python reads without a package (`lzma`). Commit the files it changes.

The published run is the default of `chart` and `--update`, so anyone can redraw the published charts: `python run.py chart` draws the charts of the published run, or of a newer run in `results/`, into `results/charts/`; `python run.py chart --results <file> --charts <folder>` draws a given results file, the run's JSON or the published `.json.xz`.

**The published run now.** The suite became matched after the last timed run. That run's OneMax 1000 runs are the suite's GA, so they're published: genoxide and its Python package, DEAP, PyGAD, radiate and Evolutionary.jl, at the versions it measured. It has no runs of the suite's DE and CMA-ES: Rastrigin 30 and Rosenbrock 10 await the next full rerun.

**The full rerun.** On the machine of the published results, [rerun.sh](rerun.sh) runs every step in turn, in WSL, and stops at the first that fails: `run.py setup`, `run.py check`, the timed run (every library, 10 seeds), `run.py versions --genoxide <version>` and `run.py publish`. It logs everything to `results/rerun-<timestamp>.log`. The version is genoxide's release on crates.io, or `path` for the repository's genoxide before the release. It takes about an hour; nothing else may run meanwhile.

```sh
benchmarks/rerun.sh 0.9.1
```

**Rerunning some libraries.** `--update` reruns only the libraries of `--libraries`, on every problem of the published run (or of `--update <file>`) with its seeds and time caps, and keeps the others' results. Then publish it:

```sh
python run.py --update --libraries genoxide genoxide_python
python run.py publish
```

- It can't be combined with `--scenarios` or `--quick`. To add a problem, rerun every library.
- It first reruns DEAP's GA on OneMax 1000, seeds 0 to 2. It refuses to go on if the median time differs from the file's by more than 3%, or if the evaluations differ. `--allow-drift` reruns anyway. After a change to DEAP's adapter (such as its 2026-09-29 switch to `array.array` individuals), the reference itself changed: rerun every library, or DEAP alone with `--allow-drift` on the machine of the published results.
- If the platform differs from the file's, the new file records both.

**New releases.** `python run.py outdated` lists each library's pinned version, the version in the published results and its latest release. It marks a newer release "newer", and a pin that differs from the published results "not rerun". To benchmark a newer release, update the library's pin, check that its page still maps its configuration to the definitions, run `python run.py check --libraries <name>`, then rerun it alone with `--update` and publish it. A weekly workflow keeps an issue, "New releases of benchmarked libraries", open while any library has a newer release or isn't rerun at its pin.

**The version measured.** `--version-label genoxide=0.7.0` records genoxide and its Python package as 0.7.0, still followed by the commit, for a release benchmarked before `Cargo.toml` is bumped. It works for any library of `--libraries`.

## genoxide's versions: instruction counts

CPU instructions are counted for genoxide only, to see how its versions solve the same problems ([rule 10](../docs/benchmarks/rules.md#10-instruction-counts-genoxides-versions)). The other libraries have times and evaluations, not instruction counts.

`python run.py versions --genoxide 0.9.1` measures genoxide 0.9.1, with Callgrind, on Linux with [Valgrind](https://valgrind.org/):

- **The build.** The genoxide adapter, as it is in the repository, is copied to `adapters/genoxide/target/versions/0.9.1/` with `genoxide = "=0.9.1"` from crates.io in its Cargo.toml, and built there with the adapter's Cargo.lock. Nothing in the repository changes. `--genoxide path` builds it against the repository's genoxide instead, for an unreleased version: its row is labelled with the version and commit, e.g. `0.9.1+034f3cd`. There's at most one such row, and a newer release replaces it. A release whose API the adapter doesn't compile against is reported and skipped. The history starts at 0.6.0: the adapter doesn't compile against older releases, which have no `de::Restarts`. Several versions can be given at once.
- **The runs.** In every problem, genoxide's method runs once with seed 0, to its target or its evaluation budget, with no time cap. The adapter runs one method per process when `GENOXIDE_BENCH_SOLVER` names it, and none with a name no method has: that process, the startup, is subtracted from each method's count. Each run is made without Callgrind too, and the two must be the same run.
- **The history.** Each version is a row of [genoxide-versions.json](../docs/benchmarks/genoxide-versions.json), added or replaced: the version, its release date, the day it was measured, the machine, rustc and Valgrind, and per problem the startup and, per method, the instructions, the evaluations, and whether it reached the target and its best value.
- **The chart.** It redraws [genoxide_versions.svg](../docs/benchmarks/genoxide_versions.svg), a panel per problem with the versions on the x axis, and its numbers in `charts.json` beside it. `python run.py versions` without `--genoxide` only redraws them. `python run.py chart` draws it too, from the same history file (`--history` for another).

The counts don't depend on the load, so WSL needn't be pinned: it runs up to `--jobs` processes at once (default: one per core). Under Callgrind a run takes about 50 times as long; a version takes a few minutes. A count changes with genoxide and also with the adapter, rustc or Valgrind: compare versions measured with the same ones, which each row records. After a change to the adapter or the toolchain, measure every version again. The build differs a little too: the same sources, built from crates.io and from the repository, count up to about 0.1% apart, so a change between a repository row and a release smaller than that isn't genoxide's.

**A new release.** After a release is published on crates.io, add it to the history (a step of the release checklist in [CONTRIBUTING.md](../CONTRIBUTING.md#benchmarks)):

```sh
cd benchmarks
python run.py versions --genoxide 0.9.2   # measures it, adds its row, redraws the chart
```

Then commit `docs/benchmarks/genoxide-versions.json`, `genoxide_versions.svg` and `charts.json`.

## Adding a library

A library joins the suite for each problem whose method it implements itself. Its adapter follows [the rules](../docs/benchmarks/rules.md), sets the library's implementation to the definitions of rule 6, and has two commands. The first runs the method of a problem:

```
<adapter> <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
```

The mode is `matched`. It prints one JSON line per seed, with the best solution found:

```json
{"library": "deap", "solver": "ga", "problem": "onemax", "size": 1000, "mode": "matched", "seed": 0,
 "time_s": 15.07, "generations": 566, "evaluations": 102333, "last_generation": 183, "best": 1000,
 "target": 1000, "success": true, "first_hit": {"evaluations": 102188, "time_s": 15.066},
 "solution": [1, 1, 1, ...]}
```

- `solver`: `ga`, `de` or `cma_es`, the problem's method.
- `first_hit`: the number and clock of the first evaluation that reaches the target, or `null` ([rule 3.3](../docs/benchmarks/rules.md#3-counting-evaluations)). A problem without a target (Rastrigin) prints `"target": null`, `"success": false` and `"first_hit": null`.
- `outside`, in a continuous run: the evaluated solutions outside the bounds, which must be 0 (rule 2.4).
- `last_generation`: the evaluations the adapter counted since the start of the run's last generation, in every run ([rule 2.3](../docs/benchmarks/rules.md#2-the-budget)).
- `ended_by`, only in a run the library ended with an error ([rule 8.4](../docs/benchmarks/rules.md#8-reporting)): the error.
- For a problem whose method its library doesn't have, an adapter prints nothing.

The second reads one JSON solution per line and prints its value, with the adapter's own fitness functions:

```
<adapter> values <problem> <size>
```

Register the adapter in `ADAPTERS` in `run.py`, with its language and where its releases are published (`release`), map its methods to their code in `METHOD_CODE` in [method_code.py](method_code.py), write its page in [docs/benchmarks/libraries/](../docs/benchmarks/libraries/), and run `python run.py check --libraries <name>`. A timed run measures only adapters that passed the check as they are now. A change to the adapter, to `problems.py` or to `requirements.txt` (and, for genoxide, to its sources) needs a new check.
