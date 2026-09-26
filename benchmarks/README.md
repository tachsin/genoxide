# Benchmarks

16 evolutionary computation libraries in 5 languages, and genoxide's Python package, on the same problems, with the same fitness functions and evaluation budgets.

| Library | Language | Version | Solvers (single-objective; multi-objective) |
|---|---|---|---|
| [genoxide](../docs/benchmarks/libraries/genoxide.md) | Rust | this repository | GA, local search, CMA-ES (IPOP), DE, ES; NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA |
| [genoxide (Python)](../docs/benchmarks/libraries/genoxide_python.md) | Rust via Python | this repository ([python/](../python)) | GA, local search, CMA-ES (IPOP), DE; NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA |
| [genetic_algorithm](../docs/benchmarks/libraries/genetic_algorithm.md) | Rust | 0.27.3 | GA, hill climbing |
| [radiate](../docs/benchmarks/libraries/radiate.md) | Rust | 1.3.1 | GA, as its examples set it and with the crossovers its guide recommends; NSGA-II/III |
| [moors](../docs/benchmarks/libraries/moors.md) | Rust | 0.2.11 | GA |
| [openGA](../docs/benchmarks/libraries/openga.md) | C++ | 1.0.5+f9b15e7 | GA, as its Rastrigin example and its code generator set it |
| [pygmo](../docs/benchmarks/libraries/pygmo.md) | C++ via Python | 2.19.8 | SGA, IHS, GACO, SaDE, CMA-ES, simulated annealing, xNES; NSGA-II |
| [DEAP](../docs/benchmarks/libraries/deap.md) | Python | 1.4.4 | GA, BIPOP-CMA-ES, DE; NSGA-II/III |
| [pymoo](../docs/benchmarks/libraries/pymoo.md) | Python | 0.6.2 | GA, BRKGA, CMA-ES (IPOP), DE, ES, Nelder-Mead; NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA |
| [PyGAD](../docs/benchmarks/libraries/pygad.md) | Python | 3.7.0 | GA; NSGA-II/III |
| [pycma](../docs/benchmarks/libraries/pycma.md) | Python | 4.5.0 | IPOP-CMA-ES, BIPOP-CMA-ES, lq-CMA-ES |
| [Nevergrad](../docs/benchmarks/libraries/nevergrad.md) | Python | 1.0.12 | NgIohTuned, DiscreteOnePlusOne, PortfolioDiscreteOnePlusOne, RotatedTwoPointsDE, GeneticDE, ScrHammersleySearchPlusMiddlePoint, OnePlusOne, CMA |
| [SciPy](../docs/benchmarks/libraries/scipy.md) | Python | 1.18.1 | `differential_evolution`, `dual_annealing`, `direct`, `minimize` (L-BFGS-B, Nelder-Mead) |
| [Jenetics](../docs/benchmarks/libraries/jenetics.md) | Java | 9.1.0 | GA |
| [jMetal](../docs/benchmarks/libraries/jmetal.md) | Java | 7.5 | GA, ES, DE, CMA-ES; NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA |
| [Evolutionary.jl](../docs/benchmarks/libraries/evolutionary_jl.md) | Julia | 0.12.0 | GA, ES, CMA-ES, DE; NSGA-II |
| [Metaheuristics.jl](../docs/benchmarks/libraries/metaheuristics_jl.md) | Julia | 3.5.0 | GA, BRKGA, ECA, DE, PSO; NSGA-II/III, SPEA2, SMS-EMOA |

Each library's page gives its methods, their sources, what it leaves out, its separate test runs and its bugs. [notes.md](../docs/benchmarks/notes.md) indexes what each library can't run and the bugs found.

## Methodology

The full protocol is in [rules.md](../docs/benchmarks/rules.md). In short:

- **Problems:** every adapter implements the fitness functions below in its library's language. `run.py check` compares them with a Python reference.
- **Stop:** the target, the evaluation budget or 60 seconds, whichever comes first. A method that converges starts again ([rule 2.2](../docs/benchmarks/rules.md#2-the-budget)).
- **Seeds:** 10 per scenario. A solver whose first 3 seeds all reach the 60-second cap without the target stops there; `run.py` applies this to every library.
- **Time:** measured inside the adapter, around the optimization only. Every library runs single-threaded, one run at a time.
- **Results:** time and evaluations to target are the expected running time (ERT, [rule 8.1](../docs/benchmarks/rules.md#8-reporting)). Runs stopped by the time cap before their budget are "capped". The other results are medians.
- **Validation:** every timed run passes the checks of `run.py check`, or it's left out and listed.
- **Matched:** configurations as equal as the libraries allow, with each library's own components. They measure framework cost and algorithm implementations.
- **Idiomatic:** each library's recommended configuration, from its docs and examples, up to 3 solvers per problem type. They measure what its users get.
- **Multi-objective:** no target; each run uses its budget. `run.py` computes the hypervolume of the final front with the same code for every library.
- **Bugs** in the libraries aren't worked around, except where a library's page says so and shows both results.

**What time to target is made of.** Time to target is the number of evaluations times the cost of one evaluation, framework and fitness function together. With a cheap fitness function, as here, the library's own cost dominates. With an expensive one, the number of evaluations does, so the charts show evaluations to target and instructions per evaluation too.

**Cores.** Times must be measured on fixed cores. The published results come from WSL on an Intel Core Ultra 7 265K, pinned to its two fastest P-cores, 8 and 19. Under WSL, `run.py` refuses to measure times unless the virtual machine is pinned. Pin it with [pin-wsl.ps1](pin-wsl.ps1) (below), or pin WSL's `vmmemWSL` process to the same cores with [Process Lasso](https://bitsum.com/).

## Scenarios

| Problem | Genome | Fitness (minimized, except OneMax) | Target | Size (budget) |
|---|---|---|---|---|
| OneMax | bits | number of ones | all ones | matched 100 (200k), 1000 (2M); idiomatic 100 (200k) |
| N-Queens | permutation | diagonal conflicts | 0 | 32 (500k), 64 (1M) |
| Rastrigin, shifted | reals in [−5.12, 5.12] | 10 n + Σ (yᵢ² − 10 cos 2π yᵢ), yᵢ = xᵢ − sᵢ | ≤ 0.01 | 10 (500k), 30 (2M) |
| Rosenbrock | reals in [−5, 10] | Σ 100 (xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)² | ≤ 0.01 | 10 (500k) |
| Ackley, shifted | reals in [−32.768, 32.768] | Ackley's function of yᵢ = xᵢ − sᵢ | ≤ 0.01 | 30 (1M) |

Rastrigin and Ackley are shifted so that an optimum at the origin doesn't favour operators that drift towards 0. The shift sᵢ is in [rule 1.4](../docs/benchmarks/rules.md#1-the-problems).

**Matched OneMax** is DEAP's `examples/ga/onemax.py` with `eaSimple`:
- 300 individuals;
- tournament selection of 3, with replacement;
- two-point crossover on each consecutive pair, with probability 0.5;
- each individual mutated with probability 0.2, each bit flipping with probability 1 / n;
- generational replacement, no elitism;
- an individual neither crossed nor mutated keeps its fitness and isn't evaluated again.

A library that lacks one of these components doesn't run it ([rule 6.1](../docs/benchmarks/rules.md#6-which-methods-run)).

Multi-objective scenarios:

| Problem | Variables | Objectives | Budget | Hypervolume reference point |
|---|---|---|---|---|
| ZDT1 (convex front) | 30 in [0, 1] | 2 | 25,000 | (1.1, 1.1) |
| ZDT2 (concave front) | 30 in [0, 1] | 2 | 25,000 | (1.1, 1.1) |
| ZDT3 (disconnected front) | 30 in [0, 1] | 2 | 25,000 | (1.1, 1.1) |
| DTLZ2 (spherical front) | 12 in [0, 1] | 3 | 25,000 | (1.1, 1.1, 1.1) |
| DTLZ1 (linear front, multimodal) | 7 in [0, 1] | 3 | 40,000 | (0.55, 0.55, 0.55) |

They use the matched settings, with the library's own SBX and polynomial mutation and no duplicate elimination:
- **NSGA-II, SPEA2 and SMS-EMOA:** 100 individuals (92 with 3 objectives), SBX with η 15 at 0.9, polynomial mutation with η 20 at 1 / n. SMS-EMOA is steady-state: one child per step.
- **NSGA-III:** Das-Dennis reference directions (99 divisions with 2 objectives, 12 with 3), SBX with η 30 at 1. It runs on every problem, with 100 individuals on ZDT.
- **MOEA/D:** 100 weight vectors (91 with 3 objectives), 20 neighbors, parents from the neighborhood with probability 0.9, Tchebycheff (PBI with θ 5 for DTLZ), SBX with η 20 at 1.

## Running

This needs Linux or WSL, with Python 3.11+, Rust (cargo) and g++. Under WSL, pin the virtual machine first, from an Administrator PowerShell. `-Install` adds a scheduled task that pins it at logon and every minute. On another machine, pass its cores to both the script (`-Cores`) and `run.py` (`--cores`):

```powershell
benchmarks\pin-wsl.ps1 -Install
```

The Java and Julia adapters download their runtimes and packages into `~/opt` on their first build. The C++, Java and Julia builds go to `~/bench-targets`. The charts use the [Inter](https://rsms.me/inter/) font if it's in `~/.local/share/fonts`, and DejaVu Sans otherwise.

```sh
cd benchmarks
python run.py setup      # .venv with the Python libraries from requirements.txt (uses uv if installed)
python run.py check     # test the adapters against the rules (a timed run needs it)
python run.py --quick    # small scenarios, 3 seeds
python run.py            # all scenarios, 10 seeds
python run.py --scenarios nqueens-64-idiomatic --seeds 20 --libraries genetic_algorithm deap
python run.py chart      # redraw the charts of the latest results (--png also draws PNG previews)
```

`--allow-unpinned` skips the pinning check, for runs whose times don't count.

Each run writes the raw runs to `results/<timestamp>.json`, tables to `results/latest.md` (published as [docs/benchmarks/results.md](../docs/benchmarks/results.md)), and charts to `results/charts/`.

**Rerunning some libraries.** `--update` reruns only the libraries of `--libraries`, on every scenario of the file with its seeds and time cap, and keeps the others' results:

```sh
python run.py --update results/<timestamp>.json --libraries genoxide genoxide_python
```

- It can't be combined with `--scenarios` or `--quick`. To add a scenario, rerun every library.
- It first reruns DEAP's GA in matched OneMax 100, seeds 0 to 2. It refuses to go on if the median time differs from the file's by more than 3%, or if the evaluations differ. `--allow-drift` reruns anyway.
- Without Valgrind, or with `--no-instructions`, the previous instruction counts are kept.
- If the platform differs from the file's, the new file records both.

**The version measured.** `--version-label genoxide=0.7.0` records genoxide and its Python package as 0.7.0, still followed by the commit, for a release benchmarked before `Cargo.toml` is bumped. It works for any library of `--libraries`.

## Instructions per evaluation

On Linux with [Valgrind](https://valgrind.org/), each run also counts CPU instructions with Callgrind; it can't run the Java and Julia runtimes. Each adapter runs matched OneMax 1000 with budgets of N = 3,000 and 2N evaluations. The difference, I(2N) − I(N), divided by the difference in evaluations, cancels startup and setup: what's left is the cost of one evaluation. A library that can't run matched OneMax has no count.

## Adding a library

An adapter follows [the rules](../docs/benchmarks/rules.md) and has two commands. The first runs the solvers:

```
<adapter> <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
```

It prints one JSON line per solver per seed, with the best solution found:

```json
{"library": "deap", "solver": "ga", "problem": "onemax", "size": 100, "mode": "matched", "seed": 0,
 "time_s": 0.21, "generations": 37, "evaluations": 7041, "last_generation": 183, "best": 100,
 "target": 100, "success": true, "first_hit": {"evaluations": 6912, "time_s": 0.206},
 "solution": [1, 1, 1, ...]}
```

- `first_hit`: the number and clock of the first evaluation that reaches the target, or `null` ([rule 3.3](../docs/benchmarks/rules.md#3-counting-evaluations)).
- `outside`, in a continuous or multi-objective run: the evaluated solutions outside the bounds, which must be 0 (rule 2.4).
- `last_generation`: the evaluations the adapter counted since the start of the run's last generation, in every run ([rule 2.3](../docs/benchmarks/rules.md#2-the-budget)). A restart's initial population counts as a generation, and a method that evaluates one solution per step has generations of 1.
- A multi-objective run prints `"front": [[f1, f2], ...]` and `"solutions": [[x1, x2, ...], ...]` instead of `best`, `target`, `success`, `first_hit` and `solution`.
- For a problem its library can't do, an adapter prints nothing.

The second reads one JSON solution per line and prints its value, or its list of objectives, with the adapter's own fitness functions:

```
<adapter> values <problem> <size>
```

Register the adapter in `ADAPTERS` in `run.py`, with its language, write its page in [docs/benchmarks/libraries/](../docs/benchmarks/libraries/), and run `python run.py check --libraries <name>`. A timed run measures only adapters that passed the check as they are now. A change to the adapter, to `problems.py` or to `requirements.txt` (and, for genoxide, to its sources) needs a new check.
