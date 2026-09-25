# Benchmarks

16 evolutionary computation libraries in 5 languages, and genoxide's Python package, on the same problems, with the same fitness functions and evaluation budgets.

| Library | Language | Version | Solvers benchmarked (single-objective; multi-objective) |
|---|---|---|---|
| [genoxide](../docs/benchmarks/libraries/genoxide.md) | Rust | this repository | GA, GA with islands, DE, CMA-ES (IPOP), PSO, local search, tabu search; NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA |
| [genoxide (Python)](../docs/benchmarks/libraries/genoxide_python.md) | Rust via Python | this repository ([python/](../python)) | GA, tabu search, CMA-ES (IPOP), L-SHADE, DE, PSO; NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA |
| [genetic_algorithm](../docs/benchmarks/libraries/genetic_algorithm.md) | Rust | 0.27.3 | GA, hill climbing |
| [radiate](../docs/benchmarks/libraries/radiate.md) | Rust | 1.3.1 | GA, as its examples set it and with the crossovers its guide recommends; NSGA-II/III |
| [moors](../docs/benchmarks/libraries/moors.md) | Rust | 0.2.11 | GA; NSGA-II/III, SPEA2 |
| [openGA](../docs/benchmarks/libraries/openga.md) | C++ | 1.0.5+f9b15e7 | GA, as its Rastrigin example and its code generator set it; NSGA-III |
| [pygmo](../docs/benchmarks/libraries/pygmo.md) | C++ via Python | 2.19.8 | SGA, IHS, GACO, SaDE, CMA-ES, simulated annealing, xNES; NSGA-II |
| [DEAP](../docs/benchmarks/libraries/deap.md) | Python | 1.4.4 | GA, BIPOP-CMA-ES, DE; NSGA-II/III |
| [pymoo](../docs/benchmarks/libraries/pymoo.md) | Python | 0.6.2 | GA, BRKGA, CMA-ES (IPOP), DE, ES, Nelder-Mead; NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA |
| [PyGAD](../docs/benchmarks/libraries/pygad.md) | Python | 3.7.0 | GA; NSGA-II/III |
| [pycma](../docs/benchmarks/libraries/pycma.md) | Python | 4.5.0 | IPOP-CMA-ES, BIPOP-CMA-ES, lq-CMA-ES |
| [Nevergrad](../docs/benchmarks/libraries/nevergrad.md) | Python | 1.0.12 | NgIohTuned, DiscreteOnePlusOne, PortfolioDiscreteOnePlusOne, RotatedTwoPointsDE, GeneticDE, ScrHammersleySearchPlusMiddlePoint, OnePlusOne, CMA |
| [SciPy](../docs/benchmarks/libraries/scipy.md) | Python | 1.18.1 | `differential_evolution`, `dual_annealing`, `direct`, `minimize` (L-BFGS-B, Nelder-Mead) |
| [Jenetics](../docs/benchmarks/libraries/jenetics.md) | Java | 9.1.0 | GA; NSGA-II |
| [jMetal](../docs/benchmarks/libraries/jmetal.md) | Java | 7.5 | GA, ES, DE, CMA-ES; NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA |
| [Evolutionary.jl](../docs/benchmarks/libraries/evolutionary_jl.md) | Julia | 0.12.0 | GA, ES, CMA-ES, DE; NSGA-II |
| [Metaheuristics.jl](../docs/benchmarks/libraries/metaheuristics_jl.md) | Julia | 3.5.0 | GA, BRKGA, ECA, DE, PSO; NSGA-II/III, SPEA2, SMS-EMOA |

Each library's page in [docs/benchmarks/libraries/](../docs/benchmarks/libraries/) gives its methods and where its docs recommend them, the methods left out and why, its separate test runs, and the bugs found. [notes.md](../docs/benchmarks/notes.md) indexes what each library can't run, the bugs, and the rule-level choices that affect many libraries.

## Methodology

**The same problems.** Every adapter implements the fitness functions below in its library's language, as the library's users would. Their values are checked against a Python reference at the optimum and at fixed random points.

**The same rules.** A run stops at the first of these:
- the target is reached
- the scenario's evaluation budget is used up
- 60 seconds have passed

Libraries that check the stop between generations can go past the budget by up to one generation. Every result reports the true number of evaluations. A method that ends by its own convergence criterion starts again, with the library's restart mechanism or from a new random start ([rule 2.2](../docs/benchmarks/rules.md#2-the-budget)); a limit that's only a budget, such as a number of generations, is lifted.

A solver whose first 3 seeds all run for the full 60 seconds without reaching the target (a multi-objective run has none) stops there: the other 7 would take a minute each for the same result. Its result shows 3 runs instead of 10: "0 of 3". The adapters of Nevergrad, PyGAD and Metaheuristics.jl skip those seeds themselves. `run.py` applies the same rule to every library's results.

**Time.** It's measured inside the adapter, around the optimization only. It doesn't include interpreter or JVM startup, imports or setup, and the Java and Julia adapters make an untimed run first, so JIT compilation isn't included either. Every library runs single-threaded, one run at a time, on one machine. Each scenario runs 10 seeds.

**Time and evaluations to target** are the expected running time (ERT, [rule 8.1](../docs/benchmarks/rules.md#8-reporting)): what all runs spent, up to the first hit of the target in the runs that reached it and in full in the others, divided by the number of runs that reached it. Each run records its first hit at the evaluation itself, even where the library checks its stop only between generations. Runs stopped by the 60-second cap before their budget are counted as "capped": they're limited by speed, not by the search. The other results are medians.

**Every timed run is validated** with the checks of `run.py check`: its solution re-evaluated, its evaluations within the budget, its first hit and its front consistent. A run that fails is left out of every table and chart, and the results list it.

**Cores.** The published results come from WSL on an Intel Core Ultra 7 265K, whose 8 P-cores and 12 E-cores run at different speeds, and whose P-cores don't all reach the same turbo frequency. Windows decides which core runs the WSL virtual machine, so without pinning, a run's time depends on where it lands. So the virtual machine is pinned to the two fastest P-cores, 8 and 19, and `run.py` refuses to measure times unless it is.

Fixed single-threaded loops, each measured 30 times; the spread is the slowest run against the fastest:

| Where it runs | C loop: spread | C loop: standard deviation | Python loop: spread |
|---|---|---|---|
| unpinned | not measured | not measured | 16.4% |
| the 8 P-cores | 5.6% | 1.25% | 15.7% |
| **P-cores 8 and 19** | **4.0%** | **0.99%** | **14.1%** |

The C loop's remaining variation is the Windows host itself. Python adds its own variation on top, from memory management, which is part of what a Python library costs. With the median of 10 seeds, a time's noise is about 0.5%.

**What time to target is made of.** Time to target is the number of evaluations multiplied by the cost of one evaluation, and the charts show both factors:
- **Evaluations to target** measure how efficient the search is. They don't depend on the language, and they're what matters when the fitness function is expensive.
- **Instructions per evaluation** measure the cost of one evaluation: the framework and the fitness function together. Callgrind counts them exactly, for every library it can run; it can't run the Java and Julia runtimes.

The fitness functions here are cheap, so the time charts mostly show framework and language cost. OneMax 1000 in the matched configuration is an example:

| | Time to target | Evaluations | Evaluations per second |
|---|---|---|---|
| genoxide | 15.4 ms | 54,038 | 3,395,068 |
| DEAP | 15.82 s | 105,287 | 6,681 |
| Ratio | 1,000× | 1.9× | 508× |

- **The evaluations:** genoxide needs half as many as DEAP, because it doesn't evaluate a child identical to its parent again; the child inherits the fitness.
- **The evaluations per second:** genoxide stores the genome as packed bits in compiled Rust. DEAP keeps a Python object per individual, and its operators and fitness function are Python.
- **An expensive fitness function:** with a fitness function that takes a millisecond, the second factor would nearly vanish, and genoxide would be about 2× faster, not 1,000×.

**Matched and idiomatic.**
- **Matched:** configurations as equal as the libraries allow. This measures framework cost and algorithm implementations. They're close, not identical: each library's page lists its differences.
- **Idiomatic:** each library's recommended configuration, from its docs and examples. This measures what its users get. A library can run up to 3 solvers per problem type here.

**Multi-objective quality.** These runs have no target: each one uses its evaluation budget. The adapter prints its final non-dominated front and the solutions of its points. `run.py` computes the objectives of the solutions itself, keeps the non-dominated ones, and computes their hypervolume with the same exact code, so no library's own values or indicator are involved.

**Bugs in the libraries** aren't worked around, except where the [notes](../docs/benchmarks/notes.md) say so. When a library's own operator is wrong, its results show it. The notes list the bug, and the library's page shows what the library reaches without it, where that was measured.

## Scenarios

| Problem | Genome | Fitness (minimized, except OneMax) | Target | Size (budget) |
|---|---|---|---|---|
| OneMax | bits | number of ones | all ones | matched 100 (200k), 1000 (2M); idiomatic 100 (200k) |
| N-Queens | permutation | diagonal conflicts | 0 | 32 (500k), 64 (1M) |
| Rastrigin, shifted | reals in [−5.12, 5.12] | 10 n + Σ (yᵢ² − 10 cos 2π yᵢ), yᵢ = xᵢ − sᵢ | ≤ 0.01 | 10 (500k), 30 (2M) |
| Rosenbrock | reals in [−5, 10] | Σ 100 (xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)² | ≤ 0.01 | 10 (500k) |
| Ackley, shifted | reals in [−32.768, 32.768] | Ackley's function of yᵢ = xᵢ − sᵢ | ≤ 0.01 | 30 (1M) |

**Why shifted:** Rastrigin and Ackley usually have their optimum at the origin, which favours operators that drift towards 0. radiate's arithmetic mutation, for example, solves Rastrigin 10 in 17,000 to 27,000 evaluations with the optimum at the origin, and needs 250,000 to 400,000 with it moved. So gene i, from 0, is measured from sᵢ = 0.8 · upper · (2 ((37 i + 11) mod 101) / 101 − 1), where upper is the box's upper bound, 5.12 or 32.768: a fixed shift spread over 80% of the box, that every language computes exactly in this order ([rule 1.4](../docs/benchmarks/rules.md#1-the-problems)).

**Matched OneMax** is DEAP's `examples/ga/onemax.py` with `eaSimple`:
- 300 individuals;
- tournament selection of 3, with replacement;
- two-point crossover on each consecutive pair, with probability 0.5;
- each individual mutated with probability 0.2, each bit flipping with probability 1 / n;
- generational replacement, no elitism;
- an individual neither crossed nor mutated keeps its fitness, and isn't evaluated again.

A library runs it with its own components only. One that lacks one, or can only run it with elitism, as (μ+λ) or with another replacement, doesn't run it ([rule 6.1](../docs/benchmarks/rules.md#6-which-methods-run)).

Multi-objective scenarios:

| Problem | Variables | Objectives | Budget | Hypervolume reference point |
|---|---|---|---|---|
| ZDT1 (convex front) | 30 in [0, 1] | 2 | 25,000 | (1.1, 1.1) |
| ZDT2 (concave front) | 30 in [0, 1] | 2 | 25,000 | (1.1, 1.1) |
| ZDT3 (disconnected front) | 30 in [0, 1] | 2 | 25,000 | (1.1, 1.1) |
| DTLZ2 (spherical front) | 12 in [0, 1] | 3 | 25,000 | (1.1, 1.1, 1.1) |
| DTLZ1 (linear front, multimodal) | 7 in [0, 1] | 3 | 40,000 | (0.55, 0.55, 0.55) |

The reference point is 1.1 times the nadir of the optimal front: 1 for ZDT and DTLZ2, 0.5 for DTLZ1.

They use the matched settings:
- **NSGA-II, SPEA2 and SMS-EMOA:** 100 individuals (92 with 3 objectives), SBX with η 15 at 0.9, and polynomial mutation with η 20 at 1 / n.
- **SMS-EMOA** is steady-state: one child per step. A library whose SMS-EMOA can't do that doesn't run it.
- **NSGA-III:** Das-Dennis reference directions (99 divisions with 2 objectives, 12 with 3), SBX with η 30 at 1. It runs on the ZDT problems too, with 100 individuals, in every library that has it.
- **MOEA/D:** 100 weight vectors (91 with 3 objectives), 20 neighbors, parents from the neighborhood with probability 0.9, Tchebycheff (PBI with θ 5 for DTLZ), SBX with η 20 at 1.
- **In every algorithm:** the library's own SBX and polynomial mutation, bugs included, and no duplicate elimination.

## Notes on the libraries

How each library is run, and how its settings differ from the others', is on its page in [docs/benchmarks/libraries/](../docs/benchmarks/libraries/). What each library doesn't run and why, and the bugs found in the libraries, are indexed in [docs/benchmarks/notes.md](../docs/benchmarks/notes.md).

## Running

This needs Linux or WSL, with Python 3.11+, Rust (cargo) and g++. Under WSL, pin the virtual machine first, from an Administrator PowerShell. `-Install` adds a scheduled task that pins it at logon and every minute, since WSL's virtual machine restarts whenever WSL starts again. Another machine passes its own cores to both the script (`-Cores`) and `run.py` (`--cores`):

```powershell
benchmarks\pin-wsl.ps1 -Install
```

 The Java and Julia adapters download a JDK, Julia, their jars and their packages into `~/opt` on their first build. The C++, Java and Julia builds go to `~/bench-targets`. The charts use the [Inter](https://rsms.me/inter/) font if it's in `~/.local/share/fonts`, and DejaVu Sans otherwise.

```sh
cd benchmarks
python run.py setup      # .venv with the Python libraries from requirements.txt (uses uv if installed)
python run.py check     # test the adapters against the rules (a timed run needs it)
python run.py --quick    # small scenarios, 3 seeds
python run.py            # all scenarios, 10 seeds
python run.py --scenarios nqueens-64-idiomatic --seeds 20 --libraries genetic_algorithm deap
python run.py chart      # redraw the charts of the latest results (--png also draws PNG previews)
```

Each run writes:
- the raw runs, with the date, the library versions and languages, and the platform, to `results/<timestamp>.json`
- tables to `results/latest.md`, which becomes [docs/benchmarks/results.md](../docs/benchmarks/results.md): the coverage, the single-objective results with the evaluations to target, the multi-objective results, and the instructions per evaluation
- charts to `results/charts/`: time and evaluations to target, instructions per evaluation, and the hypervolume and time of the multi-objective runs

**Rerunning some libraries.** `--update` reruns only the libraries of `--libraries` and keeps the other libraries' results:

```sh
python run.py --update results/<timestamp>.json --libraries genoxide genoxide_python
```

The rerun libraries run every scenario of the results file, with its seeds and wall time cap, and their old runs, instruction counts and versions are replaced. The other libraries' runs are kept as they are. So `--update` can't be combined with `--scenarios` or `--quick`: some of a library's runs would be old ones under its new version. To add a scenario, rerun every library. If the platform differs from the file's, the new file records both. Without Valgrind, or with `--no-instructions`, the previous instruction counts are kept.

The kept times must still be comparable, so `--update` first runs a reference, DEAP's GA in matched OneMax 100 with seeds 0 to 2, and refuses to go on if its median time differs from the file's by more than 3%, or if its evaluations differ. `--allow-drift` reruns anyway.

**The version measured.** A minor release is benchmarked before its release PR bumps `Cargo.toml`, so `--version-label genoxide=0.7.0` records genoxide and its Python package as 0.7.0, still followed by the commit. Any library of `--libraries` can be labeled this way.

## Instructions per evaluation

On Linux with [Valgrind](https://valgrind.org/), each run also counts CPU instructions with Callgrind. Instruction counts are exact and don't depend on the machine's load or clock speed. They're the cost per evaluation: a generation's work is divided by the children the library evaluates in it, so the count depends on how many children a library evaluates per generation.

Each adapter runs OneMax 1000 in the matched configuration twice, with budgets of N and 2N evaluations. None of them can reach the target within that budget. The difference, I(2N) − I(N), divided by the difference in evaluations, cancels the interpreter startup, the imports and the setup. What's left is the cost of one evaluation, framework and fitness function together.

## Adding a library

An adapter follows [the rules](../docs/benchmarks/rules.md) and has two commands. The first runs the solvers:

```
<adapter> <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
```

It prints one JSON line per solver per seed, with the best solution found:

```json
{"library": "deap", "solver": "ga", "problem": "onemax", "size": 100, "mode": "matched", "seed": 0,
 "time_s": 0.21, "generations": 37, "evaluations": 7041, "best": 100, "target": 100, "success": true,
 "first_hit": {"evaluations": 6912, "time_s": 0.206}, "solution": [1, 1, 1, ...]}
```

`first_hit` is the first evaluation whose value reaches the target: its number, counting it, and the run's clock at that moment. It's `null` if the target is never reached. The adapter's counter records it at each evaluation ([rule 3.3](../docs/benchmarks/rules.md#3-counting-evaluations)).

A run of a continuous or multi-objective problem also prints `"outside"`, the number of evaluated solutions outside the bounds, counted around the fitness function. It must be 0 (rule 2.4). A solver whose generations change size, such as CMA-ES with IPOP restarts, also prints `"last_generation"`, the evaluations of its last generation: a run may go past its budget by at most that (rule 2.3). A multi-objective run prints `"front": [[f1, f2], ...]` and `"solutions": [[x1, x2, ...], ...]`, the solutions of those points in the same order, instead of `best`, `target`, `success`, `first_hit` and `solution`. For a problem its library can't do, an adapter prints nothing.

The second evaluates solutions with the adapter's own fitness functions:

```
<adapter> values <problem> <size>
```

It reads one JSON solution per line and prints its value, or its list of objectives, one per line.

Register the adapter in `ADAPTERS` in `run.py`, with its language, write its page in [docs/benchmarks/libraries/](../docs/benchmarks/libraries/), and run `python run.py check --libraries <name>`. A timed run measures only adapters that passed the check as they are now: a change to the adapter, to `problems.py` or to `requirements.txt` (and, for genoxide, to its sources) needs a new check.
