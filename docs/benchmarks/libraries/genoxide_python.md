# genoxide for Python (Rust via Python, 0.6.0)

genoxide's Python package, from this repository's [python/](../../../python/) folder: genoxide's algorithms, in Rust, calling fitness functions in Python and numpy. Its docs are [python/README.md](../../../python/README.md), its [examples](../../../python/examples/) and the docstrings of [python/genoxide/\_\_init\_\_.py](../../../python/genoxide/__init__.py).

The methods come from the Python package's docs, not the Rust library's ([genoxide.md](genoxide.md)). [Rule 6.6](../rules.md#6-which-methods-run) applies: the methods run with their defaults, and standard literature values (cited below) where there's no default. Settings from the examples aren't used.

Adapter: [benchmarks/adapters/genoxide_python/](../../../benchmarks/adapters/genoxide_python/).
Know a better way to solve one of these problems with genoxide's Python package? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs the package

- **Fitness functions:** numpy, as the docs write them ([bench.py, lines 62-224](../../../benchmarks/adapters/genoxide_python/bench.py#L62-L224)): a function per genome for OneMax and N-Queens, as python/README.md's first example, [onemax.py](../../../python/examples/onemax.py) and [n_queens.py](../../../python/examples/n_queens.py); `batch=True` (a generation per call, "one call per generation") for the real-valued and multi-objective problems, as [rastrigin.py](../../../python/examples/rastrigin.py) and [zdt1.py](../../../python/examples/zdt1.py). A batch doesn't change the algorithm ("The same seed repeats a run exactly: one genome at a time, in batches or in parallel"). `bench.py --self-check` compares every function with problems.py.
- **Evaluations:** each function counts its genomes, one per call or the rows of a batch ([`count`](../../../benchmarks/adapters/genoxide_python/bench.py#L77-L82)), and records the first hit. Any difference from `result.evaluations` is printed to stderr ([`solve`, lines 385-452](../../../benchmarks/adapters/genoxide_python/bench.py#L385-L452)).
- **Stop:** `run(..., target=..., evaluations=...)` after every generation, and the 60 s cap: `time=...` for CMA-ES, DE and local search, and an `on_generation` callback returning False for the GA.
- **Keeping going (rule 2.2):** DE and CMA-ES (IPOP) restart by themselves. Local search redraws a neighbor that didn't change, so it can't stall. The GA has no convergence criterion; the package ends a run by itself only as "stalled" (10,000 generations with nothing to evaluate), and then the adapter restarts it with the seeds of rule 2.2. The GA's time cap is a callback because the package stalls only when every stop condition needs new evaluations. No test run stalled.
- **Bounds (rule 2.4):** genoxide's: CMA-ES redraws a sample outside the bounds up to 100 times, then clips it; DE sets an outside trial gene halfway between the parent's gene and the bound; SBX and polynomial mutation are bounded.
- **Time:** from before the algorithm object is created to the return of `run`, which creates the initial population.
- **One thread:** `parallel` off; numpy's BLAS (OpenMP, OpenBLAS, MKL) set to 1 thread before import ([lines 49-51](../../../benchmarks/adapters/genoxide_python/bench.py#L49-L51)).
- **Seeds:** the algorithm's `seed`; a seed repeats a run exactly.
- **Solutions:** `result.best_genome`; for several objectives, `result.front_genomes` and `result.front_objectives`, the final non-dominated front without copies.
- **Separate tests:** 2026-09-25, the package 0.6.0 built from 03e237b, seeds 0 to 4, the scenario's budget, 60 s cap. The counts equalled `result.evaluations`, and `outside` was 0, in every run.

**Settings from the literature**, for settings without a default ([lines 226-243](../../../benchmarks/adapters/genoxide_python/bench.py#L226-L243)):
- a GA's population of 100, binary tournament (`Tournament(2)`), SBX η 20 and polynomial mutation η 20 at 1 / n per gene: Deb, Pratap, Agarwal and Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation 6(2), 2002;
- bit-flip mutation at 1 / n per gene: Mühlenbein, "How genetic algorithms really work: mutation and hillclimbing", PPSN 1992;
- swap mutation (`SwapMutation()`, one swap) for permutations; for local search on reals, the GA's polynomial mutation as the neighbor.

A GA's crossover is the first python/README.md lists for the genome: `UniformCrossover()` (bits), `OrderCrossover()` (permutations), `SimulatedBinaryCrossover(eta)` (reals). The rates (crossover 0.9, mutation 1) and the scheme (generational, elitism 1) are the defaults.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched:** the [README](../../../benchmarks/README.md)'s matched GA with the package's components: population 300, `Tournament(3)` (with replacement), `PointCrossover(2)` on each consecutive pair at 0.5, each child mutated at 0.2 by `BitFlip(rate=1 / n)`, `Generational(elitism=0)` ([lines 248-263](../../../benchmarks/adapters/genoxide_python/bench.py#L248-L263)). Difference from eaSimple: a child identical to a parent isn't evaluated, also one crossed or mutated back to it.
- **Idiomatic (OneMax 100):** the GA, the method of both OneMax examples: population 100, `Tournament(2)`, `UniformCrossover()`, `BitFlip(rate=1 / n)`, the default rates and scheme, a function per genome, `bits.sum()` ([lines 264-275](../../../benchmarks/adapters/genoxide_python/bench.py#L264-L275)).

**Keeping going:** to the target or the budget.

**Left out:** the settings of the two OneMax examples, python/README.md's first example (tournament 3, `BitFlip(rate=0.01)`) and [onemax.py](../../../python/examples/onemax.py) (DEAP's first tutorial): example settings (rule 6.6).

**Separate tests:**

OneMax 100, matched (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 4,789 | 100 | 100 | 100 | 0 |

OneMax 1000, matched (budget 2,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 53,597 | 1,000 | 1,000 | 1,000 | 0 |

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 3,418 | 100 | 100 | 100 | 0 |

## Permutation: N-Queens 32 and 64

**Methods:** no stated preference for permutations, so the methods the docs list first for every genome, `Ga` and `LocalSearch` ([lines 278-300](../../../benchmarks/adapters/genoxide_python/bench.py#L278-L300)):
- **`ga`:** population 100, `Tournament(2)`, `OrderCrossover()`, `SwapMutation()`, the default rates and scheme.
- **`local_search`:** hill climbing with its defaults (1 neighbor per step, `NotWorse()` acceptance), `SwapMutation()` neighbors.

**Keeping going:** no convergence criterion.

**Left out:** tabu search (`acceptance=Tabu(tenure)`), the method of [n_queens.py](../../../python/examples/n_queens.py): `Tabu(tenure)` has no default or standard value, and the example's tenure 20 and 32 neighbors are example settings (rule 6.6).

**Separate tests:**

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 2 | 237,595 | 1 | 0 | 2 | 0 |
| local_search | 5 | 5 | 1,145 | 0 | 0 | 0 | 0 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 2 | 1 | 3 | 1 |
| local_search | 5 | 5 | 2,385 | 0 | 0 | 0 | 0 |

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:** the package's Rastrigin example, [rastrigin.py](../../../python/examples/rastrigin.py), runs CMA-ES and DE ([lines 303-333](../../../benchmarks/adapters/genoxide_python/bench.py#L303-L333)):
- **`cma_es`:** `gx.Cmaes(genome, restarts="ipop")`, with the defaults (4 + ⌊3 ln n⌋ samples, an initial step of 0.3 of each range); the example, python/README.md's first example and the `Cmaes` docstring ("for multimodal functions") use IPOP.
- **`de`:** `gx.De(genome)` with its defaults (python/README.md, the `De` docstring): SHADE's published settings (Tanabe and Fukunaga, IEEE CEC 2013: current-to-pbest/1 with an archive, SHADE's adaptation of F and CR, 100 individuals), and genoxide's own restarts when the population converges or stalls (#113).

**Keeping going:** CMA-ES restarts with a doubled population, DE when its population converges or stalls.

**Left out:**
- L-SHADE (`gx.De(genome, l_shade=budget)`), which the example runs: a setting, not the default (rule 6.6).
- A GA, PSO and local search: the example for this problem type runs CMA-ES and DE.

**Separate tests:**

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 124,444 | 0.004704 | 0.003576 | 0.007748 | 0 |
| de | 5 | 5 | 41,016 | 0.00921 | 0.00476 | 0.009991 | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 551,188 | 0.009033 | 0.008715 | 0.009513 | 0 |
| de | 5 | 5 | 101,455 | 0.00905 | 0.008015 | 0.00999 | 0 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 3,203 | 0.00943 | 0.008762 | 0.009948 | 0 |
| de | 5 | 5 | 19,144 | 0.00973 | 0.008253 | 0.009884 | 0 |

## Continuous, unimodal: Rosenbrock 10

**Methods:** no example or stated preference for a unimodal function, so the first three methods python/README.md lists for real genomes, with their defaults ([lines 303-326](../../../benchmarks/adapters/genoxide_python/bench.py#L303-L326)):
- **`ga`:** population 100, `Tournament(2)`, `SimulatedBinaryCrossover(20)` at the default 0.9, `PolynomialMutation(20, rate=1 / n)`, the default scheme.
- **`local_search`:** hill climbing with its defaults, `PolynomialMutation(20, rate=1 / n)` as the neighbor.
- **`de`:** `gx.De(genome)` (above).

**Keeping going:** DE restarts by itself; the GA and local search run to the budget.

**Left out:** `Cmaes` and `Pso`, listed after these three. PSO's population size has no default; the docstring's "e.g. 40" is an example setting.

**Separate tests:**

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 3.346 | 0.5505 | 6.311 | 0 |
| local_search | 5 | 1 | 29,694 | 3.204 | 0.008596 | 4.718 | 0 |
| de | 5 | 5 | 38,077 | 0.008082 | 0.004516 | 0.008217 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

The matched settings of the [README](../../../benchmarks/README.md#scenarios), with the package's own operators.

**Methods** ([lines 336-370](../../../benchmarks/adapters/genoxide_python/bench.py#L336-L370)), with `batch=True` functions returning a row of objectives per genome, as [zdt1.py](../../../python/examples/zdt1.py):
- **`nsga2`, `spea2`, `sms_emoa`:** 100 individuals (92 with 3 objectives), `SimulatedBinaryCrossover(15)` at the default 0.9, `PolynomialMutation(20, rate=1 / n)`. SMS-EMOA breeds one child per generation (`offspring=1`).
- **`nsga3`:** `das_dennis(2, 99)` (100 directions, population 100) or `das_dennis(3, 12)` (91, population 92), SBX η 30 at the default 1, the same mutation.
- **`moead`:** `das_dennis(2, 99)` or `das_dennis(3, 12)` weights, its defaults of 20 neighbors, neighborhood parents at 0.9 and at most 2 replacements; `Tchebycheff()` with 2 objectives, `Pbi(5.0)` with 3; SBX η 20 at the default 1, the same mutation.
- **No duplicate elimination:** `eliminate_duplicates=False` for NSGA-II, NSGA-III, SPEA2 and SMS-EMOA, which drop duplicates by default (python/README.md); MOEA/D has none.

**Keeping going:** no convergence criterion; every run uses its budget.

**Separate tests** (hypervolume as `run.py` computes it):

ZDT1 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | Capped |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.8693 | 0.8697 | 0.8691 | 25,024 | 99 | 0 |
| nsga3 | 5 | 0.8701 | 0.8704 | 0.8699 | 25,022 | 100 | 0 |
| spea2 | 5 | 0.8703 | 0.8707 | 0.8700 | 25,083 | 100 | 0 |
| sms_emoa | 5 | 0.8719 | 0.8720 | 0.8719 | 25,000 | 100 | 0 |
| moead | 5 | 0.8687 | 0.8697 | 0.8684 | 25,006 | 100 | 0 |

ZDT2 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | Capped |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.5363 | 0.5365 | 0.5360 | 25,041 | 100 | 0 |
| nsga3 | 5 | 0.5366 | 0.5370 | 0.5365 | 25,016 | 100 | 0 |
| spea2 | 5 | 0.5368 | 0.5373 | 0.5363 | 25,041 | 100 | 0 |
| sms_emoa | 5 | 0.5386 | 0.5387 | 0.5386 | 25,000 | 100 | 0 |
| moead | 5 | 0.5352 | 0.5366 | 0.5351 | 25,070 | 98 | 0 |

ZDT3 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | Capped |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 1.3269 | 1.3273 | 1.2447 | 25,059 | 100 | 0 |
| nsga3 | 5 | 1.3255 | 1.3259 | 1.3241 | 25,015 | 100 | 0 |
| spea2 | 5 | 1.3273 | 1.3277 | 1.3270 | 25,063 | 100 | 0 |
| sms_emoa | 5 | 1.3244 | 1.3293 | 1.2458 | 25,000 | 100 | 0 |
| moead | 5 | 1.3227 | 1.3236 | 1.3213 | 25,049 | 76 | 0 |

1 NSGA-II run and 2 SMS-EMOA runs end without the front's last segment (f1 above 0.65), at a hypervolume of about 1.245.

DTLZ2, 3 objectives (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | Capped |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.6972 | 0.7021 | 0.6929 | 25,062 | 92 | 0 |
| nsga3 | 5 | 0.7443 | 0.7444 | 0.7441 | 25,023 | 92 | 0 |
| spea2 | 5 | 0.7292 | 0.7316 | 0.7282 | 25,029 | 92 | 0 |
| sms_emoa | 5 | 0.7556 | 0.7557 | 0.7554 | 25,000 | 92 | 0 |
| moead | 5 | 0.7441 | 0.7441 | 0.7440 | 25,014 | 91 | 0 |

DTLZ1, 3 objectives (budget 40,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | Capped |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.1338 | 0.1369 | 0.1307 | 40,036 | 91 | 0 |
| nsga3 | 5 | 0.1398 | 0.1400 | 0.1390 | 40,038 | 92 | 0 |
| spea2 | 5 | 0.1392 | 0.1398 | 0.1391 | 40,032 | 92 | 0 |
| sms_emoa | 5 | 0.1401 | 0.1402 | 0.1401 | 40,000 | 92 | 0 |
| moead | 5 | 0.1397 | 0.1398 | 0.1393 | 40,039 | 91 | 0 |

## Can't run

Nothing. The package has no multi-objective algorithm beyond the five of rule 6.1.

## Bugs found

None.
