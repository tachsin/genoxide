# genoxide for Python (Rust via Python, 0.6.0)

genoxide's Python package, from this repository's [python/](../../../python/) folder: genoxide's algorithms, in Rust, calling fitness functions written in Python and numpy. Its documentation is [python/README.md](../../../python/README.md), its [examples](../../../python/examples/) and the docstrings of [python/genoxide/\_\_init\_\_.py](../../../python/genoxide/__init__.py).

Adapter: [benchmarks/adapters/genoxide_python/](../../../benchmarks/adapters/genoxide_python/).
Know a better way to solve one of these problems with genoxide's Python package? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs the package

The methods come from the Python package's own docs, not from the Rust library's ([genoxide.md](genoxide.md)): rule 6.2 takes each library entry's docs, in its order: a preference they state, then their example for the problem type, then the first they list. genoxide's authors run this benchmark, so [rule 6.6](../rules.md#6-which-methods-run) applies: the methods run with their defaults, and a setting without a default takes a standard value from the literature, cited below. Settings from the examples aren't used: some were written on these problems. The separate tests below are shown, and decided nothing.

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), in numpy, written as the package's docs write them ([bench.py, lines 62-221](../../../benchmarks/adapters/genoxide_python/bench.py#L62-L221)):
  - a function per genome for OneMax and N-Queens, as [python/README.md](../../../python/README.md)'s first example and [python/examples/onemax.py](../../../python/examples/onemax.py) and [n_queens.py](../../../python/examples/n_queens.py);
  - `batch=True`, a generation per call as a 2-D array with vectorized numpy, for the real-valued and multi-objective problems, as [python/examples/rastrigin.py](../../../python/examples/rastrigin.py) and [zdt1.py](../../../python/examples/zdt1.py), and what python/README.md presents for vectorized numpy ("one call per generation"). A batch doesn't change the algorithm: the package asks for the same genomes either way (python/README.md: "the same seed repeats a run exactly, with a genome at a time, in batches or in parallel");
  - `values` evaluates a solution with the same functions as the runs, and a self-check (`bench.py --self-check`) compares every function with problems.py at random points and at the optimum.
- **Evaluations:** each fitness function counts the genomes it evaluates itself: one per call, or the rows of a batch ([`count`](../../../benchmarks/adapters/genoxide_python/bench.py#L75-L79)). The adapter reports that count, compares it with the package's `result.evaluations` and prints any difference to stderr ([`solve`, lines 382-449](../../../benchmarks/adapters/genoxide_python/bench.py#L382-L449)). In every separate test below, the two counts were equal.
- **First hit:** each single-objective fitness function records the first evaluation whose value reaches the target, and the clock at that moment; in a batch, the first row that reaches it (`first_hit`).
- **Stop:** `run(..., target=..., evaluations=...)`, checked after every generation, and the 60 s cap: `time=...` for CMA-ES, DE and local search, and for the GA an `on_generation` callback that returns False once the cap is reached (below). CMA-ES with IPOP restarts also prints `last_generation`, from `on_generation`, for the budget check.
- **Restarts on convergence (rule 2.2):** DE and CMA-ES (IPOP) restart by themselves when a run converges. Local search evaluates a new neighbor every step (the package redraws a neighbor that didn't change), so it can't stall. The GA has no convergence criterion; the package ends a run by itself only as "stalled" (10,000 generations without a genome to evaluate), and only when every stop condition needs new evaluations, which is why the GA's time cap is its callback rather than `time`. After a stall the adapter makes the algorithm again with the seed `(seed + 1) * 1,000,000 + restart` and runs it on the rest of the budget and time, keeping the best and counting every evaluation. No separate test stalled; a run forced to stall (a GA without mutation) restarted as it should.
- **Bounds (rule 2.4):** the package's operators are genoxide's, with its bound handling: CMA-ES draws a sample outside the bounds again, up to 100 times, and then clips it; DE sets a trial gene outside the bounds halfway between the parent's gene and the bound; SBX and polynomial mutation are bounded. The real-valued and multi-objective fitness functions count the evaluated genomes with a gene outside the bounds, and each run prints `outside`: 0 in every run of the separate tests.
- **Time:** the clock starts before the algorithm object is created and stops when `run` returns, whose first step creates the random initial population.
- **One thread:** `parallel` is off, and the adapter sets numpy's BLAS (OpenMP, OpenBLAS, MKL) to 1 thread before importing numpy ([lines 49-51](../../../benchmarks/adapters/genoxide_python/bench.py#L49-L51)).
- **Seeds:** each seed goes to the algorithm's `seed`; a seed repeats a run exactly.
- **Solutions:** each run prints `result.best_genome` (bits as 0 and 1), and a multi-objective run `result.front_genomes` with `result.front_objectives`: the final non-dominated front of the population, without copies of the same solution.

**Settings from the literature**, for settings without a default ([lines 223-240](../../../benchmarks/adapters/genoxide_python/bench.py#L223-L240)):
- a GA's population of 100, binary tournament selection (`Tournament(2)`), SBX with η 20 and polynomial mutation with η 20 at 1 / n per gene: Deb, Pratap, Agarwal and Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation 6(2), 2002;
- bit-flip mutation at 1 / n per gene: Mühlenbein, "How genetic algorithms really work: mutation and hillclimbing", PPSN 1992;
- swap mutation (`SwapMutation()`, one swap) for permutations; for local search on reals, the GA's polynomial mutation as the neighbor.

The crossover of a GA, required too, is the first one python/README.md lists for the genome (rule 6.2): `UniformCrossover()` for bits, `OrderCrossover()` for permutations, `SimulatedBinaryCrossover(eta)` for reals. The rates (crossover 0.9, mutation 1) and the scheme (generational, elitism 1) are the defaults.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched (OneMax 100 and 1000):** the matched GA of the [README](../../../benchmarks/README.md), with the package's own components: population 300, `Tournament(3)` (with replacement), `PointCrossover(2)` applied to each consecutive pair of parents with probability 0.5, each child mutated with probability 0.2 by `BitFlip(rate=1 / n)`, and `Generational(elitism=0)`: generational, no elitism ([lines 245-260](../../../benchmarks/adapters/genoxide_python/bench.py#L245-L260)). One difference from DEAP's eaSimple: the package doesn't evaluate any child identical to a parent, also one that was crossed or mutated back to a parent's genome; DEAP evaluates those again.
- **Idiomatic (OneMax 100):** the GA, the method of both OneMax examples, with the settings above: population 100, `Tournament(2)`, `UniformCrossover()`, `BitFlip(rate=1 / n)`, the default rates and scheme, and a function per genome, `bits.sum()` ([lines 261-272](../../../benchmarks/adapters/genoxide_python/bench.py#L261-L272)).

**Keeping going:** a GA has no convergence criterion and runs to the budget.

**Left out:** the settings of the two OneMax examples, python/README.md's first example (tournament 3, `BitFlip(rate=0.01)`) and [python/examples/onemax.py](../../../python/examples/onemax.py) (DEAP's first tutorial), which the adapter used before: example settings (rule 6.6).

**Separate tests** (2026-09-25, the package 0.6.0 built from 03e237b, seeds 0 to 4, the scenario's budget, 60 s cap):

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

**Methods:** the package's example for N-Queens, [python/examples/n_queens.py](../../../python/examples/n_queens.py), is tabu search (`LocalSearch` with `acceptance=Tabu(tenure)`), whose tenure has no default and no standard value in the literature: it's left out (below). The docs state no preference for permutations, so the methods they list first for every genome, `Ga` and `LocalSearch` ([lines 275-297](../../../benchmarks/adapters/genoxide_python/bench.py#L275-L297)):
- **`ga`:** population 100, `Tournament(2)`, `OrderCrossover()`, `SwapMutation()`, the default rates and scheme.
- **`local_search`:** hill climbing with its defaults: 1 neighbor per step and the acceptance `NotWorse()`, which moves to equal neighbors across plateaus; `SwapMutation()` neighbors.

**Keeping going:** neither has a convergence criterion: they run to the budget.

**Left out:** tabu search, the example's method: `Tabu(tenure)` has no default, and the example's tenure of 20 with 32 neighbors per step are example settings (rule 6.6).

**Separate tests** (2026-09-25, the package 0.6.0 built from 03e237b, seeds 0 to 4):

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

**Methods:** the package's example for Rastrigin, [python/examples/rastrigin.py](../../../python/examples/rastrigin.py), runs CMA-ES and DE ([lines 300-330](../../../benchmarks/adapters/genoxide_python/bench.py#L300-L330)):
- **`cma_es`:** `gx.Cmaes(genome, restarts="ipop")`, with the defaults (4 + ⌊3 ln n⌋ samples, an initial step of 0.3 of each range) and IPOP restarts, which the example, python/README.md's first example and the `Cmaes` docstring ("for multimodal functions") give it.
- **`de`:** `gx.De(genome)`, with its defaults (python/README.md's table, the `De` docstring): SHADE's published settings (Tanabe and Fukunaga, IEEE CEC 2013: current-to-pbest/1 with an archive, SHADE's adaptation of F and CR, 100 individuals), and restarts when the population converges or stalls: genoxide's own choice, from #113.

**Keeping going:** CMA-ES restarts with a doubled population when a run converges, and DE when its population converges or stalls.

**Left out:**
- L-SHADE (`gx.De(genome, l_shade=budget)`), which the example runs: a setting of `De`, not its default (rule 6.6).
- A GA, PSO and local search: the docs' example for this problem type runs CMA-ES and DE.

**Separate tests** (2026-09-25, the package 0.6.0 built from 03e237b, seeds 0 to 4):

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

**Methods:** the Python docs have no example for a unimodal function and state no preference for one, so the first three methods python/README.md lists for real genomes (rule 6.2): `Ga`, `LocalSearch` and `De`, with their defaults ([lines 300-323](../../../benchmarks/adapters/genoxide_python/bench.py#L300-L323)):
- **`ga`:** population 100, `Tournament(2)`, `SimulatedBinaryCrossover(20)` at the default rate of 0.9, `PolynomialMutation(20, rate=1 / n)`, the default scheme.
- **`local_search`:** hill climbing with its defaults (1 neighbor per step, `NotWorse()`), with `PolynomialMutation(20, rate=1 / n)` as the neighbor.
- **`de`:** `gx.De(genome)`, with its defaults (above).

**Keeping going:** DE restarts by itself; the GA and local search have no convergence criterion and run to the budget.

**Left out:** CMA-ES and PSO, which python/README.md lists after these three. PSO's population size is required and has no default; the `Pso` docstring's "e.g. 40", which the adapter used before, is an example setting.

**Separate tests** (2026-09-25, the package 0.6.0 built from 03e237b, seeds 0 to 4):

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 3.346 | 0.5505 | 6.311 | 0 |
| local_search | 5 | 1 | 29,694 | 3.204 | 0.008596 | 4.718 | 0 |
| de | 5 | 5 | 38,077 | 0.008082 | 0.004516 | 0.008217 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

These scenarios are matched: every library runs the settings of the [README](../../../benchmarks/README.md#scenarios), with its own operators, and rule 6.1 runs only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA.

**Methods** ([lines 333-367](../../../benchmarks/adapters/genoxide_python/bench.py#L333-L367)), with the fitness functions returning a row of objective values per genome (`batch=True`), as [python/examples/zdt1.py](../../../python/examples/zdt1.py):
- **`nsga2`, `spea2`, `sms_emoa`:** 100 individuals (92 with 3 objectives), `SimulatedBinaryCrossover(15)` at their default rate of 0.9, `PolynomialMutation(20, rate=1 / n)`. SMS-EMOA is steady-state: one child per generation (`offspring=1`; the default is a population's worth).
- **`nsga3`:** `das_dennis(2, 99)` reference directions (100) with a population of 100, or `das_dennis(3, 12)` (91) with a population of 92, SBX with η 30 at its default rate of 1, and the same polynomial mutation.
- **`moead`:** `das_dennis(2, 99)` weights (100), or `das_dennis(3, 12)` (91), with its defaults of 20 neighbors, parents from the neighborhood with probability 0.9 and at most 2 replacements; `Tchebycheff()` with 2 objectives and `Pbi(5.0)` with 3; SBX with η 20 at its default rate of 1, and the same polynomial mutation.
- **No duplicate elimination:** NSGA-II, NSGA-III, SPEA2 and SMS-EMOA drop a child that equals a member of the population or an earlier child by default (python/README.md); the matched scenarios turn it off (`eliminate_duplicates=False`). MOEA/D has none.

**Keeping going:** none of the five has a convergence criterion; every run uses its whole budget. `outside` was 0 in every run.

**Separate tests** (2026-09-25, the package 0.6.0 built from 03e237b, seeds 0 to 4; the hypervolume as `run.py` computes it):

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

The package runs every scenario. It has no multi-objective algorithm beyond the five that rule 6.1 runs.

## Bugs found

None in the package in this review. Its `result.evaluations` matched the adapter's own count in every run.

Observations for the library, not bugs:
- The GA with the settings above reaches N-Queens 32 in 2 of 5 runs and N-Queens 64 in none, where hill climbing reaches both in every run.
- On Rosenbrock 10, the GA reaches the target in no run and local search in 1 of 5; DE reaches it in every run.
- On ZDT3, 1 NSGA-II run and 2 SMS-EMOA runs of 5 end without the front's last segment (f1 above 0.65), with a hypervolume of about 1.245 instead of 1.33.
