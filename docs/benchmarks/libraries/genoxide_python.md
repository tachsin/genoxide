# genoxide for Python (Rust via Python, 0.6.0)

genoxide's Python package, from this repository's [python/](../../../python/) folder: genoxide's algorithms, in Rust, calling fitness functions written in Python and numpy. Its documentation is [python/README.md](../../../python/README.md), the [examples](../../../python/examples/) and the docstrings of [python/genoxide/\_\_init\_\_.py](../../../python/genoxide/__init__.py); the Rust library's docs ([genoxide.md](genoxide.md)) describe the same algorithms.

Adapter: [benchmarks/adapters/genoxide_python/](../../../benchmarks/adapters/genoxide_python/).
Know a better way to solve one of these problems with genoxide's Python package? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs the package

It runs the same methods with the same settings as the [Rust adapter](genoxide.md), so the two measure the cost of the Python fitness functions and of the package's calls into them, not different searches. The methods, where genoxide's docs recommend them, what was left out and why are on the [genoxide page](genoxide.md); this page gives what differs, and the separate tests.

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), in numpy, as the package's docs write them ([bench.py, lines 54-200](../../../benchmarks/adapters/genoxide_python/bench.py#L54-L200)):
  - `batch=True`, with a generation per call as a 2-D array and vectorized numpy: what [python/README.md](../../../python/README.md#fitness-functions) recommends for vectorized numpy ("one call per generation"), as [python/examples/rastrigin.py](../../../python/examples/rastrigin.py) and [zdt1.py](../../../python/examples/zdt1.py) do. The GA, DE, CMA-ES, particle swarm and every multi-objective algorithm use it, and so does the idiomatic OneMax GA.
  - A function per genome for local search and tabu search, which evaluate 4 or 32 neighbors per step, as [python/examples/n_queens.py](../../../python/examples/n_queens.py) does.
  - A function per genome in the matched OneMax runs, like DEAP's: a Python call per genome that counts the ones with numpy's `sum`, as README.md's first example and [python/examples/onemax.py](../../../python/examples/onemax.py) do. The matched scenarios compare framework costs, and this is the cost of one Python call per evaluation.
  - A self-check (`bench.py --self-check`) compares every function with problems.py at random points, and the per-genome N-Queens function with the batch one.
- **Evaluations:** each fitness function counts the genomes it evaluates itself: one per call, or the rows of a batch ([`EVALUATIONS`](../../../benchmarks/adapters/genoxide_python/bench.py#L58)). The adapter reports that count, compares it with the package's `result.evaluations` and prints any difference to stderr ([lines 342-346](../../../benchmarks/adapters/genoxide_python/bench.py#L342-L346)). In every separate test below, on every scenario, seed and solver, the two counts were equal.
- **Stop:** `run(..., target=..., evaluations=..., time=...)`, checked after every generation, as in Rust.
- **Time:** from before the algorithm object is created to the end of `run`, whose first step creates the random initial population.
- **One thread:** `parallel` is off, and the adapter sets numpy's BLAS (OpenMP, OpenBLAS, MKL) to 1 thread before importing numpy ([lines 41-43](../../../benchmarks/adapters/genoxide_python/bench.py#L41-L43)). Without that, OpenBLAS's threads took CPU time in the DTLZ2 runs: 1.55 times the wall time in `run.py check`. The v0.6.0 results were measured before the adapter set it.
- **Seeds:** each seed goes to the algorithm's `seed`; a seed repeats a run exactly.
- **Solutions:** each run prints `result.best_genome` (bits as 0 and 1), and a multi-objective run `result.front_genomes` with `result.front_objectives`: the final non-dominated front of the population. Unlike the Rust `outcome.front()`, the package leaves out copies of the same solution (in MOEA/D's population); the hypervolume is the same.

## Differences from the Rust adapter's results

With the same seed and the same fitness values, the package runs the same search as genoxide in Rust, evaluation for evaluation: the separate tests below match the Rust ones exactly for the GA, local search, tabu search, the particle swarm, and NSGA-II, NSGA-III, SPEA2 and SMS-EMOA. numpy adds up a row in another order than Rust's iterator, so some fitness values differ in the last bit; then DE, CMA-ES and MOEA/D can take other paths, e.g. a median of 4,500 evaluations for DE on Rastrigin 10 against 4,400 in Rust, and 76,980 against 80,200 for CMA-ES.

## Binary: OneMax 100 and 1000

**Methods:** the matched GA ([lines 211-225](../../../benchmarks/adapters/genoxide_python/bench.py#L211-L225)) and the idiomatic GA of every OneMax example in the docs: population 100, tournament 3, uniform crossover, bit-flip at 1 / n per gene, the default rates and scheme ([lines 226-236](../../../benchmarks/adapters/genoxide_python/bench.py#L226-L236)); see [genoxide.md](genoxide.md#binary-onemax-100-and-1000).

**Keeping going:** a GA runs to the budget by itself.

**Left out:** as in Rust. A per-genome function in the idiomatic run: README.md's first example writes one (`lambda bits: bits.sum()`), but its section on fitness functions recommends `batch=True` for vectorized numpy.

**Separate tests** (2026-09-25, the package 0.6.0 built from ad4eaf5, seeds 0 to 4, the scenario's budget, 60 s cap):

OneMax 100, matched (budget 200,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 4,823 | 100 | 100 | 100 | 0 |

OneMax 1000, matched (budget 2,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 53,615 | 1,000 | 1,000 | 1,000 | 0 |

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 2,182 | 100 | 100 | 100 | 0 |

## Permutation: N-Queens 32 and 64

**Methods** ([lines 239-271](../../../benchmarks/adapters/genoxide_python/bench.py#L239-L271)): the (20 + 20) GA of examples/n_queens.rs, hill climbing with 4 swap neighbors per step (the rustdoc's N-Queens example), and tabu search with 32 swap neighbors and tenure 20 ([python/examples/n_queens.py](../../../python/examples/n_queens.py)); see [genoxide.md](genoxide.md#permutation-n-queens-32-and-64).

**Keeping going:** all three run to the budget by themselves.

**Left out:** as in Rust.

**Separate tests** (2026-09-25, seeds 0 to 4):

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 2,560 | 0 | 0 | 0 | 0 |
| local_search | 5 | 5 | 925 | 0 | 0 | 0 | 0 |
| tabu_search | 5 | 5 | 4,097 | 0 | 0 | 0 | 0 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 5,978 | 0 | 0 | 0 | 0 |
| local_search | 5 | 5 | 4,069 | 0 | 0 | 0 | 0 |
| tabu_search | 5 | 5 | 165,377 | 0 | 0 | 0 | 0 |

The tabu search of the package's own N-Queens example needs 40 times the evaluations of hill climbing on N-Queens 64 (see [genoxide.md](genoxide.md#permutation-n-queens-32-and-64)).

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** ([lines 274-300](../../../benchmarks/adapters/genoxide_python/bench.py#L274-L300)): the GA of examples/rastrigin.rs, DE with its defaults, and CMA-ES with its defaults and IPOP restarts, which python/README.md's first example and [python/examples/rastrigin.py](../../../python/examples/rastrigin.py) run on Rastrigin; see [genoxide.md](genoxide.md#continuous-multimodal-rastrigin-10-and-30-ackley-30), which also says where the DE defaults come from.

**Keeping going:** the GA runs to the budget by itself; DE and CMA-ES (IPOP) restart by themselves.

**Left out:** as in Rust. L-SHADE (`gx.De(genome, l_shade=budget)`), which python/examples/rastrigin.py runs on Rastrigin 30, aims at the best value at the end of a budget, not the fewest evaluations to a target; the island model and the evolution strategy aren't in the package.

**Separate tests** (2026-09-25, seeds 0 to 4):

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 23,530 | 0.008099 | 0.003897 | 0.009707 | 0 |
| de | 5 | 5 | 4,500 | 0.008197 | 0.003399 | 0.009838 | 0 |
| cma_es | 5 | 5 | 76,980 | 0.008206 | 0.005702 | 0.009597 | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 101,527 | 0.009632 | 0.009398 | 0.00982 | 0 |
| de | 5 | 5 | 27,960 | 0.00931 | 0.005445 | 0.00955 | 0 |
| cma_es | 5 | 5 | 530,796 | 0.008887 | 0.007419 | 0.009968 | 0 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 245,513 | 0.009644 | 0.008412 | 0.009885 | 0 |
| de | 5 | 5 | 8,040 | 0.009627 | 0.009334 | 0.00982 | 0 |
| cma_es | 5 | 5 | 3,164 | 0.008987 | 0.008779 | 0.009896 | 0 |

## Continuous, unimodal: Rosenbrock 10

**Methods** ([lines 279-288](../../../benchmarks/adapters/genoxide_python/bench.py#L279-L288)): CMA-ES with IPOP restarts, DE with its defaults, and the particle swarm of AGENTS.md's Rosenbrock template (`gx.Pso(genome, population_size=40)`, the global topology); see [genoxide.md](genoxide.md#continuous-unimodal-rosenbrock-10), which also gives the result of the GA left out.

**Keeping going:** CMA-ES and DE restart by themselves; the particle swarm runs to the budget by itself.

**Left out:** as in Rust.

**Separate tests** (2026-09-25, seeds 0 to 4):

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 5,840 | 0.009198 | 0.008564 | 0.009741 | 0 |
| de | 5 | 5 | 7,400 | 0.009191 | 0.007153 | 0.009787 | 0 |
| pso | 5 | 5 | 98,160 | 0.009987 | 0.009948 | 0.009991 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Methods** ([lines 303-334](../../../benchmarks/adapters/genoxide_python/bench.py#L303-L334)): NSGA-II, SPEA2, SMS-EMOA and MOEA/D, and NSGA-III with 3 objectives, with the matched settings of the [README](../../../benchmarks/README.md#scenarios), as in Rust ([genoxide.md](genoxide.md#multi-objective-zdt1-zdt2-zdt3-dtlz2-dtlz1)); the fitness functions return a row of objective values per genome (`batch=True`), as [python/examples/zdt1.py](../../../python/examples/zdt1.py).

**Keeping going:** every run uses its whole budget.

**Separate tests** (2026-09-25, seeds 0 to 4; the hypervolume as `run.py` computes it):

ZDT1 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | At the cap |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.8693 | 0.8698 | 0.8690 | 25,000 | 100 | 0 |
| spea2 | 5 | 0.8702 | 0.8706 | 0.8694 | 25,000 | 100 | 0 |
| sms_emoa | 5 | 0.8716 | 0.8717 | 0.8714 | 25,000 | 100 | 0 |
| moead | 5 | 0.8687 | 0.8697 | 0.8684 | 25,006 | 100 | 0 |

ZDT2 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | At the cap |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.5360 | 0.5366 | 0.5355 | 25,000 | 100 | 0 |
| spea2 | 5 | 0.5369 | 0.5371 | 0.5361 | 25,000 | 100 | 0 |
| sms_emoa | 5 | 0.5380 | 0.5382 | 0.5379 | 25,000 | 100 | 0 |
| moead | 5 | 0.5352 | 0.5366 | 0.5351 | 25,070 | 98 | 0 |

ZDT3 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | At the cap |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 1.3274 | 1.3278 | 1.3271 | 25,000 | 100 | 0 |
| spea2 | 5 | 1.3273 | 1.3277 | 1.3269 | 25,000 | 100 | 0 |
| sms_emoa | 5 | 1.3288 | 1.3289 | 1.3287 | 25,000 | 100 | 0 |
| moead | 5 | 1.3227 | 1.3236 | 1.3213 | 25,049 | 76 | 0 |

DTLZ2, 3 objectives (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | At the cap |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.6969 | 0.7037 | 0.6912 | 25,024 | 92 | 0 |
| nsga3 | 5 | 0.7442 | 0.7442 | 0.7434 | 25,024 | 92 | 0 |
| spea2 | 5 | 0.7274 | 0.7311 | 0.7230 | 25,024 | 92 | 0 |
| sms_emoa | 5 | 0.7544 | 0.7545 | 0.7543 | 25,024 | 92 | 0 |
| moead | 5 | 0.7441 | 0.7441 | 0.7440 | 25,014 | 91 | 0 |

DTLZ1, 3 objectives (budget 40,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | At the cap |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 1.2987 | 1.3009 | 1.2956 | 40,020 | 92 | 0 |
| nsga3 | 5 | 1.3045 | 1.3046 | 1.3035 | 40,020 | 92 | 0 |
| spea2 | 5 | 1.3032 | 1.3034 | 1.2997 | 40,020 | 92 | 0 |
| sms_emoa | 5 | 1.3046 | 1.3046 | 1.3045 | 40,020 | 92 | 0 |
| moead | 5 | 1.3043 | 1.3044 | 1.3039 | 40,039 | 91 | 0 |

## Can't run

The package runs every scenario.

## Bugs found

None in the package in this review. Its `result.evaluations` matched the adapter's own count in every run.
