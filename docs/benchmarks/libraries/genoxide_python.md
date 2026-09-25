# genoxide for Python (Rust via Python, 0.6.0)

genoxide's Python package, from this repository's [python/](../../../python/) folder: genoxide's algorithms, in Rust, calling fitness functions written in Python and numpy. Its documentation is [python/README.md](../../../python/README.md), its [examples](../../../python/examples/) and the docstrings of [python/genoxide/\_\_init\_\_.py](../../../python/genoxide/__init__.py).

Adapter: [benchmarks/adapters/genoxide_python/](../../../benchmarks/adapters/genoxide_python/).
Know a better way to solve one of these problems with genoxide's Python package? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs the package

The methods and settings come from the Python package's own docs, not from the Rust library's ([genoxide.md](genoxide.md)): rule 6.2 takes each library entry's docs. Where the docs offer several, the order is a preference they state, then their example for the problem type, then the defaults. The separate tests below are shown, and decided nothing.

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), in numpy, written as the package's examples write them ([bench.py, lines 61-207](../../../benchmarks/adapters/genoxide_python/bench.py#L61-L207)):
  - a function per genome for OneMax, as [python/README.md](../../../python/README.md)'s first example and [python/examples/onemax.py](../../../python/examples/onemax.py) (`bits.sum()`), in the matched and the idiomatic runs;
  - a function per genome for N-Queens, as [python/examples/n_queens.py](../../../python/examples/n_queens.py);
  - `batch=True`, a generation per call as a 2-D array with vectorized numpy, for the real-valued and multi-objective problems, as [python/examples/rastrigin.py](../../../python/examples/rastrigin.py) and [zdt1.py](../../../python/examples/zdt1.py), and what python/README.md recommends for vectorized numpy ("one call per generation");
  - a self-check (`bench.py --self-check`) compares every function with problems.py at random points.
- **Evaluations:** each fitness function counts the genomes it evaluates itself: one per call, or the rows of a batch ([`EVALUATIONS`](../../../benchmarks/adapters/genoxide_python/bench.py#L65), [`count`](../../../benchmarks/adapters/genoxide_python/bench.py#L69-L73)). The adapter reports that count, compares it with the package's `result.evaluations` and prints any difference to stderr ([`solve`, lines 340-389](../../../benchmarks/adapters/genoxide_python/bench.py#L340-L389)). In every separate test below, on every scenario, seed and solver, the two counts were equal.
- **Stop:** `run(..., target=..., evaluations=..., time=...)`, checked after every generation. CMA-ES with IPOP restarts also prints `last_generation`, from `on_generation`, for the budget check.
- **Restarts on convergence (rule 2.2):** DE and CMA-ES (IPOP) restart by themselves when a run converges. L-SHADE has no restarts and no convergence criterion: its population shrinks over the budget. The GA, tabu search and PSO have no convergence criterion either; the package ends a run by itself only as "stalled" (10,000 generations without a genome to evaluate), after which the adapter makes the algorithm again with the seed `seed * 1000 + restart` and runs it on the rest of the budget and time, keeping the best and counting every evaluation. No test run stalled.
- **Bounds (rule 2.4):** the package's operators are genoxide's, with its bound handling: CMA-ES draws a sample outside the bounds again, up to 100 times, and then clips it; DE and L-SHADE set a trial gene outside the bounds halfway between the parent's gene and the bound; a particle stops at the bound; SBX and polynomial mutation are bounded. The real-valued and multi-objective fitness functions count the evaluated genomes with a gene outside the bounds, as the package passed them, and each run prints `outside`: 0 in every run of the separate tests.
- **Time:** from before the algorithm object is created to the end of `run`, whose first step creates the random initial population.
- **One thread:** `parallel` is off, and the adapter sets numpy's BLAS (OpenMP, OpenBLAS, MKL) to 1 thread before importing numpy ([lines 47-49](../../../benchmarks/adapters/genoxide_python/bench.py#L47-L49)). Without that, OpenBLAS's threads took CPU time in the DTLZ2 runs: 1.55 times the wall time in `run.py check`. The v0.6.0 results were measured before the adapter set it.
- **Seeds:** each seed goes to the algorithm's `seed`; a seed repeats a run exactly.
- **Solutions:** each run prints `result.best_genome` (bits as 0 and 1), and a multi-objective run `result.front_genomes` with `result.front_objectives`: the final non-dominated front of the population, without copies of the same solution.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched (OneMax 100 and 1000):** the matched GA of the [README](../../../benchmarks/README.md): population 300, tournament 3, two-point crossover with probability 0.5, bit-flip at 1 / n per gene on 20% of the children, no elitism ([lines 219-233](../../../benchmarks/adapters/genoxide_python/bench.py#L219-L233)).
- **Idiomatic (OneMax 100):** [python/README.md](../../../python/README.md)'s first example, which is OneMax with 100 bits: `gx.Ga(gx.Binary(100), population_size=100, select=gx.Tournament(3), crossover=gx.UniformCrossover(), mutation=gx.BitFlip(rate=0.01))`, with the default rates and scheme (generational, elitism 1), and a function per genome, `bits.sum()` ([lines 234-247](../../../benchmarks/adapters/genoxide_python/bench.py#L234-L247)).

**Keeping going:** a GA has no convergence criterion and runs to the budget.

**Left out:** the second OneMax example, [python/examples/onemax.py](../../../python/examples/onemax.py), which copies DEAP's first tutorial (population 300, two-point crossover at 0.5, `BitFlip(rate=0.05)` on 20% of the children). The docs state no preference between the two; the README's is the package's own OneMax, and comes first in the docs.

**Separate tests** (2026-09-25, the package 0.6.0 built from 4fff05e, seeds 0 to 4, the scenario's budget, 60 s cap):

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

**Methods:**
- **`tabu_search`:** the package's example for N-Queens, [python/examples/n_queens.py](../../../python/examples/n_queens.py) (lines 17-24): tabu search with swap neighbors, 32 per step, tenure 20, and a function per genome ([lines 250-263](../../../benchmarks/adapters/genoxide_python/bench.py#L250-L263)). The example is for N-Queens 64; N-Queens 32 runs with the same settings.

**Keeping going:** tabu search has no convergence criterion and runs to the budget.

**Left out:** the GA and hill climbing, which the Rust adapter runs from the Rust docs: the Python docs present no other method for permutations.

**Separate tests** (2026-09-25, seeds 0 to 4):

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| tabu_search | 5 | 5 | 4,097 | 0 | 0 | 0 | 0 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| tabu_search | 5 | 5 | 165,377 | 0 | 0 | 0 | 0 |

The example's tabu search needs up to 919,713 of the 1,000,000 evaluations on N-Queens 64 (seeds 0 to 4); every run reached the target. Its tabu list holds whole solutions, and it always moves to the best of the 32 neighbors that aren't tabu, even a worse one.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:** the package's example for Rastrigin, [python/examples/rastrigin.py](../../../python/examples/rastrigin.py), runs two methods, as it sets them ([lines 266-289](../../../benchmarks/adapters/genoxide_python/bench.py#L266-L289)):
- **`cma_es`:** `gx.Cmaes(genome, restarts="ipop")`, with the defaults (4 + ⌊3 ln n⌋ samples, an initial step of 0.3 of each range), as in the example and in python/README.md's first example. The `Cmaes` docstring gives the restarts "for multimodal functions".
- **`l_shade`:** `gx.De(genome, l_shade=budget)`, L-SHADE for the scenario's evaluation budget, and the run stops at that budget, as the `De` docstring asks ("Stop the run at the same number of evaluations"): current-to-pbest/1, F and CR adapted during the run, and a population that shrinks linearly from 18 times the number of genes to 4.

**Keeping going:** CMA-ES restarts with a doubled population when a run converges. L-SHADE has no convergence criterion: its population shrinks until the budget.

**Left out:**
- DE with its defaults (`gx.De(genome)`), which the Rust adapter runs: the package's example for these problems runs L-SHADE instead.
- A GA, PSO and local search: the Python docs present none of them for continuous multimodal functions. The Pso docstring offers a ring topology "for multimodal functions", but the docs' example for this problem type decides first.

**Separate tests** (2026-09-25, seeds 0 to 4):

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 76,980 | 0.008206 | 0.005702 | 0.009597 | 0 |
| l_shade | 5 | 5 | 48,003 | 0.007674 | 0.007149 | 0.008394 | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 530,796 | 0.008887 | 0.007419 | 0.009968 | 0 |
| l_shade | 5 | 5 | 405,783 | 0.009588 | 0.00875 | 0.009988 | 0 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 3,164 | 0.008987 | 0.008779 | 0.009896 | 0 |
| l_shade | 5 | 5 | 78,847 | 0.009758 | 0.008207 | 0.00977 | 0 |

L-SHADE spreads its search over the whole budget by design, so it reaches the targets late, and with little spread between seeds.

## Continuous, unimodal: Rosenbrock 10

**Methods:** the Python docs have no example for a unimodal function and state no preference among their real-valued algorithms, so the algorithms that run with their documented defaults ([lines 266-283](../../../benchmarks/adapters/genoxide_python/bench.py#L266-L283)):
- **`cma_es`:** `gx.Cmaes(genome, restarts="ipop")`: the defaults, with IPOP for rule 2.2, which restarts a converged method with the library's own restarts.
- **`de`:** `gx.De(genome)`, the defaults of python/README.md's table and the `De` docstring: SHADE's published settings (current-to-pbest/1 with an archive, SHADE's adaptation of F and CR, 100 individuals), and restarts when the population converges or stalls ([genoxide.md](genoxide.md#continuous-multimodal-rastrigin-10-and-30-ackley-30)).
- **`pso`:** `gx.Pso(genome, population_size=40)`: the population size is required, and 40 is the `Pso` docstring's ("e.g. 40"); the default global topology.

**Keeping going:** CMA-ES and DE restart by themselves; PSO has no convergence criterion and runs to the budget.

**Left out:** L-SHADE, which the docs' example runs on Rastrigin, a multimodal function; and a GA, which has no defaults for its operators.

**Separate tests** (2026-09-25, seeds 0 to 4):

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 5,840 | 0.009198 | 0.008564 | 0.009741 | 0 |
| de | 5 | 5 | 38,100 | 0.008082 | 0.004516 | 0.008217 | 0 |
| pso | 5 | 5 | 98,160 | 0.009987 | 0.009948 | 0.009991 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

These scenarios are matched: every library runs the settings of the [README](../../../benchmarks/README.md#scenarios), and rule 6.1 runs only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA.

**Methods** ([lines 292-325](../../../benchmarks/adapters/genoxide_python/bench.py#L292-L325)), with the fitness functions returning a row of objective values per genome (`batch=True`), as [python/examples/zdt1.py](../../../python/examples/zdt1.py):
- **`nsga2`, `spea2`, `sms_emoa`:** 100 individuals (92 with 3 objectives), `SimulatedBinaryCrossover(15)` at their default rate of 0.9, `PolynomialMutation(20, rate=1 / n)`. SMS-EMOA breeds as many children per generation as its population (its default `offspring`).
- **`nsga3`:** with 3 objectives only (python/README.md presents it for fronts where NSGA-II's crowding distance "works poorly"): `das_dennis(3, 12)` reference directions (91), a population of 92, SBX with η 30 at its default rate of 1, and the same polynomial mutation.
- **`moead`:** `das_dennis(2, 99)` weights (100), or `das_dennis(3, 12)` (91), with its defaults of 20 neighbors, parents from the neighborhood with probability 0.9 and at most 2 replacements; `Tchebycheff()` with 2 objectives and `Pbi(5.0)` with 3 (python/README.md: PBI "spreads fronts of 3 or more objectives well"); SBX with η 20 at its default rate of 1, and the same polynomial mutation.
- NSGA-II, NSGA-III, SPEA2 and SMS-EMOA drop a child that equals a member of the population or an earlier child, and breed another (python/README.md).

**Keeping going:** none of the five has a convergence criterion; every run uses its whole budget. `outside` was 0 in every run.

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

The package runs every scenario. It has no multi-objective algorithm beyond the five that rule 6.1 runs.

## Bugs found

None in the package in this review. Its `result.evaluations` matched the adapter's own count in every run.
