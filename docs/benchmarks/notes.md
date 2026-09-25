# Notes on the libraries

What each library in the [benchmarks](../../benchmarks/README.md) doesn't run and why, the bugs found in the libraries, and how their settings differ. [results.md](results.md) has the numbers and a table of which library ran which scenario.

## What each library doesn't run

Every library runs every scenario its features allow. A library missing from a chart can't run that scenario:

| Library | Doesn't run | Why |
|---|---|---|
| genetic_algorithm | the multi-objective scenarios | it has no multi-objective algorithm |
| pygmo | N-Queens | it has no permutation genome |
| pycma | OneMax, N-Queens, the multi-objective scenarios | CMA-ES optimizes one objective of real numbers |
| SciPy | OneMax, N-Queens, the multi-objective scenarios | `differential_evolution` optimizes one objective of real numbers |
| Nevergrad | OneMax (matched), N-Queens | it has no GA with the matched operators, and no permutation genome |

The runs that failed, as opposed to not running, are in the charts: a cross for a target never reached, and a hatched bar for a hypervolume far below the others.

## Bugs found

A bug in a library's own operator isn't worked around: the benchmark runs what the library's users get, and its results show the bug. The exceptions are in the last column.

| Library | Bug | Effect | Worked around | Reported |
|---|---|---|---|---|
| radiate 1.3.1 | `SimulatedBinaryCrossover` centres the child on (a − b)/2 instead of (a + b)/2, and changes only one parent | lower multi-objective hypervolumes: with textbook operators, its NSGA-II reaches pymoo's (ZDT1: 0.868 against 0.870) | no | [pkalivas/radiate#28](https://github.com/pkalivas/radiate/issues/28) |
| radiate 1.3.1 | `PolynomialMutator` computes the new value from a bound instead of from the gene | as above | no | [pkalivas/radiate#28](https://github.com/pkalivas/radiate/issues/28) |
| Jenetics 9.1.0 | `SimulatedBinaryCrossover` centres the child on (a − b)/2 | with a corrected copy, the DTLZ2 hypervolume goes from 0.51 to 0.69 | no | [jenetics/jenetics#969](https://github.com/jenetics/jenetics/issues/969) |
| Evolutionary.jl 0.12.0 | `NSGA2` reorders the parents but not their objective values, ranks and crowding distances, so from the second generation it selects by other individuals' values | hypervolumes near 0 | no | [SciML/Evolutionary.jl#174](https://github.com/SciML/Evolutionary.jl/issues/174) |
| moors 0.2.11 | AGE-MOEA panics when its first front is degenerate: an assertion that the central point is positive, or a division of 0 by 0 | its ZDT2 and DTLZ1 runs fail, with a hypervolume of 0 | no | [andresliszt/moo-rs#301](https://github.com/andresliszt/moo-rs/issues/301) |
| openGA f9b15e7 | the single-objective survival's rank roulette can't pick a child, so children survive only through the elite slots | children are almost never kept | in the matched OneMax runs, every slot is elite: the best of parents and children survive | [Arash-codedev/openGA#30](https://github.com/Arash-codedev/openGA/issues/30) |
| PyGAD 3.7.0 | its `sbx` crossover always makes the child below the parents' midpoint, which pulls every gene towards the lower bound, and it can't cross a pair with probability 0.9 | a bias towards ZDT's optimum | yes: SBX is a PyGAD crossover function in the adapter | [ahmedfgad/GeneticAlgorithmPython#369](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/369) |
| jMetal 7.5 | CMA-ES throws `ArrayIndexOutOfBoundsException` in `tql2` when its covariance degenerates | the run ends there | the run counts with what it reached | |
| Nevergrad 1.0.12 | its metamodel crashes with NumPy 2.5 | no run at all | the adapter restores the old scalar conversion in that one module, which doesn't change the algorithm | |

## Settings and differences

Every adapter's source has its full settings, and where its idiomatic settings come from.

- **genoxide:**
  - A child identical to one of its parents inherits the parent's fitness, without an evaluation.
  - Its OneMax genomes are bit-packed.
- **genoxide (Python):** genoxide's Python package, whose algorithms are genoxide's Rust, calling Python fitness functions, with the Rust adapter's solvers and settings.
  - In the matched OneMax runs, the fitness function is pure Python, called with one genome at a time, as in DEAP. In the other runs, it's vectorized numpy, called with a generation at a time (`batch=True`), as the package's README recommends.
  - With the same seed and the same fitness values, it runs the same search as genoxide in Rust, evaluation for evaluation. numpy's `sum` adds in another order, so some fitness values differ in the last bit, and its CMA-ES, DE and MOEA/D runs take other paths with the same settings.
- **genetic_algorithm:** it selects survivors from parents and offspring by tournament, without replacement.
- **radiate:** its recommended settings don't reach the idiomatic OneMax, N-Queens and Rosenbrock targets.
- **moors:**
  - It evaluates the parents again every generation, so a budget buys it half the generations.
  - It has no polynomial mutation, so the adapter implements Deb's.
- **openGA:**
  - It selects parents by a rank roulette.
  - The idiomatic real-valued runs use its Rastrigin example's population of 10,000.
- **pygmo:**
  - pagmo's algorithms are C++, and they call the Python fitness function.
  - Its SGA mutates a bit by drawing it again, which flips it half the time, so the matched rate is doubled. Its SGA keeps the best of parents and children, which can't be turned off.
  - Its CMA-ES runs with `force_bounds`.
- **DEAP:** its CMA-ES starts Rastrigin from its example's (5, …, 5) with σ 5. For the other problems, it starts from a random point with σ a quarter of the range.
- **pymoo:**
  - The survival of its GA is elitist: the best of parents and offspring survive.
  - Its multi-objective algorithms eliminate duplicate children.
- **PyGAD:**
  - Its NSGA-II selects the parents before the survivors. So the adapter runs a population of 2N with N elites: the elites are the best N of the previous elites and their offspring, as in NSGA-II, but the parents' binary tournament draws from all 2N, and the initial population is 2N.
  - A child identical to an elite or a parent of the previous generation takes its fitness without an evaluation.
  - Its non-dominated sorting compares every pair of individuals in Python, three times per generation.
- **pycma:** IPOP-CMA-ES through `fmin2`, with 9 restarts, each with twice the population.
- **Nevergrad:**
  - NGOpt chooses from a portfolio of optimizers and costs about 10 ms per evaluation, so it reaches the 60-second limit after a few thousand evaluations.
  - It has no NSGA-II, SBX or polynomial mutation, so the multi-objective scenarios run the DE its docs recommend for several objectives, with its defaults: an idiomatic run in a matched scenario. Its front is the non-dominated set of every point it evaluated, not a final population.
- **SciPy:** `differential_evolution` runs with its defaults. These include the L-BFGS-B polish at the end, whose evaluations count. Its default `maxiter` and `tol` often stop it before the budget.
- **Jenetics:**
  - Its crossover probability is per individual, not per pair, so the matched runs use half the rate.
  - It has no bit-flip or polynomial mutation, so the adapter adds them.
- **jMetal:** its CMA-ES starts at a random point in [0, 1)ⁿ, whatever the bounds, which is next to the shifted optimum. The adapter starts it at a random point within the bounds, like the other libraries.
- **Evolutionary.jl:** its CMA-ES with the default σ of 0.5 doesn't reach the targets.
- **Metaheuristics.jl:**
  - It has no CMA-ES.
  - With 3 objectives, its SMS-EMOA estimates the hypervolume by Monte Carlo, and reaches the 60-second limit before the budget.
