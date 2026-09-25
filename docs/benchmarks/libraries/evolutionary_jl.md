# Evolutionary.jl (Julia, 0.12.0)

Evolutionary.jl is a Julia package of evolution strategies (ES), CMA-ES, genetic algorithms (GA), differential evolution (DE), NSGA-II and genetic programming, with a set of mutation, crossover and selection operators. Its documentation is at [wildart.github.io/Evolutionary.jl](https://wildart.github.io/Evolutionary.jl/stable/), built from `docs/src` of the package. Besides the documentation's own examples, its worked examples for our problem types are its tests (`test/*.jl` of the package), which the adapter cites. The repository is now [SciML/Evolutionary.jl](https://github.com/SciML/Evolutionary.jl).

Adapter: [benchmarks/adapters/evolutionary_jl/](../../../benchmarks/adapters/evolutionary_jl/).
Know a better way to solve one of these problems with Evolutionary.jl? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How every run is set up

- **Evaluations** (rule 3): `counted` and `counted!` ([bench.jl](../../../benchmarks/adapters/evolutionary_jl/bench.jl), lines 188-212) count every call of the fitness function, including the one `EvolutionaryObjective` makes before each attempt to learn the type of the value (`zero(f(x))` in `src/api/objective.jl`).
- **Stops and restarts** (rule 2.2), `options` and `run_restarting` (lines 219-248):
  - The iteration limit is only a budget (the library's default is 1,000 generations, 1,500 for CMA-ES), so it's lifted to `typemax(Int)`.
  - The convergence test is the library's: each method's default metric (`AbsDiff(1e-12)` of the best value for GA and CMA-ES, `AbsDiff(1e-10)` for DE and ES, `GD` and `GD(true)` for NSGA2) within its tolerance for more than `successive_f_tol` generations (`src/api/optimize.jl`). `successive_f_tol` is the default of 10, or the value of the library's example for the problem type (25 for DE on Rastrigin, 30 for the GA on N-Queens).
  - These convergence criteria count: they are part of each method's own settings (the `metrics` keyword of `GA`, `ES`, `CMAES`, `DE` and `NSGA2`, with its default), and the library's ways to set a budget (`iterations`, `time_limit`, a `callback`) leave them in effect.
  - CMA-ES also ends when the eigendecomposition of its covariance matrix fails (`update_state!` returns `true`, `src/cmaes.jl`).
  - An attempt that ends one of these ways starts again from a new random start, with the seed `seed * 1000 + restart`: Evolutionary.jl has no restart mechanism. The best solution and every evaluation are kept. Each run prints its `restarts`.
  - The `callback` of [`Options`](https://wildart.github.io/Evolutionary.jl/stable/tutorial/#General-options) ends the run at the target, the budget or the time cap, after every generation.
- **Bounds** (rule 2.4): the continuous and multi-objective problems use `BoxConstraints` ([docs](https://wildart.github.io/Evolutionary.jl/stable/constraints/#Box-Constrained-Optimization)): the initial population is drawn within the bounds (`initial_population`, `src/api/utilities.jl`), and every new solution of GA, ES, CMA-ES, DE and NSGA2 is clipped to them (`apply!` → `clip!`, `src/api/constraints.jl`) before it's evaluated. The adapter counts the evaluated solutions outside the bounds (`outside`): 0 in every run.
- **Time** (rule 4): the clock starts before `optimize` creates the initial population. Each solver first makes an untimed warm-up run of the same problem with 1,000 evaluations and seed 1000.
- **One thread** (rule 4.3): [run.sh](../../../benchmarks/adapters/evolutionary_jl/run.sh) runs Julia 1.13 with `--threads=1 --gcthreads=1,0` (one GC mark thread, no concurrent sweep thread) and BLAS with one thread. `run.py check` measured CPU/wall 1.00.
- **Seeds** (rule 5.2): the seed goes to a `Xoshiro` passed as `rng` to `Options`, which the library's operators use; the initial populations use the same generator.
- **How the documentation decides** (rule 6.2): first what the documentation states a method is for; then its examples for the problem type, the published documentation's own (the tutorial, the getting-started example) before the package's tests; then the defaults. Where an example presents several settings as equals (the N-Queens and DE tests loop over operators), the settings it fixes are used, the ones it varies take the library's default when the default is among them, and otherwise the first of its list (a tie, rule 6.2).

## Binary: OneMax 100 and 1000 (matched), OneMax 100 (idiomatic)

**Methods** (`onemax_solvers`, lines 280-313):
- Matched: `GA` with DEAP's `eaSimple` settings: population 300, `tournament(3)`, two-point crossover (`TPX`) at 0.5, bit-flip at 1/n on 20% of the children, no elitism. The library's `flip` flips exactly one bit, so the adapter's `bitflip_per_gene` does DEAP's per-gene flip; the crossover probability is the wrapper `crossover_with_probability` with `crossoverRate = 1`, which copies the parents it doesn't cross (see "Bugs found" for why). Evolutionary's `tournament` draws its contestants without replacement.
- Idiomatic: `GA(selection = tournament(3), mutation = flip, crossover = TPX, mutationRate = 0.05, crossoverRate = 0.85, populationSize = 100)` with the default options, the library's only binary example, `test/onemax.jl`, whose genome also has 100 bits.

**Keeping going:** the GA's convergence test (the best value unchanged for more than 10 generations) restarts it, in the matched runs too. Without elitism, the matched GA's best value often stays the same for 10 generations near the end: on OneMax 1000, 47 restarts in 5 runs, and 3 of 5 runs reached the target, where the same GA without restarts reached it in 5 of 5 (median 135,001 evaluations, in the separate tests before rule 2.2 was amended).

**Left out:** none: the documentation has no other binary configuration.

**Separate tests** (2026-09-25, Evolutionary.jl 0.12.0, seeds 0 to 4, the scenario's budget, 60 s cap):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap | Restarts (5 runs) |
|---|---|---|---|---|---|---|---|
| OneMax 100, matched | ga | 5 | 5 | 9,901 | 100 (100, 100) | 0 | 0 |
| OneMax 1000, matched | ga | 5 | 3 | 766,806 | 1000 (1000, 999) | 0 | 47 |
| OneMax 100, idiomatic | ga | 5 | 5 | 7,602 | 100 (100, 100) | 0 | 6 |

## Permutation: N-Queens 32 and 64

**Methods** (`nqueens_solvers`, lines 315-339), both from the library's only permutation example, `test/n-queens.jl`:
- `ga`: `GA(populationSize = 100, selection = tournament(5), crossover = PMX, crossoverRate = 0.89, mutation = inversion, mutationRate = 0.06)` with `successive_f_tol = 30`. The test tries 5 mutations × 5 crossovers as equals; the library's defaults (`genop`, which does nothing) aren't among them, so the adapter takes the first of each list.
- `es`: `ES(mutation = mutationwrapper(inversion), μ = 20, ρ = 1, λ = 100, selection = :plus)` with the default options. The test tries 5 mutations with `:plus` and `:comma`: the first mutation, and `:plus`, the library's default selection.

**Keeping going:** both converge often, and restart (see the table).

**Left out:** the other mutation/crossover pairs and ES variants of the same test, for the reason above.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap | Restarts (5 runs) |
|---|---|---|---|---|---|---|---|
| N-Queens 32 | ga | 5 | 3 | 188,437 | 0 (0, 1) | 0 | 310 |
| N-Queens 32 | es | 5 | 2 | 216,341 | 1 (0, 1) | 0 | 641 |
| N-Queens 64 | ga | 5 | 0 | – | 5 (4, 6) | 0 | 766 |
| N-Queens 64 | es | 5 | 0 | – | 7 (6, 7) | 0 | 1,204 |

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** (`real_solvers`, lines 341-387). The library presents four methods for real numbers (the [README](https://github.com/SciML/Evolutionary.jl#algorithms) and the [documentation's index](https://wildart.github.io/Evolutionary.jl/stable/) list ES, CMA-ES, GA and DE), and rule 6.4 allows three. The documentation states what two of them are for, so they come first; the third is the library's example for the problem type, `test/rastrigin.jl`, whose first method is the ES:
- `cma_es`: the [CMA-ES page](https://wildart.github.io/Evolutionary.jl/stable/cmaes/): for "difficult (non-convex, ill-conditioned, multi-modal, rugged, noisy) optimization problems in continuous search spaces". `CMAES(lambda = 100)` with the default options, as in `test/rastrigin.jl`: a (50,100)-CMA-ES with the default σ0 of 0.5, from a random point of the domain.
- `de`: the [DE page](https://wildart.github.io/Evolutionary.jl/stable/de/): DE "is used for multidimensional real-valued functions". `test/rastrigin.jl` runs DE with population 100 and F = 0.9, with `successive_f_tol = 25`, and varies the selection, the recombination and the number of differences, which take the library's defaults (`random`, `BINX(0.5)`, 1): DE/rand/1/bin.
- `es`: the (15/15,100)-σ-SA-ES of `test/rastrigin.jl` ([ES page](https://wildart.github.io/Evolutionary.jl/stable/es/)): `AnisotropicStrategy`, `average` recombination of the solutions and strategies, `gaussian` mutation of both, comma selection.

**Keeping going:** all three converge and restart (see the table).

**Left out:**
- `GA`: the fourth method. Its documentation presents it as a general framework; its examples on real numbers are the getting-started example on the Sphere function and `test/rastrigin.jl`, where it comes after the ES and CMA-ES.
- `TreeGP`: genetic programming, for expressions, not vectors.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap | Restarts (5 runs) |
|---|---|---|---|---|---|---|---|
| Rastrigin 10 | cma_es | 5 | 0 | – | 26.86 (20.89, 35.82) | 0 | 134 |
| Rastrigin 10 | de | 5 | 5 | 154,804 | 0.00858 (0.00785, 0.00974) | 0 | 31 |
| Rastrigin 10 | es | 5 | 3 | 437,876 | 0.00820 (0.00394, 0.995) | 0 | 40 |
| Rastrigin 30 | cma_es | 5 | 0 | – | 148.5 (125.4, 161.4) | 0 | 91 |
| Rastrigin 30 | de | 5 | 0 | – | 192.4 (176.3, 199.7) | 0 | 909 |
| Rastrigin 30 | es | 5 | 0 | – | 4.97 (4.97, 4.97) | 0 | 120 |
| Ackley 30 | cma_es | 5 | 0 | – | 19.24 (18.68, 19.34) | 0 | 49 |
| Ackley 30 | de | 5 | 0 | – | 0.359 (0.125, 3.70) | 0 | 253 |
| Ackley 30 | es | 5 | 5 | 9,516 | 0.00903 (0.00876, 0.00983) | 0 | 0 |

DE's convergence test ends an attempt when its best value hasn't changed for 26 generations. On Rastrigin 30 and Ackley 30 that happens often (909 and 253 restarts in 5 runs), and each restart throws the population away. Before rule 2.2 was amended, with the test turned off, the same method (F = 0.9, the default) reached Ackley 30 in 5 of 5 runs (median 155,201 evaluations) and Rastrigin 30 to a median of 23.4.

## Continuous, unimodal: Rosenbrock 10

**Methods** (`real_solvers`): CMA-ES and DE, for the reasons above, and the GA, the documentation's own example for a continuous unimodal function. The settings are the documentation's examples for this problem type:
- `cma_es`: the [tutorial](https://wildart.github.io/Evolutionary.jl/stable/tutorial/) minimizes Rosenbrock with `CMAES()` and the default options: a (10,20)-CMA-ES with σ0 0.5, from a random point of the domain.
- `de`: `DE(populationSize = 100)` with the default options, as in `test/rosenbrock.jl`: DE/rand/1/bin with F = 0.9, Cr = 0.5.
- `ga`: the [getting-started example](https://wildart.github.io/Evolutionary.jl/stable/#Getting-started), on the Sphere function: `GA(populationSize = 100, selection = susinv, crossover = DC, mutation = PLM())` with the default rates (crossover 0.8, mutation 0.1) and options.

**Keeping going:** all three converge and restart (see the table).

**Left out:** the ES: `test/rosenbrock.jl` starts with it, but the documentation's own examples for this problem type (the tutorial and the getting-started example) show CMA-ES and the GA.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap | Restarts (5 runs) |
|---|---|---|---|---|---|---|---|
| Rosenbrock 10 | cma_es | 5 | 0 | – | 1.236 (0.224, 1.314) | 0 | 62 |
| Rosenbrock 10 | de | 5 | 0 | – | 0.967 (0.294, 4.70) | 0 | 266 |
| Rosenbrock 10 | ga | 5 | 0 | – | 6.00 (3.88, 11.86) | 0 | 71 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1 (matched)

**Methods:** [`NSGA2`](https://wildart.github.io/Evolutionary.jl/stable/moea/), the library's only multi-objective algorithm, with the matched settings (`run_front`, lines 407-434): population 100 (92 with 3 objectives), SBX with η 15 at 0.9, polynomial mutation with η 20 at 1/n, binary tournament on rank and crowding distance (the library's default). Differences: Evolutionary's `SBX` and `PLM` are the unbounded variants, and the box constraints clip the offspring to [0, 1]; its `SBX` crosses each variable with probability 0.5, as pymoo's. The crossover probability is `crossover_with_probability` with `crossoverRate = 1`, for the reason under "Bugs found". The objectives use the in-place form `f!(F, x)` of the [tutorial](https://wildart.github.io/Evolutionary.jl/stable/tutorial/#Objective-Function-Definition).

**The front** (rule 7.2): the non-dominated part of the final population of 100 (92), which NSGA2 leaves in the population vector passed to `optimize` (after a restart, the last attempt's). Its objective values are the ones recorded when those solutions were evaluated (`counted!` keeps them), not NSGA2's own, which the bug below mixes up.

**Keeping going:** its convergence test (the `GD` metrics, default options as in `test/moea.jl`) restarts it; it never triggered in the separate tests.

**Left out:** no other multi-objective algorithm exists in the library.

**Separate tests** (2026-09-25; hypervolume computed by `run.py`'s code from the reported solutions):

| Scenario | Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap | Restarts |
|---|---|---|---|---|---|---|
| ZDT1 | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 25,000 | 0 | 0 |
| ZDT2 | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 25,000 | 0 | 0 |
| ZDT3 | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 25,000 | 0 | 0 |
| DTLZ2 | nsga2 | 5 | 0.3084 (0.3773, 0.2609) | 25,024 | 0 | 0 |
| DTLZ1 | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 40,020 | 0 | 0 |

The hypervolumes of 0 are the NSGA2 bug below. `EVOLUTIONARY_JL_NSGA2=0` makes the adapter skip these scenarios.

## Can't run

Every scenario runs.

## Bugs found

- **NSGA2 selects on other individuals' values** ([SciML/Evolutionary.jl#174](https://github.com/SciML/Evolutionary.jl/issues/174)): `update_state!` (`src/nsga2.jl`) reorders the parents (`parents .= state.population[fitidx]`) but not their objective values in `state.fitpop[:, 1:N]`, nor their ranks and crowding distances. From the second generation on, it sorts and selects on the values of other individuals. The results show it: hypervolumes of 0 on ZDT1, ZDT2, ZDT3 and DTLZ1. Not worked around.
- **CMA-ES doesn't converge in 30 dimensions.** On a 30-dimensional sphere centred at 3, from a random start within [−5.12, 5.12]³⁰ with those bounds, `CMAES()` ends at f = 95.8 after 60,000 evaluations and `CMAES(lambda = 100)` at 227 after 300,000; without bounds, from a random start in [0, 1]³⁰, at 17.6 and 19.8. pycma 4.5.0, with the same σ0 of 0.5, start and bounds, reaches 1e-10 in 5,180 and 5,446 evaluations (seeds 1 and 2). In 10 dimensions it converges. The step-size update in `src/cmaes.jl` uses ‖s_σ‖/N, where the form of Hansen's tutorial it follows (the `min(1, …)/2` variant) has ‖s_σ‖²/N, so σ shrinks even when selection is neutral. Squaring the norm alone doesn't fix the runs (they then diverge with bounds, and reach 0.25 and 2.1 without), so the cause isn't only that line; it wasn't investigated further. The results show it: CMA-ES reaches no multimodal target, and its covariance matrix breaks down now and then. Not reported upstream yet; not worked around.
- **GA offspring share their parents** (`recombine!`, `src/ga.jl`): a pair that isn't crossed gets the parents themselves, not copies, and `mutate!` then mutates them in place. Two offspring of the same parent become one object, and with elitism (ε > 0) an elite can be mutated. With the N-Queens settings, a median of 1 individual per generation is the same object as another. The matched GA avoids it with `crossoverRate = 1` and a wrapper that copies; the idiomatic GAs run as users get them. Not reported upstream yet.
- **The (μ+λ)-ES can lose a surviving parent** (`update_state!`, `src/es.jl`): it writes each surviving offspring into the slot of its rank, which can overwrite a parent that also survived. A (15/3+100)-ES on a sphere, from 2,000 random starts, lost its best parent in 3 of 40,000 generations: a small effect. Not reported upstream yet.
