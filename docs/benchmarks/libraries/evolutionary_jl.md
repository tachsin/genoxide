# Evolutionary.jl (Julia, 0.12.0)

Evolutionary.jl is a Julia package of evolution strategies (ES), CMA-ES, genetic algorithms (GA), differential evolution (DE), NSGA-II and genetic programming, with a set of mutation, crossover and selection operators. Its documentation is at [wildart.github.io/Evolutionary.jl](https://wildart.github.io/Evolutionary.jl/stable/), built from `docs/src` of the package; its only worked examples for our problem types are its tests (`test/*.jl` of the package), which the adapter cites. The repository is now [SciML/Evolutionary.jl](https://github.com/SciML/Evolutionary.jl).

Adapter: [benchmarks/adapters/evolutionary_jl/](../../../benchmarks/adapters/evolutionary_jl/).
Know a better way to solve one of these problems with Evolutionary.jl? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How every run is set up

- **Evaluations** (rule 3): `counted` and `counted!` ([bench.jl](../../../benchmarks/adapters/evolutionary_jl/bench.jl), lines 186-205) count every call of the fitness function, including the one `EvolutionaryObjective` makes before the run to learn the type of the value (`zero(f(x))` in `src/api/objective.jl`).
- **Stops** (rule 2): `options` (lines 212-222) sets the iteration limit to `typemax(Int)` and `successive_f_tol` to `typemax(Int)`, which turns off the library's convergence test (a metric such as `AbsDiff(1e-12)` below its tolerance for more than `successive_f_tol` generations, `src/api/optimize.jl`). The run ends in the `callback` of [`Options`](https://wildart.github.io/Evolutionary.jl/stable/tutorial/#General-options), after every generation, at the target, the budget or the time cap.
- **Restarts** (rule 2.2): one stop can't be turned off: CMA-ES ends when the eigendecomposition of its covariance matrix fails (`update_state!` returns `true`, `src/cmaes.jl`). Evolutionary.jl has no restart mechanism, so `run_restarting` (lines 224-239) starts the method again from a new random start, with the seed `seed * 1000 + restart`, and keeps the best solution and every evaluation. Only CMA-ES restarts in practice; each run prints its `restarts`.
- **Time** (rule 4): the clock starts before `optimize` creates the initial population. Each solver first makes an untimed warm-up run of the same problem with 1,000 evaluations and seed 1000.
- **One thread** (rule 4.3): [run.sh](../../../benchmarks/adapters/evolutionary_jl/run.sh) runs Julia 1.13 with `--threads=1 --gcthreads=1,0` (one GC mark thread, no concurrent sweep thread) and BLAS with one thread. `run.py check` measured CPU/wall 1.00 in every scenario.
- **Seeds** (rule 5.2): the seed goes to a `Xoshiro` passed as `rng` to `Options`, which the library's operators use; the initial populations use the same generator.

## Binary: OneMax 100 and 1000 (matched), OneMax 100 (idiomatic)

**Methods:**
- Matched: `GA` with DEAP's `eaSimple` settings: population 300, `tournament(3)`, two-point crossover (`TPX`) at 0.5, bit-flip at 1/n on 20% of the children, no elitism (`onemax_solvers`, lines 271-303). The library's `flip` flips exactly one bit, so the adapter's `bitflip_per_gene` does DEAP's per-gene flip; the crossover probability is the wrapper `crossover_with_probability` with `crossoverRate = 1`, which copies the parents it doesn't cross (see "Bugs found" for why). Evolutionary's `tournament` draws its contestants without replacement.
- Idiomatic: `GA(selection = tournament(3), mutation = flip, crossover = TPX, mutationRate = 0.05, crossoverRate = 0.85, populationSize = 100)`, the library's only binary example, `test/onemax.jl`, whose genome also has 100 bits.

**Keeping going:** runs to the budget by itself, with the options above.

**Left out:** none: the documentation has no other binary configuration.

**Separate tests** (2026-09-25, Evolutionary.jl 0.12.0, seeds 0 to 4, the scenario's budget, 60 s cap):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|---|
| OneMax 100, matched | ga | 5 | 5 | 9,901 | 100 (100, 100) | 0 |
| OneMax 1000, matched | ga | 5 | 5 | 135,001 | 1000 (1000, 1000) | 0 |
| OneMax 100, idiomatic | ga | 5 | 5 | 5,301 | 100 (100, 100) | 0 |

## Permutation: N-Queens 32 and 64

**Methods** (`nqueens_solvers`, lines 305-327), both from the library's only permutation example, `test/n-queens.jl`:
- `ga`: `GA(populationSize = 100, selection = tournament(5), crossover = PMX, crossoverRate = 0.89, mutation = inversion, mutationRate = 0.06)`. The test tries 5 mutations × 5 crossovers with these settings; the adapter takes the first pair of its lists.
- `es`: `ES(mutation = mutationwrapper(inversion), μ = 20, ρ = 1, λ = 100, selection = :plus)`. The test tries 5 mutations with `:plus` and `:comma`; the adapter takes the first.

**Keeping going:** runs to the budget by itself, with the options above. The test stops these runs with `successive_f_tol = 30`, which the adapter turns off.

**Left out:** the other 24 mutation/crossover pairs and 9 ES variants of the same test: the test presents them as equals, and choosing among them by their results would be tuning.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|---|
| N-Queens 32 | ga | 5 | 0 | – | 1 (1, 3) | 0 |
| N-Queens 32 | es | 5 | 0 | – | 1 (1, 2) | 0 |
| N-Queens 64 | ga | 5 | 0 | – | 5 (3, 5) | 0 |
| N-Queens 64 | es | 5 | 0 | – | 2 (1, 3) | 0 |

Neither method solves N-Queens 32 or 64 within the budget. Both run to their budget; the best values are the numbers of diagonal conflicts left.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** (`real_solvers`, lines 329-366). The bounds are a `BoxConstraints` ([docs](https://wildart.github.io/Evolutionary.jl/stable/constraints/#Box-Constrained-Optimization)): the initial population is random within them, and offspring are clipped to them.
- `es`: the (μ/ρ(+/,)λ)-ES of the [ES page](https://wildart.github.io/Evolutionary.jl/stable/es/), with the settings of `test/rastrigin.jl`: a (15/15,100)-σ-SA-ES, `AnisotropicStrategy`, `average` recombination of the solutions and strategies, `gaussian` mutation of both.
- `cma_es`: the [CMA-ES page](https://wildart.github.io/Evolutionary.jl/stable/cmaes/) presents it for "difficult (non-convex, ill-conditioned, multi-modal, rugged, noisy) optimization problems in continuous search spaces". `CMAES(lambda = 100)`, the setting of `test/rastrigin.jl`, i.e. a (50,100)-CMA-ES with the default σ0 of 0.5, from a random point of the domain.
- `de`: the [DE page](https://wildart.github.io/Evolutionary.jl/stable/de/): DE "is used for multidimensional real-valued functions". `DE(populationSize = 100)`, the setting of `test/rosenbrock.jl`; `test/rastrigin.jl` uses the same population with F = 0.9, the default. So DE/rand/1/bin with F = 0.9 and Cr = 0.5.

**Keeping going:** ES and DE run to the budget by themselves. CMA-ES restarts after each breakdown of its covariance matrix, as above: 15 restarts in the 5 Rastrigin 30 runs and 5 in the Ackley 30 runs. Between breakdowns, a run continues to the budget without reaching the target (see "Bugs found").

**Left out:**
- `GA`: Evolutionary.jl presents four methods for real numbers, and rule 6.4 allows three. The [README](https://github.com/SciML/Evolutionary.jl#algorithms) and the [documentation's index](https://wildart.github.io/Evolutionary.jl/stable/) list ES, CMA-ES, GA and DE. The ES, CMA-ES and DE pages present their method for real-valued or continuous problems. The GA page presents the GA as a general framework, and shows it on real numbers only in the getting-started example on the Sphere function and in the tests. So the GA is the one left out. For reference, the 0.6.0 benchmark ran the GA of `test/rosenbrock.jl` (`rouletteinv`, `IC(0.2)`, `BGA`, ε = 0.1): it reached the target in 0 of 10 runs on every continuous scenario. This choice is a judgment call: if you read the documentation otherwise, open an issue.
- CMA-ES with another σ0 or λ: the documented defaults and the tests' λ = 100 are used; nothing else is documented for these problems.
- `TreeGP`: genetic programming, for expressions, not vectors.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|---|
| Rastrigin 10 | es | 5 | 1 | 51,216 | 0.995 (0.00626, 1.99) | 0 |
| Rastrigin 10 | cma_es | 5 | 0 | – | 69.65 (44.77, 99.50) | 0 |
| Rastrigin 10 | de | 5 | 2 | 31,801 | 0.995 (0.00785, 3.98) | 0 |
| Rastrigin 30 | es | 5 | 0 | – | 11.94 (4.97, 17.91) | 0 |
| Rastrigin 30 | cma_es | 5 | 0 | – | 223.7 (189.1, 254.2) | 0 |
| Rastrigin 30 | de | 5 | 0 | – | 23.39 (19.90, 35.82) | 0 |
| Ackley 30 | es | 5 | 5 | 9,516 | 0.00903 (0.00876, 0.00983) | 0 |
| Ackley 30 | cma_es | 5 | 0 | – | 19.47 (19.34, 19.59) | 0 |
| Ackley 30 | de | 5 | 5 | 155,201 | 0.00962 (0.00879, 0.00990) | 0 |

## Continuous, unimodal: Rosenbrock 10

**Methods:** the same three as above, with the settings of `test/rosenbrock.jl`:
- `es`: a (15/3+100)-ES, `IsotropicStrategy`, `average` recombination, `gaussian` mutation: the first of the settings the test tries.
- `cma_es`: `CMAES(lambda = 100)`. The [tutorial](https://wildart.github.io/Evolutionary.jl/stable/tutorial/) also minimizes Rosenbrock with CMA-ES, with the defaults.
- `de`: `DE(populationSize = 100)`.

**Keeping going:** as above. No CMA-ES run restarted on Rosenbrock.

**Left out:** the GA, as above.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|---|
| Rosenbrock 10 | es | 5 | 0 | – | 0.0734 (0.0526, 0.469) | 0 |
| Rosenbrock 10 | cma_es | 5 | 3 | 102,801 | 0.00998 (0.00987, 0.961) | 0 |
| Rosenbrock 10 | de | 5 | 5 | 88,001 | 0.00964 (0.00815, 0.00974) | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1 (matched)

**Methods:** [`NSGA2`](https://wildart.github.io/Evolutionary.jl/stable/moea/), the library's only multi-objective algorithm, with the matched settings (`run_front`, lines 386-411): population 100 (92 with 3 objectives), SBX with η 15 at 0.9, polynomial mutation with η 20 at 1/n, binary tournament on rank and crowding distance (the library's default). Differences: Evolutionary's `SBX` and `PLM` are the unbounded variants, and the box constraints clip the offspring to [0, 1]; its `SBX` crosses each variable with probability 0.5, as pymoo's. The crossover probability is `crossover_with_probability` with `crossoverRate = 1`, for the reason under "Bugs found". The objectives use the in-place form `f!(F, x)` of the [tutorial](https://wildart.github.io/Evolutionary.jl/stable/tutorial/#Objective-Function-Definition).

**The front** (rule 7.2): the non-dominated part of the final population of 100 (92), which NSGA2 leaves in the population vector passed to `optimize`. Its objective values are the ones recorded when those solutions were evaluated (`counted!` keeps them), not NSGA2's own, which the bug below mixes up.

**Keeping going:** runs to the budget by itself, with the options above (its convergence metric, `GD`, is turned off like the others).

**Left out:** no other multi-objective algorithm exists in the library.

**Separate tests** (2026-09-25; hypervolume computed by `run.py`'s code from the reported solutions):

| Scenario | Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|---|
| ZDT1 | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 25,000 | 0 |
| ZDT2 | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 25,000 | 0 |
| ZDT3 | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 25,000 | 0 |
| DTLZ2 | nsga2 | 5 | 0.3084 (0.3773, 0.2609) | 25,024 | 0 |
| DTLZ1 | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 40,020 | 0 |

The hypervolumes of 0 are the NSGA2 bug below. `EVOLUTIONARY_JL_NSGA2=0` makes the adapter skip these scenarios.

## Can't run

Every scenario runs.

## Bugs found

- **NSGA2 selects on other individuals' values** ([SciML/Evolutionary.jl#174](https://github.com/SciML/Evolutionary.jl/issues/174)): `update_state!` (`src/nsga2.jl`) reorders the parents (`parents .= state.population[fitidx]`) but not their objective values in `state.fitpop[:, 1:N]`, nor their ranks and crowding distances. From the second generation on, it sorts and selects on the values of other individuals. The results show it: hypervolumes of 0 on ZDT1, ZDT2, ZDT3 and DTLZ1. Not worked around.
- **CMA-ES doesn't converge in 30 dimensions.** On a 30-dimensional sphere centred at 3, from a random start within [−5.12, 5.12]³⁰ with those bounds, `CMAES()` ends at f = 95.8 after 60,000 evaluations and `CMAES(lambda = 100)` at 227 after 300,000; without bounds, from a random start in [0, 1]³⁰, at 17.6 and 19.8. pycma 4.5.0, with the same σ0 of 0.5, start and bounds, reaches 1e-10 in 5,180 and 5,446 evaluations (seeds 1 and 2). In 10 dimensions it converges. The step-size update in `src/cmaes.jl` uses ‖s_σ‖/N, where the form of Hansen's tutorial it follows (the `min(1, …)/2` variant) has ‖s_σ‖²/N, so σ shrinks even when selection is neutral. Squaring the norm alone doesn't fix the runs (they then diverge with bounds, and reach 0.25 and 2.1 without), so the cause isn't only that line; it wasn't investigated further. The results show it: CMA-ES reaches no multimodal target, and its covariance matrix breaks down now and then (the restarts above). Not reported upstream yet; not worked around.
- **GA offspring share their parents** (`recombine!`, `src/ga.jl`): a pair that isn't crossed gets the parents themselves, not copies, and `mutate!` then mutates them in place. Two offspring of the same parent become one object, and with elitism (ε > 0) an elite can be mutated. With the N-Queens settings, a median of 1 individual per generation is the same object as another. The matched GA avoids it with `crossoverRate = 1` and a wrapper that copies; the idiomatic GAs run as users get them. Not reported upstream yet.
- **The (μ+λ)-ES can lose a surviving parent** (`update_state!`, `src/es.jl`): it writes each surviving offspring into the slot of its rank, which can overwrite a parent that also survived. A (15/3+100)-ES on a sphere, from 2,000 random starts, lost its best parent in 3 of 40,000 generations: a small effect. Not reported upstream yet.
