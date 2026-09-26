# Evolutionary.jl (Julia, 0.12.0)

Evolutionary.jl is a Julia package of evolution strategies (ES), CMA-ES, genetic algorithms (GA), differential evolution (DE), NSGA-II and genetic programming, with mutation, crossover and selection operators. Its docs are at [docs.sciml.ai/Evolutionary](https://docs.sciml.ai/Evolutionary/stable/), from `docs/src` of [SciML/Evolutionary.jl](https://github.com/SciML/Evolutionary.jl); its tests (`test/*.jl`) are its other worked examples for these problem types.

Adapter: [benchmarks/adapters/evolutionary_jl/](../../../benchmarks/adapters/evolutionary_jl/).
Know a better way to solve one of these problems with Evolutionary.jl? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs Evolutionary.jl

- **Evaluations:** `counted` and `counted!` count every call, including the one `EvolutionaryObjective` makes before each attempt to learn the value's type (`zero(f(x))`), and record the first hit ([bench.jl, lines 211-234](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L211-L234)).
- **Stop:** the `callback` of [`Options`](https://docs.sciml.ai/Evolutionary/stable/tutorial/#General-options), after every generation.
- **Keeping going (rule 2.2)** ([lines 245-276](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L245-L276)):
  - the iteration limit (1,000; 1,500 for CMA-ES) is lifted;
  - the library's convergence test counts: each method's default metric (`AbsDiff(1e-12)` for GA and CMA-ES, `AbsDiff(1e-10)` for DE and ES, `GD` and `GD(true)` for NSGA2) within tolerance for more than `successive_f_tol` generations (`src/api/optimize.jl`): 10 by default, or the example's value (25 for DE on Rastrigin, 30 for the GA on N-Queens). It's part of each method's settings (`metrics`) and stays in effect with `iterations`, `time_limit` or a `callback`;
  - CMA-ES also ends when the eigendecomposition fails (`update_state!` returns `true`, `src/cmaes.jl`);
  - Evolutionary.jl has no restart mechanism, so an attempt restarts from a new random start with the seeds of rule 2.2. Runs print `restarts`;
  - the matched scenarios have no convergence criterion (`successive_f_tol = typemax(Int)`).
- **Bounds (rule 2.4):** `BoxConstraints` ([docs](https://docs.sciml.ai/Evolutionary/stable/constraints/#Box-Constrained-Optimization)): the initial population is drawn within the bounds (`src/api/utilities.jl`), and every new solution is clipped before it's evaluated (`apply!` → `clip!`, `src/api/constraints.jl`).
- **Time:** from the run's `Budget`, before `optimize` creates the initial population. Warm-up as rule 4.2.
- **One thread:** [run.sh](../../../benchmarks/adapters/evolutionary_jl/run.sh) runs Julia 1.13 with `--threads=1 --gcthreads=1,0` and BLAS with one thread.
- **Seeds:** a `Xoshiro` passed as `rng` to `Options`, used by the operators and the initial populations.
- **Choosing among the docs (rule 6.2):** first what the docs say a method is for; then examples, the published docs' own before the tests; then defaults. Where an example tries several settings as equals, the ones it varies take the library's default when it's among them, else the first listed.
- **Separate tests:** 2026-09-25, Evolutionary.jl 0.12.0, seeds 0 to 4, the scenario's budget, 60 s cap. `outside` was 0 in every run.

## Binary: OneMax 100 and 1000 (matched), OneMax 100 (idiomatic)

**Methods** (`onemax_solvers`, [lines 282-316](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L282-L316)):
- **Matched:** `GA` with its own operators: population 300, `tournament(3)`, `TPX` with `crossoverRate = 0.5`, `flip` with `mutationRate = 0.2`, `ɛ = 0`. Differences:
  - `flip` flips exactly one random bit of a mutated child (the same mean as 1/n per bit);
  - `tournament` draws without replacement from a shuffled population;
  - every child is evaluated, changed or not;
  - an uncrossed pair passes the parents themselves, not copies (see [Bugs found](#bugs-found)), included (rule 6.1).
- **Idiomatic:** the [tutorial](https://docs.sciml.ai/Evolutionary/stable/tutorial/#General-options)'s GA, which counts the ones of a `BitVector`: `GA(selection = uniformranking(5), mutation = flip, crossover = SPX)` with the defaults (population 50, `crossoverRate` 0.8, `mutationRate` 0.1, `ɛ` 0). Its `Options(iterations = 10)` is lifted. The bits start random (the tutorial: zeros).

**Keeping going:** matched: to the target or the budget. Idiomatic: the convergence test (the best unchanged for more than 10 generations) restarts it.

**Left out:** the tests' binary GAs, after the published example: `test/onemax.jl` (`tournament(3)`, `TPX`, `flip`, `crossoverRate` 0.85, `mutationRate` 0.05, population 100) and `test/knapsack.jl` (`roulette`, `swap2`, `SPX`; `tournament(3)`, `inversion`, `SPX`; both elitist).

**Separate tests:**

| Scenario | Solver | Runs | Reached | Median first hit (evaluations) | Best value: median (best, worst) | Capped | Restarts (all runs) |
|---|---|---|---|---|---|---|---|
| OneMax 100, matched | ga | 5 | 5 | 9,645 | 100 (100, 100) | 0 | 0 |
| OneMax 1000, matched | ga | 5 | 5 | 117,036 | 1,000 (1,000, 1,000) | 0 | 0 |
| OneMax 100, idiomatic | ga | 5 | 5 | 10,811 | 100 (100, 100) | 0 | 14 |

## Permutation: N-Queens 32 and 64

**Methods** (`nqueens_solvers`, [lines 318-342](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L318-L342)), from the only permutation example, `test/n-queens.jl`:
- **`ga`:** `GA(populationSize = 100, selection = tournament(5), crossover = PMX, crossoverRate = 0.89, mutation = inversion, mutationRate = 0.06)`, `successive_f_tol = 30`. The test tries 5 mutations × 5 crossovers; the defaults (`genop`, which does nothing) aren't among them, so the first of each list.
- **`es`:** `ES(mutation = mutationwrapper(inversion), μ = 20, ρ = 1, λ = 100, selection = :plus)`: the first mutation of the test, and `:plus`, the default selection.

**Keeping going:** both converge and restart (see the table).

**Left out:** the test's other operator pairs and ES variants.

**Separate tests:**

| Scenario | Solver | Runs | Reached | Median first hit (evaluations) | Best value: median (best, worst) | Capped | Restarts (all runs) |
|---|---|---|---|---|---|---|---|
| N-Queens 32 | ga | 5 | 2 | 262,084 | 1 (0, 1) | 0 | 377 |
| N-Queens 32 | es | 5 | 1 | 377,775 | 1 (0, 1) | 0 | 774 |
| N-Queens 64 | ga | 5 | 0 | - | 6 (5, 7) | 0 | 768 |
| N-Queens 64 | es | 5 | 0 | - | 7 (6, 7) | 0 | 1,219 |

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** (`real_solvers`, [lines 344-390](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L344-L390)): the [README](https://github.com/SciML/Evolutionary.jl#algorithms) and the [docs' index](https://docs.sciml.ai/Evolutionary/stable/) list four methods for real numbers (ES, CMA-ES, GA, DE). The docs say what two are for, so they come first; the third is the first method of the example `test/rastrigin.jl`:
- **`cma_es`:** for "difficult (non-convex, ill-conditioned, multi-modal, rugged, noisy) optimization problems" ([CMA-ES page](https://docs.sciml.ai/Evolutionary/stable/cmaes/)). `CMAES(lambda = 100)` as in `test/rastrigin.jl`: a (50,100)-CMA-ES, σ0 0.5, from a random point.
- **`de`:** "used for multidimensional real-valued functions" ([DE page](https://docs.sciml.ai/Evolutionary/stable/de/)). `test/rastrigin.jl`'s population 100, F = 0.9, `successive_f_tol = 25`, and the defaults for what it varies (`random`, `BINX(0.5)`, 1 difference): DE/rand/1/bin.
- **`es`:** `test/rastrigin.jl`'s (15/15,100)-σ-SA-ES ([ES page](https://docs.sciml.ai/Evolutionary/stable/es/)): `AnisotropicStrategy`, `average` recombination, `gaussian` mutation, comma selection.

**Keeping going:** all three converge and restart (see the table).

**Left out:**
- `GA`: presented as a general framework; its real-valued examples (getting-started on the Sphere, `test/rastrigin.jl`) put it after the ES and CMA-ES.
- `TreeGP`: genetic programming.

**Separate tests:**

| Scenario | Solver | Runs | Reached | Median first hit (evaluations) | Best value: median (best, worst) | Capped | Restarts (all runs) |
|---|---|---|---|---|---|---|---|
| Rastrigin 10 | cma_es | 5 | 0 | - | 52.7 (37.8, 57.7) | 0 | 134 |
| Rastrigin 10 | de | 5 | 5 | 137,414 | 0.00948 (0.00827, 0.00957) | 0 | 22 |
| Rastrigin 10 | es | 5 | 2 | 257,606 | 0.995 (0.00528, 12.7) | 0 | 14 |
| Rastrigin 30 | cma_es | 5 | 0 | - | 286.3 (192.2, 317.2) | 4 | 56 |
| Rastrigin 30 | de | 5 | 0 | - | 142.1 (122.5, 149.7) | 0 | 741 |
| Rastrigin 30 | es | 5 | 0 | - | 9.88 (3.98, 16) | 5 | 1 |
| Ackley 30 | cma_es | 5 | 0 | - | 19.7 (19.4, 19.8) | 0 | 53 |
| Ackley 30 | de | 5 | 0 | - | 2.8 (1.07, 3.43) | 0 | 157 |
| Ackley 30 | es | 5 | 5 | 12,969 | 0.009 (0.0085, 0.0092) | 0 | 0 |

## Continuous, unimodal: Rosenbrock 10

**Methods** (`real_solvers`): CMA-ES and DE as above, and the GA, the docs' own example for a unimodal function, each with the docs' example settings:
- **`cma_es`:** the [tutorial](https://docs.sciml.ai/Evolutionary/stable/tutorial/) on Rosenbrock: `CMAES()`, a (10,20)-CMA-ES, σ0 0.5, from a random point.
- **`de`:** `DE(populationSize = 100)`, as in `test/rosenbrock.jl`: DE/rand/1/bin, F = 0.9, Cr = 0.5.
- **`ga`:** the [getting-started example](https://docs.sciml.ai/Evolutionary/stable/#Getting-started) on the Sphere: `GA(populationSize = 100, selection = susinv, crossover = DC, mutation = PLM())`, default rates (0.8, 0.1).

**Keeping going:** all three converge and restart.

**Left out:** the ES: `test/rosenbrock.jl` starts with it, but the published docs' examples show CMA-ES and the GA.

**Separate tests:**

| Scenario | Solver | Runs | Reached | Median first hit (evaluations) | Best value: median (best, worst) | Capped | Restarts (all runs) |
|---|---|---|---|---|---|---|---|
| Rosenbrock 10 | cma_es | 5 | 0 | - | 1.19 (0.0312, 1.39) | 0 | 62 |
| Rosenbrock 10 | de | 5 | 0 | - | 3.47 (0.118, 5.31) | 0 | 275 |
| Rosenbrock 10 | ga | 5 | 0 | - | 8.18 (6.52, 9.02) | 0 | 75 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1 (matched)

**Methods:** [`NSGA2`](https://docs.sciml.ai/Evolutionary/stable/moea/), the only multi-objective algorithm, with the matched settings (`run_front`, [lines 418-441](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L418-L441)): population 100 (92), SBX η 15 at 0.9, polynomial mutation η 20 at 1/n, binary tournament on rank and crowding distance (the default); objectives in the tutorial's in-place form `f!(F, x)` ([tutorial](https://docs.sciml.ai/Evolutionary/stable/tutorial/#Objective-Function-Definition)). Differences:
- `SBX` and `PLM` are unbounded; the box constraints clip the offspring to [0, 1].
- `SBX` crosses each variable with probability 0.5, as pymoo's.
- An uncrossed pair passes the parents themselves, which `PLM` then mutates in place (see [Bugs found](#bugs-found)), included (rule 6.1).

**Keeping going:** NSGA2's `GD` test is lifted; one attempt.

**The front:** the non-dominated part of the final population, which NSGA2 leaves in the vector passed to `optimize`.

**Left out:** no other multi-objective algorithm exists.

**Separate tests** (hypervolume by `run.py`'s code):

| Scenario | Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Capped |
|---|---|---|---|---|---|
| ZDT1 (25,000) | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 25,000 | 0 |
| ZDT2 (25,000) | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 25,000 | 0 |
| ZDT3 (25,000) | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 25,000 | 0 |
| DTLZ2 (25,000) | nsga2 | 5 | 0.2835 (0.3162, 0.2302) | 25,024 | 0 |
| DTLZ1 (40,000) | nsga2 | 5 | 0.0000 (0.0000, 0.0000) | 40,020 | 0 |

## Can't run

Nothing: every scenario runs.

## Bugs found

None is worked around.
- **NSGA2 selects on other individuals' values** ([SciML/Evolutionary.jl#174](https://github.com/SciML/Evolutionary.jl/issues/174)). `update_state!` (`src/nsga2.jl`) reorders the parents (`parents .= state.population[fitidx]`) but not their objective values in `state.fitpop[:, 1:N]`, ranks or crowding distances, so from the second generation on it selects on other individuals' values. Effect: the multi-objective results above.
- **CMA-ES doesn't converge in 30 dimensions.** On a 30-dimensional sphere centred at 3, from a random start in [−5.12, 5.12]³⁰ with those bounds, `CMAES()` ends at 95.8 after 60,000 evaluations and `CMAES(lambda = 100)` at 227 after 300,000; without bounds, from [0, 1]³⁰, at 17.6 and 19.8. pycma 4.5.0 with the same σ0, start and bounds reaches 1e-10 in 5,180 and 5,446 evaluations (seeds 1 and 2). In 10 dimensions it converges. The step-size update in `src/cmaes.jl` uses ‖s_σ‖/N where Hansen's tutorial form it follows has ‖s_σ‖²/N, but squaring alone doesn't fix the runs (they then diverge with bounds, and reach 0.25 and 2.1 without). Effect: no multimodal target reached, and occasional covariance breakdowns. Not reported upstream yet.
- **GA offspring share their parents** (`recombine!`, `src/ga.jl`, also used by `NSGA2`). An uncrossed pair gets the parents themselves, and `mutate!` mutates them in place: two offspring become one object, and with elitism (ε > 0) an elite can be mutated. With the N-Queens settings, a median of 1 individual per generation is the same object as another. Effect: every GA and NSGA2 run here. Not reported upstream yet.
- **The (μ+λ)-ES can lose a surviving parent** (`update_state!`, `src/es.jl`): it writes each surviving offspring into the slot of its rank, which can overwrite a surviving parent. A (15/3+100)-ES on a sphere, from 2,000 random starts, lost its best parent in 3 of 40,000 generations. Not reported upstream yet.
