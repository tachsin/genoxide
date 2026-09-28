# Metaheuristics.jl (Julia, 3.5.0)

Metaheuristics.jl, by Jesús-Adolfo Mejía-de-Dios, is a Julia package of single- and multi-objective metaheuristics: ECA, DE, PSO, ABC, SA, SHADE and others for real numbers, a GA framework for binary, permutation and real encodings, BRKGA, and NSGA-II, NSGA-III, SPEA2, SMS-EMOA, MOEA/D-DE and CCMO. Its docs are at [jmejia8.github.io/Metaheuristics.jl](https://jmejia8.github.io/Metaheuristics.jl/stable/), from `docs/src` of the package. Its [algorithms index](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/) has a "Quick Selection Guide" by problem type, "the guide" below.

Adapter: [benchmarks/adapters/metaheuristics_jl/](../../../benchmarks/adapters/metaheuristics_jl/).
Know a better way to solve one of these problems with Metaheuristics.jl? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs Metaheuristics.jl

- **Evaluations:** `counted` counts every call and records the first hit ([bench.jl, lines 156-171](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L156-L171)).
- **Stop:** each attempt gets the rest of the budget as `f_calls_limit` and of the cap as `time_limit`; a user-defined `BudgetTermination` ([lines 150-153](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L150-L153)) ends the run at the target, the budget or the cap after every iteration.
- **Keeping going (rule 2.2)** (`options`, `algorithm_kwargs`, `run_restarting`, [lines 173-212](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L173-L212)):
  - the iteration limit is lifted;
  - the library's convergence criteria count, with the default tolerances (`f_tol` 1e-12, `f_tol_rel` eps, `x_tol` 1e-8): `default_stop_check`'s `CheckConvergence` (`src/termination/default.jl`), which needs all of `AbsoluteFunctionConvergence`, `RelativeFunctionConvergence`, `SmallStandardDeviation` and `RelativeParameterConvergence`, and the one `optimize` adds without a user termination (`src/optimize/before.jl`): `CheckConvergence` for one objective. The documented budget (`Options(f_calls_limit = …)`, [FAQ](https://jmejia8.github.io/Metaheuristics.jl/stable/faq/#How-to-set-stopping-criteria?)) leaves them in effect;
  - a converged attempt restarts from a new random start with the seeds of rule 2.2; runs print `restarts`. The library's [`Restart`](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/#Restart) replaces the population every 100 iterations whatever happens and keeps the base method's stops, so it isn't used.
- **Bounds (rule 2.4):** `boxconstraints` for the continuous problems and BRKGA's random keys; the initial population within the bounds; repairs by `evo_boundary_repairer!` (ECA, DE: `DE.jl` line 205) and `reset_to_violated_bounds!` (PSO).
- **Rule 5.3:** `EARLY_SEEDS` ([line 298](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L298)) and `main`.
- **Time:** from the run's `Budget`, before `optimize` creates the initial population. Warm-up as rule 4.2.
- **One thread:** [run.sh](../../../benchmarks/adapters/metaheuristics_jl/run.sh) runs Julia 1.13 with `--threads=1 --gcthreads=1,0` and BLAS with one thread.
- **Seeds:** `Options(seed = seed)`, which seeds Julia's global generator, the library's own (`default_rng_mh`).
- **Choosing among the docs (rule 6.2):** the guide's first recommendations for each problem type, with the docs' example settings or the defaults; where it lists several without preference ("ECA, DE, PSO, or SHADE"), the first listed.
- **Separate tests:** 2026-09-25, Metaheuristics.jl 3.5.0, seeds 0 to 4, the scenario's budget, 60 s cap. `outside` was 0 in every run.

## Binary: OneMax 100 (idiomatic)

**Methods** (`onemax_solvers`, [lines 221-235](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L221-L235)): the guide: "Binary: Use GA with BitFlipMutation". The [GA docstring](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/#GA)'s binary example: `GA()` on a `BitArraySpace` with its defaults: population 100, binary tournament, uniform crossover at 0.5, `BitFlipMutation` at 1e-5, `ElitistReplacement`.

**Keeping going:** no attempt converged here.

**Left out:** `MCCGA`, a compact GA "for real-valued optimization problems".

**Not run: matched OneMax 100 and 1000** (rule 6.1): no two-point crossover (only `UniformCrossover`, `OrderCrossover`, `SBX`, `BinomialCrossover`, `src/operators/crossover/`).

**Separate tests:**

| Scenario | Solver | Runs | Reached | Median first hit (evaluations) | Best value: median (best, worst) | Capped | Restarts (all runs) |
|---|---|---|---|---|---|---|---|
| OneMax 100, idiomatic | ga | 5 | 5 | 1,878 | 100 (100, 100) | 0 | 0 |

## Permutation: N-Queens 32 and 64

**Methods** (`nqueens_solvers`, [lines 237-265](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L237-L265)): the guide: "Permutation-based: Use GA with OrderCrossover or BRKGA".
- **`ga`:** the [N-Queens tutorial](https://jmejia8.github.io/Metaheuristics.jl/stable/tutorials/n-queens/)'s `optimize(attacks, PermutationSpace(N), GA)`, the GA's permutation defaults (`get_parameters`, `GA.jl`): population 100, binary tournament, `OrderCrossover`, `SlightMutation`, `ElitistReplacement`, built explicitly to pass the options.
- **`brkga`:** [`BRKGA`](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/combinatorial/#BRKGA) with its defaults (20 elites, 10 mutants, 70 offspring, bias 0.7), random keys in [0, 1]ⁿ decoded by `sortperm`, as in its docstring; the decoded permutation is reported.

**Keeping going:** both converge and restart: the GA when the whole population has the same conflicts and best and worst permutation, BRKGA when its population has one value.

**Left out:** `GRASP`, `VNS`, `VND`: they need a problem-specific constructor or neighbourhoods, and the guide doesn't list them for permutations.

**Separate tests** (the best values are the diagonal conflicts left):

| Scenario | Solver | Runs | Reached | Median first hit (evaluations) | Best value: median (best, worst) | Capped | Restarts (all runs) |
|---|---|---|---|---|---|---|---|
| N-Queens 32 | ga | 5 | 2 | 131,248 | 1 (0, 1) | 0 | 95 |
| N-Queens 32 | brkga | 5 | 0 | - | 4 (4, 5) | 0 | 3,560 |
| N-Queens 64 | ga | 5 | 0 | - | 3 (3, 4) | 0 | 133 |
| N-Queens 64 | brkga | 5 | 0 | - | 15 (14, 15) | 0 | 6,873 |

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** (`real_solvers`, [lines 267-289](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L267-L289)): the guide: "Box-constrained (continuous): Use ECA, DE, PSO, or SHADE"; the [FAQ](https://jmejia8.github.io/Metaheuristics.jl/stable/faq/#How-to-choose-between-algorithms?): "ECA, DE, PSO are good starting points". The first three, with their defaults ([docstrings](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/)):
- **`eca`:** the default of `optimize`, run on Rastrigin in the [Quick Start](https://jmejia8.github.io/Metaheuristics.jl/stable/#Quick-Start): K = 7, population K·D, η_max = 2, p_exploit = 0.95, p_bin = 0.02. It switches to exploitation after 95% of `f_calls_limit`, the rest of the budget at the attempt's start.
- **`de`:** DE/rand/1/bin, population 10·D, F = 0.7, CR = 0.5 ("Good for multimodal").
- **`pso`:** population 10·D, C1 = C2 = 2, ω = 0.8 ("Good for multimodal").

**Keeping going:** attempts converge and restart (see the table).

**Left out:**
- `SHADE`: fourth in the guide's list.
- `ABC`: "Good for multimodal", but not in the guide's list for box-constrained problems nor the FAQ's.
- `GA` with SBX and polynomial mutation: its docstring has a Rastrigin example, but neither the guide nor the FAQ lists it for continuous problems.
- `CGSA`, `SA`, `WOA`, `MCCGA`: not in the guide's recommendations; `CSO` is for large-scale problems, `εDE` for constrained ones.
- CMA-ES: the library has none.

**Separate tests:**

| Scenario | Solver | Runs | Reached | Median first hit (evaluations) | Best value: median (best, worst) | Capped | Restarts (all runs) |
|---|---|---|---|---|---|---|---|
| Rastrigin 10 | eca | 5 | 4 | 135,736 | 0.00788 (0.00389, 0.995) | 0 | 12 |
| Rastrigin 10 | de | 5 | 5 | 166,009 | 0.00907 (0.00795, 0.00932) | 0 | 0 |
| Rastrigin 10 | pso | 5 | 0 | - | 11.9 (3.98, 17.9) | 0 | 3 |
| Rastrigin 30 | eca | 5 | 0 | - | 9.95 (8.95, 13.9) | 0 | 21 |
| Rastrigin 30 | de | 5 | 0 | - | 93.3 (86.8, 105.5) | 0 | 0 |
| Rastrigin 30 | pso | 5 | 0 | - | 90.5 (62.7, 101.5) | 0 | 0 |
| Ackley 30 | eca | 5 | 4 | 69,902 | 0.00978 (0.0091, 1.16) | 0 | 1 |
| Ackley 30 | de | 5 | 5 | 468,550 | 0.00989 (0.00929, 0.00999) | 0 | 0 |
| Ackley 30 | pso | 5 | 0 | - | 3.74 (3.52, 7.7) | 0 | 0 |

## Continuous, unimodal: Rosenbrock 10

**Methods:** ECA, DE and PSO with their defaults: the guide doesn't separate unimodal problems, and lists ECA under "Fast convergence".

**Keeping going** and **Left out:** as above.

**Separate tests:**

| Scenario | Solver | Runs | Reached | Median first hit (evaluations) | Best value: median (best, worst) | Capped | Restarts (all runs) |
|---|---|---|---|---|---|---|---|
| Rosenbrock 10 | eca | 5 | 5 | 18,070 | 0.00759 (0.00614, 0.00918) | 0 | 0 |
| Rosenbrock 10 | de | 5 | 5 | 196,333 | 0.00842 (0.00751, 0.00968) | 0 | 0 |
| Rosenbrock 10 | pso | 5 | 5 | 96,150 | 0.00881 (0.00854, 0.00989) | 0 | 0 |

## Can't run

- Matched OneMax 100 and 1000: no two-point crossover.

## Bugs found

- **Documentation:** the `DE` docstring gives F = 1.0 as the default; the code's is F = 0.7 (`DE.jl`), which the adapter uses.
