# Metaheuristics.jl (Julia, 3.5.0)

Metaheuristics.jl, by Jesús-Adolfo Mejía-de-Dios, is a Julia package of single- and multi-objective metaheuristics: ECA, DE, PSO, ABC, SA, SHADE and others for real numbers, a GA framework for binary, permutation and real encodings, BRKGA, and NSGA-II, NSGA-III, SPEA2, SMS-EMOA, MOEA/D-DE and CCMO. Its docs are at [jmejia8.github.io/Metaheuristics.jl](https://jmejia8.github.io/Metaheuristics.jl/stable/), from `docs/src` of the package. Its [algorithms index](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/) has a "Quick Selection Guide" by problem type, "the guide" below.

Adapter: [benchmarks/adapters/metaheuristics_jl/](../../../benchmarks/adapters/metaheuristics_jl/).
Know a better way to solve one of these problems with Metaheuristics.jl? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs Metaheuristics.jl

- **Evaluations:** `counted` and `counted_front` count every call; `counted` records the first hit ([bench.jl, lines 217-242](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L217-L242)).
- **Stop:** each attempt gets the rest of the budget as `f_calls_limit` and of the cap as `time_limit`; a user-defined `BudgetTermination` ([lines 211-214](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L211-L214)) ends the run at the target, the budget or the cap after every iteration.
- **Keeping going (rule 2.2)** (`options`, `algorithm_kwargs`, `run_restarting`, [lines 244-289](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L244-L289)):
  - the iteration limit is lifted;
  - the library's convergence criteria count, with the default tolerances (`f_tol` 1e-12, `f_tol_rel` eps, `x_tol` 1e-8): `default_stop_check`'s `CheckConvergence` (`src/termination/default.jl`), which needs all of `AbsoluteFunctionConvergence`, `RelativeFunctionConvergence`, `SmallStandardDeviation` and `RelativeParameterConvergence`, and the one `optimize` adds without a user termination (`src/optimize/before.jl`): `CheckConvergence` for one objective, `RobustConvergence(ftol = f_tol)` for several. The documented budget (`Options(f_calls_limit = …)`, [FAQ](https://jmejia8.github.io/Metaheuristics.jl/stable/faq/#How-to-set-stopping-criteria?)) leaves them in effect;
  - a converged attempt restarts from a new random start with the seeds of rule 2.2; runs print `restarts`. The library's [`Restart`](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/#Restart) replaces the population every 100 iterations whatever happens and keeps the base method's stops, so it isn't used;
  - the matched (multi-objective) scenarios have no convergence criterion: `BudgetTermination` only, and `f_tol = -1`.
- **Bounds (rule 2.4):** `boxconstraints` for the continuous problems and BRKGA's random keys; the initial population within the bounds; repairs by `evo_boundary_repairer!` (ECA, DE: `DE.jl` line 205) and `reset_to_violated_bounds!` (PSO; the multi-objective algorithms through `GA_reproduction` in `NSGA2.jl`, and `SMS_EMOA.jl`).
- **Rule 5.3:** `EARLY_SEEDS` ([line 411](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L411)) and `main`.
- **Time:** from the run's `Budget`, before `optimize` creates the initial population. Warm-up as rule 4.2.
- **One thread:** [run.sh](../../../benchmarks/adapters/metaheuristics_jl/run.sh) runs Julia 1.13 with `--threads=1 --gcthreads=1,0` and BLAS with one thread.
- **Seeds:** `Options(seed = seed)`, which seeds Julia's global generator, the library's own (`default_rng_mh`).
- **Choosing among the docs (rule 6.2):** the guide's first recommendations for each problem type, with the docs' example settings or the defaults; where it lists several without preference ("ECA, DE, PSO, or SHADE"), the first listed.
- **Separate tests:** 2026-09-25, Metaheuristics.jl 3.5.0, seeds 0 to 4, the scenario's budget, 60 s cap. `outside` was 0 in every run.

## Binary: OneMax 100 (idiomatic)

**Methods** (`onemax_solvers`, [lines 298-312](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L298-L312)): the guide: "Binary: Use GA with BitFlipMutation". The [GA docstring](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/#GA)'s binary example: `GA()` on a `BitArraySpace` with its defaults: population 100, binary tournament, uniform crossover at 0.5, `BitFlipMutation` at 1e-5, `ElitistReplacement`.

**Keeping going:** no attempt converged here.

**Left out:** `MCCGA`, a compact GA "for real-valued optimization problems".

**Not run: matched OneMax 100 and 1000** (rule 6.1): no two-point crossover (only `UniformCrossover`, `OrderCrossover`, `SBX`, `BinomialCrossover`, `src/operators/crossover/`).

**Separate tests:**

| Scenario | Solver | Runs | Reached | Median first hit (evaluations) | Best value: median (best, worst) | Capped | Restarts (all runs) |
|---|---|---|---|---|---|---|---|
| OneMax 100, idiomatic | ga | 5 | 5 | 1,878 | 100 (100, 100) | 0 | 0 |

## Permutation: N-Queens 32 and 64

**Methods** (`nqueens_solvers`, [lines 314-342](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L314-L342)): the guide: "Permutation-based: Use GA with OrderCrossover or BRKGA".
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

**Methods** (`real_solvers`, [lines 344-366](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L344-L366)): the guide: "Box-constrained (continuous): Use ECA, DE, PSO, or SHADE"; the [FAQ](https://jmejia8.github.io/Metaheuristics.jl/stable/faq/#How-to-choose-between-algorithms?): "ECA, DE, PSO are good starting points". The first three, with their defaults ([docstrings](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/)):
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

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1 (matched)

**Methods** (`front_solvers`, [lines 368-402](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L368-L402)): NSGA-II, NSGA-III, SPEA2 and SMS-EMOA with the matched settings; the guide recommends NSGA2, SPEA2 or SMS_EMOA for 2 and 3 objectives. Objectives are returned as `(f, [0.0], [0.0])`, the unconstrained form of the [docstrings](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/multiobjective/).
- **`nsga2`, `spea2`, `sms_emoa`:** population 100 (92), SBX η 15 with `p_cr = 0.9`, polynomial mutation η 20 with `p_m = 1/n`.
- **`nsga3`:** population 100 (92), Das-Dennis directions with 99 (12) partitions, SBX η 30 with `p_cr = 1`, the same mutation.

Differences:
- `p_cr` is per variable, and every pair is crossed (pymoo: pairs at 0.9, variables at 0.5).
- NSGA-II and SPEA2 create 2N children per generation. SMS-EMOA creates one per step, N per iteration.
- With 3 objectives, SMS-EMOA estimates hypervolume contributions by Monte Carlo, `n_samples = 10,000` by default, for every child.

**Keeping going:** one attempt to the budget.

**The front** (rule 7.2): the non-dominated part of `status.population` (92 or 100).

**Left out (rule 6.1):**
- `MOEAD_DE`: its reproduction, DE/rand/1 (F 0.5, CR 1) with polynomial mutation, is fixed in `MOEAD_DE_reproduction`, so it can't use the matched SBX; it also has only Tchebycheff and at most 2 replacements.
- `CCMO`: for constrained problems.

**Separate tests** (hypervolume by `run.py`'s code):

| Scenario | Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Capped |
|---|---|---|---|---|---|
| ZDT1 (25,000) | nsga2 | 5 | 0.8663 (0.8672, 0.8654) | 25,100 | 0 |
| ZDT1 (25,000) | nsga3 | 5 | 0.8675 (0.8679, 0.8664) | 25,000 | 0 |
| ZDT1 (25,000) | spea2 | 5 | 0.8658 (0.8662, 0.8645) | 25,100 | 0 |
| ZDT1 (25,000) | sms_emoa | 5 | 0.8719 (0.8720, 0.8719) | 25,000 | 0 |
| ZDT2 (25,000) | nsga2 | 5 | 0.5336 (0.5348, 0.5334) | 25,100 | 0 |
| ZDT2 (25,000) | nsga3 | 5 | 0.5344 (0.5363, 0.5338) | 25,000 | 0 |
| ZDT2 (25,000) | spea2 | 5 | 0.5306 (0.5317, 0.5294) | 25,100 | 0 |
| ZDT2 (25,000) | sms_emoa | 5 | 0.5387 (0.5387, 0.5386) | 25,000 | 0 |
| ZDT3 (25,000) | nsga2 | 5 | 1.3229 (1.3243, 1.3226) | 25,100 | 0 |
| ZDT3 (25,000) | nsga3 | 5 | 1.3221 (1.3234, 1.3197) | 25,000 | 0 |
| ZDT3 (25,000) | spea2 | 5 | 1.3148 (1.3175, 1.3112) | 25,100 | 0 |
| ZDT3 (25,000) | sms_emoa | 5 | 1.3291 (1.3293, 1.3289) | 25,000 | 0 |
| DTLZ2 (25,000) | nsga2 | 5 | 0.6259 (0.6479, 0.6012) | 25,116 | 0 |
| DTLZ2 (25,000) | nsga3 | 5 | 0.7385 (0.7395, 0.7374) | 25,024 | 0 |
| DTLZ2 (25,000) | spea2 | 5 | 0.7358 (0.7380, 0.7275) | 25,116 | 0 |
| DTLZ2 (25,000) | sms_emoa | 3 | 0.7146 (0.7156, 0.7015) | 1,932 | 3 |
| DTLZ1 (40,000) | nsga2 | 5 | 0.1222 (0.1242, 0.0971) | 40,020 | 0 |
| DTLZ1 (40,000) | nsga3 | 5 | 0.1194 (0.1385, 0.1110) | 40,020 | 0 |
| DTLZ1 (40,000) | spea2 | 5 | 0.1376 (0.1389, 0.1371) | 40,020 | 0 |
| DTLZ1 (40,000) | sms_emoa | 3 | 0.0063 (0.0311, 0.0000) | 8,832 | 3 |

With 3 objectives, SMS-EMOA recomputes the Monte Carlo estimate from scratch for every child (`calculate_hv`, `SMS_EMOA.jl`; `update_population!`, `update-population.jl`): a 10,000 × 3 matrix per member of the last front. One call takes 1.4 ms for 10 points, 5.3 ms for 50 and 11.3 ms (12.7 MiB) for 93. On DTLZ2 the whole population is one front after 3,036 evaluations, so 25,000 children would take about 5 minutes. A ZDT run, with exact contributions, takes about 1.3 s. `n_samples` stays at its documented default (rule 6.3).

## Can't run

- Matched OneMax 100 and 1000: no two-point crossover.
- MOEA/D in the matched multi-objective scenarios: only MOEA/D-DE.

## Bugs found

- **The bounded SBX computes the second child's spread from the lower bound** (`SBX_crossover`, `src/operators/crossover/sbx.jl`): both children use β = 1 + 2 (y1 − xl) / Δ, where Deb's bounded SBX (and pymoo's) uses β = 1 + 2 (xu − y2) / Δ for the second. All the multi-objective algorithms and the real-valued GA use it. Found by reading the code; the effect wasn't measured. Not reported upstream yet; not worked around.
- **Documentation:** the `DE` docstring gives F = 1.0 as the default; the code's is F = 0.7 (`DE.jl`), which the adapter uses.
