# Metaheuristics.jl (Julia, 3.5.0)

Metaheuristics.jl, by Jesús-Adolfo Mejía-de-Dios, is a Julia package of single- and multi-objective metaheuristics: ECA, DE, PSO, ABC, SA, SHADE and others for real numbers, a GA framework for binary, permutation and real encodings, BRKGA, and NSGA-II, NSGA-III, SPEA2, SMS-EMOA, MOEA/D-DE and CCMO. Its documentation is at [jmejia8.github.io/Metaheuristics.jl](https://jmejia8.github.io/Metaheuristics.jl/stable/), built from `docs/src` of the package. Its [algorithms index](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/) has a "Quick Selection Guide" by problem type, called "the guide" below.

Adapter: [benchmarks/adapters/metaheuristics_jl/](../../../benchmarks/adapters/metaheuristics_jl/).
Know a better way to solve one of these problems with Metaheuristics.jl? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How every run is set up

- **Evaluations** (rule 3): `counted` and `counted_front` ([bench.jl](../../../benchmarks/adapters/metaheuristics_jl/bench.jl), lines 186-208) count every call of the fitness function.
- **Stops and restarts** (rule 2.2), `options`, `algorithm_kwargs` and `run_restarting` (lines 210-256):
  - Each attempt gets what is left of the budget as `f_calls_limit` and of the cap as `time_limit`, and a user-defined termination criterion, `BudgetTermination` (lines 181-184), ends the run at the target, the budget or the cap, after every iteration.
  - The iteration limit is only a budget, so it's lifted.
  - The convergence criteria are the library's, with the default tolerances (`f_tol` 1e-12, `f_tol_rel` eps, `x_tol` 1e-8): the one `optimize` always checks (`default_stop_check`, `src/termination/default.jl`: `CheckConvergence`, which needs all of `AbsoluteFunctionConvergence`, `RelativeFunctionConvergence`, `SmallStandardDeviation` and `RelativeParameterConvergence`), and the one it adds when the user gives no termination (`src/optimize/before.jl`): the same `CheckConvergence` for one objective, `RobustConvergence(ftol = f_tol)` for several. The adapter passes both, next to `BudgetTermination`. These criteria count: setting the budget the documented way (`Options(f_calls_limit = …)`, [FAQ](https://jmejia8.github.io/Metaheuristics.jl/stable/faq/#How-to-set-stopping-criteria?)) leaves them in effect alongside it.
  - These apply in the idiomatic scenarios. The matched scenarios (the matched OneMax and the multi-objective ones) run the matched configuration, which has no convergence criterion: they get `BudgetTermination` only and `f_tol = -1`, which makes `default_stop_check`'s `CheckConvergence` impossible, and run to the target or the budget in one attempt.
  - An attempt that converges starts again from a new random start, with the seed `seed * 1000 + restart`, keeping the best solution and every evaluation. Each run prints its `restarts`. The library's [`Restart`](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/#Restart) isn't a restart after convergence: it replaces the population every 100 iterations whatever happens, and keeps the base method's stops (`src/algorithms/singleobjective/Restart/Restart.jl`), so it isn't used.
- **Bounds** (rule 2.4): the continuous problems and the random keys of BRKGA use `boxconstraints`; the initial population is drawn within the bounds, and each method repairs its new solutions: ECA with `evo_boundary_repairer!`, DE, PSO and the multi-objective algorithms with `reset_to_violated_bounds!` (`src/common/repair.jl`). The adapter counts the evaluated solutions outside the bounds (`outside`): 0 in every run.
- **Early stop** (rule 5.3): a solver whose first 3 seeds all hit the 60 s cap runs no more seeds (`EARLY_SEEDS`, line 424, and `main`).
- **Time** (rule 4): the clock starts before `optimize` creates the initial population. Each solver first makes an untimed warm-up run of the same problem with 1,000 evaluations and seed 1000.
- **One thread** (rule 4.3): [run.sh](../../../benchmarks/adapters/metaheuristics_jl/run.sh) runs Julia 1.13 with `--threads=1 --gcthreads=1,0` (one GC mark thread, no concurrent sweep thread) and BLAS with one thread. `run.py check` measured CPU/wall 1.00.
- **Seeds** (rule 5.2): `Options(seed = seed)`. `optimize` seeds Julia's global generator with it, and the library's own generator (`default_rng_mh`) is that same global generator, so the adapter's operators (which call `rand()`) are seeded too.
- **How the documentation decides** (rule 6.2): the guide states a preference for each problem type, and the adapter takes its first recommendations, with the settings of the documentation's example for the problem type, or the defaults. Where the guide lists several methods without a preference among them ("ECA, DE, PSO, or SHADE"), the first ones it lists are taken (a tie, rule 6.2).

## Binary: OneMax 100 and 1000 (matched), OneMax 100 (idiomatic)

**Methods** (`onemax_solvers`, lines 309-334):
- Matched: the library's `GA` framework with DEAP's `eaSimple` settings: population 300, `TournamentSelection(K = 3)`, two-point crossover at 0.5, bit-flip at 1/n on 20% of the children, `GenerationalReplacement` (no elitism). The GA framework dispatches on operator types ([tutorial](https://jmejia8.github.io/Metaheuristics.jl/stable/tutorials/create-metaheuristic/)), and the library has no two-point crossover and no per-child mutation probability, so the adapter adds them (`TwoPointCrossover`, `BitFlipSomeChildren`, lines 264-300). Its GA pairs the selected parents (i, i + N/2) instead of neighbours.
- Idiomatic: the guide: "Binary: Use GA with BitFlipMutation". The binary example of the [GA docstring](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/#GA) runs `GA()` on a `BitArraySpace` with its defaults: population 100, binary tournament, uniform crossover at 0.5, `BitFlipMutation` at 1e-5, `ElitistReplacement`.

**Keeping going:** matched: no convergence criterion, it runs to the target or the budget. Idiomatic: the convergence criteria never ended an attempt here.

**Left out:** `MCCGA`, a compact GA "for real-valued optimization problems"; nothing else is presented for bits.

**Separate tests** (2026-09-25, Metaheuristics.jl 3.5.0, seeds 0 to 4, the scenario's budget, 60 s cap):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap | Restarts (5 runs) |
|---|---|---|---|---|---|---|---|
| OneMax 100, matched | ga | 5 | 5 | 9,900 | 100 (100, 100) | 0 | 0 |
| OneMax 1000, matched | ga | 5 | 5 | 170,100 | 1000 (1000, 1000) | 0 | 0 |
| OneMax 100, idiomatic | ga | 5 | 5 | 1,900 | 100 (100, 100) | 0 | 0 |

## Permutation: N-Queens 32 and 64

**Methods** (`nqueens_solvers`, lines 336-361): the guide: "Permutation-based: Use GA with OrderCrossover or BRKGA".
- `ga`: the [N-Queens tutorial](https://jmejia8.github.io/Metaheuristics.jl/stable/tutorials/n-queens/) runs `optimize(attacks, PermutationSpace(N), GA)`, the GA's defaults for permutations (`get_parameters` in `src/algorithms/singleobjective/GA/GA.jl`): population 100, binary tournament, `OrderCrossover`, `SlightMutation`, `ElitistReplacement`. The adapter builds the same GA explicitly, to pass the options.
- `brkga`: [`BRKGA`](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/combinatorial/#BRKGA) with its defaults (20 elites, 10 mutants, 70 offspring, bias 0.7), on random keys in [0, 1]ⁿ decoded by `sortperm`, as in its docstring's permutation example. The reported solution is the decoded permutation.

**Keeping going:** both converge and restart: the GA when its whole population has the same number of conflicts, and the same best and worst permutation; BRKGA very often, as its population soon has one value (see the table).

**Left out:** `GRASP`, `VNS` and `VND` need a problem-specific constructor or neighbourhood structures from the user (their docstrings), and the guide doesn't list them for permutations.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap | Restarts (5 runs) |
|---|---|---|---|---|---|---|---|
| N-Queens 32 | ga | 5 | 2 | 291,550 | 1 (0, 1) | 0 | 117 |
| N-Queens 32 | brkga | 5 | 0 | – | 4 (3, 5) | 0 | 3,554 |
| N-Queens 64 | ga | 5 | 0 | – | 4 (3, 4) | 0 | 135 |
| N-Queens 64 | brkga | 5 | 0 | – | 14 (14, 15) | 0 | 6,834 |

The best values are the numbers of diagonal conflicts left.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** (`real_solvers`, lines 363-382): the guide: "Box-constrained (continuous): Use ECA, DE, PSO, or SHADE"; the [FAQ](https://jmejia8.github.io/Metaheuristics.jl/stable/faq/#How-to-choose-between-algorithms?): "Single-objective, unconstrained: ECA, DE, PSO are good starting points". The first three, with their defaults ([docstrings](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/)), on a `boxconstraints` search space:
- `eca`: ECA, the default method of `optimize` and the one the [Quick Start](https://jmejia8.github.io/Metaheuristics.jl/stable/#Quick-Start) runs on Rastrigin. K = 7, population K·D, η_max = 2, p_exploit = 0.95, p_bin = 0.02. It switches to exploitation after 95% of `f_calls_limit`, which is what is left of the scenario's budget when the attempt starts.
- `de`: DE/rand/1/bin, population 10·D, F = 0.7, CR = 0.5. The guide lists DE as "Good for multimodal".
- `pso`: population 10·D, C1 = C2 = 2, ω = 0.8. The guide lists PSO as "Good for multimodal".

**Keeping going:** ECA converges and restarts on Rastrigin, and PSO on Rastrigin 10 (see the table); the others run to the target or the budget in one attempt.

**Left out:**
- `SHADE`: fourth in the guide's list, and rule 6.4 allows three.
- `ABC`: listed as "Good for multimodal", but not in the guide's list for box-constrained problems, nor in the FAQ's.
- `GA` with SBX and polynomial mutation: its docstring has a real-encoding example on Rastrigin, but neither the guide nor the FAQ lists the GA for continuous problems. For reference, the 0.6.0 benchmark ran it (the defaults for real numbers: population 100, SBX, polynomial mutation, elitist replacement) and it reached the target in 10 of 10 runs on Rastrigin 10 and 30 and Ackley 30, and 0 of 10 on Rosenbrock 10: better than ECA, DE and PSO on the multimodal problems. The documentation decides (rule 6.2), and it doesn't recommend the GA for continuous problems; if you read it otherwise, open an issue.
- `CGSA`, `SA`, `WOA`, `MCCGA`: in the algorithm list, but not in the guide's recommendations; `CSO` is for large-scale problems; `εDE` for constrained ones.
- A CMA-ES: the library has none.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap | Restarts (5 runs) |
|---|---|---|---|---|---|---|---|
| Rastrigin 10 | eca | 5 | 3 | 91,630 | 0.00854 (0.00648, 1.99) | 0 | 19 |
| Rastrigin 10 | de | 5 | 5 | 161,500 | 0.00776 (0.00257, 0.00962) | 0 | 0 |
| Rastrigin 10 | pso | 5 | 0 | – | 6.97 (6.04, 10.94) | 0 | 5 |
| Rastrigin 30 | eca | 5 | 0 | – | 2.985 (1.99, 5.97) | 0 | 22 |
| Rastrigin 30 | de | 5 | 0 | – | 115.3 (105.8, 122.5) | 0 | 0 |
| Rastrigin 30 | pso | 5 | 0 | – | 41.79 (29.85, 46.76) | 0 | 0 |
| Ackley 30 | eca | 5 | 5 | 70,980 | 0.00967 (0.00834, 0.00989) | 0 | 0 |
| Ackley 30 | de | 5 | 5 | 497,700 | 0.00943 (0.00899, 0.00992) | 0 | 0 |
| Ackley 30 | pso | 5 | 0 | – | 2.12 (1.90, 2.50) | 0 | 0 |

## Continuous, unimodal: Rosenbrock 10

**Methods:** the same three, ECA, DE and PSO, with their defaults, for the same reasons: the guide recommends them for box-constrained problems without distinguishing unimodal ones, and lists ECA under "Fast convergence".

**Keeping going:** no attempt converged before the target here.

**Left out:** as above.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap | Restarts (5 runs) |
|---|---|---|---|---|---|---|---|
| Rosenbrock 10 | eca | 5 | 5 | 18,130 | 0.00759 (0.00614, 0.00918) | 0 | 0 |
| Rosenbrock 10 | de | 5 | 5 | 196,400 | 0.00842 (0.00751, 0.00968) | 0 | 0 |
| Rosenbrock 10 | pso | 5 | 5 | 96,200 | 0.00881 (0.00854, 0.00989) | 0 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1 (matched)

**Methods** (`front_solvers`, lines 384-416): the library's NSGA-II, NSGA-III, SPEA2 and SMS-EMOA, with the matched settings (rule 6.1). The guide recommends NSGA2, SPEA2 or SMS_EMOA for 2 and 3 objectives and NSGA3 for more; the objectives are returned as `(f, [0.0], [0.0])`, the unconstrained form of the [docstrings](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/multiobjective/).
- `nsga2`, `spea2`, `sms_emoa`: population 100 (92 with 3 objectives), SBX with η 15 and `p_cr = 0.9`, polynomial mutation with η 20 and `p_m = 1/n`.
- `nsga3`: population 100 (92), Das-Dennis directions with 99 (12) partitions, SBX with η 30 and `p_cr = 1`, polynomial mutation with η 20 and `p_m = 1/n`.

Differences from the matched settings:
- `p_cr` is the probability of crossing each variable, and every pair is crossed (pymoo: pairs at 0.9, variables at 0.5).
- NSGA-II and SPEA2 create 2N children per generation, not N. SMS-EMOA is steady-state: one child at a time, N per iteration.
- With 3 objectives, SMS-EMOA estimates the hypervolume contributions by Monte Carlo, with its default `n_samples = 10,000` samples, for every child.

**The front** (rule 7.2): the non-dominated part of the final population (`status.population`), whose size is the scenario's (92 or 100), with the objective values the library computed.

**Keeping going:** the matched configuration has no convergence criterion, so `RobustConvergence` isn't given and `f_tol = -1`: each run goes to the budget in one attempt.

**Not run** (rule 6.1):
- `MOEAD_DE`, the library's MOEA/D: it is MOEA/D-DE, whose reproduction, DE/rand/1 (F 0.5, CR 1) with polynomial mutation, is fixed in `MOEAD_DE_reproduction` (`src/algorithms/multiobjective/MOEAD_DE/MOEAD_DE.jl`), so it can't use the matched SBX. It also replaces at most 2 neighbours per child, and has only Tchebycheff. The 0.6.0 benchmark ran it.
- `CCMO`, which the guide recommends for constrained multi-objective problems.

**Separate tests** (2026-09-25; hypervolume computed by `run.py`'s code from the reported solutions):

| Scenario | Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|---|
| ZDT1 | nsga2 | 5 | 0.8663 (0.8672, 0.8654) | 25,100 | 0 |
| ZDT1 | nsga3 | 5 | 0.8675 (0.8679, 0.8664) | 25,000 | 0 |
| ZDT1 | spea2 | 5 | 0.8658 (0.8662, 0.8645) | 25,100 | 0 |
| ZDT1 | sms_emoa | 5 | 0.8719 (0.8720, 0.8719) | 25,000 | 0 |
| ZDT2 | nsga2 | 5 | 0.5336 (0.5348, 0.5334) | 25,100 | 0 |
| ZDT2 | nsga3 | 5 | 0.5344 (0.5363, 0.5338) | 25,000 | 0 |
| ZDT2 | spea2 | 5 | 0.5306 (0.5317, 0.5294) | 25,100 | 0 |
| ZDT2 | sms_emoa | 5 | 0.5387 (0.5387, 0.5386) | 25,000 | 0 |
| ZDT3 | nsga2 | 5 | 1.3229 (1.3243, 1.3226) | 25,100 | 0 |
| ZDT3 | nsga3 | 5 | 1.3221 (1.3234, 1.3197) | 25,000 | 0 |
| ZDT3 | spea2 | 5 | 1.3148 (1.3175, 1.3112) | 25,100 | 0 |
| ZDT3 | sms_emoa | 5 | 1.3291 (1.3293, 1.3289) | 25,000 | 0 |
| DTLZ2 | nsga2 | 5 | 0.6259 (0.6479, 0.6012) | 25,116 | 0 |
| DTLZ2 | nsga3 | 5 | 0.7385 (0.7395, 0.7374) | 25,024 | 0 |
| DTLZ2 | spea2 | 5 | 0.7358 (0.7380, 0.7275) | 25,116 | 0 |
| DTLZ2 | sms_emoa | 3 | 0.7364 (0.7376, 0.7337) | 8,832 | 3 |
| DTLZ1 | nsga2 | 5 | 1.2708 (1.2740, 1.1927) | 40,020 | 0 |
| DTLZ1 | nsga3 | 5 | 1.2832 (1.3024, 1.2756) | 40,020 | 0 |
| DTLZ1 | spea2 | 5 | 1.3020 (1.3035, 1.3009) | 40,020 | 0 |
| DTLZ1 | sms_emoa | 3 | 1.2684 (1.2981, 1.1066) | 17,756 | 3 |

**Why SMS-EMOA hits the 60 s cap on DTLZ2 and DTLZ1.** It's the library, not its use. SMS-EMOA is steady-state: every child is added to the population, which is sorted again, and the child or a member of the last front is removed by its hypervolume contribution (`update_population!`, `src/algorithms/multiobjective/SMS_EMOA/update-population.jl`). With 2 objectives, the contributions are computed exactly, and a ZDT run takes about 1.3 s. With 3 objectives, `calculate_hv` (`SMS_EMOA.jl`) estimates them by Monte Carlo with `n_samples` = 10,000 points, from scratch, for every child: it builds a 10,000 × 3 matrix per member of the last front. Measured on its own, one call takes 1.4 ms for a front of 10 points, 5.3 ms for 50 and 11.3 ms for 93, and allocates 12.7 MiB for 93. On DTLZ2, the whole population of 92 is already one front after 3,036 evaluations, so every child costs about 11 ms, and 25,000 children would take about 5 minutes; the runs reach a median of 8,832 evaluations in 60 s, and on DTLZ1 17,756 (between 5,980 and 7,912, and 14,628 and 15,364, in earlier rounds of the same tests: the machine was shared with other runs, so these numbers vary). `n_samples` is a documented parameter, but its default is the documented setting, and lowering it would be tuning. The 0.6.0 results showed the same (9,016 and 20,056 evaluations).

## Can't run

Every scenario runs. The matched multi-objective scenarios run without MOEA/D, which the library has only as MOEA/D-DE (see above).

## Bugs found

- **The bounded SBX computes the second child's spread from the lower bound** (`SBX_crossover`, `src/operators/crossover/sbx.jl`): both children use β = 1 + 2 (y1 − xl) / Δ, where Deb's bounded SBX (and pymoo's) uses β = 1 + 2 (xu − y2) / Δ for the second child. All the multi-objective algorithms and the real-valued GA use it. Found by reading the code; its effect on the results wasn't measured. Not reported upstream yet; not worked around.
- **Documentation:** the `DE` docstring gives F = 1.0 as the default; the code's default is F = 0.7 (`src/algorithms/singleobjective/DE/DE.jl`), which the adapter uses.
