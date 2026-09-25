# Metaheuristics.jl (Julia, 3.5.0)

Metaheuristics.jl, by Jesús-Adolfo Mejía-de-Dios, is a Julia package of single- and multi-objective metaheuristics: ECA, DE, PSO, ABC, SA, SHADE and others for real numbers, a GA framework for binary, permutation and real encodings, BRKGA, and NSGA-II, NSGA-III, SPEA2, SMS-EMOA, MOEA/D-DE and CCMO. Its documentation is at [jmejia8.github.io/Metaheuristics.jl](https://jmejia8.github.io/Metaheuristics.jl/stable/), built from `docs/src` of the package. Its [algorithms index](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/) has a "Quick Selection Guide" by problem type, called "the guide" below.

Adapter: [benchmarks/adapters/metaheuristics_jl/](../../../benchmarks/adapters/metaheuristics_jl/).
Know a better way to solve one of these problems with Metaheuristics.jl? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How every run is set up

- **Evaluations** (rule 3): `counted` and `counted_front` ([bench.jl](../../../benchmarks/adapters/metaheuristics_jl/bench.jl), lines 182-201) count every call of the fitness function.
- **Stops** (rule 2): `options` (lines 203-224) gives the library the budget as `f_calls_limit` and the cap as `time_limit`, and a user-defined termination criterion, `BudgetTermination` (lines 177-180), ends the run at the target, the budget or the cap, after every iteration. The library's own stops are turned off:
  - its convergence stop (`default_stop_check`, `src/termination/default.jl`), which needs all of `AbsoluteFunctionConvergence(f_tol)`, `RelativeFunctionConvergence`, `SmallStandardDeviation` and `RelativeParameterConvergence`, can't pass with `f_tol = -1`;
  - its default termination criterion (`CheckConvergence` for one objective, `RobustConvergence` for several, `src/optimize/before.jl`) is only added when the user gives none;
  - the iteration limit is out of reach.
- **Restarts** (rule 2.2): none are needed: with these options, no method stops before the target, the budget or the cap. The library's [`Restart`](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/#Restart) isn't a restart after a stop: it replaces the population every 100 iterations whatever happens, and keeps the base method's stops (`src/algorithms/singleobjective/Restart/Restart.jl`). The guide doesn't recommend it for any problem type, so it isn't used.
- **Early stop** (rule 5.3): a solver whose first 3 seeds all hit the 60 s cap runs no more seeds (`EARLY_SEEDS`, line 389, and `main`).
- **Time** (rule 4): the clock starts before `optimize` creates the initial population. Each solver first makes an untimed warm-up run of the same problem with 1,000 evaluations and seed 1000.
- **One thread** (rule 4.3): [run.sh](../../../benchmarks/adapters/metaheuristics_jl/run.sh) runs Julia 1.13 with `--threads=1 --gcthreads=1,0` (one GC mark thread, no concurrent sweep thread) and BLAS with one thread. `run.py check` measured CPU/wall 0.99 to 1.00 in every scenario.
- **Seeds** (rule 5.2): `Options(seed = seed)`. `optimize` seeds Julia's global generator with it, and the library's own generator (`default_rng_mh`) is that same global generator, so the adapter's operators (which call `rand()`) are seeded too.

## Binary: OneMax 100 and 1000 (matched), OneMax 100 (idiomatic)

**Methods** (`onemax_solvers`, lines 279-304):
- Matched: the library's `GA` framework with DEAP's `eaSimple` settings: population 300, `TournamentSelection(K = 3)`, two-point crossover at 0.5, bit-flip at 1/n on 20% of the children, `GenerationalReplacement` (no elitism). The GA framework dispatches on operator types ([tutorial](https://jmejia8.github.io/Metaheuristics.jl/stable/tutorials/create-metaheuristic/)), and the library has no two-point crossover and no per-child mutation probability, so the adapter adds them (`TwoPointCrossover`, `BitFlipSomeChildren`, lines 234-270). Its GA pairs the selected parents (i, i + N/2) instead of neighbours.
- Idiomatic: the guide: "Binary: Use GA with BitFlipMutation". The binary example of the [GA docstring](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/#GA) runs `GA()` on a `BitArraySpace` with its defaults: population 100, binary tournament, uniform crossover at 0.5, `BitFlipMutation` at 1e-5, `ElitistReplacement`.

**Keeping going:** runs to the budget by itself, with the options above.

**Left out:** `MCCGA`, a compact GA "for real-valued optimization problems"; nothing else is presented for bits.

**Separate tests** (2026-09-25, Metaheuristics.jl 3.5.0, seeds 0 to 4, the scenario's budget, 60 s cap):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|---|
| OneMax 100, matched | ga | 5 | 5 | 9,900 | 100 (100, 100) | 0 |
| OneMax 1000, matched | ga | 5 | 5 | 170,100 | 1000 (1000, 1000) | 0 |
| OneMax 100, idiomatic | ga | 5 | 5 | 1,900 | 100 (100, 100) | 0 |

## Permutation: N-Queens 32 and 64

**Methods** (`nqueens_solvers`, lines 306-331): the guide: "Permutation-based: Use GA with OrderCrossover or BRKGA".
- `ga`: the [N-Queens tutorial](https://jmejia8.github.io/Metaheuristics.jl/stable/tutorials/n-queens/) runs `optimize(attacks, PermutationSpace(N), GA)`, the GA's defaults for permutations (`get_parameters` in `src/algorithms/singleobjective/GA/GA.jl`): population 100, binary tournament, `OrderCrossover`, `SlightMutation`, `ElitistReplacement`. The adapter builds the same GA explicitly, to pass the options.
- `brkga`: [`BRKGA`](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/combinatorial/#BRKGA) with its defaults (20 elites, 10 mutants, 70 offspring, bias 0.7), on random keys in [0, 1]ⁿ decoded by `sortperm`, as in its docstring's permutation example. The reported solution is the decoded permutation.

**Keeping going:** runs to the budget by itself, with the options above.

**Left out:** `GRASP`, `VNS` and `VND` need a problem-specific constructor or neighbourhood structures from the user (their docstrings), and the guide doesn't list them for permutations.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|---|
| N-Queens 32 | ga | 5 | 0 | – | 1 (1, 2) | 0 |
| N-Queens 32 | brkga | 5 | 0 | – | 7 (6, 8) | 0 |
| N-Queens 64 | ga | 5 | 0 | – | 4 (3, 5) | 0 |
| N-Queens 64 | brkga | 5 | 0 | – | 20 (16, 23) | 0 |

Neither method solves N-Queens 32 or 64 within the budget; the best values are the numbers of diagonal conflicts left.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** (`real_solvers`, lines 333-350): the guide: "Box-constrained (continuous): Use ECA, DE, PSO, or SHADE"; the [FAQ](https://jmejia8.github.io/Metaheuristics.jl/stable/faq/#How-to-choose-between-algorithms?): "Single-objective, unconstrained: ECA, DE, PSO are good starting points". The first three, with their defaults ([docstrings](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/)), on a `boxconstraints` search space:
- `eca`: ECA, the default method of `optimize` and the one the [Quick Start](https://jmejia8.github.io/Metaheuristics.jl/stable/#Quick-Start) runs on Rastrigin. K = 7, population K·D, η_max = 2, p_exploit = 0.95, p_bin = 0.02. It switches to exploitation after 95% of `f_calls_limit`, which is the scenario's budget.
- `de`: DE/rand/1/bin, population 10·D, F = 0.7, CR = 0.5. The guide lists DE as "Good for multimodal".
- `pso`: population 10·D, C1 = C2 = 2, ω = 0.8. The guide lists PSO as "Good for multimodal".

**Keeping going:** runs to the budget by itself, with the options above.

**Left out:**
- `SHADE`: fourth in the guide's list, and rule 6.4 allows three.
- `ABC`: listed as "Good for multimodal", but not in the guide's list for box-constrained problems, nor in the FAQ's.
- `GA` with SBX and polynomial mutation: its docstring has a real-encoding example on Rastrigin, but neither the guide nor the FAQ lists the GA for continuous problems. For reference, the 0.6.0 benchmark ran it (the defaults for real numbers: population 100, SBX, polynomial mutation, elitist replacement) and it reached the target in 10 of 10 runs on Rastrigin 10 and 30 and Ackley 30, and 0 of 10 on Rosenbrock 10: better than ECA, DE and PSO on the multimodal problems. The rules leave it out; if you read the documentation as recommending it, open an issue.
- `CGSA`, `SA`, `WOA`, `MCCGA`: in the algorithm list, but not in the guide's recommendations; `CSO` is for large-scale problems; `εDE` for constrained ones.
- A CMA-ES: the library has none.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|---|
| Rastrigin 10 | eca | 5 | 1 | 44,310 | 2.985 (0.00854, 5.97) | 0 |
| Rastrigin 10 | de | 5 | 5 | 161,500 | 0.00776 (0.00257, 0.00962) | 0 |
| Rastrigin 10 | pso | 5 | 0 | – | 10.94 (6.97, 15.92) | 0 |
| Rastrigin 30 | eca | 5 | 0 | – | 5.97 (2.985, 10.94) | 0 |
| Rastrigin 30 | de | 5 | 0 | – | 115.3 (105.8, 122.5) | 0 |
| Rastrigin 30 | pso | 5 | 0 | – | 41.79 (29.85, 46.76) | 0 |
| Ackley 30 | eca | 5 | 5 | 70,980 | 0.00967 (0.00834, 0.00989) | 0 |
| Ackley 30 | de | 5 | 5 | 497,700 | 0.00943 (0.00899, 0.00992) | 0 |
| Ackley 30 | pso | 5 | 0 | – | 2.12 (1.90, 2.50) | 0 |

## Continuous, unimodal: Rosenbrock 10

**Methods:** the same three, ECA, DE and PSO, with their defaults, for the same reasons: the guide recommends them for box-constrained problems without distinguishing unimodal ones, and lists ECA under "Fast convergence".

**Keeping going:** runs to the budget by itself.

**Left out:** as above.

**Separate tests** (2026-09-25):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|---|
| Rosenbrock 10 | eca | 5 | 5 | 18,130 | 0.00759 (0.00614, 0.00918) | 0 |
| Rosenbrock 10 | de | 5 | 5 | 196,400 | 0.00842 (0.00751, 0.00968) | 0 |
| Rosenbrock 10 | pso | 5 | 5 | 96,200 | 0.00881 (0.00854, 0.00989) | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1 (matched)

**Methods** (`front_solvers`, lines 352-381), the library's multi-objective algorithms with the matched settings. The guide recommends NSGA2, SPEA2 or SMS_EMOA for 2 and 3 objectives and NSGA3 for more; the objectives are returned as `(f, [0.0], [0.0])`, the unconstrained form of the [docstrings](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/multiobjective/).
- `nsga2`, `spea2`, `sms_emoa`: population 100 (92 with 3 objectives), SBX with η 15 and `p_cr = 0.9`, polynomial mutation with η 20 and `p_m = 1/n`.
- `nsga3`: population 100 (92), Das-Dennis directions with 99 (12) partitions, SBX with η 30 and `p_cr = 1`, polynomial mutation with η 20 and `p_m = 1/n`.
- `moead`: `MOEAD_DE` with the Das-Dennis weights of `gen_ref_dirs` (100 with 2 objectives, 91 with 3), 20 neighbours, parents from the neighbourhood with probability 0.9, polynomial mutation with η 20 at 1/n.

Differences from the matched settings:
- `p_cr` is the probability of crossing each variable, and every pair is crossed (pymoo: pairs at 0.9, variables at 0.5).
- NSGA-II and SPEA2 create 2N children per generation, not N. SMS-EMOA is steady-state: one child at a time, N per iteration.
- With 3 objectives, SMS-EMOA estimates the hypervolume contributions by Monte Carlo, with its default `n_samples = 10,000` samples, for every child.
- The library's MOEA/D is MOEA/D-DE: DE/rand/1 (F 0.5, CR 1) and polynomial mutation instead of SBX, at most 2 replacements per child (n_r, 2% of the population), and Tchebycheff also on DTLZ (no PBI).

**The front** (rule 7.2): the non-dominated part of the final population (`status.population`), whose size is the scenario's (92 or 100; 91 or 100 for MOEA/D, one per weight vector), with the objective values the library computed.

**Keeping going:** runs to the budget by itself, with the options above; `RobustConvergence` isn't added.

**Left out:** `CCMO`, which the guide recommends for constrained multi-objective problems.

**Separate tests** (2026-09-25; hypervolume computed by `run.py`'s code from the reported solutions):

| Scenario | Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|---|
| ZDT1 | nsga2 | 5 | 0.8663 (0.8672, 0.8654) | 25,100 | 0 |
| ZDT1 | nsga3 | 5 | 0.8675 (0.8679, 0.8664) | 25,000 | 0 |
| ZDT1 | spea2 | 5 | 0.8658 (0.8662, 0.8645) | 25,100 | 0 |
| ZDT1 | moead | 5 | 0.6485 (0.6716, 0.6260) | 25,000 | 0 |
| ZDT1 | sms_emoa | 5 | 0.8719 (0.8720, 0.8719) | 25,000 | 0 |
| ZDT2 | nsga2 | 5 | 0.5336 (0.5348, 0.5334) | 25,100 | 0 |
| ZDT2 | nsga3 | 5 | 0.5344 (0.5363, 0.5338) | 25,000 | 0 |
| ZDT2 | spea2 | 5 | 0.5306 (0.5317, 0.5294) | 25,100 | 0 |
| ZDT2 | moead | 5 | 0.2514 (0.2585, 0.2100) | 25,000 | 0 |
| ZDT2 | sms_emoa | 5 | 0.5387 (0.5387, 0.5386) | 25,000 | 0 |
| ZDT3 | nsga2 | 5 | 1.3229 (1.3243, 1.3226) | 25,100 | 0 |
| ZDT3 | nsga3 | 5 | 1.3221 (1.3234, 1.3197) | 25,000 | 0 |
| ZDT3 | spea2 | 5 | 1.3148 (1.3175, 1.3112) | 25,100 | 0 |
| ZDT3 | moead | 5 | 0.8159 (0.9353, 0.7495) | 25,000 | 0 |
| ZDT3 | sms_emoa | 5 | 1.3291 (1.3293, 1.3289) | 25,000 | 0 |
| DTLZ2 | nsga2 | 5 | 0.6259 (0.6479, 0.6012) | 25,116 | 0 |
| DTLZ2 | nsga3 | 5 | 0.7385 (0.7395, 0.7374) | 25,024 | 0 |
| DTLZ2 | spea2 | 5 | 0.7358 (0.7380, 0.7275) | 25,116 | 0 |
| DTLZ2 | moead | 5 | 0.6424 (0.6606, 0.6310) | 25,025 | 0 |
| DTLZ2 | sms_emoa | 3 | 0.7361 (0.7372, 0.7339) | 7,912 | 3 |
| DTLZ1 | nsga2 | 5 | 1.2708 (1.2740, 1.1927) | 40,020 | 0 |
| DTLZ1 | nsga3 | 5 | 1.2832 (1.3024, 1.2756) | 40,020 | 0 |
| DTLZ1 | spea2 | 5 | 1.3020 (1.3035, 1.3009) | 40,020 | 0 |
| DTLZ1 | moead | 5 | 1.2331 (1.2800, 1.1949) | 40,040 | 0 |
| DTLZ1 | sms_emoa | 3 | 1.0805 (1.3004, 0.9464) | 15,364 | 3 |

**Why SMS-EMOA hits the 60 s cap on DTLZ2 and DTLZ1.** It's the library, not its use. SMS-EMOA is steady-state: every child is added to the population, which is sorted again, and the child or a member of the last front is removed by its hypervolume contribution (`update_population!`, `src/algorithms/multiobjective/SMS_EMOA/update-population.jl`). With 2 objectives, the contributions are computed exactly, and a ZDT run takes about 1.3 s. With 3 objectives, `calculate_hv` (`SMS_EMOA.jl`) estimates them by Monte Carlo with `n_samples` = 10,000 points, from scratch, for every child: it builds a 10,000 × 3 matrix per member of the last front. Measured on its own, one call takes 1.4 ms for a front of 10 points, 5.3 ms for 50 and 11.3 ms for 93, and allocates 12.7 MiB for 93. On DTLZ2, the whole population of 92 is already one front after 3,036 evaluations, so every child costs about 11 ms, and 25,000 children would take about 5 minutes; the runs reach a median of 7,912 evaluations in 60 s, and on DTLZ1 15,364. `n_samples` is a documented parameter, but its default is the documented setting, and lowering it would be tuning. The 0.6.0 results showed the same (9,016 and 20,056 evaluations).

## Can't run

Every scenario runs. The multi-objective scenarios run MOEA/D-DE in place of the matched MOEA/D (see the differences above).

## Bugs found

- **The bounded SBX computes the second child's spread from the lower bound** (`SBX_crossover`, `src/operators/crossover/sbx.jl`): both children use β = 1 + 2 (y1 − xl) / Δ, where Deb's bounded SBX (and pymoo's) uses β = 1 + 2 (xu − y2) / Δ for the second child. All the multi-objective algorithms and the real-valued GA use it. Found by reading the code; its effect on the results wasn't measured. Not reported upstream yet; not worked around.
- **Documentation:** the `DE` docstring gives F = 1.0 as the default; the code's default is F = 0.7 (`src/algorithms/singleobjective/DE/DE.jl`), which the adapter uses.
