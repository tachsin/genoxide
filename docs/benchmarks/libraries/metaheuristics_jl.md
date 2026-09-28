# Metaheuristics.jl (Julia, 3.5.0)

Metaheuristics.jl, by Jesús-Adolfo Mejía-de-Dios, is a Julia package of single- and multi-objective metaheuristics: ECA, DE, PSO, ABC, SA, SHADE and others for real numbers, a GA framework for binary, permutation and real encodings, BRKGA, and NSGA-II, NSGA-III, SPEA2, SMS-EMOA, MOEA/D-DE and CCMO. Its docs are at [jmejia8.github.io/Metaheuristics.jl](https://jmejia8.github.io/Metaheuristics.jl/stable/), from `docs/src` of the package; the sources linked below are those of the [v3.5.0 tag](https://github.com/jmejia8/Metaheuristics.jl/tree/v3.5.0).

Adapter: [benchmarks/adapters/metaheuristics_jl/](../../../benchmarks/adapters/metaheuristics_jl/).
Know a way to set Metaheuristics.jl closer to a definition? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## What it runs

The matched suite: each problem with one method, defined the same for every library ([rule 6](../rules.md#6-the-methods)), and each library's own implementation of it.

| Scenario | Method | Metaheuristics.jl |
|---|---|---|
| Rastrigin 30, matched: no target, a budget of 300,000 evaluations | DE/rand/1/bin ([6.3](../rules.md#6-the-methods)) | `de`: `DE` with `strategy = :rand1` |
| OneMax 1000, matched | GA as DEAP's `eaSimple` ([6.2](../rules.md#6-the-methods)) | can't run: see [Can't run](#cant-run) |
| Rosenbrock 10, matched | CMA-ES ([6.4](../rules.md#6-the-methods)) | can't run: see [Can't run](#cant-run) |

The adapter prints nothing for any other scenario ([`SCENARIOS`, lines 179-182](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L179-L182)).

## How the adapter runs Metaheuristics.jl

- **Fitness function:** in Julia ([bench.jl, lines 26-52](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L26-L52)), with the shift of rule 1.4; `values` evaluates it for rule 1.2 ([lines 211-221](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L211-L221)).
- **Evaluations:** `counted` counts every call, with the best solution and the solutions outside the bounds ([lines 113-129](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L113-L129)). `optimize`'s `logger`, called after the initial population and after every iteration, marks each generation's end for `last_generation` ([lines 96-98](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L96-L98)).
- **No target:** Rastrigin 30 has none. Every run uses the whole budget of 300,000 evaluations, or stops at the time cap, and is measured by its time and its error at the end, the best value (the optimum is 0). The budget's target is −∞, never reached, so nothing stops a run early; each run prints `"target": null`, `"success": false` and `"first_hit": null` ([lines 49-52](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L49-L52)).
- **Stop:** a user-defined termination, `BudgetTermination`, as the library's own ([lines 107-111](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L107-L111)), ends the run at the budget or the time cap after every iteration; `Options` also gets the budget as `f_calls_limit` and the cap as `time_limit`.
- **No convergence criterion (rule 2.2)** ([`algorithm_kwargs`, lines 131-153](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L131-L153)): the iteration limit, only a budget, is lifted. `optimize` checks a convergence criterion after every iteration whatever the user sets, `CheckConvergence` in [`default_stop_check`](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/termination/default.jl#L1-L20): it ends the run when all of `AbsoluteFunctionConvergence(f_tol)`, `RelativeFunctionConvergence(f_tol_rel)`, `SmallStandardDeviation()` and `RelativeParameterConvergence(x_tol)` hold, each a spread of the population at most its tolerance ([convergence.jl, lines 99-119](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/termination/convergence.jl#L99-L119)). The documented [`Options`](https://jmejia8.github.io/Metaheuristics.jl/stable/api/#Metaheuristics.Options) `f_tol = f_tol_rel = x_tol = -1` turn it off: a spread is never negative. A user termination replaces the second `CheckConvergence` that `optimize` adds when there is none ([before.jl, lines 26-40](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/optimize/before.jl#L26-L40)). Nothing else ends a run, and DE evaluates every trial, so it never stalls: no restarts.
- **Time:** from the run's `Budget`, created just before `optimize` builds the initial population. The method first makes the warm-up run of rule 4.2 ([line 250](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L250)).
- **Seeds (rule 5.2):** `Options(seed = seed)`; `optimize` seeds Julia's global generator with it ([before.jl, line 10](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/optimize/before.jl#L10)), which DE's mutation, crossover and repair use. The same seed repeats a run, and seed 1 gives the same alone as after seed 0 (tested).
- **Rule 5.3:** `EARLY_SEEDS` ([lines 189-192](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L189-L192)) and `main` ([lines 252-261](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L252-L261)).
- **One thread:** [run.sh](../../../benchmarks/adapters/metaheuristics_jl/run.sh) runs Julia with `--threads=1 --gcthreads=1,0` and BLAS with one thread.
- **Separate tests:** 2026-09-28, Metaheuristics.jl 3.5.0, seeds 0 to 2, a budget of 300,000 evaluations, 60 s cap. `outside` was 0 in every run.

## Rastrigin 30: DE/rand/1/bin

**Method** ([`rastrigin_de`, lines 159-177](../../../benchmarks/adapters/metaheuristics_jl/bench.jl#L159-L177)): the library's [`DE`](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/singleobjective/#Metaheuristics.DE) ([DE.jl](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/algorithms/singleobjective/DE/DE.jl)) with its documented keywords: `DE(N = 100, F = 0.5, CR = 0.9, strategy = :rand1)`, on `boxconstraints(lb = fill(-5.12, 30), ub = fill(5.12, 30))`.

| Definition | Metaheuristics.jl | Source |
|---|---|---|
| NP = 100, uniform in the box | `N = 100`; the initial population is drawn uniformly in the box | [`generate_population`](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/solutions/individual.jl#L251-L261) |
| r1, r2, r3 uniform, distinct, ≠ i | a base index and two others, uniform and mutually distinct, from the whole population: the target's index isn't passed (below) | [`DE_mutation`, lines 11-40](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/operators/mutation/mutation.jl#L11-L40); [DE.jl, line 203](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/algorithms/singleobjective/DE/DE.jl#L203) |
| v = x_r1 + F (x_r2 − x_r3), F = 0.5 fixed | `x + F * (a - b)`; `F_min` and `F_max` default to `F`, so F isn't drawn (it would be only if `F_min < F_max`) | [mutation.jl, lines 37-40](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/operators/mutation/mutation.jl#L37-L40); [DE.jl, lines 58-75 and 90-96](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/algorithms/singleobjective/DE/DE.jl#L58-L96) |
| binomial, CR = 0.9, one forced j_rand | `DE_crossover(x, u, CR)`: gene j from the mutant if `rand() < CR` or j = j_rand, else from the target; CR fixed as F | [uniform.jl, lines 33-49](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/operators/crossover/uniform.jl#L33-L49) |
| every trial in the box | `evo_boundary_repairer!` (below) | [DE.jl, line 205](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/algorithms/singleobjective/DE/DE.jl#L205); [repair.jl, lines 68-85](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/common/repair.jl#L68-L85) |
| u replaces x_i if f(u) ≤ f(x_i) | only if strictly better (below) | [`environmental_selection`, DE.jl, lines 109-126](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/algorithms/singleobjective/DE/DE.jl#L109-L126); [`is_better`, compare.jl, lines 6-8](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/common/compare.jl#L6-L8) |
| generational | `reproduction` builds all N trials from the population, then they are evaluated together and each replaces its target or not | [DE.jl, lines 78-106 and 187-210](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/algorithms/singleobjective/DE/DE.jl#L78-L106) |
| no archive, adaptation, restarts or convergence criterion | none; the convergence check turned off (above) | |

**Differences:**
- **Indices:** `reproduction` calls `DE_mutation(population, F, strategy, 1)`, whose fourth argument is the index of the best (used by `:best1`), not the target's. So the base and the two difference vectors are uniform and mutually distinct, but the target isn't excluded: in 3% of the trials one of the three is the target (1% the base, which makes that trial a DE/current/1 step). The operator, F, CR and the selection are the definition's.
- **Replacement:** a trial replaces its target only if its value is strictly lower (`is_better` is `A.f < B.f`); the definition also replaces on a tie. A tie needs a trial with exactly its target's value: with real genes and the forced index j_rand, that has probability zero here.
- **Bounds (rule 2.4):** a trial gene below the lower bound l becomes α l + (1 − α) b_j, and one above the upper bound u becomes β u + (1 − β) b_j, with α, β uniform in [0, 1] and b the best solution of the last generation: a random point between the crossed bound and the best's gene. The reference draws it again uniformly in the box. It acts only on genes that leave the box.
- **Generations:** the adapter's `generations` is the library's `iteration`, which counts the initial population as the first.

**Keeping going:** no convergence criterion (above); nothing else ends a run, and it evaluates every trial, so it never stalls.

**Separate tests:** Rastrigin 30 (budget 300,000, no target, cap 60 s):

| Solver | Runs | Time (s): seeds 0, 1, 2 | Error at the end: median | best | worst | Capped |
|---|---|---|---|---|---|---|
| de | 3 | 0.215, 0.219, 0.210 | 134.8 | 131.3 | 170.4 | 0 |

Each run used the whole budget: 3,000 generations of 100.

## Can't run

- **Matched OneMax 1000:** no two-point crossover. The GA's crossovers are `UniformCrossover`, `OrderCrossover`, `SBX` and `BinomialCrossover` (`src/operators/crossover/`), and the matched methods use only the library's own operators (rule 6.1).
- **Matched Rosenbrock 10:** no CMA-ES. Its single-objective methods are ECA, DE, εDE, SHADE, PSO, ABC, CGSA, CSO, SA, WOA, MCCGA, RDEx and the GA framework (`src/algorithms/singleobjective/`), with `Restart` around them.

## Bugs found

| Bug | Effect here | Worked around | Reported |
|---|---|---|---|
| The `DE` docstring gives F = 1.0 as the default; the code's is F = 0.7 ([DE.jl, lines 18-26 and 58-61](https://github.com/jmejia8/Metaheuristics.jl/blob/v3.5.0/src/algorithms/singleobjective/DE/DE.jl#L18-L61)) | none: the adapter sets F | no | [jmejia8/Metaheuristics.jl#130](https://github.com/jmejia8/Metaheuristics.jl/issues/130), fix proposed in [jmejia8/Metaheuristics.jl#131](https://github.com/jmejia8/Metaheuristics.jl/pull/131) |
