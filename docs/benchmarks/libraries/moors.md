# moors (Rust, 0.2.11)

moors is the Rust core of [moo-rs](https://github.com/andresliszt/moo-rs), a library of genetic algorithms for multi-objective optimization (NSGA-II, NSGA-III, R-NSGA-II, AGE-MOEA, REVEA, SPEA2, IBEA) with a single-objective GA built from the same parts: one generation loop (`GeneticAlgorithm`) with pluggable operators. Its docs are at [andresliszt.github.io/moo-rs](https://andresliszt.github.io/moo-rs/); the links below point to their sources, the examples and the tests at moors 0.2.11 ([488063e](https://github.com/andresliszt/moo-rs/tree/488063e679f74945dd3454c1982a93d62e5e1e1f)).

The single-objective GA is an `AlgorithmBuilder` with `RankSelection` (a binary tournament on fitness rank) and `FitnessSurvival` (the best of parents and offspring), the two operators pymoors' `GeneticAlgorithmSOO` fixes ([soo/mod.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/pymoors/src/algorithms/soo/mod.rs#L82-L98)) and moors' single-objective test uses ([test_ga_soo.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/tests/test_ga_soo.rs#L21-L59)). moors has no other single-objective algorithm, so the GA runs on every single-objective problem type ([ga!](../../../benchmarks/adapters/moors/src/main.rs#L503)).

Adapter: [benchmarks/adapters/moors/](../../../benchmarks/adapters/moors/).
Know a better way to solve one of these problems with moors? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs moors

- **Evaluations:** every generation evaluates the population and its offspring together ([ga.rs line 95](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/algorithms/ga.rs#L95)): P + O evaluations, with no setting to avoid it. The fitness function counts every row and records the first hit ([soo_fitness](../../../benchmarks/adapters/moors/src/main.rs#L261)).
- **Stop:** the iteration count is set above any budget. An `AdaptiveController`, which moors calls after every generation ([controller.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/algorithms/helpers/controller.rs#L1-L7)), ends the run at the target, the budget or 60 s ([Budget](../../../benchmarks/adapters/moors/src/main.rs#L229)).
- **Keeping going (rule 2.2):** moors also ends a run when duplicate removal leaves no new offspring in 200 tries (`EmptyMatingResult`, [ga.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/algorithms/ga.rs#L125-L157)), a convergence criterion. moors has no restart mechanism, so the GA restarts from a new random population with the seeds of rule 2.2 ([run_single](../../../benchmarks/adapters/moors/src/main.rs#L438), [attempt_seed](../../../benchmarks/adapters/moors/src/main.rs#L430)). No test run restarted.
- **Bounds (rule 2.4): clipping.** SBX and Gaussian mutation ignore the bounds; moors clamps every offspring to the `lower_bound` and `upper_bound` of `impl_constraints_fn!` before it's evaluated ([docs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/fitness_and_constraints/rust/lower_upper_bounds.md), [`mating_batch`](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/evolve.rs#L49-L80)). Counted as `outside` ([is_outside](../../../benchmarks/adapters/moors/src/main.rs#L223)).
- **Best solution:** the best of each attempt's final population, which `FitnessSurvival` keeps; `best` is recomputed after the clock. moors minimizes, so OneMax is minus the number of ones, as in its [knapsack example](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/getting_started/rust/knapsack.md).
- **Shift** (rule 1.4): computed once ([shifts](../../../benchmarks/adapters/moors/src/main.rs#L89)).
- **Output:** moors prints warnings with `println!`, so the adapter writes its JSON to a copy of stdout ([results_output](../../../benchmarks/adapters/moors/src/main.rs#L315)).
- **One thread, seeds:** no rayon or BLAS dependency; the builder's `.seed()`.
- **Separate tests:** 2026-09-25, moors 0.2.11, seeds 0 to 4. `outside` was 0 in every run.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched: not run** ([adapter](../../../benchmarks/adapters/moors/src/main.rs#L439-L442)). moors has `TwoPointBinaryCrossover` and `BitFlipMutation`, but its tournaments are binary ([selection/mod.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/selection/mod.rs#L57-L139)), and its single-objective survivals keep the best of parents and offspring: no generational replacement.
- **Idiomatic** ([adapter](../../../benchmarks/adapters/moors/src/main.rs#L511-L528)): the binary example of the [README's Quickstart](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/README.md) ([quick_start.md](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/getting_started/rust/quick_start.md)): `RandomSamplingBinary`, `SinglePointBinaryCrossover` at 0.9, `BitFlipMutation` on 10% of the children, `ExactDuplicatesCleaner`, population 100, 32 offspring. The per-bit rate is 1 / size: the example's 0.5 is sized for its 5 items, and moors documents no default. With 0.5 on 100 bits (5 seeds), no run reached the target (median best 91).

**Keeping going:** runs to the target or the budget.

**Left out:** `UniformBinaryCrossover` and `UniformBinaryMutation` (no example uses them for a binary problem); `RandomSelection` (random winners).

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| onemax-100-idiomatic | ga | 5 | 5 | 11,422 | 100 | 100 | 100 | 0 |

## Permutation: N-Queens 32 and 64

**Methods** ([adapter](../../../benchmarks/adapters/moors/src/main.rs#L529-L549)): the GA with `PermutationSampling`, `OrderCrossover`, `SwapMutation` ([sampling](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/operators/rust/sampling.md), [crossover](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/operators/rust/crossover.md), [mutation](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/operators/rust/mutation.md)) and `ExactDuplicatesCleaner`, as in the README's discrete example. The docs have no permutation example and no preference: `OrderCrossover` is moors' one permutation crossover, and `SwapMutation` the mutation they describe as exploring "neighboring permutations". Rates: `AlgorithmBuilder`'s defaults, crossover 0.9 and mutation 0.2 ([builder.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/algorithms/builder.rs#L86-L89)); population and offspring 200, from the docs' [real-valued example](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/getting_started/rust/real_valued.md).

**Keeping going:** runs to the target or the budget.

**Left out:** `ScrambleMutation`, `DisplacementMutation`, `InversionMutation`.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| nqueens-32-idiomatic | ga | 5 | 5 | 62,107 | 0 | 0 | 0 | 0 |
| nqueens-64-idiomatic | ga | 5 | 5 | 188,508 | 0 | 0 | 0 | 0 |

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Methods** ([adapter](../../../benchmarks/adapters/moors/src/main.rs#L550-L581)): the GA with the docs' only [real-valued example](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/getting_started/rust/real_valued.md) (an NSGA-II example): `RandomSamplingFloat` in the bounds, `SimulatedBinaryCrossover` η 15 at 0.9, `GaussianMutation` of 10% of the genes with σ 0.01 on 20% of the children, `CloseDuplicatesCleaner(1e-16)`, population and offspring 200. The docs don't separate multimodal from unimodal problems.

**Keeping going:** runs to the budget; no run met the no-new-offspring criterion.

**Left out:**
- The single-objective test's settings ([test_ga_soo.rs lines 23-38](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/tests/test_ga_soo.rs#L23-L38): SBX η 15 at 0.9, `GaussianMutation` of 5% of the genes with σ 0.1 on 10% of the children, `CloseDuplicatesCleaner(1e-6)`, population 100, 50 offspring): a unit test on a 3-variable constrained sphere, not the docs. With them (5 seeds, the earlier shift in [−1, 1]; median best): Rastrigin 10 6.96, Rastrigin 30 30.8, Rosenbrock 10 2.80, Ackley 30 11.2, no target reached.
- `UniformRealMutation`, `ArithmeticCrossover`: no example uses them. `ExponentialCrossover` is for differential evolution, which moors doesn't have.
- `FitnessConstraintsPenaltySurvival`: for constrained problems.
- The Gaussian-process surrogate (feature `surrogate`): for expensive fitness functions.
- σ scaled to the range: `GaussianMutation` takes an absolute σ ([gaussian.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/mutation/gaussian.rs)) and the docs give no rule, so 0.01 stays, also on Ackley's [−32.768, 32.768].

**Separate tests** (with the shift of rule 1.4):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped | Outside |
|---|---|---|---|---|---|---|---|---|---|
| rastrigin-10-idiomatic | ga | 5 | 0 | – | 7.960 | 4.975 | 10.94 | 0 | 0 |
| rastrigin-30-idiomatic | ga | 5 | 0 | – | 49.75 | 45.77 | 63.68 | 0 | 0 |
| rosenbrock-10-idiomatic | ga | 5 | 0 | – | 3.603 | 0.02366 | 10.43 | 0 | 0 |
| ackley-30-idiomatic | ga | 5 | 0 | – | 14.92 | 14.02 | 15.74 | 0 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Not run** ([main](../../../benchmarks/adapters/moors/src/main.rs#L694-L695)). moors has NSGA-II, NSGA-III, SPEA2 and SBX, but no polynomial mutation ([operators/mutation](https://github.com/andresliszt/moo-rs/tree/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/mutation)), and no MOEA/D or SMS-EMOA.

**Left out:**
- AGE-MOEA, IBEA and REVEA ([algorithms](https://github.com/andresliszt/moo-rs/tree/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/algorithms)): not matched algorithms (rule 6.1).
- R-NSGA-II: it searches near a decision maker's reference points ([docs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/algorithms/rust/rnsga2.md)), not the whole front.

## Can't run

- Matched OneMax 100 and 1000: no tournament of 3 and no generational replacement.
- The multi-objective scenarios: no polynomial mutation.

## Bugs found

None is worked around. They were found in multi-objective runs with a polynomial mutation written by the adapter (Deb's, pymoo's formula), which also ran AGE-MOEA, IBEA and REVEA; those runs aren't part of the benchmark.

- **AGE-MOEA panics when its first front is degenerate** ([andresliszt/moo-rs#301](https://github.com/andresliszt/moo-rs/issues/301)): "All components of the central point must be > 0", or "There should be at least one solution in the front: UndefinedOrder" (a 0 / 0). It panicked in 4 of 5 ZDT2 runs and all 5 DTLZ1 runs, after 92 to 23,460 evaluations.
- **SPEA2's mating selection prefers the worst individuals** (not reported upstream yet). The survival stores SPEA2's fitness F(i) = rank + density, lower is better ([survival/moo/spea2.rs lines 41-69](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/survival/moo/spea2.rs#L41-L69)), and `Spea2ScoringSelection` maximizes it ([selection/moo/spea2.rs lines 9-16](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/selection/moo/spea2.rs#L9-L16)).
- **SPEA2's archive truncation keeps dominated individuals** (not reported upstream yet). With more non-dominated individuals than the archive holds, it keeps the most isolated of the whole population ([line 62](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/survival/moo/spea2.rs#L60-L63)). On ZDT1, 1 to 45 of the final archive's 100 were dominated. moors' docs call the SPEA2 population "the final non-dominated set" (algorithms/moo/spea2.rs).
  - Hypervolume medians of 5 seeds, as is → selection fixed → both fixed: ZDT1 0.689 → 0.850 → 0.863, ZDT2 0.262 → 0.516 → 0.531, ZDT3 0.841 → 1.069 → 1.233, DTLZ2 0.476 → 0.443 → 0.535.
  - The maintainer's open issue [andresliszt/moo-rs#161](https://github.com/andresliszt/moo-rs/issues/161) says SPEA2 "is not reaching the expected front in simple test"; these two bugs are a likely cause.
- **`ReveaBuilder` ends its runs before the budget.** One iteration count serves as t_max of the angle-penalized distance and as the end of the run, and REVEA keeps one survivor per non-empty reference vector ([revea.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/survival/moo/revea.rs#L54-L110)), so the population shrinks: 22,806 to 24,891 of 25,000 evaluations. A design limit rather than a bug.
- **REVEA adapts its reference vectors only when a floating-point remainder is exactly 0** (not reported upstream yet): `(t / t_max) % frequency == 0.0` ([revea.rs line 103](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/survival/moo/revea.rs#L103)), where the paper adapts every frequency × t_max generations. With frequency 0.2 it adapts at t = 0, 25, 50 and 100 for t_max 125 (ZDT), only at t = 0 for t_max 136 (DTLZ2) and 217 (DTLZ1), and at 0, 120, 240 and 480 but not 360 for the docs' t_max of 600.
- **Documentation:** the README says `CloseDuplicatesCleaner` drops individuals "within an ε-ball"; it compares ε with the squared distance ([close.rs line 38](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/duplicates/close.rs#L30-L38)), so 1e-6 is a radius of 1e-3.
