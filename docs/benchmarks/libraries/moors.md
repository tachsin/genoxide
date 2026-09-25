# moors (Rust, 0.2.11)

moors is the Rust core of [moo-rs](https://github.com/andresliszt/moo-rs), a library of genetic algorithms for multi-objective optimization (NSGA-II, NSGA-III, R-NSGA-II, AGE-MOEA, REVEA, SPEA2, IBEA), with a single-objective GA built from the same parts. Every algorithm is one generation loop (`GeneticAlgorithm`) with pluggable sampling, crossover, mutation, duplicate removal, selection and survival operators. Its docs are at [andresliszt.github.io/moo-rs](https://andresliszt.github.io/moo-rs/); the links below point to the sources of those docs, the examples and the tests at the commit of moors 0.2.11 ([488063e](https://github.com/andresliszt/moo-rs/tree/488063e679f74945dd3454c1982a93d62e5e1e1f)).

Adapter: [benchmarks/adapters/moors/](../../../benchmarks/adapters/moors/).
Know a better way to solve one of these problems with moors? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How moors runs

- **Every generation evaluates the parents again.** `GeneticAlgorithm::next` evaluates the population and its offspring together ([ga.rs line 95](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/algorithms/ga.rs#L95)), so a generation of population P and O offspring costs P + O evaluations. moors has no setting that avoids it. The adapter counts every row its fitness function is given ([soo_fitness](../../../benchmarks/adapters/moors/src/main.rs#L261)), so these evaluations are counted.
- **The first hit.** The fitness function also records the first row whose value reaches the target, and the clock at that moment (`first_hit`).
- **Stopping.** A run ends at its iteration count, or early when the duplicate removal leaves no new offspring in 200 tries: `EmptyMatingResult`, "Terminating the algorithm early" ([ga.rs lines 82 and 135](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/algorithms/ga.rs#L125-L157)). The iteration count is only a budget, so rule 2.2 lifts it: the adapter sets it above any budget and ends the run from an `AdaptiveController`, which moors calls after every generation and which "may ... request an early stop" ([controller.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/algorithms/helpers/controller.rs#L1-L7)): after the generation in which the target is reached, at the budget or at 60 seconds ([Budget](../../../benchmarks/adapters/moors/src/main.rs#L229)). A run can go past the budget by less than one generation.
- **Keeping going (rule 2.2).** "No new offspring" (`EmptyMatingResult`) is a convergence criterion: it ends the attempt. moors has no restart mechanism, so the GA starts again from a new random population, keeping the best solution and counting every evaluation ([run_single](../../../benchmarks/adapters/moors/src/main.rs#L438)). Attempt 0 uses the run's seed, restart r the seed `(seed + 1) × 1,000,000 + r` ([attempt_seed](../../../benchmarks/adapters/moors/src/main.rs#L430)). None of the separate test runs needed a restart; it's tested on N-Queens 3, which has no solution.
- **The best solution** of an attempt is the best individual of moors' final population: `FitnessSurvival` keeps the best of parents and offspring, so it's the attempt's best. The best of all attempts is reported, with its `best` recomputed after the clock with the adapter's fitness function, in the problem's direction: moors minimizes, so OneMax is given to it as minus the number of ones, as in its [knapsack example](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/getting_started/rust/knapsack.md). The genes are `f64` in moors; the solution prints bits as 0/1 and the permutation as integers.
- **The clock** starts before the first attempt's initial population and stops when the last attempt ends, before the output.
- **Output.** moors prints warnings with `println!`, so the adapter points stdout at stderr and writes its JSON lines to a copy of the original stdout ([results_output](../../../benchmarks/adapters/moors/src/main.rs#L315)).
- **One thread, seeded.** moors has no rayon or BLAS dependency; its runs take the seed through the builder's `.seed()`.
- **Bounds (rule 2.4): clipping.** moors' SBX and Gaussian mutation ignore the bounds. Its documented bound handling is the `lower_bound` and `upper_bound` of `impl_constraints_fn!` ([docs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/fitness_and_constraints/rust/lower_upper_bounds.md)): moors clamps every offspring to them after crossover and mutation, before it's evaluated ([evolve.rs `mating_batch`](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/evolve.rs#L49-L80)). The initial populations are sampled within the bounds. The fitness function counts the solutions moors gives it outside the box ([is_outside](../../../benchmarks/adapters/moors/src/main.rs#L223)), and the continuous runs print that count as `outside`: 0 in every separate test run.
- **The shift** of Rastrigin and Ackley (rule 1.4) is computed once, before any run ([shifts](../../../benchmarks/adapters/moors/src/main.rs#L89)).

moors' single-objective GA is an `AlgorithmBuilder` with `RankSelection` (a binary tournament on the fitness rank) and `FitnessSurvival` (the best of parents and offspring survive). These are the two operators pymoors' `GeneticAlgorithmSOO` fixes ([pymoors/src/algorithms/soo/mod.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/pymoors/src/algorithms/soo/mod.rs#L82-L98)) and the ones of moors' single-objective test ([test_ga_soo.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/tests/test_ga_soo.rs#L21-L59)). moors has no other single-objective algorithm (no DE, CMA-ES or PSO), so the GA is its one method for every single-objective problem type ([ga!](../../../benchmarks/adapters/moors/src/main.rs#L503)).

## Binary: OneMax

**Methods:**
- **Matched: not run** ([adapter](../../../benchmarks/adapters/moors/src/main.rs#L439)). The matched scenario is DEAP's eaSimple: a tournament of 3 and generational replacement without elitism, with the library's own components only (rule 6.1). moors has `TwoPointBinaryCrossover` and `BitFlipMutation`, but not the other two:
  - its tournaments are binary: `SelectionOperator::operate` splits the participants into pairs ([selection/mod.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/selection/mod.rs#L57-L139));
  - its single-objective survivals keep the best of parents and offspring (`FitnessSurvival`, `FitnessConstraintsPenaltySurvival`); it has no generational replacement.
- **Idiomatic** ([adapter](../../../benchmarks/adapters/moors/src/main.rs#L511-L528)): the GA with the binary example of the [README's Quickstart](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/README.md) (also [quick_start.md](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/getting_started/rust/quick_start.md)): `RandomSamplingBinary`, `SinglePointBinaryCrossover` at 0.9, `BitFlipMutation` on 10% of the children, `ExactDuplicatesCleaner`, population 100 with 32 offspring.
  - The per-bit flip rate is 1 / size, not the example's 0.5. `BitFlipMutation`'s rate is per gene, and the example's 0.5 is sized for its 5 items: 2.5 flips per mutated child. moors documents no rate for other sizes and no default, so the adapter uses 1 / size, one flip per mutated child, the usual bit-flip rate. With 0.5 on 100 bits, a mutated child is a random bit string. A separate test with 0.5 (5 seeds, shown, not used to choose) reached the target in no run, with a best of 91 (median).

**Keeping going:** runs to the target by itself.

**Left out:** `UniformBinaryCrossover` and `UniformBinaryMutation` (no moors example uses them for a binary problem); `RandomSelection` for single objectives (random winners).

**Separate tests** (2026-09-25, moors 0.2.11, seeds 0 to 4):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| onemax-100-idiomatic | ga | 5 | 5 | 11,422 | 100 | 100 | 100 | 0 |

## Permutation: N-Queens 32 and 64

**Methods:** the GA with moors' permutation operators ([adapter](../../../benchmarks/adapters/moors/src/main.rs#L529-L549)): `PermutationSampling`, `OrderCrossover` and `SwapMutation` ([sampling](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/operators/rust/sampling.md), [crossover](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/operators/rust/crossover.md), [mutation](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/operators/rust/mutation.md)), and `ExactDuplicatesCleaner` as in the README's discrete example. moors has no permutation example, so the rates are `AlgorithmBuilder`'s defaults, crossover 0.9 and mutation 0.2 ([builder.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/algorithms/builder.rs#L86-L89)), and the population and offspring are the 200 of the docs' [real-valued example](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/getting_started/rust/real_valued.md).

**Keeping going:** runs to the target by itself.

**How the docs decided (rule 6.2):** they state no preference among the permutation operators, and have no permutation example and no default operator. `OrderCrossover` is moors' one permutation crossover. Of the mutations, `SwapMutation` is the one the docs describe as exploring "neighboring permutations".

**Left out:** `ScrambleMutation`, `DisplacementMutation` and `InversionMutation`, moors' other permutation mutations.

**Separate tests** (2026-09-25, moors 0.2.11, seeds 0 to 4):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| nqueens-32-idiomatic | ga | 5 | 5 | 62,107 | 0 | 0 | 0 | 0 |
| nqueens-64-idiomatic | ga | 5 | 5 | 188,508 | 0 | 0 | 0 | 0 |

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Methods:** the GA with the docs' [real-valued example](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/getting_started/rust/real_valued.md) ([adapter](../../../benchmarks/adapters/moors/src/main.rs#L550-L586)): `RandomSamplingFloat` within the bounds, `SimulatedBinaryCrossover` with η 15 at 0.9, `GaussianMutation` of 10% of the genes with σ 0.01 on 20% of the children, `CloseDuplicatesCleaner(1e-16)`, population and offspring 200. The example is NSGA-II's. moors doesn't separate multimodal from unimodal problems, so all four use it.

**How the docs decided (rule 6.2):** they state no preference for real-valued single-objective problems, and this is their example for real-valued problems, the only one in the docs. Bounds: clipped by moors (see How moors runs).

**Keeping going:** runs to the budget by itself (no run met moors' one convergence criterion, no new offspring). It stalls long before: every run in the separate tests ended at the budget without the target.

**Left out:**
- The settings of moors' single-objective test ([test_ga_soo.rs lines 23-38](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/tests/test_ga_soo.rs#L23-L38)): SBX η 15 at 0.9, `GaussianMutation` of 5% of the genes with σ 0.1 on 10% of the children, `CloseDuplicatesCleaner(1e-6)`, population 100 with 50 offspring. It's a unit test of the GA on a 3-variable constrained sphere, not the docs. A separate test with it (5 seeds, with the earlier shift in [−1, 1], shown, not used to choose; best median): Rastrigin 10 6.96, Rastrigin 30 30.8, Rosenbrock 10 2.80, Ackley 30 11.2, no target reached.
- `UniformRealMutation`, `ArithmeticCrossover`: no moors example uses them. `ExponentialCrossover` is for differential evolution, which moors doesn't have.
- `FitnessConstraintsPenaltySurvival`: for constrained problems.
- The Gaussian-process surrogate (feature `surrogate`): for expensive fitness functions.
- σ scaled to the range: moors' `GaussianMutation` takes an absolute σ ([gaussian.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/mutation/gaussian.rs)) and its docs give no rule for it, so the example's 0.01 stays, also on Ackley's [−32.768, 32.768].

**Separate tests** (2026-09-25, moors 0.2.11, seeds 0 to 4, with the shift of rule 1.4):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped | Outside |
|---|---|---|---|---|---|---|---|---|---|
| rastrigin-10-idiomatic | ga | 5 | 0 | – | 7.960 | 4.975 | 10.94 | 0 | 0 |
| rastrigin-30-idiomatic | ga | 5 | 0 | – | 49.75 | 45.77 | 63.68 | 0 | 0 |
| rosenbrock-10-idiomatic | ga | 5 | 0 | – | 3.603 | 0.02366 | 10.43 | 0 | 0 |
| ackley-30-idiomatic | ga | 5 | 0 | – | 14.92 | 14.02 | 15.74 | 0 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Not run.** These scenarios are matched: NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA with the library's own SBX and polynomial mutation (rule 6.1). moors has NSGA-II (`Nsga2Builder`), NSGA-III (`Nsga3Builder`), SPEA2 (`Spea2Builder`) and SBX (`SimulatedBinaryCrossover`), but no polynomial mutation: its mutations are bit flip, Gaussian, uniform real, and the permutation ones ([operators/mutation](https://github.com/andresliszt/moo-rs/tree/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/mutation)). So the adapter prints nothing for them ([main](../../../benchmarks/adapters/moors/src/main.rs#L694-L695)).

Also not run:
- moors has no MOEA/D and no SMS-EMOA.
- **AGE-MOEA** (`AgeMoeaBuilder`), **IBEA** (`IbeaBuilder`) and **REVEA** (`ReveaBuilder`), moors' other algorithms for a whole front ([algorithms](https://github.com/andresliszt/moo-rs/tree/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/algorithms)): the matched scenarios run only the five algorithms above (rule 6.1).
- **R-NSGA-II** (`Rnsga2Builder`): it searches near a decision maker's reference points ([docs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/docs/user_guide/algorithms/rust/rnsga2.md)), not the whole front.

## Can't run

- Matched OneMax 100 and 1000: moors has no tournament of 3 and no generational replacement (see Binary).
- The multi-objective scenarios: moors has no polynomial mutation (see Multi-objective).

## Bugs found

Not worked around (rule 8.4). They were found in earlier multi-objective runs, when the adapter still ran them with a polynomial mutation of its own (Deb's, pymoo's formula) and ran AGE-MOEA, IBEA and REVEA.

- **AGE-MOEA panics when its first front is degenerate** ([andresliszt/moo-rs#301](https://github.com/andresliszt/moo-rs/issues/301)): "All components of the central point must be > 0", or "There should be at least one solution in the front: UndefinedOrder" (a 0 / 0). In the separate tests (before rule 6.1 took it out) it panicked in 4 of 5 ZDT2 runs and all 5 DTLZ1 runs, after 92 to 23,460 evaluations.
- **SPEA2's mating selection prefers the worst individuals** (not reported upstream yet). Its survival stores SPEA2's fitness F(i) = rank + density, lower is better, as the survival score ([survival/moo/spea2.rs lines 41-69](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/survival/moo/spea2.rs#L41-L69)), and `Spea2ScoringSelection` compares that score with `SurvivalScoringComparison::Maximize` ([selection/moo/spea2.rs lines 9-16](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/selection/moo/spea2.rs#L9-L16)), so the binary tournaments pick the dominated and crowded individuals.
- **SPEA2's archive truncation keeps dominated individuals** (not reported upstream yet). When more individuals are non-dominated than the archive holds, it keeps the most isolated of the whole population, dominated ones included, instead of the most isolated non-dominated ones ([line 62](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/survival/moo/spea2.rs#L60-L63)). On ZDT1, 1 to 45 of the final archive's 100 individuals were dominated in the separate tests (55 to 99 non-dominated). moors' own docs say the SPEA2 population is "the final non-dominated set" (algorithms/moo/spea2.rs).
  - A diagnostic outside the adapter, the same runs with the tournament minimizing F, and with both fixes (5 seeds, hypervolume median, moors as it is → selection fixed → both fixed): ZDT1 0.689 → 0.850 → 0.863, ZDT2 0.262 → 0.516 → 0.531, ZDT3 0.841 → 1.069 → 1.233, DTLZ2 0.476 → 0.443 → 0.535. With three objectives, the one-shot nearest-neighbour truncation stays weak.
  - The maintainer's open issue [andresliszt/moo-rs#161](https://github.com/andresliszt/moo-rs/issues/161) says SPEA2 "is not reaching the expected front in simple test"; these two bugs are a likely cause.
- **`ReveaBuilder` ends its runs before the budget.** It uses one iteration count both as t_max of the angle-penalized distance and as the end of the run, and REVEA keeps one survivor per non-empty reference vector ([revea.rs](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/survival/moo/revea.rs#L54-L110)), so its population shrinks and its generations cost less than planned: 22,806 to 24,891 of 25,000 evaluations in the v0.6.0 results. A design limit rather than a bug; while REVEA still ran, the adapter built it through `AlgorithmBuilder` to separate the two.
- **REVEA adapts its reference vectors only when a floating-point remainder is exactly 0** (not reported upstream yet). The check is `(t / t_max) % frequency == 0.0` ([revea.rs line 103](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/operators/survival/moo/revea.rs#L103)), where the paper adapts every frequency × t_max generations. With frequency 0.2 it adapts at t = 0, 25, 50 and 100 for t_max 125 (ZDT), only at t = 0 for t_max 136 (DTLZ2) and 217 (DTLZ1), and at 0, 120, 240 and 480 but not 360 for the docs' own t_max of 600.
- **A documentation mismatch:** the README says `CloseDuplicatesCleaner` drops individuals "within an ε-ball"; it compares ε with the squared distance ([close.rs line 38](https://github.com/andresliszt/moo-rs/blob/488063e679f74945dd3454c1982a93d62e5e1e1f/moors/src/duplicates/close.rs#L30-L38)), so 1e-6 is a radius of 1e-3.

## Changes

- 2026-09-25: runs print their solutions and the adapter has the `values` command. Matched OneMax uses a tournament of 3 and generational replacement, as the matched setting says, instead of moors' binary tournament and elitist survival. A single-objective run that moors ends by itself (no new offspring) restarts from a new random population.
- 2026-09-25, rules amended: AGE-MOEA, IBEA and REVEA no longer run (rule 6.1). The continuous and multi-objective runs report `outside` (rule 2.4); moors' clipping keeps it at 0. The choices follow the docs' order (rule 6.2); none changed.
- 2026-09-25, rules amended again: matched scenarios use only the library's own components (rule 6.1), so matched OneMax and the multi-objective scenarios no longer run: the tournament of 3, the generational replacement and the polynomial mutation were the adapter's. Restarts use the seeds of rule 2.2, runs report `first_hit` (rule 3.3), the reported solution is the best of moors' final populations, and Rastrigin and Ackley use the shift of rule 1.4.
