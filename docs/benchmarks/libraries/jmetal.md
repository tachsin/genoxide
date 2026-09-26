# jMetal (Java, 7.5)

jMetal is a Java framework for multi-objective optimization with metaheuristics, with single-objective algorithms too: "genetic algorithm (variants: generational, steady-state), evolution strategy (variants: elitist or mu+lambda, non-elitist or mu, lambda), DE, CMA-ES, PSO (Stantard 2007, Standard 2011), Coral reef optimization" ([documentation](https://jmetal.readthedocs.io/en/latest/), "Summary of features"). Its algorithms come as the component-based builders of `jmetal-component`, the current design ([Component-based algorithms](https://jmetal.readthedocs.io/en/latest/component.html)), and the older classes of `jmetal-algorithm`, each with example programs ([jmetal-component](https://github.com/jMetal/jMetal/tree/v7.5/jmetal-component/src/main/java/org/uma/jmetal/component/examples), [jmetal-algorithm](https://github.com/jMetal/jMetal/tree/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples)).

Adapter: [benchmarks/adapters/jmetal/](../../../benchmarks/adapters/jmetal/).
Know a better way to solve one of these problems with jMetal? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs jMetal

- **Fitness functions:** in Java ([Bench.java, lines 99-260](../../../benchmarks/adapters/jmetal/Bench.java#L99-L260)), in problems derived from `AbstractBinaryProblem`, `AbstractIntegerPermutationProblem` and `AbstractDoubleProblem`, as jMetal's users write their own ([lines 385-514](../../../benchmarks/adapters/jmetal/Bench.java#L385-L514); `values`, [lines 964-991](../../../benchmarks/adapters/jmetal/Bench.java#L964-L991)).
- **Evaluations:** every call of `evaluate()` counts, with the first hit ([`Budget`, lines 264-372](../../../benchmarks/adapters/jmetal/Bench.java#L264-L372)). jMetal evaluates every child.
- **Stop:** a single-objective run stops inside `evaluate()` at the target, the budget or the time cap (checked every 64 evaluations), with an exception the adapter catches; the algorithms' own limits are set out of reach. A multi-objective run stops through a `Termination` (SPEA2: its stopping condition) after every generation.
- **Keeping going (rule 2.2):** per section. jMetal has no restart mechanism; an attempt that ends by itself restarts from a new random start with the seeds of rule 2.2.
- **Bounds (rule 2.4):** `SBXCrossover`, `PolynomialMutation` and `DifferentialEvolutionCrossover` set a value outside the bounds to the bound (`RepairDoubleSolutionWithBoundValue`, their default); CMA-ES clips every sample (`Bounds.restrict`).
- **Time:** from the run's `Budget`, before the initial population, to the end of the algorithm ([`runSingle`, lines 733-760](../../../benchmarks/adapters/jmetal/Bench.java#L733-L760); [`runFront`](../../../benchmarks/adapters/jmetal/Bench.java#L835-L945)).
- **Seeds:** `JMetalRandom.getInstance().setSeed(seed)`. The CMA-ES and the initial permutations don't use it (see [Bugs found](#bugs-found)); the adapter seeds both without changing what they draw.
- **Solutions:** the best evaluated; a multi-objective run prints the non-dominated solutions of `result()`, the final population (SPEA2: its archive of N).
- **One thread and the JIT (rule 4.3):** jMetal evaluates sequentially by default (`SequentialEvaluation`, `SequentialSolutionListEvaluator`). The JVM runs with `-XX:+UseSerialGC -Xbatch` ([run.sh](../../../benchmarks/adapters/jmetal/run.sh)): the serial collector, and the JIT compiling while the calling thread waits. Each solver first makes the warm-up run of rule 4.2 ([lines 1025-1047](../../../benchmarks/adapters/jmetal/Bench.java#L1025-L1047)).
- **Separate tests:** 2026-09-25, jMetal 7.5, seeds 0 to 4, the scenario's budget, 60 s cap, on a shared machine (the capped runs depend on its load). `outside` was 0 in every run.

JVM flags on one invocation (Rastrigin 10: ga, de and cma_es, seeds 0 to 4, 500,000 evaluations, after a 1,000-evaluation warm-up):

| JVM flags | CPU / wall | ga, seeds 0-4 (s) | de, seeds 0-4 (s) | cma_es, seeds 0-4 (s) |
|---|---|---|---|---|
| `-XX:+UseSerialGC` | 1.14 | 0.11, 0.09, 0.06, 0.12, 0.09 | 0.18, 0.15, 0.18, 0.14, 0.15 | 1.08, 0.98, 1.01, 0.81, 0.80 |
| `+ -XX:ActiveProcessorCount=1` | 0.98 | 0.22, 0.16, 0.12, 0.09, 0.07 | 0.30, 0.23, 0.13, 0.13, 0.24 | 1.28, 1.16, 1.16, 1.20, 1.25 |
| `+ -XX:-TieredCompilation -XX:CICompilerCount=1` | 1.09 | 0.34, 0.23, 0.07, 0.08, 0.08 | 0.41, 0.15, 0.15, 0.13, 0.13 | 1.52, 0.96, 0.82, 0.81, 0.80 |
| `+ -XX:TieredStopAtLevel=1` (C1 only) | 1.01 | 0.13, 0.16, 0.09, 0.13, 0.12 | 0.32, 0.29, 0.31, 0.31, 0.28 | 1.43, 1.43, 1.39, 1.42, 1.43 |
| **`+ -Xbatch`** (chosen) | **0.99** | 0.33, 0.16, 0.07, 0.12, 0.08 | 0.36, 0.15, 0.13, 0.14, 0.13 | 0.98, 0.80, 0.80, 0.80, 0.71 |

C1-only makes DE and CMA-ES about 1.8 times slower throughout. `ActiveProcessorCount=1` stays within 10% here but not with Jenetics (1.37), and runs CMA-ES about 1.5 times slower. `-Xbatch` keeps the same compiled code, and moves the compilations the warm-up didn't trigger into the first timed runs.

## Binary: OneMax 100 (idiomatic)

**Methods:** the docs state no preference among the single-objective algorithms and list them as GA, evolution strategy, DE, CMA-ES, PSO and coral reef optimization. In that order, the adapter takes those with an example for the problem type, in their first variant, with the example's settings, up to 3 (rule 6.4).
- **`ga`:** [GenerationalGeneticAlgorithmBinaryExample.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-component/src/main/java/org/uma/jmetal/component/examples/singleobjective/geneticalgorithm/GenerationalGeneticAlgorithmBinaryExample.java) (OneMax 512): population 100, 100 children, binary tournament, `SinglePointCrossover(0.9)`, `BitFlipMutation(1 / n)`, (μ + λ) replacement, the builder's default ([lines 657-665](../../../benchmarks/adapters/jmetal/Bench.java#L657-L665)).
- **`es`:** [ElitistEvolutionStrategyRunner.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples/singleobjective/ElitistEvolutionStrategyRunner.java) (OneMax 512): (μ + λ) with μ 1, λ 10, `BitFlipMutation(1 / n)` ([lines 666-680](../../../benchmarks/adapters/jmetal/Bench.java#L666-L680)).

**Keeping going:** the examples' evaluation limits are replaced by the budget; no convergence criterion.

**Left out:**
- Coral reef optimization ([CoralReefsOptimizationRunner.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples/singleobjective/CoralReefsOptimizationRunner.java)): can't be seeded (rule 5.2; see [Bugs found](#bugs-found)).
- The steady-state GA and the (μ, λ) ES: second variants.
- `BasicLocalSearch` (LocalSearchBinaryProblemRunner.java) and random search: not in the docs' list.

**Not run: matched OneMax 100 and 1000** ([lines 650-655](../../../benchmarks/adapters/jmetal/Bench.java#L650-L655)). jMetal has no generational replacement without elitism (its replacements are (μ + λ), (μ, λ) with μ < λ, pairwise, random and the multi-objective ones; `GenerationalGeneticAlgorithm` keeps 2 elites), and no two-point crossover for bits (`TwoPointCrossover` is for numbers).

**Separate tests:**

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 4,395 | 100 | 100 | 100 | 0 |
| es | 5 | 5 | 969 | 100 | 100 | 100 | 0 |

## Permutation: N-Queens 32 and 64

**Methods:** `ga`, the only permutation example: [GeneticAlgorithmTSPExample.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-component/src/main/java/org/uma/jmetal/component/examples/singleobjective/geneticalgorithm/GeneticAlgorithmTSPExample.java), with the same settings as jmetal-algorithm's GenerationalGeneticAlgorithmTSPRunner.java: population 100, 100 children, binary tournament, `PMXCrossover(0.9)`, `PermutationSwapMutation(1 / n)`, (μ + λ) replacement ([lines 683-694](../../../benchmarks/adapters/jmetal/Bench.java#L683-L694)). Initial permutations drawn with `JMetalRandom` ([`createSolution`, lines 430-446](../../../benchmarks/adapters/jmetal/Bench.java#L430-L446)).

**Keeping going:** the example's `TerminationByEvaluations(250000)` is replaced by the budget.

**Separate tests:**

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 3 | 30,621 | 0 | 0 | 2 | 0 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 1 | 242,375 | 1 | 0 | 2 | 0 |

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Methods** ([lines 695-730](../../../benchmarks/adapters/jmetal/Bench.java#L695-L730)): in the docs' order, the algorithms with a continuous example are the GA, DE, CMA-ES and the PSOs; the first 3 run. jMetal documents nothing for multimodal or unimodal functions in particular, so all four problems run the same three.
- **`ga`:** [GenerationalGeneticAlgorithmExample.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-component/src/main/java/org/uma/jmetal/component/examples/singleobjective/geneticalgorithm/GenerationalGeneticAlgorithmExample.java) (Sphere 20): population 100, 100 children, binary tournament, `SBXCrossover(0.9, 20)`, `PolynomialMutation(1 / n, 20)`, (μ + λ).
- **`de`:** [DifferentialEvolutionRunner.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples/singleobjective/DifferentialEvolutionRunner.java) (Sphere 20): DE/rand/1/bin, CR 0.5, F 0.5, population 100.
- **`cma_es`:** [CovarianceMatrixAdaptationEvolutionStrategyRunner.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples/singleobjective/CovarianceMatrixAdaptationEvolutionStrategyRunner.java) (Sphere): the builder's defaults, λ 10, σ 0.3 ([`cmaes`, lines 579-641](../../../benchmarks/adapters/jmetal/Bench.java#L579-L641)). Difference: jMetal starts the mean in [0, 1)ⁿ whatever the bounds; the adapter starts it within the bounds.

**Keeping going:**
- `ga`, `de`: no convergence criterion; the evaluation limits are lifted.
- `cma_es`: an attempt ends when the covariance matrix degenerates and `checkEigenCorrectness` fails (it sets the evaluations to the maximum); the adapter restarts it from a new random point.
- **Workaround of a crash (rule 8.4):** when the covariance matrix holds NaN, `CMAESUtils.tql2` throws `ArrayIndexOutOfBoundsException`. The adapter catches it and starts a new attempt. The tables show both: `cma_es` with the workaround, `cma_es (as-is)` (`JMETAL_CMAES_AS_IS=1`) ending at the first crash with the best so far. On Rastrigin 30 and Ackley 30 the time cap stops most runs before any crash.

**Left out:**
- Standard PSO 2007 and 2011 (StandardPSO2007Runner.java, StandardPSO2011Runner.java): fourth in the docs' order.
- The steady-state GA: the second variant.
- The evolution strategy: its examples are on OneMax only.
- `BasicLocalSearch` (LocalSearchContinuousProblemRunner.java) and random search: not in the docs' list.

**Separate tests:**

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 15,803 | 0.00794 | 0.00548 | 0.00938 | 0 |
| de | 5 | 5 | 89,104 | 0.00861 | 0.00688 | 0.00974 | 0 |
| cma_es | 5 | 0 | - | 77.6 | 2.04 | 113.9 | 0 |
| cma_es (as-is) | 5 | 0 | - | 151.5 | 116.8 | 176.8 | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 61,087 | 0.00951 | 0.00819 | 0.01 | 0 |
| de | 5 | 0 | - | 62.5 | 53.3 | 75.4 | 2 |
| cma_es | 5 | 0 | - | 420.6 | 264.7 | 480.6 | 5 |
| cma_es (as-is) | 5 | 0 | - | 420.6 | 264.7 | 480.6 | 5 |

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 3.35 | 0.0677 | 5.11 | 0 |
| de | 5 | 0 | - | 0.146 | 0.11 | 0.23 | 0 |
| cma_es | 5 | 0 | - | 7.45 | 1.65 | 30.2 | 0 |
| cma_es (as-is) | 5 | 0 | - | 75.5 | 11.5 | 133.2 | 0 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 145,560 | 0.00991 | 0.00907 | 0.00997 | 0 |
| de | 5 | 5 | 55,810 | 0.00974 | 0.00949 | 0.00989 | 0 |
| cma_es | 5 | 0 | - | 19.9 | 19.8 | 19.9 | 5 |
| cma_es (as-is) | 5 | 0 | - | 19.9 | 19.8 | 19.9 | 3 |

## Multi-objective: ZDT1, ZDT2, ZDT3 (30 variables), DTLZ2 and DTLZ1 (3 objectives)

**Methods** (the matched settings, [lines 833-945](../../../benchmarks/adapters/jmetal/Bench.java#L833-L945)):
- **`nsga2`:** `NSGAIIBuilder`, N parents and N children (N = 100, 92 with 3 objectives), SBX η 15 at 0.9, polynomial mutation η 20 at 1 / n.
- **`nsga3`:** `NSGAIIIBuilder`, Das-Dennis directions (99 divisions: 100; 12: 91), population rounded up to a multiple of 4 by jMetal (100, 92), SBX η 30 at 1, the same mutation.
- **`spea2`:** jmetal-algorithm's `SPEA2` (no builder in jmetal-component): population and archive N, binary tournament, SBX η 15 at 0.9, the same mutation, k 1 ([`BudgetSPEA2`, lines 789-808](../../../benchmarks/adapters/jmetal/Bench.java#L789-L808)).
- **`moead`:** `MOEADBuilder`, 100 weight vectors (91 Das-Dennis with 3 objectives), 20 neighbors, neighborhood mating at 0.9, at most 2 replacements (the default), Tchebycheff (PBI with θ 5 for DTLZ), SBX η 20 at 1, the same mutation; one child per step.
- **`sms_emoa`:** `SMSEMOABuilder`, population N, one child per step, SBX η 15 at 0.9, the same mutation.

**Keeping going:** each runs to its budget (rule 7.1).

**Left out (rule 6.1):** SMPSO, GDE3, MOCell, IBEA, PAES, RVEA, AGE-MOEA, NSGA-II and SMS-EMOA with differential evolution, and jMetal's other multi-objective algorithms.

**Separate tests:**

| Problem (budget) | Solver | Runs | Hypervolume: median | best | worst | Median evaluations | Capped |
|---|---|---|---|---|---|---|---|
| ZDT1 (25,000) | nsga2 | 5 | 0.8696 | 0.8697 | 0.8692 | 25,000 | 0 |
|  | nsga3 | 5 | 0.8704 | 0.8707 | 0.8702 | 25,000 | 0 |
|  | spea2 | 5 | 0.8695 | 0.8699 | 0.8693 | 25,000 | 0 |
|  | moead | 5 | 0.8706 | 0.8708 | 0.8703 | 25,000 | 0 |
|  | sms_emoa | 5 | 0.8719 | 0.8719 | 0.8718 | 25,000 | 0 |
| ZDT2 (25,000) | nsga2 | 5 | 0.5358 | 0.5359 | 0.5354 | 25,000 | 0 |
|  | nsga3 | 5 | 0.5367 | 0.5375 | 0.5366 | 25,000 | 0 |
|  | spea2 | 5 | 0.5356 | 0.5367 | 0.5352 | 25,000 | 0 |
|  | moead | 5 | 0.5376 | 0.5379 | 0.5371 | 25,000 | 0 |
|  | sms_emoa | 5 | 0.5385 | 0.5385 | 0.5382 | 25,000 | 0 |
| ZDT3 (25,000) | nsga2 | 5 | 1.3271 | 1.3277 | 1.3268 | 25,000 | 0 |
|  | nsga3 | 5 | 1.3256 | 1.3266 | 1.3252 | 25,000 | 0 |
|  | spea2 | 5 | 1.3249 | 1.3259 | 1.2419 | 25,000 | 0 |
|  | moead | 5 | 1.3253 | 1.3255 | 1.3251 | 25,000 | 0 |
|  | sms_emoa | 5 | 1.3292 | 1.3293 | 1.3290 | 25,000 | 0 |
| DTLZ2 (25,000) | nsga2 | 5 | 0.7058 | 0.7126 | 0.6934 | 25,024 | 0 |
|  | nsga3 | 5 | 0.7439 | 0.7442 | 0.7438 | 25,024 | 0 |
|  | spea2 | 5 | 0.7300 | 0.7317 | 0.7246 | 25,024 | 0 |
|  | moead | 5 | 0.7444 | 0.7445 | 0.7441 | 25,000 | 0 |
|  | sms_emoa | 5 | 0.7557 | 0.7557 | 0.7556 | 25,000 | 0 |
| DTLZ1 (40,000) | nsga2 | 5 | 0.1357 | 0.1364 | 0.1320 | 40,020 | 0 |
|  | nsga3 | 5 | 0.1399 | 0.1400 | 0.1393 | 40,020 | 0 |
|  | spea2 | 5 | 0.1394 | 0.1394 | 0.1388 | 40,020 | 0 |
|  | moead | 5 | 0.1398 | 0.1400 | 0.1396 | 40,000 | 0 |
|  | sms_emoa | 5 | 0.1400 | 0.1401 | 0.1390 | 28,340 | 5 |

## Can't run

- Matched OneMax 100 and 1000: no generational replacement without elitism and no two-point crossover for bits (see [Binary](#binary-onemax-100-idiomatic)).

## Bugs found

| Bug | Effect here | Worked around | Reported |
|---|---|---|---|
| CMA-ES: once it has converged, its step size σ grows without bound. With jMetal's own `Sphere(10)` and its runner's settings, σ is 480 after 3,000 evaluations and 9 × 10¹⁴⁹ after 100,000, while the best stays at 10⁻⁶. Every sample then lands on the bounds | no CMA-ES run reaches a target; an attempt stalls until `tql2` throws, often after about 200,000 evaluations | no: restarting earlier would work around it | not yet |
| CMA-ES: `CMAESUtils.tql2` indexes past the end of its arrays (`ArrayIndexOutOfBoundsException`) when the covariance matrix holds NaN | the run aborts, often after about 200,000 evaluations | yes: the adapter catches the crash and starts a new attempt; both results are shown above | not yet |
| CMA-ES draws its samples from `new Random(System.currentTimeMillis())`, which a user can't set, so its runs can't be repeated | none: seeded by the adapter | the adapter replaces that generator by a seeded `java.util.Random`, by reflection, before each attempt ([`seedCmaes`](../../../benchmarks/adapters/jmetal/Bench.java#L633-L641)); the algorithm is unchanged | not yet |
| CMA-ES computes χₙ with integer divisions, `1 - 1 / (4 * n) + 1 / (21 * n * n)`, which is 1 in Java, so χₙ is √n instead of √n (1 − 1/(4n) + 1/(21n²)), 2.5% too large for n = 10 | σ shrinks a little faster than intended | no | not yet |
| `IntegerPermutationSolution` shuffles the initial permutation with `Collections.shuffle(list)`, whose generator ignores `JMetalRandom`'s seed, so runs on permutations can't be repeated | none: seeded by the adapter | the adapter's `createSolution()` makes the same uniform random permutation (Fisher-Yates) with `JMetalRandom` | not yet |
| Coral reef optimization creates `MersenneTwisterGenerator`s seeded with the clock, one of them inside `generateCoordinates()`, which no user can reach | it can't be seeded (rule 5.2): left out | no | not yet |
