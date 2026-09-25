# jMetal (Java, 7.5)

jMetal is a Java framework for multi-objective optimization with metaheuristics, which also has single-objective algorithms: "genetic algorithm (variants: generational, steady-state), evolution strategy (variants: elitist or mu+lambda, non-elitist or mu, lambda), DE, CMA-ES, PSO (Stantard 2007, Standard 2011), Coral reef optimization" (its [documentation](https://jmetal.readthedocs.io/en/latest/), "Summary of features"). Its algorithms come in two forms: the component-based builders of `jmetal-component`, which the documentation presents as the current design ([Component-based algorithms](https://jmetal.readthedocs.io/en/latest/component.html)), and the older classes of `jmetal-algorithm`. Each has example programs in its `examples` package ([jmetal-component](https://github.com/jMetal/jMetal/tree/v7.5/jmetal-component/src/main/java/org/uma/jmetal/component/examples), [jmetal-algorithm](https://github.com/jMetal/jMetal/tree/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples)).

Adapter: [benchmarks/adapters/jmetal/](../../../benchmarks/adapters/jmetal/).
Know a better way to solve one of these problems with jMetal? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs jMetal

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), in Java ([Bench.java, lines 99-260](../../../benchmarks/adapters/jmetal/Bench.java#L99-L260)), called from jMetal problems (`AbstractBinaryProblem`, `AbstractIntegerPermutationProblem`, `AbstractDoubleProblem`) as jMetal's users define their own ([lines 360-489](../../../benchmarks/adapters/jmetal/Bench.java#L360-L489)), not jMetal's own problem classes. `run.sh values <problem> <size>` evaluates solutions with them ([lines 912-939](../../../benchmarks/adapters/jmetal/Bench.java#L912-L939)).
- **Evaluations:** the adapter counts every call of `evaluate()` ([`Budget`, lines 264-347](../../../benchmarks/adapters/jmetal/Bench.java#L264-L347)), and records there the first evaluation that reaches the target (rule 3.3). jMetal evaluates every child, changed by the operators or not.
- **Stop:** a single-objective run stops inside `evaluate()`, at the target, the budget or the time cap (checked every 64 evaluations), with an exception the adapter catches, so it never goes past the budget. The algorithms' own evaluation limits are set out of reach. A multi-objective run stops through a `Termination` (SPEA2: its stopping condition) checked after every generation.
- **Time:** the clock starts when the run's `Budget` is created, before the algorithm creates its initial population, and stops when the algorithm ends, before the front is extracted ([`runSingle`, lines 686-712](../../../benchmarks/adapters/jmetal/Bench.java#L686-L712); [`runFront`](../../../benchmarks/adapters/jmetal/Bench.java#L786-L893)).
- **Seeds:** each run sets `JMetalRandom.getInstance().setSeed(seed)`, which jMetal's operators and algorithms draw from. Two parts of jMetal don't: its CMA-ES and its initial permutations (see Bugs found). The adapter seeds both, without changing what they draw: the same seed gives the same run.
- **Solutions:** the best solution is kept when it's evaluated; a multi-objective run prints the non-dominated solutions of the algorithm's `result()`: the final population (SPEA2: its archive of N).
- **Bounds (rule 2.4):** `SBXCrossover`, `PolynomialMutation` and `DifferentialEvolutionCrossover` repair a value outside the bounds to the bound (`RepairDoubleSolutionWithBoundValue`, their default); CMA-ES clips every sample to the bounds (`Bounds.restrict`). The continuous and multi-objective runs count the solutions evaluated outside the bounds (`outside`); it was 0 in every run below.

### One thread and the JIT (rule 4.3)

jMetal evaluates sequentially by default (`SequentialEvaluation`, `SequentialSolutionListEvaluator`), on the calling thread. The JVM runs with `-XX:+UseSerialGC -Xbatch` ([run.sh](../../../benchmarks/adapters/jmetal/run.sh)): the serial collector collects on the calling thread, and `-Xbatch` (`-XX:-BackgroundCompilation`) makes the calling thread wait while the JIT compiles a method, so the JIT's compiler threads never run beside it. Before the timed runs, every solver runs once untimed and unprinted, with the seed 999,999, 50,000 evaluations and the scenario's time cap (rule 4.2, [lines 973-995](../../../benchmarks/adapters/jmetal/Bench.java#L973-L995)).

The flags measured on one JVM invocation (Rastrigin 10: ga, de and cma_es, seeds 0 to 4, 500,000 evaluations, after a warm-up of 1,000 evaluations; 2026-09-25, on a machine shared with other work, so the times are noisy):

| JVM flags | CPU / wall | ga, seeds 0-4 (s) | de, seeds 0-4 (s) | cma_es, seeds 0-4 (s) |
|---|---|---|---|---|
| `-XX:+UseSerialGC` | 1.14 | 0.11, 0.09, 0.06, 0.12, 0.09 | 0.18, 0.15, 0.18, 0.14, 0.15 | 1.08, 0.98, 1.01, 0.81, 0.80 |
| `+ -XX:ActiveProcessorCount=1` | 0.98 | 0.22, 0.16, 0.12, 0.09, 0.07 | 0.30, 0.23, 0.13, 0.13, 0.24 | 1.28, 1.16, 1.16, 1.20, 1.25 |
| `+ -XX:-TieredCompilation -XX:CICompilerCount=1` | 1.09 | 0.34, 0.23, 0.07, 0.08, 0.08 | 0.41, 0.15, 0.15, 0.13, 0.13 | 1.52, 0.96, 0.82, 0.81, 0.80 |
| `+ -XX:TieredStopAtLevel=1` (C1 only) | 1.01 | 0.13, 0.16, 0.09, 0.13, 0.12 | 0.32, 0.29, 0.31, 0.31, 0.28 | 1.43, 1.43, 1.39, 1.42, 1.43 |
| **`+ -Xbatch`** (chosen) | **0.99** | 0.33, 0.16, 0.07, 0.12, 0.08 | 0.36, 0.15, 0.13, 0.14, 0.13 | 0.98, 0.80, 0.80, 0.80, 0.71 |

C1-only compilation makes DE and CMA-ES about 1.8 times slower for good. `ActiveProcessorCount=1` passed here but not with Jenetics (1.37), and ran CMA-ES about 1.5 times slower. `-Xbatch` keeps the same compilers and the same compiled code; its cost is that the compilations the warm-up didn't trigger happen inside the first timed runs (seed 0 of ga and de above, after the 1,000-evaluation warm-up of the time), and nowhere else.

## Binary: OneMax 100 (idiomatic)

**Methods:**
- **`ga`:** [GenerationalGeneticAlgorithmBinaryExample.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-component/src/main/java/org/uma/jmetal/component/examples/singleobjective/geneticalgorithm/GenerationalGeneticAlgorithmBinaryExample.java) (on OneMax 512): population 100, 100 children, binary tournament, `SinglePointCrossover(0.9)`, `BitFlipMutation(1 / n)`, (μ + λ) replacement, the builder's default ([lines 613-621](../../../benchmarks/adapters/jmetal/Bench.java#L613-L621)).
- **`es`:** [ElitistEvolutionStrategyRunner.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples/singleobjective/ElitistEvolutionStrategyRunner.java) (on OneMax 512): the elitist (μ + λ) evolution strategy with μ 1, λ 10 and `BitFlipMutation(1 / n)` ([lines 622-633](../../../benchmarks/adapters/jmetal/Bench.java#L622-L633)).

**How the docs decided:** the documentation states no preference among its single-objective algorithms, and lists them as GA, evolution strategy, DE, CMA-ES, PSO and coral reef optimization, each with its variants. The adapter takes, in that order, the algorithms that have an example for the problem type, in the first variant listed, with the example's settings, up to 3 (rule 6.4). For binary genomes, those are the generational GA, the elitist ES and coral reef optimization, which is left out (below).

**Keeping going:** the examples' limits are numbers of evaluations (`TerminationByEvaluations(25000)`, `setMaxEvaluations(25000)`), budgets, which the scenario's budget replaces. Neither method has a convergence criterion; both run to the budget.

**Left out:**
- Coral reef optimization ([CoralReefsOptimizationRunner.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples/singleobjective/CoralReefsOptimizationRunner.java), on OneMax): it can't be seeded. It draws from its own `MersenneTwisterGenerator`, seeded with the clock, and `generateCoordinates()` creates another one inside the method, which no user can reach, so a run can't be repeated (rule 5.2).
- The steady-state GA and the non-elitist (μ, λ) ES: the second variants of algorithms already run.
- Local search (`BasicLocalSearch`, LocalSearchBinaryProblemRunner.java) and random search: not among the algorithms the documentation lists.

**Not run: matched OneMax 100 and 1000** (rule 6.1, [lines 607-612](../../../benchmarks/adapters/jmetal/Bench.java#L607-L612)). jMetal has no generational replacement without elitism: its replacements are (μ + λ), (μ, λ) with μ < λ, pairwise, random and the multi-objective ones, and the classic `GenerationalGeneticAlgorithm` keeps 2 elites. It also has no two-point crossover for bits: `TwoPointCrossover` is for numeric solutions.

**Separate tests** (2026-09-25, jMetal 7.5, seeds 0 to 4, the scenario's budget, 60 s cap):

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 4,395 | 100 | 100 | 100 | 0 |
| es | 5 | 5 | 969 | 100 | 100 | 100 | 0 |

## Permutation: N-Queens 32 and 64

**Method, `ga`:** [GeneticAlgorithmTSPExample.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-component/src/main/java/org/uma/jmetal/component/examples/singleobjective/geneticalgorithm/GeneticAlgorithmTSPExample.java), with the settings of jmetal-algorithm's GenerationalGeneticAlgorithmTSPRunner.java too: population 100, 100 children, binary tournament, `PMXCrossover(0.9)`, `PermutationSwapMutation(1 / n)`, (μ + λ) replacement ([lines 636-647](../../../benchmarks/adapters/jmetal/Bench.java#L636-L647)). The initial permutations are random, drawn with `JMetalRandom` ([`createSolution`, lines 405-421](../../../benchmarks/adapters/jmetal/Bench.java#L405-L421)).

**How the docs decided:** the GA is the only algorithm with a permutation example.

**Keeping going:** the example's limit, `TerminationByEvaluations(250000)`, is a budget; the GA runs to the scenario's budget.

**Separate tests** (2026-09-25, jMetal 7.5, seeds 0 to 4):

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 3 | 30,621 | 0 | 0 | 2 | 0 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 1 | 242,375 | 1 | 0 | 2 | 0 |

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Methods** ([lines 648-683](../../../benchmarks/adapters/jmetal/Bench.java#L648-L683)):
- **`ga`:** [GenerationalGeneticAlgorithmExample.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-component/src/main/java/org/uma/jmetal/component/examples/singleobjective/geneticalgorithm/GenerationalGeneticAlgorithmExample.java) (on Sphere 20): population 100, 100 children, binary tournament, `SBXCrossover(0.9, 20)`, `PolynomialMutation(1 / n, 20)`, (μ + λ) replacement.
- **`de`:** [DifferentialEvolutionRunner.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples/singleobjective/DifferentialEvolutionRunner.java) (on Sphere 20): DE/rand/1/bin with CR 0.5 and F 0.5, population 100.
- **`cma_es`:** [CovarianceMatrixAdaptationEvolutionStrategyRunner.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples/singleobjective/CovarianceMatrixAdaptationEvolutionStrategyRunner.java) (on Sphere): the builder's defaults, λ 10 and σ 0.3 ([`cmaes`, lines 540-596](../../../benchmarks/adapters/jmetal/Bench.java#L540-L596)). Difference: jMetal starts the mean at a random point in [0, 1)ⁿ whatever the bounds; the adapter starts it at a random point within the bounds, like the other libraries' CMA-ES.

jMetal documents no method for multimodal or unimodal functions in particular, so all four problems run the same three.

**How the docs decided:** as for the binary problems: in the order of the documentation's list, the algorithms with an example on a continuous problem are the GA, DE, CMA-ES and the PSOs; the first 3 run (rule 6.4). The evolution strategy has examples on OneMax only.

**Keeping going:** the GA and DE have no convergence criterion; their limits of evaluations are lifted, and they run to the budget. CMA-ES ends an attempt by itself when its covariance matrix degenerates and its eigendecomposition fails its check (`checkEigenCorrectness` sets the evaluations to the maximum). jMetal has no restart mechanism, so the adapter starts it again from a new random point, keeping the best and counting every evaluation (rule 2.2). Attempt 0 uses the run's seed, restart r seeds `JMetalRandom` with (seed + 1) × 1,000,000 + r.

**Workaround of a crash (rule 8.4):** when the covariance matrix holds NaN, `CMAESUtils.tql2` throws `ArrayIndexOutOfBoundsException` and the run aborts. That's a crash, not a convergence criterion. The adapter catches it and starts a new attempt, as after a convergence. The results below show CMA-ES both ways: as the benchmark runs it, with the workaround, and as it is, where the run ends at the crash with the best value found so far (`JMETAL_CMAES_AS_IS=1`).

**Left out:**
- Standard PSO 2007 and 2011 (StandardPSO2007Runner.java, StandardPSO2011Runner.java, on Sphere): fourth in the documentation's order; at most 3 methods run.
- The steady-state GA: the second variant of the GA.
- Local search (`BasicLocalSearch` with polynomial mutation, LocalSearchContinuousProblemRunner.java) and random search: not among the algorithms the documentation lists.

**Why CMA-ES misses every target:** a jMetal bug (below) makes its step size grow without bound once it has converged. From then on, every sample lands on the bounds, and the attempt stalls without ending, until `tql2` throws, often after about 200,000 evaluations. The adapter doesn't restart it earlier: that would work around the bug (rule 8.4).

**Separate tests** (2026-09-25, jMetal 7.5, seeds 0 to 4; `outside` was 0 in every run; the machine was shared with other runs, so which runs the time cap stopped depends on its load). `cma_es (as-is)` is CMA-ES without the workaround: a run ends at the first crash of `tql2`. On Rastrigin 30 and Ackley 30 the time cap stops most runs before any crash, so both rows are nearly the same.

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

**Methods** (the matched settings; [lines 784-893](../../../benchmarks/adapters/jmetal/Bench.java#L784-L893)):
- **`nsga2`:** `NSGAIIBuilder`, N parents and N children (N = 100, 92 with 3 objectives), SBX η 15 at 0.9, polynomial mutation η 20 at 1 / n.
- **`nsga3`:** `NSGAIIIBuilder` with Das-Dennis directions (99 divisions: 100 directions with 2 objectives; 12: 91 with 3), the population rounded up to a multiple of 4 by jMetal (100, 92), SBX η 30 at 1, polynomial mutation η 20 at 1 / n.
- **`spea2`:** jmetal-algorithm's `SPEA2` (it has no builder in jmetal-component): population and archive N, binary tournament, SBX η 15 at 0.9, polynomial mutation η 20 at 1 / n, k 1 ([`BudgetSPEA2`, lines 741-759](../../../benchmarks/adapters/jmetal/Bench.java#L741-L759)).
- **`moead`:** `MOEADBuilder` with 100 weight vectors (91 Das-Dennis vectors with 3 objectives), 20 neighbors, parents from the neighborhood with probability 0.9, at most 2 replacements (jMetal's default), Tchebycheff (PBI with θ 5 for DTLZ), SBX η 20 at 1, polynomial mutation η 20 at 1 / n; one child per step.
- **`sms_emoa`:** `SMSEMOABuilder`, population N, one child per step, SBX η 15 at 0.9, polynomial mutation η 20 at 1 / n.

**Keeping going:** each runs to its budget (rule 7.1).

**Not run (rule 6.1):** SMPSO, jMetal's own multi-objective particle swarm, which the adapter ran before; and jMetal's other multi-objective algorithms (GDE3, MOCell, IBEA, PAES, RVEA, AGE-MOEA, NSGA-II and SMS-EMOA with differential evolution, ...). The matched scenarios run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA.

**Separate tests** (2026-09-25, jMetal 7.5, seeds 0 to 4; the hypervolume by run.py's code; `outside` was 0 in every run):

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

SMS-EMOA with 3 objectives is the slowest: in these tests, on a machine shared with other runs, its DTLZ1 runs reached the 60 s cap after a median of 28,340 of their 40,000 evaluations.

## Can't run

- Matched OneMax 100 and 1000: jMetal has no generational replacement without elitism and no two-point crossover for bits (see above).

## Bugs found

| Bug | Effect here | Worked around | Reported |
|---|---|---|---|
| CMA-ES: once it has converged, its step size σ grows without bound. With jMetal's own `Sphere(10)` and its runner's settings, σ is 480 after 3,000 evaluations and 9 × 10¹⁴⁹ after 100,000, while the best stays at 10⁻⁶. Every sample then lands on the bounds | no CMA-ES run reaches a target; an attempt stalls until `tql2` throws | no | not yet |
| CMA-ES: `CMAESUtils.tql2` indexes past the end of its arrays (`ArrayIndexOutOfBoundsException`) when the covariance matrix holds NaN | the run aborts, often after about 200,000 evaluations | yes: the adapter catches the crash and starts a new attempt; both results are shown above | not yet |
| CMA-ES draws its samples from `new Random(System.currentTimeMillis())`, which a user can't set, so its runs can't be repeated | none: seeded by the adapter | the adapter replaces that generator by a seeded `java.util.Random`, by reflection, before each attempt ([`seedCmaes`](../../../benchmarks/adapters/jmetal/Bench.java#L588-L596)); the algorithm is unchanged | not yet |
| CMA-ES computes χₙ with integer divisions, `1 - 1 / (4 * n) + 1 / (21 * n * n)`, which is 1 in Java, so χₙ is √n instead of √n (1 − 1/(4n) + 1/(21n²)), 2.5% too large for n = 10 | σ shrinks a little faster than intended | no | not yet |
| `IntegerPermutationSolution` shuffles the initial permutation with `Collections.shuffle(list)`, whose generator ignores `JMetalRandom`'s seed, so runs on permutations can't be repeated | none: seeded by the adapter | the adapter's `createSolution()` makes the same uniform random permutation (Fisher-Yates) with `JMetalRandom` | not yet |
| Coral reef optimization creates `MersenneTwisterGenerator`s seeded with the clock, one of them inside `generateCoordinates()` | it can't run here (rule 5.2): left out | no | not yet |
