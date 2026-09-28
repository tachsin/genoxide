# jMetal (Java, 7.5)

jMetal is a Java framework for multi-objective optimization with metaheuristics, with single-objective algorithms too: "genetic algorithm (variants: generational, steady-state), evolution strategy (variants: elitist or mu+lambda, non-elitist or mu, lambda), DE, CMA-ES, PSO (Stantard 2007, Standard 2011), Coral reef optimization" ([documentation](https://jmetal.readthedocs.io/en/latest/), "Summary of features"). Its single-objective DE and CMA-ES are classes of `jmetal-algorithm`, each with an example program ([examples](https://github.com/jMetal/jMetal/tree/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples/singleobjective)).

Adapter: [benchmarks/adapters/jmetal/](../../../benchmarks/adapters/jmetal/).
Know a better way to run one of these methods with jMetal? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## What it runs

The matched suite: each problem with one method, defined the same for every library, and each library's own implementation of it.

| Scenario | Method | jMetal |
|---|---|---|
| Rastrigin 30, matched (no target, 300,000 evaluations) | DE/rand/1/bin | `de`: `DifferentialEvolution` |
| Rosenbrock 10, matched | CMA-ES | `cma_es`: `CovarianceMatrixAdaptationEvolutionStrategy`, with its bugs (rule 8.4) |
| OneMax 1000, matched | GA as DEAP's `eaSimple` | can't run: see [Can't run](#cant-run) |

The adapter prints nothing for any other scenario ([`solver`, lines 354-362](../../../benchmarks/adapters/jmetal/Bench.java#L354-L362)).

## How the adapter runs jMetal

- **Fitness functions:** in Java ([Bench.java, lines 62-105](../../../benchmarks/adapters/jmetal/Bench.java#L62-L105)), in a problem derived from `AbstractDoubleProblem`, as jMetal's users write their own ([`RealProblem`, lines 215-245](../../../benchmarks/adapters/jmetal/Bench.java#L215-L245); `values`, [lines 408-419](../../../benchmarks/adapters/jmetal/Bench.java#L408-L419)).
- **Evaluations:** every call of `evaluate()` counts, with the first hit and the solutions outside the bounds ([`Budget`, lines 120-209](../../../benchmarks/adapters/jmetal/Bench.java#L120-L209)).
- **Stop:** a run stops inside `evaluate()` at the target (Rosenbrock 10 only), the budget or the time cap (checked every 64 evaluations), with an exception the adapter catches; the algorithms' own maximum of evaluations is set to `Integer.MAX_VALUE`.
- **Generations (rule 2.3):** DE's are counted by its evaluator, jMetal's `SequentialSolutionListEvaluator` with a counter ([lines 247-265](../../../benchmarks/adapters/jmetal/Bench.java#L247-L265)); CMA-ES evaluates λ = 10 a generation.
- **Time:** from the run's `Budget`, before the initial population, to the end of the algorithm ([`run`, lines 364-392](../../../benchmarks/adapters/jmetal/Bench.java#L364-L392)).
- **Seeds (rule 5.2):** `JMetalRandom.getInstance().setSeed(seed)` before each run. The CMA-ES draws its samples from a generator of its own, seeded by the adapter (see [Bugs found](#bugs-found)). The same seed repeats a run, and seed 1 gives the same alone as after seed 0 (tested for both methods).
- **Solutions:** the best evaluated.
- **One thread and the JIT (rule 4.3):** jMetal evaluates sequentially. The JVM runs with `-XX:+UseSerialGC -Xbatch` ([run.sh](../../../benchmarks/adapters/jmetal/run.sh)): the serial collector, and the JIT compiling while the calling thread waits. The solver first makes the warm-up run of rule 4.2 ([`main`, lines 443-465](../../../benchmarks/adapters/jmetal/Bench.java#L443-L465)).
- **Separate tests:** 2026-09-28, jMetal 7.5, seeds 0 to 2, the scenario's budget, 60 s cap, one at a time on a shared machine. `outside` was 0 in every run.

The JVM flags were chosen on one invocation of the earlier suite's methods (Rastrigin 10: ga, de and cma_es, seeds 0 to 4, 500,000 evaluations, after a 1,000-evaluation warm-up; 2026-09-25):

| JVM flags | CPU / wall | ga, seeds 0-4 (s) | de, seeds 0-4 (s) | cma_es, seeds 0-4 (s) |
|---|---|---|---|---|
| `-XX:+UseSerialGC` | 1.14 | 0.11, 0.09, 0.06, 0.12, 0.09 | 0.18, 0.15, 0.18, 0.14, 0.15 | 1.08, 0.98, 1.01, 0.81, 0.80 |
| `+ -XX:ActiveProcessorCount=1` | 0.98 | 0.22, 0.16, 0.12, 0.09, 0.07 | 0.30, 0.23, 0.13, 0.13, 0.24 | 1.28, 1.16, 1.16, 1.20, 1.25 |
| `+ -XX:-TieredCompilation -XX:CICompilerCount=1` | 1.09 | 0.34, 0.23, 0.07, 0.08, 0.08 | 0.41, 0.15, 0.15, 0.13, 0.13 | 1.52, 0.96, 0.82, 0.81, 0.80 |
| `+ -XX:TieredStopAtLevel=1` (C1 only) | 1.01 | 0.13, 0.16, 0.09, 0.13, 0.12 | 0.32, 0.29, 0.31, 0.31, 0.28 | 1.43, 1.43, 1.39, 1.42, 1.43 |
| **`+ -Xbatch`** (chosen) | **0.99** | 0.33, 0.16, 0.07, 0.12, 0.08 | 0.36, 0.15, 0.13, 0.14, 0.13 | 0.98, 0.80, 0.80, 0.80, 0.71 |

C1-only makes DE and CMA-ES about 1.8 times slower throughout. `ActiveProcessorCount=1` stays within 10% here but not with Jenetics (1.37), and runs CMA-ES about 1.5 times slower. `-Xbatch` keeps the same compiled code, and moves the compilations the warm-up didn't trigger into the first timed runs. With the matched runs, CPU / wall was 1.08 for DE (1,000,000 evaluations, JVM startup included) and 0.98 for CMA-ES.

### Where the time goes

Audited on 2026-09-29, jMetal 7.5 as the adapter runs it: Java Flight Recorder over seeds 0 to 4, the JIT's and the collector's time per run from the JVM's management beans, and seeds 0 to 2 timed on an idle machine.

- **DE:** about 60% of the time is jMetal building an error message it doesn't use. The crossover's repair, `RepairDoubleSolutionWithBoundValue`, builds the message of its bounds check, `"The lower bound (" + lowerBound + ") is greater than ..."`, on every call, whether the check fails or not ([lines 21-22](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/solution/doublesolution/repairsolution/impl/RepairDoubleSolutionWithBoundValue.java#L21-L22)). That's two `Double.toString` for each gene of each trial, 18 million in a run. `DifferentialEvolutionCrossover` always uses that repair: it has no setting for it ([line 145](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/operator/crossover/impl/DifferentialEvolutionCrossover.java#L145)). So users get it too, and it isn't worked around (rule 8.4; see [Bugs found](#bugs-found)). Most of the rest is jMetal's DE itself: copying each child, `IntStream` loops over lists of boxed `Double`s, sorting the population. The adapter's `evaluate()` takes about 3% of the samples: copying the variables into a `double[]` as jMetal's own problems do (its [`Rastrigin`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-problem/src/main/java/org/uma/jmetal/problem/singleobjective/Rastrigin.java#L34-L41)), the counter, the bounds check and the fitness function.
- **CMA-ES:** jMetal's code. Sampling takes about 50% (`java.util.Random.nextGaussian` alone 26%), and the eigendecomposition, done every generation, 36%. `evaluate()` takes about 1.5%. `ObjectiveComparator` also builds an unused message on every comparison ([lines 55-58](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/util/comparator/ObjectiveComparator.java#L55-L58)): 1.5%.
- **Nothing to set:** the sequential evaluator is `DifferentialEvolution`'s default and its runner's. `DefaultDoubleSolution` is jMetal 7.5's only real-valued solution type, and nothing is logged or observed. Garbage collection took 5 to 15 ms of a run.
- **JIT in the first timed runs:** the warm-up of rule 4.2 (50,000 evaluations) doesn't reach every path a full run takes, so some compilation is left for the first timed runs, and `-Xbatch` runs it on the timed thread. Seed 0 compiled for 110 to 125 ms of its time with DE (about 10%) and 125 to 355 ms with CMA-ES. Seed 1 compiled for 40 to 70 ms, and later seeds for at most 45 ms. A warm-up with the scenario's whole budget brings seed 0's down to 35 to 70 ms. The median of 10 runs hardly moves.

| Seeds 0, 1, 2 | Time (s) | Result |
|---|---|---|
| DE, jMetal as it is | 1.03, 1.01, 0.98 | errors 122.8, 134.6, 129.6 |
| DE, the repair's message built only when its check fails (a copy of the class ahead of jMetal's on the class path, for the bug report only) | 0.42, 0.38, 0.37 | the same runs |
| CMA-ES | 0.43, 0.31, 0.27 | as in the separate tests below |

The adapter is unchanged: nothing in it adds more than those 3%.

## Rastrigin 30: DE/rand/1/bin

This scenario has no target: every run spends a fixed budget of 300,000 evaluations, and is measured by its time for the budget and its error at the end (its best value; the optimum is 0). The adapter never stops at a value, and prints `"target": null`, `"success": false` and `"first_hit": null`.

**Method** ([`de`, lines 271-294](../../../benchmarks/adapters/jmetal/Bench.java#L271-L294)): jMetal's [`DifferentialEvolution`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/differentialevolution/DifferentialEvolution.java), built as [DifferentialEvolutionRunner.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples/singleobjective/DifferentialEvolutionRunner.java) builds it (`DifferentialEvolutionSelection`, `DifferentialEvolutionCrossover` with `RAND_1_BIN`, population 100), with the definition's CR 0.9 instead of the example's 0.5. `jmetal-component` has no single-objective DE.

| Definition | jMetal | Source |
|---|---|---|
| NP = 100, uniform in the box | population 100; `createSolution()` draws each gene with `JMetalRandom.nextDouble(lower, upper)` | [`createInitialPopulation`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/differentialevolution/DifferentialEvolution.java#L73-L80), [`DefaultDoubleSolution`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/solution/doublesolution/impl/DefaultDoubleSolution.java#L30-L32) |
| r1, r2, r3 uniform, distinct, ≠ i | `DifferentialEvolutionSelection` draws indices uniformly until it has 3 distinct ones other than the target's | [lines 85-90](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/operator/selection/impl/DifferentialEvolutionSelection.java#L85-L90) |
| v = x_r1 + F (x_r2 − x_r3), F = 0.5 | `parent[2] + f * (parent[0] - parent[1])`, F fixed | [`randMutation`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/operator/crossover/impl/DifferentialEvolutionCrossover.java#L344-L346) |
| binomial, CR = 0.9, one forced j_rand | `jrand` uniform in [0, n − 1]; gene j from the mutant if U(0, 1) < CR or j = jrand, else from the target | [lines 279-297](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/operator/crossover/impl/DifferentialEvolutionCrossover.java#L279-L297) |
| u replaces x_i if f(u) ≤ f(x_i) | the target is kept only if it's strictly better (`compare(target, trial) < 0`) | [`replacement`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/differentialevolution/DifferentialEvolution.java#L106-L120) |
| generational | all trials are built from the population, evaluated together, then replace their targets | [`reproduction`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/differentialevolution/DifferentialEvolution.java#L90-L104), [`run`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/algorithm/impl/AbstractEvolutionaryAlgorithm.java#L48-L62) |
| no adaptation, archive, restarts or convergence criterion | none; its only stop is its maximum of evaluations | [`isStoppingConditionReached`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/differentialevolution/DifferentialEvolution.java#L69-L71) |

**Differences:**
- **Bounds (rule 2.4):** a trial's gene outside the box is set to the bound, by `RepairDoubleSolutionWithBoundValue`, the crossover's default repair ([line 145](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/operator/crossover/impl/DifferentialEvolutionCrossover.java#L145) and [line 312](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/operator/crossover/impl/DifferentialEvolutionCrossover.java#L312)); the reference redraws it uniformly in the box. It acts only on genes that leave the box, as the definition allows.
- **Order:** after each generation, jMetal sorts the population by fitness ([line 118](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/differentialevolution/DifferentialEvolution.java#L118)), so the targets are visited best first. Every individual is still a target once per generation, and r1, r2, r3 are uniform among the others, so the algorithm is the same; only which random numbers go with which individual differs.

**Keeping going:** its maximum of evaluations is lifted (rule 2.2); only the budget or the time cap ends a run. It evaluates every trial, so it never stalls.

**Separate tests:** Rastrigin 30 (budget 300,000, cap 60 s):

| Seed | Evaluations | Time (s) | Error at the end |
|---|---|---|---|
| 0 | 300,000 | 0.95 | 122.8 |
| 1 | 300,000 | 0.94 | 134.6 |
| 2 | 300,000 | 0.88 | 129.6 |

No run was capped. (With 2,000,000 evaluations, the error was 8.0 to 10.7 after about 6 s.)

## Rosenbrock 10: CMA-ES

**Method** ([`cmaes`, lines 296-335](../../../benchmarks/adapters/jmetal/Bench.java#L296-L335)): jMetal's [`CovarianceMatrixAdaptationEvolutionStrategy`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java) (CMA below), built with its `Builder` as [CovarianceMatrixAdaptationEvolutionStrategyRunner.java](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/examples/singleobjective/CovarianceMatrixAdaptationEvolutionStrategyRunner.java) does, with the Builder's options set to the definition: `setLambda(10)`, `setSigma(4.5)`, `setTypicalX(mean)`. It is Hansen's CMA-ES with positive weights only, and it runs here with its bugs (rule 8.4): they keep it from reaching the target (see [Bugs found](#bugs-found)).

| Definition | jMetal | Source |
|---|---|---|
| λ = 10, μ = 5 | λ 10 (its default too), μ = ⌊λ / 2⌋ = 5 | [CMA, line 126](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L126), [line 241](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L241) |
| w_i ∝ ln((λ + 1) / 2) − ln i, positive, sum 1 | ln(μ + 0.5) − ln i, the same for λ = 10: 0.456, 0.271, 0.162, 0.085, 0.026; μ_eff = 3.167; no negative weights | [lines 243-262](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L243-L262) |
| Hansen's learning rates and damping | Hansen's 2016 formulas exactly: c_σ = 0.2844, d_σ = 1.2844, c_c = 0.2950, c_1 = 0.01528, c_μ = 0.02015 (n = 10) | [lines 264-282](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L264-L282) |
| CSA, rank-one and rank-μ updates, h_σ stall | all of them, with the covariance matrix decomposed every generation here (every λ / (c_1 + c_μ) / n / 10 = 2.8 evaluations) | [lines 325-476](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L325-L476) |
| mean uniform in the box, σ₀ = 4.5, C₀ = I | `setTypicalX` with a point the adapter draws with `JMetalRandom.nextDouble(lower, upper)`; `setSigma(4.5)`; C₀ = I | [lines 229-236](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L229-L236) |
| no restarts, no convergence criterion | none, but two ways to end by itself (below) | |

**Differences:**
- **Bounds (rule 2.4):** CMA clips every sample to the box (`Bounds.restrict` in [`sampleSolution`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L497-L523)), and the updates use the clipped points; the reference is pycma's `BoundTransform`. It acts only on genes that leave the box.
- **Initial mean:** jMetal's default is a point in [0, 1)ⁿ whatever the bounds ([lines 229-236](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L229-L236)); the adapter sets the definition's uniform point in the box with the documented `setTypicalX`.
- **An unused initial population:** as every jMetal evolutionary algorithm, CMA first creates and evaluates an initial population ([`createInitialPopulation`](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L181-L188)): λ = 10 points uniform in the box, which its updates never use. They cost 10 evaluations, counted.
- **χₙ and h_σ:** χₙ = E‖N(0, I)‖ is computed with integer divisions, √n instead of 3.0847 for n = 10 (a bug, below), and the h_σ test uses (1 − c_σ)^(2(g + 2)) instead of (1 − c_σ)^(2(g + 1)), because its generation count includes the initial population ([line 392](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L392)). Both change the constants slightly, not the updates.
- **Seed:** CMA's sampling generator is seeded by the adapter (see [Bugs found](#bugs-found)); the algorithm is unchanged.

**Keeping going:** its maximum of evaluations, a budget, is lifted. CMA can end a run by itself in two ways, both in jMetal's code, and neither is worked around (rule 8.4): no restart, which the definition doesn't have, and which after a crash would be a workaround. The run ends there with the best value found, and its line says why in `ended_by`:
- `checkEigenCorrectness`: when the eigendecomposition fails its check, or an eigenvalue is negative, CMA sets its evaluations to the maximum and stops ([lines 478-495](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L478-L495)). It didn't happen in the separate tests.
- `tql2 ArrayIndexOutOfBoundsException`: the crash of the eigendecomposition once NaN reaches the covariance matrix (below). It ended every separate test, after about 204,000 evaluations.

Such a run ends there, before the budget, as not reached ([rule 8.4](../rules.md#8-reporting)): the results show it with the value it got to, and its output names the error in `ended_by`.

**Separate tests:** Rosenbrock 10 (budget 500,000, cap 60 s):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 3 | 0 | - | 9.71 | 8.24 | 19.13 | 0 |

Every run ended at the `tql2` crash, after 203,730 to 205,290 evaluations (0.3 to 0.4 s). Their best values were already reached within the first 20,000 evaluations: from then on σ grows without bound and every sample lands on the bounds.

For the bug report only, not in the benchmark: the same adapter with a copy of CMA whose one bug below is fixed (the square roots of the eigenvalues taken before they're used) reached the target with seeds 0 to 4 in 4,145 to 6,746 evaluations.

## Can't run

- **Matched OneMax 1000:** jMetal has no generational replacement without elitism (its replacements are (μ + λ), (μ, λ) with μ < λ, pairwise, random and the multi-objective ones; the classic `GenerationalGeneticAlgorithm` keeps 2 elites), and no two-point crossover for bits (`TwoPointCrossover` is for numbers).

## Bugs found

| Bug | Effect here | Worked around | Reported |
|---|---|---|---|
| CMA-ES: C^(−1/2) is wrong. `decomposeCovarianceMatrix` takes the square roots of the eigenvalues in the same loop that uses them ([lines 457-464](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L457-L464)): row i divides column j > i by the eigenvalue instead of its square root. Once the eigenvalues fall below 1, the step-size path is too long, σ grows, and the covariance matrix shrinks further. With jMetal's `Sphere(10)` and its runner's settings, σ is 480 after 3,000 evaluations and 9 × 10¹⁴⁹ after 100,000, while the best stays at 10⁻⁶. Taking all the square roots first fixes it | no Rosenbrock run reaches the target; every sample ends up on the bounds | no | [jMetal/jMetal#490](https://github.com/jMetal/jMetal/issues/490), fix proposed in [jMetal/jMetal#493](https://github.com/jMetal/jMetal/pull/493) |
| CMA-ES: `CMAESUtils.tql2` indexes past the end of its arrays (`ArrayIndexOutOfBoundsException`) when the covariance matrix holds NaN: its search for a small subdiagonal element never stops when the comparisons are false ([lines 162-181](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/util/CMAESUtils.java#L162-L181)) | the run ends after about 204,000 evaluations | no: the run ends there | [jMetal/jMetal#490](https://github.com/jMetal/jMetal/issues/490), fix proposed in [jMetal/jMetal#493](https://github.com/jMetal/jMetal/pull/493) |
| CMA-ES draws its samples from `new Random(System.currentTimeMillis())` ([line 106](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L106)), which a user can't set, so its runs can't be repeated | none: seeded by the adapter | the adapter replaces that generator by a `java.util.Random` seeded from `JMetalRandom`, by reflection, before the run ([`seedCmaes`, lines 337-345](../../../benchmarks/adapters/jmetal/Bench.java#L337-L345)); the algorithm is unchanged | [jMetal/jMetal#492](https://github.com/jMetal/jMetal/issues/492), fix proposed in [jMetal/jMetal#495](https://github.com/jMetal/jMetal/pull/495) |
| CMA-ES computes χₙ with integer divisions, `1 - 1 / (4 * n) + 1 / (21 * n * n)`, which is 1 in Java ([line 320](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-algorithm/src/main/java/org/uma/jmetal/algorithm/singleobjective/evolutionstrategy/CovarianceMatrixAdaptationEvolutionStrategy.java#L320)), so χₙ is √n instead of √n (1 − 1/(4n) + 1/(21n²)), 2.5% too large for n = 10 | σ shrinks a little faster than intended | no | [jMetal/jMetal#491](https://github.com/jMetal/jMetal/issues/491), fix proposed in [jMetal/jMetal#494](https://github.com/jMetal/jMetal/pull/494) |
| Speed: `RepairDoubleSolutionWithBoundValue.repairSolutionVariableValue` builds the message of its bounds check on every call, whether the check fails or not: two `Double.toString` for each gene it repairs ([lines 21-22](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/solution/doublesolution/repairsolution/impl/RepairDoubleSolutionWithBoundValue.java#L21-L22)). The other repairs, `RepairDoubleSolutionWithRandomValue` and `RepairDoubleSolutionWithOppositeBoundValue`, do the same, and so does `ObjectiveComparator` with its integers ([lines 55-58](https://github.com/jMetal/jMetal/blob/v7.5/jmetal-core/src/main/java/org/uma/jmetal/util/comparator/ObjectiveComparator.java#L55-L58)). Building the message only when the check fails fixes it | about 60% of the DE's time, which runs 2.6 times slower than it would ([Where the time goes](#where-the-time-goes)); the runs are the same | no: `DifferentialEvolutionCrossover` always uses that repair | not yet |

No bug was found in the DE's algorithm.

Found in methods no longer in the suite:
- `IntegerPermutationSolution` shuffles the initial permutation with `Collections.shuffle(list)`, whose generator ignores `JMetalRandom`'s seed, so runs on permutations can't be repeated. Reported: [jMetal/jMetal#492](https://github.com/jMetal/jMetal/issues/492), fix proposed in [jMetal/jMetal#495](https://github.com/jMetal/jMetal/pull/495).
- Coral reef optimization creates `MersenneTwisterGenerator`s seeded with the clock, one of them inside `generateCoordinates()`, which no user can reach, so it can't be seeded. Reported: [jMetal/jMetal#492](https://github.com/jMetal/jMetal/issues/492), fix proposed in [jMetal/jMetal#495](https://github.com/jMetal/jMetal/pull/495); its crash once a budget is set: [jMetal/jMetal#496](https://github.com/jMetal/jMetal/issues/496), fix proposed in [jMetal/jMetal#497](https://github.com/jMetal/jMetal/pull/497).
