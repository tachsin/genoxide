# Jenetics (Java, 9.1.0)

Jenetics is a genetic algorithm, evolutionary algorithm and genetic programming library for Java 25. An `Engine` evolves a population through an `EvolutionStream`, which runs until the user limits it; `jenetics.ext` adds multi-objective selectors (`NSGA2Selector`, `UFTournamentSelector`) and operators such as `SimulatedBinaryCrossover`. Its docs are the [user's manual](https://jenetics.io/manual/manual-9.1.0.pdf) (PDF, 9.1.0), the [README](https://github.com/jenetics/jenetics/blob/v9.1.0/README.md), the [javadoc](https://jenetics.io/javadoc/jenetics/9.1/io.jenetics.base/module-summary.html) and the example programs, [jenetics.example](https://github.com/jenetics/jenetics/tree/v9.1.0/jenetics.example/src/main/java/io/jenetics/example).

Adapter: [benchmarks/adapters/jenetics/](../../../benchmarks/adapters/jenetics/).
Know a better way to solve one of these problems with Jenetics? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs Jenetics

- **Fitness functions:** in Java, as in the Jenetics examples, of a codec's decoded `double[]` or `int[]`, or of the `BitChromosome` ([Bench.java, lines 77-157](../../../benchmarks/adapters/jenetics/Bench.java#L77-L157); `values`, [lines 451-475](../../../benchmarks/adapters/jenetics/Bench.java#L451-L475)).
- **Evaluations:** every call counts, with the first hit ([`Budget`, lines 163-245](../../../benchmarks/adapters/jenetics/Bench.java#L163-L245)). Survivors and untouched offspring keep their fitness. A crossover replaces both mates even when it changed one (`Recombinator`), and `Mutator` replaces an individual it picked even when no gene changed; those are evaluated again.
- **Stop:** a stream never ends by itself ("If you don't limit the stream, the EvolutionStream will not terminate", README). The adapter limits it at the target, the budget or the time cap, after every generation ([`evolve`, lines 259-302](../../../benchmarks/adapters/jenetics/Bench.java#L259-L302)).
- **Keeping going (rule 2.2):** an example's generation limit (`limit(100)`) is replaced by the budget. `Limits.bySteadyFitness(n)`, which the permutation and continuous examples set, ends an attempt, and the run starts again from a new random population with the seeds of rule 2.2 ([`evolve`, lines 268-302](../../../benchmarks/adapters/jenetics/Bench.java#L268-L302)). So does an attempt whose generations evaluate nothing for 10 in a row, and the run prints `restarts`. Jenetics has no restart mechanism: `CyclicEngine` (manual 3.1.6.2) continues from the previous population.
- **Bounds (rule 2.4):** a `DoubleGene` stays in its range: `Mutator` draws in it, `MeanAlterer` averages two values in it.
- **Seeds:** `RandomRegistry.random(...)` with the default `L64X256MixRandom` (manual 1.4.2 and 2.7), seeded per run ([lines 254-257](../../../benchmarks/adapters/jenetics/Bench.java#L254-L257)).
- **Time:** from the run's `Budget`, before the initial population, to the end of the run ([`runSingle`, lines 405-432](../../../benchmarks/adapters/jenetics/Bench.java#L405-L432)).
- **One thread and the JIT (rule 4.3):** `.executor(Runnable::run)` (manual 2.7) keeps everything on the calling thread (the default is the ForkJoin common pool). The JVM runs with `-XX:+UseSerialGC -Xbatch` ([run.sh](../../../benchmarks/adapters/jenetics/run.sh)): the serial collector, and the JIT compiling while the calling thread waits. Each solver first makes the warm-up run of rule 4.2 ([lines 505-524](../../../benchmarks/adapters/jenetics/Bench.java#L505-L524)).
- **Separate tests:** 2026-09-25, Jenetics 9.1.0, seeds 0 to 4, the scenario's budget, 60 s cap, on a shared machine. `outside` was 0 in every run.

JVM flags on one invocation (Rastrigin 10, seeds 0 to 4, 500,000 evaluations, after a 1,000-evaluation warm-up):

| JVM flags | CPU / wall, Rastrigin 10 | Time of the 5 Rastrigin runs (s) |
|---|---|---|
| `-XX:+UseSerialGC` | 1.59 | 0.68, 0.42, 0.20, 0.27, 0.40 |
| `+ -XX:ActiveProcessorCount=1` | 1.37 | 0.78, 0.45, 0.22, 0.34, 0.44 |
| `+ -XX:-TieredCompilation -XX:CICompilerCount=1` | 1.26 | 0.93, 0.47, 0.19, 0.25, 0.47 |
| `+ -XX:TieredStopAtLevel=1` (C1 only) | 1.03 | 1.52, 1.20, 0.54, 0.68, 1.23 |
| **`+ -Xbatch`** (chosen) | **0.97** | 1.21, 0.45, 0.54, 0.25, 0.48 |

Only C1-only and `-Xbatch` stay within 10%. C1-only makes the code 2 to 3 times slower throughout; `-Xbatch` keeps the same compiled code, and moves the compilations the warm-up didn't trigger into the first timed run.

## Binary: OneMax 100 (idiomatic)

**Methods:**
- **Matched:** not run (rule 6.1): Jenetics has no bit-flip mutation among its own components. `Mutator` redraws a bit, flipping it half the time ([lines 329-331](../../../benchmarks/adapters/jenetics/Bench.java#L329-L331)).
- **Idiomatic, `ga`:** the README's "Hello World (Ones counting)" and [OnesCounting.java](https://github.com/jenetics/jenetics/blob/v9.1.0/jenetics.example/src/main/java/io/jenetics/example/OnesCounting.java): a `BitChromosome` with random bits and the engine defaults ([lines 332-344](../../../benchmarks/adapters/jenetics/Bench.java#L332-L344); [Engine.Builder](https://jenetics.io/javadoc/jenetics/9.1/io.jenetics.base/io/jenetics/engine/Engine.Builder.html)): population 50, `TournamentSelector(3)`, `SinglePointCrossover(0.2)`, `Mutator(0.15)` (about 7.5% of the bits flipped), 60% offspring, maximal age 70. Of the three OneMax examples, with no stated preference, it's listed first, on [jenetics.io](https://jenetics.io/) and in the README.

**Keeping going:** the examples' `limit(100)` and `limit(10)` are replaced by the budget; no convergence criterion.

**Left out:**
- The manual's listing (section 6.1: population 500, `RouletteWheelSelector`, `Mutator(0.55)`, `SinglePointCrossover(0.06)`): listed after the Hello World. Without its `bySteadyFitness(7)` (5 seeds), it reached no target (best 73, from 72 to 74).
- Evolution strategies (manual 2.9): no values for μ, λ or the mutation rate, and no problem type.

**Separate tests:**

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 90 | 94 | 87 | 0 |

## Permutation: N-Queens 32 and 64

**Methods:** `ga`, the manual's "Traveling salesman" example (section 6.5): `Codecs.ofPermutation(n)`, population 500, maximal age 11, `SwapMutator(0.2)`, `PartiallyMatchedCrossover(0.35)`, the other engine defaults ([lines 346-370](../../../benchmarks/adapters/jenetics/Bench.java#L346-L370)). Of the two permutation examples, with no stated preference, the docs list only the manual's; the program [TravelingSalesman.java](https://github.com/jenetics/jenetics/blob/v9.1.0/jenetics.example/src/main/java/io/jenetics/example/TravelingSalesman.java) is only in the source tree. The engine's default crossover and `Mutator` break a permutation.

**Keeping going:** the example's `limit(250)` is lifted; `Limits.bySteadyFitness(25)` ends an attempt.

**Left out:**
- TravelingSalesman.java's settings (`SwapMutator(0.15)`, `PartiallyMatchedCrossover(0.15)`, population 50, maximal age 70). With its `limit(1_000)` lifted: N-Queens 32, 4 of 5 runs reached the target (median 50,934 evaluations); N-Queens 64, 2 of 5 (median 526,007).
- `ShiftMutator`, `ShuffleMutator`, `UniformOderBasedCrossover` and other permutation operators: no example uses them.

**Separate tests:**

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 4 | 149,864 | 0 | 0 | 1 | 0 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 3 | 1 | 3 | 0 |

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Methods:** `ga`, the manual's "Rastrigin function" example (section 6.3), with the engine of its "Real function" example (section 6.2, [RealFunction.java](https://github.com/jenetics/jenetics/blob/v9.1.0/jenetics.example/src/main/java/io/jenetics/example/RealFunction.java)): `Codecs.ofVector(new DoubleRange(lower, upper), n)`, population 500, `Mutator(0.03)`, `MeanAlterer(0.6)`, the other defaults ([lines 371-397](../../../benchmarks/adapters/jenetics/Bench.java#L371-L397)). Jenetics has no other recommendation for real functions, so Rosenbrock runs the same.

**Keeping going:** both examples set `Limits.bySteadyFitness(7)`, which ends an attempt; `limit(100)` is lifted.

**Left out:**
- Evolution strategies (manual 2.9): no values for μ, λ or the mutation rate.
- `GaussianMutator`, `LineCrossover`, `IntermediateCrossover` (manual 1.3.2.2): no example uses them for a real function.

**Separate tests:**

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 5.1 | 4.26 | 5.88 | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 127.3 | 109.6 | 132.0 | 0 |

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 7.5 | 1.68 | 11.6 | 0 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 11.9 | 11.5 | 12.4 | 0 |

## Can't run

- OneMax 100 and 1000, matched: no bit-flip mutation among Jenetics' own components.

## Bugs found

None in the single-objective scenarios.
