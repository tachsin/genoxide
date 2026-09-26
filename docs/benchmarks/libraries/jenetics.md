# Jenetics (Java, 9.1.0)

Jenetics is a genetic algorithm, evolutionary algorithm and genetic programming library for Java 25. An `Engine` evolves a population through an `EvolutionStream`, which runs until the user limits it; `jenetics.ext` adds multi-objective selectors (`NSGA2Selector`, `UFTournamentSelector`) and operators such as `SimulatedBinaryCrossover`. Its docs are the [user's manual](https://jenetics.io/manual/manual-9.1.0.pdf) (PDF, 9.1.0), the [README](https://github.com/jenetics/jenetics/blob/v9.1.0/README.md), the [javadoc](https://jenetics.io/javadoc/jenetics/9.1/io.jenetics.base/module-summary.html) and the example programs, [jenetics.example](https://github.com/jenetics/jenetics/tree/v9.1.0/jenetics.example/src/main/java/io/jenetics/example).

Adapter: [benchmarks/adapters/jenetics/](../../../benchmarks/adapters/jenetics/).
Know a better way to solve one of these problems with Jenetics? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs Jenetics

- **Fitness functions:** in Java, as in the Jenetics examples, of a codec's decoded `double[]` or `int[]`, or of the `BitChromosome` ([Bench.java, lines 79-159](../../../benchmarks/adapters/jenetics/Bench.java#L79-L159); `values`, [lines 469-493](../../../benchmarks/adapters/jenetics/Bench.java#L469-L493)).
- **Evaluations:** every call counts, with the first hit ([`Budget`, lines 165-247](../../../benchmarks/adapters/jenetics/Bench.java#L165-L247)). Survivors and untouched offspring keep their fitness. A crossover replaces both mates even when it changed one (`Recombinator`), and `Mutator` replaces an individual it picked even when no gene changed; those are evaluated again.
- **Stop:** a stream never ends by itself ("If you don't limit the stream, the EvolutionStream will not terminate", README). The adapter limits it at the target, the budget or the time cap, after every generation ([`evolve`, lines 262-298](../../../benchmarks/adapters/jenetics/Bench.java#L262-L298)).
- **Keeping going (rule 2.2):** an example's generation limit (`limit(100)`) is replaced by the budget. `Limits.bySteadyFitness(n)`, which the permutation and continuous examples set, ends an attempt, and the run starts again from a new random population with the seeds of rule 2.2 ([`evolve`, lines 267-298](../../../benchmarks/adapters/jenetics/Bench.java#L267-L298)). Jenetics has no restart mechanism: `CyclicEngine` (manual 3.1.6.2) continues from the previous population.
- **Bounds (rule 2.4):** a `DoubleGene` stays in its range: `Mutator` draws in it, `MeanAlterer` averages two values in it.
- **Seeds:** `RandomRegistry.random(...)` with the default `L64X256MixRandom` (manual 1.4.2 and 2.7), seeded per run ([lines 256-259](../../../benchmarks/adapters/jenetics/Bench.java#L256-L259)).
- **Time:** from the run's `Budget`, before the initial population, to the end of the run ([`runSingle`, lines 423-450](../../../benchmarks/adapters/jenetics/Bench.java#L423-L450)).
- **One thread and the JIT (rule 4.3):** `.executor(Runnable::run)` (manual 2.7) keeps everything on the calling thread (the default is the ForkJoin common pool). The JVM runs with `-XX:+UseSerialGC -Xbatch` ([run.sh](../../../benchmarks/adapters/jenetics/run.sh)): the serial collector, and the JIT compiling while the calling thread waits. Each solver first makes the warm-up run of rule 4.2 ([lines 523-542](../../../benchmarks/adapters/jenetics/Bench.java#L523-L542)).
- **Separate tests:** 2026-09-25, Jenetics 9.1.0, seeds 0 to 4, the scenario's budget, 60 s cap, on a shared machine. `outside` was 0 in every run.

JVM flags on one invocation (Rastrigin 10, seeds 0 to 4, 500,000 evaluations, then ZDT1 with 25,000, after a 1,000-evaluation warm-up):

| JVM flags | CPU / wall, Rastrigin 10 | CPU / wall, ZDT1 | Time of the 5 Rastrigin runs (s) |
|---|---|---|---|
| `-XX:+UseSerialGC` | 1.59 | 1.65 | 0.68, 0.42, 0.20, 0.27, 0.40 |
| `+ -XX:ActiveProcessorCount=1` | 1.37 | 1.65 | 0.78, 0.45, 0.22, 0.34, 0.44 |
| `+ -XX:-TieredCompilation -XX:CICompilerCount=1` | 1.26 | 1.48 | 0.93, 0.47, 0.19, 0.25, 0.47 |
| `+ -XX:TieredStopAtLevel=1` (C1 only) | 1.03 | 1.04 | 1.52, 1.20, 0.54, 0.68, 1.23 |
| **`+ -Xbatch`** (chosen) | **0.97** | **0.98** | 1.21, 0.45, 0.54, 0.25, 0.48 |

Only C1-only and `-Xbatch` stay within 10%. C1-only makes the code 2 to 3 times slower throughout; `-Xbatch` keeps the same compiled code, and moves the compilations the warm-up didn't trigger into the first timed run.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched:** Jenetics' own components: population 300, `TournamentSelector(3)` for the parents (with replacement), no survivors (`offspringFraction(1.0)`), `MultiPointCrossover` with 2 points, `Mutator`, no maximal age ([lines 325-347](../../../benchmarks/adapters/jenetics/Bench.java#L325-L347)). Differences:
  - the crossover picks each individual with probability p and mates it with a random other; p = 0.25 gives the matched N / 2 pairs at 0.5;
  - there's no bit flip: `Mutator(p)` redraws a bit (flipping it half the time), picking the individual, chromosome and bit each with probability p^(1/3). p = 0.4 / n gives the matched mean of 0.2 flipped bits per child, in bursts: with n = 100, 2.5% of the children get about 16 bits redrawn and 8 flipped;
  - picked or mated individuals are evaluated again even when unchanged.
- **Idiomatic, `ga`:** the README's "Hello World (Ones counting)" and [OnesCounting.java](https://github.com/jenetics/jenetics/blob/v9.1.0/jenetics.example/src/main/java/io/jenetics/example/OnesCounting.java): a `BitChromosome` with random bits and the engine defaults ([lines 348-363](../../../benchmarks/adapters/jenetics/Bench.java#L348-L363); [Engine.Builder](https://jenetics.io/javadoc/jenetics/9.1/io.jenetics.base/io/jenetics/engine/Engine.Builder.html)): population 50, `TournamentSelector(3)`, `SinglePointCrossover(0.2)`, `Mutator(0.15)` (about 7.5% of the bits flipped), 60% offspring, maximal age 70. Of the three OneMax examples, with no stated preference, it's listed first, on [jenetics.io](https://jenetics.io/) and in the README.

**Keeping going:** the examples' `limit(100)` and `limit(10)` are replaced by the budget; no convergence criterion.

**Left out:**
- The manual's listing (section 6.1: population 500, `RouletteWheelSelector`, `Mutator(0.55)`, `SinglePointCrossover(0.06)`): listed after the Hello World. Without its `bySteadyFitness(7)` (5 seeds), it reached no target (best 73, from 72 to 74).
- Evolution strategies (manual 2.9): no values for μ, λ or the mutation rate, and no problem type.

**Separate tests:**

OneMax 100, matched (budget 200,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 4 | 87,834 | 100 | 100 | 99 | 0 |

OneMax 1000, matched (budget 2,000,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 775 | 781 | 757 | 5 |

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 90 | 94 | 87 | 0 |

## Permutation: N-Queens 32 and 64

**Methods:** `ga`, the manual's "Traveling salesman" example (section 6.5): `Codecs.ofPermutation(n)`, population 500, maximal age 11, `SwapMutator(0.2)`, `PartiallyMatchedCrossover(0.35)`, the other engine defaults ([lines 364-388](../../../benchmarks/adapters/jenetics/Bench.java#L364-L388)). Of the two permutation examples, with no stated preference, the docs list only the manual's; the program [TravelingSalesman.java](https://github.com/jenetics/jenetics/blob/v9.1.0/jenetics.example/src/main/java/io/jenetics/example/TravelingSalesman.java) is only in the source tree. The engine's default crossover and `Mutator` break a permutation.

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

**Methods:** `ga`, the manual's "Rastrigin function" example (section 6.3), with the engine of its "Real function" example (section 6.2, [RealFunction.java](https://github.com/jenetics/jenetics/blob/v9.1.0/jenetics.example/src/main/java/io/jenetics/example/RealFunction.java)): `Codecs.ofVector(new DoubleRange(lower, upper), n)`, population 500, `Mutator(0.03)`, `MeanAlterer(0.6)`, the other defaults ([lines 389-415](../../../benchmarks/adapters/jenetics/Bench.java#L389-L415)). Jenetics has no other recommendation for real functions, so Rosenbrock runs the same.

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

## Multi-objective: ZDT1, ZDT2, ZDT3 (30 variables), DTLZ2 and DTLZ1 (3 objectives)

**Not run (rule 6.1).** Jenetics has no polynomial mutation (its mutators: `Mutator`, `GaussianMutator`, `SwapMutator`, `ShiftMutator`, `ShuffleMutator`, and in jenetics.ext `HPRMutator`, `RSMutator` and the tree and weasel mutators), and no NSGA-III, SPEA2, MOEA/D or SMS-EMOA.

An NSGA-II can be built from the engine, `NSGA2Selector`, `UFTournamentSelector` and `SimulatedBinaryCrossover`, but only with a polynomial mutation written by the adapter. With one, and SBX η 15 at 0.9 (5 seeds, run.py's hypervolume), for reference:

| Problem (budget) | Solver | Runs | Hypervolume: median | best | worst | Median evaluations |
|---|---|---|---|---|---|---|
| ZDT1 (25,000) | nsga2 | 5 | 0.8679 | 0.8685 | 0.8673 | 25,033 |
| ZDT2 (25,000) | nsga2 | 5 | 0.5341 | 0.5347 | 0.4819 | 25,025 |
| ZDT3 (25,000) | nsga2 | 5 | 1.3247 | 1.3264 | 1.2446 | 25,044 |
| DTLZ2 (25,000) | nsga2 | 5 | 0.4835 | 0.5270 | 0.4600 | 25,036 |
| DTLZ1 (40,000) | nsga2 | 5 | 0 | 0 | 0 | 40,023 |

**Left out:** Jenetics' own multi-objective setups, the manual's DTLZ1 example (section 6.9: population 100, `SimulatedBinaryCrossover(1)`, `Mutator(1 / n)`, `TournamentSelector(5)`, `NSGA2Selector`) and the `MOEA` javadoc's (`Mutator(0.1)`, `MeanAlterer`, `TournamentSelector(2)`, `UFTournamentSelector`): not the matched algorithms.

## Can't run

- The 5 multi-objective scenarios: no polynomial mutation, and no NSGA-III, SPEA2, MOEA/D or SMS-EMOA.

## Bugs found

| Bug | Effect here | Worked around | Reported |
|---|---|---|---|
| `SimulatedBinaryCrossover` centres the child on (a − b) / 2 instead of (a + b) / 2, and clamps it to the range | none, as the multi-objective scenarios don't run; the NSGA-II above reaches a hypervolume of 0 on DTLZ1 | no | [jenetics/jenetics#969](https://github.com/jenetics/jenetics/issues/969) |
| With `.minimizing()` and `Vec.of(...)`, the crowding distance is 0 for all but each objective's extremes: `CrowdedComparator` reverses the element comparator for `Optimize.MINIMUM` but not the element distance, so `Pareto.crowdingDistance` sees a range max − min ≤ 0 and adds nothing. `NSGA2Selector` and `UFTournamentSelector` lose their diversity. The manual's DTLZ1 example is written this way | none, as the multi-objective scenarios don't run; the NSGA-II above with SBX at 0.45 (5 seeds): ZDT1 0.236 instead of 0.868, DTLZ2 0.358 instead of 0.586 | the NSGA-II above minimizes through `VecFactory` (manual 3.1.7.4), whose distance follows the direction | not yet |
| `UFTournamentSelector` pairs individuals by position: `Subsets.next` returns its sample sorted, so when it samples the whole population (selecting half of it or more), the same pairs are drawn every round | none, as the multi-objective scenarios don't run; small: with the survivors shuffled first (SBX at 0.45, 5 seeds), ZDT1 0.8679 and DTLZ2 0.552, against 0.8680 and 0.586 | no | not yet |
