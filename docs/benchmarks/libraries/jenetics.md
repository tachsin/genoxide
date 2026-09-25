# Jenetics (Java, 9.1.0)

Jenetics is a genetic algorithm, evolutionary algorithm and genetic programming library for Java 25. An `Engine` evolves a population through an `EvolutionStream`, which runs until the user limits it; the module `jenetics.ext` adds multi-objective selectors (`NSGA2Selector`, `UFTournamentSelector`) and more operators. Its documentation is the [user's manual](https://jenetics.io/manual/manual-9.1.0.pdf) (PDF, 9.1.0), the [README](https://github.com/jenetics/jenetics/blob/v9.1.0/README.md), the [javadoc](https://jenetics.io/javadoc/jenetics/9.1/io.jenetics.base/module-summary.html) and the example programs delivered with the library, [jenetics.example](https://github.com/jenetics/jenetics/tree/v9.1.0/jenetics.example/src/main/java/io/jenetics/example).

Adapter: [benchmarks/adapters/jenetics/](../../../benchmarks/adapters/jenetics/).
Know a better way to solve one of these problems with Jenetics? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs Jenetics

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), in Java, as the fitness functions of the Jenetics examples: of the decoded `double[]` or `int[]` of a codec, or of the `BitChromosome` ([Bench.java, lines 90-239](../../../benchmarks/adapters/jenetics/Bench.java#L90-L239)). `run.sh values <problem> <size>` evaluates solutions with them ([lines 718-745](../../../benchmarks/adapters/jenetics/Bench.java#L718-L745)).
- **Evaluations:** the adapter counts every call of the fitness function, and records there the first evaluation that reaches the target (rule 3.3) ([`Budget`, lines 245-310](../../../benchmarks/adapters/jenetics/Bench.java#L245-L310)). Jenetics calls it only for individuals it hasn't evaluated: survivors, and offspring the alterers left alone, keep their fitness. But a crossover replaces both mates by new individuals even when it changed one of them only (`Recombinator`), and `Mutator` replaces an individual it picked even when it changed none of its genes; those are evaluated again, as Jenetics' users get it.
- **Stop:** an evolution stream never ends by itself ("If you don't limit the stream, the EvolutionStream will not terminate and run forever", README). The adapter limits it with the target, the budget and the time cap, checked after every generation, so a run can go past the budget by one generation ([`evolve`, lines 385-420](../../../benchmarks/adapters/jenetics/Bench.java#L385-L420)). The examples also limit it: a generation limit such as `limit(100)` is a budget, and the scenario's budget replaces it; `Limits.bySteadyFitness(n)`, which the permutation and continuous examples set, is a convergence criterion, and ends an attempt (rule 2.2, see those problems).
- **Time:** the clock starts when the run's `Budget` is created, before the stream creates its random initial population, and stops when the run ends, before the output ([`runSingle`, lines 545-571](../../../benchmarks/adapters/jenetics/Bench.java#L545-L571)).
- **Seeds:** each run seeds the engine's generator, `RandomRegistry.random(...)` with Jenetics' default algorithm `L64X256MixRandom` (manual 1.4.2 and 2.7, "Reproducibility") ([lines 379-382](../../../benchmarks/adapters/jenetics/Bench.java#L379-L382)). The same seed gives the same run.
- **Solutions:** the best solution is kept when it's evaluated; a multi-objective run prints the non-dominated individuals of its final population with their genomes.
- **Bounds (rule 2.4):** a `DoubleGene` always lies in its range: `Mutator` draws a new value in it, `MeanAlterer` takes the mean of two values in it, `SimulatedBinaryCrossover` clamps to it, and the adapter's polynomial mutation clips to it. The continuous and multi-objective runs count the solutions evaluated outside the bounds (`outside`); it was 0 in every run below.

### One thread and the JIT (rule 4.3)

The engine gets `.executor(Runnable::run)` (manual 2.7), so selection, alteration and evaluation run on the calling thread; by default Jenetics uses the ForkJoin common pool. The JVM runs with `-XX:+UseSerialGC -Xbatch` ([run.sh](../../../benchmarks/adapters/jenetics/run.sh)): the serial collector collects on the calling thread, and `-Xbatch` (`-XX:-BackgroundCompilation`) makes the calling thread wait while the JIT compiles a method, so the JIT's compiler threads never run beside it. Before the timed runs, every solver runs once untimed and unprinted, with the seed 999,999, 50,000 evaluations and the scenario's time cap (rule 4.2, [lines 779-798](../../../benchmarks/adapters/jenetics/Bench.java#L779-L798)).

The flags measured on the same JVM invocation (Rastrigin 10, seeds 0 to 4, 500,000 evaluations, then ZDT1 with 25,000, after a warm-up of 1,000 evaluations; 2026-09-25, on a machine shared with other work, so the times are noisy):

| JVM flags | CPU / wall, Rastrigin 10 | CPU / wall, ZDT1 | Time of the 5 Rastrigin runs (s) |
|---|---|---|---|
| `-XX:+UseSerialGC` | 1.59 | 1.65 | 0.68, 0.42, 0.20, 0.27, 0.40 |
| `+ -XX:ActiveProcessorCount=1` | 1.37 | 1.65 | 0.78, 0.45, 0.22, 0.34, 0.44 |
| `+ -XX:-TieredCompilation -XX:CICompilerCount=1` | 1.26 | 1.48 | 0.93, 0.47, 0.19, 0.25, 0.47 |
| `+ -XX:TieredStopAtLevel=1` (C1 only) | 1.03 | 1.04 | 1.52, 1.20, 0.54, 0.68, 1.23 |
| **`+ -Xbatch`** (chosen) | **0.97** | **0.98** | 1.21, 0.45, 0.54, 0.25, 0.48 |

Only C1-only compilation and `-Xbatch` stay within 10%. C1 alone makes the code 2 to 3 times slower for good. `-Xbatch` keeps the same compilers and the same compiled code; its cost is that the compilations the warm-up didn't trigger happen inside the first timed run (seed 0 above, after the 1,000-evaluation warm-up of the time: 1.21 s against 0.68 s), and nowhere else.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched (OneMax 100 and 1000):** the matched GA with Jenetics' own components: population 300, `TournamentSelector(3)` for the parents (it draws with replacement), no survivors (`offspringFraction(1.0)`, so generational without elitism), two-point crossover (`MultiPointCrossover` with 2 points), `Mutator`, no maximal age ([lines 447-469](../../../benchmarks/adapters/jenetics/Bench.java#L447-L469)). Differences, where Jenetics can't express the setting exactly:
  - Jenetics' crossover picks each individual with probability p and mates it with a random other one, so p = 0.25 gives the matched expected number of crossovers (N / 2 pairs at 0.5).
  - Jenetics has no bit-flip mutation. `Mutator(p)` gives a bit a new random value, which flips it half the time, and picks the individual, the chromosome and the bit each with probability p^(1/3). p = 0.4 / n gives the matched expected number of flipped bits per child, 0.2. The flips come in bursts: with n = 100, 2.5% of the children get a new chromosome, with about 16 bits redrawn and 8 flipped each (the matched GA: 20% of the children, 1 flip each on average).
  - `Mutator` replaces the individuals it picks, and `Recombinator` both mates, even when a bit didn't change; those are evaluated again.
- **Idiomatic (OneMax 100), `ga`:** the README's "Hello World (Ones counting)" and the delivered example program [OnesCounting.java](https://github.com/jenetics/jenetics/blob/v9.1.0/jenetics.example/src/main/java/io/jenetics/example/OnesCounting.java): a `BitChromosome` and the engine defaults ([lines 470-485](../../../benchmarks/adapters/jenetics/Bench.java#L470-L485)). The defaults ([Engine.Builder](https://jenetics.io/javadoc/jenetics/9.1/io.jenetics.base/io/jenetics/engine/Engine.Builder.html)): population 50, `TournamentSelector(3)` for the offspring and the survivors, `SinglePointCrossover(0.2)` and `Mutator(0.15)`, 60% offspring, maximal age 70. The bits start random (the README's `BitChromosome.of(10, 0.5)`).

**How the docs decided:** three examples count ones, with no stated preference. The README and OnesCounting.java use the engine defaults; the manual's listing (section 6.1) sets population 500, `RouletteWheelSelector`, `Mutator(0.55)` and `SinglePointCrossover(0.06)`. The first listed is the Hello World: the first example of [jenetics.io](https://jenetics.io/) and of the README.

**Keeping going:** the README ends the stream with `limit(100)` and OnesCounting.java with `limit(10)`, generation limits, which the budget replaces. Neither sets a convergence criterion, so a run goes to the budget. The matched GA has none either.

**Left out:**
- The manual's settings (section 6.1): see above. In a separate test, run to the budget without the listing's `bySteadyFitness(7)`, they reached no target either (5 seeds, the best at the end 73, from 72 to 74).
- Evolution strategies (manual 2.9): the manual shows how to configure the engine as a (μ, λ) or (μ + λ) ES, without values for μ, λ or the mutation rate, and not for a problem type.

**Why the idiomatic run likely misses the target:** `Mutator(0.15)` gives each bit of an offspring a new random value with probability 0.15 (0.15^(1/3) for the individual, for its chromosome and for the bit), so it flips about 7.5% of the bits, 7 or 8 of 100. Near the optimum, nearly every mutated child loses ones, and with 50 individuals selection doesn't keep up.

**Separate tests** (2026-09-25, Jenetics 9.1.0, seeds 0 to 4, the scenario's budget, 60 s cap; the machine was shared with other work, so the capped runs depend on its load):

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

**Method, `ga`:** the manual's "Traveling salesman" example (section 6.5), Jenetics' permutation example: `Codecs.ofPermutation(n)`, population 500, maximal age 11, `SwapMutator(0.2)` and `PartiallyMatchedCrossover(0.35)`, the other settings the engine defaults (tournament of 3, 60% offspring) ([lines 486-510](../../../benchmarks/adapters/jenetics/Bench.java#L486-L510)).

**How the docs decided:** two examples solve a permutation problem with their own settings, with no stated preference: the manual's listing (section 6.5) and the delivered program [TravelingSalesman.java](https://github.com/jenetics/jenetics/blob/v9.1.0/jenetics.example/src/main/java/io/jenetics/example/TravelingSalesman.java), which sets `SwapMutator(0.15)` and `PartiallyMatchedCrossover(0.15)` and keeps the other defaults (population 50, maximal age 70). The engine defaults themselves don't fit a permutation (their crossover and `Mutator` break it). The one the documentation lists is the manual's; the example programs aren't listed anywhere but in the source tree.

**Keeping going:** the example ends the stream with `Limits.bySteadyFitness(25)` and `limit(250)`. The generation limit is a budget, lifted; the steady fitness is a convergence criterion the example sets, so an attempt ends after 25 generations without a better best, and the run starts again from a new random population (rule 2.2, [`evolve`, lines 390-420](../../../benchmarks/adapters/jenetics/Bench.java#L390-L420)). Attempt 0 uses the run's seed, restart r the seed (seed + 1) × 1,000,000 + r.

**Left out:**
- The delivered program's settings (see above). In a separate test, with its `limit(1_000)` lifted: N-Queens 32, 4 of 5 runs reached the target (median 50,934 evaluations); N-Queens 64, 2 of 5 (median 526,007).
- Other permutation operators of Jenetics (`ShiftMutator`, `ShuffleMutator`, `UniformOderBasedCrossover`, ...): no example uses them for a permutation problem.

**Separate tests** (2026-09-25, Jenetics 9.1.0, seeds 0 to 4):

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 4 | 149,864 | 0 | 0 | 1 | 0 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 0 | - | 3 | 1 | 3 | 0 |

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Method, `ga`:** the manual's "Rastrigin function" example (section 6.3), whose engine is the one of its "Real function" example (section 6.2, the delivered program [RealFunction.java](https://github.com/jenetics/jenetics/blob/v9.1.0/jenetics.example/src/main/java/io/jenetics/example/RealFunction.java)): `Codecs.ofVector(new DoubleRange(lower, upper), n)`, population 500, `Mutator(0.03)` and `MeanAlterer(0.6)`, the other settings the defaults ([lines 511-542](../../../benchmarks/adapters/jenetics/Bench.java#L511-L542)). Jenetics has no other recommendation for real functions, so Rosenbrock runs the same.

**Keeping going:** both examples end the stream with `Limits.bySteadyFitness(7)`: 7 generations without a better best. That's a convergence criterion, so it ends an attempt, and the run starts again from a new random population, with the seeds above, keeping the best and counting every evaluation (rule 2.2). Jenetics has no restart mechanism: `CyclicEngine` (manual 3.1.6.2) starts each stream from the previous one's population. The "Real function" example's `limit(100)` is a generation limit, lifted.

Before rule 2.2 was amended, the adapter ran without the criterion, with the optimum then at the old shift. Then, in separate tests (5 seeds): Rastrigin 10 reached the target in 4 of 5 runs (median 327,618 evaluations), and the best at the end was a median of 16.1 on Rastrigin 30, 1.05 on Rosenbrock and 2.04 on Ackley. With the restarts, below, no run reaches the target: 7 steady generations end most attempts long before the search has converged.

**Left out:**
- Evolution strategies (manual 2.9): no values for μ, λ or the mutation rate.
- Jenetics' other real-valued alterers (`GaussianMutator`, `LineCrossover`, `IntermediateCrossover`), described in the manual's alterer section (1.3.2.2): no example uses them for a real function.

**Separate tests** (2026-09-25, Jenetics 9.1.0, seeds 0 to 4; `outside` was 0 in every run):

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

**Method, `nsga2`:** NSGA-II with the matched settings, built from the engine and jenetics.ext's selectors ([lines 599-699](../../../benchmarks/adapters/jenetics/Bench.java#L599-L699)). Jenetics has no NSGA-II as such: its manual says its multi-objective classes extend the engine and "doesn't exactly follow an established algorithm, like NSGA2 or SPEA2" (3.1.7.2). The engine keeps "survivors" and adds "offspring" to them, so:
- the population is 2N (N = 100, 92 with 3 objectives), N survivors and N offspring: NSGA-II's parents and children;
- the survivors are chosen by `NSGA2Selector` (rank, then crowding distance): NSGA-II's selection of N from parents and children;
- the parents of the offspring are drawn from those same N by `UFTournamentSelector`, the crowded binary tournament of Fortin and Parizeau (2013);
- the final population is `NSGA2Selector`'s N from the last population of 2N, chosen after the clock stops;
- SBX with η 15 at 0.9 is Jenetics' `SimulatedBinaryCrossover`: it picks each individual with that probability and mates it with a random other one, and writes one child only, into one of the two mates; the other stays unchanged but is evaluated again. Jenetics has no polynomial mutation, so the adapter adds Deb's, η 20 at 1 / n per gene ([`PolynomialMutator`, lines 316-370](../../../benchmarks/adapters/jenetics/Bench.java#L316-L370)).

Differences from NSGA-II: the initial population has 2N random individuals; SBX makes one child per crossing; Jenetics computes the crowding distance over the whole population, not per front; `UFTournamentSelector` compares each pair by rank and crowding distance of the N parents, and picks among individuals of equal fitness at random.

The objectives are minimized with `VecFactory.ofDoubleVec(MINIMUM, ...)`, the manual's way to set each objective's direction (3.1.7.4), with the engine's default direction. The manual's DTLZ1 example (section 6.9) writes `Vec.of(...)` and minimizes with the engine, which meets a Jenetics bug (see below).

**Keeping going:** runs to the budget; each run uses its whole budget (rule 7.1).

**Not run (rule 6.1):**
- Jenetics' own multi-objective setup, the manual's DTLZ1 example (section 6.9): population 100, `SimulatedBinaryCrossover(1)`, `Mutator(1 / n)`, `TournamentSelector(5)` for the offspring and `NSGA2Selector` for the survivors. The matched scenarios run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA with their matched settings.
- The setup of the `MOEA` javadoc (`Mutator(0.1)`, `MeanAlterer`, `TournamentSelector(2)`, `UFTournamentSelector` for the survivors), which the adapter ran before as `moea`, for the same reason.
- NSGA-III, SPEA2, MOEA/D, SMS-EMOA: Jenetics doesn't have them.

**Separate tests** (2026-09-25, Jenetics 9.1.0, seeds 0 to 4; the hypervolume by run.py's code; `outside` was 0 in every run):

| Problem (budget) | Solver | Runs | Hypervolume: median | best | worst | Median evaluations | Capped |
|---|---|---|---|---|---|---|---|
| ZDT1 (25,000) | nsga2 | 5 | 0.8679 | 0.8685 | 0.8673 | 25,033 | 0 |
| ZDT2 (25,000) | nsga2 | 5 | 0.5341 | 0.5347 | 0.4819 | 25,025 | 0 |
| ZDT3 (25,000) | nsga2 | 5 | 1.3247 | 1.3264 | 1.2446 | 25,044 | 0 |
| DTLZ2 (25,000) | nsga2 | 5 | 0.4835 | 0.5270 | 0.4600 | 25,036 | 0 |
| DTLZ1 (40,000) | nsga2 | 5 | 0.0000 | 0.0000 | 0.0000 | 40,023 | 0 |

## Can't run

- Matched multi-objective: NSGA-III, SPEA2, MOEA/D and SMS-EMOA; Jenetics doesn't have them.

## Bugs found

| Bug | Effect here | Worked around | Reported |
|---|---|---|---|
| `SimulatedBinaryCrossover` centres the child on (a − b) / 2 instead of (a + b) / 2, and clamps it to the range | NSGA-II's DTLZ1 hypervolume is 0; its results show it | no | [jenetics/jenetics#969](https://github.com/jenetics/jenetics/issues/969) |
| With the engine minimizing (`.minimizing()`) and `Vec.of(...)`, the crowding distance is 0 for every individual but the extremes of each objective: `CrowdedComparator` reverses the element comparator for `Optimize.MINIMUM` but not the element distance, so `Pareto.crowdingDistance` sees a negative range (max − min ≤ 0) and adds nothing. `NSGA2Selector` and `UFTournamentSelector` then lose their diversity. The manual's DTLZ1 example is written this way | before the SBX rate was set to 0.9, NSGA-II, 5 seeds: ZDT1 0.236 instead of 0.868, DTLZ2 0.358 instead of 0.586 | the objectives are minimized through `VecFactory` (another documented way, manual 3.1.7.4), whose element distance follows the direction | not yet |
| `UFTournamentSelector` pairs the individuals by their position: `Subsets.next` returns its sample sorted, and when it samples the whole population (every call that selects half the population or more), the same fixed pairs are drawn in every round | small: before the SBX rate was set to 0.9, with the survivors shuffled before the tournament (separate test, 5 seeds), ZDT1 0.8679 and DTLZ2 0.552, against 0.8680 and 0.586 | no | not yet |
