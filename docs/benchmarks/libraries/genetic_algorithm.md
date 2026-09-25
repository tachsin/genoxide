# genetic_algorithm (Rust, 0.27.3)

A Rust genetic algorithm library with three strategies: Evolve (a GA), HillClimb (local search, Stochastic or SteepestAscent) and Permutate (exhaustive search). The fitness is an `isize`, so real values are scaled by a precision. Its docs are the [README](https://docs.rs/crate/genetic_algorithm/0.27.3/source/README.md), [AGENTS.md](https://docs.rs/crate/genetic_algorithm/0.27.3/source/AGENTS.md) (decision matrices and recommended settings), [AGENTS_TEMPLATES.md](https://docs.rs/crate/genetic_algorithm/0.27.3/source/AGENTS_TEMPLATES.md), the [examples](https://docs.rs/crate/genetic_algorithm/0.27.3/source/examples/) and [docs.rs](https://docs.rs/genetic_algorithm/0.27.3/genetic_algorithm/).

Adapter: [benchmarks/adapters/genetic_algorithm/](../../../benchmarks/adapters/genetic_algorithm/).
Know a better way to solve one of these problems with genetic_algorithm? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How every run works

- **Evaluations** are counted in the fitness functions ([main.rs#L143](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L143)), every call.
- **The end of a run:** the strategy's own `with_target_fitness_score`, or the abort flag (`with_abort_flag`, checked once per generation), which the adapter sets when the budget or the time is used up ([#L42](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L42)). No run uses `with_max_generations`, a limit that's only a budget.
- **Real values:** the fitness is `isize`, so a value is divided by the precision 1e-5 of AGENTS.md and the examples ([#L174](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L174)). It's rounded up instead of truncated, so that the library's target of 1000 units means a value of at most 0.01. The reported `best` is the f64 value of the best genes, not the scaled score.
- **Restarts (rule 2.2):** `with_max_stale_generations` detects convergence (generations without improvement), so it ends an attempt and the method starts again. The library's restart mechanism is `call_repeatedly(n)` (AGENTS.md, "Choosing a call variant"). With `with_rng_seed_from_u64`, every repeat gets the same seed and runs the same search again (see "Bugs found"). So the adapter's `restarts` ([#L274](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L274)) does what `call_repeatedly` does, from a new random start with the seed `seed * 1000 + restart`: attempts one after the other until one is conclusive (the target, or the abort flag), keeping the best genes and counting every evaluation. No run in the tests below needed more than 43 restarts.
- **Bounds (rule 2.4):** the library's own handling. A `RangeGenotype` draws its initial genes from the allele range and clamps every mutation to it ([genotype/range.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/src/genotype/range.rs), `clamped_add` and `clamped_sub`); uniform crossover only exchanges genes. The fitness counts the evaluated solutions outside the bounds as the library proposed them ([#L186](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L186)), and each continuous run reports the count as `outside`: 0 in every run.
- **One thread:** no `with_par_fitness` and no `call_par_*`. The only other thread is the timer, which sleeps. Measured: 0.90 s of CPU for 0.90 s of wall time (Rosenbrock, 10 seeds), and 0.98 over all of `run.py check`'s runs.
- **Seeds:** `with_rng_seed_from_u64`.

## Binary: OneMax 100 and 1000

**Methods:**
- Matched (OneMax 100 and 1000, [#L332](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L332)): Evolve with population 300, `SelectTournament(0.5, 0.0, 3)`, `CrossoverMultiPoint(1.0, 0.5, 2, false)` (two-point crossover at 0.5) and `MutateSingleGene(0.2)`. Differences from DEAP's `eaSimple`:
  - the library selects the survivors from parents and offspring by a tournament without replacement, not the parents by a tournament with replacement, so the pressure comes from the surplus of offspring. A replacement rate of 1.0 would keep all 300 offspring and select nothing, hence 0.5;
  - `MutateSingleGene(0.2)` mutates exactly one bit of 20% of the children, where a bit flip at 1/n flips one bit on average.
- Idiomatic (OneMax 100, [#L342](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L342)): Evolve with the binary preset of AGENTS.md "If unsure, start here" (a stated preference): `SelectTournament(0.5, 0.02, 4)`, `CrossoverUniform(0.7, 0.8)`, `MutateSingleGene(0.2)`. The preset leaves the population to the user; 100 is the default, and also the README's "Quick Usage" example, which is this problem (100 genes, count the ones, target 100).

**Keeping going:** the only ending condition is the target, so the run goes to the target or the budget.

**Left out:**
- HillClimb: the README recommends it for a "convex search space, few local optima". OneMax is one, but a binary problem in general isn't, and the benchmark doesn't pick methods by knowing the answer.
- `SelectElite(0.5, 0.02)`, the README's Quick Usage: AGENTS.md calls `SelectTournament` the "default choice".

**Separate tests** (2026-09-25, 0.27.3, seeds 0 to 4, the scenario's budget, 60 s cap):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best (median, best, worst) | At the cap |
|---|---|---|---|---|---|---|
| onemax-100-matched | evolve | 5 | 5 | 10,042 | 100, 100, 100 | 0 |
| onemax-1000-matched | evolve | 5 | 5 | 114,293 | 1000, 1000, 1000 | 0 |
| onemax-100-idiomatic | evolve | 5 | 5 | 1,939 | 100, 100, 100 | 0 |

## Permutation: N-Queens 32 and 64

**Methods:** HillClimb, Stochastic ([#L379](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L379)):
- the README recommends HillClimb for "permutation problems (ordering, assignment)": "crossover is inefficient for permutations";
- AGENTS.md "Which HillClimb Variant?" prefers Stochastic for a large genome and for plateaus, and says "Use Stochastic with call_repeatedly for genomes >20 genes";
- the settings of [examples/hill_climb_nqueens.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/examples/hill_climb_nqueens.rs), the 64-queens board: `UniqueGenotype<u8>`, `HillClimbVariant::Stochastic`, `with_max_stale_generations(10000)` and `with_replace_on_equal_fitness(true)` ("crucial for this problem").

**Keeping going:** an attempt ends after 10,000 steps without improvement, and the method restarts, as `call_repeatedly` would (see above). In the tests below no run needed a restart.

**Left out:**
- Evolve: AGENTS.md says `CrossoverClone` with `UniqueGenotype` "is almost always less efficient than HillClimb + call_repeatedly(n). Only use Evolve + CrossoverClone for UniqueGenotype when you need Extensions or speciation", and [examples/evolve_nqueens.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/examples/evolve_nqueens.rs) prints "The Evolve strategy is very inefficient for this problem". It ran in v0.6.0 (the AGENTS.md unique preset); it's no longer run.
- SteepestAscent: AGENTS.md warns that it evaluates n(n-1)/2 neighbours per step for a `UniqueGenotype` (2,016 for 64) and says to use it for fewer than 20 genes.
- Permutate: exhaustive, for search spaces below a million (README); 32! isn't.

**Separate tests** (2026-09-25, 0.27.3, seeds 0 to 4):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best (median, best, worst) | At the cap |
|---|---|---|---|---|---|---|
| nqueens-32-idiomatic | hill_climb | 5 | 5 | 1,059 | 0, 0, 0 | 0 |
| nqueens-64-idiomatic | hill_climb | 5 | 5 | 5,731 | 0, 0, 0 | 0 |

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Methods:**
- **evolve** (all four, [#L446](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L446)): Evolve on a `RangeGenotype<f64>` with population 100, `SelectTournament(0.5, 0.02, 4)`, `CrossoverUniform(0.7, 0.8)`, `MutateMultiGene(2, 0.2)`, `MutationType::StepScaled` with the steps 0.1, 0.01, 0.001 and 0.0001 of the range (1.024, 0.1024, ... for Rastrigin), `with_max_stale_generations(100)` and precision 1e-5.
- **hill_climb** (Rosenbrock only, [#L517](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L517)): HillClimb, `SteepestAscent`, `StepScaled` with the steps 0.1 to 0.00001 of the range, `with_max_stale_generations(1)`. Each step evaluates the 2n neighbours, one step up and one down per gene.

**How the docs decided (rule 6.2: a stated preference, then the example for the problem type, then the default):**
- *Strategies.* The README's table recommends Evolve for "general optimization" and HillClimb for a "convex search space, few local optima". So Evolve for all four, and HillClimb for the unimodal Rosenbrock too. AGENTS.md "Which HillClimb Variant?" prefers SteepestAscent for a small genome.
- *The example.* The library's Evolve example for a real function is [examples/evolve_range_float.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/examples/evolve_range_float.rs): population 100, `SelectTournament(0.5, 0.02, 4)`, `CrossoverMultiPoint(0.7, 0.8, 9, false)`, `MutateMultiGene(2, 0.2)`, `StepScaled(vec![0.1, 0.01, 0.001, 0.0001])` on the range 0..=1, `with_max_stale_generations(100_000)`, precision 1e-5.
- *Stated preferences, which come before the example:*
  - the example's comments call its `StepScaled` the "best approach for this problem, converges fast, but needs low max_stale_generations to trigger next scale". The low value they give is `.with_max_stale_generations(100)`, commented out next to the 100,000 the example runs with. So `StepScaled` with 100;
  - AGENTS.md "Which Crossover?" recommends `CrossoverUniform` or `CrossoverSinglePoint` for a `RangeGenotype`. So `CrossoverUniform` instead of the example's `CrossoverMultiPoint`, with the rates 0.7 and 0.8 that the example and AGENTS.md's presets share.
- *From the example, where no stated preference covers a 10- or 30-gene genome:* `MutateMultiGene(2, 0.2)`, population 100, the precision. AGENTS.md's float advice (`MutateMultiGene(10, 0.8)` in its preset, "prefer RangeScaled or StepScaled") is stated "for float genomes >50 genes"; its `MutateSingleGene` is labelled the "default choice", which comes after the example.
- *HillClimb:* [examples/hill_climb_range.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/examples/hill_climb_range.rs), the HillClimb example for a real function, as it is: `SteepestAscent`, `StepScaled(vec![0.1, 0.01, 0.001, 0.0001, 0.00001])` on 0..=1, `with_max_stale_generations(1)`.
- *The range.* Both examples use the range 0..=1, so their steps are taken as shares of each problem's range. The docs call the steps "relative", meaning around the current value, so this is the adapter's reading.
- *A conflicting statement:* the README says of the mutation types "Random leads to the best results overall", and that the others "might converge faster, but are all more sensitive to local optima than Random". It covers every numeric genotype. The example's statement is specific to a real function, so it's the one followed. `MutationType::Random` draws a new value from the whole range and wasn't tested.

**Keeping going:** both end attempts by themselves. A scaled mutation moves to its next step after `max_stale_generations` without improvement, and the attempt ends when the last step goes stale (AGENTS.md "Scale advancement"). The adapter restarts them, as `call_repeatedly` would; AGENTS.md recommends `call_repeatedly` for local optima ("Typical n: 10") and "SteepestAscent + call_repeatedly(n)".

**Left out:**
- HillClimb on Rastrigin and Ackley: they have many local optima, and the README presents HillClimb for "few local optima".
- The extensions (`MassExtinction`, `MassGenesis`, `MassDegeneration`, `MassDeduplication`): AGENTS.md says they "should not be needed when hyperparameters are properly tuned" and are "a fallback when the population keeps collapsing to clones".
- `call_speciated`: AGENTS.md presents it for "complex combinatorial problems".
- The copy-paste template [AGENTS_TEMPLATES.md "Continuous Optimization"](https://docs.rs/crate/genetic_algorithm/0.27.3/source/AGENTS_TEMPLATES.md), a 100-gene genome with `MutateMultiGene(10, 1.0)`, `StepScaled([0.1, 0.01, 0.001])` and `with_max_stale_generations(1000)`, and AGENTS.md Troubleshooting's `RangeScaled`: other documented settings, below the example's stated preference.

**Separate tests** (2026-09-25, 0.27.3, seeds 0 to 4; `outside` 0 in every run):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best (median, best, worst) | At the cap |
|---|---|---|---|---|---|---|
| rastrigin-10-idiomatic | evolve | 5 | 5 | 55,274 | 0.00946, 0.00884, 0.00954 | 0 |
| rastrigin-30-idiomatic | evolve | 5 | 0 | | 1.99, 1.99, 3.98 | 0 |
| ackley-30-idiomatic | evolve | 5 | 5 | 134,377 | 0.00982, 0.00956, 0.00989 | 0 |
| rosenbrock-10-idiomatic | evolve | 5 | 0 | | 0.0761, 0.0634, 4.00 | 0 |
| rosenbrock-10-idiomatic | hill_climb | 5 | 0 | | 0.0204, 0.0184, 1.69 | 0 |

The runs that don't reach the target use their whole budget: Rastrigin 30 about 42 attempts of Evolve per run, Rosenbrock up to 3 attempts of HillClimb.

Other documented settings were run while auditing, before rule 6.2 said the docs decide (5 seeds each; reached out of 5, median best). They're shown for information only; they didn't pick anything:

| Mutation type (shares of the range), max_stale_generations, mutation, crossover | Rastrigin 10 | Rastrigin 30 | Ackley 30 | Rosenbrock 10 |
|---|---|---|---|---|
| **StepScaled [0.1, 0.01, 0.001, 0.0001], 100, MultiGene(2, 0.2), Uniform(0.7, 0.8): the docs' choice** | **5, 0.0095** | **0, 1.99** | **5, 0.0098** | **0, 0.076** |
| the same with the example's MultiPoint(0.7, 0.8, 9) (the example as it is, with 100) | 5, 0.0095 | 0, 1.99 | 5, 0.0098 | 1, 0.96 |
| the same, 100, MultiGene(n/10, 1.0), Uniform | 5, 0.0096 | 5, 0.0096 | 5, 0.0097 | 3, 0.0099 |
| the same, 1,000, MultiGene(n/10, 1.0), Uniform | 5, 0.0090 | 2, 0.996 | 5, 0.0097 | 1, 0.028 |
| the same, 100, MultiGene(10, 1.0), Uniform | 5, 0.0076 | 2, 0.997 | 0, 0.0137 | 0, 0.026 |
| StepScaled [0.1, 0.01, 0.001] (the template's), 1,000, MultiGene(n/10, 1.0), Uniform | 0, 0.012 | 0, 0.061 | 0, 0.091 | 0, 0.18 |
| the same, 100 | 2, 0.011 | 0, 0.042 | 0, 0.087 | 0, 0.19 |
| RangeScaled [1, 1, 0.5, 0.1, 0.01] (AGENTS.md Troubleshooting), 1,000, MultiGene(n/10, 1.0), Uniform | 3, 0.0095 | 0, 2.25 | 0, 0.26 | 0, 5.57 |
| the same, 100 | 5, 0.0070 | 0, 0.75 | 0, 0.36 | 0, 0.93 |
| RangeScaled [1, 1, 0.1, 0.01, 0.001] (the example's commented alternative), 1,000 | 3, 0.0093 | 0, 3.26 | 0, 0.019 | 0, 5.57 |
| the same, 100 | 5, 0.0076 | 0, 1.00 | 0, 0.024 | 0, 1.23 |
| RangeScaled [1, 1, 0.5, 0.1, 0.01], 1,000, MultiGene(10, 1.0), Uniform (the template with Troubleshooting's RangeScaled) | 0, 24.3 | 0, 12.7 | 0, 1.41 | 0, 6.61 |

The rows other than the first ran with the seed `seed * 1,000,000 + restart` for the restarts, before rule 2.2 set `seed * 1000 + restart`.

In v0.6.0 the adapter ran Evolve with `RangeScaled([range, range/2, 1, 0.1, 0.01, 0.001])`, `max_stale_generations(50)` and `MutateMultiGene(n/5, 0.8)`, and stopped when the last step went stale. Rastrigin 30 reached 0 of 10 after a median of 202,786 of its 2,000,000 evaluations, and Rosenbrock 1 of 10.

## Can't run

The multi-objective scenarios (ZDT1 to 3, DTLZ1 and 2): the library optimizes one `isize` fitness. AGENTS.md suggests a weighted sum for several objectives, which gives one point of the front, not a front.

## Bugs found

- **`call_repeatedly` with a seed repeats the same run.** Each repeat is built from a clone of the builder, whose random number generator is `SmallRng::seed_from_u64(seed)` ([strategy/evolve/builder.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/src/strategy/evolve/builder.rs), `call_repeatedly` and `rng`; the same in HillClimb's builder and `call_speciated`). With `with_rng_seed_from_u64(0)` and `call_repeatedly(4)` on OneMax, all four runs end at the same generation with the same genes. So seeded restarts can't use it; the adapter repeats the runs itself with a seed per run. Not reported upstream yet.
