# genetic_algorithm (Rust, 0.27.3)

A Rust genetic algorithm library with three strategies: Evolve (a GA), HillClimb (local search, Stochastic or SteepestAscent) and Permutate (exhaustive search). The fitness is an `isize`, so real values are scaled by a precision. Its docs are the [README](https://docs.rs/crate/genetic_algorithm/0.27.3/source/README.md), [AGENTS.md](https://docs.rs/crate/genetic_algorithm/0.27.3/source/AGENTS.md) (decision matrices and recommended settings), [AGENTS_TEMPLATES.md](https://docs.rs/crate/genetic_algorithm/0.27.3/source/AGENTS_TEMPLATES.md), the [examples](https://docs.rs/crate/genetic_algorithm/0.27.3/source/examples/) and [docs.rs](https://docs.rs/genetic_algorithm/0.27.3/genetic_algorithm/).

Adapter: [benchmarks/adapters/genetic_algorithm/](../../../benchmarks/adapters/genetic_algorithm/).
Know a better way to solve one of these problems with genetic_algorithm? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs genetic_algorithm

- **Evaluations:** counted in the fitness functions ([main.rs#L171](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L171)), with the first hit ([#L71](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L71)).
- **Stop:** `with_target_fitness_score`, or `with_abort_flag` (checked once per generation), set when the budget or the time is used up ([#L48](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L48)). No run uses `with_max_generations`.
- **Real values:** a value is divided by the precision 1e-5 of AGENTS.md and the examples, rounded up, so the target of 1000 units means at most 0.01 ([#L208](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L208)). The reported `best` is the f64 value.
- **Keeping going (rule 2.2):** `with_max_stale_generations` ends an attempt. The library's restart mechanism, `call_repeatedly(n)` (AGENTS.md, "Choosing a call variant"), repeats the same run when seeded (see [Bugs found](#bugs-found)), so the adapter's `restarts` ([#L328](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L328)) does the same with the seeds of rule 2.2 ([#L316](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L316)): attempts until the target or the abort flag, keeping the best genes by the library's score.
- **Bounds (rule 2.4):** a `RangeGenotype` draws its initial genes in the allele range and clamps every mutation to it ([genotype/range.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/src/genotype/range.rs), `clamped_add`, `clamped_sub`); uniform crossover only exchanges genes. Counted as `outside` ([#L221](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L221)).
- **Shift** (rule 1.4): computed once ([#L134](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L134)).
- **Time:** from before the strategy is built to its return ([run](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L297)).
- **One thread:** no `with_par_fitness`, no `call_par_*`; the only other thread is the sleeping timer.
- **Seeds:** `with_rng_seed_from_u64`.
- **Separate tests:** 2026-09-25, 0.27.3, seeds 0 to 4, the scenario's budget, 60 s cap. `outside` was 0 in every run.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched: not run** ([#L360](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L360)). Evolve has no generational replacement without elitism: its selection keeps the survivors from parents and offspring together ([select/tournament.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/src/select/tournament.rs)), and the crossover breeds from them. With a replacement rate of 1.0 all offspring survive and nothing is selected; below it, parents compete with offspring.
- **Idiomatic** ([#L355](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L355)): Evolve with AGENTS.md's binary preset ("If unsure, start here"): `SelectTournament(0.5, 0.02, 4)`, `CrossoverUniform(0.7, 0.8)`, `MutateSingleGene(0.2)`, and population 100, the default and the README's "Quick Usage" example, which is this problem.

**Keeping going:** runs to the target or the budget.

**Left out:**
- HillClimb: the README recommends it for a "convex search space, few local optima", which a binary problem in general isn't.
- `SelectElite(0.5, 0.02)`, the Quick Usage's: AGENTS.md calls `SelectTournament` the "default choice".

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best (median, best, worst) | Capped |
|---|---|---|---|---|---|---|
| onemax-100-idiomatic | evolve | 5 | 5 | 1,893 | 100, 100, 100 | 0 |

## Permutation: N-Queens 32 and 64

**Methods:** HillClimb, Stochastic ([#L416](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L416)): the README recommends HillClimb for "permutation problems (ordering, assignment)"; AGENTS.md's "Which HillClimb Variant?" says "Use Stochastic with call_repeatedly for genomes >20 genes". The settings are [examples/hill_climb_nqueens.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/examples/hill_climb_nqueens.rs)'s: `UniqueGenotype<u8>`, `with_max_stale_generations(10000)`, `with_replace_on_equal_fitness(true)` ("crucial for this problem").

**Keeping going:** an attempt ends after 10,000 steps without improvement and restarts (above). No test run restarted.

**Left out:**
- Evolve: AGENTS.md says `CrossoverClone` with `UniqueGenotype` "is almost always less efficient than HillClimb + call_repeatedly(n)", and [examples/evolve_nqueens.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/examples/evolve_nqueens.rs) prints "The Evolve strategy is very inefficient for this problem".
- SteepestAscent: n(n-1)/2 neighbours per step (2,016 for 64); AGENTS.md says to use it for fewer than 20 genes.
- Permutate: for search spaces below a million (README).

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best (median, best, worst) | Capped |
|---|---|---|---|---|---|---|
| nqueens-32-idiomatic | hill_climb | 5 | 5 | 1,720 | 0, 0, 0 | 0 |
| nqueens-64-idiomatic | hill_climb | 5 | 5 | 4,010 | 0, 0, 0 | 0 |

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Methods:**
- **`evolve`** (all four, [#L495](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L495)): a `RangeGenotype<f64>`, population 100, `SelectTournament(0.5, 0.02, 4)`, `CrossoverUniform(0.7, 0.8)`, `MutateMultiGene(2, 0.2)`, `MutationType::StepScaled` with steps of 0.1, 0.01, 0.001 and 0.0001 of the range, `with_max_stale_generations(100)`, precision 1e-5.
- **`hill_climb`** (Rosenbrock, [#L562](../../../benchmarks/adapters/genetic_algorithm/src/main.rs#L562)): `SteepestAscent`, `StepScaled` with steps of 0.1 to 0.00001 of the range, `with_max_stale_generations(1)`; each step evaluates the 2n neighbours.

**Where the settings come from** (rule 6.2):
- *Strategies:* the README's table recommends Evolve for "general optimization" and HillClimb for a "convex search space, few local optima", so HillClimb for Rosenbrock too. AGENTS.md prefers SteepestAscent for a small genome.
- *The example:* [evolve_range_float.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/examples/evolve_range_float.rs): population 100, `SelectTournament(0.5, 0.02, 4)`, `CrossoverMultiPoint(0.7, 0.8, 9, false)`, `MutateMultiGene(2, 0.2)`, `StepScaled(vec![0.1, 0.01, 0.001, 0.0001])` on 0..=1, `with_max_stale_generations(100_000)`, precision 1e-5.
- *Stated preferences, before the example:* the example's comments call its `StepScaled` the "best approach for this problem ... but needs low max_stale_generations to trigger next scale", with `.with_max_stale_generations(100)` commented out next to it: so 100. AGENTS.md's "Which Crossover?" recommends `CrossoverUniform` or `CrossoverSinglePoint` for a `RangeGenotype`: so `CrossoverUniform`, at the rates the example and the presets share.
- *From the example:* `MutateMultiGene(2, 0.2)`, population 100, the precision. AGENTS.md's float advice (`MutateMultiGene(10, 0.8)`, "prefer RangeScaled or StepScaled") is "for float genomes >50 genes"; its "default choice" `MutateSingleGene` comes after the example.
- *HillClimb:* [hill_climb_range.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/examples/hill_climb_range.rs) as it is: `SteepestAscent`, `StepScaled(vec![0.1, 0.01, 0.001, 0.0001, 0.00001])` on 0..=1, `with_max_stale_generations(1)`.
- *The range:* both examples use 0..=1, so their steps are taken as shares of each problem's range; the docs call the steps "relative", so this is the adapter's reading.

**Keeping going:** a scaled mutation moves to its next step after `max_stale_generations` without improvement, and the attempt ends when the last step goes stale (AGENTS.md, "Scale advancement"); it then restarts, as AGENTS.md recommends for local optima (`call_repeatedly`, "Typical n: 10").

**Left out:**
- HillClimb on Rastrigin and Ackley: many local optima.
- The extensions (`MassExtinction`, `MassGenesis`, `MassDegeneration`, `MassDeduplication`): AGENTS.md calls them "a fallback when the population keeps collapsing to clones".
- `call_speciated`: for "complex combinatorial problems".
- The template [AGENTS_TEMPLATES.md "Continuous Optimization"](https://docs.rs/crate/genetic_algorithm/0.27.3/source/AGENTS_TEMPLATES.md) (100 genes, `MutateMultiGene(10, 1.0)`, `StepScaled([0.1, 0.01, 0.001])`, `with_max_stale_generations(1000)`) and Troubleshooting's `RangeScaled`: below the example's stated preference.
- `MutationType::Random`: the README says it "leads to the best results overall" for every numeric genotype; the example's statement, specific to real functions, comes first.

**Separate tests** (with the shift of rule 1.4):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best (median, best, worst) | Capped |
|---|---|---|---|---|---|---|
| rastrigin-10-idiomatic | evolve | 5 | 5 | 54,291 | 0.00959, 0.00860, 0.00989 | 0 |
| rastrigin-30-idiomatic | evolve | 5 | 0 | | 2.98, 1.99, 3.98 | 0 |
| ackley-30-idiomatic | evolve | 5 | 5 | 83,675 | 0.00992, 0.00967, 0.00993 | 0 |
| rosenbrock-10-idiomatic | evolve | 5 | 0 | | 0.0742, 0.0562, 0.267 | 0 |
| rosenbrock-10-idiomatic | hill_climb | 5 | 0 | | 0.0201, 0.0186, 0.0227 | 0 |

Other documented settings (5 seeds each, the earlier shift in [−1, 1], restart seed `seed * 1,000,000 + restart` except in the first row; reached out of 5, median best). They didn't pick anything:

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

## Can't run

- Matched OneMax 100 and 1000: no generational replacement without elitism.
- The multi-objective scenarios: the library optimizes one `isize` fitness. AGENTS.md suggests a weighted sum, which gives one point of the front.

## Bugs found

- **`call_repeatedly` with a seed repeats the same run.** Each repeat is built from a clone of the builder, whose generator is `SmallRng::seed_from_u64(seed)` ([strategy/evolve/builder.rs](https://docs.rs/crate/genetic_algorithm/0.27.3/source/src/strategy/evolve/builder.rs), `call_repeatedly` and `rng`; the same in HillClimb's builder and `call_speciated`). With `with_rng_seed_from_u64(0)` and `call_repeatedly(4)` on OneMax, all four runs end at the same generation with the same genes. Effect: none here; the adapter repeats the runs itself with a seed per run. Not reported upstream yet.
