# radiate (Rust, 1.3.2)

radiate is a Rust library for genetic algorithms, with Python bindings, genetic programming and neuroevolution. Its one search method is the `GeneticEngine`, a GA: each generation keeps `population_size × (1 − offspring_fraction)` survivors and breeds the rest. Its user guide is at [pkalivas.github.io/radiate](https://pkalivas.github.io/radiate/); the links below point to the guide's sources at the tag [v1.3.2](https://github.com/pkalivas/radiate/tree/v1.3.2).

In the matched suite ([rule 6](../rules.md#6-the-methods)), radiate runs OneMax 1000 with its `GeneticEngine`, set to the GA of [rule 6.2](../rules.md#6-the-methods). It has no differential evolution and no CMA-ES, so it doesn't run Rastrigin or Rosenbrock.

Adapter: [benchmarks/adapters/radiate/](../../../benchmarks/adapters/radiate/). The published results measure 1.3.2.
Found a setting that brings radiate closer to the definition, or a difference this page misses? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs radiate

- **Generation:** radiate evaluates only individuals whose genome changed ([engine/index.md, "Life of an epoch"](https://github.com/pkalivas/radiate/blob/v1.3.2/docs/source/engine/index.md#L42-L63)).
- **Evaluations:** the fitness wrapper counts every call and records the first hit ([`Budget::record`](../../../benchmarks/adapters/radiate/src/main.rs#L107-L121)).
- **Stop and stalls (rule 2.2):** radiate has no stop criterion of its own ("an engine with no limit attached runs forever in Rust", [engine/index.md](https://github.com/pkalivas/radiate/blob/v1.3.2/docs/source/engine/index.md#L88-L90)). The adapter's `until` ([`run_engine`](../../../benchmarks/adapters/radiate/src/main.rs#L195-L217)) ends the run after the generation that reaches the target, the budget or the time cap. After 10 generations in a row without an evaluation, it ends the attempt, and the engine starts again from a new random population with the next restart seed ([`run_single`](../../../benchmarks/adapters/radiate/src/main.rs#L256-L311)); the run prints `restarts`.
- **The run:** `engine.iter().until(closure).last()`, the way radiate's guide calls the cheap one ([engine/runtime.md](https://github.com/pkalivas/radiate/blob/v1.3.2/docs/source/engine/runtime.md), [engine/generations.md](https://github.com/pkalivas/radiate/blob/v1.3.2/docs/source/engine/generations.md)): the runtime's own `last()` is `run()`, and the closure gets a borrowed `GenerationView` of the live engine, so a `Generation`, a clone of the population and the metrics, is built once, at the end. Iterating, or the engine's `run(closure)`, would build one every generation ("Materializing a `Generation` you don't need", [engine/index.md](https://github.com/pkalivas/radiate/blob/v1.3.2/docs/source/engine/index.md#L88-L93)). Everything else that could cost is off by default and stays off: no diversity (no species step), no Pareto front for one objective, no event subscribers, the serial executor. The metrics step runs every generation and has no setting to turn it off.
- **Where the time goes** (Callgrind, seed 0): the fitness function 31% (a `BitChromosome` holds a `bool` per gene, so OneMax adds 1,000 bytes per evaluation; genoxide counts the ones of 16 words), evaluation and recombination (selection, crossover, mutation) most of the rest, the metrics step 21%. The adapter's `until` closure and its counter and first-hit check are under 0.5%.
- **Best solution:** `Generation::value`, radiate's best, recomputed after the clock. If it misses the target while an evaluated solution reached it, the run reports the first hit's solution.
- **Fitness:** `raw_fitness_fn`, the documented way to skip decoding ([fitness.md](https://github.com/pkalivas/radiate/blob/v1.3.2/docs/source/fitness.md#L81-L96)).
- **One thread:** no `rayon` feature, default `Executor::Serial`.
- **Seeds:** each run is inside `random_provider::scoped_seed(seed, ..)`, which reseeds radiate's thread-local generator ([random_provider.rs](https://github.com/pkalivas/radiate/blob/v1.3.2/crates/radiate-core/src/domain/random_provider.rs#L40-L64)). `random_provider::seed`, which the examples call, only seeds new threads.
- **Separate tests:** 2026-09-28, radiate 1.3.2, seeds 0 to 4, the scenario's budget and cap. 2026-09-29, after this audit: seeds 0 to 2 gave the same evaluations, best values and first hits as before, and `run.py check` passed.

## OneMax 1000: the GA

**Configuration** ([`run_onemax`, lines 319-373](../../../benchmarks/adapters/radiate/src/main.rs#L319-L373)), against [rule 6.2](../rules.md#6-the-methods):

| Definition | radiate |
|---|---|
| 300 individuals, uniform random bits | `.population_size(300)`, `BitCodec::vector(1000)` |
| tournaments of 3, with replacement | `.offspring_selector(TournamentSelector::new(3))` |
| two-point crossover, probability 0.5 | `MultiPointCrossover::new(0.5, 2)` |
| each child mutated with probability 0.2, each bit at 1/1000 | `BitFlipMutator::new(0.2 / 1000)` |
| generational, no elitism | `.offspring_fraction(1.0)`: no survivors; `.max_age(usize::MAX)`: no replacement of old individuals by random ones (radiate's default replaces those older than 20 generations) |

**Differences** (rule 6.1):
- No per-child mutation probability: each bit flips with probability 0.2 / n, the same mean of 0.2 flips per child, over more children (18% mutated instead of 12.6%).
- Each child is crossed with probability 0.5 with a random other child, so a child can be crossed more than once; DEAP crosses disjoint pairs. The expected number of crossovers per generation is the same, 150.
- Cut points are drawn from 0..n (a cut at 0 changes nothing), DEAP's from 1..n.
- Only changed children are evaluated; DEAP also evaluates a child it chose to mutate when no bit flipped.

**Separate tests:** 5 of 5 reached the target, first hits at 73,745 to 82,840 evaluations (median 79,149), in about 36 ms each.

## Can't run

- Rastrigin 30 (DE/rand/1/bin) and Rosenbrock 10 (CMA-ES): radiate has only a GA.

## Bugs found

None in the GA run here. Found in methods no longer in the suite, reported in [pkalivas/radiate#29](https://github.com/pkalivas/radiate/issues/29), fix proposed in [pkalivas/radiate#30](https://github.com/pkalivas/radiate/pull/30):
- The guide recommends `ShuffleCrossover` for permutations ([alters/index.md, line 37](https://github.com/pkalivas/radiate/blob/v1.3.2/docs/source/alters/index.md#L37)), but its children aren't permutations, and the engine replaces them with random individuals without a warning.
- The guide describes `GaussianMutator` as producing "small, incremental changes" ([mutators.md](https://github.com/pkalivas/radiate/blob/v1.3.2/docs/source/alters/mutators.md#L45-L56)); its standard deviation is a quarter of the gene's initial range.
