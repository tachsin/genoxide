# radiate (Rust, 1.3.1)

radiate is a Rust library for genetic algorithms, with Python bindings, genetic programming and neuroevolution. Its one search method is the `GeneticEngine`: a GA whose generation keeps `population_size × (1 − offspring_fraction)` survivors (a survivor selector) and breeds the rest from an offspring selector, crossovers and mutators. Multi-objective problems use the same engine with the NSGA-II or NSGA-III selectors. Its user guide is at [pkalivas.github.io/radiate](https://pkalivas.github.io/radiate/); the links below point to the guide's sources and the examples in the repository at the tag [v1.3.1](https://github.com/pkalivas/radiate/tree/v1.3.1) (579a943), the version benchmarked.

Adapter: [benchmarks/adapters/radiate/](../../../benchmarks/adapters/radiate/).
Know a better way to solve one of these problems with radiate? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How radiate runs

- **The generation.** Each generation evaluates the unscored individuals, selects survivors and offspring from the same population, alters the offspring, replaces invalid individuals and those older than `max_age`, and evaluates the ones whose genome changed ([engine/index.md, "Life of an epoch"](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/engine/index.md#L42-L63)). A crossover marks both parents as changed when it changed either, so a crossover that writes one parent (`MeanCrossover`, `SimulatedBinaryCrossover`) costs an evaluation of an unchanged individual ([alter.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-core/src/alter.rs#L263-L266)). The adapter counts every call of its fitness function, so these evaluations are counted.
- **Defaults.** Population 100, roulette offspring selection, a tournament of 3 for the survivors, offspring fraction 0.8, `UniformCrossover(0.5)` and `UniformMutator(0.1)` ([engine/index.md, "Engine Defaults"](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/engine/index.md#L7-L26)). Not in that table: every individual older than `max_age` = 20 generations is replaced by a random one ([builder/mod.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-engines/src/builder/mod.rs#L552-L556), [steps/filter.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-engines/src/steps/filter.rs#L27-L52)). The idiomatic runs keep it, as radiate's users get it. It isn't a stop or a restart: it replaces individuals within the run.
- **Stopping and restarts (rule 2.2).** radiate has no stop criterion of its own, neither a budget nor a convergence test: "an engine with no limit attached runs forever in Rust" ([engine/index.md, "Common Pitfalls"](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/engine/index.md#L88-L90)). The adapter's one limit, `until` ([run_engine](../../../benchmarks/adapters/radiate/src/main.rs#L304-L316)), ends the run after the generation in which an evaluation reaches the target, the budget or 60 seconds ([Budget::done](../../../benchmarks/adapters/radiate/src/main.rs#L281-L287)). So no run needs a restart; every run goes on to the target or the budget, and can go past the budget by less than one generation.
- **Bounds (rule 2.4).** `FloatCodec::vector` draws the genes in the problem's range and sets that range as their bounds; radiate's float alterers write through `FloatGene::set_allele` or `safe_clamp`, which clip to the bounds ([float.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-core/src/genome/chromosomes/float.rs#L82-L85)). So radiate's own bound handling is clipping, inside its operators. The adapter's fitness wrapper counts every evaluated solution outside the bounds, as radiate proposed it, without clipping ([Budget::check_bounds](../../../benchmarks/adapters/radiate/src/main.rs#L262-L267)); continuous and multi-objective runs print it as `outside`, and it was 0 in every run.
- **Evaluations and the first hit.** The fitness wrapper counts every call, and records the first evaluation whose `f64` value reaches the target, with the clock at that moment (`first_hit`, [Budget::record](../../../benchmarks/adapters/radiate/src/main.rs#L236-L251)).
- **The best solution** is radiate's own: `Generation::value`, the best individual of the run by radiate's `f32` scores. Its value is recomputed in `f64` after the clock ([run_single](../../../benchmarks/adapters/radiate/src/main.rs#L353-L391)). If it doesn't reach the target in `f64` while an evaluated solution did (two scores equal in `f32`), the run reports the solution of the first hit, which the fitness wrapper copies once per run. This didn't happen in any test run.
- **Fitness.** Through `raw_fitness_fn`, radiate's documented way to evaluate the genotype without decoding it ([fitness.md, "Raw Fitness"](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/fitness.md#L81-L96)). The fitness functions compute in `f64` ([fitness functions](../../../benchmarks/adapters/radiate/src/main.rs#L50-L165)); the shift of Rastrigin and Ackley (rule 1.4) is computed once, before any run.
- **The clock** starts before the engine creates its initial population and stops when the engine's last generation ends, before the output ([run_single](../../../benchmarks/adapters/radiate/src/main.rs#L353-L391)).
- **One thread.** radiate is built without its `rayon` feature and no executor is set, so the fitness, the species and the events run on the default `Executor::Serial` ([executors.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/executors.md)). `run.py check` measures a CPU time equal to the wall time.
- **Seeded.** Every run is inside `random_provider::scoped_seed(seed, ..)`, which reseeds the thread-local generator radiate draws all its random numbers from ([random_provider.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-core/src/domain/random_provider.rs#L40-L58)). `random_provider::seed`, which radiate's examples call, only reseeds the global generator new threads start from, so it can't reseed a second run in the same thread. The same seed gives the same evaluations and results (rule 5.2, checked).

**The idiomatic methods** ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L398-L421)). Each problem type runs up to 3 solvers:
- `ga`: radiate's own example for the problem type, as it is.
- the recipe of radiate's guide for its genome type, one solver per crossover the recipe names. The recipes are in the alterers' "Best Practices" ([alters/index.md, lines 30-48](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/alters/index.md#L30-L48)):

  > 1. **Rate Selection**: Start with conservative rates (0.01 for mutation, 0.5-0.8 for crossover) [...]
  > 2. **Choosing the Right Alterer**:
  >    - For continuous problems: Use Gaussian or Arithmetic mutators with Blend/Intermediate crossover
  >    - For permutation problems: Use Swap/Scramble mutators with PMX or Shuffle crossover
  >    - For binary problems: Use Uniform mutator with Multi-point or Uniform crossover

**How the docs decide the recipes' settings** (rule 6.2: a preference the docs state, else their example for the problem type, else the default). A recipe names only alterers, so a recipe solver is the problem type's example (its population and selectors) with the recipe's alterers:
- **Mutation rate 0.01:** the stated starting rate.
- **Crossover rate:** the stated range is 0.5 to 0.8. Within it, the example's own crossover rate: 0.5 in the knapsack example (the engine's default `UniformCrossover(0.5)`) and the Rastrigin example, 0.75 in the Rosenbrock example. The TSP example's 0.4 is outside the range, so the permutation recipe takes the rate of the engine's default crossover, 0.5 (`UniformCrossover(0.5)`, "Engine Defaults").
- **The mutator, where the recipe names two:** the example's. The TSP example uses Swap; the Rastrigin and Rosenbrock examples use Arithmetic.
- **The crossover, where the recipe names two:** no preference is stated and the examples use neither, so both run as separate solvers. Shuffle crossover is left out for another reason (see Permutation).
- **Operator parameters:** `MultiPointCrossover` with 2 points, as in every radiate example that uses it; `BlendCrossover` and `IntermediateCrossover` with α 0.5, the guide's snippets ([src/rust/alters/crossovers.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/src/rust/alters/crossovers.rs#L4-L12)).

The separate tests below are shown, but they didn't pick any of these.

The guide's alterers example ([src/rust/alters/example.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/src/rust/alters/example.rs#L30-L50): `BoltzmannSelector(4)`, `BlendCrossover(0.8, 0.5)` with `GaussianMutator(0.1)`, or `MultiPointCrossover(0.8, 2)`) shows how to wire alterers into an engine on a 2-variable float problem; it isn't an example for any of these problem types, so it decides nothing here.

Diversity (speciation) is left out everywhere: the guide says to "reach for it when a problem is converging too early or not reaching its global optimum, not by default" ([diversity/index.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/diversity/index.md#L32-L34)), and its threshold has to be scaled to the distance measure by trial ([species.md, "Species threshold"](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/diversity/species.md)), which would be tuning to the benchmark. Rate schedules (`Expr` rates) are left out for the same reason: the guide presents them as "examples of common shapes that _might_ be useful" ([alters/rate.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/alters/rate.md)), not as settings for a problem type.

## Binary: OneMax

**Methods:**
- **Matched** ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L437-L476)): DEAP's eaSimple, the matched setting, with radiate's own selector and alterers only (rule 6.1): population 300, offspring fraction 1.0 (every generation is 300 selected and altered copies, no survivors, so generational and without elitism), `TournamentSelector(3)` (with replacement), `MultiPointCrossover(0.5, 2)`, `BitFlipMutator(0.2 / size)`, max age off. radiate evaluates only the children whose genome changed. The differences radiate forces:
  - radiate has no per-individual mutation probability, so `BitFlipMutator` flips each bit with probability 0.2 / size: the same expected 0.2 flipped bits per child, spread over more children (18% instead of 12.6% mutated).
  - radiate visits every child and with probability 0.5 crosses it with a random other child (both change); DEAP crosses the disjoint pairs (0, 1), (2, 3), ... with probability 0.5. The same expected 150 crossovers per generation, but a child can be crossed more than once. radiate draws the cut points from 0..size (a cut at 0 changes nothing), DEAP from 1..size.
  - max age is turned off: radiate would otherwise replace individuals older than 20 generations with random ones, which eaSimple doesn't do.
- **Idiomatic**, on radiate's binary example, the knapsack ([examples/rust/knapsack](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/knapsack/src/main.rs#L11-L18)): a `SubSetCodec`, whose genome is a `BitChromosome` with one bit per item (here the items are the gene indices, and the fitness counts the ones), `max_age(50)`, and the other engine defaults (population 100, roulette offspring selection, a tournament of 3 for the survivors, offspring fraction 0.8) ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L477-L515)):
  - `ga` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L489-L492)): the example as it is, with the default alterers `UniformCrossover(0.5)` and `UniformMutator(0.1)`. `UniformMutator` draws the gene again, so it flips a bit with probability 0.05: 5 bits of every child of OneMax 100.
  - `ga_uniform` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L493-L496)): the binary recipe with Uniform crossover: `UniformCrossover(0.5)` and `UniformMutator(0.01)`.
  - `ga_multipoint` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L497-L500)): the binary recipe with Multi-point crossover: `MultiPointCrossover(0.5, 2)` and `UniformMutator(0.01)`.

**Keeping going:** runs to the target or the budget by itself.

**Left out:**
- The README's "Hello, Radiate!" example ([examples/rust/hello-world](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/hello-world/src/main.rs#L8-L19)): it evolves characters towards a string, not bits; the knapsack is radiate's example with a binary genome.
- `BitFlipMutator` in the idiomatic runs: the guide's mutator table doesn't list it, and the recipe for binary problems names the Uniform mutator.
- Why `ga` doesn't reach the target: its mutation flips about 5 bits of every child, so a child of a near-optimal parent almost never keeps all its ones (0.95^99 ≈ 0.6% for 99 ones). Its best stayed at 81 or 82 ones.

**Separate tests** (2026-09-25, radiate 1.3.1, seeds 0 to 4):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| onemax-100-matched | ga | 5 | 5 | 5,484 | 100 | 100 | 100 | 0 |
| onemax-1000-matched | ga | 5 | 5 | 77,342 | 1000 | 1000 | 1000 | 0 |
| onemax-100-idiomatic | ga | 5 | 0 | | 82 | 82 | 81 | 0 |
| onemax-100-idiomatic | ga_uniform | 5 | 5 | 9,681 | 100 | 100 | 100 | 0 |
| onemax-100-idiomatic | ga_multipoint | 5 | 5 | 12,156 | 100 | 100 | 100 | 0 |

## Permutation: N-Queens 32 and 64

**Methods**, on radiate's permutation example, the TSP ([examples/rust/TSP](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/TSP/src/main.rs#L13-L19)): `PermutationCodec`, population 250, minimizing, the other engine defaults (roulette offspring selection, a tournament of 3 for the survivors) ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L526-L540)):
- `ga` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L542-L545)): the example as it is, `PMXCrossover(0.4)` and `SwapMutator(0.05)`.
- `ga_pmx` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L546-L552)): the permutation recipe, `PMXCrossover(0.5)` and `SwapMutator(0.01)`.

**Keeping going:** runs to the target or the budget by itself.

**Left out:**
- radiate's own N-Queens example ([examples/rust/nqueens](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/nqueens/src/main.rs), also on the guide's [examples page](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/examples.md#L31-L53)) evolves integers and counts row conflicts in its fitness; it's not a permutation, which this scenario's genome is.
- `ShuffleCrossover`, which the recipe also names: it swaps genes between the parents position by position ([shuffle.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-alters/src/crossovers/shuffle.rs#L35-L47)), so its children are no longer permutations, and the engine replaces every invalid child with a random one. The guide's own pitfall applies: "Some alterers may produce invalid solutions" ([alters/index.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/alters/index.md#L61-L63)).
- `ScrambleMutator`: the recipe names Swap or Scramble, and the TSP example uses Swap.

Why neither reaches the target reliably: the TSP example's roulette selection. radiate's `RouletteSelector` weighs a minimized score by reversing the sorted scores ([roulette.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-selectors/src/roulette.rs#L20-L35)); with conflict counts close to each other, the pressure is weak. It's not the max age (with it off, `ga` ended at the same median of 5 on N-Queens 32). For comparison only, not used to choose: in the separate tests the TSP example with `BoltzmannSelector(4)` instead of roulette reached the target of N-Queens 32 in 5 of 5 runs (median 21,809 evaluations).

**Separate tests** (2026-09-25, radiate 1.3.1, seeds 0 to 4):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| nqueens-32-idiomatic | ga | 5 | 0 | | 5 | 3 | 5 | 0 |
| nqueens-32-idiomatic | ga_pmx | 5 | 2 | 312,840 | 4 | 0 | 4 | 0 |
| nqueens-64-idiomatic | ga | 5 | 0 | | 16 | 15 | 16 | 0 |
| nqueens-64-idiomatic | ga_pmx | 5 | 0 | | 15 | 14 | 16 | 0 |

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods**, on radiate's Rastrigin example ([examples/rust/rastrigin](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/rastrigin/src/main.rs#L10-L17), also on the guide's [examples page](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/examples.md#L57-L85)): population 500, the other engine defaults (roulette offspring selection) ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L585-L597)). radiate has no Ackley example; the Rastrigin example is its example of a multimodal real function, so Ackley uses it too.
- `ga` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L607-L609)): the example as it is, `UniformCrossover(0.5)` and `ArithmeticMutator(0.01)`.
- `ga_blend` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L610-L614)): the continuous recipe with Blend crossover, `BlendCrossover(0.5, 0.5)` and `ArithmeticMutator(0.01)`.
- `ga_intermediate` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L615-L622)): the continuous recipe with Intermediate crossover, `IntermediateCrossover(0.5, 0.5)` and `ArithmeticMutator(0.01)`.

The genes are `f64` in the problem's bounds; see "Bounds" above ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L573-L584)).

**Keeping going:** runs to the target or the budget by itself.

**Left out:**
- `GaussianMutator`: the recipe names Gaussian or Arithmetic mutators, and radiate's examples for continuous problems (Rastrigin, Rosenbrock) use Arithmetic.
- `JitterMutator` (uniform noise of a fixed magnitude): the recipe doesn't name it.
- `MeanCrossover` (the Rosenbrock example's) and `SimulatedBinaryCrossover`: not in the continuous recipe; SBX has the bug below.

**Separate tests** (2026-09-25, radiate 1.3.1, seeds 0 to 4, with the shift of rule 1.4):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| rastrigin-10-idiomatic | ga | 5 | 5 | 294,787 | 0.00940 | 0.00700 | 0.00961 | 0 |
| rastrigin-10-idiomatic | ga_blend | 5 | 0 | | 39.11 | 1.117 | 39.65 | 0 |
| rastrigin-10-idiomatic | ga_intermediate | 5 | 5 | 57,144 | 0.00667 | 0.00161 | 0.00920 | 0 |
| rastrigin-30-idiomatic | ga | 5 | 5 | 698,687 | 0.00944 | 0.00891 | 0.01000 | 0 |
| rastrigin-30-idiomatic | ga_blend | 5 | 0 | | 421.5 | 385.2 | 425.2 | 0 |
| rastrigin-30-idiomatic | ga_intermediate | 5 | 5 | 494,697 | 0.00873 | 0.00598 | 0.00967 | 0 |
| ackley-30-idiomatic | ga | 5 | 0 | | 0.0432 | 0.0350 | 0.0464 | 0 |
| ackley-30-idiomatic | ga_blend | 5 | 0 | | 20.64 | 20.62 | 20.64 | 0 |
| ackley-30-idiomatic | ga_intermediate | 5 | 2 | 808,175 | 0.0132 | 0.00758 | 0.0227 | 0 |

`ga_blend`: radiate's Blend crossover moves each child away from the other parent (child = a − α (b − a), [blend.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-alters/src/crossovers/blend.rs#L57-L64), as its docs define it), outside the segment between the parents. It spreads the population instead of combining it; in these runs the best solutions stayed far from the optimum (Ackley's outer plateau, at about 20).

## Continuous, unimodal: Rosenbrock 10

**Methods**, on radiate's Rosenbrock example ([examples/rust/rosenbrock](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/rosenbrock/src/main.rs#L13-L20)): `BoltzmannSelector(4)` for the offspring, the other engine defaults (population 100) ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L585-L597)):
- `ga` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L604-L606)): the example as it is, `MeanCrossover(0.75)` and `ArithmeticMutator(0.1)`.
- `ga_blend` and `ga_intermediate`: the continuous recipe, as for the multimodal problems, at the example's crossover rate: `BlendCrossover(0.75, 0.5)` or `IntermediateCrossover(0.75, 0.5)`, with `ArithmeticMutator(0.01)`.

**Keeping going:** runs to the target or the budget by itself.

**Left out:** as for the multimodal problems.

Why `ga` stops at about 6.9: in every run its best solution ends with its last genes at about 0.01 and the first ones between 0.7 and 0.05 (e.g. 0.67, 0.45, 0.21, 0.05, 0.013, 0.010, ...), a point on Rosenbrock's curved valley. `MeanCrossover` pulls the population to its mean and `ArithmeticMutator` adds, subtracts, multiplies or divides by a random value of the whole range ([arithmetic.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-alters/src/mutators/arithmetic.rs#L38-L58)), so it can't take the small steps the valley needs. It's not the scores' `f32` precision (the values are near 7) nor the max age: with max age off it ended at the same median (6.91). The example itself is for 2 variables in [−2, 2].

**Separate tests** (2026-09-25, radiate 1.3.1, seeds 0 to 4):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| rosenbrock-10-idiomatic | ga | 5 | 0 | | 6.885 | 6.858 | 7.213 | 0 |
| rosenbrock-10-idiomatic | ga_blend | 5 | 0 | | 0.240 | 0.0349 | 5.421 | 0 |
| rosenbrock-10-idiomatic | ga_intermediate | 5 | 1 | 16,326 | 0.0284 | 0.00972 | 2.089 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Methods** ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L661-L788)), the matched settings (rule 6.1), with radiate's own selectors, SBX and polynomial mutation, bugs included (rule 8.4). radiate has no other multi-objective algorithm (no SPEA2, MOEA/D or SMS-EMOA); its multi-objective support is these two survivor selectors in the same engine ([selectors/index.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/selectors/index.md#L20-L34)), so nothing is left out.
- `nsga2`: `TournamentNSGA2Selector` (binary tournament on rank and crowding distance) for the mating pool, `NSGA2Selector` for the survivors, `SimulatedBinaryCrossover(0.9, 15)`, `PolynomialMutator(1 / n, 20)`; population 100, 92 with 3 objectives.
- `nsga3`: `RandomSelector` for the mating pool (NSGA-III's random mating), `NSGA3Selector` with Das-Dennis reference directions, 99 divisions with 2 objectives (100 directions, population 100) and 12 with 3 (91 directions, population 92), `SimulatedBinaryCrossover(1.0, 30)`, `PolynomialMutator(1 / n, 20)`.
  - It runs on the ZDT problems too, which the earlier adapter left out: radiate's selector works for any number of objectives, and radiate's own ZDT example uses `nsga3(12)` with 2 objectives ([examples/python/zdt.py](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/python/zdt.py#L30-L36)). The guide lists NSGA-III for "Multi-objective problems with many objectives" ([selectors/index.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/selectors/index.md#L29)).

NSGA-II and NSGA-III select the next population from the parents and their offspring. In radiate's engine that's `population_size(2 mu)` with `offspring_fraction(0.5)`: each generation keeps mu survivors of the 2 mu individuals by the NSGA-II or NSGA-III selector and breeds mu offspring. The front reported is the non-dominated part of the mu individuals the survivor selector keeps of the last 2 mu, i.e. NSGA-II's (NSGA-III's) population after the last generation (rule 7.2), with the objectives recomputed in `f64` from the solutions. radiate's own Pareto archive (`front_size`, by default 800 to 900 of the non-dominated individuals it has seen, [builder/mod.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-engines/src/builder/mod.rs#L575-L579)) is not reported.

The differences radiate forces:
- The initial population has 2 mu random individuals (200, or 184 with 3 objectives), not mu. radiate keeps the population at the size of its initial one: the survivor and offspring counts are fractions of the initial population's length ([config.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-engines/src/builder/config.rs#L81-L87)). A first population of mu given with `population(..)` would keep mu / 2 survivors and breed mu / 2 offspring every generation. So the first generation costs mu more evaluations than in the other libraries, out of the same budget.
- The mating pool is drawn from all 2 mu individuals, not from the mu survivors.
- NSGA-II's crowding distance is computed over the whole population, not per front ([pareto.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-core/src/objectives/pareto.rs#L30-L85)).
- NSGA-III normalizes the objectives by the population's ideal and nadir points, without the hyperplane through the extreme points ([nsga3.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-selectors/src/nsga3.rs#L144-L188)).
- The crossover rate is per individual: radiate visits every offspring and crosses it with a random other one with the given probability ([alter.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-core/src/alter.rs#L225-L243)), so an individual can be crossed more than once.
- Scores are `f32`, so the selections compare the objectives rounded to `f32`.
- Max age is off (radiate would replace individuals older than 20 generations with random ones).
- radiate's own multi-objective examples use `TournamentSelector(5)` for the offspring and `UniformMutator` ([examples/rust/DTLZ](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/DTLZ/src/main.rs#L13-L24)); the matched settings don't.

**Keeping going:** multi-objective runs use their whole budget; radiate has no convergence criterion to restart from.

**Bounds:** the variables are `f64` in [0, 1], clipped by radiate's operators (see "Bounds" above). `outside` was 0 in every run, also with the SBX bug below, whose children below 0 are clipped to 0 before they're evaluated.

**Separate tests** (2026-09-25, radiate 1.3.1, seeds 0 to 4; hypervolume with `run.py`'s code and reference points):

| Scenario | Solver | Runs | Hypervolume: median | best | worst | Median evaluations | Capped |
|---|---|---|---|---|---|---|---|
| zdt1-30-matched | nsga2 | 5 | 0.8047 | 0.8382 | 0.7606 | 25,024 | 0 |
| zdt1-30-matched | nsga3 | 5 | 0.7837 | 0.8348 | 0.7548 | 25,000 | 0 |
| zdt2-30-matched | nsga2 | 5 | 0.4384 | 0.4427 | 0.4054 | 25,034 | 0 |
| zdt2-30-matched | nsga3 | 5 | 0.4198 | 0.4448 | 0.3585 | 25,000 | 0 |
| zdt3-30-matched | nsga2 | 5 | 1.1912 | 1.2383 | 1.1577 | 25,028 | 0 |
| zdt3-30-matched | nsga3 | 5 | 1.1903 | 1.2067 | 1.1016 | 25,000 | 0 |
| dtlz2-3-matched | nsga2 | 5 | 0.0797 | 0.1416 | 0.0516 | 25,019 | 0 |
| dtlz2-3-matched | nsga3 | 5 | 0.0436 | 0.1220 | 0.0156 | 25,023 | 0 |
| dtlz1-3-matched | nsga2 | 5 | 0 | 0 | 0 | 40,053 | 0 |
| dtlz1-3-matched | nsga3 | 5 | 0 | 0 | 0 | 40,079 | 0 |

Why the DTLZ hypervolumes are near 0: on DTLZ2 mostly the operator bugs below; DTLZ1 stays at 0 with textbook operators too (see Bugs found). Not the adapter's reporting: the non-dominated set of all 2 mu individuals of the last generation, instead of the mu survivors, gives the same hypervolumes in every run of the separate tests. Not the `f32` scores either, as far as the numbers show: the objectives of these populations are between about 0.01 and a few hundred, where `f32`'s 7 significant digits separate all but nearly identical points; the front itself is recomputed in `f64`.

## Can't run

Nothing: radiate runs all 14 scenarios.

## Bugs found

| Bug | Effect | Worked around | Reported |
|---|---|---|---|
| `SimulatedBinaryCrossover` centres the child on (p1 − p2) / 2 instead of (p1 + p2) / 2, and writes only the first parent ([simulated_binary.rs, lines 59-68](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-alters/src/crossovers/simulated_binary.rs#L59-L68)). For parents in [0, 1] the child lands around 0, or below and is clamped to 0. | every multi-objective run, see below | no | [pkalivas/radiate#28](https://github.com/pkalivas/radiate/issues/28) |
| `PolynomialMutator` computes the new value from a bound instead of from the gene: `min + mutq (max − min)` without the `− 1` of the first branch ([polynomial.rs, lines 46-58](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-alters/src/mutators/polynomial.rs#L46-L58)), so a mutated gene jumps next to a bound | every multi-objective run | no | [pkalivas/radiate#28](https://github.com/pkalivas/radiate/issues/28) |

Both are in 1.3.1's source as described, unchanged since the issue. How the results show them: the crossover pulls the variables to 0, which is where ZDT's distance variables have their optimum (on ZDT1, every run's front reaches g = 1, the minimum, and in 9 of the 10 runs no front point has g above 1.02), and away from DTLZ's, which have it at 0.5, so the DTLZ hypervolumes are near 0. A diagnostic, not part of the adapter: a copy of it with the two operators written as in the textbook (SBX with two children around the parents' mean, Deb's bounded polynomial mutation), seeds 0 to 4, median hypervolumes:

| Operators | ZDT1 NSGA-II | ZDT1 NSGA-III | DTLZ2 NSGA-II | DTLZ2 NSGA-III | DTLZ1 NSGA-II | DTLZ1 NSGA-III |
|---|---|---|---|---|---|---|
| radiate's (the benchmark) | 0.805 | 0.784 | 0.080 | 0.044 | 0 | 0 |
| textbook SBX and polynomial mutation | 0.428 | 0.022 | 0.581 | 0.653 | 0 | 0 (best 0.12) |
| textbook SBX, radiate's mutation | 0.792 | | 0.330 | | | |
| radiate's SBX, textbook mutation | 0.862 | | 0.314 | | | |

This diagnostic used the earlier DTLZ1 reference point (1.1, 1.1, 1.1). So on DTLZ2 the bugs cost most of the hypervolume; on ZDT they help radiate, and radiate's NSGA-II and NSGA-III with textbook operators converge slowly on ZDT1 (after 25,000 evaluations, NSGA-II's fronts have g between 1.35 and 1.9, NSGA-III's between 2.1 and 2.8); and DTLZ1 stays at 0 either way within 40,000 evaluations.

Also found, not filed:
- The guide recommends `ShuffleCrossover` for permutation problems ([alters/index.md, line 37](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/alters/index.md#L37)), but on a `PermutationChromosome` its children aren't permutations; the engine replaces them with random individuals without a warning (see Permutation above).
- The guide describes `GaussianMutator` as producing "small, incremental changes" ([mutators.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/alters/mutators.md#L45-L56)); its standard deviation is a quarter of the gene's initial range.
