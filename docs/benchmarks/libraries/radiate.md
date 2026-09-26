# radiate (Rust, 1.3.1)

radiate is a Rust library for genetic algorithms, with Python bindings, genetic programming and neuroevolution. Its one search method is the `GeneticEngine`, a GA: each generation keeps `population_size × (1 − offspring_fraction)` survivors and breeds the rest. Multi-objective problems use the same engine with the NSGA-II or NSGA-III selectors. Its user guide is at [pkalivas.github.io/radiate](https://pkalivas.github.io/radiate/); the links below point to the guide's sources and examples at the tag [v1.3.1](https://github.com/pkalivas/radiate/tree/v1.3.1) (579a943).

Adapter: [benchmarks/adapters/radiate/](../../../benchmarks/adapters/radiate/).
Know a better way to solve one of these problems with radiate? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs radiate

- **Generation:** radiate evaluates only individuals whose genome changed ([engine/index.md, "Life of an epoch"](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/engine/index.md#L42-L63)). A crossover that writes one parent (`MeanCrossover`, `SimulatedBinaryCrossover`) marks both as changed ([alter.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-core/src/alter.rs#L263-L266)), so one unchanged individual is evaluated again.
- **Defaults** ([engine/index.md, "Engine Defaults"](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/engine/index.md#L7-L26)): population 100, roulette offspring selection, a tournament of 3 for the survivors, offspring fraction 0.8, `UniformCrossover(0.5)`, `UniformMutator(0.1)`. Individuals older than `max_age` = 20 generations are replaced by random ones ([builder/mod.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-engines/src/builder/mod.rs#L552-L556), [steps/filter.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-engines/src/steps/filter.rs#L27-L52)); the idiomatic runs keep this.
- **Evaluations:** the fitness wrapper counts every call and records the first hit in `f64` ([Budget::record](../../../benchmarks/adapters/radiate/src/main.rs#L236-L251)).
- **Stop and keeping going (rule 2.2):** radiate has no stop criterion of its own ("an engine with no limit attached runs forever in Rust", [engine/index.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/engine/index.md#L88-L90)). The adapter's `until` ([run_engine](../../../benchmarks/adapters/radiate/src/main.rs#L304-L316)) ends the run after the generation that reaches the target, the budget or 60 s ([Budget::done](../../../benchmarks/adapters/radiate/src/main.rs#L281-L287)). No run restarts.
- **Bounds (rule 2.4): clipping.** `FloatCodec::vector` sets the problem's range as the genes' bounds, and the float alterers clip to them (`FloatGene::set_allele`, `safe_clamp`, [float.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-core/src/genome/chromosomes/float.rs#L82-L85)). The wrapper counts solutions outside the bounds ([Budget::check_bounds](../../../benchmarks/adapters/radiate/src/main.rs#L262-L267)).
- **Best solution:** `Generation::value`, radiate's best by its `f32` scores, recomputed in `f64` ([run_single](../../../benchmarks/adapters/radiate/src/main.rs#L353-L391)). If it misses the target in `f64` while an evaluated solution reached it, the run reports the first hit's solution; no test run needed this.
- **Fitness:** `raw_fitness_fn`, the documented way to skip decoding ([fitness.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/fitness.md#L81-L96)), in `f64` ([fitness functions](../../../benchmarks/adapters/radiate/src/main.rs#L50-L165)).
- **One thread:** no `rayon` feature, default `Executor::Serial` ([executors.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/executors.md)).
- **Seeds:** each run is inside `random_provider::scoped_seed(seed, ..)`, which reseeds radiate's thread-local generator ([random_provider.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-core/src/domain/random_provider.rs#L40-L58)). `random_provider::seed`, which the examples call, only seeds new threads.
- **Separate tests:** 2026-09-25, radiate 1.3.1, seeds 0 to 4, the scenario's budget, 60 s cap. `outside` was 0 in every run.

**The idiomatic methods** ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L398-L421)), up to 3 per problem type:
- `ga`: radiate's example for the problem type, as it is.
- one solver per crossover in the guide's recipe for the genome type, the alterers' "Best Practices" ([alters/index.md, lines 30-48](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/alters/index.md#L30-L48)):

  > 1. **Rate Selection**: Start with conservative rates (0.01 for mutation, 0.5-0.8 for crossover) [...]
  > 2. **Choosing the Right Alterer**:
  >    - For continuous problems: Use Gaussian or Arithmetic mutators with Blend/Intermediate crossover
  >    - For permutation problems: Use Swap/Scramble mutators with PMX or Shuffle crossover
  >    - For binary problems: Use Uniform mutator with Multi-point or Uniform crossover

A recipe solver is the problem type's example (population, selectors) with the recipe's alterers:
- mutation rate 0.01, the stated starting rate;
- crossover rate: the example's own if it's in the stated 0.5 to 0.8 (0.5 for knapsack and Rastrigin, 0.75 for Rosenbrock); the TSP example's 0.4 isn't, so the permutation recipe takes the default crossover's 0.5;
- where the recipe names two mutators, the example's (Swap for TSP, Arithmetic for Rastrigin and Rosenbrock);
- where it names two crossovers, both, as separate solvers (Shuffle is left out, see Permutation);
- `MultiPointCrossover` with 2 points, as in every radiate example; `BlendCrossover` and `IntermediateCrossover` with α 0.5, the guide's snippets ([crossovers.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/src/rust/alters/crossovers.rs#L4-L12)).

**Left out everywhere:**
- The guide's alterers example ([example.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/src/rust/alters/example.rs#L30-L50)): it shows wiring on a 2-variable float problem, not a problem type here.
- Speciation: the guide says to use it "not by default" ([diversity/index.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/diversity/index.md#L32-L34)), and its threshold is tuned by trial ([species.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/diversity/species.md)).
- Rate schedules (`Expr` rates): shapes that "_might_ be useful" ([alters/rate.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/alters/rate.md)), not settings for a problem type.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched** ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L437-L476)): population 300, offspring fraction 1.0 (no survivors: generational, no elitism), `TournamentSelector(3)` (with replacement), `MultiPointCrossover(0.5, 2)`, `BitFlipMutator(0.2 / size)`, max age off. Differences from eaSimple:
  - no per-individual mutation probability: each bit flips with probability 0.2 / size, the same mean of 0.2 flips per child (18% of children mutated instead of 12.6%);
  - each child is crossed with probability 0.5 with a random other child, so a child can be crossed more than once (DEAP: disjoint pairs); the same expected 150 crossovers per generation;
  - cut points are drawn from 0..size (a cut at 0 changes nothing), DEAP's from 1..size.
- **Idiomatic**, radiate's binary example, the knapsack ([examples/rust/knapsack](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/knapsack/src/main.rs#L11-L18)): a `SubSetCodec` (a `BitChromosome`, one bit per item), `max_age(50)`, the other engine defaults ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L477-L515)):
  - `ga` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L489-L492)): the default alterers `UniformCrossover(0.5)` and `UniformMutator(0.1)`, which redraws a bit, flipping it with probability 0.05;
  - `ga_uniform` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L493-L496)): `UniformCrossover(0.5)`, `UniformMutator(0.01)`;
  - `ga_multipoint` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L497-L500)): `MultiPointCrossover(0.5, 2)`, `UniformMutator(0.01)`.

**Keeping going:** runs to the target or the budget.

**Left out:**
- The README's "Hello, Radiate!" ([hello-world](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/hello-world/src/main.rs#L8-L19)): it evolves characters, not bits.
- `BitFlipMutator` in the idiomatic runs: not in the guide's mutator table; the binary recipe names the Uniform mutator.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| onemax-100-matched | ga | 5 | 5 | 5,484 | 100 | 100 | 100 | 0 |
| onemax-1000-matched | ga | 5 | 5 | 77,342 | 1000 | 1000 | 1000 | 0 |
| onemax-100-idiomatic | ga | 5 | 0 | | 82 | 82 | 81 | 0 |
| onemax-100-idiomatic | ga_uniform | 5 | 5 | 9,681 | 100 | 100 | 100 | 0 |
| onemax-100-idiomatic | ga_multipoint | 5 | 5 | 12,156 | 100 | 100 | 100 | 0 |

`ga` flips about 5 bits per child: a child of a 99-ones parent keeps them all with probability 0.95^99 ≈ 0.6%.

## Permutation: N-Queens 32 and 64

**Methods**, radiate's permutation example, the TSP ([examples/rust/TSP](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/TSP/src/main.rs#L13-L19)): `PermutationCodec`, population 250, the other engine defaults ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L526-L540)):
- `ga` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L542-L545)): `PMXCrossover(0.4)`, `SwapMutator(0.05)`, as in the example;
- `ga_pmx` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L546-L552)): the recipe, `PMXCrossover(0.5)`, `SwapMutator(0.01)`.

**Keeping going:** runs to the target or the budget.

**Left out:**
- radiate's N-Queens example ([nqueens](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/nqueens/src/main.rs), [examples page](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/examples.md#L31-L53)): it evolves integers, not a permutation.
- `ShuffleCrossover`: its children aren't permutations ([shuffle.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-alters/src/crossovers/shuffle.rs#L35-L47)), and the engine replaces them with random ones ("Some alterers may produce invalid solutions", [alters/index.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/alters/index.md#L61-L63)).
- `ScrambleMutator`: the TSP example uses Swap.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| nqueens-32-idiomatic | ga | 5 | 0 | | 5 | 3 | 5 | 0 |
| nqueens-32-idiomatic | ga_pmx | 5 | 2 | 312,840 | 4 | 0 | 4 | 0 |
| nqueens-64-idiomatic | ga | 5 | 0 | | 16 | 15 | 16 | 0 |
| nqueens-64-idiomatic | ga_pmx | 5 | 0 | | 15 | 14 | 16 | 0 |

`RouletteSelector` weighs a minimized score by reversing the sorted scores ([roulette.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-selectors/src/roulette.rs#L20-L35)), which gives little selection pressure between close conflict counts. With max age off, `ga` ended at the same median of 5 on N-Queens 32. With `BoltzmannSelector(4)` instead (not used to choose), it reached N-Queens 32 in 5 of 5 runs (median 21,809 evaluations).

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods**, radiate's Rastrigin example ([examples/rust/rastrigin](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/rastrigin/src/main.rs#L10-L17), [examples page](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/examples.md#L57-L85)): population 500, the other engine defaults, `f64` genes in the bounds ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L573-L597)). Ackley, which has no example, uses it too.
- `ga` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L607-L609)): `UniformCrossover(0.5)`, `ArithmeticMutator(0.01)`, as in the example;
- `ga_blend` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L610-L614)): `BlendCrossover(0.5, 0.5)`, `ArithmeticMutator(0.01)`;
- `ga_intermediate` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L615-L622)): `IntermediateCrossover(0.5, 0.5)`, `ArithmeticMutator(0.01)`.

**Keeping going:** runs to the target or the budget.

**Left out:**
- `GaussianMutator`: radiate's continuous examples use Arithmetic.
- `JitterMutator`: not in the recipe.
- `MeanCrossover` and `SimulatedBinaryCrossover`: not in the recipe (SBX has the bug below).

**Separate tests** (with the shift of rule 1.4):

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

radiate's Blend crossover puts the child outside the segment between the parents (child = a − α (b − a), [blend.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-alters/src/crossovers/blend.rs#L57-L64), as its docs define it).

## Continuous, unimodal: Rosenbrock 10

**Methods**, radiate's Rosenbrock example ([examples/rust/rosenbrock](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/rosenbrock/src/main.rs#L13-L20), for 2 variables in [−2, 2]): `BoltzmannSelector(4)` for the offspring, the other engine defaults ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L585-L597)):
- `ga` ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L604-L606)): `MeanCrossover(0.75)`, `ArithmeticMutator(0.1)`, as in the example;
- `ga_blend`, `ga_intermediate`: the recipe at the example's rate, `BlendCrossover(0.75, 0.5)` or `IntermediateCrossover(0.75, 0.5)`, with `ArithmeticMutator(0.01)`.

**Keeping going** and **Left out:** as for the multimodal problems.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| rosenbrock-10-idiomatic | ga | 5 | 0 | | 6.885 | 6.858 | 7.213 | 0 |
| rosenbrock-10-idiomatic | ga_blend | 5 | 0 | | 0.240 | 0.0349 | 5.421 | 0 |
| rosenbrock-10-idiomatic | ga_intermediate | 5 | 1 | 16,326 | 0.0284 | 0.00972 | 2.089 | 0 |

`ga` ends every run in Rosenbrock's curved valley (e.g. 0.67, 0.45, 0.21, 0.05, 0.013, 0.010, ...): `ArithmeticMutator` changes a gene by a random value of the whole range ([arithmetic.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-alters/src/mutators/arithmetic.rs#L38-L58)). With max age off it ended at the same median (6.91).

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Methods** ([adapter](../../../benchmarks/adapters/radiate/src/main.rs#L661-L788)): the matched settings with radiate's own selectors, SBX and polynomial mutation, bugs included (rule 8.4):
- `nsga2`: `TournamentNSGA2Selector` for mating, `NSGA2Selector` for survivors, `SimulatedBinaryCrossover(0.9, 15)`, `PolynomialMutator(1 / n, 20)`; population 100, 92 with 3 objectives.
- `nsga3`: `RandomSelector` for mating, `NSGA3Selector` with Das-Dennis directions (99 divisions: 100, population 100; 12 divisions: 91, population 92), `SimulatedBinaryCrossover(1.0, 30)`, `PolynomialMutator(1 / n, 20)`. It runs on ZDT too, as radiate's own ZDT example does (`nsga3(12)`, [zdt.py](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/python/zdt.py#L30-L36)); the guide lists NSGA-III for "many objectives" ([selectors/index.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/selectors/index.md#L29)).
- Survival from parents and offspring: `population_size(2 mu)` with `offspring_fraction(0.5)` keeps mu of 2 mu and breeds mu.

Differences:
- The initial population is 2 mu (200 or 184): radiate sizes survivors and offspring from the initial population ([config.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-engines/src/builder/config.rs#L81-L87)), so the first generation costs mu more evaluations.
- Mating draws from all 2 mu, not the mu survivors.
- NSGA-II's crowding distance is over the whole population, not per front ([pareto.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-core/src/objectives/pareto.rs#L30-L85)).
- NSGA-III normalizes by the ideal and nadir points, without the extreme-point hyperplane ([nsga3.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-selectors/src/nsga3.rs#L144-L188)).
- The crossover rate is per individual, so an individual can be crossed more than once ([alter.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-core/src/alter.rs#L225-L243)).
- Scores are `f32`. Max age is off.
- radiate's own multi-objective examples use `TournamentSelector(5)` and `UniformMutator` ([examples/rust/DTLZ](https://github.com/pkalivas/radiate/blob/v1.3.1/examples/rust/DTLZ/src/main.rs#L13-L24)); the matched settings don't.

**Keeping going:** runs to the budget; no convergence criterion.

**Bounds:** radiate's operators clip the variables to [0, 1], including the children the SBX bug puts below 0.

**The front** (rule 7.2): the non-dominated part of the mu survivors of the last generation, objectives recomputed in `f64`. radiate's own Pareto archive (`front_size`, 800 to 900 by default, [builder/mod.rs](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-engines/src/builder/mod.rs#L575-L579)) isn't reported.

**Left out:** nothing; radiate has no other multi-objective algorithm ([selectors/index.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/selectors/index.md#L20-L34)).

**Separate tests:**

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

The non-dominated set of all 2 mu individuals gives the same hypervolumes, and `f32` separates all but nearly identical points at these objective values (0.01 to a few hundred).

## Can't run

Nothing: radiate runs all 14 scenarios.

## Bugs found

| Bug | Effect | Worked around | Reported |
|---|---|---|---|
| `SimulatedBinaryCrossover` centres the child on (p1 − p2) / 2 instead of (p1 + p2) / 2, and writes only the first parent ([simulated_binary.rs, lines 59-68](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-alters/src/crossovers/simulated_binary.rs#L59-L68)). For parents in [0, 1] the child lands around 0, or below and is clamped to 0. | every multi-objective run, see below | no | [pkalivas/radiate#28](https://github.com/pkalivas/radiate/issues/28) |
| `PolynomialMutator` computes the new value from a bound instead of from the gene: `min + mutq (max − min)` without the `− 1` of the first branch ([polynomial.rs, lines 46-58](https://github.com/pkalivas/radiate/blob/v1.3.1/crates/radiate-alters/src/mutators/polynomial.rs#L46-L58)), so a mutated gene jumps next to a bound | every multi-objective run | no | [pkalivas/radiate#28](https://github.com/pkalivas/radiate/issues/28) |

Effect: the crossover pulls the variables to 0, the optimum of ZDT's distance variables (on ZDT1 every front reaches g = 1, and in 9 of 10 runs no front point has g above 1.02) and away from DTLZ's, at 0.5. A copy of the adapter with textbook operators (SBX with two children around the parents' mean, Deb's bounded polynomial mutation), seeds 0 to 4, median hypervolumes, with the earlier DTLZ1 reference point (1.1, 1.1, 1.1):

| Operators | ZDT1 NSGA-II | ZDT1 NSGA-III | DTLZ2 NSGA-II | DTLZ2 NSGA-III | DTLZ1 NSGA-II | DTLZ1 NSGA-III |
|---|---|---|---|---|---|---|
| radiate's (the benchmark) | 0.805 | 0.784 | 0.080 | 0.044 | 0 | 0 |
| textbook SBX and polynomial mutation | 0.428 | 0.022 | 0.581 | 0.653 | 0 | 0 (best 0.12) |
| textbook SBX, radiate's mutation | 0.792 | | 0.330 | | | |
| radiate's SBX, textbook mutation | 0.862 | | 0.314 | | | |

With textbook operators, after 25,000 evaluations on ZDT1, NSGA-II's fronts have g between 1.35 and 1.9 and NSGA-III's between 2.1 and 2.8; DTLZ1 stays at 0 within 40,000 evaluations either way.

Also found, not filed:
- The guide recommends `ShuffleCrossover` for permutations ([alters/index.md, line 37](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/alters/index.md#L37)), but its children aren't permutations, and the engine replaces them with random individuals without a warning.
- The guide describes `GaussianMutator` as producing "small, incremental changes" ([mutators.md](https://github.com/pkalivas/radiate/blob/v1.3.1/docs/source/alters/mutators.md#L45-L56)); its standard deviation is a quarter of the gene's initial range.
