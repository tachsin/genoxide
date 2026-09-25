# genoxide (Rust, 0.6.0)

genoxide is this repository's library: genetic algorithms, evolution strategies, CMA-ES, differential evolution, particle swarms, local search and multi-objective algorithms in Rust. Its documentation is the [README](../../../README.md), the guide [AGENTS.md](../../../AGENTS.md) (decision tables, settings and templates), the [examples](../../../examples/) and the API docs ([docs.rs/genoxide](https://docs.rs/genoxide), from the rustdoc in [src/](../../../src/)).

genoxide's authors run this benchmark, so [rule 6.6](../rules.md#6-which-methods-run) applies: the methods are the ones genoxide's docs prefer for each problem type (rule 6.2), and they run with their builder's defaults. A setting without a default takes a standard value from the literature, cited below. Settings from the templates and examples aren't used: some were written on these problems.

Adapter: [benchmarks/adapters/genoxide/](../../../benchmarks/adapters/genoxide/).
Know a better way to solve one of these problems with genoxide? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs genoxide

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), in Rust, a closure per genome, as every template in AGENTS.md writes them ([main.rs, lines 49-180](../../../benchmarks/adapters/genoxide/src/main.rs#L49-L180)). The multi-objective ones are the adapter's own too, not genoxide's `multi::problems`; a unit test checks that the two agree.
- **Evaluations:** the adapter counts every call of the fitness function itself, with a counter around it ([`solve`, lines 256-388](../../../benchmarks/adapters/genoxide/src/main.rs#L256-L388), [`solve_front`, lines 555-607](../../../benchmarks/adapters/genoxide/src/main.rs#L555-L607)), and reports that count. It also compares it with genoxide's own `Outcome::evaluations()` and prints any difference to stderr. In every separate test below, the two counts were equal.
- **First hit:** the counter records the first evaluation whose value reaches the target, and the clock at that moment (`first_hit`).
- **A child identical to a parent** inherits the parent's fitness, without a call of the fitness function: genoxide's documented behavior (AGENTS.md, [Fitness functions](../../../AGENTS.md#fitness-functions)). That evaluation is saved, and isn't counted.
- **Stop:** the target or the evaluation budget (`Stop::target(..).or(Stop::evaluations(..))`), checked by genoxide after every generation, and the 60 s cap, an abort flag that the adapter sets after the generation that reaches it. A run can go past the budget by at most one generation. CMA-ES with IPOP restarts also prints `last_generation`, the evaluations of its last generation, which grows at every restart, for the budget check.
- **Restarts on convergence (rule 2.2):** DE and CMA-ES (IPOP) restart by themselves when a run converges (their own criteria, inside one genoxide run). The GA, the evolution strategy and local search have no convergence criterion. genoxide's engine ends a run by itself only with `StopReason::Stalled`, after 10,000 generations in a row without a genome to evaluate (e.g. every child a copy of its parent; AGENTS.md, [Troubleshooting](../../../AGENTS.md#troubleshooting)). The engine stalls only when every stop condition needs new evaluations, which is why the time cap is an abort flag rather than `Stop::time`. After a stall, the adapter starts the method again from a new random start with the seed `(seed + 1) * 1,000,000 + restart`, keeps the best solution and counts every evaluation, until the target, the budget or the time cap; a run that restarted prints `restarts`. No separate test stalled; a run forced to stall (a GA without mutation) restarted as it should.
- **Bounds (rule 2.4):** genoxide's own bound handling keeps every evaluated solution inside the box: SBX and polynomial mutation are the bounded versions (their rustdoc); CMA-ES draws a sample outside the bounds again, up to 100 times, and then clips it (the rustdoc of `Cmaes`); DE sets a trial gene outside the bounds halfway between the parent's gene and the bound ("bounce-back", the rustdoc of `De`); the evolution strategy reflects its mutations into the bounds (the rustdoc of `Es`). The adapter's counter counts the evaluated solutions outside the bounds and prints `outside`: 0 in every run of the separate tests.
- **Time:** the clock starts before the algorithm is built (which creates its random initial population) and stops when the run ends, before the output is formatted.
- **One thread:** genoxide is built without its `parallel` feature, and the engines evaluate sequentially.
- **Seeds:** each seed goes to the algorithm's `.seed(...)`; a seed repeats a run exactly.
- **Solutions:** each run prints its best genome (`outcome.best_genome()`), and a multi-objective run the non-dominated individuals of its final population (`outcome.front()`) with their genomes.

**Settings from the literature**, for settings without a default ([lines 394-406](../../../benchmarks/adapters/genoxide/src/main.rs#L394-L406)):
- a GA's population of 100, binary tournament selection (`Tournament::new(2)`), SBX with η 20 and polynomial mutation with η 20 at 1 / n per gene: Deb, Pratap, Agarwal and Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation 6(2), 2002;
- bit-flip mutation at 1 / n per gene: Mühlenbein, "How genetic algorithms really work: mutation and hillclimbing", PPSN 1992;
- swap mutation (`SwapMutation::new()`, one swap) for permutations;
- the evolution strategy's 15 parents and 100 offspring: Bäck and Schwefel, "An overview of evolutionary algorithms for parameter optimization", Evolutionary Computation 1(1), 1993.

The crossover of a GA, required too, is the first one AGENTS.md's table ([Choosing the pieces](../../../AGENTS.md#choosing-the-pieces)) lists for the genome (rule 6.2): `UniformCrossover` for bits, `OrderCrossover` for permutations, `SimulatedBinaryCrossover` for reals. The rates (crossover 0.9, mutation 1) and the scheme (generational, elitism 1) are the builder's defaults.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched (OneMax 100 and 1000):** the matched GA of the [README](../../../benchmarks/README.md), with genoxide's own components: population 300, `Tournament::new(3)` (with replacement), `PointCrossover::two_point()` applied to each consecutive pair of parents with probability 0.5, each child mutated with probability 0.2 by `BitFlip::per_gene(1 / n)`, and `Scheme::Generational { elitism: 0 }`: generational, no elitism ([lines 411-429](../../../benchmarks/adapters/genoxide/src/main.rs#L411-L429)). One difference from DEAP's eaSimple: genoxide doesn't evaluate any child identical to a parent, also one that was crossed or mutated back to a parent's genome; DEAP evaluates those again.
- **Idiomatic (OneMax 100):** the GA, the one method AGENTS.md presents for binary genomes, with the settings above: population 100, binary tournament, `UniformCrossover`, `BitFlip::per_gene(1 / n)`, the default rates and scheme ([lines 430-443](../../../benchmarks/adapters/genoxide/src/main.rs#L430-L443)).

**Keeping going:** a GA runs to the budget by itself.

**Left out:**
- The settings of the OneMax examples (examples/one_max.rs, the README's "A first look", AGENTS.md's first program: tournament 3), which the adapter used before: example settings (rule 6.6).
- Local search: the docs present it for permutations, not for binary genomes.

**Separate tests** (2026-09-25, genoxide 0.6.0 at 03e237b, seeds 0 to 4, the scenario's budget, 60 s cap):

OneMax 100, matched (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 4,789 | 100 | 100 | 100 | 0 |

OneMax 1000, matched (budget 2,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 53,597 | 1,000 | 1,000 | 1,000 | 0 |

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 3,418 | 100 | 100 | 100 | 0 |

## Permutation: N-Queens 32 and 64

**Methods:** AGENTS.md states one preference for permutations: local search "often beats a GA on permutations". The GA comes second: the docs present it for permutations too.
- **`local_search`:** hill climbing with its defaults: 1 neighbor per step and the acceptance `NotWorse`, which moves to equal neighbors across plateaus; swap neighbors ([lines 448-458](../../../benchmarks/adapters/genoxide/src/main.rs#L448-L458)).
- **`ga`:** population 100, binary tournament, `OrderCrossover`, `SwapMutation::new()`, the default rates and scheme ([lines 460-473](../../../benchmarks/adapters/genoxide/src/main.rs#L460-L473)).

**Keeping going:** neither has a convergence criterion: they run to the budget. Local search evaluates at least one new neighbor every step (genoxide redraws a neighbor that didn't change), so it can't stall.

**Left out:**
- The settings of the N-Queens examples, which the adapter used before: examples/n_queens.rs's (20 + 20) GA without crossover, the rustdoc example's 4 neighbors per step, and tabu search as python/examples/n_queens.py sets it (32 neighbors, tenure 20): example settings (rule 6.6).
- Tabu search (`Acceptance::Tabu { tenure }`) and simulated annealing: the Rust docs present them as options of local search, not for this problem type, and their tenure or temperature have no default and no standard value.

**Separate tests** (2026-09-25, genoxide 0.6.0 at 03e237b, seeds 0 to 4):

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| local_search | 5 | 5 | 1,145 | 0 | 0 | 0 | 0 |
| ga | 5 | 2 | 237,595 | 1 | 0 | 2 | 0 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| local_search | 5 | 5 | 2,385 | 0 | 0 | 0 | 0 |
| ga | 5 | 0 | - | 2 | 1 | 3 | 0 |

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:** AGENTS.md states a preference for two: CMA-ES is "the strongest general choice for continuous problems", with restarts "for multimodal functions"; differential evolution "often needs far fewer evaluations than a GA". The third is the GA, which [Choosing the pieces](../../../AGENTS.md#choosing-the-pieces) presents for real numbers, and which AGENTS.md prefers on multimodal problems as an island model (below).
- **`cma_es`:** CMA-ES with its defaults (AGENTS.md: "Nothing needs tuning": 4 + ⌊3 ln n⌋ samples, an initial step of 0.3 of each range) and IPOP restarts, which AGENTS.md ([CMA-ES](../../../AGENTS.md#cma-es)) and the rustdoc of `Restarts::Ipop` ("suits multimodal functions with a global structure, like Rastrigin") give for multimodal functions ([lines 494-507](../../../benchmarks/adapters/genoxide/src/main.rs#L494-L507)).
- **`de`:** differential evolution with the builder's defaults ([lines 509-512](../../../benchmarks/adapters/genoxide/src/main.rs#L509-L512)): SHADE's published settings (Tanabe and Fukunaga, IEEE CEC 2013, as the rustdoc of `De::builder` cites it: current-to-pbest/1 with an archive and a random p per trial, SHADE's adaptation of F and CR, 100 individuals), and restarts when the population converges or stalls: genoxide's own choice, from #113.
- **`ga`:** population 100, binary tournament, SBX with η 20 at the default rate of 0.9, polynomial mutation with η 20 at 1 / n, the default scheme ([lines 534-548](../../../benchmarks/adapters/genoxide/src/main.rs#L534-L548)).

**Keeping going:** the GA has no convergence criterion and runs to the budget. DE restarts by itself when its population converges or stalls (its default), and CMA-ES with IPOP restarts with a doubled population when a run meets one of its stop criteria.

**Left out:**
- **The island model**, which AGENTS.md prefers to one GA population on multimodal problems ("more diverse than one large population, and often faster on multimodal problems"): the number and size of its islands have no default and no standard value, and the docs' only values are the template's, written on Rastrigin (rule 6.6). The GA runs as one population.
- **L-SHADE** (`De::l_shade(real, budget)`), which [python/examples/rastrigin.py](../../../python/examples/rastrigin.py) runs: a setting of DE, not its default.
- **BIPOP restarts** (`cmaes::Restarts::Bipop`, "good on a wider range of multimodal functions than IPOP"): the rustdoc names this kind of function for IPOP ("with a global structure, like Rastrigin").
- **Particle swarm:** AGENTS.md: "On separable functions like Rastrigin, differential evolution with a small CR does much better."
- **The evolution strategy:** AGENTS.md: "For hard problems (rotated, badly conditioned or multimodal), CMA-ES is stronger."

**Separate tests** (2026-09-25, genoxide 0.6.0 at 03e237b, seeds 0 to 4):

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 74,822 | 0.007264 | 0.005887 | 0.008683 | 0 |
| de | 5 | 5 | 41,016 | 0.00921 | 0.00476 | 0.009991 | 0 |
| ga | 5 | 5 | 87,539 | 0.009155 | 0.006517 | 0.009792 | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 907,857 | 0.007948 | 0.007243 | 0.009791 | 0 |
| de | 5 | 5 | 101,455 | 0.00905 | 0.008014 | 0.00999 | 0 |
| ga | 5 | 5 | 782,489 | 0.00982 | 0.008953 | 0.009988 | 0 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 3,203 | 0.00943 | 0.008762 | 0.009948 | 0 |
| de | 5 | 5 | 19,144 | 0.00973 | 0.008253 | 0.009884 | 0 |
| ga | 5 | 0 | - | 0.02309 | 0.02079 | 0.03297 | 0 |

## Continuous, unimodal: Rosenbrock 10

**Methods:** AGENTS.md states a preference for three.
- **`cma_es`:** "the strongest general choice for continuous problems ... especially when the genes interact (rotated or badly conditioned functions)" ([CMA-ES](../../../AGENTS.md#cma-es)), with its defaults and IPOP restarts. The restarts are for rule 2.2: a converged run starts again with the library's restart mechanism; AGENTS.md's troubleshooting table says to add them when a CMA-ES stops improving.
- **`de`:** "For continuous problems on `Real` genomes, differential evolution often needs far fewer evaluations than a GA" ([Differential evolution](../../../AGENTS.md#differential-evolution)), with its defaults (above).
- **`es`:** the evolution strategy, "For smooth real-valued problems that need precise answers" ([Evolution strategy](../../../AGENTS.md#evolution-strategy-with-self-adaptation)), with its defaults (intermediate recombination of all parents, comma selection, a step size per gene starting at 0.3 of each range) and 15 parents and 100 offspring ([lines 517-529](../../../benchmarks/adapters/genoxide/src/main.rs#L517-L529)).

**Keeping going:** CMA-ES and DE restart by themselves (above); the evolution strategy has no convergence criterion and runs to the budget.

**Left out:**
- **Particle swarm**, whose template (AGENTS.md, [Particle swarm optimization](../../../AGENTS.md#particle-swarm-optimization)) is this problem: the docs state no preference for it, and the three above come first.
- **The GA:** AGENTS.md prefers DE to a GA on continuous problems.

**Separate tests** (2026-09-25, genoxide 0.6.0 at 03e237b, seeds 0 to 4):

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 5,837 | 0.009198 | 0.008564 | 0.009741 | 0 |
| de | 5 | 5 | 38,077 | 0.008082 | 0.004516 | 0.008217 | 0 |
| es | 5 | 1 | 489,151 | 0.02353 | 0.009992 | 0.1455 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

These scenarios are matched: every library runs the settings of the [README](../../../benchmarks/README.md#scenarios), with its own operators.

**Methods** ([lines 609-683](../../../benchmarks/adapters/genoxide/src/main.rs#L609-L683)), the five that rule 6.1 runs in the matched multi-objective scenarios:
- **`nsga2`, `spea2`, `sms_emoa`:** 100 individuals (92 with 3 objectives), `SimulatedBinaryCrossover::new(15.0)` at their default rate of 0.9, `PolynomialMutation::per_gene(1 / n, 20.0)`. SMS-EMOA is steady-state: one child per generation (`.offspring(1)`; genoxide's default breeds a population's worth).
- **`nsga3`:** Das-Dennis directions with 99 divisions (100) with 2 objectives and 12 (91) with 3, a population of 100 (92 with 3 objectives: genoxide's default would be the 91 directions; 92 is the size the scenario gives the other algorithms), SBX with η 30 at its default rate of 1, and the same polynomial mutation.
- **`moead`:** 100 weight vectors (99 divisions; 91 with 3 objectives and 12 divisions), its defaults of 20 neighbors, parents from the neighborhood with probability 0.9 and at most 2 replacements, Tchebycheff with 2 objectives and PBI with θ 5 with 3, SBX with η 20 at its default rate of 1, and the same polynomial mutation.
- **No duplicate elimination:** NSGA-II, NSGA-III, SPEA2 and SMS-EMOA drop a child that equals a member of the population or an earlier child by default; the matched scenarios turn it off (`.eliminate_duplicates(false)`). MOEA/D has none.

**Keeping going:** none of the five has a convergence criterion; every run uses its whole budget, and ends after the generation that reaches it. SBX and polynomial mutation keep every gene in [0, 1]: `outside` was 0 in every run.

**The front:** the non-dominated individuals of the final population, `outcome.front()`: for SPEA2, of its archive of 100 (92). The hypervolume comes from the solutions, as `run.py` computes it.

**Separate tests** (2026-09-25, genoxide 0.6.0 at 03e237b, seeds 0 to 4; the hypervolume as `run.py` computes it):

ZDT1 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | Capped |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.8693 | 0.8697 | 0.8691 | 25,024 | 100 | 0 |
| nsga3 | 5 | 0.8701 | 0.8704 | 0.8699 | 25,022 | 100 | 0 |
| spea2 | 5 | 0.8703 | 0.8707 | 0.8700 | 25,083 | 100 | 0 |
| sms_emoa | 5 | 0.8719 | 0.8720 | 0.8719 | 25,000 | 100 | 0 |
| moead | 5 | 0.8686 | 0.8689 | 0.8683 | 25,020 | 100 | 0 |

ZDT2 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | Capped |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.5363 | 0.5365 | 0.5360 | 25,041 | 100 | 0 |
| nsga3 | 5 | 0.5366 | 0.5370 | 0.5365 | 25,016 | 100 | 0 |
| spea2 | 5 | 0.5368 | 0.5373 | 0.5363 | 25,041 | 100 | 0 |
| sms_emoa | 5 | 0.5386 | 0.5387 | 0.5386 | 25,000 | 100 | 0 |
| moead | 5 | 0.5357 | 0.5358 | 0.5345 | 25,083 | 100 | 0 |

ZDT3 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | Capped |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 1.3269 | 1.3273 | 1.2447 | 25,059 | 100 | 0 |
| nsga3 | 5 | 1.3255 | 1.3259 | 1.3241 | 25,015 | 100 | 0 |
| spea2 | 5 | 1.3273 | 1.3277 | 1.3270 | 25,063 | 100 | 0 |
| sms_emoa | 5 | 1.3244 | 1.3293 | 1.2458 | 25,000 | 100 | 0 |
| moead | 5 | 1.3216 | 1.3226 | 1.3208 | 25,032 | 83 | 0 |

DTLZ2, 3 objectives (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | Capped |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.6972 | 0.7021 | 0.6929 | 25,062 | 92 | 0 |
| nsga3 | 5 | 0.7443 | 0.7444 | 0.7441 | 25,023 | 92 | 0 |
| spea2 | 5 | 0.7292 | 0.7316 | 0.7282 | 25,029 | 92 | 0 |
| sms_emoa | 5 | 0.7556 | 0.7557 | 0.7554 | 25,000 | 92 | 0 |
| moead | 5 | 0.7441 | 0.7441 | 0.7440 | 25,014 | 91 | 0 |

DTLZ1, 3 objectives (budget 40,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | Capped |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.1338 | 0.1369 | 0.1307 | 40,036 | 92 | 0 |
| nsga3 | 5 | 0.1398 | 0.1400 | 0.1390 | 40,038 | 92 | 0 |
| spea2 | 5 | 0.1392 | 0.1398 | 0.1391 | 40,032 | 92 | 0 |
| sms_emoa | 5 | 0.1401 | 0.1402 | 0.1401 | 40,000 | 92 | 0 |
| moead | 5 | 0.1397 | 0.1398 | 0.1393 | 40,039 | 91 | 0 |

## Can't run

genoxide runs every scenario. It has no multi-objective algorithm beyond the five that rule 6.1 runs.

## Bugs found

None in genoxide in this review. The adapter's own evaluation count matched genoxide's `Outcome::evaluations()` in every run, and its fitness functions agree with genoxide's `multi::problems` (ZDT1 to 3, DTLZ1 and 2).

Observations for the library, not bugs:
- The GA with the settings above reaches N-Queens 32 in 2 of 5 runs and N-Queens 64 in none, where hill climbing reaches both in every run.
- The GA doesn't reach the Ackley 30 target within 1,000,000 evaluations: its runs end between 0.021 and 0.033.
- The evolution strategy with 15 parents and 100 offspring reaches the Rosenbrock 10 target in 1 of 5 runs.
- On ZDT3, 1 NSGA-II run and 2 SMS-EMOA runs of 5 end without the front's last segment (f1 above 0.65), with a hypervolume of about 1.245 instead of 1.33.
