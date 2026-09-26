# genoxide (Rust, 0.6.0)

genoxide is this repository's library: genetic algorithms, evolution strategies, CMA-ES, differential evolution, particle swarms, local search and multi-objective algorithms in Rust. Its docs are the [README](../../../README.md), the guide [AGENTS.md](../../../AGENTS.md) (decision tables, settings and templates), the [examples](../../../examples/) and the API docs ([docs.rs/genoxide](https://docs.rs/genoxide), from the rustdoc in [src/](../../../src/)).

genoxide's authors run this benchmark, so [rule 6.6](../rules.md#6-which-methods-run) applies: the methods its docs prefer for each problem type, with their builder's defaults, and standard literature values (cited below) where there's no default. Settings from the templates and examples aren't used.

Adapter: [benchmarks/adapters/genoxide/](../../../benchmarks/adapters/genoxide/).
Know a better way to solve one of these problems with genoxide? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs genoxide

- **Fitness functions:** in Rust, a closure per genome, as AGENTS.md's templates write them ([main.rs, lines 49-180](../../../benchmarks/adapters/genoxide/src/main.rs#L49-L180)). The multi-objective ones are the adapter's, not `multi::problems`; a unit test checks that they agree.
- **Evaluations:** a counter around the fitness function counts every call and records the first hit ([`solve`, lines 256-380](../../../benchmarks/adapters/genoxide/src/main.rs#L256-L380), [`solve_front`, lines 547-607](../../../benchmarks/adapters/genoxide/src/main.rs#L547-L607)). Any difference from `Outcome::evaluations()` is printed to stderr. A child identical to a parent inherits its fitness without an evaluation (AGENTS.md, [Fitness functions](../../../AGENTS.md#fitness-functions)).
- **Stop:** `Stop::target(..).or(Stop::evaluations(..))`, after every generation, and an abort flag for the 60 s cap.
- **Keeping going (rule 2.2):** DE and CMA-ES (IPOP) restart by themselves. The GA, the evolution strategy and local search have no convergence criterion. The engine ends a run by itself only as `StopReason::Stalled`, after 10,000 generations with nothing to evaluate (AGENTS.md, [Troubleshooting](../../../AGENTS.md#troubleshooting)); the adapter then restarts the method with the seeds of rule 2.2 and prints `restarts`. The time cap is an abort flag rather than `Stop::time`, because the engine stalls only when every stop condition needs new evaluations. No test run stalled.
- **Bounds (rule 2.4):** genoxide's own, from each type's rustdoc: SBX and polynomial mutation are bounded; CMA-ES redraws a sample outside the bounds up to 100 times, then clips it (`Cmaes`); DE sets an outside trial gene halfway between the parent's gene and the bound (`De`); the evolution strategy reflects its mutations into the bounds (`Es`).
- **Time:** from before the algorithm is built (it creates the initial population) to the end of the run.
- **One thread:** built without the `parallel` feature.
- **Seeds:** `.seed(...)`; a seed repeats a run exactly.
- **Solutions:** `outcome.best_genome()`; for several objectives, `outcome.front()`.
- **Separate tests:** 2026-09-25, genoxide 0.6.0 at 03e237b, seeds 0 to 4, the scenario's budget, 60 s cap. The counts equalled `Outcome::evaluations()`, and `outside` was 0, in every run.

**Settings from the literature**, for settings without a default ([lines 386-398](../../../benchmarks/adapters/genoxide/src/main.rs#L386-L398)):
- a GA's population of 100, binary tournament (`Tournament::new(2)`), SBX η 20 and polynomial mutation η 20 at 1 / n per gene: Deb, Pratap, Agarwal and Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation 6(2), 2002;
- bit-flip mutation at 1 / n per gene: Mühlenbein, "How genetic algorithms really work: mutation and hillclimbing", PPSN 1992;
- swap mutation (`SwapMutation::new()`, one swap) for permutations;
- the evolution strategy's 15 parents and 100 offspring: Bäck and Schwefel, "An overview of evolutionary algorithms for parameter optimization", Evolutionary Computation 1(1), 1993.

A GA's crossover is the first AGENTS.md's table ([Choosing the pieces](../../../AGENTS.md#choosing-the-pieces)) lists for the genome: `UniformCrossover` (bits), `OrderCrossover` (permutations), `SimulatedBinaryCrossover` (reals). The rates (crossover 0.9, mutation 1) and the scheme (generational, elitism 1) are the builder's defaults.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched:** the [README](../../../benchmarks/README.md)'s matched GA with genoxide's components: population 300, `Tournament::new(3)` (with replacement), `PointCrossover::two_point()` on each consecutive pair at 0.5, each child mutated at 0.2 by `BitFlip::per_gene(1 / n)`, `Scheme::Generational { elitism: 0 }` ([lines 403-421](../../../benchmarks/adapters/genoxide/src/main.rs#L403-L421)). Difference from eaSimple: a child identical to a parent isn't evaluated, also one crossed or mutated back to it.
- **Idiomatic (OneMax 100):** the GA, the one method AGENTS.md presents for binary genomes: population 100, binary tournament, `UniformCrossover`, `BitFlip::per_gene(1 / n)`, the default rates and scheme ([lines 422-435](../../../benchmarks/adapters/genoxide/src/main.rs#L422-L435)).

**Keeping going:** to the target or the budget.

**Left out:**
- The OneMax examples' settings (examples/one_max.rs, the README's first example, AGENTS.md's first program: tournament 3): example settings (rule 6.6).
- Local search: presented for permutations, not binary genomes.

**Separate tests:**

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

**Methods:** AGENTS.md: local search "often beats a GA on permutations"; the GA, also presented for permutations, comes second.
- **`local_search`:** hill climbing with its defaults: 1 neighbor per step, `NotWorse` acceptance (moves across plateaus), swap neighbors ([lines 440-450](../../../benchmarks/adapters/genoxide/src/main.rs#L440-L450)).
- **`ga`:** population 100, binary tournament, `OrderCrossover`, `SwapMutation::new()`, the default rates and scheme ([lines 452-465](../../../benchmarks/adapters/genoxide/src/main.rs#L452-L465)).

**Keeping going:** no convergence criterion. Local search redraws a neighbor that didn't change, so it can't stall.

**Left out:**
- The N-Queens examples' settings (examples/n_queens.rs's (20 + 20) GA without crossover, the rustdoc example's 4 neighbors per step, python/examples/n_queens.py's tabu search with 32 neighbors and tenure 20): example settings (rule 6.6).
- Tabu search (`Acceptance::Tabu { tenure }`) and simulated annealing: options of local search, not presented for this problem type; tenure and temperature have no default or standard value.

**Separate tests:**

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

**Methods:** AGENTS.md prefers CMA-ES ("the strongest general choice for continuous problems", with restarts "for multimodal functions") and DE ("often needs far fewer evaluations than a GA"). The third is the GA, which [Choosing the pieces](../../../AGENTS.md#choosing-the-pieces) presents for real numbers.
- **`cma_es`:** its defaults ("Nothing needs tuning": 4 + ⌊3 ln n⌋ samples, an initial step of 0.3 of each range) and IPOP restarts, which AGENTS.md ([CMA-ES](../../../AGENTS.md#cma-es)) and the rustdoc of `Restarts::Ipop` ("suits multimodal functions with a global structure, like Rastrigin") give for multimodal functions ([lines 487-499](../../../benchmarks/adapters/genoxide/src/main.rs#L487-L499)).
- **`de`:** the builder's defaults ([lines 501-504](../../../benchmarks/adapters/genoxide/src/main.rs#L501-L504)): SHADE's published settings (Tanabe and Fukunaga, IEEE CEC 2013, cited in the rustdoc of `De::builder`: current-to-pbest/1 with an archive and a random p per trial, SHADE's adaptation of F and CR, 100 individuals), and genoxide's own restarts when the population converges or stalls (#113).
- **`ga`:** population 100, binary tournament, SBX η 20 at the default 0.9, polynomial mutation η 20 at 1 / n, the default scheme ([lines 526-540](../../../benchmarks/adapters/genoxide/src/main.rs#L526-L540)).

**Keeping going:** the GA runs to the budget; DE and CMA-ES (IPOP, with a doubled population) restart.

**Left out:**
- **The island model**, which AGENTS.md prefers to one GA population on multimodal problems: the number and size of its islands have no default or standard value, and the only values are the template's (rule 6.6).
- **L-SHADE** (`De::l_shade(real, budget)`), which [python/examples/rastrigin.py](../../../python/examples/rastrigin.py) runs: a setting of DE, not its default.
- **BIPOP restarts:** the rustdoc names this kind of function for IPOP.
- **Particle swarm:** AGENTS.md's stated preferences for continuous problems are CMA-ES, DE and the evolution strategy, so PSO isn't among the first three (rules 6.2 and 6.4).
- **The evolution strategy:** AGENTS.md: "For hard problems (rotated, badly conditioned or multimodal), CMA-ES is stronger."

**Separate tests:**

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

**Methods:** AGENTS.md prefers three:
- **`cma_es`:** "The strongest general choice for continuous problems ... especially when the genes interact (rotated or badly conditioned functions)" ([CMA-ES](../../../AGENTS.md#cma-es)), with its defaults and IPOP restarts (rule 2.2; AGENTS.md's troubleshooting table gives them for a CMA-ES that converged without restarts).
- **`de`:** "For continuous problems on `Real` genomes, differential evolution often needs far fewer evaluations than a GA" ([Differential evolution](../../../AGENTS.md#differential-evolution)), with its defaults.
- **`es`:** "For smooth real-valued problems that need precise answers" ([Evolution strategy](../../../AGENTS.md#evolution-strategy-with-self-adaptation)), with its defaults (intermediate recombination of all parents, comma selection, a step per gene starting at 0.3 of each range) and 15 parents and 100 offspring ([lines 509-521](../../../benchmarks/adapters/genoxide/src/main.rs#L509-L521)).

**Keeping going:** CMA-ES and DE restart; the evolution strategy runs to the budget.

**Left out:**
- **Particle swarm**, whose template ([Particle swarm optimization](../../../AGENTS.md#particle-swarm-optimization)) is this problem: no stated preference for it, and the three above come first.
- **The GA:** AGENTS.md prefers DE to a GA on continuous problems.

**Separate tests:**

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 5,837 | 0.009198 | 0.008564 | 0.009741 | 0 |
| de | 5 | 5 | 38,077 | 0.008082 | 0.004516 | 0.008217 | 0 |
| es | 5 | 1 | 489,151 | 0.02353 | 0.009992 | 0.1455 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

The matched settings of the [README](../../../benchmarks/README.md#scenarios), with genoxide's own operators.

**Methods** ([lines 609-683](../../../benchmarks/adapters/genoxide/src/main.rs#L609-L683)):
- **`nsga2`, `spea2`, `sms_emoa`:** 100 individuals (92 with 3 objectives), `SimulatedBinaryCrossover::new(15.0)` at the default 0.9, `PolynomialMutation::per_gene(1 / n, 20.0)`. SMS-EMOA breeds one child per generation (`.offspring(1)`; the default is a population's worth).
- **`nsga3`:** Das-Dennis directions, 99 divisions (100) or 12 (91), population 100 or 92 (the scenario's size; the default would be the 91 directions), SBX η 30 at the default 1, the same mutation.
- **`moead`:** 100 weight vectors (99 divisions) or 91 (12), its defaults of 20 neighbors, neighborhood parents at 0.9 and at most 2 replacements, Tchebycheff with 2 objectives and PBI with θ 5 with 3, SBX η 20 at the default 1, the same mutation.
- **No duplicate elimination:** `.eliminate_duplicates(false)` for NSGA-II, NSGA-III, SPEA2 and SMS-EMOA, which drop duplicates by default; MOEA/D has none.

**Keeping going:** no convergence criterion; every run uses its budget.

**The front:** `outcome.front()`, the non-dominated individuals of the final population (SPEA2: of its archive of 100 or 92).

**Separate tests** (hypervolume as `run.py` computes it):

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

1 NSGA-II run and 2 SMS-EMOA runs end without the front's last segment (f1 above 0.65), at a hypervolume of about 1.245.

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

Nothing. genoxide has no multi-objective algorithm beyond the five of rule 6.1.

## Bugs found

None.
