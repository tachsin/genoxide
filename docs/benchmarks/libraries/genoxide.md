# genoxide (Rust, 0.6.0)

genoxide is this repository's library: genetic algorithms, evolution strategies, CMA-ES, differential evolution, particle swarms, local search and multi-objective algorithms in Rust. Its documentation is the [README](../../../README.md), the guide [AGENTS.md](../../../AGENTS.md) (decision tables, settings and templates), the [examples](../../../examples/) and the API docs ([docs.rs/genoxide](https://docs.rs/genoxide), from the rustdoc in [src/](../../../src/)). This page applies the same rules to genoxide as to every other library: its methods are the ones those docs recommend, and where a recommendation does badly, the page says so.

Adapter: [benchmarks/adapters/genoxide/](../../../benchmarks/adapters/genoxide/).
Know a better way to solve one of these problems with genoxide? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs genoxide

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), in Rust, a closure per genome, as every template in AGENTS.md writes them ([main.rs, lines 48-169](../../../benchmarks/adapters/genoxide/src/main.rs#L48-L169)). The multi-objective ones are the adapter's own too, not genoxide's `multi::problems`; a unit test checks that the two agree.
- **Evaluations:** the adapter counts every call of the fitness function itself, with a counter around it ([`solve`, lines 245-344](../../../benchmarks/adapters/genoxide/src/main.rs#L245-L344), [`solve_front`, lines 518-571](../../../benchmarks/adapters/genoxide/src/main.rs#L518-L571)), and reports that count. It also compares it with genoxide's own `Outcome::evaluations()` and prints any difference to stderr. In every separate test below, on every scenario, seed and solver, the two counts were equal.
- **A child identical to a parent** inherits the parent's fitness, without a call of the fitness function: genoxide's documented behavior (AGENTS.md, [Fitness functions](../../../AGENTS.md#fitness-functions)). That evaluation is saved, and isn't counted.
- **Stop:** the target, the evaluation budget or the time cap, checked by genoxide after every generation (`Stop::target(..).or(Stop::evaluations(..)).or(Stop::time(..))`). A run can go past the budget by at most one generation. CMA-ES with IPOP restarts also prints `last_generation`, the evaluations of its last generation, which grows at every restart, for the budget check.
- **Restarts on convergence (rule 2.2):** DE and CMA-ES (IPOP) restart by themselves when a run converges (their own criteria, inside one genoxide run). The GA, the island model, the particle swarm and local search have no convergence criterion: they run until a stop condition. genoxide's engine ends a run by itself only with `StopReason::Stalled`, after 10,000 generations in a row without a genome to evaluate (e.g. every child a copy of its parent; AGENTS.md, [Troubleshooting](../../../AGENTS.md#troubleshooting)). The adapter then starts the method again from a new random start with the seed `seed * 1000 + restart`, keeps the best solution and counts every evaluation, until the target, the budget or the time cap ([`solve`](../../../benchmarks/adapters/genoxide/src/main.rs#L245-L344)); a run that restarted prints `restarts`. No test run stalled.
- **Bounds (rule 2.4):** genoxide's own bound handling keeps every evaluated solution inside the box: uniform crossover exchanges genes between parents; SBX and polynomial mutation are the bounded versions (their rustdoc); CMA-ES draws a sample outside the bounds again, up to 100 times, and then clips it (the rustdoc of `Cmaes`); DE sets a trial gene outside the bounds halfway between the parent's gene and the bound ("bounce-back", the rustdoc of `De`); a particle that would leave the bounds stops at the bound (the rustdoc of `Pso`). The adapter's fitness wrapper counts the evaluated solutions outside the bounds, as genoxide proposed them, and prints `outside`: 0 in every run of the separate tests.
- **Choosing among the docs' options (rule 6.2):** where genoxide's docs offer several methods or settings for a problem type, a preference they state decides, then their example for that problem type, then the default. The separate tests below are shown, and decided nothing.
- **Time:** from before the algorithm is built (which creates its random initial population) to the end of the run.
- **One thread:** genoxide is built without its `parallel` feature, and the engines evaluate sequentially.
- **Seeds:** each seed goes to the algorithm's `.seed(...)`; a seed repeats a run exactly.
- **Solutions:** each run prints its best genome (`outcome.best_genome()`), and a multi-objective run the non-dominated individuals of its final population (`outcome.front()`) with their genomes.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched (OneMax 100 and 1000):** the matched GA of the [README](../../../benchmarks/README.md): population 300, tournament 3, two-point crossover with probability 0.5, bit-flip mutation at 1 / n per gene on 20% of the children, no elitism ([lines 353-369](../../../benchmarks/adapters/genoxide/src/main.rs#L353-L369)). A child that comes out identical to a parent isn't evaluated.
- **Idiomatic (OneMax 100):** the GA of every OneMax example in the docs: [examples/one_max.rs](../../../examples/one_max.rs) (lines 15-21), the README's "A first look", AGENTS.md's [first program](../../../AGENTS.md#the-shape-of-every-program) and [python/README.md](../../../python/README.md): population 100, tournament 3, uniform crossover, `BitFlip::per_gene(1 / n)`, with the default rates (crossover 0.9, mutation 1) and the default scheme (generational, elitism 1) ([lines 371-384](../../../benchmarks/adapters/genoxide/src/main.rs#L371-L384)).

**Keeping going:** a GA runs to the budget by itself.

**Left out:**
- Two-point crossover in the idiomatic run, which the adapter used before this review: the docs' example for OneMax uses uniform crossover, and they use two-point only in the knapsack template (AGENTS.md, [Constraints](../../../AGENTS.md#constraints-with-a-hall-of-fame)), with `BitFlip::count(1)` and 30 individuals. For information: with two-point crossover, the idiomatic run needed a median of 3,230 evaluations (5 seeds).
- Local search: the docs present it for permutations, not for binary genomes.

**Separate tests** (2026-09-25, genoxide 0.6.0 at 4fff05e, seeds 0 to 4, the scenario's budget, 60 s cap):

OneMax 100, matched (budget 200,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 4,823 | 100 | 100 | 100 | 0 |

OneMax 1000, matched (budget 2,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 53,615 | 1,000 | 1,000 | 1,000 | 0 |

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 2,182 | 100 | 100 | 100 | 0 |

## Permutation: N-Queens 32 and 64

**Methods:** the docs have an example for N-Queens for each of the three, and state one preference: AGENTS.md says local search "often beats a GA on permutations".
- **`ga`:** genoxide's example for N-Queens, [examples/n_queens.rs](../../../examples/n_queens.rs) (lines 29-37), also AGENTS.md's [permutation template](../../../AGENTS.md#permutations) and its scheme table ("(μ+λ): mutation-only search (with `NoCrossover`)"): a (20 + 20) GA with tournament 2, no crossover and `SwapMutation` ([lines 389-403](../../../benchmarks/adapters/genoxide/src/main.rs#L389-L403)).
- **`local_search`:** hill climbing, as the rustdoc's example for N-Queens sets it (the example of `LocalSearch` in [src/algorithm/local_search.rs](../../../src/algorithm/local_search.rs), lines 110-115): swap neighbors, the best of 4 per step (the example's setting, rather than the default of 1), and the default acceptance `NotWorse`, which moves to equal neighbors across plateaus. AGENTS.md ([Local search](../../../AGENTS.md#local-search-hill-climbing-and-simulated-annealing)): it "often beats a GA on permutations" ([lines 405-417](../../../benchmarks/adapters/genoxide/src/main.rs#L405-L417)).
- **`tabu_search`:** tabu search, the Python package's example for N-Queens: [python/examples/n_queens.py](../../../python/examples/n_queens.py) (lines 17-24) runs N-Queens 64: swap neighbors, 32 per step, tenure 20 ([lines 419-430](../../../benchmarks/adapters/genoxide/src/main.rs#L419-L430)). The same settings run N-Queens 32: the docs don't give others.

**Keeping going:** none of the three has a convergence criterion: they run to the budget. Local search evaluates at least one new neighbor every step (genoxide redraws a neighbor that didn't change), so it can't stall.

**Left out:**
- One neighbor per step (the default of `neighbors`), which the adapter used before: the docs' example for N-Queens uses 4, and an example for the problem type comes before the default (rule 6.2). For information: with 1 neighbor, local search needed a median of 1,145 (N-Queens 32) and 2,385 (N-Queens 64) evaluations (5 seeds each).
- Permutation crossovers (`OrderCrossover`, `PartiallyMappedCrossover`, ...) in the GA: the N-Queens example and the permutation template use no crossover.
- Simulated annealing: its temperature needs setting to the fitness differences of the problem (AGENTS.md, Troubleshooting), and no example gives one for N-Queens.

**Separate tests** (2026-09-25, genoxide 0.6.0 at 4fff05e, seeds 0 to 4):

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 2,560 | 0 | 0 | 0 | 0 |
| local_search | 5 | 5 | 925 | 0 | 0 | 0 | 0 |
| tabu_search | 5 | 5 | 4,097 | 0 | 0 | 0 | 0 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 5,978 | 0 | 0 | 0 | 0 |
| local_search | 5 | 5 | 4,069 | 0 | 0 | 0 | 0 |
| tabu_search | 5 | 5 | 165,377 | 0 | 0 | 0 | 0 |

The tabu search of the Python example is kept, as the docs' example for this problem, though it needs 40 times the evaluations of hill climbing on N-Queens 64, and varies widely: from 18,625 to 919,713 evaluations in seeds 0 to 4, and up to 947,297 in seeds 5 to 9 (all reached the target within the budget of 1,000,000). Its tabu list holds whole solutions, and it always moves to the best of the 32 neighbors that aren't tabu, even a worse one.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:** AGENTS.md states a preference for each: CMA-ES is "the strongest general choice for continuous problems", with restarts "for multimodal functions"; differential evolution "often needs far fewer evaluations than a GA"; and a GA runs best as an island model: "more diverse than one large population, and often faster on multimodal problems". Particle swarm and the evolution strategy are the ones it advises against here (below).
- **`islands`:** the GA as AGENTS.md's [island model](../../../AGENTS.md#island-model), with its template, which runs Rastrigin: 4 islands of 25, each with its own seed (`4 * seed + island`), tournament 3, uniform crossover, polynomial mutation with η 20 at 0.1 per gene, the default scheme (elitism 1); a ring, migration every 10 generations, 2 migrants ([lines 485-509](../../../benchmarks/adapters/genoxide/src/main.rs#L485-L509)). The template's rate of 0.1 is used as it is, also with 30 genes.
- **`de`:** differential evolution with its defaults, as AGENTS.md's [DE template](../../../AGENTS.md#differential-evolution) runs Rastrigin: DE/current-to-pbest/1 with an archive, SHADE's adaptation of F and CR, the number of genes + 10 individuals, and restarts when the population converges or stalls ([lines 463-467](../../../benchmarks/adapters/genoxide/src/main.rs#L463-L467)).
- **`cma_es`:** CMA-ES with its defaults (AGENTS.md: "nothing needs tuning": 4 + ⌊3 ln n⌋ samples, an initial step of 0.3 of each range) and IPOP restarts, which AGENTS.md's [CMA-ES template](../../../AGENTS.md#cma-es) (on Rastrigin), [python/README.md](../../../python/README.md) and the rustdoc of `Restarts::Ipop` ("suits multimodal functions with a global structure, like Rastrigin") recommend for multimodal functions ([lines 450-462](../../../benchmarks/adapters/genoxide/src/main.rs#L450-L462)). IPOP rather than BIPOP: the rustdoc says IPOP "suits multimodal functions with a global structure, like Rastrigin", and the docs' examples for Rastrigin use it.

**Keeping going:** the island model has no convergence criterion and runs to the budget. DE restarts by itself when its population converges or stalls (its default), and CMA-ES with IPOP restarts with a doubled population when a run meets one of its stop criteria.

**Left out** (5 seeds each, with the scenario's budget):
- **L-SHADE** (`De::l_shade(real, budget)`), which [python/examples/rastrigin.py](../../../python/examples/rastrigin.py) runs on Rastrigin 30: AGENTS.md says it "aims at the best final value rather than the fewest evaluations to a target". It reached every target, after a median of 48,003 (Rastrigin 10), 405,783 (Rastrigin 30) and 78,847 (Ackley 30) evaluations, against 4,400, 27,960 and 8,040 for the default DE.
- **BIPOP restarts** (`cmaes::Restarts::Bipop`, "good on a wider range of multimodal functions than IPOP"): the docs' statement about IPOP names this kind of function ("with a global structure, like Rastrigin"), and their examples for Rastrigin use IPOP. For information: BIPOP needed a median of 138,893 (Rastrigin 10) and 868,670 (Rastrigin 30) evaluations, and the same as IPOP on Ackley 30 (3,164).
- **DE with a fixed small CR** (`de::Control::Fixed { f: 0.5, cr: 0.1 }`), which AGENTS.md suggests for separable functions: it's a setting for a known property of the function, which the benchmark doesn't give the libraries. For information: it needed 4,820, 46,800 and 14,800 evaluations.
- **The GA as one population**, the settings of [examples/rastrigin.rs](../../../examples/rastrigin.rs) (population 100, tournament 3, uniform crossover, polynomial mutation with η 20 at 1 / n, elitism 2), which the adapter ran before the rules' amendment of rule 6.2: AGENTS.md prefers the island model for multimodal problems. For information: it reached every target, after a median of 23,530 (Rastrigin 10), 101,527 (Rastrigin 30) and 245,513 (Ackley 30) evaluations. The Python package, which has no island model, still runs it ([genoxide_python.md](genoxide_python.md)).
- **Particle swarm:** AGENTS.md: "On separable functions like Rastrigin, differential evolution with a small CR does much better." With a ring topology, the one it suggests for multimodal functions, it reached none of the Rastrigin targets and Ackley 30 in 32,400 evaluations.
- **The evolution strategy** (AGENTS.md's (5/5, 35)-ES): "For hard problems (rotated, badly conditioned or multimodal), CMA-ES is stronger." It reached no target (Rastrigin 10 median best 7.96), and the Python package doesn't have it.

**Separate tests** (2026-09-25, genoxide 0.6.0 at 4fff05e, seeds 0 to 4):

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| islands | 5 | 5 | 16,108 | 0.008556 | 0.007568 | 0.009915 | 0 |
| de | 5 | 5 | 4,400 | 0.008197 | 0.003399 | 0.009838 | 0 |
| cma_es | 5 | 5 | 80,200 | 0.009076 | 0.004135 | 0.009887 | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| islands | 5 | 3 | 1,875,790 | 0.00942 | 0.007191 | 0.01347 | 0 |
| de | 5 | 5 | 27,960 | 0.009308 | 0.005445 | 0.00955 | 0 |
| cma_es | 5 | 5 | 525,238 | 0.009529 | 0.007666 | 0.009915 | 0 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| islands | 5 | 0 | - | 0.09492 | 0.06938 | 0.1146 | 0 |
| de | 5 | 5 | 8,040 | 0.009627 | 0.009334 | 0.00982 | 0 |
| cma_es | 5 | 5 | 3,164 | 0.008987 | 0.008779 | 0.009896 | 0 |

The island model with its template's settings does worse than the single GA it's preferred to, except on Rastrigin 10: 2 of 5 runs miss the Rastrigin 30 target within 2,000,000 evaluations, and no run reaches the Ackley 30 target within 1,000,000 (they end between 0.069 and 0.115). This is a result against the docs' stated preference.

**Where the DE defaults come from:** the rustdoc of `De::builder` ([src/algorithm/de.rs](../../../src/algorithm/de.rs), lines 194-197) says they are "the settings that reached targets in the fewest evaluations in genoxide's measurements (shifted Rastrigin, Rosenbrock and Ackley with 10 and 30 genes)": the problems of this benchmark. They're the documented defaults, which the rules allow, but they were chosen on these problems, which rule 6.3 doesn't allow for a setting chosen by an adapter. Readers should weigh the DE results with that in mind.

## Continuous, unimodal: Rosenbrock 10

**Methods:** AGENTS.md states preferences for two, and has an example for this problem for the third.
- **`cma_es`:** AGENTS.md's first choice for this kind of function, "the strongest general choice for continuous problems ... especially when the genes interact (rotated or badly conditioned functions)" ([CMA-ES](../../../AGENTS.md#cma-es)), with its defaults and IPOP restarts ([lines 450-462](../../../benchmarks/adapters/genoxide/src/main.rs#L450-L462)). The restarts are for rule 2.2: a converged run starts again with the library's restart mechanism; without them, a converged CMA-ES goes on sampling around the same point (the rustdoc of `Restarts::Never`), and AGENTS.md's troubleshooting table says to add them when a CMA-ES stops improving. On Rosenbrock 10 no run converged before the target: the runs are the same with and without restarts.
- **`de`:** differential evolution with its defaults, "for continuous problems on `Real` genomes, differential evolution often needs far fewer evaluations than a GA" ([DE](../../../AGENTS.md#differential-evolution); [lines 463-467](../../../benchmarks/adapters/genoxide/src/main.rs#L463-L467)). See above for where its defaults come from.
- **`pso`:** AGENTS.md's example for this problem, its [PSO template](../../../AGENTS.md#particle-swarm-optimization), which is Rosenbrock with the target 0.01: 40 particles ("20 to 50"), the default global topology and Clerc and Kennedy's constriction coefficients ([lines 472-482](../../../benchmarks/adapters/genoxide/src/main.rs#L472-L482)).

**Keeping going:** CMA-ES and DE restart by themselves (above); the particle swarm has no convergence criterion and runs to the budget.

**Left out** (5 seeds each, budget 500,000):
- **The GA of examples/rastrigin.rs**, which the adapter ran on Rosenbrock before this review, and which the v0.6.0 results show failing: the docs present it for a multimodal function, and their preferences and example for a function like this one name CMA-ES, DE and PSO. It reached the target in none of 5 runs; its best values had a median of 1.14 (best 0.58, worst 5.17) after the whole budget. The docs recommend it for real-valued problems in general ([Choosing the pieces](../../../AGENTS.md#choosing-the-pieces)), so this is a result against genoxide's GA.
- **The evolution strategy** (AGENTS.md's (5/5, 35)-ES, "for smooth real-valued problems that need precise answers"; "for hard problems (rotated, badly conditioned or multimodal), CMA-ES is stronger"): it reached the target in 1 of 5 runs (median best 0.0168), and the Python package doesn't have it.

**Separate tests** (2026-09-25, genoxide 0.6.0 at 4fff05e, seeds 0 to 4):

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 5,840 | 0.009198 | 0.008564 | 0.009741 | 0 |
| de | 5 | 5 | 7,360 | 0.009191 | 0.008339 | 0.009786 | 0 |
| pso | 5 | 5 | 98,160 | 0.009987 | 0.009948 | 0.009991 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

These scenarios are matched: every library runs the settings of the [README](../../../benchmarks/README.md#scenarios).

**Methods** ([lines 573-642](../../../benchmarks/adapters/genoxide/src/main.rs#L573-L642)), the five that rule 6.1 runs in the matched multi-objective scenarios:
- **`nsga2`, `spea2`, `sms_emoa`:** 100 individuals (92 with 3 objectives), `SimulatedBinaryCrossover::new(15.0)` at their default rate of 0.9, `PolynomialMutation::per_gene(1 / n, 20.0)`. SMS-EMOA breeds as many children per generation as its population (genoxide's default).
- **`nsga3`:** with 3 objectives only, as AGENTS.md presents it ("for 3 or more objectives"): Das-Dennis directions with 12 divisions (91), a population of 92 (genoxide's default would be the 91 directions; 92 is the size the scenario gives the other algorithms), SBX with η 30 at its default rate of 1, and the same polynomial mutation.
- NSGA-II, NSGA-III, SPEA2 and SMS-EMOA drop a child that equals a member of the population or an earlier child, and breed another: genoxide's default, as pymoo's ([python/README.md](../../../python/README.md#algorithms)).
- **`moead`:** 100 weight vectors (99 divisions; 91 with 3 objectives and 12 divisions), its defaults of 20 neighbors, parents from the neighborhood with probability 0.9 and at most 2 replacements, Tchebycheff with 2 objectives and PBI with θ 5 with 3 (AGENTS.md: "PBI for 3 or more objectives"), SBX with η 20 at its default rate of 1, and the same polynomial mutation.

**Keeping going:** none of the five has a convergence criterion; every run uses its whole budget, and ends after the generation that reaches it. SBX and polynomial mutation keep every gene in [0, 1]: `outside` was 0 in every run.

**The front:** the non-dominated individuals of the final population, `outcome.front()`: for SPEA2, of its archive of 100 (92). MOEA/D's population can hold the same solution for several weight vectors, and genoxide's front keeps those copies (e.g. 83 points, 75 distinct, in ZDT3 seed 0); they don't change the hypervolume.

**Left out:** NSGA-III on the 2-objective ZDT problems, which some libraries run with 99 divisions: genoxide's docs present it for 3 or more objectives.

**Separate tests** (2026-09-25, genoxide 0.6.0 at 4fff05e, seeds 0 to 4; the hypervolume as `run.py` computes it):

ZDT1 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | At the cap |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.8693 | 0.8698 | 0.8690 | 25,000 | 100 | 0 |
| spea2 | 5 | 0.8702 | 0.8706 | 0.8694 | 25,000 | 100 | 0 |
| sms_emoa | 5 | 0.8716 | 0.8717 | 0.8714 | 25,000 | 100 | 0 |
| moead | 5 | 0.8686 | 0.8689 | 0.8683 | 25,020 | 100 | 0 |

ZDT2 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | At the cap |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.5360 | 0.5366 | 0.5355 | 25,000 | 100 | 0 |
| spea2 | 5 | 0.5369 | 0.5371 | 0.5361 | 25,000 | 100 | 0 |
| sms_emoa | 5 | 0.5380 | 0.5382 | 0.5379 | 25,000 | 100 | 0 |
| moead | 5 | 0.5357 | 0.5358 | 0.5345 | 25,083 | 100 | 0 |

ZDT3 (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | At the cap |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 1.3274 | 1.3278 | 1.3271 | 25,000 | 100 | 0 |
| spea2 | 5 | 1.3273 | 1.3277 | 1.3269 | 25,000 | 100 | 0 |
| sms_emoa | 5 | 1.3288 | 1.3289 | 1.3287 | 25,000 | 100 | 0 |
| moead | 5 | 1.3216 | 1.3226 | 1.3208 | 25,032 | 83 | 0 |

DTLZ2, 3 objectives (budget 25,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | At the cap |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 0.6969 | 0.7037 | 0.6912 | 25,024 | 92 | 0 |
| nsga3 | 5 | 0.7442 | 0.7442 | 0.7434 | 25,024 | 92 | 0 |
| spea2 | 5 | 0.7274 | 0.7311 | 0.7230 | 25,024 | 92 | 0 |
| sms_emoa | 5 | 0.7544 | 0.7545 | 0.7543 | 25,024 | 92 | 0 |
| moead | 5 | 0.7441 | 0.7441 | 0.7440 | 25,014 | 91 | 0 |

DTLZ1, 3 objectives (budget 40,000):

| Solver | Runs | Hypervolume: median | best | worst | Evaluations (median) | Front size (median) | At the cap |
|---|---|---|---|---|---|---|---|
| nsga2 | 5 | 1.2987 | 1.3009 | 1.2956 | 40,020 | 92 | 0 |
| nsga3 | 5 | 1.3045 | 1.3046 | 1.3035 | 40,020 | 92 | 0 |
| spea2 | 5 | 1.3032 | 1.3034 | 1.2997 | 40,020 | 92 | 0 |
| sms_emoa | 5 | 1.3046 | 1.3046 | 1.3045 | 40,020 | 92 | 0 |
| moead | 5 | 1.3043 | 1.3044 | 1.3039 | 40,039 | 91 | 0 |

## Can't run

genoxide runs every scenario. It has no multi-objective algorithm beyond the five that rule 6.1 runs.

## Bugs found

None in genoxide in this review. The adapter's own evaluation count matched genoxide's `Outcome::evaluations()` in every run, and its fitness functions agree with genoxide's `multi::problems` (ZDT1 to 3, DTLZ1 and 2).

Observations for the library, not bugs:
- The GA with the documented real-valued operators doesn't reach Rosenbrock 10 (above).
- The tabu search settings of python/examples/n_queens.py need 40 times the evaluations of hill climbing on N-Queens 64 (above).
- The DE defaults were chosen on this benchmark's problems (above).
- AGENTS.md prefers the island model to one population on multimodal problems, but with its template's settings it does worse here than the GA of examples/rastrigin.rs on Rastrigin 30 and Ackley 30 (above).
