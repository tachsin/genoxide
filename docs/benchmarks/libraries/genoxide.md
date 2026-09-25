# genoxide (Rust, 0.6.0)

genoxide is this repository's library: genetic algorithms, evolution strategies, CMA-ES, differential evolution, particle swarms, local search and multi-objective algorithms in Rust. Its documentation is the [README](../../../README.md), the guide [AGENTS.md](../../../AGENTS.md) (decision tables, settings and templates), the [examples](../../../examples/) and the API docs ([docs.rs/genoxide](https://docs.rs/genoxide), from the rustdoc in [src/](../../../src/)). This page applies the same rules to genoxide as to every other library: its methods are the ones those docs recommend, and where a recommendation does badly, the page says so.

Adapter: [benchmarks/adapters/genoxide/](../../../benchmarks/adapters/genoxide/).
Know a better way to solve one of these problems with genoxide? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs genoxide

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), in Rust, a closure per genome, as every template in AGENTS.md writes them ([main.rs, lines 40-162](../../../benchmarks/adapters/genoxide/src/main.rs#L40-L162)). The multi-objective ones are the adapter's own too, not genoxide's `multi::problems`; a unit test checks that the two agree.
- **Evaluations:** the adapter counts every call of the fitness function itself, with a counter around it ([`solve`, lines 247-283](../../../benchmarks/adapters/genoxide/src/main.rs#L247-L283), [`solve_front`, lines 440-481](../../../benchmarks/adapters/genoxide/src/main.rs#L440-L481)), and reports that count. It also compares it with genoxide's own `Outcome::evaluations()` and prints any difference to stderr. In every separate test below, on every scenario, seed and solver, the two counts were equal.
- **A child identical to a parent** inherits the parent's fitness, without a call of the fitness function: genoxide's documented behavior (AGENTS.md, [Fitness functions](../../../AGENTS.md#fitness-functions)). That evaluation is saved, and isn't counted.
- **Stop:** the target, the evaluation budget or the time cap, checked by genoxide after every generation (`Stop::target(..).or(Stop::evaluations(..)).or(Stop::time(..))`). A run can go past the budget by at most one generation.
- **Time:** from before the algorithm is built (which creates its random initial population) to the end of the run.
- **One thread:** genoxide is built without its `parallel` feature, and the engines evaluate sequentially.
- **Seeds:** each seed goes to the algorithm's `.seed(...)`; a seed repeats a run exactly.
- **Solutions:** each run prints its best genome (`outcome.best_genome()`), and a multi-objective run the non-dominated individuals of its final population (`outcome.front()`) with their genomes.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched (OneMax 100 and 1000):** the matched GA of the [README](../../../benchmarks/README.md): population 300, tournament 3, two-point crossover with probability 0.5, bit-flip mutation at 1 / n per gene on 20% of the children, no elitism ([lines 288-305](../../../benchmarks/adapters/genoxide/src/main.rs#L288-L305)). A child that comes out identical to a parent isn't evaluated.
- **Idiomatic (OneMax 100):** the GA of every OneMax example in the docs: [examples/one_max.rs](../../../examples/one_max.rs) (lines 15-21), the README's "A first look", AGENTS.md's [first program](../../../AGENTS.md#the-shape-of-every-program) and [python/README.md](../../../python/README.md): population 100, tournament 3, uniform crossover, `BitFlip::per_gene(1 / n)`, with the default rates (crossover 0.9, mutation 1) and the default scheme (generational, elitism 1) ([lines 306-319](../../../benchmarks/adapters/genoxide/src/main.rs#L306-L319)).

**Keeping going:** a GA runs to the budget by itself.

**Left out:**
- Two-point crossover in the idiomatic run, which the adapter used before this review: the docs use it only in the knapsack template (AGENTS.md, [Constraints](../../../AGENTS.md#constraints-with-a-hall-of-fame)), with `BitFlip::count(1)` and 30 individuals, not for OneMax. With it, the idiomatic run needed a median of 3,230 evaluations (5 seeds), against 2,182 with uniform crossover.
- Local search: the docs present it for permutations, not for binary genomes.

**Separate tests** (2026-09-25, genoxide 0.6.0 at ad4eaf5, seeds 0 to 4, the scenario's budget, 60 s cap):

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

**Methods:**
- **`ga`:** [examples/n_queens.rs](../../../examples/n_queens.rs) (lines 29-37), also AGENTS.md's [permutation template](../../../AGENTS.md#permutations) and its scheme table ("(μ+λ): mutation-only search (with `NoCrossover`)"): a (20 + 20) GA with tournament 2, no crossover and `SwapMutation` ([lines 324-338](../../../benchmarks/adapters/genoxide/src/main.rs#L324-L338)).
- **`local_search`:** hill climbing, the N-Queens example of `LocalSearch` in the rustdoc ([src/algorithm/local_search.rs](../../../src/algorithm/local_search.rs), lines 110-115): swap neighbors, the best of 4 per step, and the default acceptance `NotWorse`, which moves to equal neighbors across plateaus. AGENTS.md ([Local search](../../../AGENTS.md#local-search-hill-climbing-and-simulated-annealing)): it "often beats a GA on permutations" ([lines 340-352](../../../benchmarks/adapters/genoxide/src/main.rs#L340-L352)).
- **`tabu_search`:** tabu search as [python/examples/n_queens.py](../../../python/examples/n_queens.py) (lines 17-24) runs N-Queens 64: swap neighbors, 32 per step, tenure 20 ([lines 354-365](../../../benchmarks/adapters/genoxide/src/main.rs#L354-L365)). The same settings run N-Queens 32: the docs don't give others.

**Keeping going:** all three run to the budget by themselves. Local search evaluates at least one new neighbor every step (genoxide redraws a neighbor that didn't change), so it can't stall.

**Left out:**
- One neighbor per step (the default of `neighbors`), which the adapter used before: the rustdoc's N-Queens example uses 4, the setting documented for this problem. With 1 neighbor, local search needed a median of 1,145 (N-Queens 32) and 2,385 (N-Queens 64) evaluations, against 925 and 4,069 with 4 (5 seeds each).
- Permutation crossovers (`OrderCrossover`, `PartiallyMappedCrossover`, ...) in the GA: the N-Queens example and the permutation template use no crossover.
- Simulated annealing: its temperature needs setting to the fitness differences of the problem (AGENTS.md, Troubleshooting), and no example gives one for N-Queens.

**Separate tests** (2026-09-25, genoxide 0.6.0 at ad4eaf5, seeds 0 to 4):

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

Tabu search with the Python example's settings needs 40 times the evaluations of hill climbing on N-Queens 64, and varies widely: from 18,625 to 919,713 evaluations in seeds 0 to 4, and up to 947,297 in seeds 5 to 9 (all reached the target within the budget of 1,000,000). Its tabu list holds whole solutions, and it always moves to the best of the 32 neighbors that aren't tabu, even a worse one.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:**
- **`ga`:** the GA of [examples/rastrigin.rs](../../../examples/rastrigin.rs) (lines 27-36): population 100, tournament 3, uniform crossover, polynomial mutation with η 20 at 1 / n per gene ("the usual rate"), elitism 2 ([lines 414-428](../../../benchmarks/adapters/genoxide/src/main.rs#L414-L428)).
- **`de`:** differential evolution with its defaults, as AGENTS.md's [DE template](../../../AGENTS.md#differential-evolution) runs Rastrigin: DE/current-to-pbest/1 with an archive, SHADE's adaptation of F and CR, the number of genes + 10 individuals, and restarts when the population converges or stalls ([line 397](../../../benchmarks/adapters/genoxide/src/main.rs#L397)).
- **`cma_es`:** CMA-ES with its defaults (AGENTS.md: "nothing needs tuning": 4 + ⌊3 ln n⌋ samples, an initial step of 0.3 of each range) and IPOP restarts, which AGENTS.md's [CMA-ES template](../../../AGENTS.md#cma-es) (on Rastrigin), [python/README.md](../../../python/README.md) and the rustdoc of `Restarts::Ipop` ("suits multimodal functions with a global structure, like Rastrigin") recommend for multimodal functions ([lines 381-393](../../../benchmarks/adapters/genoxide/src/main.rs#L381-L393)).

**Keeping going:** the GA runs to the budget by itself. DE restarts by itself when its population converges or stalls (its default), and CMA-ES with IPOP restarts with a doubled population when a run meets one of its stop criteria.

**Left out** (5 seeds each, with the scenario's budget):
- **L-SHADE** (`De::l_shade(real, budget)`), which [python/examples/rastrigin.py](../../../python/examples/rastrigin.py) runs on Rastrigin 30: AGENTS.md says it "aims at the best final value rather than the fewest evaluations to a target". It reached every target, after a median of 48,003 (Rastrigin 10), 405,783 (Rastrigin 30) and 78,847 (Ackley 30) evaluations, against 4,400, 27,960 and 8,040 for the default DE.
- **BIPOP restarts** (`cmaes::Restarts::Bipop`, "good on a wider range of multimodal functions than IPOP"): the templates use IPOP. BIPOP needed a median of 138,893 (Rastrigin 10) and 868,670 (Rastrigin 30) evaluations, and the same as IPOP on Ackley 30 (3,164).
- **DE with a fixed small CR** (`de::Control::Fixed { f: 0.5, cr: 0.1 }`), which AGENTS.md suggests for separable functions: it's a setting for a known property of the function, which the benchmark doesn't give the libraries. It needed 4,820, 46,800 and 14,800 evaluations: more than the defaults.
- **The island model** ([AGENTS.md](../../../AGENTS.md#island-model), 4 islands of 25 on Rastrigin): a way to run the GA, not another method, and the Python package doesn't have it. With the template's settings it needed 17,565 evaluations on Rastrigin 10, reached Rastrigin 30 in 4 of 5 runs (median 1,772,255) and Ackley 30 in none (median best 0.084).
- **Particle swarm:** AGENTS.md: "On separable functions like Rastrigin, differential evolution with a small CR does much better." With a ring topology, the one it suggests for multimodal functions, it reached none of the Rastrigin targets and Ackley 30 in 32,400 evaluations.
- **The evolution strategy** (AGENTS.md's (5/5, 35)-ES): "For hard problems (rotated, badly conditioned or multimodal), CMA-ES is stronger." It reached no target (Rastrigin 10 median best 7.96), and the Python package doesn't have it.

**Separate tests** (2026-09-25, genoxide 0.6.0 at ad4eaf5, seeds 0 to 4):

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 23,530 | 0.008099 | 0.003897 | 0.009707 | 0 |
| de | 5 | 5 | 4,400 | 0.008197 | 0.003399 | 0.009838 | 0 |
| cma_es | 5 | 5 | 80,200 | 0.009076 | 0.004135 | 0.009887 | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 101,527 | 0.009632 | 0.009398 | 0.00982 | 0 |
| de | 5 | 5 | 27,960 | 0.009308 | 0.005445 | 0.00955 | 0 |
| cma_es | 5 | 5 | 525,238 | 0.009529 | 0.007666 | 0.009915 | 0 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| ga | 5 | 5 | 245,513 | 0.009644 | 0.008412 | 0.009885 | 0 |
| de | 5 | 5 | 8,040 | 0.009627 | 0.009334 | 0.00982 | 0 |
| cma_es | 5 | 5 | 3,164 | 0.008987 | 0.008779 | 0.009896 | 0 |

**Where the DE defaults come from:** the rustdoc of `De::builder` ([src/algorithm/de.rs](../../../src/algorithm/de.rs), lines 194-197) says they are "the settings that reached targets in the fewest evaluations in genoxide's measurements (shifted Rastrigin, Rosenbrock and Ackley with 10 and 30 genes)": the problems of this benchmark. They're the documented defaults, which the rules allow, but they were chosen on these problems, which rule 6.3 doesn't allow for a setting chosen by an adapter. Readers should weigh the DE results with that in mind.

## Continuous, unimodal: Rosenbrock 10

**Methods:**
- **`cma_es`:** AGENTS.md's first choice for this kind of function, "the strongest general choice for continuous problems ... especially when the genes interact (rotated or badly conditioned functions)" ([CMA-ES](../../../AGENTS.md#cma-es)), with its defaults and IPOP restarts ([lines 381-393](../../../benchmarks/adapters/genoxide/src/main.rs#L381-L393)). The restarts are for rule 2.2: without them, a CMA-ES whose run has converged goes on sampling around the same point (the rustdoc of `Restarts::Never`), and AGENTS.md's troubleshooting table says to add them when a CMA-ES stops improving. On Rosenbrock 10 no run converged before the target: the runs are the same with and without restarts.
- **`de`:** differential evolution with its defaults, "for continuous problems on `Real` genomes, differential evolution often needs far fewer evaluations than a GA" ([DE](../../../AGENTS.md#differential-evolution); [line 397](../../../benchmarks/adapters/genoxide/src/main.rs#L397)). See above for where its defaults come from.
- **`pso`:** AGENTS.md's [PSO template](../../../AGENTS.md#particle-swarm-optimization), which is Rosenbrock with the target 0.01: 40 particles ("20 to 50"), the default global topology and Clerc and Kennedy's constriction coefficients ([lines 402-411](../../../benchmarks/adapters/genoxide/src/main.rs#L402-L411)).

**Keeping going:** CMA-ES and DE restart by themselves (above); the particle swarm runs to the budget by itself.

**Left out** (5 seeds each, budget 500,000):
- **The GA of examples/rastrigin.rs**, which the adapter ran on Rosenbrock before this review, and which the v0.6.0 results show failing: the docs present it for a multimodal function, and name CMA-ES, DE, the ES and PSO for continuous problems like this one. It reached the target in none of 5 runs; its best values had a median of 1.14 (best 0.58, worst 5.17) after the whole budget. The docs recommend it for real-valued problems in general ([Choosing the pieces](../../../AGENTS.md#choosing-the-pieces)), so this is a result against genoxide's GA.
- **The evolution strategy** (AGENTS.md's (5/5, 35)-ES, "for smooth real-valued problems that need precise answers"; "for hard problems (rotated, badly conditioned or multimodal), CMA-ES is stronger"): it reached the target in 1 of 5 runs (median best 0.0168), and the Python package doesn't have it.

**Separate tests** (2026-09-25, genoxide 0.6.0 at ad4eaf5, seeds 0 to 4):

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | At the cap |
|---|---|---|---|---|---|---|---|
| cma_es | 5 | 5 | 5,840 | 0.009198 | 0.008564 | 0.009741 | 0 |
| de | 5 | 5 | 7,360 | 0.009191 | 0.008339 | 0.009786 | 0 |
| pso | 5 | 5 | 98,160 | 0.009987 | 0.009948 | 0.009991 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

These scenarios are matched: every library runs the settings of the [README](../../../benchmarks/README.md#scenarios).

**Methods** ([lines 483-552](../../../benchmarks/adapters/genoxide/src/main.rs#L483-L552)):
- **`nsga2`, `spea2`, `sms_emoa`:** 100 individuals (92 with 3 objectives), `SimulatedBinaryCrossover::new(15.0)` at their default rate of 0.9, `PolynomialMutation::per_gene(1 / n, 20.0)`. SMS-EMOA breeds as many children per generation as its population (genoxide's default).
- **`nsga3`:** with 3 objectives only, as AGENTS.md presents it ("for 3 or more objectives"): Das-Dennis directions with 12 divisions (91), a population of 92 (genoxide's default would be the 91 directions; 92 is the size the scenario gives the other algorithms), SBX with η 30 at its default rate of 1, and the same polynomial mutation.
- NSGA-II, NSGA-III, SPEA2 and SMS-EMOA drop a child that equals a member of the population or an earlier child, and breed another: genoxide's default, as pymoo's ([python/README.md](../../../python/README.md#algorithms)).
- **`moead`:** 100 weight vectors (99 divisions; 91 with 3 objectives and 12 divisions), its defaults of 20 neighbors, parents from the neighborhood with probability 0.9 and at most 2 replacements, Tchebycheff with 2 objectives and PBI with θ 5 with 3 (AGENTS.md: "PBI for 3 or more objectives"), SBX with η 20 at its default rate of 1, and the same polynomial mutation.

**Keeping going:** every run uses its whole budget, and ends after the generation that reaches it.

**The front:** the non-dominated individuals of the final population, `outcome.front()`: for SPEA2, of its archive of 100 (92). MOEA/D's population can hold the same solution for several weight vectors, and genoxide's front keeps those copies (e.g. 83 points, 75 distinct, in ZDT3 seed 0); they don't change the hypervolume.

**Left out:** NSGA-III on the 2-objective ZDT problems, which some libraries run with 99 divisions: genoxide's docs present it for 3 or more objectives.

**Separate tests** (2026-09-25, genoxide 0.6.0 at ad4eaf5, seeds 0 to 4; the hypervolume as `run.py` computes it):

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

genoxide runs every scenario.

## Bugs found

None in genoxide in this review. The adapter's own evaluation count matched genoxide's `Outcome::evaluations()` in every run, and its fitness functions agree with genoxide's `multi::problems` (ZDT1 to 3, DTLZ1 and 2).

Observations for the library, not bugs:
- The GA with the documented real-valued operators doesn't reach Rosenbrock 10 (above).
- The tabu search settings of python/examples/n_queens.py need 40 times the evaluations of hill climbing on N-Queens 64 (above).
- The DE defaults were chosen on this benchmark's problems (above).
