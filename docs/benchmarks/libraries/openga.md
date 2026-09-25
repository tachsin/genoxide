# openGA (C++, 1.0.5+f9b15e7)

A header-only C++ genetic algorithm library, [openGA.hpp](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp), with a single-objective GA (`SOGA`), an interactive GA (`IGA`) and NSGA-III. It ships no operators: its users write the initialization, the evaluation, the crossover (one child per call) and the mutation, and set the population and rates. Its docs are the [user manual](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/openGA.pdf), the [README](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/README.md), the [examples](https://github.com/Arash-codedev/openGA/tree/f9b15e70600e20491504391dec6de5c64eb18913/examples) and the code generator openGA assist ([assist/main.js](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js)), which the README and the manual (p. 6) recommend for starting a program. The benchmark uses the header at commit f9b15e7 (2026-03-22), 16 commits after the last release, v1.0.5 (2020). Header lines below are of that commit.

Adapter: [benchmarks/adapters/openga/](../../../benchmarks/adapters/openga/).
Know a better way to solve one of these problems with openGA? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How every run works

- **openGA's loop**, which the adapter can't change ([bench.cpp#L20](../../../benchmarks/adapters/openga/bench.cpp#L20)): every generation keeps the whole population ([`transfer`, L582-599](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L582-L599)) and adds round(population × `crossover_fraction`) children ([L1669](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1669)). Each child comes from two distinct parents drawn by a rank roulette, with chance 1/√(rank + 1) ([L1132-1147](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1132-L1147), [L1586-1594](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1586-L1594)), the user's crossover and, with probability `mutation_rate`, the user's mutation ([L1613-1628](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1613-L1628)). Single-objective survival keeps the `elite_count` best of parents and children, and fills the rest with a rank roulette ([L1021-1061](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1021-L1061)), which can't pick a child (see "Bugs found").
- **Evaluations** are counted in `eval_solution`, every call ([bench.cpp#L147](../../../benchmarks/adapters/openga/bench.cpp#L147)). The counter also records the first evaluation whose value reaches the target, and the clock at that moment (`first_hit`). openGA evaluates each child once and never evaluates a parent again, also when a child is a copy of its parent.
- **The end of a run:** the adapter calls openGA's `solve_init` and then `solve_next_generation`, the two halves of `solve()` ([L402-503](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L402-L503)), and stops after the generation in which an evaluation reaches the target or the budget or the time is used up ([`run_ga`, bench.cpp#L397](../../../benchmarks/adapters/openga/bench.cpp#L397)). A run may go past its budget by one generation, or by one initial population when a new attempt starts; the adapter reports that size as `last_generation`.
- **The clock** starts before the first GA and its initial population are created, and stops when the last attempt ends, before the output.
- **Keeping going (rule 2.2):** openGA's stop criteria are in [`stop_critera`, L1705-1740](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1705-L1740).
  - `generation_max` is only a budget, so it's lifted (`INT_MAX`, [bench.cpp#L341](../../../benchmarks/adapters/openga/bench.cpp#L341)). This doesn't change the mutation's schedule: the `shrink_scale` passed to `mutate` is `default_shrink_scale(generation)` ([L531-539](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L531-L539)), 1 up to generation 5 and 1/√(generation − 4) after, squared with probability 0.4 and reset to 1 with probability 0.06: a function of the generation number only.
  - The best and average stalls detect convergence: `best_stall_max` generations in a row whose best cost changes by less than `tol_stall_best`, or `average_stall_max` generations whose average cost changes by less than `tol_stall_average`. They're set as each method's source sets them (below). When one fires, `solve_next_generation` returns its `StopReason`, and that attempt ends. openGA has no restart mechanism, so the adapter starts the method again from a new random population, a new GA; every evaluation counts. The runs report the number of `restarts`.
  - Nothing else in openGA ends a run: `user_request_stop` is never set.
- **The best solution** of an attempt is the best of its last generation (`best_chromosome_index`): the elite keeps the best of parents and children, so it's the attempt's best. The best of all attempts is reported, with `best` recomputed from it after the clock, in the problem's direction ([bench.cpp#L471](../../../benchmarks/adapters/openga/bench.cpp#L471)). The `values` command uses the same fitness functions ([bench.cpp#L747](../../../benchmarks/adapters/openga/bench.cpp#L747)).
- **Bounds (rule 2.4):** the initial genes are drawn inside the bounds; the examples' mutation draws the whole child again while a gene is out of range; their crossover mixes the two parents per gene, which stays between them. The adapter counts the solutions evaluated outside the bounds, as openGA proposed them ([bench.cpp#L655](../../../benchmarks/adapters/openga/bench.cpp#L655)); every test run reports 0.
- **The shift** of Rastrigin and Ackley (rule 1.4) is computed once, before any run ([bench.cpp#L203](../../../benchmarks/adapters/openga/bench.cpp#L203)).
- **One thread:** `multi_threading = false`, so the population is created and evaluated sequentially ([L1557-1561](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1557-L1561), [L1678-1682](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1678-L1682)).
- **Seeds:** openGA seeds its private `std::mt19937_64` from the clock in its constructor ([L371-374](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L371-L374)) and has no setter. All its randomness, and the operators' `rnd01`, comes from that generator, so the adapter seeds it through the standard explicit-instantiation access to a private member ([bench.cpp#L114](../../../benchmarks/adapters/openga/bench.cpp#L114)). Attempt 0 uses the run's seed, restart r the seed `(seed + 1) × 1,000,000 + r` ([bench.cpp#L136](../../../benchmarks/adapters/openga/bench.cpp#L136)). The same seed gives the same evaluations and best (checked by `run.py check`).
- **The build:** `build.sh` compiles with `g++ -O3` for the default target (rule 4.5).
- **How the docs decide (rule 6.2):** openGA states no preference between its examples and its generator. For each problem type, the adapter uses openGA's example for that type where there is one (so-rastrigin, for multimodal real functions), and the program openGA assist generates, openGA's default starting point, for every type. Neither choice comes from the separate tests.

## Binary: OneMax 100 and 1000

**Methods:**
- Matched (OneMax 100 and 1000): **not run** ([bench.cpp#L499](../../../benchmarks/adapters/openga/bench.cpp#L499)). A matched scenario uses only the library's own components (rule 6.1), and openGA has none of the matched algorithm's: no tournament selection (its parent selection is the rank roulette), no two-point crossover and no bit flip (it ships no operators), and no generational replacement without elitism (its survival keeps the `elite_count` best of parents and children).
- Idiomatic (OneMax 100, [bench.cpp#L525-L548](../../../benchmarks/adapters/openga/bench.cpp#L525-L548)): openGA has no binary example, so the settings openGA assist generates, with its default choices: population 200 ("medium", [main.js#L435-436](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L435-L436), selected by default in [index.html#L24](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/index.html#L24)), `crossover_fraction` 0.7, `mutation_rate` 0.2 ([#L460-461](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L460-L461)), `elite_count` 10 ([#L465](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L465)). The generated crossover mixes the parents at random per gene ([#L282-295](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L282-L295)), which for 0/1 genes is uniform crossover. The generated mutation adds a real-valued step to each gene ([#L259-279](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L259-L279)), which doesn't apply to bits, and the manual says to edit the operators to the genes' type (p. 6); the adapter uses a bit flip at 1/n per bit.

**Keeping going:** the assist program's `best_stall_max` 10 ([#L464](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L464)), with openGA's other stall defaults ([L346-349](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L346-L349), manual Table 1, p. 4): `tol_stall_best` 1e-6, `average_stall_max` 10, `tol_stall_average` 1e-4 ([bench.cpp#L365](../../../benchmarks/adapters/openga/bench.cpp#L365)). An attempt that stalls restarts (see above). Its `generation_max` 1000 ([#L447](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L447)) is lifted.

**Left out:**
- The examples' settings for a 2-variable problem ([so-1](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-1/example_so1.cpp#L138-L149): population 20, `mutation_rate` 0.4): an example for one real-valued function, not for binary genes.
- openGA has no binary example and no binary operators.

**Separate tests** (2026-09-25, 1.0.5+f9b15e7, seeds 0 to 4, the scenario's budget, 60 s cap):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best (median, best, worst) | Restarts (median) | Capped |
|---|---|---|---|---|---|---|---|
| onemax-100-idiomatic | ga | 5 | 5 | 8,950 | 100, 100, 100 | 1 | 0 |

In three of the five runs, the best stalls for 10 generations before the target, and the target is reached in a later attempt.

## Permutation: N-Queens 32 and 64

**Methods:** openGA has no permutation example, so the assist settings, as for OneMax: population 200, `crossover_fraction` 0.7, `mutation_rate` 0.2, `elite_count` 10 ([bench.cpp#L554](../../../benchmarks/adapters/openga/bench.cpp#L554)). openGA has no permutation operators, so the adapter uses the usual ones: a random permutation, order crossover (OX1, one child) and a swap of two genes.

**Keeping going:** the assist program's stall criteria, as for OneMax; an attempt that stalls restarts.

**Left out:** the examples' real-valued operators, which don't keep a permutation.

**Why it doesn't reach the target:** the single-objective survival bug (see "Bugs found"). A roulette slot can take the same individual again, so the population fills with copies: in an attempt without the stall criteria (seed 0 of N-Queens 32), 117 distinct permutations of 200 after one generation, 14 after 20, and 1 from generation 100 on, at 1 conflict. With the collapse, the best and the average stop changing, so the stall criteria end attempts after about 30 generations on average (110 to 140 restarts per run), and each new attempt starts again from random permutations. Few attempts get to 0 conflicts in that time.

**Separate tests** (2026-09-25, 1.0.5+f9b15e7, seeds 0 to 4, the scenario's budget, 60 s cap):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best (median, best, worst) | Restarts (median) | Capped |
|---|---|---|---|---|---|---|---|
| nqueens-32-idiomatic | ga | 5 | 0 | - | 1, 1, 1 | 116 | 0 |
| nqueens-64-idiomatic | ga | 5 | 0 | - | 4, 3, 5 | 136 | 0 |

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** ([bench.cpp#L699](../../../benchmarks/adapters/openga/bench.cpp#L699)), both with the operators of openGA's examples ([bench.cpp#L636](../../../benchmarks/adapters/openga/bench.cpp#L636)): genes uniform in the bounds; a crossover that mixes the parents at random per gene, r·a + (1 − r)·b with a new r per gene; a mutation that moves every gene by mu·(rnd01() − rnd01()) and draws the whole child again while a gene is out of bounds ([bench.cpp#L667](../../../benchmarks/adapters/openga/bench.cpp#L667)).
- `ga`: [examples/so-rastrigin](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp), openGA's example for this problem type (Rastrigin, 5 variables) and its only example with an n-dimensional real vector, as written: population 10,000, `elite_count` 10, `crossover_fraction` 0.7, `mutation_rate` 0.1 ([#L145-159](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L145-L159)), and mu = 1.7 · rnd01() · shrink_scale per gene ([#L66](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L66)), on Ackley too.
- `ga_assist`: the program openGA assist generates for real variables, openGA's default starting point: population 200, `crossover_fraction` 0.7, `mutation_rate` 0.2, `elite_count` 10 (as for OneMax), and mu = 0.2 · shrink_scale ([main.js#L265](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L265), "adjustable", used as generated).

**Keeping going:**
- `ga`: the example's stall criteria, 20 generations at 1e-6 for both the best and the average cost ([#L153-156](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L153-L156)); its `generation_max` 1000 ([#L146](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L146)) is lifted. With population 10,000, the budgets allow 70 generations (Rastrigin 10), 142 (Ackley 30) and 285 (Rastrigin 30), so the stalls rarely fire.
- `ga_assist`: the assist program's stall criteria, as for OneMax.

**Left out:**
- The 2-variable examples [so-1](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-1/example_so1.cpp#L138-L149) (population 20, `mutation_rate` 0.4, mu = 0.2 · shrink_scale) and [so-init-solutions](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-init-solutions/example_so-init-solutions.cpp#L161-L180) (Rastrigin with population 10 from given initial solutions): so-rastrigin is the example for an n-dimensional function of this type, and the assist program is the default.
- The CI test [various-so-rastrigin.cpp](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/ci-test-cases/various-so-rastrigin.cpp#L147-L161): the example with population 5,000 and 50 generations, a test, not a recommendation.
- A custom `get_shrink_scale`: the manual allows one (p. 2), but no example sets it.

**Why no run reaches the target:** the same survival bug. In Rastrigin 10, seed 0 (with the earlier shift in [−1, 1]), the population of 10,000 has 285 distinct individuals after 10 generations and about 210 from then on; at the last generation, 70, its average is 1.006 and its best 0.997: the whole population sits in one local minimum, one coordinate a basin away from the optimum. The final values are near whole numbers (0.997, 1.99, 2.99, ...) for that reason. With the bug fixed, the same settings reached Rastrigin 10 in every run (see "Bugs found").

**Separate tests** (2026-09-25, 1.0.5+f9b15e7, seeds 0 to 4, the scenario's budget, 60 s cap, with the shift of rule 1.4; no solution evaluated outside the bounds):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best (median, best, worst) | Restarts (median) | Capped |
|---|---|---|---|---|---|---|---|
| rastrigin-10-idiomatic | ga | 5 | 0 | - | 4.977, 1.993, 5.970 | 0 | 0 |
| rastrigin-10-idiomatic | ga_assist | 5 | 0 | - | 3.980, 2.985, 5.970 | 25 | 0 |
| rastrigin-30-idiomatic | ga | 5 | 0 | - | 34.84, 32.84, 45.77 | 0 | 4 |
| rastrigin-30-idiomatic | ga_assist | 5 | 0 | - | 46.77, 38.81, 49.75 | 25 | 0 |
| ackley-30-idiomatic | ga | 5 | 0 | - | 8.696, 8.040, 10.86 | 0 | 0 |
| ackley-30-idiomatic | ga_assist | 5 | 0 | - | 13.97, 13.38, 14.12 | 20 | 0 |

The time cap stopped 4 of the 5 `ga` runs of Rastrigin 30, after 1.1 to 1.3 million of the 2 million evaluations. The tests ran on a shared machine, so their times are noisy.

## Continuous, unimodal: Rosenbrock 10

**Methods:** `ga_assist`, as for the multimodal problems ([bench.cpp#L715](../../../benchmarks/adapters/openga/bench.cpp#L715)). openGA has no example for a unimodal function, so by rule 6.2 its default, the assist program, is the method for this type.

**Keeping going:** as for the multimodal problems.

**Left out:**
- `ga`, the so-rastrigin example: it's an example for a multimodal function (a population of 10,000 against Rastrigin's many local minima), not for this type ([bench.cpp#L706](../../../benchmarks/adapters/openga/bench.cpp#L706)).
- The rest as for the multimodal problems.

**Separate tests** (2026-09-25, 1.0.5+f9b15e7, seeds 0 to 4, the scenario's budget, 60 s cap; no solution evaluated outside the bounds):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best (median, best, worst) | Restarts (median) | Capped |
|---|---|---|---|---|---|---|---|
| rosenbrock-10-idiomatic | ga_assist | 5 | 0 | - | 1.308, 0.02888, 3.093 | 1 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Not run.** These scenarios are matched: NSGA-III with the library's own SBX and polynomial mutation (rule 6.1). openGA has NSGA-III (`GA_MODE::NSGA_III`, its only multi-objective algorithm), but ships no operators, so no SBX and no polynomial mutation. The adapter prints nothing for them ([bench.cpp#L811](../../../benchmarks/adapters/openga/bench.cpp#L811)).

openGA has no NSGA-II, SPEA2, MOEA/D or SMS-EMOA, and no other multi-objective algorithm (`GA_MODE`, [L42-47](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L42-L47)).

## Can't run

- Matched OneMax 100 and 1000: openGA has no tournament selection, two-point crossover, bit flip or generational replacement (see Binary).
- The multi-objective scenarios: openGA has no SBX and no polynomial mutation (see Multi-objective).

## Bugs found

**The single-objective survival can't keep a child** ([Arash-codedev/openGA#30](https://github.com/Arash-codedev/openGA/issues/30), open). After `transfer` and the crossover, a generation holds the previous population at indices 0 to population − 1 and the children after them. `generate_selection_chance` builds the cumulative rank chances over all of them but divides by the value at index population − 1 ([L1145](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1145)), so every child's value is above 1, and `select_parent` ([L1586-1594](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1586-L1594)), with r in [0, 1), never returns a child. In `select_population_SO` ([L1021-1061](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1021-L1061)), the non-elite slots come from that roulette, so a child survives only among the `elite_count` best of parents and children. The same issue reports a second slip in that loop ([L1057](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1057)): after drawing index j it blocks `sorted_indices[j]` instead of j, so j can be drawn again, and the population fills with copies. NSGA-III has its own survival and isn't affected.

How the results show it: with `elite_count` 10, at most 10 children per generation survive, and the population collapses to copies of a few individuals (N-Queens 32, seed 0: 1 distinct permutation of 200 by generation 100; Rastrigin 10, seed 0: 285 distinct of 10,000 by generation 10). The idiomatic N-Queens attempts converge to 1 or more conflicts, and no continuous run reaches its target. The runs keep the library as it is (rule 8.4).

With both fixes of the issue applied to a copy of the header (normalize by the last cumulative value, block j), the same adapter reached (not benchmark results; 2026-09-25, seeds 0 to 4, the scenario's budget; with the earlier shift in [−1, 1], the `ga` radius scaled to the width of the domain, and the earlier seeds):

| Scenario | Solver | Reached (as is → fixed) | Median evaluations to target (fixed) | Best, median (as is → fixed) |
|---|---|---|---|---|
| onemax-100-idiomatic | ga | 5 → 5 | 3,280 (27,180 as is) | 100 → 100 |
| nqueens-32-idiomatic | ga | 1 → 3 | 357,740 | 1 → 0 |
| nqueens-64-idiomatic | ga | 0 → 0 | - | 4 → 3 |
| rastrigin-10-idiomatic | ga | 0 → 5 | 178,000 | 3.980 → 0.006180 |
| rastrigin-10-idiomatic | ga_assist | 0 → 2 | 220,770 | 2.985 → 0.9950 |
| rastrigin-30-idiomatic | ga | 0 → 0 | - | 27.86 → 0.9950 |
| rastrigin-30-idiomatic | ga_assist | 0 → 0 | - | 15.92 → 7.960 |
| ackley-30-idiomatic | ga | 0 → 0 | - | 2.965 → 0.01439 |
| ackley-30-idiomatic | ga_assist | 0 → 0 | - | 6.883 → 3.223 |
| rosenbrock-10-idiomatic | ga_assist | 0 → 1 | 122,480 | 3.228 → 0.3406 |

**Fixed at the pinned commit:** [openGA#23](https://github.com/Arash-codedev/openGA/issues/23): from 2023 (commit c82d71b) to 2025, `solve_next_generation` stored `last_generation` only when it had fronts, so single-objective runs kept their first generation. At f9b15e7 it stores every generation ([L484-491](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L484-L491)). The adapter takes each attempt's best from `last_generation`, so it depends on this fix.
