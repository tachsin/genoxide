# openGA (C++, 1.0.5+f9b15e7)

A header-only C++ genetic algorithm library, [openGA.hpp](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp), with a single-objective GA (`SOGA`), an interactive GA (`IGA`) and NSGA-III. It ships no operators: its users write the initialization, the evaluation, the crossover (one child per call) and the mutation, and set the population and rates. Its docs are the [user manual](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/openGA.pdf), the [README](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/README.md), the [examples](https://github.com/Arash-codedev/openGA/tree/f9b15e70600e20491504391dec6de5c64eb18913/examples) and the code generator openGA assist ([assist/main.js](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js)), which the README and the manual (p. 6) recommend for starting a program. The benchmark uses the header at commit f9b15e7 (2026-03-22), 16 commits after the last release, v1.0.5 (2020). Header lines below are of that commit.

Adapter: [benchmarks/adapters/openga/](../../../benchmarks/adapters/openga/).
Know a better way to solve one of these problems with openGA? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How every run works

- **openGA's loop**, which the adapter can't change ([bench.cpp#L16](../../../benchmarks/adapters/openga/bench.cpp#L16)): every generation keeps the whole population ([`transfer`, L582-599](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L582-L599)) and adds round(population × `crossover_fraction`) children ([L1669](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1669)). Each child comes from two distinct parents drawn by a rank roulette, with chance 1/√(rank + 1) ([L1132-1147](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1132-L1147), [L1586-1594](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1586-L1594)), the user's crossover and, with probability `mutation_rate`, the user's mutation ([L1613-1628](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1613-L1628)). Single-objective survival keeps the `elite_count` best of parents and children, and fills the rest with a rank roulette ([L1021-1061](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1021-L1061)), which can't pick a child (see "Bugs found").
- **Evaluations** are counted in `eval_solution`, every call, which also keeps the genes of the best cost over the whole run ([bench.cpp#L140](../../../benchmarks/adapters/openga/bench.cpp#L140)). openGA evaluates each child once and never evaluates a parent again, also when a child is a copy of its parent.
- **The end of a run:** the adapter calls openGA's `solve_init` and then `solve_next_generation`, the two halves of `solve()` ([L402-503](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L402-L503)), and stops after the generation in which the target is reached or the budget or the time is used up ([`run_ga`, bench.cpp#L369](../../../benchmarks/adapters/openga/bench.cpp#L369)). The clock starts before the first GA and its initial population are created. A run may go past its budget by one generation, or by one initial population when a new attempt starts; the adapter reports that size as `last_generation`.
- **Keeping going (rule 2.2):** openGA's stop criteria are in [`stop_critera`, L1705-1740](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1705-L1740).
  - `generation_max` is only a budget, so it's lifted (`INT_MAX`, [bench.cpp#L323](../../../benchmarks/adapters/openga/bench.cpp#L323)). This doesn't change the mutation's schedule: the `shrink_scale` passed to `mutate` is `default_shrink_scale(generation)` ([L531-539](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L531-L539)), 1 up to generation 5 and 1/√(generation − 4) after, squared with probability 0.4 and reset to 1 with probability 0.06: a function of the generation number only.
  - The best and average stalls detect convergence: `best_stall_max` generations in a row whose best cost changes by less than `tol_stall_best`, or `average_stall_max` generations whose average cost changes by less than `tol_stall_average`. They're set as each method's source sets them (below). When one fires, `solve_next_generation` returns its `StopReason`, and that attempt ends. openGA has no restart mechanism, so the adapter starts the method again from a new random population, a new GA seeded with `seed × 1000 + attempt`; the best solution is kept over all attempts, and every evaluation counts. The runs report the number of `restarts`.
  - Nothing else in openGA ends a run: `user_request_stop` is never set.
- **Bounds (rule 2.4):** the initial genes are drawn inside the bounds; the examples' mutation draws the whole child again while a gene is out of range; their crossover mixes the two parents per gene, which stays between them; SBX and polynomial mutation clip to [0, 1]. The adapter counts the solutions evaluated outside the bounds, as openGA proposed them ([bench.cpp#L666](../../../benchmarks/adapters/openga/bench.cpp#L666)); every test run reports 0.
- **One thread:** `multi_threading = false`, so the population is created and evaluated sequentially ([L1557-1561](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1557-L1561), [L1678-1682](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1678-L1682)). `run.py check`: CPU 3.4 s over 3.4 s of wall time.
- **Seeds:** openGA seeds its private `std::mt19937_64` from the clock in its constructor ([L371-374](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L371-L374)) and has no setter. All its randomness, and the operators' `rnd01`, comes from that generator, so the adapter seeds it through the standard explicit-instantiation access to a private member ([bench.cpp#L106](../../../benchmarks/adapters/openga/bench.cpp#L106)), with `seed × 1000 + attempt` (attempt 0, 1, ...; no test run needed more than 138 restarts). The same seed gives the same evaluations and best on the same machine and build (checked by `run.py check`).
- **Reproducibility across machines:** `build.sh` compiles with `-O3 -march=native`, so the compiler may use the CPU's fused multiply-add and other instructions, which round differently. On another CPU, or with other flags, the same seed can follow another path: a build with `-O2` and no `-march` already takes another path in Rastrigin 10, seed 0.
- **The output:** `solution` is the best genes evaluated, and `best` its value recomputed from it in the problem's direction ([bench.cpp#L459](../../../benchmarks/adapters/openga/bench.cpp#L459)). The `values` command uses the same fitness functions ([bench.cpp#L899](../../../benchmarks/adapters/openga/bench.cpp#L899)).
- **How the docs decide (rule 6.2):** openGA states no preference between its examples and its generator. For each problem type, the adapter uses openGA's example for that type where there is one (so-rastrigin, for Rastrigin), and the program openGA assist generates, openGA's default starting point, for every type. Neither choice comes from the separate tests.

## Binary: OneMax 100 and 1000

**Methods:**
- Matched (OneMax 100 and 1000, [bench.cpp#L500](../../../benchmarks/adapters/openga/bench.cpp#L500)): population 300, 300 children per generation (`crossover_fraction` 1), each from DEAP's two-point crossover with probability 0.5 (else a copy of the first parent), then with probability 0.2 (`mutation_rate`) a bit flip at 1/n per bit. `elite_count` = population. Differences from DEAP's `eaSimple`:
  - parents come from openGA's rank roulette, two distinct ones per child, not from a tournament of 3;
  - the crossover gives one child per call;
  - openGA evaluates every child, also the unchanged copies;
  - survival is the best 300 of parents and children. "No elitism" can't be expressed: with fewer elite slots, a child can only enter the next population through one (see "Bugs found"). This is the one workaround of the bug, listed in the [notes](../notes.md).
- Idiomatic (OneMax 100, [bench.cpp#L532](../../../benchmarks/adapters/openga/bench.cpp#L532)): openGA has no binary example, so the settings openGA assist generates, with its default choices: population 200 ("medium", [main.js#L435-436](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L435-L436), selected by default in [index.html#L24](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/index.html#L24)), `crossover_fraction` 0.7, `mutation_rate` 0.2 ([#L460-461](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L460-L461)), `elite_count` 10 ([#L465](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L465)). The generated crossover mixes the parents at random per gene ([#L282-295](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L282-L295)), which for 0/1 genes is uniform crossover. The generated mutation adds a real-valued step to each gene ([#L259-279](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L259-L279)), which doesn't apply to bits, and the manual says to edit the operators to the genes' type (p. 6); the adapter uses a bit flip at 1/n per bit.

**Keeping going:**
- Matched: `eaSimple` has no convergence criterion, so the matched configuration sets none, and a run is one attempt to the target or the budget.
- Idiomatic: the assist program's `best_stall_max` 10 ([#L464](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L464)), with openGA's other stall defaults ([L346-349](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L346-L349), manual Table 1, p. 4): `tol_stall_best` 1e-6, `average_stall_max` 10, `tol_stall_average` 1e-4 ([bench.cpp#L352](../../../benchmarks/adapters/openga/bench.cpp#L352)). An attempt that stalls restarts (see above). Its `generation_max` 1000 ([#L447](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L447)) is lifted.

**Left out:**
- The examples' settings for a 2-variable problem ([so-1](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-1/example_so1.cpp#L138-L149): population 20, `mutation_rate` 0.4): an example for one real-valued function, not for binary genes.
- openGA has no binary example and no binary operators.

**Separate tests** (2026-09-25, 1.0.5+f9b15e7, seeds 0 to 4, the scenario's budget, 60 s cap):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best (median, best, worst) | Restarts (median) | At the cap |
|---|---|---|---|---|---|---|---|
| onemax-100-matched | ga | 5 | 5 | 12,000 | 100, 100, 100 | 0 | 0 |
| onemax-1000-matched | ga | 5 | 5 | 177,300 | 1000, 1000, 1000 | 0 | 0 |
| onemax-100-idiomatic | ga | 5 | 5 | 27,180 | 100, 100, 100 | 4 | 0 |

In three of the five idiomatic runs, the best stalls for 10 generations before the target, and the target is reached in a later attempt.

## Permutation: N-Queens 32 and 64

**Methods:** openGA has no permutation example, so the assist settings, as for OneMax: population 200, `crossover_fraction` 0.7, `mutation_rate` 0.2, `elite_count` 10 ([bench.cpp#L564](../../../benchmarks/adapters/openga/bench.cpp#L564)). openGA has no permutation operators, so the adapter uses the usual ones: a random permutation, order crossover (OX1, one child) and a swap of two genes.

**Keeping going:** the assist program's stall criteria, as for OneMax idiomatic; an attempt that stalls restarts.

**Left out:** the examples' real-valued operators, which don't keep a permutation.

**Why it doesn't reach the target:** the single-objective survival bug (see "Bugs found"). A roulette slot can take the same individual again, so the population fills with copies: in an attempt without the stall criteria (seed 0 of N-Queens 32), 117 distinct permutations of 200 after one generation, 14 after 20, and 1 from generation 100 on, at 1 conflict. With the collapse, the best and the average stop changing, so the stall criteria end attempts after about 30 generations on average (111 to 138 restarts per run), and each new attempt starts again from random permutations. Few attempts get to 0 conflicts in that time.

**Separate tests** (2026-09-25, 1.0.5+f9b15e7, seeds 0 to 4, the scenario's budget, 60 s cap):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best (median, best, worst) | Restarts (median) | At the cap |
|---|---|---|---|---|---|---|---|
| nqueens-32-idiomatic | ga | 5 | 1 | 186,560 | 1, 0, 1 | 113 | 0 |
| nqueens-64-idiomatic | ga | 5 | 0 | - | 4, 3, 4 | 131 | 0 |

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** ([bench.cpp#L711](../../../benchmarks/adapters/openga/bench.cpp#L711)), both with the operators of openGA's examples ([bench.cpp#L647](../../../benchmarks/adapters/openga/bench.cpp#L647)): genes uniform in the bounds; a crossover that mixes the parents at random per gene, r·a + (1 − r)·b with a new r per gene; a mutation that moves every gene by mu·(rnd01() − rnd01()) and draws the whole child again while a gene is out of bounds ([bench.cpp#L678](../../../benchmarks/adapters/openga/bench.cpp#L678)).
- `ga`: [examples/so-rastrigin](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp), openGA's example for this problem type (Rastrigin, 5 variables) and its only example with an n-dimensional real vector: population 10,000, `elite_count` 10, `crossover_fraction` 0.7, `mutation_rate` 0.1 ([#L145-159](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L145-L159)), and mu = 1.7 · rnd01() · shrink_scale per gene ([#L66](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L66)).
  - **The radius for Ackley is the adapter's reading.** The example's 1.7 is for Rastrigin's width of 10.24. The adapter scales it to the width of the domain, so the step is the same share of the domain: 1.7 × 65.536 / 10.24 = 10.88 for Ackley. openGA doesn't document this; the literal example would keep 1.7.
- `ga_assist`: the program openGA assist generates for real variables, openGA's default starting point: population 200, `crossover_fraction` 0.7, `mutation_rate` 0.2, `elite_count` 10 (as for OneMax), and mu = 0.2 · shrink_scale ([main.js#L265](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L265), "adjustable", used as generated).

**Keeping going:**
- `ga`: the example's stall criteria, 20 generations at 1e-6 for both the best and the average cost ([#L153-156](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L153-L156)); its `generation_max` 1000 ([#L146](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L146)) is lifted. With population 10,000, the budgets allow 70 generations (Rastrigin 10), 142 (Ackley 30) and 285 (Rastrigin 30), so the stalls rarely fire (at most one restart in the tests).
- `ga_assist`: the assist program's stall criteria, as for OneMax idiomatic (17 to 27 restarts per run).

**Left out:**
- The 2-variable examples [so-1](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-1/example_so1.cpp#L138-L149) (population 20, `mutation_rate` 0.4, mu = 0.2 · shrink_scale) and [so-init-solutions](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-init-solutions/example_so-init-solutions.cpp#L161-L180) (Rastrigin with population 10 from given initial solutions): so-rastrigin is the example for an n-dimensional function of this type, and the assist program is the default.
- The CI test [various-so-rastrigin.cpp](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/ci-test-cases/various-so-rastrigin.cpp#L147-L161): the example with population 5,000 and 50 generations, a test, not a recommendation.
- A custom `get_shrink_scale`: the manual allows one (p. 2), but no example sets it.

**Why no run reaches the target:** the same survival bug. In Rastrigin 10, seed 0, the population of 10,000 has 285 distinct individuals after 10 generations and about 210 from then on; at the last generation, 70, its average is 1.006 and its best 0.997: the whole population sits in one local minimum, one coordinate a basin away from the optimum. The final values are near whole numbers (0.997, 1.99, 2.99, ...) for that reason. With the bug fixed, the same settings reach Rastrigin 10 in every run (see "Bugs found").

**Separate tests** (2026-09-25, 1.0.5+f9b15e7, seeds 0 to 4, the scenario's budget, 60 s cap; no solution evaluated outside the bounds):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best (median, best, worst) | Restarts (median) | At the cap |
|---|---|---|---|---|---|---|---|
| rastrigin-10-idiomatic | ga | 5 | 0 | - | 3.980, 0.9973, 3.985 | 0 | 0 |
| rastrigin-10-idiomatic | ga_assist | 5 | 0 | - | 2.985, 1.990, 3.980 | 26 | 0 |
| rastrigin-30-idiomatic | ga | 5 | 0 | - | 27.86, 25.54, 35.83 | 1 | 0 |
| rastrigin-30-idiomatic | ga_assist | 5 | 0 | - | 15.92, 13.93, 23.88 | 25 | 0 |
| ackley-30-idiomatic | ga | 5 | 0 | - | 2.965, 2.501, 5.539 | 0 | 0 |
| ackley-30-idiomatic | ga_assist | 5 | 0 | - | 6.883, 6.605, 7.587 | 18 | 0 |

## Continuous, unimodal: Rosenbrock 10

**Methods:** `ga` and `ga_assist` as for the multimodal problems ([bench.cpp#L711](../../../benchmarks/adapters/openga/bench.cpp#L711)). openGA has no example for a unimodal function, so by rule 6.2 its default, the assist program, is the method for this type; `ga`, from its only n-dimensional real-valued example, also runs, so the two continuous types have the same solvers. For `ga`, the radius is scaled to Rosenbrock's width, 1.7 × 15 / 10.24 = 2.49: the adapter's reading, as for Ackley.

**Keeping going:** as for the multimodal problems.

**Left out:** as for the multimodal problems.

**Separate tests** (2026-09-25, 1.0.5+f9b15e7, seeds 0 to 4, the scenario's budget, 60 s cap; no solution evaluated outside the bounds):

| Scenario | Solver | Runs | Reached | Median evaluations to target | Best (median, best, worst) | Restarts (median) | At the cap |
|---|---|---|---|---|---|---|---|
| rosenbrock-10-idiomatic | ga | 5 | 0 | - | 7.320, 6.340, 9.604 | 0 | 0 |
| rosenbrock-10-idiomatic | ga_assist | 5 | 0 | - | 3.228, 0.02461, 9.600 | 1 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1 (matched)

**Methods:** openGA's NSGA-III (`GA_MODE::NSGA_III`, its only multi-objective algorithm), with the matched settings ([bench.cpp#L791](../../../benchmarks/adapters/openga/bench.cpp#L791)): `reference_vector_divisions` 99 with 2 objectives and 12 with 3 (Das-Dennis, 100 and 91 directions), population 100 and 92, `crossover_fraction` 1 (population children per generation), SBX with η 30 on each variable with probability 0.5 (as DEAP's `cxSimulatedBinaryBounded`), and on every child (`mutation_rate` 1) polynomial mutation with η 20 at 1/n per variable (as DEAP's `mutPolynomialBounded`). openGA has neither operator, so the adapter implements them ([bench.cpp#L737](../../../benchmarks/adapters/openga/bench.cpp#L737), [#L768](../../../benchmarks/adapters/openga/bench.cpp#L768)); both clip to [0, 1], as DEAP's bounded operators do (rule 2.4). Differences from the matched NSGA-III:
- parents come from openGA's rank roulette on the front index ([L1149-1200](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1149-L1200)), two distinct ones per child, not at random;
- the crossover gives one of the two SBX children, at random.

**The front:** `fronts[0]` of `last_generation`, as openGA's examples save it ([mo-dtlz2.cpp#L116-119](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/mo-dtlz2/mo-dtlz2.cpp#L116-L119)). `solve_next_generation` selects exactly `population` survivors from parents and children ([L477-479](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L477-L479), filled up to `population` in [L805-870](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L805-L870)), ranks them into fronts ([L480](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L480)) and stores them as `last_generation` ([L488-491](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L488-L491)). So the front is the non-dominated part of the final population of 100 or 92, not of parents and children. The adapter prints those solutions and their objectives, recomputed from them ([bench.cpp#L842](../../../benchmarks/adapters/openga/bench.cpp#L842)).

**Keeping going:** NSGA-III has no convergence criterion in openGA (the stall counters are single-objective only, [L1712-1725](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1712-L1725)), so a run is one attempt; `generation_max` is lifted. Each run uses its budget, plus the rest of the last generation.

**Left out:** openGA has no NSGA-II, SPEA2, MOEA/D or SMS-EMOA, and no other multi-objective algorithm (`GA_MODE`, [L42-47](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L42-L47)). The examples' settings for NSGA-III ([mo-dtlz2.cpp#L142-151](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/mo-dtlz2/mo-dtlz2.cpp#L142-L151): population 40, `crossover_fraction` 0.7, `mutation_rate` 0.4, the blend crossover and uniform step mutation): these scenarios are matched.

**Separate tests** (2026-09-25, 1.0.5+f9b15e7, seeds 0 to 4, the scenario's budget, 60 s cap; hypervolume computed with `run.py`'s code; no solution evaluated outside the bounds):

| Scenario | Solver | Runs | Evaluations | Hypervolume (median, best, worst) | Front size | At the cap |
|---|---|---|---|---|---|---|
| zdt1-30-matched | nsga3 | 5 | 25,000 | 0.8578, 0.8642, 0.8545 | 100 | 0 |
| zdt2-30-matched | nsga3 | 5 | 25,000 | 0.5104, 0.5155, 0.4229 | 100 | 0 |
| zdt3-30-matched | nsga3 | 5 | 25,000 | 1.3211, 1.3225, 1.3189 | 100 | 0 |
| dtlz2-3-matched | nsga3 | 5 | 25,024 | 0.6902, 0.6993, 0.6893 | 92 | 0 |
| dtlz1-3-matched | nsga3 | 5 | 40,020 | 1.2897, 1.2907, 1.0340 | 92 | 0 |

The whole final population is non-dominated in every run.

## Can't run

Nothing: the adapter runs all 14 scenarios.

## Bugs found

**The single-objective survival can't keep a child** ([Arash-codedev/openGA#30](https://github.com/Arash-codedev/openGA/issues/30), open). After `transfer` and the crossover, a generation holds the previous population at indices 0 to population − 1 and the children after them. `generate_selection_chance` builds the cumulative rank chances over all of them but divides by the value at index population − 1 ([L1145](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1145)), so every child's value is above 1, and `select_parent` ([L1586-1594](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1586-L1594)), with r in [0, 1), never returns a child. In `select_population_SO` ([L1021-1061](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1021-L1061)), the non-elite slots come from that roulette, so a child survives only among the `elite_count` best of parents and children. The same issue reports a second slip in that loop ([L1057](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1057)): after drawing index j it blocks `sorted_indices[j]` instead of j, so j can be drawn again, and the population fills with copies. NSGA-III has its own survival and isn't affected.

How the results show it: with `elite_count` 10, at most 10 children per generation survive, and the population collapses to copies of a few individuals (N-Queens 32, seed 0: 1 distinct permutation of 200 by generation 100; Rastrigin 10, seed 0: 285 distinct of 10,000 by generation 10). The idiomatic N-Queens attempts converge to 1 or more conflicts, and no continuous run reaches its target.

Worked around only in the matched OneMax runs, as the [notes](../notes.md) say: `elite_count` = population empties the roulette loop, and survival is the best of parents and children. The idiomatic runs keep the library as it is (rule 8.4).

With both fixes of the issue applied to a copy of the header (normalize by the last cumulative value, block j), the same adapter reaches (not benchmark results; 2026-09-25, seeds 0 to 4, the scenario's budget):

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
| rosenbrock-10-idiomatic | ga | 0 → 0 | - | 7.320 → 6.920 |
| rosenbrock-10-idiomatic | ga_assist | 0 → 1 | 122,480 | 3.228 → 0.3406 |

**Fixed at the pinned commit:** [openGA#23](https://github.com/Arash-codedev/openGA/issues/23): from 2023 (commit c82d71b) to 2025, `solve_next_generation` stored `last_generation` only when it had fronts, so single-objective runs kept their first generation. At f9b15e7 it stores every generation ([L484-491](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L484-L491)). The adapter doesn't depend on it for single-objective runs: it keeps the best genes itself.
