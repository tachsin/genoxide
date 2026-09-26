# openGA (C++, 1.0.5+f9b15e7)

openGA is a header-only C++ genetic algorithm library, [openGA.hpp](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp), with a single-objective GA (`SOGA`), an interactive GA (`IGA`) and NSGA-III. It ships no operators: users write the initialization, evaluation, crossover (one child per call) and mutation. Its docs are the [user manual](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/openGA.pdf), the [README](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/README.md), the [examples](https://github.com/Arash-codedev/openGA/tree/f9b15e70600e20491504391dec6de5c64eb18913/examples) and the code generator openGA assist ([assist/main.js](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js)), which the README and the manual (p. 6) recommend for starting a program. The benchmark uses the header at commit f9b15e7 (2026-03-22), 16 commits after v1.0.5 (2020); header lines below are of that commit.

Adapter: [benchmarks/adapters/openga/](../../../benchmarks/adapters/openga/).
Know a better way to solve one of these problems with openGA? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs openGA

- **openGA's loop** ([bench.cpp#L20](../../../benchmarks/adapters/openga/bench.cpp#L20)): each generation keeps the whole population ([`transfer`, L582-599](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L582-L599)) and adds round(population × `crossover_fraction`) children ([L1669](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1669)). Each child has two distinct parents from a rank roulette with chance 1/√(rank + 1) ([L1132-1147](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1132-L1147), [L1586-1594](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1586-L1594)), and is mutated with probability `mutation_rate` ([L1613-1628](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1613-L1628)). Survival keeps the `elite_count` best of parents and children and fills the rest by rank roulette ([L1021-1061](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1021-L1061)).
- **Evaluations:** counted in `eval_solution`, with the first hit ([bench.cpp#L147](../../../benchmarks/adapters/openga/bench.cpp#L147)). openGA evaluates each child once and never a parent again.
- **Stop:** the adapter calls `solve_init` and `solve_next_generation`, the two halves of `solve()` ([L402-503](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L402-L503)), and stops after the generation that reaches the target, the budget or the time ([`run_ga`, bench.cpp#L397](../../../benchmarks/adapters/openga/bench.cpp#L397)). A run may pass its budget by one generation, or one initial population when an attempt starts (`last_generation`).
- **Keeping going (rule 2.2)** ([`stop_critera`, L1705-1740](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1705-L1740)):
  - `generation_max` is lifted (`INT_MAX`, [bench.cpp#L341](../../../benchmarks/adapters/openga/bench.cpp#L341)). The mutation's `shrink_scale` ([L531-539](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L531-L539)) depends only on the generation number (1 up to generation 5, then 1/√(generation − 4), squared with probability 0.4 and reset to 1 with probability 0.06), so this doesn't change it.
  - The best and average stalls (`best_stall_max` generations with a best-cost change below `tol_stall_best`, or `average_stall_max` with an average-cost change below `tol_stall_average`) end an attempt. openGA has no restart mechanism, so the adapter starts a new GA from a new random population with the seeds of rule 2.2 ([bench.cpp#L136](../../../benchmarks/adapters/openga/bench.cpp#L136)). Runs report `restarts`.
  - `user_request_stop` is never set.
- **Best solution:** the best of each attempt's last generation (`best_chromosome_index`), which the elite keeps; `best` is recomputed after the clock ([bench.cpp#L471](../../../benchmarks/adapters/openga/bench.cpp#L471)). `values` uses the same fitness functions ([bench.cpp#L747](../../../benchmarks/adapters/openga/bench.cpp#L747)).
- **Bounds (rule 2.4):** initial genes inside the bounds; the examples' mutation redraws the whole child while a gene is out of range; their crossover stays between the parents. Counted as `outside` ([bench.cpp#L655](../../../benchmarks/adapters/openga/bench.cpp#L655)).
- **Shift** (rule 1.4): computed once ([bench.cpp#L203](../../../benchmarks/adapters/openga/bench.cpp#L203)).
- **One thread:** `multi_threading = false` ([L1557-1561](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1557-L1561), [L1678-1682](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1678-L1682)).
- **Seeds:** openGA seeds its private `std::mt19937_64` from the clock ([L371-374](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L371-L374)) and has no setter. The adapter seeds it through the standard explicit-instantiation access to a private member ([bench.cpp#L114](../../../benchmarks/adapters/openga/bench.cpp#L114)).
- **Build:** `g++ -O3`, default target (rule 4.5).
- **Choosing among the docs (rule 6.2):** openGA states no preference. The adapter uses openGA's example for the problem type where there is one (so-rastrigin), and the program openGA assist generates, openGA's default starting point, for every type.
- **Separate tests:** 2026-09-25, 1.0.5+f9b15e7, seeds 0 to 4, the scenario's budget, 60 s cap, on a shared machine. `outside` was 0 in every run.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched: not run** ([bench.cpp#L499](../../../benchmarks/adapters/openga/bench.cpp#L499)). openGA has none of the matched GA's components: no tournament selection, no two-point crossover or bit flip (no operators), and no generational replacement without elitism.
- **Idiomatic** ([bench.cpp#L525-L548](../../../benchmarks/adapters/openga/bench.cpp#L525-L548)): no binary example, so openGA assist's default settings: population 200 ("medium", [main.js#L435-436](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L435-L436), the default in [index.html#L24](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/index.html#L24)), `crossover_fraction` 0.7, `mutation_rate` 0.2 ([#L460-461](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L460-L461)), `elite_count` 10 ([#L465](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L465)). The generated crossover mixes the parents per gene ([#L282-295](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L282-L295)): uniform crossover for 0/1 genes. The generated mutation adds a real-valued step ([#L259-279](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L259-L279)); the manual says to adapt the operators to the genes' type (p. 6), so the adapter uses a bit flip at 1/n.

**Keeping going:** the assist program's `best_stall_max` 10 ([#L464](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L464)) and openGA's other stall defaults ([L346-349](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L346-L349), manual p. 4): `tol_stall_best` 1e-6, `average_stall_max` 10, `tol_stall_average` 1e-4 ([bench.cpp#L365](../../../benchmarks/adapters/openga/bench.cpp#L365)). Its `generation_max` 1000 ([#L447](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L447)) is lifted.

**Left out:** the 2-variable real-valued example's settings ([so-1](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-1/example_so1.cpp#L138-L149): population 20, `mutation_rate` 0.4): not for binary genes.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best (median, best, worst) | Restarts (median) | Capped |
|---|---|---|---|---|---|---|---|
| onemax-100-idiomatic | ga | 5 | 5 | 8,950 | 100, 100, 100 | 1 | 0 |

In 3 of the 5 runs, a later attempt reaches the target.

## Permutation: N-Queens 32 and 64

**Methods:** no permutation example, so the assist settings, as for OneMax ([bench.cpp#L554](../../../benchmarks/adapters/openga/bench.cpp#L554)). openGA has no permutation operators, so the adapter uses the usual ones: a random permutation, order crossover (OX1, one child) and a swap of two genes.

**Keeping going:** the assist program's stall criteria, as for OneMax.

**Left out:** the examples' real-valued operators, which don't keep a permutation.

**Separate tests** (see the survival bug in [Bugs found](#bugs-found)):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best (median, best, worst) | Restarts (median) | Capped |
|---|---|---|---|---|---|---|---|
| nqueens-32-idiomatic | ga | 5 | 0 | - | 1, 1, 1 | 116 | 0 |
| nqueens-64-idiomatic | ga | 5 | 0 | - | 4, 3, 5 | 136 | 0 |

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** ([bench.cpp#L699](../../../benchmarks/adapters/openga/bench.cpp#L699)), with the operators of openGA's examples ([bench.cpp#L636](../../../benchmarks/adapters/openga/bench.cpp#L636)): genes uniform in the bounds; crossover r·a + (1 − r)·b with a new r per gene; mutation of every gene by mu·(rnd01() − rnd01()), redrawing the child while a gene is out of bounds ([bench.cpp#L667](../../../benchmarks/adapters/openga/bench.cpp#L667)).
- **`ga`:** [examples/so-rastrigin](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp), openGA's only example with an n-dimensional real vector, as written: population 10,000, `elite_count` 10, `crossover_fraction` 0.7, `mutation_rate` 0.1 ([#L145-159](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L145-L159)), mu = 1.7 · rnd01() · shrink_scale ([#L66](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L66)). Ackley uses it too.
- **`ga_assist`:** the assist program for real variables: the OneMax settings and mu = 0.2 · shrink_scale ([main.js#L265](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/assist/main.js#L265), "adjustable", used as generated).

**Keeping going:**
- `ga`: the example's stalls, 20 generations at 1e-6 for the best and the average ([#L153-156](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L153-L156)); its `generation_max` 1000 ([#L146](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-rastrigin/so-rastrigin.cpp#L146)) is lifted. The budgets allow 70 generations (Rastrigin 10), 142 (Ackley 30) and 285 (Rastrigin 30).
- `ga_assist`: as for OneMax.

**Left out:**
- The 2-variable examples [so-1](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-1/example_so1.cpp#L138-L149) (population 20, `mutation_rate` 0.4, mu = 0.2 · shrink_scale) and [so-init-solutions](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/examples/so-init-solutions/example_so-init-solutions.cpp#L161-L180) (population 10 from given solutions): so-rastrigin is the example for an n-dimensional function.
- The CI test [various-so-rastrigin.cpp](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/ci-test-cases/various-so-rastrigin.cpp#L147-L161) (population 5,000, 50 generations): a test, not a recommendation.
- A custom `get_shrink_scale`: allowed (manual p. 2), but no example sets it.

**Separate tests** (with the shift of rule 1.4; see the survival bug in [Bugs found](#bugs-found)):

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best (median, best, worst) | Restarts (median) | Capped |
|---|---|---|---|---|---|---|---|
| rastrigin-10-idiomatic | ga | 5 | 0 | - | 4.977, 1.993, 5.970 | 0 | 0 |
| rastrigin-10-idiomatic | ga_assist | 5 | 0 | - | 3.980, 2.985, 5.970 | 25 | 0 |
| rastrigin-30-idiomatic | ga | 5 | 0 | - | 34.84, 32.84, 45.77 | 0 | 4 |
| rastrigin-30-idiomatic | ga_assist | 5 | 0 | - | 46.77, 38.81, 49.75 | 25 | 0 |
| ackley-30-idiomatic | ga | 5 | 0 | - | 8.696, 8.040, 10.86 | 0 | 0 |
| ackley-30-idiomatic | ga_assist | 5 | 0 | - | 13.97, 13.38, 14.12 | 20 | 0 |

The capped `ga` runs of Rastrigin 30 stopped after 1.1 to 1.3 million of the 2 million evaluations.

## Continuous, unimodal: Rosenbrock 10

**Methods:** `ga_assist`, as above ([bench.cpp#L715](../../../benchmarks/adapters/openga/bench.cpp#L715)): openGA has no example for a unimodal function, so its default, the assist program.

**Keeping going:** as above.

**Left out:** `ga`, the so-rastrigin example: an example for a multimodal function ([bench.cpp#L706](../../../benchmarks/adapters/openga/bench.cpp#L706)). The rest as above.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit, median evaluations | Best (median, best, worst) | Restarts (median) | Capped |
|---|---|---|---|---|---|---|---|
| rosenbrock-10-idiomatic | ga_assist | 5 | 0 | - | 1.308, 0.02888, 3.093 | 1 | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Not run** ([bench.cpp#L811](../../../benchmarks/adapters/openga/bench.cpp#L811)). The matched NSGA-III needs the library's own SBX and polynomial mutation, and openGA ships no operators. openGA has no NSGA-II, SPEA2, MOEA/D or SMS-EMOA (`GA_MODE`, [L42-47](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L42-L47)).

## Can't run

- Matched OneMax 100 and 1000: no tournament selection, two-point crossover, bit flip or generational replacement.
- The multi-objective scenarios: no SBX and no polynomial mutation.

## Bugs found

**The single-objective survival can't keep a child** ([Arash-codedev/openGA#30](https://github.com/Arash-codedev/openGA/issues/30), open; not worked around).
- `generate_selection_chance` builds cumulative rank chances over parents and children but divides by the value at index population − 1 ([L1145](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1145)). Every child's value is then above 1, and `select_parent` ([L1586-1594](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1586-L1594)), with r in [0, 1), never returns a child. So a child survives only among the `elite_count` best ([`select_population_SO`, L1021-1061](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1021-L1061)).
- In the same loop ([L1057](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L1057)), after drawing index j it blocks `sorted_indices[j]` instead of j, so j can be drawn again and the population fills with copies.
- NSGA-III has its own survival and isn't affected.

Effect: the population collapses to copies of a few individuals. N-Queens 32, seed 0, without the stall criteria: 117 distinct permutations of 200 after one generation, 14 after 20, and 1 from generation 100 on; with them, attempts end after about 30 generations (110 to 140 restarts per run). Rastrigin 10, seed 0 (earlier shift in [−1, 1]): 285 distinct individuals of 10,000 after 10 generations, about 210 after; at generation 70 the average is 1.006 and the best 0.997, one coordinate a basin away from the optimum. So the continuous results are near whole numbers.

With both fixes of the issue in a copy of the header (normalize by the last cumulative value, block j), not benchmark results (seeds 0 to 4, the scenario's budget, the earlier shift in [−1, 1], the `ga` radius scaled to the domain width, earlier seeds):

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

**Fixed at the pinned commit:** [openGA#23](https://github.com/Arash-codedev/openGA/issues/23): from c82d71b (2023) to 2025, `solve_next_generation` stored `last_generation` only when it had fronts. f9b15e7 stores every generation ([L484-491](https://github.com/Arash-codedev/openGA/blob/f9b15e70600e20491504391dec6de5c64eb18913/src/openGA.hpp#L484-L491)); the adapter depends on this fix.
