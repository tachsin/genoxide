# PyGAD (Python, 3.7.0)

PyGAD is a genetic-algorithm library: one class, `pygad.GA`, whose parameters choose the parent selection, crossover, mutation, elitism and callbacks, for single-objective problems and, through its NSGA-II and NSGA-III parent selections, multi-objective ones. Its documentation is [pygad.readthedocs.io](https://pygad.readthedocs.io/en/latest/), built from the `docs/source/` folder of its repository ([ahmedfgad/GeneticAlgorithmPython](https://github.com/ahmedfgad/GeneticAlgorithmPython), tag [3.7.0](https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/3.7.0)). Version 3.7.0 added benchmark problems (`pygad.benchmarks`: Rastrigin, Rosenbrock, Ackley, ZDT, DTLZ, knapsack, TSP) with "a runnable example per benchmark" under [examples/benchmarks/](https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/3.7.0/examples/benchmarks) ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html)). Those examples are PyGAD's own configurations for each problem type, and the idiomatic runs below take their settings.

Adapter: [benchmarks/adapters/pygad/](../../../benchmarks/adapters/pygad/).
Know a better way to solve one of these problems with PyGAD? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs PyGAD

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), as numpy functions of one solution ([bench.py, lines 101-205](../../../benchmarks/adapters/pygad/bench.py#L101-L205)): PyGAD calls the fitness function with one solution, a numpy array, at a time, and its own benchmark classes (`pygad/benchmarks/classic.py`, `zdt.py`) are written this way. PyGAD maximizes, so a minimized problem's fitness is its negated value. Batch fitness (`fitness_batch_size`) isn't the default and none of PyGAD's examples for these problems uses it.
- **Evaluations:** every call of the fitness function counts, in the adapter's `fitness_func` ([`Budget.count`, line 62](../../../benchmarks/adapters/pygad/bench.py#L62)). PyGAD doesn't call it for a solution identical to one of the previous generation's elites or kept parents (`cal_pop_fitness` in `utils/engine.py`), so its generations differ in size; each run prints `last_generation`, the evaluations of its last one.
- **Stop:** `on_generation` returns `"stop"` at the target, the budget or the time cap, checked after each generation; `num_generations` is 10⁷, never reached.
- **Keeping going (rule 2.2):** the examples' `num_generations` is only a budget, so it's lifted. PyGAD's convergence criterion (`stop_criteria="saturate_N"`) is off by default and none of the examples sets it, so no convergence criterion applies and every run goes to the target, the budget or the time cap. The GA has no restart mechanism, and none is needed.
- **Bounds (rule 2.4):** see each section. Every run of a continuous or multi-objective problem prints `outside`, the number of evaluated solutions outside the box, counted in `fitness_func`. PyGAD's bounded operators keep it at 0; it was 0 in every run of `run.py check` and in the tests that printed it.
- **Time:** from before the `pygad.GA` constructor, which creates the initial population, to the end of the run.
- **One thread:** numpy's BLAS is set to one thread before numpy is imported ([line 19](../../../benchmarks/adapters/pygad/bench.py#L19)); `parallel_processing` is left at its default, `None`.
- **Seeds:** `random_seed=seed`, which seeds numpy's and Python's generators.
- **Solutions:** a single-objective run prints the best solution `fitness_func` saw evaluated; a multi-objective run the non-dominated part of its final survivors with their variables ([lines 451-458](../../../benchmarks/adapters/pygad/bench.py#L451-L458)).
- **Rule 5.3:** the adapter stops a solver after 3 seeds that all ran the full 60 s without reaching the target ([line 533](../../../benchmarks/adapters/pygad/bench.py#L533)).

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched (OneMax 100 and 1000):** the matched GA ([lines 230-244](../../../benchmarks/adapters/pygad/bench.py#L230-L244)): 300 solutions, all of them parents (`num_parents_mating` 300), `parent_selection_type="tournament"` with `K_tournament` 3, generational (`keep_elitism` 0, `keep_parents` 0), two-point crossover with probability 0.5, `mutation_probability` 0.2 / n per bit. The genes are PyGAD's binary genes, `gene_space=[0, 1]` with `gene_type=int` ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html), "Knapsack"); a mutated gene takes the other value of its space (`generate_gene_value_from_space` in `helper/misc.py` removes the current value), a bit flip. The crossover is PyGAD's own `two_points_crossover`, applied to a pair with probability 0.5, else the child copies its first parent ([`two_points_at`, lines 210-224](../../../benchmarks/adapters/pygad/bench.py#L210-L224)): PyGAD's `crossover_probability` means something else, it makes each parent eligible with that probability and crosses two eligible parents, so with 300 parents nearly every child is crossed. Differences from the other libraries: PyGAD makes one child from parents k and k + 1, and evaluates every child, copies included (with `keep_parents` 0 it doesn't reuse the parents' fitness).
- **Idiomatic (OneMax 100):** [examples/benchmarks/example_knapsack.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_knapsack.py), PyGAD's binary example: `gene_space=[0, 1]`, `gene_type=int`, 30 solutions, 10 parents, and PyGAD's defaults ([`pygad.GA`](https://pygad.readthedocs.io/en/latest/pygad.html)): steady-state selection (`"sss"`), single-point crossover, random mutation of 10% of the genes (`mutation_percent_genes="default"`), `keep_elitism` 1 ([line 249](../../../benchmarks/adapters/pygad/bench.py#L249)).

**Keeping going:** both run to the target or the budget.

**Left out:** adaptive mutation ([Adaptive Mutation](https://pygad.readthedocs.io/en/latest/adaptive_mutation.html)): the page argues against a constant mutation rate, but its rates (`mutation_num_genes=(3, 1)` for 6 genes, and a list of syntax examples) aren't a setting for this problem type, so the idiomatic run keeps the example's defaults (rule 6.2).

**Separate tests** (2026-09-25, PyGAD 3.7.0, numpy 2.5.3, Python 3.13.9, seeds 0 to 4, the scenario's budget, 60 s cap; other processes ran on the machine, so times are indicative):

OneMax 100, matched (budget 200,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 9,600 | 100 (100, 100) | 0 |

OneMax 1000, matched (budget 2,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 135,300 | 1,000 (1,000, 1,000) | 0 |

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 0 | - | 90 (92, 89) | 0 |

The idiomatic GA flips 10 of the 100 bits of every child, the default 10% of the genes; near the optimum nearly every child is worse than its parents, and the runs end the budget at 89 to 92.

For comparison, not benchmarked: the matched GA with PyGAD's `crossover_probability=0.5` instead of the per-pair crossover needs 7,200 and 113,700 evaluations (median of 5), and the adapter up to 0.6.0, which also had `gene_space={"low": 0, "high": 2}` (a mutated bit drawn again, which flips it half the time), 7,200 and 147,900.

## Permutation: N-Queens 32 and 64

**Methods:**
- **`ga`:** [examples/benchmarks/example_tsp.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_tsp.py), PyGAD's permutation example ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html), "Travelling Salesman Problem": `gene_space=list(range(num_cities))`, `gene_type=int` and `allow_duplicate_genes=False` "keep the permutation constraint"): 30 solutions, 10 parents, PyGAD's defaults otherwise, the same as for OneMax ([lines 252-265](../../../benchmarks/adapters/pygad/bench.py#L252-L265)). As `pygad/benchmarks/tsp.py` does for tours, a solution that isn't a permutation (PyGAD couldn't remove a duplicate) gets a value worse than any permutation, scaled by its missing values ([lines 286-297](../../../benchmarks/adapters/pygad/bench.py#L286-L297)).

**Keeping going:** it runs to the target, the budget or the time cap.

**Left out:** the swap, inversion and scramble mutations: they're among PyGAD's mutation types ([`pygad.GA`](https://pygad.readthedocs.io/en/latest/pygad.html), `mutation_type`), but its permutation example uses the default, random mutation. The community notebook [example_travelling_salesman.ipynb](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/example_travelling_salesman.ipynb) (PyGAD 2.17) writes its own PMX crossover and inversion mutation, "as the out of the box genetic algorithm does not support a TSP"; the 3.7.0 benchmark example is the current one.

**Separate tests** (2026-09-25, as above):

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 3 | 0 | - | 7 (7, 7) | 3 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 3 | 0 | - | 18 (15, 18) | 3 |

PyGAD's random mutation can't change a permutation (see [Bugs found](#bugs-found)), so the population converges to a single board within a few dozen generations, after which every child is a copy of it, which PyGAD doesn't evaluate: the runs used 184 to 216 evaluations in their 60 s. For comparison, not benchmarked: the same example with PyGAD's other mutation types. `mutation_type="swap"` reached N-Queens 32 in 5 runs of 5 (median 32,462 evaluations) and 64 in 5 of 5 (90,251); `"inversion"` 1 of 5 and 0 of 3; `"scramble"` 0 of 5 and 0 of 3.

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Methods:**
- **`ga`:** [example_classic_rastrigin.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_rastrigin.py) and [example_classic_ackley.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_ackley.py): 40 solutions, 10 parents, `crossover_type="sbx"` with `sbx_crossover_eta` 20, `mutation_type="polynomial"` with `polynomial_mutation_eta` 20 (each gene with probability 1 / n, PyGAD's default when `mutation_probability` isn't set), `init_range_low` and `init_range_high` at the problem's bounds, PyGAD's defaults otherwise (steady-state selection, `keep_elitism` 1). [example_classic_rosenbrock.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_rosenbrock.py): the same with `sbx_crossover_eta` 30 ([lines 267-280](../../../benchmarks/adapters/pygad/bench.py#L267-L280)). The same settings appear in the docs' "Example: SOO" ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html)).

**Bounds:** PyGAD's `sbx` and `polynomial` operators clip each gene to its initial range (`get_initial_population_range`), which is the problem's box: every evaluated solution is inside it.

**Keeping going:** it runs to the target or the budget.

**Left out:** PyGAD has one algorithm, the GA; its other crossover and mutation types (single-point, uniform, random mutation) aren't what its examples use for these problems.

**Separate tests** (2026-09-25, as above):

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 61,011 | 0.0091 (0.0063, 0.0093) | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 440,692 | 0.0100 (0.0086, 0.0100) | 0 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 0 | - | 18.7 (14.8, 19.3) | 0 |

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 0 | - | 7.01 (6.95, 7.16) | 0 |

PyGAD's `sbx` always makes the child below its parents' midpoint (see [Bugs found](#bugs-found)), which pulls every gene towards the lower bound. The runs show it. For comparison, not benchmarked: the same settings with an SBX that makes either child with probability 0.5, and is otherwise PyGAD's: Rastrigin 10 in 5 of 5 runs (median 11,372 evaluations), Rastrigin 30 in 5 of 5 (41,008), Ackley 30 in 5 of 5 (100,222), and Rosenbrock 10 in 0 of 5, at 1.90 (1.21, 4.12).

## Multi-objective (matched): ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

PyGAD runs NSGA-II or NSGA-III when the fitness function returns several values and the parent selection is one of theirs ([Multi-Objective Optimization](https://pygad.readthedocs.io/en/latest/multi_objective.html)). But they're parent selections: a PyGAD generation selects the parents from the population, crosses and mutates them, and the next population is either the `keep_elitism` best of the current one by PyGAD's NSGA-II sort (front, then crowding distance), followed by the offspring, or, with `keep_parents=-1`, the parents followed by the offspring (`run` in `utils/engine.py`). Each algorithm's survival is built from these, with N the matched population, 100 (92 with 3 objectives), and a PyGAD population of 2N ([`run_front`, lines 415-458](../../../benchmarks/adapters/pygad/bench.py#L415-L458)).

**Methods:**
- **`nsga2`:** `keep_elitism` N and N offspring: the N elites of each generation are the best N of the previous elites and their offspring by the NSGA-II sort, which is NSGA-II's survival. The parents come from PyGAD's own crowded binary tournament (`parent_selection_type="tournament_nsga2"`, `K_tournament` 2). Because PyGAD selects the parents before the elites, it draws them from all 2N, not from the N survivors only.
- **`nsga3`** (DTLZ2 and DTLZ1): `parent_selection_type="nsga3"` with `nsga3_num_divisions` 12 (Das-Dennis reference points), N parents, `keep_elitism` 0 and `keep_parents` -1: the parents are the N survivors of the 2N by NSGA-III's niching, and the next population is those survivors and their N offspring, which is NSGA-III's survival. Its mating is random, as NSGA-III's: the crossover pairs the survivors in a random order. PyGAD 3.7.0 added NSGA-III; the adapter up to 0.6.0 ran NSGA-II only.
- **Operators:** PyGAD's `sbx` can't take the matched settings: it makes one child per pair, always the lower one (see [Bugs found](#bugs-found)), and its `crossover_probability` isn't a per-pair probability. So the crossover is bounded SBX as DEAP's `cxSimulatedBinaryBounded`, given to PyGAD as a crossover function ([User-Defined Crossover, Mutation, and Parent Selection Operators](https://pygad.readthedocs.io/en/latest/user_defined_operators.html); [`make_sbx`, lines 363-404](../../../benchmarks/adapters/pygad/bench.py#L363-L404)): η 15 on each pair with probability 0.9 (NSGA-II), η 30 on every pair (NSGA-III). The mutation is PyGAD's own `"polynomial"`, Deb's bounded polynomial mutation, with η 20 and `mutation_probability` 1 / n.

Differences from the textbook algorithms and the other libraries: the initial population has 2N random solutions, so the first generation costs N more evaluations; PyGAD's crowding distance normalizes each objective by its range over the whole population, not the front; a child identical to a previous elite or parent takes its fitness without an evaluation; and PyGAD's non-dominated sorting compares every pair of solutions in Python, two or three times per generation, which is slow enough for the 60 s cap to end runs before the budget.

**Bounds:** the SBX clips to [0, 1], and PyGAD's polynomial mutation clips to the initial range [0, 1].

**Keeping going:** each run uses its whole budget, or reaches the time cap.

**The front:** the non-dominated part of the final N survivors: the elites (NSGA-II) or the parents (NSGA-III) that PyGAD selects from the last population after the last generation.

**Parent selection:** the docs list `nsga2`, `tournament_nsga2`, `nsga3` and `tournament_nsga3`, with no stated preference, and PyGAD's ZDT example uses `nsga2`, which takes the best N by front and crowding distance as parents. The matched NSGA-II's parents come from a crowded binary tournament, which is `tournament_nsga2`; NSGA-III's survivors are `nsga3`, as in PyGAD's DTLZ2 example ([example_dtlz2.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_dtlz2.py)). In matched mode the matched algorithm decides (rule 6.1).

**Not run:** SPEA2, SMS-EMOA and MOEA/D: PyGAD has none of them. Its `tournament_nsga3` parent selection is a variant of NSGA-III's mating, not another algorithm.

**Separate tests** (2026-09-25, as above; hypervolume with `run.py`'s reference points):

With the 60 s cap (5 seeds; rule 5.3 stopped a solver after 3 seeds that all reached the cap):

| Problem | Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|---|
| ZDT1 | nsga2 | 3 | 0.8596 (0.8651, 0.8544) | 16,872 | 3 |
| ZDT2 | nsga2 | 3 | 0.5242 (0.5307, 0.3228) | 17,413 | 3 |
| ZDT3 | nsga2 | 3 | 1.3241 (1.3248, 1.3211) | 23,023 | 3 |
| DTLZ2 | nsga2 | 3 | 0.6851 (0.6860, 0.6732) | 18,559 | 3 |
| DTLZ2 | nsga3 | 5 | 0.7437 (0.7441, 0.7435) | 25,022 | 1 |
| DTLZ1 | nsga2 | 3 | 1.2308 (1.2935, 0.9296) | 19,585 | 3 |
| DTLZ1 | nsga3 | 3 | 1.3044 (1.3046, 1.3030) | 24,573 | 3 |

Without the 60 s cap, for comparison (seeds 0 to 2, a cap of 300 s, 600 s for ZDT2, ZDT3 and DTLZ1, 2 to 3 runs at a time): every run used its whole budget, in 64 to 89 s (NSGA-II) and about 55 s (NSGA-III), with median hypervolumes of 0.8676 (ZDT1), 0.5338 (ZDT2), 1.3252 (ZDT3), 0.6874 and 0.7436 (DTLZ2, NSGA-II and NSGA-III) and 1.3010 and 1.3046 (DTLZ1). So the cap cuts the NSGA-II runs at 65 to 90% of their budget on this loaded machine; the benchmark's pinned runs, one at a time, will be faster.

PyGAD's NSGA-II tournament draws from all 2N (see above). For comparison, not benchmarked: a crowded tournament among the N survivors only, as a user-defined parent selection, reached 0.8694 on ZDT1 and 0.6922 on DTLZ2 (medians of 3, full budget), against 0.8676 and 0.6874 with PyGAD's own tournament. The adapter keeps PyGAD's.

## Can't run

PyGAD runs every scenario. In the multi-objective scenarios it runs NSGA-II and NSGA-III, not SPEA2, SMS-EMOA or MOEA/D, which it doesn't have.

## Bugs found

| Bug | Effect | Worked around | Reported |
|---|---|---|---|
| `sbx` makes one child per pair, always `0.5 * ((y1 + y2) - beta_q * (y2 - y1))` with `beta_q > 0` ([utils/crossover.py, line 329](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/crossover.py#L329)): the child below the parents' midpoint, which pulls every crossed gene towards the lower bound | the continuous runs above: Ackley 30 and Rosenbrock 10 fail, Rastrigin needs 5 to 11 times the evaluations of an unbiased SBX; in the multi-objective runs it would favour ZDT's optimum, at the lower bound | in the matched multi-objective runs only, where the matched SBX is a crossover function in the adapter; not in the idiomatic runs | [ahmedfgad/GeneticAlgorithmPython#369](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/369) |
| With `allow_duplicate_genes=False` and a gene space of n values for n genes, as in PyGAD's permutation example, the random mutation never changes a solution: it picks a value from the space that isn't already in the solution (`select_unique_value`, [helper/unique.py, lines 269-280](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/helper/unique.py#L269-L280)), and in a permutation there's none, so the gene keeps its value. In 1,000 mutations of random permutations of 8, none changed | the N-Queens runs above: the population converges to one board, and the runs reach the time cap after about 200 evaluations | no | not yet |
| The docs describe `swap_mutation` as swapping "2 randomly selected genes" ([utils.md](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/docs/source/utils.md)), but it swaps a gene of the first half with the gene half the length after it ([utils/mutation.py, lines 363-387](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/mutation.py#L363-L387)) | none here: the adapter doesn't use it | - | not yet |
