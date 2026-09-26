# PyGAD (Python, 3.7.0)

PyGAD is a genetic-algorithm library: one class, `pygad.GA`, whose parameters choose the parent selection, crossover, mutation, elitism and callbacks, for single-objective problems and, through its NSGA-II and NSGA-III parent selections, multi-objective ones. Its documentation is [pygad.readthedocs.io](https://pygad.readthedocs.io/en/latest/), built from the `docs/source/` folder of its repository ([ahmedfgad/GeneticAlgorithmPython](https://github.com/ahmedfgad/GeneticAlgorithmPython), tag [3.7.0](https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/3.7.0)). Version 3.7.0 added benchmark problems (`pygad.benchmarks`: Rastrigin, Rosenbrock, Ackley, ZDT, DTLZ, knapsack, TSP) with "a runnable example per benchmark" under [examples/benchmarks/](https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/3.7.0/examples/benchmarks) ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html)). Those examples are PyGAD's own configurations for each problem type, and the idiomatic runs below take their settings.

Adapter: [benchmarks/adapters/pygad/](../../../benchmarks/adapters/pygad/).
Know a better way to solve one of these problems with PyGAD? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs PyGAD

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), as numpy functions of a batch of solutions, one per row ([bench.py, lines 115-231](../../../benchmarks/adapters/pygad/bench.py#L115-L231)). PyGAD's batch fitness, `fitness_batch_size` ([Batch Fitness Calculation](https://pygad.readthedocs.io/en/latest/fitness_calculation.html#batch-fitness-calculation)), passes the solutions of a generation to one call (rule 3.4). It changes nothing else: PyGAD decides which solutions to evaluate as without it, then evaluates those in one batch (`cal_pop_fitness` in `utils/engine.py`); runs with and without it give the same evaluations and the same best values. PyGAD maximizes, so a minimized problem's fitness is its negated value.
- **Evaluations:** each row of a batch counts as one evaluation, in the adapter's `fitness_func` ([`Budget.count`, lines 66-71](../../../benchmarks/adapters/pygad/bench.py#L66-L71)). PyGAD doesn't evaluate a solution identical to one of the previous generation's elites or kept parents, so its generations differ in size; each run prints `last_generation`, the evaluations of its last one.
- **First hit (rule 3.3):** the counter records the first row of a batch that reaches the target, its number and the clock ([`Budget.keep`, lines 73-86](../../../benchmarks/adapters/pygad/bench.py#L73-L86)).
- **Stop:** `on_generation` returns `"stop"` at the target, the budget or the time cap, checked after each generation; `num_generations` is 10⁷, never reached.
- **Keeping going (rule 2.2):** the examples' `num_generations` is only a budget, so it's lifted. PyGAD's convergence criterion (`stop_criteria="saturate_N"`) is off by default and none of the examples sets it, so no convergence criterion applies and every run goes to the target, the budget or the time cap. The GA has no restart mechanism, and none is needed.
- **Bounds (rule 2.4):** see each section. Every run of a continuous or multi-objective problem prints `outside`, the number of evaluated solutions outside the box, counted in `fitness_func`. PyGAD's bounded operators keep it at 0; it was 0 in every test run.
- **Time:** from before the `pygad.GA` constructor, which creates the initial population, to the end of the run. The front of a multi-objective run is extracted after the clock stops ([line 504](../../../benchmarks/adapters/pygad/bench.py#L504)).
- **One thread:** numpy's BLAS is set to one thread before numpy is imported, by assigning the variables ([lines 18-20](../../../benchmarks/adapters/pygad/bench.py#L18-L20)); `parallel_processing` is left at its default, `None`.
- **Seeds:** `random_seed=seed`, which seeds numpy's and Python's generators.
- **Solutions:** a single-objective run prints the best solution `fitness_func` saw evaluated; a multi-objective run the non-dominated part of its final survivors with their variables ([`front_of`, lines 425-436](../../../benchmarks/adapters/pygad/bench.py#L425-L436)).
- **Rule 5.3:** the adapter stops a solver after 3 seeds that all ran the full 60 s without reaching the target ([line 519](../../../benchmarks/adapters/pygad/bench.py#L519)).

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched (OneMax 100 and 1000):** the matched GA with PyGAD's own components ([lines 242-259](../../../benchmarks/adapters/pygad/bench.py#L242-L259)): 300 solutions, all of them parents (`num_parents_mating` 300), `parent_selection_type="tournament"` with `K_tournament` 3, generational without elitism (`keep_elitism` 0, `keep_parents` 0), `crossover_type="two_points"` with `crossover_probability` 0.5, and `mutation_type="random"` with `mutation_probability` 0.2 / n. The genes are PyGAD's binary genes, `gene_space=[0, 1]` with `gene_type=int` ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html), "Knapsack"); a mutated gene takes the other value of its space (`generate_gene_value_from_space` in `helper/misc.py` removes the current value), a bit flip.
- **Idiomatic (OneMax 100):** [examples/benchmarks/example_knapsack.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_knapsack.py), PyGAD's binary example: `gene_space=[0, 1]`, `gene_type=int`, 30 solutions, 10 parents, and PyGAD's defaults ([`pygad.GA`](https://pygad.readthedocs.io/en/latest/pygad.html)): steady-state selection (`"sss"`), single-point crossover, random mutation of 10% of the genes (`mutation_percent_genes="default"`), `keep_elitism` 1 ([line 264](../../../benchmarks/adapters/pygad/bench.py#L264)).

**Matched differences** (PyGAD can't express these settings exactly):
- **Two-point crossover:** PyGAD's `two_points_crossover` always takes exactly n / 2 consecutive genes from the second parent (a bug, see [Bugs found](#bugs-found)), not a segment between two random points.
- **Crossover probability:** PyGAD's `crossover_probability` makes each parent eligible for crossover with that probability, and crosses two parents drawn from the eligible ones ([`crossover_probability`](https://pygad.readthedocs.io/en/latest/pygad.html)). With 300 parents there are always two eligible ones, so every child is crossed, where the matched GA crosses each pair with probability 0.5.
- **One child per crossover,** not two.
- **Mutation:** PyGAD's `mutation_probability` is per gene, and it has no probability per individual. So each gene of every child flips with probability 0.2 / n: the matched mean of 0.2 bits per child, where the matched GA mutates 20% of the individuals at 1 / n per bit.
- **Evaluations:** with `keep_parents` 0, PyGAD evaluates every child, where the matched GA keeps the fitness of an individual it neither crossed nor mutated. With every child crossed, few children are unchanged copies.

**Keeping going:** both run to the target or the budget.

**Left out:** adaptive mutation ([Adaptive Mutation](https://pygad.readthedocs.io/en/latest/adaptive_mutation.html)): the page argues against a constant mutation rate, but its rates (`mutation_num_genes=(3, 1)` for 6 genes, and a list of syntax examples) aren't a setting for this problem type, so the idiomatic run keeps the example's defaults (rule 6.2).

**Separate tests** (2026-09-25, PyGAD 3.7.0, numpy 2.5.3, Python 3.13.9, seeds 0 to 4, the scenario's budget, 60 s cap; other processes ran on the machine, so times are indicative):

OneMax 100, matched (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 7,091 | 100 (100, 100) | 0 |

OneMax 1000, matched (budget 2,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 113,594 | 1,000 (1,000, 1,000) | 0 |

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 0 | - | 90 (92, 89) | 0 |

The idiomatic GA flips 10 of the 100 bits of every child, the default 10% of the genes; near the optimum nearly every child is worse than its parents, and the runs end the budget short of the optimum.

## Permutation: N-Queens 32 and 64

**Methods:**
- **`ga`:** [examples/benchmarks/example_tsp.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_tsp.py), PyGAD's permutation example ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html), "Travelling Salesman Problem": `gene_space=list(range(num_cities))`, `gene_type=int` and `allow_duplicate_genes=False` "keep the permutation constraint"): 30 solutions, 10 parents, PyGAD's defaults otherwise, the same as for OneMax ([lines 267-276](../../../benchmarks/adapters/pygad/bench.py#L267-L276)). As `pygad/benchmarks/tsp.py` does for tours, a solution that isn't a permutation (PyGAD couldn't remove a duplicate) gets a value worse than any permutation, scaled by its missing values ([lines 297-315](../../../benchmarks/adapters/pygad/bench.py#L297-L315)).

**Keeping going:** it runs to the target, the budget or the time cap.

**Left out:** the swap, inversion and scramble mutations: they're among PyGAD's mutation types ([`pygad.GA`](https://pygad.readthedocs.io/en/latest/pygad.html), `mutation_type`), but its permutation example uses the default, random mutation. The community notebook [example_travelling_salesman.ipynb](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/example_travelling_salesman.ipynb) (PyGAD 2.17) writes its own PMX crossover and inversion mutation, "as the out of the box genetic algorithm does not support a TSP"; the 3.7.0 benchmark example is the current one.

**Separate tests** (as above):

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 3 | 0 | - | 7 (7, 7) | 3 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 3 | 0 | - | 18 (15, 18) | 3 |

PyGAD's random mutation can't change a permutation (see [Bugs found](#bugs-found)), so the population converges to a single board within a few dozen generations, after which every child is a copy of it, which PyGAD doesn't evaluate: the runs use about 200 evaluations in their 60 s. For comparison, not benchmarked: the same example with PyGAD's other mutation types. `mutation_type="swap"` reached N-Queens 32 in 5 runs of 5 (median 32,462 evaluations) and 64 in 5 of 5 (90,251); `"inversion"` 1 of 5 and 0 of 3; `"scramble"` 0 of 5 and 0 of 3.

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Methods:**
- **`ga`:** [example_classic_rastrigin.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_rastrigin.py) and [example_classic_ackley.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_ackley.py): 40 solutions, 10 parents, `crossover_type="sbx"` with `sbx_crossover_eta` 20, `mutation_type="polynomial"` with `polynomial_mutation_eta` 20 (each gene with probability 1 / n, PyGAD's default when `mutation_probability` isn't set), `init_range_low` and `init_range_high` at the problem's bounds, PyGAD's defaults otherwise (steady-state selection, `keep_elitism` 1). [example_classic_rosenbrock.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_rosenbrock.py): the same with `sbx_crossover_eta` 30 ([lines 278-291](../../../benchmarks/adapters/pygad/bench.py#L278-L291)). The same settings appear in the docs' "Example: SOO" ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html)).

**Bounds:** PyGAD's `sbx` and `polynomial` operators clip each gene to its initial range (`get_initial_population_range`), which is the problem's box: every evaluated solution is inside it.

**Keeping going:** it runs to the target or the budget.

**Left out:** PyGAD has one algorithm, the GA; its other crossover and mutation types (single-point, uniform, random mutation) aren't what its examples use for these problems.

**Separate tests** (as above):

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 50,683 | 0.00865 (0.00524, 0.009) | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 3 | 409,849 | 0.00954 (0.00942, 0.0375) | 2 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 3 | 0 | - | 19.7 (16.7, 19.9) | 3 |

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 0 | - | 7.01 (6.95, 7.15) | 0 |

PyGAD's `sbx` always makes the child below its parents' midpoint (see [Bugs found](#bugs-found)), which pulls every gene towards the lower bound. The runs show it. Without the 60 s cap (seeds 0 to 2, a 900 s cap), for comparison: Rastrigin 30 in 3 runs of 3 (median first hit 434,052 evaluations), and Ackley 30 in none, at 16.7 to 19.9 after the whole budget.

## Multi-objective (matched): ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

PyGAD runs NSGA-II or NSGA-III when the fitness function returns several values and the parent selection is one of theirs ([Multi-Objective Optimization](https://pygad.readthedocs.io/en/latest/multi_objective.html)). But they're parent selections: a PyGAD generation selects the parents from the population, crosses and mutates them, and the next population is either the `keep_elitism` best of the current one by PyGAD's NSGA-II sort (front, then crowding distance), followed by the offspring, or, with `keep_parents=-1`, the parents followed by the offspring (`run` in `utils/engine.py`). Each algorithm's survival is built from these settings, with N the matched population, 100 (92 with 3 objectives), and a PyGAD population of 2N ([`run_front`, lines 387-422](../../../benchmarks/adapters/pygad/bench.py#L387-L422)).

**Methods:**
- **`nsga2`:** `keep_elitism` N and N offspring: the N elites of each generation are the best N of the previous elites and their offspring by the NSGA-II sort, which is NSGA-II's survival. The parents come from PyGAD's own crowded binary tournament (`parent_selection_type="tournament_nsga2"`, `K_tournament` 2). Because PyGAD selects the parents before the elites, it draws them from all 2N, not from the N survivors only.
- **`nsga3`** (every problem): `parent_selection_type="nsga3"` with `nsga3_num_divisions` 99 with 2 objectives (100 Das-Dennis reference points) and 12 with 3 (91), N parents, `keep_elitism` 0 and `keep_parents` -1: the parents are the N survivors of the 2N by NSGA-III's niching, and the next population is those survivors and their N offspring, which is NSGA-III's survival.
- **Operators:** PyGAD's own, bugs included (rule 6.1). The crossover is `crossover_type="sbx"`: `sbx_crossover_eta` 15 with `crossover_probability` 0.9 for NSGA-II, and `sbx_crossover_eta` 30 with every child crossed (`crossover_probability` unset) for NSGA-III. The mutation is `"polynomial"`, Deb's bounded polynomial mutation, with `polynomial_mutation_eta` 20 and `mutation_probability` 1 / n.

**Differences** from the matched settings and the other libraries:
- PyGAD's `sbx` makes one child per crossover, always the one below the parents' midpoint (see [Bugs found](#bugs-found)), and crosses every gene, not each with probability 0.5. It pulls every variable towards 0: on ZDT the first one too, which sets a solution's place along the front, so the fronts crowd towards f1 = 0; on DTLZ the distance variables, whose optimum is 0.5.
- Its `crossover_probability` isn't a per-pair probability: each parent is eligible with probability 0.9, and a child comes from two parents drawn from the eligible ones. Without it (NSGA-III), child k comes from parents k and k + 1, in the order PyGAD's NSGA-III selection returns them: by front, not at random.
- The initial population has 2N random solutions, so the first generation costs N more evaluations.
- PyGAD's crowding distance normalizes each objective by its range over the whole population, not the front.
- A child identical to a previous elite or parent takes its fitness without an evaluation.
- PyGAD's non-dominated sorting compares every pair of solutions in Python, two or three times per generation, which is slow enough for the 60 s cap to end runs before the budget.

**Bounds:** PyGAD's `sbx` and polynomial mutation clip each gene to the initial range, [0, 1].

**Keeping going:** each run uses its whole budget, or reaches the time cap.

**The front:** the non-dominated part of the final N survivors: the elites (NSGA-II) or the parents (NSGA-III) that PyGAD selects from the last population after the last generation.

**Parent selection:** the docs list `nsga2`, `tournament_nsga2`, `nsga3` and `tournament_nsga3`, with no stated preference, and PyGAD's ZDT example uses `nsga2`, which takes the best N by front and crowding distance as parents. The matched NSGA-II's parents come from a crowded binary tournament, which is `tournament_nsga2`; NSGA-III's survivors are `nsga3`, as in PyGAD's DTLZ2 example ([example_dtlz2.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_dtlz2.py)). In matched mode the matched algorithm decides (rule 6.1).

**Not run:** SPEA2, SMS-EMOA and MOEA/D: PyGAD has none of them. Its `tournament_nsga3` parent selection is a variant of NSGA-III's mating, not another algorithm.

**Separate tests** (as above; hypervolume with `run.py`'s reference points; rule 5.3 stops a solver after 3 seeds that all reached the cap):

ZDT1 (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 3 | 0.4875 (0.5261, 0.4731) | 2,396 | 3 |
| nsga3 | 3 | 0.6538 (0.6850, 0.5757) | 3,800 | 3 |

ZDT2 (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 3 | 0.1517 (0.1559, 0.0997) | 1,997 | 3 |
| nsga3 | 3 | 0.1525 (0.2080, 0.1316) | 3,600 | 3 |

ZDT3 (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 3 | 0.6005 (0.7153, 0.5309) | 2,393 | 3 |
| nsga3 | 3 | 0.8357 (0.9190, 0.5910) | 3,800 | 3 |

DTLZ2, 3 objectives (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 3 | 0.1336 (0.1770, 0.0530) | 3,120 | 3 |
| nsga3 | 3 | 0.0949 (0.1091, 0.0690) | 4,416 | 3 |

DTLZ1, 3 objectives (budget 40,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 3 | 0.0000 (0.0000, 0.0000) | 5,691 | 3 |
| nsga3 | 3 | 0.0000 (0.0000, 0.0000) | 7,084 | 3 |

PyGAD's multi-objective runs are slow per evaluation (see above), and every one reached the 60 s cap. Without the cap (seeds 0 to 2, a 3,000 s cap), for comparison: NSGA-II used its budget in 404 to 567 s, with median hypervolumes of 0.6886 (ZDT1), 0.2755 (ZDT2), 0.8424 (ZDT3), 0.0780 (DTLZ2) and 0 (DTLZ1); NSGA-III in 240 to 385 s, with 0.7887, 0.3335, 0.9704, 0.1177 and 0. The DTLZ1 fronts stay far above its reference point. With the unbiased SBX the adapter wrote before these runs used PyGAD's own, the same NSGA-II reached 0.8676 on ZDT1 and 0.6874 on DTLZ2 (medians of 3, full budget).

## Can't run

PyGAD runs every scenario. In the multi-objective scenarios it runs NSGA-II and NSGA-III, not SPEA2, SMS-EMOA or MOEA/D, which it doesn't have.

## Bugs found

| Bug | Effect | Worked around | Reported |
|---|---|---|---|
| `sbx` makes one child per pair, always `0.5 * ((y1 + y2) - beta_q * (y2 - y1))` with `beta_q > 0` ([utils/crossover.py, line 329](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/crossover.py#L329)): the child below the parents' midpoint, which pulls every crossed gene towards the lower bound | the continuous runs above: Ackley 30 and Rosenbrock 10 fail; the multi-objective runs above: the fronts crowd towards f1 = 0 on ZDT and stay far from DTLZ's | no | [ahmedfgad/GeneticAlgorithmPython#369](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/369) |
| `two_points_crossover` draws only the first point; the second is always the first plus n / 2 ([utils/crossover.py, lines 122-127](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/crossover.py#L122-L127)), so a child always takes exactly n / 2 consecutive genes from its second parent. The docs say it "selects the 2 points randomly" ([utils.md](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/docs/source/utils.md)) | the matched OneMax runs above | no | not yet |
| With `allow_duplicate_genes=False` and a gene space of n values for n genes, as in PyGAD's permutation example, the random mutation never changes a solution: it picks a value from the space that isn't already in the solution (`select_unique_value`, [helper/unique.py, lines 269-280](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/helper/unique.py#L269-L280)), and in a permutation there's none, so the gene keeps its value. In 1,000 mutations of random permutations of 8, none changed | the N-Queens runs above: the population converges to one board, and the runs reach the time cap after about 200 evaluations | no | not yet |
| The docs describe `swap_mutation` as swapping "2 randomly selected genes" ([utils.md](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/docs/source/utils.md)), but it swaps a gene of the first half with the gene half the length after it ([utils/mutation.py, lines 363-387](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/mutation.py#L363-L387)) | none here: the adapter doesn't use it | - | not yet |
