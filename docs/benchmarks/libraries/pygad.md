# PyGAD (Python, 3.7.0)

PyGAD is a genetic-algorithm library: one class, `pygad.GA`, whose parameters choose the parent selection, crossover, mutation, elitism and callbacks; its NSGA-II and NSGA-III parent selections handle several objectives. Its docs are [pygad.readthedocs.io](https://pygad.readthedocs.io/en/latest/), from `docs/source/` of [ahmedfgad/GeneticAlgorithmPython](https://github.com/ahmedfgad/GeneticAlgorithmPython) (tag [3.7.0](https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/3.7.0)). Version 3.7.0 has benchmark problems (`pygad.benchmarks`) with "a runnable example per benchmark" in [examples/benchmarks/](https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/3.7.0/examples/benchmarks) ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html)); the idiomatic runs take their settings.

Adapter: [benchmarks/adapters/pygad/](../../../benchmarks/adapters/pygad/).
Know a better way to solve one of these problems with PyGAD? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs PyGAD

- **Fitness functions:** numpy functions of a batch, one solution per row ([bench.py, lines 115-231](../../../benchmarks/adapters/pygad/bench.py#L115-L231)), through `fitness_batch_size` ([Batch Fitness Calculation](https://pygad.readthedocs.io/en/latest/fitness_calculation.html#batch-fitness-calculation), rule 3.4). PyGAD picks the solutions to evaluate as without it (`cal_pop_fitness`, `utils/engine.py`); runs with and without it give the same evaluations and best values. PyGAD maximizes, so a minimized problem is negated.
- **Evaluations:** each row counts ([`Budget.count`, lines 66-71](../../../benchmarks/adapters/pygad/bench.py#L66-L71)), with the first hit ([`Budget.keep`, lines 73-86](../../../benchmarks/adapters/pygad/bench.py#L73-L86)). PyGAD doesn't evaluate a solution identical to a previous elite or kept parent, so generations differ in size; runs print `last_generation`.
- **Stop:** `on_generation` returns `"stop"` at the target, the budget or the time cap; `num_generations` is 10⁷.
- **Keeping going (rule 2.2):** the examples' `num_generations` is lifted. PyGAD's `stop_criteria="saturate_N"` is off by default and no example sets it.
- **Bounds (rule 2.4):** per section.
- **Time:** from before the `pygad.GA` constructor, which creates the initial population; the front is extracted after the clock ([line 504](../../../benchmarks/adapters/pygad/bench.py#L504)).
- **One thread:** numpy's BLAS set to one thread before import ([lines 18-20](../../../benchmarks/adapters/pygad/bench.py#L18-L20)); `parallel_processing=None`.
- **Seeds:** `random_seed=seed` (numpy's and Python's generators).
- **Solutions:** the best evaluated; a multi-objective run prints the non-dominated part of its final survivors ([`front_of`, lines 425-436](../../../benchmarks/adapters/pygad/bench.py#L425-L436)).
- **Rule 5.3:** [line 519](../../../benchmarks/adapters/pygad/bench.py#L519).
- **Separate tests:** 2026-09-25, PyGAD 3.7.0, numpy 2.5.3, Python 3.13.9, seeds 0 to 4, the scenario's budget, 60 s cap, with other processes on the machine. `outside` was 0 in every run.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched** ([lines 242-259](../../../benchmarks/adapters/pygad/bench.py#L242-L259)): 300 solutions, all parents (`num_parents_mating` 300), `parent_selection_type="tournament"` with `K_tournament` 3, `keep_elitism` 0 and `keep_parents` 0, `crossover_type="two_points"` with `crossover_probability` 0.5, `mutation_type="random"` with `mutation_probability` 0.2 / n. The genes are PyGAD's binary genes, `gene_space=[0, 1]`, `gene_type=int` ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html), "Knapsack"); a mutated gene takes the other value (`generate_gene_value_from_space`, `helper/misc.py`).
- **Idiomatic (OneMax 100):** [example_knapsack.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_knapsack.py), the binary example: `gene_space=[0, 1]`, `gene_type=int`, 30 solutions, 10 parents, and the defaults ([`pygad.GA`](https://pygad.readthedocs.io/en/latest/pygad.html)): steady-state selection (`"sss"`), single-point crossover, random mutation of 10% of the genes, `keep_elitism` 1 ([line 264](../../../benchmarks/adapters/pygad/bench.py#L264)).

**Matched differences:**
- `two_points_crossover` always takes exactly n / 2 consecutive genes from the second parent (a bug, see [Bugs found](#bugs-found)).
- `crossover_probability` makes each parent eligible with that probability, and crosses two eligible parents; with 300 parents every child is crossed (the matched GA: each pair at 0.5).
- One child per crossover, not two.
- `mutation_probability` is per gene, with none per individual: each gene flips with probability 0.2 / n, the matched mean of 0.2 bits per child.
- With `keep_parents` 0, every child is evaluated.

**Keeping going:** to the target or the budget.

**Left out:** adaptive mutation ([Adaptive Mutation](https://pygad.readthedocs.io/en/latest/adaptive_mutation.html)): its rates (`mutation_num_genes=(3, 1)` for 6 genes, and syntax examples) aren't settings for this problem type.

**Separate tests:**

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

The idiomatic GA flips 10 of the 100 bits of every child (the default 10%).

## Permutation: N-Queens 32 and 64

**Methods:** `ga`, [example_tsp.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_tsp.py), the permutation example ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html): `gene_space=list(range(num_cities))`, `gene_type=int`, `allow_duplicate_genes=False` "keep the permutation constraint"): 30 solutions, 10 parents, the defaults otherwise ([lines 267-276](../../../benchmarks/adapters/pygad/bench.py#L267-L276)). As `pygad/benchmarks/tsp.py` does, a non-permutation scores worse than any permutation ([lines 297-315](../../../benchmarks/adapters/pygad/bench.py#L297-L315)).

**Keeping going:** to the target, the budget or the time cap.

**Left out:**
- The swap, inversion and scramble mutations: the example uses the default, random mutation.
- The community notebook [example_travelling_salesman.ipynb](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/example_travelling_salesman.ipynb) (PyGAD 2.17, its own PMX and inversion): the 3.7.0 benchmark example is the current one.

**Separate tests:**

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 3 | 0 | - | 7 (7, 7) | 3 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 3 | 0 | - | 18 (15, 18) | 3 |

The random mutation can't change a permutation (see [Bugs found](#bugs-found)), so the population converges to one board, and the unevaluated copies leave about 200 evaluations in 60 s. With PyGAD's other mutation types (not benchmarked): `"swap"` reached N-Queens 32 in 5 of 5 runs (median 32,462 evaluations) and 64 in 5 of 5 (90,251); `"inversion"` 1 of 5 and 0 of 3; `"scramble"` 0 of 5 and 0 of 3.

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Methods:** `ga`, [example_classic_rastrigin.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_rastrigin.py) and [example_classic_ackley.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_ackley.py) (also the docs' "Example: SOO"): 40 solutions, 10 parents, `crossover_type="sbx"` with `sbx_crossover_eta` 20, `mutation_type="polynomial"` with `polynomial_mutation_eta` 20 (1 / n per gene, the default), `init_range_low` and `init_range_high` at the bounds, the defaults otherwise (steady-state selection, `keep_elitism` 1). [example_classic_rosenbrock.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_rosenbrock.py): `sbx_crossover_eta` 30 ([lines 278-291](../../../benchmarks/adapters/pygad/bench.py#L278-L291)).

**Bounds:** `sbx` and `polynomial` clip each gene to its initial range (`get_initial_population_range`), the box.

**Keeping going:** to the target or the budget.

**Left out:** the other crossover and mutation types: the examples don't use them.

**Separate tests** (see the `sbx` bug in [Bugs found](#bugs-found)):

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

With a 900 s cap (seeds 0 to 2): Rastrigin 30 in 3 of 3 runs (median first hit 434,052 evaluations); Ackley 30 in none, at 16.7 to 19.9.

## Multi-objective (matched): ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

PyGAD's NSGA-II and NSGA-III are parent selections ([Multi-Objective Optimization](https://pygad.readthedocs.io/en/latest/multi_objective.html)). A generation selects parents, crosses and mutates them, and the next population is either the `keep_elitism` best of the current one by PyGAD's NSGA-II sort followed by the offspring, or, with `keep_parents=-1`, the parents followed by the offspring (`run`, `utils/engine.py`). Each algorithm's survival is built from these, with N = 100 (92 with 3 objectives) and a PyGAD population of 2N ([`run_front`, lines 387-422](../../../benchmarks/adapters/pygad/bench.py#L387-L422)).

**Methods:**
- **`nsga2`:** `keep_elitism` N and N offspring, so the elites are the best N of elites and offspring by the NSGA-II sort. Parents: PyGAD's crowded binary tournament (`"tournament_nsga2"`, `K_tournament` 2), drawn from all 2N, since PyGAD selects parents before elites.
- **`nsga3`** (every problem): `parent_selection_type="nsga3"`, `nsga3_num_divisions` 99 (100 reference points) or 12 (91), N parents, `keep_elitism` 0, `keep_parents` -1: the parents are NSGA-III's N survivors of 2N, and the next population is them and their N offspring.
- **Operators**, PyGAD's own, bugs included (rule 6.1): `crossover_type="sbx"`, η 15 with `crossover_probability` 0.9 (NSGA-II), η 30 with every child crossed (NSGA-III); `"polynomial"` mutation, Deb's bounded one, η 20, `mutation_probability` 1 / n.
- **Parent selection:** the docs list `nsga2`, `tournament_nsga2`, `nsga3` and `tournament_nsga3` without preference; the ZDT example uses `nsga2`. In matched mode the matched algorithm decides: a crowded binary tournament for NSGA-II; `nsga3`, as in [example_dtlz2.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_dtlz2.py), for NSGA-III.

**Differences:**
- `sbx` makes one child, always below the parents' midpoint (see [Bugs found](#bugs-found)), and crosses every gene, not each at 0.5.
- `crossover_probability` makes parents eligible at 0.9, not pairs. Without it (NSGA-III), child k comes from parents k and k + 1, in NSGA-III's order: by front.
- The initial population is 2N, so the first generation costs N more evaluations.
- The crowding distance normalizes by the whole population's range, not the front's.
- A child identical to a previous elite or parent isn't evaluated.
- The non-dominated sorting compares every pair in Python, two or three times per generation; the 60 s cap ends runs before the budget.

**Bounds:** `sbx` and polynomial mutation clip to [0, 1].

**Keeping going:** to the budget or the time cap.

**The front:** the non-dominated part of the final N survivors (the elites, or NSGA-III's parents).

**Left out:** SPEA2, SMS-EMOA and MOEA/D: PyGAD has none. `tournament_nsga3` is a variant of NSGA-III's mating, not another algorithm.

**Separate tests:**

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

With a 3,000 s cap (seeds 0 to 2): NSGA-II used its budget in 404 to 567 s, with median hypervolumes of 0.6886 (ZDT1), 0.2755 (ZDT2), 0.8424 (ZDT3), 0.0780 (DTLZ2) and 0 (DTLZ1); NSGA-III in 240 to 385 s, with 0.7887, 0.3335, 0.9704, 0.1177 and 0.

## Can't run

SPEA2, SMS-EMOA and MOEA/D in the multi-objective scenarios: PyGAD doesn't have them. It runs every other scenario.

## Bugs found

| Bug | Effect | Worked around | Reported |
|---|---|---|---|
| `sbx` makes one child per pair, always `0.5 * ((y1 + y2) - beta_q * (y2 - y1))` with `beta_q > 0` ([utils/crossover.py, line 329](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/crossover.py#L329)): the child below the parents' midpoint, which pulls every crossed gene towards the lower bound | the continuous runs above (Ackley 30 and Rosenbrock 10 don't reach the target) and the multi-objective runs: on ZDT the first variable is pulled too, so the fronts crowd towards f1 = 0; on DTLZ the distance variables stay away from their optimum, 0.5. With an unbiased SBX, the same NSGA-II reached 0.8676 on ZDT1 and 0.6874 on DTLZ2 (medians of 3, full budget) | no | [ahmedfgad/GeneticAlgorithmPython#369](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/369) |
| `two_points_crossover` draws only the first point; the second is always the first plus n / 2 ([utils/crossover.py, lines 122-127](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/crossover.py#L122-L127)), so a child always takes exactly n / 2 consecutive genes from its second parent. The docs say it "selects the 2 points randomly" ([utils.md](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/docs/source/utils.md)) | the matched OneMax runs above | no | not yet |
| With `allow_duplicate_genes=False` and a gene space of n values for n genes, as in PyGAD's permutation example, the random mutation never changes a solution: it picks a value from the space that isn't already in the solution (`select_unique_value`, [helper/unique.py, lines 269-280](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/helper/unique.py#L269-L280)), and in a permutation there's none, so the gene keeps its value. In 1,000 mutations of random permutations of 8, none changed | the N-Queens runs above: the population converges to one board, and the runs reach the time cap after about 200 evaluations | no | not yet |
| The docs describe `swap_mutation` as swapping "2 randomly selected genes" ([utils.md](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/docs/source/utils.md)), but it swaps a gene of the first half with the gene half the length after it ([utils/mutation.py, lines 363-387](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/mutation.py#L363-L387)) | none here: the adapter doesn't use it | - | not yet |
