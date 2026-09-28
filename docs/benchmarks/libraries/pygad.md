# PyGAD (Python, 3.7.0)

PyGAD is a genetic-algorithm library: one class, `pygad.GA`, whose parameters choose the parent selection, crossover, mutation, elitism and callbacks; its NSGA-II and NSGA-III parent selections handle several objectives. Its docs are [pygad.readthedocs.io](https://pygad.readthedocs.io/en/latest/), from `docs/source/` of [ahmedfgad/GeneticAlgorithmPython](https://github.com/ahmedfgad/GeneticAlgorithmPython) (tag [3.7.0](https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/3.7.0)). Version 3.7.0 has benchmark problems (`pygad.benchmarks`) with "a runnable example per benchmark" in [examples/benchmarks/](https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/3.7.0/examples/benchmarks) ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html)); the idiomatic runs take their settings.

Adapter: [benchmarks/adapters/pygad/](../../../benchmarks/adapters/pygad/).
Know a better way to solve one of these problems with PyGAD? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs PyGAD

- **Fitness functions:** numpy functions of a batch, one solution per row ([bench.py, lines 114-171](../../../benchmarks/adapters/pygad/bench.py#L114-L171)), through `fitness_batch_size` ([Batch Fitness Calculation](https://pygad.readthedocs.io/en/latest/fitness_calculation.html#batch-fitness-calculation), rule 3.4). PyGAD picks the solutions to evaluate as without it (`cal_pop_fitness`, `utils/engine.py`); runs with and without it give the same evaluations and best values. PyGAD maximizes, so a minimized problem is negated.
- **Evaluations:** each row counts ([`Budget.count`, lines 65-70](../../../benchmarks/adapters/pygad/bench.py#L65-L70)), with the first hit ([`Budget.keep`, lines 72-85](../../../benchmarks/adapters/pygad/bench.py#L72-L85)). PyGAD doesn't evaluate a solution identical to a previous elite or kept parent, so generations differ in size; runs print `last_generation`.
- **Stop:** `on_generation` returns `"stop"` at the target, the budget or the time cap; `num_generations` is 10⁷.
- **Keeping going (rule 2.2):** the examples' `num_generations` is lifted. PyGAD's `stop_criteria="saturate_N"` is off by default and no example sets it. After 10 generations in a row without an evaluation, `on_generation` ends the attempt, and a new `pygad.GA` starts from a new random population with the next restart seed ([`run_single`, lines 234-291](../../../benchmarks/adapters/pygad/bench.py#L234-L291)); the run prints `restarts`.
- **Bounds (rule 2.4):** per section.
- **Time:** from before the `pygad.GA` constructor, which creates the initial population.
- **One thread:** numpy's BLAS set to one thread before import ([lines 18-20](../../../benchmarks/adapters/pygad/bench.py#L18-L20)); `parallel_processing=None`.
- **Seeds:** `random_seed=seed` (numpy's and Python's generators).
- **Solutions:** the best evaluated.
- **Rule 5.3:** in the scenarios with a target ([lines 359-360](../../../benchmarks/adapters/pygad/bench.py#L359-L360)).
- **Separate tests:** 2026-09-25, PyGAD 3.7.0, numpy 2.5.3, Python 3.13.9, seeds 0 to 4, the scenario's budget, 60 s cap, with other processes on the machine. `outside` was 0 in every run.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched** ([lines 182-199](../../../benchmarks/adapters/pygad/bench.py#L182-L199)): 300 solutions, all parents (`num_parents_mating` 300), `parent_selection_type="tournament"` with `K_tournament` 3, `keep_elitism` 0 and `keep_parents` 0, `crossover_type="two_points"` with `crossover_probability` 0.5, `mutation_type="random"` with `mutation_probability` 0.2 / n. The genes are PyGAD's binary genes, `gene_space=[0, 1]`, `gene_type=int` ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html), "Knapsack"); a mutated gene takes the other value (`generate_gene_value_from_space`, `helper/misc.py`).
- **Idiomatic (OneMax 100):** [example_knapsack.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_knapsack.py), the binary example: `gene_space=[0, 1]`, `gene_type=int`, 30 solutions, 10 parents, and the defaults ([`pygad.GA`](https://pygad.readthedocs.io/en/latest/pygad.html)): steady-state selection (`"sss"`), single-point crossover, random mutation of 10% of the genes, `keep_elitism` 1 ([line 204](../../../benchmarks/adapters/pygad/bench.py#L204)).

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

**Methods:** `ga`, [example_tsp.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_tsp.py), the permutation example ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html): `gene_space=list(range(num_cities))`, `gene_type=int`, `allow_duplicate_genes=False` "keep the permutation constraint"): 30 solutions, 10 parents, the defaults otherwise ([lines 207-216](../../../benchmarks/adapters/pygad/bench.py#L207-L216)). As `pygad/benchmarks/tsp.py` does, a non-permutation scores worse than any permutation ([lines 252-270](../../../benchmarks/adapters/pygad/bench.py#L252-L270)).

**Keeping going:** to the target, the budget or the time cap; an attempt that stalls restarts.

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

The random mutation can't change a permutation (see [Bugs found](#bugs-found)), so the population converges to one board and stops evaluating. These tests ran before the stall restart: about 200 evaluations in 60 s. With PyGAD's other mutation types (not benchmarked): `"swap"` reached N-Queens 32 in 5 of 5 runs (median 32,462 evaluations) and 64 in 5 of 5 (90,251); `"inversion"` 1 of 5 and 0 of 3; `"scramble"` 0 of 5 and 0 of 3.

## Continuous: Rastrigin 10 and 30, Ackley 30 (multimodal), Rosenbrock 10 (unimodal)

**Methods:** `ga`, [example_classic_rastrigin.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_rastrigin.py) and [example_classic_ackley.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_ackley.py) (also the docs' "Example: SOO"): 40 solutions, 10 parents, `crossover_type="sbx"` with `sbx_crossover_eta` 20, `mutation_type="polynomial"` with `polynomial_mutation_eta` 20 (1 / n per gene, the default), `init_range_low` and `init_range_high` at the bounds, the defaults otherwise (steady-state selection, `keep_elitism` 1). [example_classic_rosenbrock.py](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/examples/benchmarks/example_classic_rosenbrock.py): `sbx_crossover_eta` 30 ([lines 218-231](../../../benchmarks/adapters/pygad/bench.py#L218-L231)).

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

## Can't run

Nothing: PyGAD runs all 9 scenarios.

## Bugs found

| Bug | Effect | Worked around | Reported |
|---|---|---|---|
| `sbx` makes one child per pair, always `0.5 * ((y1 + y2) - beta_q * (y2 - y1))` with `beta_q > 0` ([utils/crossover.py, line 329](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/crossover.py#L329)): the child below the parents' midpoint, which pulls every crossed gene towards the lower bound | the continuous runs above (Ackley 30 and Rosenbrock 10 don't reach the target) | no | [ahmedfgad/GeneticAlgorithmPython#369](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/369) |
| `two_points_crossover` draws only the first point; the second is always the first plus n / 2 ([utils/crossover.py, lines 122-127](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/crossover.py#L122-L127)), so a child always takes exactly n / 2 consecutive genes from its second parent. The docs say it "selects the 2 points randomly" ([utils.md](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/docs/source/utils.md)) | the matched OneMax runs above | no | not yet |
| With `allow_duplicate_genes=False` and a gene space of n values for n genes, as in PyGAD's permutation example, the random mutation never changes a solution: it picks a value from the space that isn't already in the solution (`select_unique_value`, [helper/unique.py, lines 269-280](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/helper/unique.py#L269-L280)), and in a permutation there's none, so the gene keeps its value. In 1,000 mutations of random permutations of 8, none changed | the N-Queens runs above: the population converges to one board and stops evaluating, until the stall restart | no | not yet |
| The docs describe `swap_mutation` as swapping "2 randomly selected genes" ([utils.md](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/docs/source/utils.md)), but it swaps a gene of the first half with the gene half the length after it ([utils/mutation.py, lines 363-387](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/mutation.py#L363-L387)) | none here: the adapter doesn't use it | - | not yet |
