# PyGAD (Python, 3.7.0)

PyGAD is a genetic-algorithm library: one class, `pygad.GA`, whose parameters choose the parent selection, crossover, mutation, elitism and callbacks. Its docs are [pygad.readthedocs.io](https://pygad.readthedocs.io/en/latest/), from `docs/source/` of [ahmedfgad/GeneticAlgorithmPython](https://github.com/ahmedfgad/GeneticAlgorithmPython) (tag [3.7.0](https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/3.7.0)).

In the matched suite ([rule 6](../rules.md#6-the-methods)), PyGAD runs OneMax 1000 with its GA, set to the GA of [rule 6.2](../rules.md#6-the-methods). It has no differential evolution and no CMA-ES, so it doesn't run Rastrigin or Rosenbrock.

Adapter: [benchmarks/adapters/pygad/](../../../benchmarks/adapters/pygad/).
Found a setting that brings PyGAD closer to the definition, or a difference this page misses? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs PyGAD

- **Fitness function:** numpy, of a batch, one solution per row ([bench.py, lines 94-104](../../../benchmarks/adapters/pygad/bench.py#L94-L104)), through `fitness_batch_size` ([Batch Fitness Calculation](https://pygad.readthedocs.io/en/latest/fitness_calculation.html#batch-fitness-calculation), rule 3.4). PyGAD picks the solutions to evaluate as without it (`cal_pop_fitness`, `utils/engine.py`); runs with and without it give the same evaluations and best values.
- **Evaluations:** each row counts ([`Budget.count`](../../../benchmarks/adapters/pygad/bench.py#L59-L61)), with the first hit ([`Budget.keep`](../../../benchmarks/adapters/pygad/bench.py#L63-L73)). PyGAD doesn't evaluate a solution identical to a previous elite or kept parent, so generations can differ in size; runs print `last_generation`.
- **Stop:** `on_generation` returns `"stop"` at the target, the budget or the time cap; `num_generations` is 10⁷. PyGAD's `stop_criteria` is off by default.
- **Stalls (rule 2.2):** after 10 generations in a row without an evaluation, `on_generation` ends the attempt, and a new `pygad.GA` starts from a new random population with the next restart seed ([`run_ga`, lines 139-176](../../../benchmarks/adapters/pygad/bench.py#L139-L176)); the run prints `restarts`.
- **Time:** from before the `pygad.GA` constructor, which creates the initial population.
- **One thread:** numpy's BLAS set to one thread before import ([lines 21-22](../../../benchmarks/adapters/pygad/bench.py#L21-L22)); `parallel_processing=None`.
- **Seeds:** `random_seed=seed` (numpy's and Python's generators).
- **Rule 5.3:** applied by the adapter too ([`main`](../../../benchmarks/adapters/pygad/bench.py#L197-L239)).
- **Separate tests:** 2026-09-25, PyGAD 3.7.0, numpy 2.5.3, Python 3.13.9, seeds 0 to 4, the scenario's budget, 60 s cap.

## OneMax 1000: the GA

**Configuration** ([`onemax_ga`, lines 111-127](../../../benchmarks/adapters/pygad/bench.py#L111-L127)), against [rule 6.2](../rules.md#6-the-methods):

| Definition | PyGAD |
|---|---|
| 300 individuals, uniform random bits | `sol_per_pop=300`, with PyGAD's binary genes, `gene_space=[0, 1]`, `gene_type=int` ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html), "Knapsack") |
| 300 tournaments of 3 | `num_parents_mating=300`, `parent_selection_type="tournament"`, `K_tournament=3` |
| two-point crossover of each consecutive pair, probability 0.5 | `crossover_type="two_points"`, `crossover_probability=0.5` |
| each child mutated with probability 0.2, each bit at 1/1000 | `mutation_type="random"`, `mutation_probability=0.2 / 1000` per gene; a mutated gene takes the other value of its space (`generate_gene_value_from_space`, `helper/misc.py`), a bit flip |
| generational, no elitism | `keep_elitism=0`, `keep_parents=0` |

**Differences** (rule 6.1):
- `crossover_probability` makes each parent eligible with that probability and crosses it with the next eligible one, one child per crossover, not two per pair.
- `mutation_probability` is per gene, with none per child: each gene flips with probability 0.2 / n, the definition's mean of 0.2 bits per child, spread over more children.
- `two_points_crossover` draws only the first cut point; the second is always n / 2 after it (a bug, below, included by rule 8.4).
- With `keep_parents` 0, every child is evaluated, also one that is a copy of its parent.

**Separate tests:** 5 of 5 reached the target, first hits at a median of 113,594 evaluations.

**Speed** (checked 2026-09-29, PyGAD 3.7.0, WSL with other processes running): seeds 0 to 2 took 7.4, 7.6 and 7.1 s, for 114,000, 116,100 and 107,700 evaluations. A profile of seed 0 puts nearly all of it in PyGAD's own operators, which loop in Python: `mutation_probs_by_space` draws a probability per gene and checks every gene of every child (about half of the run), `two_points_crossover` draws the parents' probabilities anew for each child (a sixth), `tournament_selection` (a sixth), and the initial population, built gene by gene from `gene_space` (a tenth). The batch fitness function and the adapter's counter take about 2%. PyGAD documents no faster way to run this GA:
- **Batch fitness:** already on, `fitness_batch_size` of the whole population, as [Batch Fitness Calculation](https://pygad.readthedocs.io/en/latest/fitness_calculation.html#batch-fitness-calculation) shows.
- **No extra work to turn off:** `save_solutions` and `save_best_solutions` are off (their defaults), there are no `on_parents`, `on_crossover` or `on_mutation` callbacks, `parallel_processing` is off (rule 4.3), and PyGAD logs nothing per generation. The adapter's callbacks, `on_fitness` and `on_generation`, only read counters.
- **Binary genes:** `gene_space=[0, 1]` is PyGAD's documented binary gene ([Benchmark Problems](https://pygad.readthedocs.io/en/latest/benchmarks.html), "Knapsack"). Without it, `mutation_type="random"` adds a random value from a range to the gene, or replaces the gene with one (`mutation_by_replacement`), which can leave it unchanged or outside {0, 1}: not a bit flip, another algorithm. `mutation_num_genes` instead of `mutation_probability` changes a fixed number of genes per child, not 0.2 on average.
- **A user-built initial population** (`initial_population`) would skip PyGAD's gene-by-gene initialization, but it's the adapter's code instead of the library's (rule 6.1), so it isn't used.

So the adapter is unchanged.

## Can't run

- Rastrigin 30 (DE/rand/1/bin) and Rosenbrock 10 (CMA-ES): PyGAD has only a GA.

## Bugs found

| Bug | Effect | Worked around | Reported |
|---|---|---|---|
| `two_points_crossover` draws only the first point; the second is always the first plus n / 2 ([utils/crossover.py, lines 122-127](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/pygad/utils/crossover.py#L122-L127)), so a child always takes exactly n / 2 consecutive genes from its second parent. The docs say it "selects the 2 points randomly" ([utils.md](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/3.7.0/docs/source/utils.md)) | the OneMax runs | no | [ahmedfgad/GeneticAlgorithmPython#370](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/370), fix proposed in [ahmedfgad/GeneticAlgorithmPython#371](https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/371) |

Found in methods no longer in the suite: `sbx` always makes the child below the parents' midpoint ([ahmedfgad/GeneticAlgorithmPython#369](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/369), fix proposed in [#376](https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/376)); with `allow_duplicate_genes=False`, as in PyGAD's permutation example, the random mutation never changes a permutation ([#372](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/372), fix proposed in [#373](https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/373)); the docs say `swap_mutation` swaps 2 random genes, but it swaps a gene with the one half the length after it ([#374](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/374), fix proposed in [#375](https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/375)).
