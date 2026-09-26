# Nevergrad (Python, 1.0.12)

Nevergrad is Meta's platform for derivative-free optimization: many optimizers behind one ask and tell interface, and "wizards" that choose among them from the budget, the number of workers and the parametrization. Its docs are at [facebookresearch.github.io/nevergrad](https://facebookresearch.github.io/nevergrad/), with the choice of optimizer in [How to perform optimization](https://facebookresearch.github.io/nevergrad/optimization.html), and the [README](https://github.com/facebookresearch/nevergrad).

Adapter: [benchmarks/adapters/nevergrad/](../../../benchmarks/adapters/nevergrad/).
Know a better way to solve one of these problems with Nevergrad? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs Nevergrad

- **The loop** ([`run`](../../../benchmarks/adapters/nevergrad/bench.py#L219-L248)): the docs' [ask and tell loop](https://facebookresearch.github.io/nevergrad/optimization.html#ask-and-tell-interface), one worker, the optimizer given the scenario's budget ("the budget tells NgIohTuned which algorithm to choose"). Every optimizer runs with its defaults.
- **Stop:** right after the evaluation that reaches the target, or before one past the budget or the time cap.
- **Keeping going (rule 2.2):** the optimizers have no stop criterion. The CMA-ES inside `CMA` and `NgIohTuned` restarts from the middle of the domain, with the same population size, when pycma's criteria end it (`optimizerlib.py`, `_CMA.es`).
- **Imports (rule 4.2):** pycma with matplotlib, scikit-learn and SciPy's COBYLA, which Nevergrad imports during its first run (about 0.4 s), are imported before the clock ([the imports](../../../benchmarks/adapters/nevergrad/bench.py#L46-L53)).
- **Seeds:** the docs' two ways ([Reproducibility](https://facebookresearch.github.io/nevergrad/optimization.html#reproducibility)): numpy's global random state and the parametrization's `random_state`.
- **Rule 5.3:** applied in the adapter too.
- **Choosing among the docs (rule 6.2):** [Choosing an optimizer](https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer) lists "rules of thumb", each for its case. The first is the default: "`NgIohTuned` is 'meta'-optimizer which adapts to the provided settings ... and should therefore be a good default". It runs on every problem; the other methods come from the list's cases or the docs' example for the problem type.
- **Separate tests:** 2026-09-25, Nevergrad 1.0.12, 5 seeds, the scenario's budget, 60 s cap, rule 5.3. Other tests ran at the same time, and Nevergrad ran several times slower than in a benchmark run, so capped runs stopped after fewer evaluations.

## Binary: OneMax 100 (idiomatic)

**Methods** ([`solvers`](../../../benchmarks/adapters/nevergrad/bench.py#L169-L216)), on the docs' OneMax parameter, `ng.p.TransitionChoice(range(2), repetitions=100)` ([Basic example](https://facebookresearch.github.io/nevergrad/optimization.html#basic-example)):
- **`ngiohtuned`:** the default; here it chooses `DoubleFastGADiscreteOnePlusOne`.
- **`discrete_one_plus_one`:** `DiscreteOnePlusOne`, the docs' OneMax example.
- **`portfolio_discrete_one_plus_one`:** "`PortfolioDiscreteOnePlusOne` is excellent in discrete settings of mixed settings when high precision on parameters is not relevant".

**Keeping going:** no stop criterion.

**Left out:**
- `NGOpt`, the README's example wizard: `NgIohTuned` is the list's default, and one wizard runs.
- `TwoPointsDE` on a `Choice`: suggested for hyperparameters "if you have more budget", not for a bit string.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| OneMax 100 (200k) | ngiohtuned | 5 | 5 | 3,553 | 100 | 100 | 100 | 0 |
| OneMax 100 (200k) | discrete_one_plus_one | 5 | 5 | 3,209 | 100 | 100 | 100 | 0 |
| OneMax 100 (200k) | portfolio_discrete_one_plus_one | 3 | 0 | | 80 | 81 | 78 | 3 |

`PortfolioDiscreteOnePlusOne` reached the cap after about 9,000 evaluations.

## Permutation: N-Queens 32 and 64

Nevergrad has no permutation parameter. The docs' [Example with permutation](https://facebookresearch.github.io/nevergrad/optimization.html#example-with-permutation) links to [How to optimize permutations with Nevergrad](https://docs.google.com/document/d/1B5yVOx1H1nnjY3EOf14487hAr8CzwJ9zEkDwQnZ5nbE/edit): a real `ng.p.Array(shape=(n,))` whose `argsort` is the permutation, as in Nevergrad's TSP benchmark (`nevergrad/functions/stsp/core.py`). The adapter does the same and prints the permutation.

**Methods:**
- **`ngiohtuned`:** the default; here it chooses a metamodel over `VLPCMA` (CMA-ES with a large population).
- **`rotated_two_points_de`, `genetic_de`:** the two optimizers the permutation page names: they "will cut and paste part of the permutation into other parts".

**Keeping going:** no stop criterion; `NgIohTuned`'s CMA-ES restarts.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| N-Queens 32 (500k) | ngiohtuned | 3 | 0 | | 6 | 6 | 7 | 3 |
| N-Queens 32 (500k) | rotated_two_points_de | 3 | 0 | | 6 | 5 | 6 | 3 |
| N-Queens 32 (500k) | genetic_de | 3 | 0 | | 3 | 3 | 5 | 3 |
| N-Queens 64 (1M) | ngiohtuned | 3 | 0 | | 20 | 20 | 21 | 3 |
| N-Queens 64 (1M) | rotated_two_points_de | 3 | 0 | | 18 | 17 | 18 | 3 |
| N-Queens 64 (1M) | genetic_de | 3 | 0 | | 16 | 15 | 16 | 3 |

The runs reached the cap after about 25,000 evaluations (`NgIohTuned`) and 48,000 to 77,000 (the DEs).

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

The parameter is a bounded `ng.p.Array(shape=(n,), lower=..., upper=...)`, the docs' way to give bounds ("sigma will be adapted so that the range spans 6 sigma"), starting at the middle.

**Methods:**
- **`ngiohtuned`:** the default; here a metamodel over `VLPCMA`, which fits a quadratic model to the best points every few dozen evaluations (about 1 ms or more per evaluation).
- **`scr_hammersley`:** "`ScrHammersleySearchPlusMiddlePoint` is excellent for super parallel cases ... or for very multimodal cases", the list's only case for multimodal functions.
- **`one_plus_one`:** with one worker and a budget over 1000 times the dimension, two cases match: "`OnePlusOne` is a simple robust method for continuous parameters with `num_workers` < 8" and "`CMA` is excellent for control ... when the budget is large". The first listed runs (rule 6.2).

**Bounds (rule 2.4):** the `Array`'s default "bouncing" (`set_bounds`: "bounce on border (at most once). This is a variant of clipping"). `outside` was 0 in every run.

**Keeping going:** as above.

**Left out:**
- `NGOpt`: one wizard runs. On these budgets it chooses a portfolio of 24 to 33 metamodel CMA-ES (`NGOpt36`), at 75 to 120 evaluations per second.
- `CMA`: see `one_plus_one`; it runs on Rosenbrock.
- `TwoPointsDE` ("excellent in many cases") and `PSO` ("excellent in terms of robustness"): the list gives them no case that singles out this one.
- `CmaFmin2` (pycma's `fmin` with IPOP): not in the docs; pycma itself runs in this benchmark.
- `TBPSA`: for noisy functions.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| Rastrigin 10 (500k) | ngiohtuned | 3 | 0 | | 63.4 | 43.3 | 64.8 | 3 |
| Rastrigin 10 (500k) | scr_hammersley | 3 | 0 | | 70.5 | 56.2 | 74.6 | 3 |
| Rastrigin 10 (500k) | one_plus_one | 3 | 0 | | 89.5 | 59.7 | 105 | 3 |
| Rastrigin 30 (2M) | ngiohtuned | 3 | 0 | | 257 | 236 | 278 | 3 |
| Rastrigin 30 (2M) | scr_hammersley | 3 | 0 | | 392 | 382 | 408 | 3 |
| Rastrigin 30 (2M) | one_plus_one | 3 | 0 | | 162 | 161 | 183 | 3 |
| Ackley 30 (1M) | ngiohtuned | 3 | 0 | | 12.4 | 11.8 | 12.7 | 3 |
| Ackley 30 (1M) | scr_hammersley | 3 | 0 | | 20.3 | 19.7 | 20.4 | 3 |
| Ackley 30 (1M) | one_plus_one | 3 | 0 | | 19.0 | 18.6 | 19.0 | 3 |

The runs reached the cap after 2,300 evaluations (`ngiohtuned`, Rastrigin 10) and 8,000 to 9,000 (in 30 dimensions), 17,000 to 21,000 (`scr_hammersley`) and 63,000 to 76,000 (`one_plus_one`).

## Continuous, unimodal: Rosenbrock 10

**Methods:** `ngiohtuned`, `one_plus_one`, and `cma_es`: `CMA`, the list's other matching case, which with bounds is `CMAbounded` (an elitist, diagonal CMA-ES from `MetaCMA`).

**Bounds** and **keeping going:** as above.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| Rosenbrock 10 (500k) | ngiohtuned | 3 | 0 | | 115 | 111 | 116 | 3 |
| Rosenbrock 10 (500k) | one_plus_one | 3 | 0 | | 0.166 | 0.105 | 0.191 | 3 |
| Rosenbrock 10 (500k) | cma_es | 5 | 1 | 491 | 0.055 | 0.0090 | 3.99 | 4 |

The runs reached the cap after about 2,200 evaluations (`ngiohtuned`), 75,000 (`one_plus_one`) and 15,000 to 25,000 (`cma_es`).

## Can't run

- OneMax 100 and 1000, matched: no GA with the matched operators.
- The multi-objective scenarios: Nevergrad has none of the five matched algorithms. Its docs recommend DE for several objectives; its result, `optimizer.pareto_front()`, is the non-dominated set of every point evaluated, an unbounded archive that rule 7.2 excludes (DE's `optimizer.population` would give a front of the scenario's size).

## Bugs found

- **`NgIohTuned`'s metamodel crashes with NumPy 2.5.** It calls `float()` on a one-element array (`metamodel.py`, `loss_function_sm`), which NumPy 2.5 refuses (`TypeError`). Worked around: the adapter restores the old conversion in that module only ([the line](../../../benchmarks/adapters/nevergrad/bench.py#L58-L62)); the algorithm is unchanged. Both results: without it, every `ngiohtuned` run on Rastrigin, Rosenbrock and Ackley crashes at the metamodel's first fit (after 118 evaluations in 10 dimensions, 892 or 1,784 in 30), with no result; with it, the results above. Not reported upstream yet.
