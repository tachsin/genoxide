# Nevergrad (Python, 1.0.12)

Nevergrad is Meta's platform for derivative-free optimization: many optimizers behind one ask and tell interface, and "wizards" that choose among them from the budget, the number of workers and the parametrization. Its docs are at [facebookresearch.github.io/nevergrad](https://facebookresearch.github.io/nevergrad/), with the choice of optimizer in [How to perform optimization](https://facebookresearch.github.io/nevergrad/optimization.html) ("Choosing an optimizer", "Example with permutation", "Multiobjective minimization"), and the [README](https://github.com/facebookresearch/nevergrad).

Adapter: [benchmarks/adapters/nevergrad/](../../../benchmarks/adapters/nevergrad/).
Know a better way to solve one of these problems with Nevergrad? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

**How every run works** ([`run`](../../../benchmarks/adapters/nevergrad/bench.py#L219-L248)): the docs' [ask and tell loop](https://facebookresearch.github.io/nevergrad/optimization.html#ask-and-tell-interface) with one worker, the optimizer given the scenario's budget ("the budget tells NgIohTuned which algorithm to choose, as a user would give it"), seeded the docs' two ways ([Reproducibility](https://facebookresearch.github.io/nevergrad/optimization.html#reproducibility): numpy's global random state and the parametrization's `random_state`). The loop stops right after the evaluation that reaches the target, the first hit (rule 3.3), or before one past the budget or the time cap. Every optimizer runs with its defaults.

**Imports (rule 4.2):** Nevergrad imports some modules inside the first run that needs them: pycma with matplotlib for the CMA-ES in `NgIohTuned` and `CMA`, scikit-learn for `NgIohTuned`'s metamodel, and the COBYLA code of SciPy. That took about 0.4 s of the first timed run. The adapter imports them at the start, before the clock ([the imports](../../../benchmarks/adapters/nevergrad/bench.py#L46-L53)).

**Keeping going (rule 2.2):** Nevergrad's optimizers have no stop criterion: every `ask` gives a candidate until the loop stops. The CMA-ES inside `CMA` and `NgIohTuned` starts again from the middle of the domain when pycma's own criteria end it (`optimizerlib.py`, `_CMA.es`), with the same population size: that is Nevergrad's restart.

**Rule 5.3:** Nevergrad's wizards cost milliseconds per evaluation (below), so many runs reach the 60-second cap. The adapter stops a solver after 3 seeds that all reach the cap without the target, as `run.py` does.

**Which method, by the docs (rule 6.2):** the docs' [Choosing an optimizer](https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer) is a list of "rules of thumb", each for its case. Their first is the default: "`NgIohTuned` is 'meta'-optimizer which adapts to the provided settings (budget, number of workers, parametrization) and should therefore be a good default." It runs on every problem. The others follow from the list's cases, or from the docs' example for the problem type.

## Binary: OneMax 100 (idiomatic)

**Methods** ([`solvers`](../../../benchmarks/adapters/nevergrad/bench.py#L169-L216)), on the parameter of the docs' own OneMax example, `ng.p.TransitionChoice(range(2), repetitions=100)` ([Basic example](https://facebookresearch.github.io/nevergrad/optimization.html#basic-example)):
- `ngiohtuned`: the default. On this parameter it chooses `DoubleFastGADiscreteOnePlusOne`.
- `discrete_one_plus_one`: `DiscreteOnePlusOne`, the docs' OneMax example.
- `portfolio_discrete_one_plus_one`: "`PortfolioDiscreteOnePlusOne` is excellent in discrete settings of mixed settings when high precision on parameters is not relevant".

**Left out:** `NGOpt`, the README's example wizard: the docs' list names `NgIohTuned` as the default, and one wizard runs. `TwoPointsDE` on a `Choice`: the machine-learning page's suggestion "if you have more budget" for hyperparameters, not for a bit string.

**Separate tests** (2026-09-25, Nevergrad 1.0.12, 5 seeds, 60 s cap, rule 5.3; the machine ran other tests at the same time, and Nevergrad ran several times slower than in a benchmark run, so capped runs stopped after far fewer evaluations):

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| OneMax 100 (200k) | ngiohtuned | 5 | 5 | 3,553 | 100 | 100 | 100 | 0 |
| OneMax 100 (200k) | discrete_one_plus_one | 5 | 5 | 3,209 | 100 | 100 | 100 | 0 |
| OneMax 100 (200k) | portfolio_discrete_one_plus_one | 3 | 0 | | 80 | 81 | 78 | 3 |

`PortfolioDiscreteOnePlusOne` reached the cap after about 9,000 evaluations.

## Permutation: N-Queens 32 and 64

Nevergrad has no permutation parameter, but its docs' [Example with permutation](https://facebookresearch.github.io/nevergrad/optimization.html#example-with-permutation) links to [How to optimize permutations with Nevergrad](https://docs.google.com/document/d/1B5yVOx1H1nnjY3EOf14487hAr8CzwJ9zEkDwQnZ5nbE/edit): a real `ng.p.Array(shape=(n,))`, "if you consider permutations in [0,1,2,3,...,499]", whose `argsort` is the permutation, as in Nevergrad's own TSP benchmark (`nevergrad/functions/stsp/core.py`). The adapter does the same, and prints the permutation.

**Methods:**
- `ngiohtuned`: the default. On this parameter it chooses a metamodel over `VLPCMA` (CMA-ES with a large population).
- `rotated_two_points_de`, `genetic_de`: `RotatedTwoPointsDE` and `GeneticDE`, the two optimizers the permutation page names: they "will cut and paste part of the permutation into other parts".

**Separate tests** (as above):

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| N-Queens 32 (500k) | ngiohtuned | 3 | 0 | | 6 | 6 | 7 | 3 |
| N-Queens 32 (500k) | rotated_two_points_de | 3 | 0 | | 6 | 5 | 6 | 3 |
| N-Queens 32 (500k) | genetic_de | 3 | 0 | | 3 | 3 | 5 | 3 |
| N-Queens 64 (1M) | ngiohtuned | 3 | 0 | | 20 | 20 | 21 | 3 |
| N-Queens 64 (1M) | rotated_two_points_de | 3 | 0 | | 18 | 17 | 18 | 3 |
| N-Queens 64 (1M) | genetic_de | 3 | 0 | | 16 | 15 | 16 | 3 |

All reached the cap: `NgIohTuned` after about 25,000 evaluations, the DEs after 48,000 to 77,000.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

The parameter is a bounded `ng.p.Array(shape=(n,), lower=..., upper=...)`, the docs' way to give bounds: "if both lower and upper bounds are provided, sigma will be adapted so that the range spans 6 sigma", and the initial value is the middle.

**Methods:**
- `ngiohtuned`: the default. On these problems and budgets it chooses a metamodel over `VLPCMA`: every few dozen evaluations it fits a quadratic model to the best points and proposes its optimum. That costs about 1 ms per evaluation or more, so its runs reach the cap after far fewer evaluations than the budget.
- `scr_hammersley`: "`ScrHammersleySearchPlusMiddlePoint` is excellent for super parallel cases (fully one-shot, i.e. `num_workers` = budget included) or for very multimodal cases", the only one the list gives for multimodal functions: a scrambled Hammersley sequence over the box, and its middle point.
- `one_plus_one`: two of the list's cases match every continuous scenario here, one worker and a budget over 1000 times the dimension: "`OnePlusOne` is a simple robust method for continuous parameters with `num_workers` < 8" and "`CMA` is excellent for control ... when the budget is large (e.g. 1000 x the dimension)". With no preference between them, the first the list gives runs (rule 6.2).

**Bounds (rule 2.4):** the `Array`'s bound layer, by default "bouncing" (`set_bounds`: "bounce on border (at most once). This is a variant of clipping"), so every value evaluated is inside the box. The adapter counts the evaluated values outside it: 0 in every run.

**Left out:**
- `NGOpt`, the README's example wizard: one wizard runs, the list's default. On these budgets (over 5000 times the dimension) `NGOpt` chooses a portfolio of 24 to 33 metamodel CMA-ES (`NGOpt36`), at 75 to 120 evaluations per second: in 0.6.0 it reached none of these targets, after 4,700 to 7,000 evaluations.
- `CMA`: see `one_plus_one` above; it runs on Rosenbrock. In 0.6.0 it reached Ackley 30 in 10 of 10 runs, and Rastrigin in none.
- `TwoPointsDE` ("excellent in many cases") and `PSO` ("excellent in terms of robustness"): the list gives them no case that singles out this one. In 0.6.0 neither reached a Rastrigin or Ackley target.
- `CmaFmin2` (pycma's `fmin` with IPOP restarts): not in the docs; `NGOpt` picks it only for budgets between 500 and 5000 times the dimension in 20 dimensions or more. pycma itself runs in this benchmark.
- `TBPSA`: for noisy functions.

**Separate tests** (as above):

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

All reached the cap: `ngiohtuned` after 2,300 evaluations on Rastrigin 10 and 8,000 to 9,000 in 30 dimensions, `scr_hammersley` after 17,000 to 21,000, `one_plus_one` after 63,000 to 76,000. In the earlier separate tests, with the previous optimum and a less loaded machine, `one_plus_one` used the whole budget of Rastrigin 10, and `ngiohtuned` reached the cap after 45,000 to 65,000 evaluations and reached 0.01 on Ackley 30 in 5 of 5 runs.

## Continuous, unimodal: Rosenbrock 10

**Methods:** `ngiohtuned`; `one_plus_one`; and `cma_es`, `CMA`, the other of the list's two matching cases. With bounds `CMA` is `CMAbounded` (an elitist, diagonal CMA-ES from `MetaCMA`). Parameter and bounds as above: 0 values outside the box.

**Separate tests** (as above):

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| Rosenbrock 10 (500k) | ngiohtuned | 3 | 0 | | 115 | 111 | 116 | 3 |
| Rosenbrock 10 (500k) | one_plus_one | 3 | 0 | | 0.166 | 0.105 | 0.191 | 3 |
| Rosenbrock 10 (500k) | cma_es | 5 | 1 | 491 | 0.055 | 0.0090 | 3.99 | 4 |

`ngiohtuned` reached the cap after about 2,200 evaluations, `one_plus_one` after about 75,000 and `cma_es` after 15,000 to 25,000. In the earlier separate tests, on a less loaded machine, `one_plus_one` and `cma_es` reached 0.01 in 5 of 5 runs, after medians of 156,521 and 24,829 evaluations.

## Changes from the 0.6.0 benchmark

- `NgIohTuned`, the docs' stated default, replaces `NGOpt`, the README's example.
- The continuous methods follow the docs' list of cases: `ScrHammersleySearchPlusMiddlePoint` and `OnePlusOne` on Rastrigin and Ackley, `OnePlusOne` and `CMA` on Rosenbrock. 0.6.0 ran `NGOpt`, `CMA`, `TwoPointsDE` and `PSO` on all three, four methods where rule 6.4 allows three.
- OneMax adds `PortfolioDiscreteOnePlusOne`.
- N-Queens runs, with the docs' permutation encoding.
- The multi-objective scenarios don't run (below).
- Each run prints its solution, the continuous runs the values evaluated outside the bounds, and the adapter has the `values` command.
- The modules that Nevergrad imports in its first run are imported before the clock.
- Each run records its first hit, the evaluation that reached the target (rule 3.3).
- The optimum of Rastrigin and Ackley is further from the origin (rule 1.4).

## Can't run

- OneMax 100 and 1000, matched: Nevergrad has no GA with the matched operators (rule 6.1).
- The multi-objective scenarios: they run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA (rule 6.1), and Nevergrad has none of them. Its docs recommend DE for several objectives ("`DE` and its variants have however been updated to make use of the full multi-objective losses"), which 0.6.0 ran. Its result, `optimizer.pareto_front()`, is the non-dominated set of every point evaluated, an unbounded archive that rule 7.2 excludes; DE's own population (`optimizer.population`) would give a front of the scenario's size.

## Bugs found

- Nevergrad 1.0.12 with NumPy 2.5: the metamodel that `NgIohTuned` uses calls `float()` on a one-element array (`metamodel.py`, `loss_function_sm`), which NumPy 2.5 refuses, so the run crashes with a `TypeError`. The adapter restores the old conversion in that module only ([the line](../../../benchmarks/adapters/nevergrad/bench.py#L58-L62)); the algorithm is unchanged. Both results (rule 8.4): without it, every `ngiohtuned` run on Rastrigin, Rosenbrock and Ackley crashes at the metamodel's first fit, after 118 evaluations in 10 dimensions and 892 or 1,784 in 30, with no result; with it, the results above. Not reported upstream yet.
