# Nevergrad (Python, 1.0.12)

Nevergrad is Meta's platform for derivative-free optimization: many optimizers behind one ask and tell interface, and "wizards" that choose among them from the budget, the number of workers and the parametrization. Its docs are at [facebookresearch.github.io/nevergrad](https://facebookresearch.github.io/nevergrad/), with the choice of optimizer in [How to perform optimization](https://facebookresearch.github.io/nevergrad/optimization.html) ("Choosing an optimizer", "Example with permutation", "Multiobjective minimization"), and the [README](https://github.com/facebookresearch/nevergrad).

Adapter: [benchmarks/adapters/nevergrad/](../../../benchmarks/adapters/nevergrad/).
Know a better way to solve one of these problems with Nevergrad? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

**How every run works** ([`run`](../../../benchmarks/adapters/nevergrad/bench.py#L209-L235)): the docs' [ask and tell loop](https://facebookresearch.github.io/nevergrad/optimization.html#ask-and-tell-interface) with one worker, the optimizer given the scenario's budget ("the budget tells NgIohTuned which algorithm to choose, as a user would give it"), seeded the docs' two ways ([Reproducibility](https://facebookresearch.github.io/nevergrad/optimization.html#reproducibility): numpy's global random state and the parametrization's `random_state`). The loop stops right after the evaluation that reaches the target, or before one past the budget or the time cap. Every optimizer runs with its defaults.

**Keeping going (rule 2.2):** Nevergrad's optimizers have no stop criterion: every `ask` gives a candidate until the loop stops. The CMA-ES inside `CMA` and `NgIohTuned` starts again from the middle of the domain when pycma's own criteria end it (`optimizerlib.py`, `_CMA.es`), with the same population size: that is Nevergrad's restart.

**Rule 5.3:** Nevergrad's wizards cost milliseconds per evaluation (below), so many runs reach the 60-second cap. The adapter stops a solver after 3 seeds that all reach the cap without the target, as `run.py` does.

**Which method, by the docs (rule 6.2):** the docs' [Choosing an optimizer](https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer) is a list of "rules of thumb", each for its case. Their first is the default: "`NgIohTuned` is 'meta'-optimizer which adapts to the provided settings (budget, number of workers, parametrization) and should therefore be a good default." It runs on every problem. The others follow from the list's cases, or from the docs' example for the problem type.

## Binary: OneMax 100 (idiomatic)

**Methods** ([`solvers`](../../../benchmarks/adapters/nevergrad/bench.py#L159-L206)), on the parameter of the docs' own OneMax example, `ng.p.TransitionChoice(range(2), repetitions=100)` ([Basic example](https://facebookresearch.github.io/nevergrad/optimization.html#basic-example)):
- `ngiohtuned`: the default. On this parameter it chooses `DoubleFastGADiscreteOnePlusOne`.
- `discrete_one_plus_one`: `DiscreteOnePlusOne`, the docs' OneMax example.
- `portfolio_discrete_one_plus_one`: "`PortfolioDiscreteOnePlusOne` is excellent in discrete settings of mixed settings when high precision on parameters is not relevant".

**Left out:** `NGOpt`, the README's example wizard: the docs' list names `NgIohTuned` as the default, and one wizard runs. `TwoPointsDE` on a `Choice`: the machine-learning page's suggestion "if you have more budget" for hyperparameters, not for a bit string.

**Separate tests** (2026-09-25, Nevergrad 1.0.12, 5 seeds, 60 s cap, rule 5.3; the machine ran other tests at the same time, so capped runs reached fewer evaluations than a benchmark run would):

| Scenario | Solver | Runs | Reached | Median evaluations to the target | Best value: median | best | worst | Runs at the cap |
|---|---|---|---|---|---|---|---|---|
| OneMax 100 (200k) | ngiohtuned | 5 | 5 | 3,553 | 100 | 100 | 100 | 0 |
| OneMax 100 (200k) | discrete_one_plus_one | 5 | 5 | 3,209 | 100 | 100 | 100 | 0 |
| OneMax 100 (200k) | portfolio_discrete_one_plus_one | 3 | 0 | | 88 | 96 | 87 | 3 |

`PortfolioDiscreteOnePlusOne` evaluates about 700 solutions per second, and reached the cap after about 40,000.

## Permutation: N-Queens 32 and 64

Nevergrad has no permutation parameter, but its docs' [Example with permutation](https://facebookresearch.github.io/nevergrad/optimization.html#example-with-permutation) links to [How to optimize permutations with Nevergrad](https://docs.google.com/document/d/1B5yVOx1H1nnjY3EOf14487hAr8CzwJ9zEkDwQnZ5nbE/edit): a real `ng.p.Array(shape=(n,))`, "if you consider permutations in [0,1,2,3,...,499]", whose `argsort` is the permutation, as in Nevergrad's own TSP benchmark (`nevergrad/functions/stsp/core.py`). The adapter does the same, and prints the permutation.

**Methods:**
- `ngiohtuned`: the default. On this parameter it chooses a metamodel over `VLPCMA` (CMA-ES with a large population).
- `rotated_two_points_de`, `genetic_de`: `RotatedTwoPointsDE` and `GeneticDE`, the two optimizers the permutation page names: they "will cut and paste part of the permutation into other parts".

**Separate tests** (as above):

| Scenario | Solver | Runs | Reached | Median evaluations to 0 | Best value: median | best | worst | Runs at the cap |
|---|---|---|---|---|---|---|---|---|
| N-Queens 32 (500k) | ngiohtuned | 3 | 0 | | 2 | 1 | 2 | 3 |
| N-Queens 32 (500k) | rotated_two_points_de | 3 | 0 | | 4 | 4 | 4 | 3 |
| N-Queens 32 (500k) | genetic_de | 3 | 0 | | 2 | 2 | 3 | 3 |
| N-Queens 64 (1M) | ngiohtuned | 3 | 0 | | 16 | 16 | 16 | 3 |
| N-Queens 64 (1M) | rotated_two_points_de | 3 | 0 | | 17 | 16 | 17 | 3 |
| N-Queens 64 (1M) | genetic_de | 3 | 0 | | 12 | 11 | 13 | 3 |

All reached the cap: `NgIohTuned` after about 150,000 evaluations, the DEs after 300,000 to 400,000.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

The parameter is a bounded `ng.p.Array(shape=(n,), lower=..., upper=...)`, the docs' way to give bounds: "if both lower and upper bounds are provided, sigma will be adapted so that the range spans 6 sigma", and the initial value is the middle.

**Methods:**
- `ngiohtuned`: the default. On these problems and budgets it chooses a metamodel over `VLPCMA`: every few dozen evaluations it fits a quadratic model to the best points and proposes its optimum. That costs about 1 ms per evaluation, so its runs reach the cap after 45,000 to 65,000 evaluations.
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

| Scenario | Solver | Runs | Reached | Median evaluations to 0.01 | Best value: median | best | worst | Runs at the cap |
|---|---|---|---|---|---|---|---|---|
| Rastrigin 10 (500k) | ngiohtuned | 3 | 0 | | 1.99 | 1.99 | 3.98 | 3 |
| Rastrigin 10 (500k) | scr_hammersley | 3 | 0 | | 61.7 | 56.7 | 62.2 | 3 |
| Rastrigin 10 (500k) | one_plus_one | 5 | 0 | | 9.95 | 4.97 | 22.9 | 0 |
| Rastrigin 30 (2M) | ngiohtuned | 3 | 0 | | 14.9 | 13.9 | 15.9 | 3 |
| Rastrigin 30 (2M) | scr_hammersley | 3 | 0 | | 310 | 310 | 310 | 3 |
| Rastrigin 30 (2M) | one_plus_one | 3 | 0 | | 74.6 | 73.6 | 80.6 | 3 |
| Ackley 30 (1M) | ngiohtuned | 5 | 5 | 47,723 | 0.0045 | 0.0029 | 0.0098 | 0 |
| Ackley 30 (1M) | scr_hammersley | 3 | 0 | | 3.90 | 3.90 | 3.90 | 3 |
| Ackley 30 (1M) | one_plus_one | 3 | 0 | | 2.12 | 2.01 | 2.32 | 3 |

`one_plus_one` used the whole budget on Rastrigin 10, stuck in a local minimum. The Hammersley search's best on Rastrigin 30 is its middle point in every run.

## Continuous, unimodal: Rosenbrock 10

**Methods:** `ngiohtuned`; `one_plus_one`; and `cma_es`, `CMA`, the other of the list's two matching cases. With bounds `CMA` is `CMAbounded` (an elitist, diagonal CMA-ES from `MetaCMA`). Parameter and bounds as above: 0 values outside the box.

**Separate tests** (as above):

| Scenario | Solver | Runs | Reached | Median evaluations to 0.01 | Best value: median | best | worst | Runs at the cap |
|---|---|---|---|---|---|---|---|---|
| Rosenbrock 10 (500k) | ngiohtuned | 3 | 0 | | 1.67 | 1.46 | 1.80 | 3 |
| Rosenbrock 10 (500k) | one_plus_one | 5 | 5 | 156,521 | 0.0100 | 0.0087 | 0.0100 | 0 |
| Rosenbrock 10 (500k) | cma_es | 5 | 5 | 24,829 | 0.0100 | 0.0090 | 0.0100 | 0 |

`ngiohtuned` reached the cap after about 16,500 evaluations: its metamodel costs more here than on Rastrigin.

## Changes from the 0.6.0 benchmark

- `NgIohTuned`, the docs' stated default, replaces `NGOpt`, the README's example.
- The continuous methods follow the docs' list of cases: `ScrHammersleySearchPlusMiddlePoint` and `OnePlusOne` on Rastrigin and Ackley, `OnePlusOne` and `CMA` on Rosenbrock. 0.6.0 ran `NGOpt`, `CMA`, `TwoPointsDE` and `PSO` on all three, four methods where rule 6.4 allows three.
- OneMax adds `PortfolioDiscreteOnePlusOne`.
- N-Queens runs, with the docs' permutation encoding.
- The multi-objective scenarios don't run (below).
- Each run prints its solution, the continuous runs the values evaluated outside the bounds, and the adapter has the `values` command.

## Can't run

- OneMax 100 and 1000, matched: Nevergrad has no GA with the matched operators.
- The multi-objective scenarios: they run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA (rule 6.1), and Nevergrad has none of them. Its docs recommend DE for several objectives ("`DE` and its variants have however been updated to make use of the full multi-objective losses"), which 0.6.0 ran. Its result, `optimizer.pareto_front()`, is the non-dominated set of every point evaluated, an unbounded archive that rule 7.2 excludes; DE's own population (`optimizer.population`) would give a front of the scenario's size.

## Bugs found

- Nevergrad 1.0.12 with NumPy 2.5: the metamodel that `NgIohTuned` uses calls `float()` on a one-element array (`metamodel.py`, `loss_function_sm`), which NumPy 2.5 refuses, so the run crashes with a `TypeError`. The adapter restores the old conversion in that module only ([the line](../../../benchmarks/adapters/nevergrad/bench.py#L49-L53)); the algorithm is unchanged. Not reported upstream yet.
