# pycma (Python, 4.5.0)

pycma is the reference Python implementation of CMA-ES by Nikolaus Hansen and co-authors. Its docs are its docstrings, published as the [API docs](https://cma-es.github.io/apidocs-pycma/) (in particular [`cma.fmin2`](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.html#fmin2) and [`CMAEvolutionStrategy`](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html)), with the [practical hints](https://cma-es.github.io/cmaes_sourcecode_page.html) on the CMA-ES site.

Adapter: [benchmarks/adapters/pycma/](../../../benchmarks/adapters/pycma/).
Know a better way to solve one of these problems with pycma? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs pycma

Every method is `cma.fmin2` or its surrogate variant ([`solve`](../../../benchmarks/adapters/pycma/bench.py#L159-L212)), with settings from the `fmin2` docstring or the defaults:
- `restarts=9`: "the recommended setting is `restarts <= 9` and `x0` passed as a `callable`";
- `x0` a callable drawing a uniform point in the box, "to restart from different points (recommended)";
- `sigma0` a quarter of the box width: "`sigma0` should be about 1/4th of the search domain width";
- the box as `bounds`, with the default `BoundTransform`;
- the other options at their defaults: population 4 + 3 ln n, and the stop criteria (`tolfun`, `tolx`, `tolstagnation`, ...);
- `parallel_objective`, the batch interface (rule 3.4): each population in one numpy call, with the same solutions, order and first hit as one call per solution.

The rest:
- **Evaluations and stop:** the counter ([`Budget`](../../../benchmarks/adapters/pycma/bench.py#L44-L107)) counts every solution, including the final mean `fmin2` evaluates after each run (`eval_final_mean`), and records the first hit. It ends the run after the population that reaches 0.01, or before one past the budget (a population is cut at the budget) or the time cap.
- **Keeping going (rule 2.2):** pycma's restarts. The stop criteria stay in effect with `maxfevals`, so each ends a run and `fmin2` restarts. The limit of 9 restarts is a budget: if all 9 ended early, the adapter would call `fmin2` again from its first population size, with a note on stderr; it never happened.
- **Bounds (rule 2.4):** `BoundTransform` maps every sample into the box.
- **Seeds:** pycma samples from numpy's global random state, which the adapter seeds with the run's seed, and with `(seed + 1) * 1_000_000 + call` for a further `fmin2` call. The `seed` option is `np.nan` ("do nothing"), because pycma reads 0 as "seed from the clock"; the restarts continue the same random stream.
- **Separate tests:** 2026-09-25, pycma 4.5.0, 5 seeds, the scenario's budget, 60 s cap, rule 5.3, with other tests on the machine (capped runs stopped after fewer evaluations than in a benchmark run). `outside` was 0 in every run.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:** two, each with its own example on Rastrigin and no stated preference between them (rule 6.2):
- **`ipop_cma_es`:** `restarts=9, incpopsize=2`, each restart doubling the population: the `CMAEvolutionStrategy` docstring's "Example implementing restarts with increasing popsize (IPOP)", on Rastrigin ("usually after five restarts the global optimum is located").
- **`bipop_cma_es`:** `restarts=9, bipop=True`: the `fmin2` docstring's Rastrigin example ("the BIPOP restart strategy"), which interleaves small-population restarts with smaller steps.

**Keeping going:** pycma's restarts (above).

**Left out:**
- CMA-ES without restarts: rule 6.3.
- `restart_from_best=True`: "CAVE: restart_from_best is often not useful" (`fmin2`).
- `BoundPenalty`: `BoundTransform` is the default.
- lq-CMA-ES: its docs and [page](https://cma-es.github.io/lq-cma/) show it on Rosenbrock (below).
- `noise_handler`: the functions aren't noisy.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| Rastrigin 10 (500k) | ipop_cma_es | 5 | 5 | 63,850 | 0.0071 | 0.0050 | 0.0085 | 0 |
| Rastrigin 10 (500k) | bipop_cma_es | 5 | 5 | 230,680 | 0.0070 | 0.0036 | 0.0081 | 0 |
| Rastrigin 30 (2M) | ipop_cma_es | 3 | 0 | | 9.95 | 6.96 | 13.9 | 3 |
| Rastrigin 30 (2M) | bipop_cma_es | 3 | 0 | | 12.9 | 10.9 | 13.9 | 3 |
| Ackley 30 (1M) | ipop_cma_es | 5 | 5 | 3,603 | 0.0094 | 0.0087 | 0.0099 | 0 |
| Ackley 30 (1M) | bipop_cma_es | 5 | 5 | 3,603 | 0.0094 | 0.0087 | 0.0099 | 0 |

The Rastrigin 30 runs reached the cap after about 300,000 evaluations. On Ackley both solve it in their first run, which is the same for both.

## Continuous, unimodal: Rosenbrock 10

**Methods** ([`solvers`](../../../benchmarks/adapters/pycma/bench.py#L215-L219)): both are the docs' examples on Rosenbrock, with no stated preference:
- **`cma_es`:** `cma.fmin2` as in its docstring's Rosenbrock example, with the settings above and IPOP restarts.
- **`lq_cma_es`:** [`cma.fmin_lq_surr2`](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.html#fmin_lq_surr2) ([page](https://cma-es.github.io/lq-cma/)): a linear or quadratic model decides which part of each population to evaluate, one solution at a time (no `parallel_objective`). Same arguments; its restarts also double the population.

**Keeping going:** as above.

**Left out:** the docstring example's `CMA_diagonal: 100`, a speed-up: the default full covariance is kept.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| Rosenbrock 10 (500k) | cma_es | 5 | 5 | 4,341 | 0.0081 | 0.0065 | 0.0090 | 0 |
| Rosenbrock 10 (500k) | lq_cma_es | 5 | 5 | 1,219 | 0.0095 | 0.0086 | 0.0100 | 0 |

lq-CMA-ES's model takes about 30 times as much time per evaluation as `cma_es`.

## Can't run

- OneMax, N-Queens: CMA-ES optimizes real numbers; pycma's `integer_variables` option isn't presented for binary strings or permutations.
- The multi-objective scenarios: pycma optimizes one objective.

## Bugs found

None.
