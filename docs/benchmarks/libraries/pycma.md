# pycma (Python, 4.5.0)

pycma is the reference Python implementation of CMA-ES by Nikolaus Hansen and co-authors. Its documentation is the docstrings of the package, published as the [API docs](https://cma-es.github.io/apidocs-pycma/) (in particular [`cma.fmin2`](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.html#fmin2) and the [`CMAEvolutionStrategy` class](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html)), with the [practical hints](https://cma-es.github.io/cmaes_sourcecode_page.html) on the CMA-ES site.

Adapter: [benchmarks/adapters/pycma/](../../../benchmarks/adapters/pycma/).
Know a better way to solve one of these problems with pycma? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:** two, both documented for multimodal functions with the same function, [`cma.fmin2`](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.html#fmin2) ([`solve`](../../../benchmarks/adapters/pycma/bench.py#L148-L201)):
- `ipop_cma_es`: IPOP-CMA-ES, `restarts=9, incpopsize=2`: each restart doubles the population. The `CMAEvolutionStrategy` docstring's "Example implementing restarts with increasing popsize (IPOP)" is on Rastrigin: "On the Rastrigin function, usually after five restarts the global optimum is located. When `fmin2` with the `restarts` parameter is used, ...".
- `bipop_cma_es`: BIPOP-CMA-ES, `restarts=9, bipop=True`: the `fmin2` docstring's example on Rastrigin ("the BIPOP restart strategy (that progressively increases population)"), which interleaves restarts with small populations and smaller step-sizes.

The docs state no preference between the two, and each has its own example on Rastrigin, so both run (rule 6.2).

Settings, all from the `fmin2` docstring or pycma's defaults:
- `restarts=9`: "the recommended setting is `restarts <= 9` and `x0` passed as a `callable`".
- `x0` a callable drawing a uniform random point in the box, "to restart from different points (recommended)".
- `sigma0` a quarter of the box width: "`sigma0` should be about 1/4th of the search domain width".
- the box as the `bounds` option, with the default boundary handler `BoundTransform`.
- every other option at its default: population 4 + 3 ln n, the stop criteria (`tolfun`, `tolx`, `tolstagnation`, ...) that end each run and start the next.
- `parallel_objective`, `fmin2`'s batch interface: "an objective function that accepts a list of `numpy.ndarray` as input and returns a `list`". Each population is evaluated in one call of a numpy function (rule 3.4). The search is the same: the same solutions in the same order, and the same best value at the same evaluation as with one call per solution. The objective function still evaluates the final mean.
- seeds: pycma samples from numpy's global random state (its `randn` option is `np.random.randn`). The adapter seeds it with the run's seed, and with `(seed + 1) * 1_000_000 + call` if it calls `fmin2` again (rule 2.2). The `seed` option is `np.nan`, which pycma documents as "do nothing", because pycma reads a seed of 0 as "seed from the clock". So pycma doesn't seed again at a restart: its runs continue the same random stream.

**Bounds (rule 2.4):** `BoundTransform` maps every sampled point into the box before it is evaluated. The adapter counts the evaluated solutions outside the box ([`Budget`](../../../benchmarks/adapters/pycma/bench.py#L44-L96)): 0 in every run.

**Keeping going:** pycma's restart mechanism (rule 2.2). CMA-ES's stop criteria (`tolfun`, `tolx`, `tolstagnation`, ...) are the method's own settings, and they stay in effect when a budget is set pycma's way (`maxfevals`), so they count: each ends a run, and `fmin2` restarts. The limit of 9 restarts is only a budget: if all 9 ended before the budget, the adapter would call `fmin2` again, from its first population size, and say so on stderr. That never happened in the separate tests.

**Stopping:** the adapter's counter ends the run after the population whose evaluation reaches 0.01, or before one that would start past the budget or the time cap. A population that would go past the budget is evaluated only up to it. The counter counts every solution evaluated, including the final mean that `fmin2` evaluates after each run (`eval_final_mean`), and records the first hit: the first evaluation that reaches 0.01 (rule 3.3).

**Left out:**
- CMA-ES without restarts: pycma offers restarts, and rule 6.3 excludes it on a multimodal function.
- `restart_from_best=True`: `fmin2` warns "CAVE: restart_from_best is often not useful".
- `BoundPenalty`: the other boundary handler; `BoundTransform` is the default.
- lq-CMA-ES (`cma.fmin_lq_surr2`): its docs and [page](https://cma-es.github.io/lq-cma/) show it on Rosenbrock, not on multimodal functions; it runs on Rosenbrock below.
- `noise_handler`: the functions aren't noisy.

**Separate tests** (2026-09-25, pycma 4.5.0, 5 seeds, the scenario's budget, 60 s cap, rule 5.3; the machine ran other tests at the same time, so capped runs stopped after fewer evaluations than a benchmark run would):

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| Rastrigin 10 (500k) | ipop_cma_es | 5 | 5 | 63,850 | 0.0071 | 0.0050 | 0.0085 | 0 |
| Rastrigin 10 (500k) | bipop_cma_es | 5 | 5 | 230,680 | 0.0070 | 0.0036 | 0.0081 | 0 |
| Rastrigin 30 (2M) | ipop_cma_es | 3 | 0 | | 9.95 | 6.96 | 13.9 | 3 |
| Rastrigin 30 (2M) | bipop_cma_es | 3 | 0 | | 12.9 | 10.9 | 13.9 | 3 |
| Ackley 30 (1M) | ipop_cma_es | 5 | 5 | 3,603 | 0.0094 | 0.0087 | 0.0099 | 0 |
| Ackley 30 (1M) | bipop_cma_es | 5 | 5 | 3,603 | 0.0094 | 0.0087 | 0.0099 | 0 |

The Rastrigin 30 runs reached the cap after about 300,000 of the 2 million evaluations. In the earlier separate tests, with the previous optimum and a less loaded machine, IPOP reached 0.01 there in 5 of 5 runs, after a median of 783,423 evaluations. On Ackley both solvers solve it in their first run, which is the same for both.

## Continuous, unimodal: Rosenbrock 10

**Methods** ([`solvers`](../../../benchmarks/adapters/pycma/bench.py#L204-L208)):
- `cma_es`: `cma.fmin2` as in its docstring's Rosenbrock example, with the settings above and IPOP restarts, which only start if a run ends before the target.
- `lq_cma_es`: lq-CMA-ES, [`cma.fmin_lq_surr2`](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.html#fmin_lq_surr2), whose docstring example and [page](https://cma-es.github.io/lq-cma/) are on Rosenbrock. It fits a linear or quadratic model to the evaluated points and evaluates only part of each population. Same arguments; its restarts also double the population. It has no `parallel_objective`: its model decides which solutions to evaluate, one at a time.

Both are the docs' examples on Rosenbrock, with no stated preference between them, so both run (rule 6.2).

**Bounds, keeping going, stopping:** as above; 0 solutions outside the box.

**Left out:** the docstring example's `CMA_diagonal: 100` (a diagonal covariance for the first 100 iterations, a speed-up): the default, full covariance from the start, is kept, as rule 6.2 asks when the docs state no preference.

**Separate tests** (as above):

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| Rosenbrock 10 (500k) | cma_es | 5 | 5 | 4,341 | 0.0081 | 0.0065 | 0.0090 | 0 |
| Rosenbrock 10 (500k) | lq_cma_es | 5 | 5 | 1,219 | 0.0095 | 0.0086 | 0.0100 | 0 |

lq-CMA-ES needs a third of the evaluations, but its model costs about 30 times as much time per evaluation as `cma_es`.

## Changes from the 0.6.0 benchmark

- BIPOP-CMA-ES runs next to IPOP on the multimodal problems, and lq-CMA-ES on Rosenbrock.
- `fmin2` evaluates each population in one numpy call (`parallel_objective`). The run stops after the population that reaches the target, and records the first hit, the evaluation that reached it.
- Each run prints its solution and the evaluated solutions outside the bounds, and the adapter has the `values` command.
- The seeds of different runs no longer overlap: `seed + 1`, plus 1 at each restart, made seed 0's second run start like seed 1's first. The run's seed now seeds numpy, and pycma doesn't seed again.
- The fitness functions use numpy, like pycma's own `cma.ff`.
- The optimum of Rastrigin and Ackley is further from the origin (rule 1.4).

## Can't run

- OneMax, N-Queens: CMA-ES optimizes real numbers. pycma has an `integer_variables` option for mixed-integer problems, but its docs don't present it for binary strings or permutations.
- The multi-objective scenarios: pycma optimizes one objective.

## Bugs found

None.
