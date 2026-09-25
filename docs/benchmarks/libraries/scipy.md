# SciPy (Python, 1.18.1)

SciPy's [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) minimizes functions of real numbers, with local methods ([`minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)) and global ones ([`differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html), [`dual_annealing`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html), [`direct`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.direct.html), [`shgo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html), `basinhopping`). Its docs are the reference pages and the [optimization tutorial](https://docs.scipy.org/doc/scipy/tutorial/optimize.html).

Adapter: [benchmarks/adapters/scipy/](../../../benchmarks/adapters/scipy/).
Know a better way to solve one of these problems with SciPy? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

Every method runs through the adapter's counter ([`Budget`](../../../benchmarks/adapters/scipy/bench.py#L45-L82)): it counts every call of the fitness function, from the method, its restarts, its local searches, its polish and its finite differences; it keeps the best solution and records the first hit, the first evaluation that reaches 0.01 (rule 3.3); it ends the run right after that evaluation, or before one past the budget or the time cap; and it counts the evaluated solutions outside the box (rule 2.4), 0 in every run of the separate tests. The fitness functions are written with numpy, like SciPy's own `scipy.optimize.rosen`.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:** the tutorial's section [Global optimization](https://docs.scipy.org/doc/scipy/tutorial/optimize.html#global-optimization) shows `shgo`, `dual_annealing` and `differential_evolution` on a function with many local minima, and its table of global optimizers gives `direct`, `dual_annealing`, `differential_evolution` and `shgo` for bounds ("If there are multiple candidates, try several"). `shgo` is left out (below), so three run ([`solvers`](../../../benchmarks/adapters/scipy/bench.py#L219-L231)):
- `de`: [`differential_evolution`](../../../benchmarks/adapters/scipy/bench.py#L132-L148) with its defaults, as its docstring's examples call it (on `rosen` and on Ackley): `best1bin`, 15 n individuals, mutation (0.5, 1) with dithering, recombination 0.7, Latin hypercube initialization, `updating='immediate'`, and the final L-BFGS-B polish. The seed is `rng`.
- `dual_annealing`: [`dual_annealing`](../../../benchmarks/adapters/scipy/bench.py#L157-L170) with its defaults, whose docstring example is "a 10-D problem, with many local minima. The function involved is called Rastrigin": generalized simulated annealing with an L-BFGS-B local search. The seed is `rng`.
- `direct`: [`direct`](../../../benchmarks/adapters/scipy/bench.py#L173-L192) with `locally_biased=False`, which its docs recommend: "For hard problems with many local minima, `False` is recommended". And `vol_tol=0`, because its docs say the volume tolerance "decreases exponentially with increasing dimensionality of the problem. Therefore `vol_tol` should be decreased to avoid premature termination of the algorithm for higher dimensions": with the default 1e-16 it ended Rastrigin 10 after 1,965 evaluations, at 33. DIRECT is deterministic: every seed gives the same run.

**Bounds (rule 2.4):**
- `differential_evolution` keeps its population in the box: a mutant coordinate outside it is replaced by a random one, and the polish is bounded.
- `dual_annealing` wraps a visit outside the box back into it (modulo the range), and its local search is bounded.
- `direct` samples the centres of the hyperrectangles that divide the box.

**Keeping going (rule 2.2):**
- `differential_evolution`: its `maxiter` (1,000 generations by default) is only a budget, so it's lifted. Its convergence test, `tol` 0.01 (the standard deviation of the population's values at most 0.01 times their mean), is a setting of the method, so it counts: it ends the attempt, the polish runs, and `differential_evolution` starts again. SciPy has no restart mechanism for it. The first attempt's seed is the run's seed, and restart r's is `(seed + 1) * 1_000_000 + r` ([`restart_seed`](../../../benchmarks/adapters/scipy/bench.py#L151-L154)), so no two runs share a seed.
- `dual_annealing` has no convergence criterion. It restarts by itself: it re-anneals from a new random point when its temperature falls to `initial_temp * restart_temp_ratio`. Its `maxiter` global iterations and `maxfun` evaluations are only budgets, so they're lifted (`maxfun` is twice the budget, so that the adapter's counter, not `dual_annealing`'s own, ends the run).
- `direct`: `maxiter` and `maxfun` are only budgets, so they're lifted. `maxfun` is twice the budget, because DIRECT ends before an iteration that could go past `maxfun`: with `maxfun` at the budget, it ended Rastrigin 30 after 1,886,939 of its 2,000,000 evaluations. Its convergence criterion `len_tol` (1e-6, a setting that stays alongside `maxfun`) counts, but DIRECT has no random start and no restart mechanism, so a restart would repeat the same run. It never converged in the separate tests; if it did, the run would end there, with a message on stderr.

**Left out:**
- `shgo`: its docs say it's "appropriate for solving general purpose NLP and blackbox optimization problems to global optimality (low-dimensional problems)".
- `basinhopping`: the tutorial's table gives it no bounds, and rule 2.4 requires every evaluated solution inside them.
- `differential_evolution` with `vectorized=True`: the docs say it "may aid" an inexpensive objective by calling it once per generation, but it forces `updating='deferred'`, which changes the algorithm: an improving trial vector replaces its target only at the end of the generation, where the default replaces it at once. Rule 3.4 uses no batch interface that changes the algorithm.
- `differential_evolution` with `polish=False`, and other strategies: the defaults run.

**Separate tests** (2026-09-25, SciPy 1.18.1, 5 seeds, the scenario's budget, 60 s cap, rule 5.3; the machine ran other tests at the same time, so capped runs stopped after fewer evaluations than a benchmark run would):

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| Rastrigin 10 (500k) | de | 5 | 5 | 87,692 | 0.0094 | 0.0089 | 0.0099 | 0 |
| Rastrigin 10 (500k) | dual_annealing | 5 | 5 | 3,671 | 0.000026 | 4.8e-7 | 0.00066 | 0 |
| Rastrigin 10 (500k) | direct | 5 | 0 | | 17.2 | 17.2 | 17.2 | 0 |
| Rastrigin 30 (2M) | de | 3 | 0 | | 145 | 115 | 150 | 3 |
| Rastrigin 30 (2M) | dual_annealing | 5 | 5 | 19,434 | 0.000046 | 1.2e-8 | 0.0043 | 0 |
| Rastrigin 30 (2M) | direct | 3 | 0 | | 76.6 | 76.6 | 76.6 | 3 |
| Ackley 30 (1M) | de | 5 | 5 | 114,213 | 0.0098 | 0.0090 | 0.0099 | 0 |
| Ackley 30 (1M) | dual_annealing | 5 | 5 | 25,894 | 0.0059 | 0.0030 | 0.0097 | 0 |
| Ackley 30 (1M) | direct | 5 | 5 | 652,801 | 0.0099 | 0.0099 | 0.0099 | 0 |

On Rastrigin 30, `de` reached the cap after about 370,000 of the 2,000,000 evaluations, and `direct` after about 780,000. `direct` on Rastrigin 10 used the whole budget.

## Continuous, unimodal: Rosenbrock 10

**Methods:** the tutorial's [local minimization](https://docs.scipy.org/doc/scipy/tutorial/optimize.html#local-minimization-of-multivariate-scalar-functions-minimize) section uses the Rosenbrock function as its example, and `differential_evolution`'s docstring example is `rosen`. Three run:
- `lbfgsb`: [`minimize`](../../../benchmarks/adapters/scipy/bench.py#L195-L216) from a uniform random point, with the box as `bounds`, whose default method is then L-BFGS-B ("If not given, chosen to be one of BFGS, L-BFGS-B, SLSQP, depending on whether or not the problem has constraints or bounds"). Without a gradient it estimates one by finite differences, whose evaluations count. The tutorial says BFGS "typically requires fewer function calls than the simplex algorithm even when the gradient must be estimated"; L-BFGS-B is its bounded form.
- `nelder_mead`: `minimize` with `method='Nelder-Mead'`, the tutorial's first example on Rosenbrock, with the box as `bounds` and its default tolerances (the example's `xatol: 1e-8` is left at the default 1e-4).
- `de`: `differential_evolution` as above.

**Bounds (rule 2.4):** L-BFGS-B projects onto the box and its finite differences step inside it; Nelder-Mead clips its points to the box.

**Keeping going (rule 2.2):** the limits that are only budgets (`maxiter`, `maxfun` / `maxfev`) are lifted. The convergence tests (`ftol` and `gtol` for L-BFGS-B, `xatol` and `fatol` for Nelder-Mead) are the methods' own settings, so they count: each ends an attempt, and `minimize` starts again from a new uniform random point, drawn with the seeds above: the run's seed first, then `(seed + 1) * 1_000_000 + restart`. `de` as above.

**Left out:**
- Powell: the tutorial mentions it as "another optimization algorithm that needs only function calls", after Nelder-Mead.
- BFGS, Newton-CG, trust-region methods: the tutorial's examples give them the gradient or the Hessian, which a black-box benchmark doesn't have; BFGS takes no bounds.
- `dual_annealing`, `direct`: the docs present them for many local minima.
- `shgo`: as above; its docstring example is `rosen` in 5 dimensions.

**Separate tests** (as above):

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| Rosenbrock 10 (500k) | lbfgsb | 5 | 5 | 562 | 0.0057 | 0.0044 | 0.0091 | 0 |
| Rosenbrock 10 (500k) | nelder_mead | 5 | 5 | 12,838 | 0.0098 | 0.0077 | 0.0099 | 0 |
| Rosenbrock 10 (500k) | de | 5 | 5 | 37,062 | 0.0087 | 0.0073 | 0.0092 | 0 |

## Changes from the 0.6.0 benchmark

- `dual_annealing` and `direct` run next to `differential_evolution` on Rastrigin and Ackley; `minimize` with L-BFGS-B and Nelder-Mead on Rosenbrock.
- `differential_evolution` no longer ends a run at its default `maxiter`, or at its convergence test: it starts again (rule 2.2). In 0.6.0 it ended Rastrigin 30 after a median of 450,822 of the 2,000,000 evaluations.
- The run stops at the evaluation that reaches the target: in 0.6.0, the callback stopped after the generation and the polish still ran.
- Each run prints its solution and the evaluated solutions outside the bounds, and the adapter has the `values` command.
- The fitness functions use numpy.
- The restarts' seeds no longer overlap: `seed * 1000 + restart` gave seed 0's restarts the seeds 1 to 9, and the first attempt of seeds 1 to 9 used 1000 to 9000, not the seed itself.
- Each run records its first hit, the evaluation that reached the target (rule 3.3).
- The optimum of Rastrigin and Ackley is further from the origin (rule 1.4).

## Can't run

- OneMax: `differential_evolution` has an `integrality` option for integer variables, but SciPy doesn't present a method for binary strings.
- N-Queens: SciPy has no permutation search.
- The multi-objective scenarios: SciPy's optimizers minimize one objective.

## Bugs found

None.
