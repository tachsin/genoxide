# SciPy (Python, 1.18.1)

SciPy's [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) minimizes functions of real numbers, with local methods ([`minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)) and global ones ([`differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html), [`dual_annealing`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html), [`direct`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.direct.html), [`shgo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html), `basinhopping`). Its docs are the reference pages and the [optimization tutorial](https://docs.scipy.org/doc/scipy/tutorial/optimize.html).

Adapter: [benchmarks/adapters/scipy/](../../../benchmarks/adapters/scipy/).
Know a better way to solve one of these problems with SciPy? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs SciPy

- **Fitness functions:** numpy, like SciPy's `scipy.optimize.rosen`.
- **Evaluations and stop:** the counter ([`Budget`](../../../benchmarks/adapters/scipy/bench.py#L45-L82)) counts every call, from the method, its restarts, local searches, polish and finite differences, keeps the best and records the first hit. It ends the run right after the evaluation that reaches 0.01, or before one past the budget or the time cap.
- **Seeds:** `rng` for the stochastic methods; restarts as rule 2.2 ([`restart_seed`](../../../benchmarks/adapters/scipy/bench.py#L151-L154)).
- **Separate tests:** 2026-09-25, SciPy 1.18.1, 5 seeds, the scenario's budget, 60 s cap, rule 5.3, with other tests on the machine (capped runs stopped after fewer evaluations than in a benchmark run). `outside` was 0 in every run.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** ([`solvers`](../../../benchmarks/adapters/scipy/bench.py#L219-L231)): the tutorial's [Global optimization](https://docs.scipy.org/doc/scipy/tutorial/optimize.html#global-optimization) shows `shgo`, `dual_annealing` and `differential_evolution` on a function with many local minima, and its table gives `direct`, `dual_annealing`, `differential_evolution` and `shgo` for bounds ("If there are multiple candidates, try several"). Without `shgo` (below):
- **`de`:** [`differential_evolution`](../../../benchmarks/adapters/scipy/bench.py#L132-L148) with its defaults, as its docstring examples call it (on `rosen` and Ackley): `best1bin`, 15 n individuals, mutation (0.5, 1) with dithering, recombination 0.7, Latin hypercube initialization, `updating='immediate'`, the final L-BFGS-B polish.
- **`dual_annealing`:** [`dual_annealing`](../../../benchmarks/adapters/scipy/bench.py#L157-L170) with its defaults; its docstring example is "a 10-D problem, with many local minima ... called Rastrigin". Generalized simulated annealing with an L-BFGS-B local search.
- **`direct`:** [`direct`](../../../benchmarks/adapters/scipy/bench.py#L173-L192) with `locally_biased=False` ("For hard problems with many local minima, `False` is recommended") and `vol_tol=0` ("`vol_tol` should be decreased to avoid premature termination of the algorithm for higher dimensions"; with the default 1e-16 it ended Rastrigin 10 after 1,965 evaluations, at 33). Deterministic: every seed gives the same run.

**Bounds (rule 2.4):** `differential_evolution` replaces a mutant coordinate outside the box by a random one, and polishes within bounds; `dual_annealing` wraps a visit back into the box and bounds its local search; `direct` samples centres of hyperrectangles of the box.

**Keeping going (rule 2.2):**
- `differential_evolution`: `maxiter` (1,000 by default) is lifted. Its convergence test, `tol` 0.01 (the population's value spread at most 0.01 times their mean), counts: it ends the attempt, the polish runs, and it restarts; SciPy has no restart mechanism for it.
- `dual_annealing`: no convergence criterion; it re-anneals by itself from a new random point when its temperature falls to `initial_temp * restart_temp_ratio`. `maxiter` and `maxfun` are lifted (`maxfun` twice the budget).
- `direct`: `maxiter` and `maxfun` are lifted; `maxfun` is twice the budget, because DIRECT stops before an iteration that could pass it (at the budget, it ended Rastrigin 30 after 1,886,939 of 2,000,000). Its `len_tol` (1e-6) counts, but a restart of a deterministic method without a random start would repeat the run; it never converged in the tests, and if it did, the run would end there with a note on stderr.

**Left out:**
- `shgo`: "appropriate for ... global optimality (low-dimensional problems)".
- `basinhopping`: the tutorial's table gives it no bounds (rule 2.4).
- `differential_evolution` with `vectorized=True`: it forces `updating='deferred'`, which changes the algorithm (rule 3.4).
- `polish=False` and other strategies: the defaults run.

**Separate tests:**

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

On Rastrigin 30, `de` reached the cap after about 370,000 evaluations and `direct` after about 780,000.

## Continuous, unimodal: Rosenbrock 10

**Methods:** the tutorial's [local minimization](https://docs.scipy.org/doc/scipy/tutorial/optimize.html#local-minimization-of-multivariate-scalar-functions-minimize) section uses Rosenbrock as its example, and `differential_evolution`'s docstring example is `rosen`:
- **`lbfgsb`:** [`minimize`](../../../benchmarks/adapters/scipy/bench.py#L195-L216) from a uniform random point with the box as `bounds`, so L-BFGS-B, its default then; the gradient is estimated by finite differences, counted. The tutorial says BFGS "typically requires fewer function calls than the simplex algorithm even when the gradient must be estimated"; L-BFGS-B is its bounded form.
- **`nelder_mead`:** `method='Nelder-Mead'`, the tutorial's first Rosenbrock example, with `bounds` and the default tolerances (the example's `xatol: 1e-8` is left at 1e-4).
- **`de`:** as above.

**Bounds (rule 2.4):** L-BFGS-B projects onto the box and steps its finite differences inside it; Nelder-Mead clips its points.

**Keeping going (rule 2.2):** `maxiter`, `maxfun` and `maxfev` are lifted. The convergence tests (`ftol`, `gtol`; `xatol`, `fatol`) end an attempt, and `minimize` restarts from a new random point.

**Left out:**
- Powell: "another optimization algorithm that needs only function calls", after Nelder-Mead in the tutorial.
- BFGS, Newton-CG, trust-region methods: the tutorial gives them the gradient or the Hessian; BFGS takes no bounds.
- `dual_annealing`, `direct`: presented for many local minima.
- `shgo`: as above; its docstring example is `rosen` in 5 dimensions.

**Separate tests:**

| Scenario | Solver | Runs | Reached | First hit: median evaluations | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|---|
| Rosenbrock 10 (500k) | lbfgsb | 5 | 5 | 562 | 0.0057 | 0.0044 | 0.0091 | 0 |
| Rosenbrock 10 (500k) | nelder_mead | 5 | 5 | 12,838 | 0.0098 | 0.0077 | 0.0099 | 0 |
| Rosenbrock 10 (500k) | de | 5 | 5 | 37,062 | 0.0087 | 0.0073 | 0.0092 | 0 |

## Can't run

- OneMax: `differential_evolution` has an `integrality` option, but SciPy presents no method for binary strings.
- N-Queens: no permutation search.
- The multi-objective scenarios: one objective only.

## Bugs found

None.
