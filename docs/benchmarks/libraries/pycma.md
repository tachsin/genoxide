# pycma (Python, 4.5.0)

pycma is the reference Python implementation of CMA-ES by Nikolaus Hansen and co-authors, and the suite's reference for CMA-ES ([rule 6.4](../rules.md#6-the-methods)). Its docs are its docstrings, published as the [API docs](https://cma-es.github.io/apidocs-pycma/) (in particular [`CMAEvolutionStrategy`](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html), with its ask-and-tell interface), and `cma.CMAOptions()`, which lists every option with its default and a description. The source cited below is [CMA-ES/pycma](https://github.com/CMA-ES/pycma/tree/r4.5.0) at tag r4.5.0.

Adapter: [benchmarks/adapters/pycma/](../../../benchmarks/adapters/pycma/).
Know a way to set pycma closer to a definition? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## What it runs

The matched suite: each problem with one method, defined the same for every library ([rules, section 6](../rules.md#6-the-methods)), and each library's own implementation of it.

| Scenario | Method | pycma |
|---|---|---|
| Rosenbrock 10, matched | CMA-ES | `cma_es`: `CMAEvolutionStrategy` in an ask-and-tell loop |
| Rastrigin 30, matched | DE/rand/1/bin | can't run: see [Can't run](#cant-run) |
| OneMax 1000, matched | GA as DEAP's `eaSimple` | can't run: see [Can't run](#cant-run) |

The adapter prints nothing for any other scenario ([`main`, lines 187-189](../../../benchmarks/adapters/pycma/bench.py#L187-L189)). The methods of the earlier suite (`cma.fmin2` with IPOP and BIPOP restarts, lq-CMA-ES) are gone with it.

## How the adapter runs pycma

- **Fitness function:** numpy, taking a generation as the rows of an array, like pycma's own test functions (`cma.ff`) ([lines 103-111](../../../benchmarks/adapters/pycma/bench.py#L103-L111); `values`, [lines 166-173](../../../benchmarks/adapters/pycma/bench.py#L166-L173)).
- **Evaluations and stop:** the counter ([`Budget`, lines 40-100](../../../benchmarks/adapters/pycma/bench.py#L40-L100)) evaluates each generation `es.ask()` returns in one numpy call (rule 3.4: ask-and-tell evaluates a generation at once anyway, so the algorithm is the same). It counts every solution, keeps the best, records the first hit and the solutions outside the bounds (a gene below, above or not a number). It ends the run after the generation that reaches 0.01, or before one past the budget (a generation is cut at the budget) or the time cap.
- **Generations (rule 2.3):** one per `ask` and `tell`, 10 evaluations.
- **Seeds (rule 5.2):** pycma samples from numpy's global random state (option `randn`, `np.random.randn`). The adapter seeds it with the run's seed, `np.random.seed(seed)`, and draws the initial mean from it too. The `seed` option is `np.nan`, "do nothing": pycma reads 0 as "seed from the clock" (`CMAOptions`: "`None` and `0` equate to `time`"). The same seed repeats a run, and seed 1 gives the same alone as after seed 0 (tested).
- **One thread (rule 4.3):** numpy's BLAS set to one thread before the import ([lines 22-24](../../../benchmarks/adapters/pycma/bench.py#L22-L24)).
- **Separate tests:** 2026-09-28, pycma 4.5.0, numpy 2.5.3, Python 3.13.9, the scenario's budget, 60 s cap, on a shared machine. `outside` was 0 in every run.

## Rosenbrock 10: CMA-ES

**Method** ([`solve_cma_es`, lines 123-160](../../../benchmarks/adapters/pycma/bench.py#L123-L160)): `cma.CMAEvolutionStrategy(x0, 4.5, {"bounds": [-5, 10], "CMA_active": False, "maxstd_boundrange": np.inf, "seed": np.nan, ...})`, then `es.tell(X, f(X))` for each `X = es.ask()`, never consulting `es.stop()`. Every other option is pycma's default.

| Definition | pycma | Source |
|---|---|---|
| λ = 4 + ⌊3 ln n⌋ = 10 | default `popsize` `4 + 3 * math.log(N)`, truncated: 10 | `CMAOptions`; [options_parameters.py, lines 929-936](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/options_parameters.py#L929-L936) |
| μ = 5, w_i ∝ ln((λ + 1) / 2) − ln i, positive, sum 1 | default `CMA_mu` (`popsize // 2`) and `RecombinationWeights(popsize)`: 0.4563, 0.2708, 0.1622, 0.0852, 0.0255; μ_eff = 3.167 | [lines 938-946](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/options_parameters.py#L938-L946) |
| no negative weights | `CMA_active=False`: the negative weights of the other 5 are set to 0 ("zero_negative_weights"); the mean uses the positive weights | `CMAOptions`, `CMA_active`; [lines 1037-1042](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/options_parameters.py#L1037-L1042) |
| CSA, rank-one and rank-μ updates, h_σ stall | `CMAAdaptSigmaCSA` (the default `AdaptSigma`), `GaussFullSampler` (full covariance, `CMA_diagonal` 0), with h_σ and the variance-loss correction `c1 * (1 - (1 - hsig**2) * cc * (2 - cc))` | [evolution_strategy.py, lines 2836-2882](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/evolution_strategy.py#L2836-L2882) |
| learning rates and damping: Hansen's | pycma's defaults, below | |
| no mirrored samples | default `CMA_mirrors` `popsize < 6`: none for λ = 10 | `CMAOptions` |
| mean uniform in the box, σ₀ = 4.5, C₀ = I | `x0` drawn with `np.random.uniform(-5, 10, 10)`; `sigma0` 0.3 × 15 = 4.5; C₀ = I | docstring, `x0` and `sigma0` |
| bounds: `BoundTransform` | the reference itself: `bounds` with the default `BoundaryHandler`, `BoundTransform`, which maps every sample into the box before it's evaluated | `CMAOptions`, `BoundaryHandler` |
| no restarts, no convergence criterion | pycma's termination criteria live in `es.stop()`, which the loop never calls; `CMAEvolutionStrategy` has no restarts ("restarts not available with class CMAEvolutionStrategy, use function fmin") | [evolution_strategy.py, lines 1092-1093](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/evolution_strategy.py#L1092-L1093) |

**Learning rates and damping for n = 10, λ = 10** (printed from `es.sp` and `es.adapt_sigma`):

| Constant | pycma | Hansen's 2016 tutorial | pycma's formula |
|---|---|---|---|
| c_σ | 0.3196 | 0.2844 | (μ_eff + 2) / (n + μ_eff + 3), the formula of Hansen's earlier papers ([sigma_adaptation.py, lines 394-423](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/sigma_adaptation.py#L394-L423)) |
| d_σ | 1.3196 | 1.2844 | 1 + c_σ here, as in the tutorial; it differs only through c_σ ([lines 363-393](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/sigma_adaptation.py#L363-L393)) |
| c_c | 0.2950 | 0.2950 | the tutorial's ([options_parameters.py, line 982](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/options_parameters.py#L982-L983)) |
| c_1 | 0.01528 | 0.01528 | the tutorial's, times min(1, λ / 6) = 1 ([lines 996-999](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/options_parameters.py#L996-L999)) |
| c_μ | 0.02355 | 0.02015 | min(1 − c_1, 2 (1/4 + μ_eff + 1/μ_eff − 2) / ((n + 2)² + μ_eff)): the tutorial's with 1/4 added to the numerator (`rankmu_offset`, [lines 1009-1025](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/options_parameters.py#L1009-L1025)) |
| h_σ | | ‖p_σ‖ / √(1 − (1 − c_σ)^(2(g+1))) < (1.4 + 2 / (n + 1)) E‖N(0, I)‖ | ‖p_σ‖² / (1 − (1 − c_σ)^(2g)) / n − 1 < 1 + 4 / (n + 1), on the squared length ([sigma_adaptation.py, lines 73-95](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/sigma_adaptation.py#L73-L95)) |

The weights, μ_eff, c_c and c_1 are the tutorial's; c_σ, d_σ and c_μ are pycma's own defaults, which rule 6.4 keeps.

**Differences:**
- **The learning rates** c_σ and c_μ, and the h_σ test, as above: pycma's defaults, Hansen's own.
- **No cap on the standard deviations.** With bounds, pycma by default caps each coordinate's standard deviation at a third of the box width (`maxstd_boundrange`, "maximal std relative to bound_range per coordinate", [evolution_strategy.py, lines 1178-1183](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/evolution_strategy.py#L1178-L1183)): here 5, against σ₀ = 4.5. It isn't part of the definition, and it acts on the distribution, not on genes that leave the box: it capped a coordinate in the first 20 generations of 26 of 40 test runs, and rescales the coordinates from then on. The adapter turns it off with the documented option, `maxstd_boundrange=np.inf`. For the record, with the cap: 37 of 40 seeds reached the target, median 5,680 evaluations; without it: 37 of 40, median 5,630 (100,000 evaluations at most).
- **Numerical safeguards that don't act here:** σ changes by at most a factor e per generation (`max_delta_log_sigma`, [sigma_adaptation.py, line 533](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/sigma_adaptation.py#L533)); the largest change in the 37 runs that reached the target was a factor 1.30. `conditioncov_alleviate` (default `[1e8, 1e12]`) rescales C once its condition number passes 10⁸; it stayed below 5,500 in those runs, and passes it only in a run that has converged at the local optimum.
- **The initial mean:** pycma keeps its mean in the coordinates before `BoundTransform`, the transformation's inverse of `x0`: the same point, except within a small margin of the bounds. How a uniform initial mean is drawn is an allowed difference.

**Keeping going (rule 2.2):** no criterion ends a run: the loop never calls `es.stop()`, so pycma's termination criteria (`tolfun`, `tolx`, `tolstagnation`, `noeffectaxis`, ...) never act. A run that converges at Rosenbrock's local minimum (3.9866, near (−0.99, 0.99, ..., 0.99)) samples on there until the budget or the time cap, as the definition says. It evaluates every sample, so it never stalls.

**Errors (rule 8.4):** a run pycma ends with an error ends there, not reached, and its line says so in `ended_by` ([lines 150-160](../../../benchmarks/adapters/pycma/bench.py#L150-L160)). Long after converging at the local minimum, the distribution can degenerate until σ is nan and `GaussFullSampler.update` fails its assertion (`AssertionError` in `sampler.py`, line 334): with pycma's default cap (above), seed 25 did so after 160,540 evaluations. With the adapter's settings it didn't happen in 40 seeds.

**Separate tests:** Rosenbrock 10 (budget 500,000, cap 60 s), seeds 0 to 2:

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 3 | 3 | 4,806 | 0.00970 | 0.00887 | 0.00975 | 0 |

Each run took about 0.15 s. With seeds 0 to 39, 37 runs reached the target (median first hit 5,622 evaluations, 2,988 to 8,535); seeds 9, 25 and 30 converged at the local minimum, 3.9866, and sampled on to the budget, in 15 to 18 s.

## Can't run

- **Rastrigin 30, DE/rand/1/bin:** pycma has no differential evolution.
- **OneMax 1000, the GA:** pycma has no genetic algorithm; its `integer_variables` option isn't meant for binary strings.

## Bugs found

None in pycma's intended use. The degeneration above comes from running CMA-ES for thousands of generations after it has converged, which its termination criteria (`noeffectaxis`, `tolflatfitness`, ...) prevent in normal use; the definition turns them off. None were found in the methods no longer in the suite.
