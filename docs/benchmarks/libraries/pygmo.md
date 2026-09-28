# pygmo (C++ via Python, 2.19.8)

pygmo is the Python interface of pagmo, ESA's C++ library of optimization algorithms built around the island model; the 2.19.8 wheel bundles pagmo 2.19.1. The algorithms run in C++ and call a Python user-defined problem (UDP) for every evaluation. Its docs are at [esa.github.io/pygmo2](https://esa.github.io/pygmo2/), with the [list of algorithms](https://esa.github.io/pygmo2/overview.html#list-of-algorithms); pagmo's C++ docs are at [esa.github.io/pagmo2](https://esa.github.io/pagmo2/). The source links below are to pagmo [v2.19.1](https://github.com/esa/pagmo2/tree/v2.19.1).

Adapter: [benchmarks/adapters/pygmo/bench.py](../../../benchmarks/adapters/pygmo/bench.py).
Can pygmo be set closer to a definition than this page says? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## What it runs

| Problem | Method ([rules, section 6](../rules.md#6-the-methods)) | pygmo | Solver |
|---|---|---|---|
| OneMax 1000 | the GA | can't run it (below) | |
| Rastrigin 30 (no target, 300,000 evaluations) | DE/rand/1/bin | [`de`](https://esa.github.io/pygmo2/algorithms.html#pygmo.de), variant 7 | `de` |
| Rosenbrock 10 | CMA-ES | [`cmaes`](https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes) | `cma_es` |

## How the adapter runs pygmo

- **Fitness functions:** a UDP with `fitness(x)` and `batch_fitness(dvs)` ([`Problem`](../../../benchmarks/adapters/pygmo/bench.py#L192-L228)), in numpy, as [coding_udp_simple](https://esa.github.io/pygmo2/tutorials/coding_udp_simple.html) shows: `fitness` gets one decision vector, a numpy array, and returns a list of one value ([bench.py#L153-L189](../../../benchmarks/adapters/pygmo/bench.py#L153-L189)). Each function has two forms, of one vector for `fitness` and of the rows of an array for `batch_fitness`; they give the same value to the bit, which `values` checks for every point `run.py check` evaluates.
- **Batch evaluation (rule 3.4):** `cmaes` accepts a batch fitness evaluator and gets [`member_bfe`](https://esa.github.io/pygmo2/bfe.html#pygmo.member_bfe), which calls `batch_fitness` with a generation, in the same thread. It's the same search: with and without it, seeds 0 to 2 gave the same first hits and best values. `de` takes no evaluator: in pygmo 2.19.8 only `cmaes`, `gaco`, `maco`, `moead_gen`, `nsga2`, `nspso` and `pso_gen` have `set_bfe`. So `de` calls `fitness` once per evaluation, 300,000 times a run.
- **Evaluations:** the adapter's [`Counter`](../../../benchmarks/adapters/pygmo/bench.py#L80-L130) counts every decision vector, keeps the best and records the first hit; `get_fevals` isn't used.
- **The adapter's own cost (2026-09-29).** A profile of `de` showed most of a run in the adapter, not in pagmo or the fitness function: `fitness` wrapped each vector in an array of one row, evaluated it with the batch form, and counted it with array operations, about 9 µs a call, where pagmo's own call costs about 1 µs and the fitness function 2 µs. `fitness` now evaluates the vector itself, counts it with plain floats, and tests the bounds with its minimum and maximum; `batch_fitness` tests them the same way before it counts the rows outside. The search is unchanged: seeds 0 to 2 of both methods give the same evaluations, best values, solutions and first hits as before. A solution with a gene that isn't a number now counts as outside, as it does for pycma and pymoo; none occurred. Times of seeds 0 to 2, in WSL on 2 cores shared with other work, from 3 to 5 runs of each seed alternating the adapters before and after:

  | Solver | Before | After |
  |---|---|---|
  | de, Rastrigin 30, 300,000 evaluations | 2.4 to 2.7 s | 1.1 to 1.2 s |
  | cma_es, Rosenbrock 10, to the target | 10 to 12 ms | 7 to 9 ms |
- **Stop:** the counter raises an exception from inside the fitness at the target (Rosenbrock only; Rastrigin has none), the budget (a batch is cut there) or the 60 s cap; it passes through pagmo's C++ to the adapter.
- **No convergence criterion (rule 2.2):** `gen`, a limit that's only a budget, covers the whole budget ([`budget_generations`](../../../benchmarks/adapters/pygmo/bench.py#L231-L234)). `ftol` and `xtol` are 0: both algorithms stop when a spread is *below* them (`dx < xtol`, `df < ftol` in `de`; the last step's length `< xtol` and the population's `df < ftol` in `cmaes`), which a spread of 0 isn't. One call of `evolve` runs to the end; there are no restarts, so no restart seeds.
- **Errors (rule 8.4):** any other exception from `evolve` ends the run, reported in `ended_by` ([bench.py#L329-L338](../../../benchmarks/adapters/pygmo/bench.py#L329-L338)). None happened in the tests.
- **One thread:** no islands, archipelagos or parallel evaluators; numpy's BLAS with 1 thread ([bench.py#L33-L35](../../../benchmarks/adapters/pygmo/bench.py#L33-L35)). pagmo's `batch_fitness` also runs TBB worker threads, with no pygmo setting for them: the adapter starts TBB with one CPU, by a batch evaluation of a trivial problem before any run, and then restores the process's CPUs ([`start_tbb_with_one_thread`](../../../benchmarks/adapters/pygmo/bench.py#L61-L73)).
- **Seeds:** the run's seed goes to `pg.population(..., seed=)` and to the algorithm's `seed`.
- **`last_generation`:** pagmo doesn't call back between generations, but both algorithms evaluate a population of their size first and then that many a generation ([`last_generation`](../../../benchmarks/adapters/pygmo/bench.py#L136-L140)).
- **Separate tests:** 2026-09-29, pygmo 2.19.8, seeds 0 to 2, the scenario's budget (300,000 for Rastrigin 30, 500,000 for Rosenbrock 10) and 60 s cap, with other processes on the machine. `outside` was 0 in every run.

## Rastrigin 30: DE/rand/1/bin

**Code:** [`run_de`](../../../benchmarks/adapters/pygmo/bench.py#L242-L256): `pg.de(gen, F=0.5, CR=0.9, variant=7, ftol=0.0, xtol=0.0, seed)` evolving `pg.population(problem, 100, seed)`.

**No target:** Rastrigin 30 is measured by the time a run takes for its budget of 300,000 evaluations and by its error at the end. The counter never stops a run at a value, only at the budget or the time cap ([`solvers_of`](../../../benchmarks/adapters/pygmo/bench.py#L278-L288), [bench.py#L327-L328](../../../benchmarks/adapters/pygmo/bench.py#L327-L328)), and every run prints `"target": null`, `"success": false` and `"first_hit": null`.

**Configuration** ([rule 6.3](../rules.md#6-the-methods)):

| Definition | pygmo | Source |
|---|---|---|
| NP = 100, uniform in the box | a population of 100; each decision vector drawn uniformly in the bounds | `pg.population`; [population.cpp#L62-L78](https://github.com/esa/pagmo2/blob/v2.19.1/src/population.cpp#L62-L78) |
| v = x_r1 + F (x_r2 − x_r3), F = 0.5 fixed | `variant=7`, "rand/1/bin" in the docstring; `F=0.5`, fixed for the run | [de.cpp#L229-L239](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/de.cpp#L229-L239) |
| r1, r2, r3 distinct, different from i | 5 distinct indices drawn from the whole population, i included: a difference, below | [de.cpp#L142-L149](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/de.cpp#L142-L149) |
| Binomial crossover, CR = 0.9, one j_rand | `CR=0.9`; the loop over the genes starts at a uniform index and always takes the last gene it visits ("change at least one parameter"), which is uniform too: the same as j_rand | [de.cpp#L231-L238](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/de.cpp#L231-L238) |
| A trial gene outside the box drawn again uniformly | `force_bounds_random`: the reference repair ("done by creating a random number in the bounds", [de docs](https://esa.github.io/pagmo2/docs/cpp/algorithms/de.html)) | [de.cpp#L279](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/de.cpp#L279) |
| u replaces x_i if f(u) ≤ f(x_i) | `newfitness[0] <= fit[i][0]` | [de.cpp#L281-L295](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/de.cpp#L281-L295) |
| Generational | the mutants are built from `popold`, the previous generation; the replacements go to `popnew`, swapped in at the end of the generation | [de.cpp#L297-L300](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/de.cpp#L297-L300) |
| No archive, adaptation, restarts or convergence criterion | none in `de`; `ftol` and `xtol` 0; `gen` covers the budget | [de.cpp#L302-L322](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/de.cpp#L302-L322) |

**Differences:**
- **x_i can be one of r1, r2, r3.** `de` draws its indices (five, of which rand/1 uses three) from the whole population, without excluding i, so 3% of the trials use x_i as the base or a difference vector. Storn and Price's definition, and their code that pagmo says it's based on, draw them different from i. It doesn't change the algorithm: the mutation, the crossover, the rates, the selection and the replacement are rand/1/bin's, and 97% of the trials draw their indices as defined.

**Bounds (rule 2.4):** the reference repair, above.

**Separate tests** (no target, budget 300,000; the error is the best value at the end, the optimum being 0):

| Solver | Runs | Time: median (fastest, slowest) | Error at the end: median (best, worst) | Capped |
|---|---|---|---|---|
| de | 3 | 1.14 s (1.10, 1.16) | 142 (115, 182) | 0 |

Every run used the whole budget: 3,000 generations of 100. The times are the medians of 3 runs of each seed with the adapter of 2026-09-29; the adapter before it took 2.43 s (2.35, 2.49) on 2026-09-28, for the same runs.

## Rosenbrock 10: CMA-ES

**Code:** [`run_cma_es`](../../../benchmarks/adapters/pygmo/bench.py#L259-L275): `pg.cmaes(gen, sigma0=0.3, ftol=0.0, xtol=0.0, force_bounds=True, seed)` with `member_bfe`, evolving `pg.population(problem, 10, seed)`. The other parameters are at their defaults: `cc`, `cs`, `c1` and `cmu` at −1 ("automatically assigned"), `memory=False` (one call of `evolve`).

**Configuration** ([rule 6.4](../rules.md#6-the-methods)):

| Definition | pygmo | Source |
|---|---|---|
| λ = 10 | the population's size, 10 | [cmaes.cpp#L120](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/cmaes.cpp#L120) |
| μ = 5, w_i ∝ ln((λ + 1) / 2) − ln i, positive, summing to 1 | `mu = lam / 2`; `weights(i) = log(mu + 0.5) - log(i + 1)`, normalized; no negative weights (no active CMA) | [cmaes.cpp#L121](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/cmaes.cpp#L121), [#L165-L170](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/cmaes.cpp#L165-L170) |
| CSA, rank-one and rank-μ updates, h_σ, Hansen's rates | the defaults of `cc`, `cs`, `c1`, `cmu` and the damping, table below | [cmaes.cpp#L173-L191](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/cmaes.cpp#L173-L191), [#L367-L383](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/cmaes.cpp#L367-L383) |
| Mean uniform in the box | the best of the population's 10 points, each uniform in the bounds: a difference, below | [cmaes.cpp#L206-L209](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/cmaes.cpp#L206-L209) |
| σ₀ = 4.5, C₀ = I | `sigma0=0.3`: `cmaes` starts from C₀ = diag(width²), so its steps are σ₀ × width = 0.3 × 15 = 4.5. The same distribution, and the same updates: CMA-ES scales σ and √C the same way | [cmaes.cpp#L213-L226](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/cmaes.cpp#L213-L226) |
| Every evaluated solution in the box | `force_bounds=True`, clipping: a difference, below | [cmaes.cpp#L298-L311](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/cmaes.cpp#L298-L311) |
| No restarts, no convergence criterion | `cmaes` has no restarts; `ftol` and `xtol` 0; `gen` covers the budget | [cmaes.cpp#L255-L274](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/cmaes.cpp#L255-L274) |

**Learning rates and damping,** n = 10, λ = 10, μ_eff = 3.167:

| Constant | Hansen's 2016 tutorial | pygmo |
|---|---|---|
| c_σ | (μ_eff + 2) / (n + μ_eff + 5) = 0.2844 | the same, 0.2844 |
| d_σ | 1 + 2 max(0, √((μ_eff − 1) / (n + 1)) − 1) + c_σ = 1.2844 | the same, 1.2844 |
| c_c | (4 + μ_eff / n) / (n + 4 + 2 μ_eff / n) = 0.2950 | the same, 0.2950 |
| c_1 | 2 / ((n + 1.3)² + μ_eff) = 0.01528 | the same, 0.01528 |
| c_μ | min(1 − c_1, 2 (μ_eff − 2 + 1 / μ_eff) / ((n + 2)² + μ_eff)) = 0.02015 | the same without the min (it doesn't bind), 0.02015 |
| h_σ | 1 if ‖p_σ‖ / √(1 − (1 − c_σ)^(2(g+1))) < (1.4 + 2 / (n + 1)) E‖N(0, I)‖ = 4.879 | the form of Hansen's 2006 tutorial, ‖p_σ‖² / n / (1 − (1 − c_σ)^(2(g+1))) < 2 + 4 / (n + 1): a threshold of 4.862 |
| σ update | σ exp((c_σ / d_σ)(‖p_σ‖ / E‖N(0, I)‖ − 1)) | the same, with the exponent capped at 0.6 |
| Eigendecomposition | every generation, or lazily | lazily, every λ / (c_1 + c_μ) / n / 10 = 2.8 evaluations: every generation here |

**Differences:**
- **The initial mean is the best of 10 uniform points,** not one uniform point: `cmaes` starts at the best individual of the population it evolves, and that population is 10 points drawn uniformly in the box, so they cost 10 evaluations (counted). It's still a uniform draw, of 10 points instead of 1, from which the search starts at the best.
- **h_σ** is Hansen's earlier published form, with a threshold of 4.862 instead of 4.879 for the corrected length of p_σ.
- **The step-size change is capped** at a factor e^0.6 per generation. The cap acts only when ‖p_σ‖ > 11.4, 3.7 times its expected length.
- **Bounds:** `force_bounds=True`, `cmaes`'s only bound handling, clips each gene outside the box to the bound before the sample is evaluated; genes inside aren't changed. The clipped sample is also the one the update uses (pycma's `BoundTransform` evaluates a transformed point and updates with the sample itself). In seeds 0 to 2, 1.0 to 1.3% of the evaluated samples had a gene on a bound (63 of 5,120, 54 of 5,980, 68 of 5,190). pagmo's source warns that clipping "screws up the whole covariance matrix machinery and worsen performances considerably"; with so few samples outside, the runs below reach the target in about 5,000 evaluations.

**Separate tests** (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Capped |
|---|---|---|---|---|---|
| cma_es | 3 | 3 | 5,187 | 0.00928 (0.00879, 0.00958) | 0 |

Seeds 3 to 9 too: seeds 3 and 8 converged to Rosenbrock's local minimum (3.99, near x₁ = −1) and sampled around it to the end of the budget, 500,000 evaluations in under a second; the other five reached the target in 4,308 to 5,441 evaluations.

## Can't run

- **OneMax 1000:** pagmo's only GA, [`sga`](https://esa.github.io/pygmo2/algorithms.html#pygmo.sga), has elitist reinsertion that can't be turned off ("the only reinsertion strategy provided is what we call pure elitism"), and no two-point crossover (exponential, binomial, single-point or SBX).

## Bugs found

- **`de` doesn't exclude the target from r1, r2, r3** (above): its [index draw](https://github.com/esa/pagmo2/blob/v2.19.1/src/algorithms/de.cpp#L142-L149) takes five distinct indices from the whole population. pagmo's docs say the implementation "is based on the code provided in the official DE web site", whose indices differ from the target. Effect: 3% of the trials use x_i as the base or a difference vector. Kept as it is (rule 8.4). Reported: [esa/pagmo2#653](https://github.com/esa/pagmo2/issues/653), fix proposed in [esa/pagmo2#654](https://github.com/esa/pagmo2/pull/654).
- **Documentation:** the docstring of `cmaes` describes `ftol` as "stopping criteria on the x tolerance" and `xtol` as "on the f tolerance", swapped; the code checks them the right way round. So do those of `sade`, `de1220` and `xnes`, which were in earlier versions of the benchmark. Reported: [esa/pygmo2#195](https://github.com/esa/pygmo2/issues/195), fix proposed in [esa/pygmo2#196](https://github.com/esa/pygmo2/pull/196) and [esa/pagmo2#652](https://github.com/esa/pagmo2/pull/652).
