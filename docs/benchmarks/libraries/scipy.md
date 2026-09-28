# SciPy (Python, 1.18.1)

SciPy's [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) minimizes functions of real numbers. Its [`differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) is "due to Storn and Price", with a choice of strategies, `rand1bin` among them. Its docs are that reference page (the function's docstring) and the [optimization tutorial](https://docs.scipy.org/doc/scipy/tutorial/optimize.html). The source cited below is [`_differentialevolution.py`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py) at tag v1.18.1.

Adapter: [benchmarks/adapters/scipy/](../../../benchmarks/adapters/scipy/).
Know a way to set SciPy closer to a definition? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## What it runs

The matched suite: each problem with one method, defined the same for every library ([rules, section 6](../rules.md#6-the-methods)), and each library's own implementation of it.

| Scenario | Method | SciPy |
|---|---|---|
| Rastrigin 30, matched: no target, a fixed budget of 300,000 evaluations | DE/rand/1/bin | `de`: `differential_evolution` with `strategy='rand1bin'` |
| Rosenbrock 10, matched | CMA-ES | can't run: see [Can't run](#cant-run) |
| OneMax 1000, matched | GA as DEAP's `eaSimple` | can't run: see [Can't run](#cant-run) |

The adapter prints nothing for any other scenario ([`main`, lines 181-183](../../../benchmarks/adapters/scipy/bench.py#L181-L183)). The methods of the earlier suite (`dual_annealing`, `direct`, `minimize` with L-BFGS-B and Nelder-Mead, `differential_evolution` with its defaults) are gone with it.

## How the adapter runs SciPy

- **Fitness function:** numpy, taking a generation as the columns of an array, as `differential_evolution` passes them with `vectorized=True` and as SciPy's own `scipy.optimize.rosen` accepts them ([lines 92-114](../../../benchmarks/adapters/scipy/bench.py#L92-L114); `values`, [lines 159-167](../../../benchmarks/adapters/scipy/bench.py#L159-L167)).
- **Evaluations and stop:** the objective function is the counter ([`Budget`, lines 41-89](../../../benchmarks/adapters/scipy/bench.py#L41-L89)): it counts every column, keeps the best and counts the solutions outside the bounds. Rastrigin 30 has no target: the run is measured by its time for the budget and its error at the end (the best value; the optimum is 0). The counter ends the run before a generation past the budget (a generation is cut at the budget) or the time cap. The line always has `"target": null`, `"success": false` and `"first_hit": null`.
- **Batch evaluation (rule 3.4):** `vectorized=True`, SciPy's documented vectorized interface: each generation in one numpy call. It forces `updating='deferred'`, which the definition sets anyway, so it doesn't change the algorithm: a run gives the same evaluations and best value with it as without it (tested with seeds 0 and 1, 20,000 evaluations).
- **Generations (rule 2.3):** counted by `differential_evolution`'s `callback`, called after every generation.
- **Seeds (rule 5.2):** a `numpy.random.Generator`, `np.random.default_rng(seed)`, draws the initial population and is then passed as `rng`. The same seed repeats a run, and seed 1 gives the same alone as after seed 0 (tested).
- **One thread (rule 4.3):** numpy's BLAS set to one thread before the import ([lines 24-26](../../../benchmarks/adapters/scipy/bench.py#L24-L26)); `workers` stays 1.
- **Separate tests:** 2026-09-28, SciPy 1.18.1, numpy 2.5.3, Python 3.13.9, seeds 0 to 2, the scenario's budget of 300,000 evaluations, 60 s cap, on a shared machine. `outside` was 0 in every run.
- **Speed (checked 2026-09-29).** A profile (cProfile) of a run: about 56% is `_mutate_many` building the trials (42% its `_select_samples`, a Python call per trial), 17% evaluating the population, of which the fitness function is 10% and the adapter's counting 3% (about 7 µs a generation, 20 ms a run), and the rest `differential_evolution`'s own bookkeeping per generation. The adapter already uses the documented fastest setting, `vectorized=True`; checked again on 2026-09-29: seeds 0 to 2 (20,000 evaluations) give the same evaluations and best values with `vectorized=False`, in 2.3 to 2.9 times the time. `workers` would run evaluations in other processes (rule 4.3). Nothing changed.

## Rastrigin 30: DE/rand/1/bin

**Method** ([`solve_de`, lines 122-153](../../../benchmarks/adapters/scipy/bench.py#L122-L153)): `differential_evolution(func, bounds, strategy='rand1bin', mutation=0.5, recombination=0.9, init=population, updating='deferred', vectorized=True, polish=False, tol=0, atol=0, maxiter=budget, rng=rng, callback=...)`, every keyword a documented parameter.

| Definition | SciPy | Source |
|---|---|---|
| NP = 100, uniform in the box | `init=` an array of 100 points drawn with `rng.uniform(lower, upper, (100, 30))`: "array specifying the initial population. The array should have shape (S, N), where S is the total population size". `popsize` can't give 100: it's a multiplier of N = 30, and it "is overridden if an initial population is supplied via the `init` keyword" | docs, `init` and `popsize`; [`init_population_array`, lines 1121-1154](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L1121-L1154) |
| r1, r2, r3 uniform, distinct, ≠ i | `_select_samples` shuffles the indices and takes the first ones other than the target's | [lines 1908-1915](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L1908-L1915) |
| v = x_r1 + F (x_r2 − x_r3), F = 0.5, no dither | `strategy='rand1bin'`: `population[r0] + scale * (population[r1] - population[r2])`; `mutation=0.5`, a float: "If specified as a tuple `(min, max)` dithering is employed", so a float isn't dithered | docs, `strategy` and `mutation`; [`_rand1`, lines 1867-1871](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L1867-L1871), [lines 871-874](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L871-L874) |
| binomial, CR = 0.9, one forced j_rand | `recombination=0.9`; gene j from the mutant if U(0, 1) < CR, and `fill_point`, uniform, always from the mutant: "A randomly selected parameter is always loaded from b'" | docs, Notes; [`_mutate_many`, lines 1785-1796](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L1785-L1796) |
| u replaces x_i if f(u) ≤ f(x_i) | `energy_trial <= energy_orig` | [`_accept_trial`, lines 1590-1591](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L1590-L1591) |
| generational | `updating='deferred'`: "To use the original Storn and Price behaviour, updating the best solution once per iteration, set `updating='deferred'`": all trials are built from the population, evaluated, then replace their targets | docs, Notes; [lines 1679-1722](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L1679-L1722) |
| bounds: a gene outside redrawn uniformly in the box | the reference itself: `_ensure_constraint` redraws every trial gene outside the box uniformly in it | [lines 1740-1744](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L1740-L1744) |
| no archive, adaptation or restarts | none; `polish=False` turns off the final L-BFGS-B polish, a local search the definition doesn't have | docs, `polish` |
| no convergence criterion | `maxiter`, a budget, is set to the scenario's budget; the convergence test can't be turned off, `tol=0, atol=0` (below) | docs, `maxiter`, `tol`, `atol` |

**Differences:**
- **The convergence test (rule 2.2).** `solve` ends a run when `np.std(population_energies) <= atol + tol * np.abs(np.mean(population_energies))`, after every generation ([lines 1174-1183](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L1174-L1183), [lines 1255-1257](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L1255-L1257)); no setting turns it off. Its tightest documented setting, `tol=0` and `atol=0`, ends a run only when all 100 values are equal: the population has collapsed onto one point, and DE, with every difference vector zero, can no longer move. The run ends there, before its budget, and its line says so: `"ended_by": "convergence test (tol=0, atol=0): all 100 values equal"`. No restart (rule 2.2): the definition's DE would stay at that point until the budget, with the same error at the end. Within 300,000 evaluations it didn't happen in the separate tests; with 2,000,000, in an earlier test, 2 of 3 runs collapsed onto a local minimum, after 705,500 and 870,600 evaluations.
- **Order:** after every generation SciPy swaps the best individual into position 0 (`_promote_lowest_energy`, [lines 1423-1441](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L1423-L1441)). Every individual is still a target once per generation, and r1, r2, r3 are uniform among the others, so the algorithm is the same; only which random numbers go with which individual differs.

**Keeping going:** `maxiter` is lifted; only the convergence test can end a run before the budget or the time cap, as above. DE evaluates every trial, so it never stalls.

**Separate tests:** Rastrigin 30 (no target, budget 300,000, cap 60 s):

| Solver | Runs | Time per run: median (s) | Error at the end: median | best | worst | Ended by the convergence test | Capped |
|---|---|---|---|---|---|---|---|
| de | 3 | 1.14 | 138.5 | 93.0 | 156.8 | 0 | 0 |

Every run used the whole budget: 1.14, 1.16 and 1.12 s (seeds 0, 1, 2), ending at 156.8, 93.0 and 138.5.

## Can't run

- **Rosenbrock 10, CMA-ES:** SciPy has no CMA-ES.
- **OneMax 1000, the GA:** SciPy has no genetic algorithm and no method for binary strings (`differential_evolution`'s `integrality` rounds real genes; it isn't a GA).

## Bugs found

None in `differential_evolution`. One docs inaccuracy: the Notes say "If the trial is better than the original candidate then it takes its place", but the code accepts an equal trial too (`<=`, [line 1591](https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_differentialevolution.py#L1591)), as the definition does. None were found in the methods no longer in the suite.
