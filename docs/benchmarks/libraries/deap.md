# DEAP (Python, 1.4.4)

DEAP (Distributed Evolutionary Algorithms in Python) is a framework of building blocks: the user creates the individual and fitness types (`creator`), registers operators from `tools` in a `Toolbox`, and writes the loop or takes one from `algorithms`; CMA-ES is in `cma`. Its docs are [deap.readthedocs.io](https://deap.readthedocs.io/en/master/), whose [examples pages](https://deap.readthedocs.io/en/master/examples/index.html) come from the `examples/` folder of [DEAP/deap](https://github.com/DEAP/deap). The repository has no 1.4.4 tag; the citations are to commit [8a96fd3](https://github.com/DEAP/deap/tree/8a96fd3a75026f7b30e835f595a5199c75634ddf), "Bump version to 1.4.4".

Adapter: [benchmarks/adapters/deap/bench.py](../../../benchmarks/adapters/deap/bench.py).
Can DEAP be set closer to a definition than this page says? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## What it runs

| Problem | Method ([rules, section 6](../rules.md#6-the-methods)) | DEAP | Solver |
|---|---|---|---|
| OneMax 1000 | the GA | `algorithms.eaSimple` with the OneMax example's operators and `array.array` individuals: the reference | `ga` |
| Rastrigin 30 | DE/rand/1/bin | can't run it (below) | |
| Rosenbrock 10 | CMA-ES | `cma.Strategy` with `algorithms.eaGenerateUpdate` | `cma_es` |

## How the adapter runs DEAP

- **Fitness functions:** plain Python on one individual, returning a tuple, as DEAP's examples and `deap.benchmarks` write them ([bench.py#L125-L138](../../../benchmarks/adapters/deap/bench.py#L125-L138)). DEAP evaluates one individual per call, so there's no batch interface (rule 3.4).
- **Evaluations:** a wrapper counts every call ([`Budget`](../../../benchmarks/adapters/deap/bench.py#L44-L116)) and records the first hit ([`wrap`](../../../benchmarks/adapters/deap/bench.py#L80-L94)).
- **Stop:** the target and the time cap between generations. The GA checks the budget before each evaluation, so it never passes it ([`full`](../../../benchmarks/adapters/deap/bench.py#L107-L110)); the CMA-ES evaluates a generation whole, at most 9 evaluations past the budget (rule 2.3).
- **No convergence criterion (rule 2.2):** `eaSimple` and `eaGenerateUpdate` stop only after `ngen` generations, a limit that's only a budget: the adapter's loops run to the target, the budget or the time cap. The GA starts a new attempt from a new random population, with the next restart seed, after 10 generations in a row without an evaluation (a stall); the run prints `restarts`. None of the tests below stalled. `cma.Strategy` has no stop criterion and no restarts.
- **Errors (rule 8.4):** `cma.Strategy.update` raises `numpy.linalg.LinAlgError` if its covariance matrix degenerates; that ends the run, reported in `ended_by` ([bench.py#L257-L263](../../../benchmarks/adapters/deap/bench.py#L257-L263)). It didn't happen in the tests, not even in runs that sampled around a local minimum to the end of the budget.
- **One thread:** numpy's BLAS set to one thread before import ([bench.py#L20-L22](../../../benchmarks/adapters/deap/bench.py#L20-L22)); DEAP evaluates in the calling thread.
- **Seeds:** `random` (the GA's operators) and `numpy.random` (the CMA-ES), seeded with the run's seed before each run ([bench.py#L308-L309](../../../benchmarks/adapters/deap/bench.py#L308-L309)).
- **Separate tests:** 2026-09-29, DEAP 1.4.4, numpy 2.5.3, Python 3.13.9, seeds 0 to 2, the scenario's budget and 60 s cap, with other processes on the machine.

## OneMax 1000: the GA

**Code:** [`solve_onemax`](../../../benchmarks/adapters/deap/bench.py#L191-L207) and [`ea_simple`](../../../benchmarks/adapters/deap/bench.py#L146-L188): [examples/ga/onemax.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/onemax.py) ([docs](https://deap.readthedocs.io/en/master/examples/ga_onemax.html)) with the loop of `algorithms.eaSimple` and the individuals of [onemax_short.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/onemax_short.py) ([docs](https://deap.readthedocs.io/en/master/examples/ga_onemax_short.html)), `array.array`s. The matched GA is defined from these examples ([rule 6.2](../rules.md#6-the-methods)).

**Configuration:**

| Definition | DEAP | Source |
|---|---|---|
| 300 individuals of 1000 uniform bits | `tools.initRepeat` of `random.randint(0, 1)`, 300 individuals, each a `creator` type of `array.array` with typecode `"b"` | onemax_short.py; `array.array` individuals: [Creating Types](https://deap.readthedocs.io/en/master/tutorials/basic/part1.html) |
| 300 tournaments of 3, with replacement | `tools.selTournament(tournsize=3)`, `k` the population's size: 3 drawn by `selRandom` (with replacement), the best wins | [selection.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/deap/tools/selection.py) |
| Pairs crossed with probability 0.5, two-point | `algorithms.varAnd` with `cxpb=0.5`, `tools.cxTwoPoint` | [algorithms.py, `varAnd`](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/deap/algorithms.py#L33-L82) |
| Children mutated with probability 0.2, bits flipped with probability 1/1000 | `mutpb=0.2`, `tools.mutFlipBit(indpb=1/1000)` (the example's `indpb` is 0.05; the definition takes 1 / n) | onemax.py |
| Generational, no elitism | `population[:] = offspring` | [algorithms.py, `eaSimple`](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/deap/algorithms.py#L85-L189) |
| A child neither crossed nor mutated isn't evaluated again | `varAnd` keeps its fitness valid; only the invalid ones are evaluated | `eaSimple` |

**Differences:** none. The adapter writes `eaSimple`'s loop (select, `varAnd`, evaluate the invalid, replace) instead of calling it, so as to stop at the target, the budget and the time cap and to detect a stall; `eaSimple`'s statistics and hall of fame are left out.

**The individuals: `array.array`, not `list`.** Until 2026-09-29 the adapter took onemax.py's individuals, Python lists. Profiling seed 0 showed 96% of the run in `copy.deepcopy`: `varAnd` clones the 300 selected individuals every generation with `toolbox.clone`, a deep copy, which copies a list of 1000 ints item by item. DEAP documents three kinds of individual for OneMax: a list (onemax.py), an `array.array` of typecode `"b"` ([onemax_short.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/onemax_short.py), the example of `eaSimple`) and a `numpy.ndarray` ([onemax_numpy.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/onemax_numpy.py), [docs](https://deap.readthedocs.io/en/master/examples/ga_onemax_numpy.html)). Its docs prefer `array.array` for speed ([Inheriting from Numpy, "Performance"](https://deap.readthedocs.io/en/master/tutorials/advanced/numpy.html#performance): "using an `array.array` should be preferred to `numpy.ndarray`", whose creation, also needed by the deep copy, is slower). `creator` gives an `array.array` type a deep copy that copies the array at once ([creator.py, `_array`](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/deap/creator.py)). The three run the same algorithm with the same operators and the same random draws, so the same search: seeds 0 to 2 make the same evaluations and first hits with each. Only the time differs, per seed (WSL, other processes running, the best of up to three runs):

| Individuals | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| `list` (onemax.py), before | 16.1 s | 26.5 s | 16.1 s |
| `numpy.ndarray` (onemax_numpy.py), fitness `sum(individual)` as the example writes it | 5.95 s | 6.14 s | 6.29 s |
| `numpy.ndarray`, fitness `individual.sum()` | 1.98 s | 1.95 s | 1.95 s |
| `array.array` (onemax_short.py), now | 1.50 s | 1.54 s | 1.52 s |

The published runs, with lists, took a median of 15.2 s on the pinned cores. `array.array` is the fastest, needs no operator of the adapter's own (the numpy example needs its own two-point crossover, `cxTwoPointCopy`, written in the example, because `tools.cxTwoPoint` swaps numpy views wrongly), and keeps onemax.py's fitness function, `sum(individual)`. What remains is DEAP's own work (profile of seed 0): `mutFlipBit`, which draws a random number per bit of every mutated child, takes about half of the run, the deep copies a fifth and the tournaments a tenth; the fitness function takes about 3%, the adapter's counter under 1%.

**Separate tests** (budget 2,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Capped |
|---|---|---|---|---|---|
| ga | 3 | 3 | 104,677 | 1000 (1000, 1000) | 0 |

With `array.array` individuals, the runs took 1.5 to 1.6 s, and made the same evaluations and first hits as the published runs, with lists (seed 0: 102,333 evaluations in 566 generations, first hit at 102,188; seed 1: 105,844 in 587, first hit at 105,722; seed 2: 104,730 in 579, first hit at 104,677).

## Rosenbrock 10: CMA-ES

**Code:** [`solve_cma_es`](../../../benchmarks/adapters/deap/bench.py#L240-L264) and [`bounded_evaluate`](../../../benchmarks/adapters/deap/bench.py#L215-L237): a [`cma.Strategy`](https://deap.readthedocs.io/en/master/api/algo.html#deap.cma.Strategy) whose `generate` and `update` are registered in the toolbox and run by [`algorithms.eaGenerateUpdate`](https://deap.readthedocs.io/en/master/api/algo.html#deap.algorithms.eaGenerateUpdate), as DEAP's CMA-ES example does ([cma_minfct.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_minfct.py), [docs](https://deap.readthedocs.io/en/master/examples/cmaes.html)). The adapter calls `eaGenerateUpdate` with `ngen=1` once per generation, so as to stop at the target, the budget or the time cap; the strategy keeps its state between calls, so it's the same run as one call.

**Configuration** ([rule 6.4](../rules.md#6-the-methods)):

| Definition | DEAP | Source |
|---|---|---|
| λ = 10 | `lambda_`'s default, `int(4 + 3 * log(N))` = 10 | [cma.py#L110](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/deap/cma.py#L110) |
| μ = 5, w_i ∝ ln((λ + 1) / 2) − ln i, positive, summing to 1 | `mu`'s default, `int(lambda_ / 2)` = 5; `weights`' default, `"superlinear"`: `log(mu + 0.5) - log(i)`, normalized. No negative weights | [cma.py#L182-L195](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/deap/cma.py#L182-L195) |
| CSA, rank-one and rank-μ updates, h_σ, Hansen's rates | `update`, with the defaults of `ccum`, `cs`, `ccov1`, `ccovmu` and `damps`, table below | [cma.py#L126-L208](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/deap/cma.py#L126-L208) |
| Mean uniform in the box, σ₀ = 4.5, C₀ = I | `centroid=numpy.random.uniform(low, high, size)`, as DEAP's BIPOP example draws its centroids; `sigma=0.3 * (high - low)` = 4.5; `cmatrix`'s default, the identity | [bench.py#L253](../../../benchmarks/adapters/deap/bench.py#L253) |
| Every evaluated solution in the box | `tools.ClosestValidPenalty`: a difference, below | [bench.py#L215-L237](../../../benchmarks/adapters/deap/bench.py#L215-L237) |
| No restarts, no convergence criterion | `cma.Strategy` has neither | |

**Learning rates and damping,** n = 10, λ = 10, μ_eff = 3.167 (the [`Strategy` docs](https://deap.readthedocs.io/en/master/api/algo.html#deap.cma.Strategy) list the formulas):

| Constant | Hansen's 2016 tutorial | DEAP |
|---|---|---|
| c_σ | (μ_eff + 2) / (n + μ_eff + 5) = 0.2844 | `cs`: (μ_eff + 2) / (n + μ_eff + 3) = 0.3196, as pycma's |
| d_σ | 1 + 2 max(0, √((μ_eff − 1) / (n + 1)) − 1) + c_σ = 1.2844 | `damps`: the same formula with its c_σ, 1.3196 |
| c_c | (4 + μ_eff / n) / (n + 4 + 2 μ_eff / n) = 0.2950 | `ccum`: 4 / (n + 4) = 0.2857, Hansen and Ostermeier's (2001) |
| c_1 | 2 / ((n + 1.3)² + μ_eff) = 0.01528 | `ccov1`: the same, 0.01528 |
| c_μ | min(1 − c_1, 2 (μ_eff − 2 + 1 / μ_eff) / ((n + 2)² + μ_eff)) = 0.02015 | `ccovmu`: the same, 0.02015 |
| h_σ | ‖p_σ‖ / √(1 − (1 − c_σ)^(2(g+1))) < (1.4 + 2 / (n + 1)) E‖N(0, I)‖ | the same |
| σ update | σ exp((c_σ / d_σ)(‖p_σ‖ / E‖N(0, I)‖ − 1)) | the same |
| Eigendecomposition | every generation, or lazily | every generation |

**Differences:**
- **c_σ, d_σ and c_c** are DEAP's defaults, Hansen's earlier published formulas: c_σ 0.3196 instead of 0.2844 (pycma's own value), d_σ 1.3196 instead of 1.2844, c_c 0.2857 instead of 0.2950.
- **Bounds:** `cma.Strategy` has no bound handling; DEAP's documented one is a penalty ([Constraint Handling](https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html)), which DEAP's box-bounded ES example ([cma_mo.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_mo.py)) uses: `tools.ClosestValidPenalty` evaluates a sample outside the box at its closest point inside (each gene outside clipped to its bound) and adds 10⁶ times the squared distance to it. A sample inside the box is evaluated as it is. The strategy updates with the sample itself and that value. pycma's `BoundTransform` also updates with the sample itself, but gives it the value of a transformed point, without a penalty. The counter sees the evaluated point, so `outside` is 0 by construction. In seeds 0 to 2, 0.85 to 2.5% of the samples were outside the box (100 of 4,070, 53 of 6,210, 82 of 5,180).

**Separate tests** (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Capped |
|---|---|---|---|---|---|
| cma_es | 3 | 3 | 5,173 | 0.00866 (0.00834, 0.00914) | 0 |

Seeds 3 to 9 reached the target too, in 4,227 to 8,964 evaluations. In a further check of seeds 10 to 40, three (17, 31, 37) converged to Rosenbrock's local minimum (3.99, near x₁ = −1) and sampled around it to the end of the budget, 500,000 evaluations in about 5 s, without an error.

**Speed** (checked 2026-09-29): a run to the target takes 45 to 70 ms (seeds 0 to 2: 4,070, 6,210 and 5,180 evaluations). Per sample, the time goes to DEAP's `generate` (an individual per sample), `update` (an eigendecomposition per generation) and the evaluation: `ClosestValidPenalty`'s check, the fitness function and the adapter's counter with its bounds check (rules 3 and 2.4). The individuals are lists, as in cma_minfct.py and cma_mo.py; DEAP has no batch evaluation and no faster documented way to run `cma.Strategy`. Calling `eaGenerateUpdate` once per generation costs nothing measurable: its loop body written out takes the same time, and a cheaper bounds check in the counter saved about 1 ms per run, within the noise. So the adapter is unchanged.

## Can't run

- **Rastrigin 30:** DEAP has no differential evolution. The `deap` package has no DE algorithm and no DE operator: no differential mutation, and no binomial or exponential crossover in `tools`. DE appears only in the examples ([examples/de](https://github.com/DEAP/deap/tree/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de): `basic.py`, `sphere.py`, `dynamic.py`), whose loops and operators (`mutDE`, `cxBinomial`, `cxExponential`) are written in the example files. The adapter doesn't write a component the library lacks ([rule 6.1](../rules.md#6-the-methods)).

## Bugs found

None in DEAP itself, and none in the methods it runs here. Earlier versions of the benchmark ran DEAP's DE and BIPOP-CMA-ES examples, and found three bugs in them:
- **The DE example's exponential crossover is inverted.** `cxExponential` ([sphere.py, lines 49-57](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/sphere.py#L49-L57)) copies a mutant gene and then stops if `random.random() < cr`; Storn and Price's goes on while it's below CR. So CR 0.8 acts as 0.2. Reported: [DEAP/deap#778](https://github.com/DEAP/deap/issues/778), fix proposed in [DEAP/deap#779](https://github.com/DEAP/deap/pull/779).
- **The BIPOP example reads the sort order backwards.** `Strategy.update` sorts best first, but [cma_bipop.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_bipop.py) takes `population[-1]` as the best, so EqualFunVals and Stagnation follow the worst samples. Reported: [DEAP/deap#780](https://github.com/DEAP/deap/issues/780), fix proposed in [DEAP/deap#782](https://github.com/DEAP/deap/pull/782).
- **The BIPOP example's EqualFunVals counts from the start of the run,** not over the last N generations: it appends 1 when the values are equal and nothing otherwise. Reported: [DEAP/deap#781](https://github.com/DEAP/deap/issues/781), fix proposed in [DEAP/deap#782](https://github.com/DEAP/deap/pull/782).
