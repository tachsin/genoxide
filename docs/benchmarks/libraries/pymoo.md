# pymoo (Python, 0.6.2)

pymoo is a Python framework for single-, multi- and many-objective optimization: genetic algorithms, differential evolution, CMA-ES (through pycma), evolution strategies, local searches, and NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA and others. Its docs are [pymoo.org](https://pymoo.org), built from `docs/source/` of [anyoptimization/pymoo](https://github.com/anyoptimization/pymoo) (tag [0.6.2](https://github.com/anyoptimization/pymoo/tree/0.6.2/docs/source)); the citations name those files.

pymoo states no preference among algorithms: users should "first understand the intuition behind an algorithm and then select one which seems to be most suitable" ([getting_started/part_2.md](https://pymoo.org/getting_started/part_2.html)). So each method is one a docs page presents for the problem type, with that page's example settings, or the defaults where it sets none.

Adapter: [benchmarks/adapters/pymoo/](../../../benchmarks/adapters/pymoo/).
Know a better way to solve one of these problems with pymoo? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs pymoo

- **Fitness functions:** numpy, vectorized over the population, as pymoo's `Problem` evaluates "a **set** of solutions" ([problems/definition.md](https://pymoo.org/problems/definition.html); [bench.py, lines 70-124](../../../benchmarks/adapters/pymoo/bench.py#L70-L124)).
- **Evaluations:** every row of every batch counts ([`SingleProblem`, lines 201-238](../../../benchmarks/adapters/pymoo/bench.py#L201-L238)), CMA-ES's initial samples, final mean and restarts included; the first hit is recorded ([lines 221-225](../../../benchmarks/adapters/pymoo/bench.py#L221-L225)).
- **Stop:** after the generation that reaches the target, or at the budget (a batch that would pass it is cut there) or the time cap ([lines 201-250](../../../benchmarks/adapters/pymoo/bench.py#L201-L250)).
- **Keeping going (rule 2.2):** a user sets a budget with `minimize`'s termination, such as `("n_evals", N)` ([interface/termination.md](https://pymoo.org/interface/termination.html)). It replaces pymoo's default termination and an algorithm's own (such as Nelder-Mead's), so a method whose example passes no termination runs to the budget. Convergence criteria end an attempt only where an example sets them, or as CMA-ES's own stops inside pycma ([lines 253-271](../../../benchmarks/adapters/pymoo/bench.py#L253-L271)); their budget limits are lifted. An attempt also ends when CMA-ES has used its restarts, or when a GA's mating can't produce a non-duplicate child. The adapter then restarts from a new random start ([`solve`, lines 418-441](../../../benchmarks/adapters/pymoo/bench.py#L418-L441)).
- **Bounds (rule 2.4):** pymoo's own, per section; counted as `outside` ([`count_outside`, lines 148-151](../../../benchmarks/adapters/pymoo/bench.py#L148-L151)).
- **One thread:** numpy's BLAS set to one thread before import ([lines 38-40](../../../benchmarks/adapters/pymoo/bench.py#L38-L40)).
- **Seeds** ([lines 274-284](../../../benchmarks/adapters/pymoo/bench.py#L274-L284)): `minimize(..., seed=seed)`; restarts as rule 2.2. CMA-ES attempt r, from 0, gets (seed + 1) × 1,000,000 + 1,000 r, because pycma reads 0 as "seed from the clock" (see [Bugs found](#bugs-found)), and pymoo's vendored `fmin` adds 1 to the seed at each IPOP restart ([vendor_cmaes.py, lines 293-296](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/vendor/vendor_cmaes.py#L293-L296)). An attempt and its 10 restarts use 11 seeds no other run uses.
- **Solutions:** the best evaluated ([lines 444-478](../../../benchmarks/adapters/pymoo/bench.py#L444-L478)).
- **Separate tests:** 2026-09-25, pymoo 0.6.2, numpy 2.5.3, Python 3.13.9, seeds 0 to 4, the scenario's budget, 60 s cap, with other processes on the machine. `outside` was 0 in every run.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched: not run** ([lines 304-307](../../../benchmarks/adapters/pymoo/bench.py#L304-L307)). pymoo's `GA` is (μ+λ): `FitnessSurvival` keeps the best of parents and children ([algorithms/soo/ga.md](https://pymoo.org/algorithms/soo/ga.html)), and pymoo has no generational survival.
- **Idiomatic:** `ga`, the binary GA of [customization/binary.md](https://pymoo.org/customization/binary.html), the docs' only binary example: population 200, `BinaryRandomSampling`, `TwoPointCrossover`, `BitflipMutation` (every child, 1 / n per bit), `eliminate_duplicates=True` ([lines 309-319](../../../benchmarks/adapters/pymoo/bench.py#L309-L319)).

**Keeping going:** the example's `("n_gen", 100)` is replaced by the budget.

**Left out:**
- `HalfUniformCrossover`: the page's text names it, its code uses `TwoPointCrossover`; the adapter follows the code.
- `BGA` (`pymoo/algorithms/soo/nonconvex/ga.py`): not in the docs.
- BRKGA: presented for permutations.

**Separate tests:**

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 5,119 | 100 (100, 100) | 0 |

## Permutation: N-Queens 32 and 64

**Methods:**
- **`ga`:** the permutation GA of [customization/permutation.md](https://pymoo.org/customization/permutation.html), its flowshop example, which "is purely optimizing the permutations": population 20, `PermutationRandomSampling`, `OrderCrossover`, `InversionMutation`, `eliminate_duplicates=True` ([lines 326-337](../../../benchmarks/adapters/pymoo/bench.py#L326-L337)).
- **`brkga`:** [algorithms/soo/brkga.md](https://pymoo.org/algorithms/soo/brkga.html) ("known to perform well on combinatorial problems", with a permutation example): random keys in [0, 1] decoded by `np.argsort`, 100 elites, 300 offspring, 50 mutants, bias 0.7, duplicates eliminated on the decoded permutations ([lines 287-297](../../../benchmarks/adapters/pymoo/bench.py#L287-L297), [lines 339-356](../../../benchmarks/adapters/pymoo/bench.py#L339-L356)).

**Keeping going:**
- `ga`: the example's `DefaultSingleObjectiveTermination(period=50, n_max_gen=10000)`, without the generation limit: an attempt ends after 50 generations without change of the best solution (xtol 1e-8) or improvement of the best value by more than 1e-6 (ftol).
- `brkga`: the example's `("n_gen", 50)` is replaced by the budget.

**Left out:**
- The TSP example's `StartFromZeroRepair`: for tours; a rotated N-Queens permutation is a different board. Its text names edge recombination crossover, its code `OrderCrossover`.
- `EdgeRecombinationCrossover` (`pymoo/operators/crossover/erx.py`): no page uses it.

**Separate tests:**

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 0 | - | 2 (1, 2) | 5 |
| brkga | 5 | 0 | - | 1 (1, 3) | 5 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 0 | - | 9 (9, 10) | 5 |
| brkga | 5 | 0 | - | 14 (5, 16) | 5 |

BRKGA's `ElementwiseDuplicateElimination`, as the docs write it, compares individuals in Python. With a 900 s cap (seeds 0 to 2), BRKGA reached N-Queens 32 in 3 of 3 runs (median first hit 317,213 evaluations) and ended N-Queens 64 at 5, 5 and 6; the GA reached neither, ending at 1 or 2 and at 9.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:** pymoo labels the DE and ES pages "Multi-modal Optimization", and the CMA-ES page says restarts "are known to work very well on multi-modal functions". Each runs its page's example (DE and ES on Ackley, CMA-ES on Rastrigin):
- **`cma_es`:** [algorithms/soo/cmaes.md](https://pymoo.org/algorithms/soo/cmaes.html): "`Rastrigin` can be solved rather quickly by: `CMAES(restarts=10, restart_from_best=True)`" ([lines 365-374](../../../benchmarks/adapters/pymoo/bench.py#L365-L374)). Start: the best of 20 Latin hypercube samples; σ 0.1 of the bounds, normalized to [0, 1]. The restarts are IPOP (docstring: "Number of restarts with increasing population size").
- **`de`:** [algorithms/soo/de.md](https://pymoo.org/algorithms/soo/de.html), its example: population 100, Latin hypercube sampling, `DE/rand/1/bin`, CR 0.3, F 0.5, no jitter ([lines 391-397](../../../benchmarks/adapters/pymoo/bench.py#L391-L397)). The example's `dither="vector"` does nothing (see [Bugs found](#bugs-found)) and is left out.
- **`es`:** [algorithms/soo/es.md](https://pymoo.org/algorithms/soo/es.html), its example: a (μ, λ) ES with self-adapted steps, 200 offspring and the 1/7 rule (29 parents), also the defaults ([lines 399-403](../../../benchmarks/adapters/pymoo/bench.py#L399-L403)).

**Keeping going:**
- `cma_es`: pycma's criteria end each run and pymoo restarts it with twice the population, 10 times; then the adapter restarts it. The example's `("n_evals", 2500)` is a budget.
- `de`: no termination in the example; runs to the budget.
- `es`: the example's `("n_gen", 200)` is replaced by the budget.

**Bounds:**
- `cma_es`: pycma's default `BoundTransform` on the normalized [0, 1] bounds.
- `de`: a donor variable outside the bounds is re-initialized between its parent and the bound (`repair_random_init`); the polynomial mutation clips.
- `es`: a variable outside the bounds is sampled again, up to 10 times, then keeps the parent's value (`es_mut_repair`).

**Left out:**
- `GA` with its defaults: presented as a modular algorithm, with a constrained example.
- PSO: its page runs Rastrigin, but doesn't present PSO for multimodal functions; it would be a fourth method.
- G3PCX, NRBO (real-parameter optimizers), SRES, ISRES (constraints): not presented for multimodal functions.
- BIPOP-CMA-ES (`bipop=True`): shown only as "an example with a few selected `cma.fmin2` parameters"; the multimodal example is IPOP.
- The DE defaults (`DE/best/1/bin`, CR 0.2): the page's example is on Ackley, a multimodal function.

**Separate tests:**

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 3 | 49,622 | 0.00854 (0.00517, 5.03) | 2 |
| de | 5 | 5 | 32,734 | 0.00837 (0.00795, 0.00976) | 0 |
| es | 5 | 1 | 33,576 | 3.98 (0.00947, 9.95) | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 0 | - | 14.9 (9.95, 21.9) | 5 |
| de | 5 | 0 | - | 33.4 (13.4, 34.8) | 5 |
| es | 5 | 0 | - | 209 (205, 219) | 5 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 5 | 3,364 | 0.00942 (0.00817, 0.00992) | 0 |
| de | 5 | 5 | 52,080 | 0.00969 (0.00899, 0.00987) | 0 |
| es | 5 | 5 | 82,375 | 0.00967 (0.00929, 0.00987) | 0 |

pymoo's ES recombines its step sizes in a Python loop over every offspring and variable. With a 900 s cap (seeds 0 to 2): on Rastrigin 30, `de` reached the target in 3 of 3 runs (median 773,626 evaluations), `cma_es` in none (5.97 to 9.95) and `es` in none (188 to 200); on Rastrigin 10, `cma_es` in 2 of 3 (the third ended at 5.03) and `es` in 1 of 3.

## Continuous, unimodal: Rosenbrock 10

**Methods:** pymoo's preface says point-by-point methods "can be highly efficient for rather unimodal fitness landscapes" ([getting_started/preface.md](https://pymoo.org/getting_started/preface.html)). Its local searches are CMA-ES, Nelder-Mead and pattern search, each starting from the best of 20 Latin hypercube samples.
- **`cma_es`:** as above ([lines 365-374](../../../benchmarks/adapters/pymoo/bench.py#L365-L374)); its page's first example is the sphere.
- **`nelder_mead`:** [algorithms/soo/nelder.md](https://pymoo.org/algorithms/soo/nelder.html), with its defaults ([lines 376-389](../../../benchmarks/adapters/pymoo/bench.py#L376-L389)).

**Keeping going:** `cma_es` as above. `nelder_mead`: `minimize`'s termination replaces `NelderAndMeadTermination` ([nelder.py, line 136](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/nelder.py#L136-L137)), and the example passes none, so it runs to the budget.

**Bounds:** `nelder_mead` limits its reflection and expansion steps to the bounds and clips the result (`set_to_bounds_if_outside_by_problem`).

**Left out:**
- Hooke and Jeeves pattern search ([algorithms/soo/pattern.md](https://pymoo.org/algorithms/soo/pattern.html)): can't be seeded in 0.6.2 (rule 5.2; see [Bugs found](#bugs-found)).
- GA, DE and ES: not presented for unimodal functions. The DE page's CR 0.9 for "parameter dependence" is general advice for DE.

**Separate tests:**

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 5 | 4,505 | 0.00886 (0.00799, 0.00942) | 0 |
| nelder_mead | 5 | 5 | 2,361 | 0.009 (0.00696, 0.00978) | 0 |

## Can't run

- **Matched OneMax 100 and 1000:** pymoo has no generational survival (see [Binary](#binary-onemax-100-and-1000)).

pymoo runs the other 7 scenarios.

## Bugs found

- **A tournament of more than 2 is binary with the GA's comparison.** `TournamentSelection(pressure=3)` draws 3 competitors, but `comp_by_cv_and_fitness` compares only `P[i, 0]` and `P[i, 1]` ([ga.py, lines 45-49](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/ga.py#L45-L49)). Effect: none here. Not reported upstream yet.
- **CMA-ES with seed 0 isn't repeatable.** `CMAES._setup` passes the seed to pycma (`cmaes.py`, lines 167-168), which treats 0 as "seed from the clock": two runs of `minimize(get_problem("rastrigin", n_var=10), CMAES(), ("n_evals", 2000), seed=0)` end at 2.99 and 24.87. Also on pymoo's main branch. Effect: none here; the adapter's seeds never pass 0. Not reported upstream yet.
- **Pattern search can't be seeded in 0.6.2.** `PatternSearch._next` calls `exploration_move` without the algorithm's `random_state`, so `@default_random_state` draws from a new unseeded generator every step. Fixed on the main branch in [54e13ec](https://github.com/anyoptimization/pymoo/commit/54e13ecd82e69880fc758561290618a8eb0998f8) (following [#794](https://github.com/anyoptimization/pymoo/issues/794)), not yet released. Effect: pattern search is left out.
- **The DE example's `dither="vector"` does nothing.** `DE` passes it to its `Variant`, which accepts any keyword and ignores it (only the `DEX` operator dithers). Effect: none; left out.
- **Docs and code disagree:** the binary page's text says "half uniform binary crossover", its code `TwoPointCrossover`; the permutation page's text says "edge recombination crossover", its code `OrderCrossover`. The adapter follows the code.
