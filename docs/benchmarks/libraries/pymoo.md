# pymoo (Python, 0.6.2)

pymoo is a Python framework for single-, multi- and many-objective optimization: genetic algorithms, differential evolution, CMA-ES (through pycma), evolution strategies, local searches, and NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA and others. Its docs are [pymoo.org](https://pymoo.org), built from `docs/source/` of [anyoptimization/pymoo](https://github.com/anyoptimization/pymoo) (tag [0.6.2](https://github.com/anyoptimization/pymoo/tree/0.6.2/docs/source)); the citations name those files.

pymoo states no preference among algorithms: users should "first understand the intuition behind an algorithm and then select one which seems to be most suitable" ([getting_started/part_2.md](https://pymoo.org/getting_started/part_2.html)). So each method is one a docs page presents for the problem type, with that page's example settings, or the defaults where it sets none.

Adapter: [benchmarks/adapters/pymoo/](../../../benchmarks/adapters/pymoo/).
Know a better way to solve one of these problems with pymoo? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs pymoo

- **Fitness functions:** numpy, vectorized over the population, as pymoo's `Problem` evaluates "a **set** of solutions" ([problems/definition.md](https://pymoo.org/problems/definition.html); [bench.py, lines 84-195](../../../benchmarks/adapters/pymoo/bench.py#L84-L195)).
- **Evaluations:** every row of every batch counts ([`SingleProblem`, lines 279-316](../../../benchmarks/adapters/pymoo/bench.py#L279-L316); [`FrontProblem`, lines 564-578](../../../benchmarks/adapters/pymoo/bench.py#L564-L578)), CMA-ES's initial samples, final mean and restarts included; the first hit is recorded ([lines 299-303](../../../benchmarks/adapters/pymoo/bench.py#L299-L303)).
- **Stop:** after the generation that reaches the target, or at the budget (a batch that would pass it is cut there) or the time cap ([lines 279-328](../../../benchmarks/adapters/pymoo/bench.py#L279-L328)). A multi-objective run stops after the generation that reaches its budget ([lines 581-594](../../../benchmarks/adapters/pymoo/bench.py#L581-L594)).
- **Keeping going (rule 2.2):** a user sets a budget with `minimize`'s termination, such as `("n_evals", N)` ([interface/termination.md](https://pymoo.org/interface/termination.html)). It replaces pymoo's default termination and an algorithm's own (such as Nelder-Mead's), so a method whose example passes no termination runs to the budget. Convergence criteria end an attempt only where an example sets them, or as CMA-ES's own stops inside pycma ([lines 331-349](../../../benchmarks/adapters/pymoo/bench.py#L331-L349)); their budget limits are lifted. An attempt also ends when CMA-ES has used its restarts, or when a GA's mating can't produce a non-duplicate child. The adapter then restarts from a new random start ([`solve`, lines 496-519](../../../benchmarks/adapters/pymoo/bench.py#L496-L519)).
- **Bounds (rule 2.4):** pymoo's own, per section; counted as `outside` ([`count_outside`, lines 226-229](../../../benchmarks/adapters/pymoo/bench.py#L226-L229)).
- **One thread:** numpy's BLAS set to one thread before import ([lines 41-43](../../../benchmarks/adapters/pymoo/bench.py#L41-L43)).
- **Seeds** ([lines 352-362](../../../benchmarks/adapters/pymoo/bench.py#L352-L362)): `minimize(..., seed=seed)`; restarts as rule 2.2. CMA-ES attempt r, from 0, gets (seed + 1) × 1,000,000 + 1,000 r, because pycma reads 0 as "seed from the clock" (see [Bugs found](#bugs-found)), and pymoo's vendored `fmin` adds 1 to the seed at each IPOP restart ([vendor_cmaes.py, lines 293-296](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/vendor/vendor_cmaes.py#L293-L296)). An attempt and its 10 restarts use 11 seeds no other run uses.
- **Solutions:** the best evaluated; a multi-objective run prints the non-dominated part of its final population ([lines 522-556](../../../benchmarks/adapters/pymoo/bench.py#L522-L556), [lines 646-664](../../../benchmarks/adapters/pymoo/bench.py#L646-L664)).
- **Separate tests:** 2026-09-25, pymoo 0.6.2, numpy 2.5.3, Python 3.13.9, seeds 0 to 4, the scenario's budget, 60 s cap, with other processes on the machine. `outside` was 0 in every run.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched: not run** ([lines 382-385](../../../benchmarks/adapters/pymoo/bench.py#L382-L385)). pymoo's `GA` is (μ+λ): `FitnessSurvival` keeps the best of parents and children ([algorithms/soo/ga.md](https://pymoo.org/algorithms/soo/ga.html)), and pymoo has no generational survival.
- **Idiomatic:** `ga`, the binary GA of [customization/binary.md](https://pymoo.org/customization/binary.html), the docs' only binary example: population 200, `BinaryRandomSampling`, `TwoPointCrossover`, `BitflipMutation` (every child, 1 / n per bit), `eliminate_duplicates=True` ([lines 387-397](../../../benchmarks/adapters/pymoo/bench.py#L387-L397)).

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
- **`ga`:** the permutation GA of [customization/permutation.md](https://pymoo.org/customization/permutation.html), its flowshop example, which "is purely optimizing the permutations": population 20, `PermutationRandomSampling`, `OrderCrossover`, `InversionMutation`, `eliminate_duplicates=True` ([lines 404-415](../../../benchmarks/adapters/pymoo/bench.py#L404-L415)).
- **`brkga`:** [algorithms/soo/brkga.md](https://pymoo.org/algorithms/soo/brkga.html) ("known to perform well on combinatorial problems", with a permutation example): random keys in [0, 1] decoded by `np.argsort`, 100 elites, 300 offspring, 50 mutants, bias 0.7, duplicates eliminated on the decoded permutations ([lines 365-375](../../../benchmarks/adapters/pymoo/bench.py#L365-L375), [lines 417-434](../../../benchmarks/adapters/pymoo/bench.py#L417-L434)).

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
- **`cma_es`:** [algorithms/soo/cmaes.md](https://pymoo.org/algorithms/soo/cmaes.html): "`Rastrigin` can be solved rather quickly by: `CMAES(restarts=10, restart_from_best=True)`" ([lines 443-452](../../../benchmarks/adapters/pymoo/bench.py#L443-L452)). Start: the best of 20 Latin hypercube samples; σ 0.1 of the bounds, normalized to [0, 1]. The restarts are IPOP (docstring: "Number of restarts with increasing population size").
- **`de`:** [algorithms/soo/de.md](https://pymoo.org/algorithms/soo/de.html), its example: population 100, Latin hypercube sampling, `DE/rand/1/bin`, CR 0.3, F 0.5, no jitter ([lines 469-475](../../../benchmarks/adapters/pymoo/bench.py#L469-L475)). The example's `dither="vector"` does nothing (see [Bugs found](#bugs-found)) and is left out.
- **`es`:** [algorithms/soo/es.md](https://pymoo.org/algorithms/soo/es.html), its example: a (μ, λ) ES with self-adapted steps, 200 offspring and the 1/7 rule (29 parents), also the defaults ([lines 477-481](../../../benchmarks/adapters/pymoo/bench.py#L477-L481)).

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
- **`cma_es`:** as above ([lines 443-452](../../../benchmarks/adapters/pymoo/bench.py#L443-L452)); its page's first example is the sphere.
- **`nelder_mead`:** [algorithms/soo/nelder.md](https://pymoo.org/algorithms/soo/nelder.html), with its defaults ([lines 454-467](../../../benchmarks/adapters/pymoo/bench.py#L454-L467)).

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

## Multi-objective: ZDT1, ZDT2, ZDT3 (30 variables), DTLZ2 and DTLZ1 (3 objectives)

**Methods:** the matched settings of the [README](../../../benchmarks/README.md#scenarios), with pymoo's own operators ([`front_solvers`, lines 597-630](../../../benchmarks/adapters/pymoo/bench.py#L597-L630)):
- **`nsga2`, `spea2`:** 100 individuals (92 with 3 objectives), `SBX(prob=0.9, eta=15)`, `PM(eta=20, prob_var=1/n)`. SPEA2 gets a new `SPEA2Survival(normalize=True)`, its default, in each run (see [Bugs found](#bugs-found)).
- **`sms_emoa`:** the same, with `n_offsprings=1`.
- **`nsga3`** (every problem): Das-Dennis directions, 99 divisions (100, population 100) or 12 (91, population 92), `SBX(prob=1.0, eta=30)`, the same mutation.
- **`moead`:** the Das-Dennis directions (100 or 91), 20 neighbors, neighborhood mating at 0.9, Tchebycheff (PBI with θ 5 for DTLZ), `SBX(prob=1.0, eta=20)`, the same mutation.
- `eliminate_duplicates=False` for NSGA-II, NSGA-III, SPEA2 and SMS-EMOA; MOEA/D has none. pymoo's SBX crosses each variable of a crossed pair with probability 0.5 (`prob_var`).

**Keeping going:** each run uses its budget (rule 7.1).

**Bounds:** SBX clamps (`repair_clamp`), polynomial mutation clips (`set_to_bounds_if_outside`).

**The front:** the non-dominated part of the final population (SPEA2: its archive), by `NonDominatedSorting` after the clock ([lines 646-649](../../../benchmarks/adapters/pymoo/bench.py#L646-L649)). Not `res.opt`, which for NSGA-III holds only the first-front solutions closest to its directions.

**Left out (rule 6.1):** R-NSGA-II, R-NSGA-III, U-NSGA-III, PI-NSGA-II, AGE-MOEA, AGE-MOEA2, C-TAEA, RVEA, Omni-Optimizer, CMOPSO, MOPSO-CD, NSDE, GDE3, NSDE-R, D-NSGA-II and KGB-DMOEA.

**Separate tests:**

ZDT1 (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 0.8696 (0.8699, 0.8688) | 25,000 | 0 |
| nsga3 | 5 | 0.8704 (0.8707, 0.8702) | 25,000 | 0 |
| spea2 | 5 | 0.8704 (0.8704, 0.8702) | 25,000 | 0 |
| sms_emoa | 5 | 0.8260 (0.8402, 0.7871) | 7,716 | 5 |
| moead | 5 | 0.8667 (0.8680, 0.8428) | 16,400 | 5 |

ZDT2 (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 0.5362 (0.5363, 0.5356) | 25,000 | 0 |
| nsga3 | 5 | 0.5368 (0.5371, 0.5365) | 25,000 | 0 |
| spea2 | 5 | 0.5370 (0.5374, 0.5363) | 25,000 | 0 |
| sms_emoa | 5 | 0.2814 (0.4769, 0.0877) | 7,680 | 5 |
| moead | 5 | 0.4837 (0.5362, 0.3227) | 16,700 | 5 |

ZDT3 (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 1.3274 (1.3276, 1.3269) | 25,000 | 0 |
| nsga3 | 5 | 1.3252 (1.3257, 1.3249) | 25,000 | 0 |
| spea2 | 5 | 1.3275 (1.3281, 1.3267) | 25,000 | 0 |
| sms_emoa | 5 | 1.2590 (1.2787, 1.2498) | 7,797 | 5 |
| moead | 5 | 1.3199 (1.3210, 1.2394) | 16,700 | 5 |

DTLZ2, 3 objectives (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 0.6962 (0.7003, 0.6891) | 25,024 | 0 |
| nsga3 | 5 | 0.7445 (0.7447, 0.7443) | 25,024 | 0 |
| spea2 | 5 | 0.7334 (0.7340, 0.7290) | 25,024 | 0 |
| sms_emoa | 5 | 0.7527 (0.7533, 0.7515) | 7,504 | 5 |
| moead | 5 | 0.7422 (0.7430, 0.7414) | 14,924 | 5 |

DTLZ1, 3 objectives (budget 40,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 0.1346 (0.1362, 0.1336) | 40,020 | 0 |
| nsga3 | 5 | 0.1399 (0.1400, 0.1395) | 40,020 | 0 |
| spea2 | 5 | 0.1394 (0.1397, 0.1393) | 40,020 | 0 |
| sms_emoa | 5 | 0.0000 (0.0124, 0.0000) | 8,040 | 5 |
| moead | 5 | 0.1375 (0.1386, 0.0306) | 15,379 | 5 |

SMS-EMOA runs pymoo's generation loop and hypervolume survival once per evaluation. With a 900 s cap (seeds 0 to 2): SMS-EMOA used its budget in 80 to 136 s, with median hypervolumes of 0.8718 (ZDT1), 0.5385 (ZDT2), 1.3292 (ZDT3), 0.7556 (DTLZ2) and 0.1401 (DTLZ1); MOEA/D in 38 to 67 s, with 0.8699, 0.5374, 1.3238, 0.7443 and 0.1400.

## Can't run

- **Matched OneMax 100 and 1000:** pymoo has no generational survival (see [Binary](#binary-onemax-100-and-1000)).

pymoo runs the other 12 scenarios.

## Bugs found

- **SPEA2's runs in one process aren't independent.** `SPEA2.__init__` has `survival=SPEA2Survival(normalize=True)` as a default argument ([spea2.py, line 178](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/moo/spea2.py#L178)), so every SPEA2 in a process shares one survival object and its normalization points (`self.norm`, [lines 66-70](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/moo/spea2.py#L66-L70)). Seed 1 after seed 0 differs from seed 1 alone (rule 5.2). Worked around: a new `SPEA2Survival(normalize=True)` per run. Both results, median hypervolumes of 5 seeds in one process, without → with the workaround: ZDT1 0.8702 → 0.8704, ZDT2 0.5365 → 0.5370, ZDT3 1.3271 → 1.3275, DTLZ2 0.7328 → 0.7334, DTLZ1 0.1397 → 0.1394. Not reported upstream yet.
- **A tournament of more than 2 is binary with the GA's comparison.** `TournamentSelection(pressure=3)` draws 3 competitors, but `comp_by_cv_and_fitness` compares only `P[i, 0]` and `P[i, 1]` ([ga.py, lines 45-49](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/ga.py#L45-L49)). Effect: none here. Not reported upstream yet.
- **CMA-ES with seed 0 isn't repeatable.** `CMAES._setup` passes the seed to pycma (`cmaes.py`, lines 167-168), which treats 0 as "seed from the clock": two runs of `minimize(get_problem("rastrigin", n_var=10), CMAES(), ("n_evals", 2000), seed=0)` end at 2.99 and 24.87. Also on pymoo's main branch. Effect: none here; the adapter's seeds never pass 0. Not reported upstream yet.
- **Pattern search can't be seeded in 0.6.2.** `PatternSearch._next` calls `exploration_move` without the algorithm's `random_state`, so `@default_random_state` draws from a new unseeded generator every step. Fixed on the main branch in [54e13ec](https://github.com/anyoptimization/pymoo/commit/54e13ecd82e69880fc758561290618a8eb0998f8) (following [#794](https://github.com/anyoptimization/pymoo/issues/794)), not yet released. Effect: pattern search is left out.
- **The DE example's `dither="vector"` does nothing.** `DE` passes it to its `Variant`, which accepts any keyword and ignores it (only the `DEX` operator dithers). Effect: none; left out.
- **Docs and code disagree:** the binary page's text says "half uniform binary crossover", its code `TwoPointCrossover`; the permutation page's text says "edge recombination crossover", its code `OrderCrossover`. The adapter follows the code.
