# pymoo (Python, 0.6.2)

pymoo is a Python framework for single-, multi- and many-objective optimization: genetic algorithms, differential evolution, CMA-ES (through pycma), evolution strategies, local searches, and NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA and others. Its docs are [pymoo.org](https://pymoo.org), built from `docs/source/` of [anyoptimization/pymoo](https://github.com/anyoptimization/pymoo) (tag [0.6.2](https://github.com/anyoptimization/pymoo/tree/0.6.2/docs/source)); the citations name those files. The source cited below is the same tag.

Adapter: [benchmarks/adapters/pymoo/](../../../benchmarks/adapters/pymoo/).
Know a way to set pymoo closer to a definition? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## What it runs

The matched suite: each problem with one method, defined the same for every library ([rules, section 6](../rules.md#6-the-methods)), and each library's own implementation of it.

| Scenario | Method | pymoo |
|---|---|---|
| Rastrigin 30, matched: no target, a fixed budget of 300,000 evaluations | DE/rand/1/bin | `de`: `DE` with `variant="DE/rand/1/bin"` |
| Rosenbrock 10, matched: target 0.01 | CMA-ES | `cma_es`: `CMAES` (pycma 4.5.0 inside) |
| OneMax 1000, matched | GA as DEAP's `eaSimple` | can't run: see [Can't run](#cant-run) |

The adapter prints nothing for any other scenario ([`main`, lines 288-290](../../../benchmarks/adapters/pymoo/bench.py#L288-L290); [`SCENARIOS`, lines 273-274](../../../benchmarks/adapters/pymoo/bench.py#L273-L274)). The methods of the earlier suite (GA, BRKGA, CMA-ES with IPOP restarts, DE as its docs example sets it, ES, Nelder-Mead) are gone with it.

## How the adapter runs pymoo

- **Fitness functions:** numpy, vectorized over the population, as pymoo's `Problem` evaluates "a **set** of solutions" ([problems/definition.md](https://pymoo.org/problems/definition.html); [bench.py, lines 50-85](../../../benchmarks/adapters/pymoo/bench.py#L50-L85)).
- **Evaluations:** every row of every batch counts, with the first hit (Rosenbrock) and the solutions outside the bounds (a gene below, above or not a number) ([`CountedProblem`, lines 146-176](../../../benchmarks/adapters/pymoo/bench.py#L146-L176); [`count_outside`, lines 93-96](../../../benchmarks/adapters/pymoo/bench.py#L93-L96)).
- **Stop:** after the generation that reaches the target (Rosenbrock 10: 0.01), or at the budget (a batch that would pass it is cut there) or the time cap: the run's termination, passed to `minimize` ([`BudgetTermination`, lines 179-194](../../../benchmarks/adapters/pymoo/bench.py#L179-L194)), as a user sets a budget with `("n_evals", N)` ([interface/termination.md](https://pymoo.org/interface/termination.html)).
- **Ending by itself (rules 2.2 and 8.4):** each run is one call of `minimize`, never restarted ([`solve`, lines 252-270](../../../benchmarks/adapters/pymoo/bench.py#L252-L270)). If the algorithm ends before the target, the budget and the time cap, the run ends there (not reached, on Rosenbrock), and its line names why in `ended_by`: for `CMAES`, the pycma criteria in its stop dictionary, which the termination reads from pycma's `CMAEvolutionStrategy` in the generator pymoo runs it in, as pymoo's own `CMAESOutput` does.
- **Generations (rule 2.3):** pymoo updates the termination after every generation, the initial population included; the termination marks its end ([`Counter`, lines 103-143](../../../benchmarks/adapters/pymoo/bench.py#L103-L143)).
- **Seeds (rule 5.2):** `minimize(..., seed=seed)` for DE; `seed + 1` for CMA-ES, whose seed pymoo passes to pycma, which reads 0 as "seed from the clock" (see [Bugs found](#bugs-found)). The same seed repeats a run, and seed 1 gives the same alone as after seed 0 (tested for both methods).
- **One thread (rule 4.3):** numpy's BLAS set to one thread before the import ([lines 33-35](../../../benchmarks/adapters/pymoo/bench.py#L33-L35)).
- **Separate tests:** 2026-09-28, pymoo 0.6.2, numpy 2.5.3, Python 3.13.9, seeds 0 to 2, the scenario's budget (Rastrigin 30: 300,000 evaluations; Rosenbrock 10: 500,000), 60 s cap, on a shared machine. `outside` was 0 in every run.

## Rastrigin 30: DE/rand/1/bin

**Method** ([`de`, lines 202-217](../../../benchmarks/adapters/pymoo/bench.py#L202-L217)): `DE(pop_size=100, sampling=FloatRandomSampling(), variant="DE/rand/1/bin", F=0.5, CR=0.9, jitter=False, prob_mut=0.0)` ([algorithms/soo/de.md](https://pymoo.org/algorithms/soo/de.html)). `DE` has no docstring: its page's example shows `pop_size`, `sampling`, `variant`, `CR`, `dither` and `jitter`, and `DE` passes its other keyword arguments to its `Variant` ([de.py, lines 62-92](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L62-L92)), where `F` and `prob_mut` are set.

| Definition | pymoo | Source |
|---|---|---|
| NP = 100, uniform in the box | `pop_size=100`; `FloatRandomSampling()`, uniform in the box (also `DE`'s default; the page's example uses `LHS()`) | [de.py, lines 238-246](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L238-L246); [sampling/rnd.py, lines 9-28](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/operators/sampling/rnd.py#L9-L28) |
| r1, r2, r3 uniform, distinct, ≠ i | `"DE/rand/1/bin"`: selection `rand`, one difference; `fast_fill_random` fills each index column with a random permutation of the population and draws again those equal to the target or an earlier column (below) | [de.py, lines 254-264](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L254-L264), [lines 131-138](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L131-L138); [selection/rnd.py, lines 26-60](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/operators/selection/rnd.py#L26-L60) |
| v = x_r1 + F (x_r2 − x_r3), F = 0.5, no dither or jitter | `F=0.5` (the `Variant`'s default too), `jitter=False`; a variant string sets `NoParameterControl`, so F stays fixed. The page example's `dither` does nothing (see [Bugs found](#bugs-found)) and isn't passed | [de.py, lines 31-54](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L31-L54), [lines 257-258](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L257-L258) |
| binomial, CR = 0.9, one forced j_rand | `CR=0.9`; `mut_binomial`: gene j from the mutant if U(0, 1) < CR; one random gene only if none was taken (below) | [de.py, lines 187-196](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L187-L196); [crossover/binx.py, lines 11-19](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/operators/crossover/binx.py#L11-L19); [util/misc.py, lines 922-935](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/util/misc.py#L922-L935) |
| no other variation | `prob_mut=0.0`: the `Variant` applies a polynomial mutation (η 20, at least one gene) to 10% of the trials by default; with 0, no trial is mutated | [de.py, lines 71, 88-90](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L71-L90), [lines 221-222](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L221-L222); [core/mutation.py, line 39](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/core/mutation.py#L39) |
| u replaces x_i if f(u) ≤ f(x_i) | `ImprovementReplacement`: if f(u) < f(x_i), and never a trial equal to a member of the population (below) | [core/replacement.py, lines 59-86](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/core/replacement.py#L59-L86) |
| generational | `_infill` builds all 100 trials from the population, `_advance` replaces their targets after they're all evaluated | [de.py, lines 286-319](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L286-L319) |
| bounds: a gene outside redrawn uniformly in the box | `repair_random_init` on the mutant (below) | [de.py, lines 169-173](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L169-L173); [repair/bounds_repair.py, lines 129-152](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/operators/repair/bounds_repair.py#L129-L152) |
| no archive, adaptation, restarts or convergence criterion | none; `DE`'s default termination, `DefaultSingleObjectiveTermination`, is replaced by `minimize`'s termination | [de.py, line 281](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/de.py#L281) |

**Differences:**
- **`prob_mut` isn't in the docs.** The DE page doesn't mention the polynomial mutation, and `prob_mut` is a keyword of the `Variant` class only (`DE` passes it through). Without it, pymoo's DE isn't DE/rand/1/bin: 10% of the trials get a second variation operator. With `prob_mut=0.0` the mutation is still computed but applied only where `random() <= 0.0`, never in practice.
- **Replacement:** strictly better (`off_F < pop_F`), and a trial equal to a member of the population never replaces. The definition replaces when not worse. On Rastrigin a trial differs from its target in at least one gene, and a different point with exactly the same value, or a duplicate of a member, has probability about 0 until the population has collapsed onto a point; so the replacement is the same here.
- **Index draws:** each of the three index columns is a random permutation of the population, the rows equal to the target or an earlier column drawn again. So every individual is the base vector x_r1 about once per generation, where the definition draws each trial's indices independently; each trial's r1, r2, r3 are still uniform among the others. A bug in the redraws leaves about 0.14% of the trials with indices that aren't distinct or equal the target (see [Bugs found](#bugs-found)); it isn't worked around (rule 8.4). Few as they are, they change the result: in about 90 trials a run, r2 = r3, so the mutant is x_r1 itself and the trial a plain crossover of x_r1 and the target, without a difference vector. On the separable Rastrigin function, combining the genes of individuals in different local minima is what helps, and these trials alone halve the error at the end.
- **The forced gene:** pymoo takes each gene from the mutant with probability CR and forces one random gene only when none was taken; the definition always takes j_rand from the mutant and the others with probability CR. Each gene comes from the mutant with probability 0.9 here, 0.9033 in the definition (27.0 and 27.1 genes of 30 on average).
- **Bounds (rule 2.4):** pymoo repairs the mutant before the crossover: a gene below the lower bound is drawn uniformly between the bound and the base vector x_r1's gene, one above the upper bound between x_r1's gene and the bound. The reference draws it uniformly in the whole box. It acts only on genes that leave the box.

**No target:** Rastrigin 30 is a fixed budget of 300,000 evaluations, measured by the time for the budget and the error at the end (the best value; the optimum is 0). The run never stops at a value, and its line always has `"target": null`, `"success": false` and `"first_hit": null`.

**Keeping going (rule 2.2):** `minimize`'s termination replaces `DE`'s default one, and `DE` has no other criterion: it runs to the budget or the time cap. It evaluates every trial, so it never stalls.

**Separate tests:** Rastrigin 30 (no target, budget 300,000, cap 60 s):

| Solver | Runs | Time per run: median (s) | Error at the end: median | best | worst | Ended by itself | Capped |
|---|---|---|---|---|---|---|---|
| de | 3 | 5.08 | 62.1 | 59.7 | 96.8 | 0 | 0 |

Every run used the whole budget: 5.08, 5.15 and 5.07 s (seeds 0, 1, 2), ending at 62.1, 59.7 and 96.8.

## Rosenbrock 10: CMA-ES

**Method** ([`cma_es`, lines 220-249](../../../benchmarks/adapters/pymoo/bench.py#L220-L249)): pymoo's [`CMAES`](https://pymoo.org/algorithms/soo/cmaes.html), which runs pycma 4.5.0 (the suite's reference, see [its page](pycma.md)) through a copy of `cma.fmin` ([vendor_cmaes.py](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/vendor/vendor_cmaes.py)): `CMAES(x0=x0, sigma=0.3, pop_size=10, restarts=0, CMA_active=False, maxstd_boundrange=np.inf, tolfun=0, tolx=0, tolfunhist=0, tolfunrel=0, tolstagnation=0, tolflatfitness=np.inf, tolconditioncov=0, tolupsigma=0, tolfacupx=np.inf, tolxstagnation=False, maxiter=np.inf)`. `x0`, `sigma`, `pop_size`, `restarts`, `tolfun` and `tolx` are `CMAES`'s documented parameters; the others go to pycma: "All parameters that can be used there either as a keyword argument or an option can also be passed to the `CMAES` constructor" (the page), "Additional CMA-ES options passed to CMAEvolutionStrategy" (the docstring).

| Definition | pymoo | Source |
|---|---|---|
| λ = 10, μ = 5, positive weights, no active CMA | `pop_size=10` (pycma's `popsize`); pycma's default μ and weights; `CMA_active=False` | docstring, `pop_size`; [cmaes.py, lines 121-123](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/cmaes.py#L121-L123), [lines 145-152](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/cmaes.py#L145-L152) |
| CSA, rank-one and rank-μ updates, h_σ, Hansen's learning rates | pycma's, the same as on [pycma's page](pycma.md#rosenbrock-10-cma-es): c_σ 0.3196, d_σ 1.3196, c_c 0.2950, c_1 0.01528, c_μ 0.02355, μ_eff 3.167 (the constants depend on n and λ only, so the normalization below doesn't change them) | pycma |
| mean uniform in the box, σ₀ = 4.5, C₀ = I | `x0` drawn with `np.random.default_rng(seed).uniform(-5, 10, 10)`; `normalize=True` (the default) maps the box to [0, 1]¹⁰, where `sigma=0.3` is 4.5 in the box | docstring, `x0`, `sigma`, `normalize`; [cmaes.py, lines 158-165](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/cmaes.py#L158-L165) |
| bounds: pycma's `BoundTransform` | pycma's `BoundTransform`, on the normalized bounds [0, 1] (below) | [cmaes.py, lines 161-162](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/cmaes.py#L161-L162) |
| no restarts | `restarts=0` | docstring, `restarts` |
| no convergence criterion | every pycma stop criterion with an option is off, and the budget replaces `CMAES`'s default termination; `noeffectcoord` and `noeffectaxis` have no option (below) | `cma.CMAOptions()` |

**Differences:**
- **Two stop criteria can't be turned off (rule 2.2).** `CMAES`'s copy of `fmin` samples `while not es.stop()` ([vendor_cmaes.py, line 200](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/vendor/vendor_cmaes.py#L200)), and pycma's `noeffectcoord` and `noeffectaxis`, "non-user defined, method specific", have no option ([evolution_strategy.py, lines 4280-4302](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/evolution_strategy.py#L4280-L4302)). They fire when adding a tenth or a fifth of a standard deviation no longer changes the mean: the distribution has collapsed onto a point at the precision of doubles, and CMA-ES can no longer move. The run ends there, not reached, with `"ended_by": "pycma's noeffectaxis"` (or whichever fired); it isn't restarted. The definition's CMA-ES samples on at that point, as pycma's adapter does, with the same result. A run ends this way only when it converges before the target: at Rosenbrock's local minimum, 3.9866.
- **Normalization and bounds:** pycma works on the box mapped to [0, 1]¹⁰, and `BoundTransform` on [0, 1]. Its quadratic margins, `max(1, |bound|) / 20` ([transformations.py, lines 177-181](https://github.com/CMA-ES/pycma/blob/r4.5.0/cma/transformations.py#L177-L181)), are 5% of the width at each bound there, and 1.7% and 3.3% in [−5, 10], so samples near a bound are mapped a little differently. It acts only near the bounds.
- **Two more evaluations, counted:** `CMAES` evaluates `x0` before the run (`LocalSearch`'s initialization, [local.py, lines 38-50](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/base/local.py#L38-L50)), and, when pycma stops, the final mean (`eval_final_mean`, [vendor_cmaes.py, lines 251-255](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/vendor/vendor_cmaes.py#L251-L255)).
- **The seed:** `minimize(..., seed=seed + 1)`, because `CMAES` passes the seed to pycma, which reads 0 as "seed from the clock" ([cmaes.py, lines 167-168](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/cmaes.py#L167-L168); see [Bugs found](#bugs-found)). It changes only which random numbers a run gets.
- **Errors:** `CMAES._advance` catches every exception pycma raises and ends the run ([cmaes.py, lines 224-230](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/cmaes.py#L224-L230)); such a run would end with `ended_by` naming no criterion. It didn't happen in the tests.
- **No cap on the standard deviations** (`maxstd_boundrange=np.inf`), as on [pycma's page](pycma.md#rosenbrock-10-cma-es).

**Separate tests:** Rosenbrock 10 (budget 500,000, cap 60 s), seeds 0 to 2:

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 3 | 3 | 6,167 | 0.00905 | 0.00871 | 0.00935 | 0 |

Each run took about 0.3 s; none ended by a criterion. With seeds 0 to 29, 25 runs reached the target (median first hit 5,251 evaluations, 3,324 to 9,630), and 5 converged at the local minimum, 3.9866, and ended there with `"ended_by": "pycma's noeffectaxis"`, after 23,002 to 37,402 evaluations (seeds 9, 13, 22, 24 and 26).

## Can't run

- **OneMax 1000, the GA:** pymoo's `GA` is (μ+λ): `FitnessSurvival` keeps the best of parents and children ([algorithms/soo/ga.md](https://pymoo.org/algorithms/soo/ga.html)), and pymoo has no generational survival.

## Bugs found

| Bug | Effect here | Worked around | Reported |
|---|---|---|---|
| `fast_fill_random` ([selection/rnd.py, line 53](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/operators/selection/rnd.py#L53)) redraws the wrong rows: after the first draw, `rem = np.where(...)[0]` holds positions within the rows just redrawn, not row numbers, so a later draw changes other rows and leaves a conflict. With 100 targets, about 0.14% of the DE trials (in 13% of the generations) have indices that aren't distinct or equal the target; keeping `rem` as row numbers (`rem = rem[np.where(...)[0]]`) leaves none. Also on the main branch | about 90 trials a run with r2 = r3: a crossover of x_r1 and the target, without a difference vector. They halve the error at the end: with seeds 1 to 10, a median of 69.6 (55 to 97) as it is, 125.7 (77 to 170) with the fix, like the other libraries (124 to 141); with only the r2 = r3 trials left unfixed, 78.0 | no | [anyoptimization/pymoo#799](https://github.com/anyoptimization/pymoo/issues/799), fix proposed in [anyoptimization/pymoo#800](https://github.com/anyoptimization/pymoo/pull/800) |
| The DE page's `dither="vector"` does nothing: `DE` passes it to its `Variant`, which accepts any keyword and ignores it (only the `DEX` operator dithers). Also on the main branch | none: not passed | no | [anyoptimization/pymoo#801](https://github.com/anyoptimization/pymoo/issues/801), fix proposed in [anyoptimization/pymoo#802](https://github.com/anyoptimization/pymoo/pull/802) |
| Docs and code disagree: the DE page describes mutation and binomial crossover only, but `DE` applies a polynomial mutation to 10% of the trials by default (`prob_mut=0.1`, a keyword the docs don't list) | turned off with `prob_mut=0.0` | no | [anyoptimization/pymoo#801](https://github.com/anyoptimization/pymoo/issues/801), fix proposed in [anyoptimization/pymoo#802](https://github.com/anyoptimization/pymoo/pull/802) |
| CMA-ES with seed 0 isn't repeatable: `CMAES._setup` passes the seed to pycma ([cmaes.py, lines 167-168](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/cmaes.py#L167-L168)), which treats 0 as "seed from the clock": two runs of `minimize(get_problem("rastrigin", n_var=10), CMAES(), ("n_evals", 2000), seed=0)` end at 2.99 and 24.87. Also on the main branch | seed 0's runs couldn't be repeated (rule 5.2) | yes: `minimize` gets `seed + 1` for CMA-ES; the algorithm is unchanged | [anyoptimization/pymoo#805](https://github.com/anyoptimization/pymoo/issues/805), fix proposed in [anyoptimization/pymoo#806](https://github.com/anyoptimization/pymoo/pull/806) |

Found in methods no longer in the suite:
- **A tournament of more than 2 is binary with the GA's comparison.** `TournamentSelection(pressure=3)` draws 3 competitors, but `comp_by_cv_and_fitness` compares only `P[i, 0]` and `P[i, 1]` ([ga.py, lines 45-49](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/ga.py#L45-L49)). Reported: [anyoptimization/pymoo#803](https://github.com/anyoptimization/pymoo/issues/803), fix proposed in [anyoptimization/pymoo#804](https://github.com/anyoptimization/pymoo/pull/804).
- **Pattern search can't be seeded in 0.6.2.** `PatternSearch._next` calls `exploration_move` without the algorithm's `random_state`, so `@default_random_state` draws from a new unseeded generator every step. Fixed on the main branch in [54e13ec](https://github.com/anyoptimization/pymoo/commit/54e13ecd82e69880fc758561290618a8eb0998f8) (following [#794](https://github.com/anyoptimization/pymoo/issues/794)), not yet released. Reported: already fixed on pymoo's main branch (commit 54e13ec), not yet released.
- **Docs and code disagree:** the binary page's text says "half uniform binary crossover", its code `TwoPointCrossover`; the permutation page's text says "edge recombination crossover", its code `OrderCrossover`. Reported: [anyoptimization/pymoo#807](https://github.com/anyoptimization/pymoo/issues/807), fix proposed in [anyoptimization/pymoo#808](https://github.com/anyoptimization/pymoo/pull/808).
