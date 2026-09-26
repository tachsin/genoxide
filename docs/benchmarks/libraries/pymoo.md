# pymoo (Python, 0.6.2)

pymoo is a Python framework for single-, multi- and many-objective optimization: genetic algorithms, differential evolution, CMA-ES (through pycma), evolution strategies, local searches, and NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA and others. Its documentation is [pymoo.org](https://pymoo.org), built from the `docs/source/` folder of its repository ([anyoptimization/pymoo](https://github.com/anyoptimization/pymoo), tag [0.6.2](https://github.com/anyoptimization/pymoo/tree/0.6.2/docs/source)); the citations below name those files.

Adapter: [benchmarks/adapters/pymoo/](../../../benchmarks/adapters/pymoo/).
Know a better way to solve one of these problems with pymoo? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the docs decide (rule 6.2)

pymoo states no preference of one algorithm over another for a kind of problem: its guide asks users "to first understand the intuition behind an algorithm and then select one which seems to be most suitable" ([getting_started/part_2.md](https://pymoo.org/getting_started/part_2.html)). So each method below is one that a page of the docs presents for the problem type, and its settings, termination included, are those of that page's example for the problem type; where the example sets nothing, pymoo's defaults, except the termination, which the benchmark's budget replaces (see "Keeping going" below). No method or setting was picked from the separate test runs.

## How the adapter runs pymoo

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), in numpy, vectorized over the population: pymoo's `Problem` evaluates "a **set** of solutions" as a matrix ([problems/definition.md](https://pymoo.org/problems/definition.html), "Problem (vectorized)"), which is how pymoo's own test problems are written ([bench.py, lines 84-195](../../../benchmarks/adapters/pymoo/bench.py#L84-L195)).
- **Evaluations:** the adapter counts every row of every batch itself ([`SingleProblem`, lines 259-296](../../../benchmarks/adapters/pymoo/bench.py#L259-L296); [`FrontProblem`, lines 542-556](../../../benchmarks/adapters/pymoo/bench.py#L542-L556)), including CMA-ES's initial samples, its evaluation of the final mean and every restart. pymoo's `Evaluator` doesn't evaluate an individual that already has its objectives; a child is a new individual and is evaluated.
- **First hit (rule 3.3):** the counter records the first row whose value reaches the target, its number and the clock, in every batch ([lines 279-283](../../../benchmarks/adapters/pymoo/bench.py#L279-L283)).
- **Stop:** a single-objective run stops after the generation that reaches the target, or at the budget or the time cap. The budget is exact: a batch that would pass it is evaluated only up to it, and the run ends there ([lines 259-307](../../../benchmarks/adapters/pymoo/bench.py#L259-L307)). A multi-objective run stops after the generation that reaches its budget ([lines 559-571](../../../benchmarks/adapters/pymoo/bench.py#L559-L571)).
- **Keeping going (rule 2.2):** the benchmark gives every run a budget, and a pymoo user gives a run a budget through `minimize`'s termination argument, such as `termination=("n_evals", N)` ([interface/termination.md](https://pymoo.org/interface/termination.html)). That argument replaces pymoo's default termination (xtol and ftol over 30 generations, 1,000 generations, 100,000 evaluations) entirely, and it also replaces the default termination an algorithm sets for itself, such as Nelder-Mead's. So a method whose docs example passes no termination runs to the budget, as it would for that user. Convergence criteria end an attempt only where a docs example sets them explicitly, or where they're CMA-ES's own stops, inside pycma ([lines 310-328](../../../benchmarks/adapters/pymoo/bench.py#L310-L328)). Their budget limits (`n_max_gen`, `n_max_evals`) are lifted, and a termination that is only a budget, such as `("n_gen", 100)`, is replaced by the benchmark's. An attempt also ends when CMA-ES has used its own restarts, or when a GA's mating can't produce a child that isn't a duplicate. The adapter then starts the method again from a new random start, keeping the best solution and counting every evaluation ([`solve`, lines 475-498](../../../benchmarks/adapters/pymoo/bench.py#L475-L498)). Each section says which applies.
- **Bounds (rule 2.4):** every evaluated solution stays inside the bounds through pymoo's own bound handling, which each section names. The adapter counts the evaluated solutions outside the bounds, as pymoo proposed them, and reports them as `outside` ([`count_outside`, lines 226-229](../../../benchmarks/adapters/pymoo/bench.py#L226-L229)). In every separate test, it was 0.
- **Time:** from before the problem and the algorithm are created (the initial population is sampled inside `minimize`) to the end of the run. The front of a multi-objective run is extracted after the clock stops.
- **One thread:** numpy's BLAS is set to one thread before numpy is imported, by assigning the variables ([lines 41-43](../../../benchmarks/adapters/pymoo/bench.py#L41-L43)); pymoo evaluates in the calling thread.
- **Seeds** ([lines 331-341](../../../benchmarks/adapters/pymoo/bench.py#L331-L341)): each seed goes to `minimize(..., seed=seed)`, which seeds the algorithm's numpy generator; restart r, from 1, gets (seed + 1) × 1,000,000 + r. **CMA-ES** gets (seed + 1) × 1,000,000 + 1,000 r for its attempt r, from 0, for two reasons. pymoo passes its seed on to pycma, which reads 0 as "seed from the clock", so seed 0 wouldn't repeat (see [Bugs found](#bugs-found)). And pymoo's vendored `fmin` adds 1 to pycma's seed at each IPOP restart ([vendor/vendor_cmaes.py, lines 293-296](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/vendor/vendor_cmaes.py#L293-L296)), so with consecutive seeds, restart k of seed s would replay seed s + k. With these seeds, an attempt and its 10 restarts use 11 consecutive seeds that no other run uses.
- **Solutions:** a single-objective run prints the best solution it evaluated, a multi-objective run the non-dominated part of its final population with their variables ([lines 501-534](../../../benchmarks/adapters/pymoo/bench.py#L501-L534), [lines 623-640](../../../benchmarks/adapters/pymoo/bench.py#L623-L640)).

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched (OneMax 100 and 1000): not run.** The matched GA is generational without elitism (rule 6.1). pymoo's `GA` is a (μ+λ) algorithm ([algorithms/soo/ga.md](https://pymoo.org/algorithms/soo/ga.html)): its survival, `FitnessSurvival`, keeps the best of parents and children, and pymoo has no generational survival to put in its place. The adapter prints nothing for these scenarios ([lines 361-364](../../../benchmarks/adapters/pymoo/bench.py#L361-L364)).
- **Idiomatic (OneMax 100):** `ga`, the binary GA of [customization/binary.md](https://pymoo.org/customization/binary.html) (its knapsack example, the docs' only binary example): population 200, `BinaryRandomSampling`, `TwoPointCrossover`, `BitflipMutation` (defaults: every child, 1 / n per bit), `eliminate_duplicates=True` ([lines 366-376](../../../benchmarks/adapters/pymoo/bench.py#L366-L376)).

**Keeping going:** the binary example's termination, `("n_gen", 100)`, is only a budget, so the GA runs to the budget. A GA whose mating can't produce a new child would restart; no test run did.

**Bounds:** bits have no bounds to leave.

**Left out:**
- `HalfUniformCrossover`: the text of the binary page names "half uniform binary crossover", but its code uses `TwoPointCrossover`. The adapter follows the code.
- `BGA`, a GA with binary operators in `pymoo/algorithms/soo/nonconvex/ga.py`: it isn't in the docs.
- BRKGA: its page presents it for permutations, not bits.

**Separate tests** (2026-09-25, pymoo 0.6.2, numpy 2.5.3, Python 3.13.9, seeds 0 to 4, the scenario's budget, 60 s cap; other processes ran on the machine, so times are indicative):

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 5,119 | 100 (100, 100) | 0 |

## Permutation: N-Queens 32 and 64

**Methods:**
- **`ga`:** the permutation GA of [customization/permutation.md](https://pymoo.org/customization/permutation.html): population 20, `PermutationRandomSampling`, `OrderCrossover`, `InversionMutation`, `eliminate_duplicates=True` ([lines 383-394](../../../benchmarks/adapters/pymoo/bench.py#L383-L394)). The page has two examples with these operators; the adapter follows the flowshop one, which "is purely optimizing the permutations", as N-Queens does. The TSP example adds a repair that is for tours (below).
- **`brkga`:** [algorithms/soo/brkga.md](https://pymoo.org/algorithms/soo/brkga.html): "BRKGAs are known to perform well on combinatorial problems", and its example is a permutation problem. As there: the variables are random keys in [0, 1] decoded into a permutation by sorting (`np.argsort`), 100 elites, 300 offspring, 50 mutants, bias 0.7, and duplicates eliminated by comparing the decoded permutations ([lines 344-354](../../../benchmarks/adapters/pymoo/bench.py#L344-L354), [lines 396-413](../../../benchmarks/adapters/pymoo/bench.py#L396-L413)). The run prints the decoded permutation.

**Keeping going:**
- `ga`: the flowshop example sets its termination explicitly, `DefaultSingleObjectiveTermination(period=50, n_max_gen=10000)`; it applies without its generation limit. Its convergence criteria end an attempt when the best solution hasn't changed (xtol 1e-8) or the best value hasn't improved by more than 1e-6 (ftol) for 50 generations in a row; the GA then restarts from new random permutations. With a population of 20, that happens often.
- `brkga`: its example's termination, `("n_gen", 50)`, is only a budget, so BRKGA runs to the budget.

**Bounds:** permutations and random keys are generated inside their domains by pymoo's sampling and operators.

**Left out:**
- The TSP example's `StartFromZeroRepair`: it's for tours, where rotating a permutation doesn't change it; an N-Queens permutation is a different board when rotated. The TSP example's text also names edge recombination crossover, but its code uses `OrderCrossover`.
- Other permutation operators (`EdgeRecombinationCrossover` in `pymoo/operators/crossover/erx.py`): no page uses them.

**Separate tests** (as above):

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

- The GA restarts whenever its population of 20 has stalled for 50 generations, so it never gets far on N-Queens 64.
- BRKGA is slow per evaluation: its duplicate elimination, as the docs write it (an `ElementwiseDuplicateElimination`), compares individuals in Python, so its runs reach the 60 s cap before the budget.
- Without the 60 s cap (seeds 0 to 2, a 900 s cap), for comparison: BRKGA reached N-Queens 32 in 3 runs of 3 (median first hit 317,213 evaluations) and ended N-Queens 64 at 5, 5 and 6; the GA reached neither, and ended at 1 or 2 and at 9.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:** the pages that present an algorithm for multimodal functions. pymoo labels the DE and ES pages "Multi-modal Optimization" (their keywords on pymoo.org), and the CMA-ES page says restarts "are known to work very well on multi-modal functions". Each runs its page's example, which for DE and ES is on Ackley and for CMA-ES on Rastrigin.
- **`cma_es`:** [algorithms/soo/cmaes.md](https://pymoo.org/algorithms/soo/cmaes.html): "Also, easily restarts can be used, which are known to work very well on multi-modal functions. For instance, `Rastrigin` can be solved rather quickly by: `CMAES(restarts=10, restart_from_best=True)`". The adapter uses exactly that ([lines 422-431](../../../benchmarks/adapters/pymoo/bench.py#L422-L431)); the other settings are the defaults of that example: the start point is the best of 20 Latin hypercube samples, and σ is 0.1 of the bounds, which pymoo normalizes to [0, 1]. The restarts are IPOP-CMA-ES (the `CMAES` docstring: "Number of restarts with increasing population size"): each doubles the population.
- **`de`:** [algorithms/soo/de.md](https://pymoo.org/algorithms/soo/de.html) ("known for its good results for global optimization"), its example: population 100, Latin hypercube sampling, `DE/rand/1/bin`, CR 0.3, F 0.5 (the default), no jitter ([lines 448-454](../../../benchmarks/adapters/pymoo/bench.py#L448-L454)). The example also passes `dither="vector"`, which pymoo's `DE` ignores (see [Bugs found](#bugs-found)); the adapter leaves it out.
- **`es`:** [algorithms/soo/es.md](https://pymoo.org/algorithms/soo/es.html), its example: 200 offspring and the 1/7 rule (29 parents), also the defaults ([lines 456-460](../../../benchmarks/adapters/pymoo/bench.py#L456-L460)).

**Keeping going:**
- `cma_es`: the library's restart mechanism. pycma's own criteria (`tolfun`, `tolx`, stagnation and others) end each CMA-ES run, and pymoo restarts it with twice the population, 10 times. After the 10th, the adapter starts it again from a new random start. The example's `("n_evals", 2500)` is only a budget.
- `de`: the example passes no termination, so DE runs to the budget: the benchmark's budget takes the place of pymoo's default termination, as `("n_evals", N)` does for a user.
- `es`: the example's termination, `("n_gen", 200)`, is only a budget, so ES runs to the budget.

**Bounds:**
- `cma_es`: pymoo normalizes the variables to [0, 1] and passes those bounds to pycma, whose default `BoundTransform` maps every sampled solution into them.
- `de`: pymoo's DE re-initializes a donor variable outside the bounds between its parent and the bound (`repair_random_init`), and its polynomial mutation clips to the bounds.
- `es`: pymoo's ES samples a variable outside the bounds again, up to 10 times, and then keeps the parent's value (`es_mut_repair`).
- All three start from Latin hypercube or uniform samples inside the bounds.

**Left out:**
- `GA` with its defaults (SBX and polynomial mutation): its page presents it as a modular algorithm, with a constrained example, not for multimodal functions.
- PSO: its page runs Rastrigin with the defaults, but doesn't present PSO for multimodal functions; with DE and ES labeled so, it would be a fourth method.
- G3PCX, NRBO, SRES, ISRES: their pages present them as real-parameter optimizers (G3PCX, NRBO) or for constraints (SRES, ISRES), not for multimodal functions.
- BIPOP-CMA-ES (`bipop=True`): the CMA-ES page shows it only as "an example with a few selected `cma.fmin2` parameters", with 2 restarts; its example for a multimodal function is the IPOP one above.
- The DE defaults (`DE/best/1/bin`, CR 0.2): the page's example is on Ackley, a multimodal function, so it's the documented setting for this problem type.

**Separate tests** (as above; no evaluated solution was outside the bounds):

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

- `es`: pymoo's ES is a (μ, λ) evolution strategy with self-adapted step sizes. Its step-size recombination is a Python loop over every offspring and variable, which makes it slow per evaluation. The docs give no other ES settings.
- Without the 60 s cap (seeds 0 to 2, a 900 s cap), for comparison: on Rastrigin 30, `de` reached the target in 3 runs of 3 (median first hit 773,626 evaluations), `cma_es` in none (5.97 to 9.95 at the budget) and `es` in none (188 to 200). On Rastrigin 10, `cma_es` reached it in 2 of 3 (the third ended the budget at 5.03) and `es` in 1 of 3.

## Continuous, unimodal: Rosenbrock 10

**Methods:** pymoo's preface says of point-by-point methods that they "can be highly efficient for rather unimodal fitness landscapes", and that a multimodal landscape "quickly shows the limitation of local search" ([getting_started/preface.md](https://pymoo.org/getting_started/preface.html)). pymoo's local searches (subclasses of `LocalSearch`) are CMA-ES, Nelder-Mead and pattern search; each starts from the best of 20 Latin hypercube samples.
- **`cma_es`:** as above, with the restarts of its page ([lines 422-431](../../../benchmarks/adapters/pymoo/bench.py#L422-L431)). Its page's first example is the sphere, a unimodal function.
- **`nelder_mead`:** [algorithms/soo/nelder.md](https://pymoo.org/algorithms/soo/nelder.html), with its defaults, as there ([lines 433-446](../../../benchmarks/adapters/pymoo/bench.py#L433-L446)).

**Keeping going:**
- `cma_es`: as above.
- `nelder_mead`: `NelderMead` sets `NelderAndMeadTermination` as its termination "if nothing else provided" ([nelder.py, line 136](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/nelder.py#L136-L137)). `minimize`'s termination replaces it, as it replaces DE's default, and the docs example passes none, so Nelder-Mead runs to the budget, as DE does.

**Bounds:**
- `cma_es`: as above.
- `nelder_mead`: pymoo's Nelder-Mead limits its reflection and expansion steps to the bounds and clips the result to them (`set_to_bounds_if_outside_by_problem`).

**Left out:**
- Hooke and Jeeves pattern search ([algorithms/soo/pattern.md](https://pymoo.org/algorithms/soo/pattern.html)), the third local search: pymoo 0.6.2 draws its coordinate order from an unseeded generator, so a seed doesn't repeat its runs (rule 5.2; see [Bugs found](#bugs-found)).
- GA, DE and ES: pymoo doesn't present them for unimodal functions. The DE page quotes Storn's advice of CR 0.9 for "parameter dependence", which Rosenbrock has, but as general advice for DE, not as a recommendation of DE for such problems.

**Separate tests** (as above; no evaluated solution was outside the bounds):

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 5 | 4,505 | 0.00886 (0.00799, 0.00942) | 0 |
| nelder_mead | 5 | 5 | 2,361 | 0.009 (0.00696, 0.00978) | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3 (30 variables), DTLZ2 and DTLZ1 (3 objectives)

**Methods:** the matched settings of the [README](../../../benchmarks/README.md#scenarios) ([`front_solvers`, lines 574-607](../../../benchmarks/adapters/pymoo/bench.py#L574-L607)), with pymoo's own operators:
- **`nsga2`, `spea2`:** 100 individuals (92 with 3 objectives), `SBX(prob=0.9, eta=15)`, `PM(eta=20, prob_var=1/n)`. SPEA2 gets a new `SPEA2Survival(normalize=True)`, its default survival, in each run, so each run is as when it's alone in its process (see [Bugs found](#bugs-found)).
- **`sms_emoa`:** the same, with `n_offsprings=1`: one child per step, SMS-EMOA's steady state.
- **`nsga3`** (every problem): Das-Dennis directions with 99 divisions (100 directions, population 100) with 2 objectives, 12 divisions (91 directions, population 92) with 3, `SBX(prob=1.0, eta=30)`, the same mutation.
- **`moead`:** the Das-Dennis directions (100, or 91 with 3 objectives), 20 neighbors, mating in the neighborhood with probability 0.9, Tchebycheff (PBI with θ 5 for DTLZ), `SBX(prob=1.0, eta=20)`, the same mutation.
- **No duplicate elimination:** NSGA-II, NSGA-III, SPEA2 and SMS-EMOA run with `eliminate_duplicates=False`; pymoo's MOEA/D has none.

pymoo's SBX crosses each variable of a crossed pair with probability 0.5 (its default `prob_var`).

**The front:** the non-dominated part of the final population (for SPEA2, its archive, which is its population), found with pymoo's `NonDominatedSorting` after the clock ([lines 623-626](../../../benchmarks/adapters/pymoo/bench.py#L623-L626)), not pymoo's `res.opt`, which for NSGA-III holds only the first-front solutions closest to its reference directions.

**Keeping going:** each run uses its budget, and stops after the generation that reaches it (rule 7.1).

**Bounds:** the initial populations are sampled inside [0, 1]. pymoo's SBX clamps its children to the bounds (`repair_clamp`), and its polynomial mutation clips them (`set_to_bounds_if_outside`).

**Not run (rule 6.1):** pymoo's other multi-objective algorithms: R-NSGA-II, R-NSGA-III, U-NSGA-III, PI-NSGA-II, AGE-MOEA, AGE-MOEA2, C-TAEA, RVEA, Omni-Optimizer, CMOPSO, MOPSO-CD, NSDE, GDE3, NSDE-R, and D-NSGA-II and KGB-DMOEA (for dynamic problems). The matched scenarios run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA.

**Separate tests** (as above; the hypervolume is `run.py`'s; no evaluated solution was outside the bounds):

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

SMS-EMOA makes one child per generation, so it runs pymoo's generation loop and its hypervolume survival once per evaluation. It and MOEA/D reached the 60 s cap before the budget in these tests. Without the cap (seeds 0 to 2, a 900 s cap), for comparison: SMS-EMOA used its budget in 80 to 136 s, with median hypervolumes of 0.8718 (ZDT1), 0.5385 (ZDT2), 1.3292 (ZDT3), 0.7556 (DTLZ2) and 0.1401 (DTLZ1); MOEA/D in 38 to 67 s, with 0.8699, 0.5374, 1.3238, 0.7443 and 0.1400.

Before the workaround for SPEA2's shared survival (see [Bugs found](#bugs-found)), with the 5 seeds run one after another in one process, SPEA2's median hypervolumes were 0.8702 (ZDT1), 0.5365 (ZDT2), 1.3271 (ZDT3), 0.7328 (DTLZ2) and 0.1397 (DTLZ1), against 0.8704, 0.5370, 1.3275, 0.7334 and 0.1394 with it.

## Can't run

- **Matched OneMax 100 and 1000:** pymoo has no generational survival; its GA keeps the best of parents and children (see [Binary](#binary-onemax-100-and-1000)).

pymoo runs the other 12 scenarios.

## Bugs found

- **SPEA2's runs in one process aren't independent.** `SPEA2.__init__` has `survival=SPEA2Survival(normalize=True)` as a default argument ([spea2.py, line 178](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/moo/spea2.py#L178)), so every SPEA2 of a process shares one survival object, and that object keeps its normalization points (`self.norm`, [lines 66-70](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/moo/spea2.py#L66-L70)) from one run to the next. A run's result then depends on the runs before it: seed 1 after seed 0 differs from seed 1 alone (rule 5.2). Worked around: the adapter passes a new `SPEA2Survival(normalize=True)` to each run, which gives every run the result it has alone, as a user's single run does. SPEA2's results without the workaround are under [Multi-objective](#multi-objective-zdt1-zdt2-zdt3-30-variables-dtlz2-and-dtlz1-3-objectives). Not reported upstream yet.
- **A tournament of more than 2 is a binary tournament with the GA's comparison.** `TournamentSelection(pressure=3)` draws 3 competitors per tournament (its docstring: "Selection pressure (tournament size)"), but the GA's default comparison, `comp_by_cv_and_fitness`, compares only the first two, `P[i, 0]` and `P[i, 1]` ([ga.py, lines 45-49](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/algorithms/soo/nonconvex/ga.py#L45-L49)), and ignores the third. None here: no run uses a tournament of more than 2. Not reported upstream yet.
- **CMA-ES with seed 0 isn't repeatable.** `CMAES._setup` passes the `minimize` seed to pycma as its `seed` option (`pymoo/algorithms/soo/nonconvex/cmaes.py`, lines 167-168), and pycma treats 0 as "seed from the clock". Two runs of `minimize(get_problem("rastrigin", n_var=10), CMAES(), ("n_evals", 2000), seed=0)` end at 2.99 and 24.87; with seed 1 they're equal. It's still so on pymoo's main branch. The adapter's CMA-ES seeds (above) never pass 0, without changing the algorithm. Not reported upstream yet.
- **Pattern search can't be seeded in 0.6.2.** `PatternSearch._next` calls `exploration_move` without the algorithm's `random_state`, so its `@default_random_state` decorator draws the coordinate order from a new, unseeded generator every step. Fixed on pymoo's main branch after 0.6.2, in [54e13ec](https://github.com/anyoptimization/pymoo/commit/54e13ecd82e69880fc758561290618a8eb0998f8) ("forward random_state across remaining stochastic call sites", following [#794](https://github.com/anyoptimization/pymoo/issues/794)), not yet released. Pattern search is left out until then.
- **The DE example's `dither="vector"` does nothing.** `DE` passes its keyword arguments to its `Variant`, which doesn't use `dither` and accepts any keyword, so the option is silently ignored (only the separate `DEX` crossover operator implements dithering). Documentation, not results: the adapter leaves the option out.
- **The docs and the code disagree:**
  - The binary page's text says "half uniform binary crossover", and its code uses `TwoPointCrossover`.
  - The permutation page's text says "edge recombination crossover", and its code uses `OrderCrossover`.

  The adapter follows the code.
