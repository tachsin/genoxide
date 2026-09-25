# pymoo (Python, 0.6.2)

pymoo is a Python framework for single-, multi- and many-objective optimization: genetic algorithms, differential evolution, CMA-ES (through pycma), evolution strategies, local searches, and NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA and others. Its documentation is [pymoo.org](https://pymoo.org), built from the `docs/source/` folder of its repository ([anyoptimization/pymoo](https://github.com/anyoptimization/pymoo), tag [0.6.2](https://github.com/anyoptimization/pymoo/tree/0.6.2/docs/source)); the citations below name those files.

Adapter: [benchmarks/adapters/pymoo/](../../../benchmarks/adapters/pymoo/).
Know a better way to solve one of these problems with pymoo? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the docs decide (rule 6.2)

pymoo states no preference of one algorithm over another for a kind of problem: its guide asks users "to first understand the intuition behind an algorithm and then select one which seems to be most suitable" ([getting_started/part_2.md](https://pymoo.org/getting_started/part_2.html)). So each method below is one that a page of the docs presents for the problem type, and its settings, termination included, are those of that page's example for the problem type; where the example sets nothing, pymoo's defaults, except the termination, which the benchmark's budget replaces (see "Keeping going" below). No method or setting was picked from the separate test runs.

## How the adapter runs pymoo

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), in numpy, vectorized over the population: pymoo's `Problem` evaluates "a **set** of solutions" as a matrix ([problems/definition.md](https://pymoo.org/problems/definition.html), "Problem (vectorized)"), which is how pymoo's own test problems are written ([bench.py, lines 89-169](../../../benchmarks/adapters/pymoo/bench.py#L89-L169)).
- **Evaluations:** the adapter counts every row of every batch itself ([`SingleProblem`, lines 248-280](../../../benchmarks/adapters/pymoo/bench.py#L248-L280); [`FrontProblem`, lines 523-537](../../../benchmarks/adapters/pymoo/bench.py#L523-L537)), including CMA-ES's initial samples, its evaluation of the final mean and every restart. pymoo's `Evaluator` doesn't evaluate an individual that already has its objectives; a child is a new individual and is evaluated.
- **Stop:** a single-objective run stops after the generation that reaches the target, or at the budget or the time cap. The budget is exact: a batch that would pass it is evaluated only up to it, and the run ends there ([lines 248-291](../../../benchmarks/adapters/pymoo/bench.py#L248-L291)). A multi-objective run stops after the generation that reaches its budget ([lines 540-552](../../../benchmarks/adapters/pymoo/bench.py#L540-L552)).
- **Keeping going (rule 2.2):** the benchmark gives every run a budget, and a pymoo user gives a run a budget through `minimize`'s termination argument, such as `termination=("n_evals", N)` ([interface/termination.md](https://pymoo.org/interface/termination.html)). That argument replaces pymoo's default termination (xtol and ftol over 30 generations, 1,000 generations, 100,000 evaluations) entirely. So a method whose docs example passes no termination runs to the budget, as it would for that user. Convergence criteria end an attempt only where they're part of the method's own settings: a termination the docs example sets explicitly, a termination the algorithm sets for itself (Nelder-Mead), or CMA-ES's own stops ([lines 294-312](../../../benchmarks/adapters/pymoo/bench.py#L294-L312)). Their budget limits (`n_max_gen`, `n_max_evals`, `n_max_iter`) are lifted, and a termination that is only a budget, such as `("n_gen", 100)`, is replaced by the benchmark's. An attempt also ends when CMA-ES has used its own restarts, or when a GA's mating can't produce a child that isn't a duplicate. The adapter then starts the method again from a new random start, seeded with seed × 1000 + restart, keeping the best solution and counting every evaluation ([`solve`, lines 459-481](../../../benchmarks/adapters/pymoo/bench.py#L459-L481)). Each section says which applies.
- **Bounds (rule 2.4):** every evaluated solution stays inside the bounds through pymoo's own bound handling, which each section names. The adapter counts the evaluated solutions outside the bounds, as pymoo proposed them, and reports them as `outside` ([`count_outside`, lines 218-221](../../../benchmarks/adapters/pymoo/bench.py#L218-L221)). In every separate test and check run, it was 0.
- **Time:** from before the problem and the algorithm are created (the initial population is sampled inside `minimize`) to the end of the run.
- **One thread:** numpy's BLAS is set to one thread before numpy is imported ([lines 39-41](../../../benchmarks/adapters/pymoo/bench.py#L39-L41)); pymoo evaluates in the calling thread.
- **Seeds:** each seed goes to `minimize(..., seed=seed)`, which seeds the algorithm's numpy generator; a restart gets seed × 1000 + restart. **CMA-ES's seed mapping:** CMA-ES gets seed + 1 (so seeds 0 to 9 run as 1 to 10, and restarts as (seed + 1) × 1000 + restart). pymoo passes its seed on to pycma, which reads 0 as "seed from the clock", so seed 0 wouldn't repeat (see [Bugs found](#bugs-found)). The shift changes only which seed each run gets.
- **Solutions:** a single-objective run prints the best solution it evaluated, a multi-objective run the non-dominated part of its final population with their variables ([lines 484-515](../../../benchmarks/adapters/pymoo/bench.py#L484-L515), [lines 598-615](../../../benchmarks/adapters/pymoo/bench.py#L598-L615)).

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched (OneMax 100 and 1000):** pymoo's `GA` with the matched settings: population 300, tournament of 3, two-point crossover with probability 0.5, bit-flip mutation on 20% of the children with 1 / n per bit, every child evaluated (`eliminate_duplicates=False`) ([lines 332-347](../../../benchmarks/adapters/pymoo/bench.py#L332-L347)). Difference: pymoo's GA is a (μ+λ) algorithm ([algorithms/soo/ga.md](https://pymoo.org/algorithms/soo/ga.html)); its survival keeps the best of parents and children and can't be made generational with its own classes.
- **Idiomatic (OneMax 100):** `ga`, the binary GA of [customization/binary.md](https://pymoo.org/customization/binary.html) (its knapsack example, the docs' only binary example): population 200, `BinaryRandomSampling`, `TwoPointCrossover`, `BitflipMutation` (defaults: every child, 1 / n per bit), `eliminate_duplicates=True` ([lines 349-359](../../../benchmarks/adapters/pymoo/bench.py#L349-L359)).

**Keeping going:** the binary example's termination, `("n_gen", 100)`, is only a budget, so both GAs, matched and idiomatic, run to the budget; the matched one has no example of its own and takes the binary example's termination. A GA whose mating can't produce a new child would restart; no test run did.

**Bounds:** bits have no bounds to leave.

**Left out:**
- `HalfUniformCrossover`: the text of the binary page names "half uniform binary crossover", but its code uses `TwoPointCrossover`. The adapter follows the code.
- `BGA`, a GA with binary operators in `pymoo/algorithms/soo/nonconvex/ga.py`: it isn't in the docs.
- BRKGA: its page presents it for permutations, not bits.

**Separate tests** (2026-09-25, pymoo 0.6.2, numpy 2.5.3, Python 3.13.9, seeds 0 to 4, the scenario's budget, 60 s cap):

OneMax 100, matched (budget 200,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 10,500 | 100 (100, 100) | 0 |

OneMax 1000, matched (budget 2,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 162,900 | 1,000 (1,000, 1,000) | 0 |

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 5,200 | 100 (100, 100) | 0 |

## Permutation: N-Queens 32 and 64

**Methods:**
- **`ga`:** the permutation GA of [customization/permutation.md](https://pymoo.org/customization/permutation.html): population 20, `PermutationRandomSampling`, `OrderCrossover`, `InversionMutation`, `eliminate_duplicates=True` ([lines 366-377](../../../benchmarks/adapters/pymoo/bench.py#L366-L377)). The page has two examples with these operators; the adapter follows the flowshop one, which "is purely optimizing the permutations", as N-Queens does. The TSP example adds a repair that is for tours (below).
- **`brkga`:** [algorithms/soo/brkga.md](https://pymoo.org/algorithms/soo/brkga.html): "BRKGAs are known to perform well on combinatorial problems", and its example is a permutation problem. As there: the variables are random keys in [0, 1] decoded into a permutation by sorting (`np.argsort`), 100 elites, 300 offspring, 50 mutants, bias 0.7, and duplicates eliminated by comparing the decoded permutations ([lines 315-325](../../../benchmarks/adapters/pymoo/bench.py#L315-L325), [lines 381-396](../../../benchmarks/adapters/pymoo/bench.py#L381-L396)). The run prints the decoded permutation.

**Keeping going:**
- `ga`: the flowshop example sets its termination explicitly, `DefaultSingleObjectiveTermination(period=50, n_max_gen=10000)`; it applies without its generation limit. Its convergence criteria end an attempt when the best solution hasn't changed (xtol 1e-8) or the best value hasn't improved by more than 1e-6 (ftol) for 50 generations in a row; the GA then restarts from new random permutations. With a population of 20, that happens often: a median of 254 attempts per N-Queens 32 run.
- `brkga`: its example's termination, `("n_gen", 50)`, is only a budget, so BRKGA runs to the budget.

**Bounds:** permutations and random keys are generated inside their domains by pymoo's sampling and operators.

**Left out:**
- The TSP example's `StartFromZeroRepair`: it's for tours, where rotating a permutation doesn't change it; an N-Queens permutation is a different board when rotated. The TSP example's text also names edge recombination crossover, but its code uses `OrderCrossover`.
- Other permutation operators (`EdgeRecombinationCrossover` in `pymoo/operators/crossover/erx.py`): no page uses them.

**Separate tests** (2026-09-25, pymoo 0.6.2, seeds 0 to 4, the scenario's budget, 60 s cap):

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 0 | - | 1 (1, 2) | 0 |
| brkga | 5 | 3 | 317,550 | 0 (0, 2) | 0 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 0 | - | 8 (7, 8) | 5 |
| brkga | 5 | 0 | - | 6 (4, 6) | 5 |

- The GA restarts whenever its population of 20 has stalled for 50 generations, so it never gets far on N-Queens 64. Before rule 2.2 was amended, with its criteria off, the same GA reached N-Queens 32 in 1 of 5 runs and N-Queens 64 in 2 of 5.
- BRKGA is slow per evaluation: its duplicate elimination, as the docs write it (an `ElementwiseDuplicateElimination`), compares individuals in Python. Its N-Queens 64 runs reached the 60 s cap after 599,000 to 754,000 evaluations. The tests ran with other processes on the machine, so the evaluations reached within 60 s vary.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:** the pages that present an algorithm for multimodal functions. pymoo labels the DE and ES pages "Multi-modal Optimization" (their keywords on pymoo.org), and the CMA-ES page says restarts "are known to work very well on multi-modal functions". Each runs its page's example, which for DE and ES is on Ackley and for CMA-ES on Rastrigin.
- **`cma_es`:** [algorithms/soo/cmaes.md](https://pymoo.org/algorithms/soo/cmaes.html): "Also, easily restarts can be used, which are known to work very well on multi-modal functions. For instance, `Rastrigin` can be solved rather quickly by: `CMAES(restarts=10, restart_from_best=True)`". The adapter uses exactly that ([lines 405-415](../../../benchmarks/adapters/pymoo/bench.py#L405-L415)); the other settings are the defaults of that example: the start point is the best of 20 Latin hypercube samples, and σ is 0.1 of the bounds, which pymoo normalizes to [0, 1]. The restarts are IPOP-CMA-ES (the `CMAES` docstring: "Number of restarts with increasing population size"): each doubles the population.
- **`de`:** [algorithms/soo/de.md](https://pymoo.org/algorithms/soo/de.html) ("known for its good results for global optimization"), its example: population 100, Latin hypercube sampling, `DE/rand/1/bin`, CR 0.3, F 0.5 (the default), no jitter ([lines 433-439](../../../benchmarks/adapters/pymoo/bench.py#L433-L439)). The example also passes `dither="vector"`, which pymoo's `DE` ignores (see [Bugs found](#bugs-found)); the adapter leaves it out.
- **`es`:** [algorithms/soo/es.md](https://pymoo.org/algorithms/soo/es.html), its example: 200 offspring and the 1/7 rule (29 parents), also the defaults ([lines 441-445](../../../benchmarks/adapters/pymoo/bench.py#L441-L445)).

**Keeping going:**
- `cma_es`: the library's restart mechanism. pycma's own criteria (`tolfun`, `tolx`, stagnation and others) end each CMA-ES run, and pymoo restarts it with twice the population, 10 times. After the 10th, the adapter would start it again from a new random start; no test run needed that. The example's `("n_evals", 2500)` is only a budget.
- `de`: the example passes no termination, so DE runs to the budget: the benchmark's budget takes the place of pymoo's default termination, as `("n_evals", N)` does for a user. (With pymoo's default termination kept as a convergence criterion, without its limits, every DE attempt on Rastrigin 30 ended within its first 170 or so generations, when the best stayed unchanged for 30, and no run reached the target in 5 seeds: a median of 121 attempts and a best of 111.)
- `es`: the example's termination, `("n_gen", 200)`, is only a budget, so ES runs to the budget.

**Bounds:**
- `cma_es`: pymoo normalizes the variables to [0, 1] and passes those bounds to pycma, whose default `BoundTransform` maps every sampled solution into them.
- `de`: pymoo's DE re-initializes a donor variable outside the bounds between its parent and the bound (`repair_random_init`), and its polynomial mutation clips to the bounds.
- `es`: pymoo's ES samples a variable outside the bounds again, up to 10 times, and then keeps the parent's value (`es_mut_repair`).
- All three start from Latin hypercube or uniform samples inside the bounds.

**Left out:**
- `GA` with its defaults (SBX and polynomial mutation), which the adapter ran before this review: its page presents it as a modular algorithm, with a constrained example, not for multimodal functions.
- PSO: its page runs Rastrigin with the defaults, but doesn't present PSO for multimodal functions; with DE and ES labeled so, it would be a fourth method.
- G3PCX, NRBO, SRES, ISRES: their pages present them as real-parameter optimizers (G3PCX, NRBO) or for constraints (SRES, ISRES), not for multimodal functions.
- BIPOP-CMA-ES (`bipop=True`): the CMA-ES page shows it only as "an example with a few selected `cma.fmin2` parameters", with 2 restarts; its example for a multimodal function is the IPOP one above.
- The DE defaults (`DE/best/1/bin`, CR 0.2), which the adapter ran before: the page's example is on Ackley, a multimodal function, so it's the documented setting for this problem type.

**Separate tests** (2026-09-25, pymoo 0.6.2 with pycma 4.5.0, seeds 0 to 4, the scenario's budget, 60 s cap; no evaluated solution was outside the bounds):

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 5 | 76,956 | 0.00836 (0.00657, 0.00895) | 0 |
| de | 5 | 5 | 32,700 | 0.00866 (0.00759, 0.00952) | 0 |
| es | 5 | 0 | - | 3.98 (0.995, 5.97) | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 5 | 412,282 | 0.00889 (0.00838, 0.00952) | 0 |
| de | 5 | 5 | 878,200 | 0.00946 (0.00813, 0.00972) | 0 |
| es | 5 | 0 | - | 207 (201, 216) | 5 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 5 | 3,226 | 0.00965 (0.00897, 0.0099) | 0 |
| de | 5 | 5 | 52,000 | 0.00968 (0.00934, 0.00972) | 0 |
| es | 5 | 5 | 84,029 | 0.00986 (0.00946, 0.00998) | 0 |

- `es`: pymoo's ES is a (μ, λ) evolution strategy with self-adapted step sizes. It reached Ackley's target but not Rastrigin's. On Rastrigin 30 it settles in a local optimum around 200, and its step-size recombination, a Python loop over every offspring and variable, makes it slow enough to reach the 60 s cap after about 1.3 to 1.5 million evaluations. The docs give no other ES settings.

## Continuous, unimodal: Rosenbrock 10

**Methods:** pymoo's preface says of point-by-point methods that they "can be highly efficient for rather unimodal fitness landscapes", and that a multimodal landscape "quickly shows the limitation of local search" ([getting_started/preface.md](https://pymoo.org/getting_started/preface.html)). pymoo's local searches (subclasses of `LocalSearch`) are CMA-ES, Nelder-Mead and pattern search; each starts from the best of 20 Latin hypercube samples.
- **`cma_es`:** as above, with the restarts of its page ([lines 405-415](../../../benchmarks/adapters/pymoo/bench.py#L405-L415)). Its page's first example is the sphere, a unimodal function.
- **`nelder_mead`:** [algorithms/soo/nelder.md](https://pymoo.org/algorithms/soo/nelder.html), with its defaults, as there ([lines 417-431](../../../benchmarks/adapters/pymoo/bench.py#L417-L431)).

**Keeping going:**
- `cma_es`: as above.
- `nelder_mead`: `NelderMead` sets its own termination, `NelderAndMeadTermination`, part of the method rather than pymoo's default; it applies without its limits (`n_max_iter`, `n_max_evals`). Its convergence criteria, a simplex whose values or points differ by less than 1e-6 or a degenerate simplex, end an attempt, and it restarts from a new random start. pymoo has no restart mechanism for it.

**Bounds:**
- `cma_es`: as above.
- `nelder_mead`: pymoo's Nelder-Mead limits its reflection and expansion steps to the bounds and clips the result to them (`set_to_bounds_if_outside_by_problem`).

**Left out:**
- Hooke and Jeeves pattern search ([algorithms/soo/pattern.md](https://pymoo.org/algorithms/soo/pattern.html)), the third local search: pymoo 0.6.2 draws its coordinate order from an unseeded generator, so a seed doesn't repeat its runs (rule 5.2; see [Bugs found](#bugs-found)).
- GA, DE and ES: pymoo doesn't present them for unimodal functions. The DE page quotes Storn's advice of CR 0.9 for "parameter dependence", which Rosenbrock has, but as general advice for DE, not as a recommendation of DE for such problems.

**Separate tests** (2026-09-25, pymoo 0.6.2 with pycma 4.5.0, seeds 0 to 4, the scenario's budget, 60 s cap; no evaluated solution was outside the bounds; no run restarted):

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | Median evaluations to the target | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 5 | 4,170 | 0.00781 (0.00722, 0.00928) | 0 |
| nelder_mead | 5 | 5 | 2,362 | 0.009 (0.00696, 0.00978) | 0 |

## Multi-objective: ZDT1, ZDT2, ZDT3 (30 variables), DTLZ2 and DTLZ1 (3 objectives)

**Methods:** the matched settings of the [README](../../../benchmarks/README.md#scenarios) ([`front_solvers`, lines 555-582](../../../benchmarks/adapters/pymoo/bench.py#L555-L582)):
- **`nsga2`, `spea2`, `sms_emoa`:** 100 individuals (92 with 3 objectives), `SBX(prob=0.9, eta=15)`, `PM(eta=20, prob_var=1/n)`.
- **`nsga3`** (3 objectives only): Das-Dennis directions with 12 divisions (91), population 92, `SBX(prob=1.0, eta=30)`, the same mutation.
- **`moead`:** the Das-Dennis directions (100, or 91 with 3 objectives), 20 neighbors, mating in the neighborhood with probability 0.9, Tchebycheff (PBI with θ 5 for DTLZ), `SBX(prob=1.0, eta=20)`, the same mutation.

Differences from the matched settings, pymoo's defaults: NSGA-II, NSGA-III, SPEA2 and SMS-EMOA eliminate duplicate children; pymoo's SBX crosses each variable with probability 0.5; its SMS-EMOA makes a population of children per generation, not one.

**The front:** the non-dominated part of the final population (for SPEA2, its archive, which is its population), found with pymoo's `NonDominatedSorting` ([lines 598-601](../../../benchmarks/adapters/pymoo/bench.py#L598-L601)). The adapter used pymoo's `res.opt` before, which for NSGA-III holds only the first-front solutions closest to its reference directions, taken before survival.

**Keeping going:** each run uses its budget, and stops after the generation that reaches it (rule 7.1).

**Bounds:** the initial populations are sampled inside [0, 1]. pymoo's SBX clamps its children to the bounds (`repair_clamp`), and its polynomial mutation clips them (`set_to_bounds_if_outside`).

**Not run (rule 6.1):**
- NSGA-III on the 2-objective ZDT problems: pymoo lists NSGA-III for many objectives and U-NSGA-III as its version "more efficient for single and bi-objective optimization problems" ([algorithms/list.md](https://pymoo.org/algorithms/list.html)).
- pymoo's other multi-objective algorithms: R-NSGA-II, R-NSGA-III, U-NSGA-III, PI-NSGA-II, AGE-MOEA, AGE-MOEA2, C-TAEA, RVEA, Omni-Optimizer, CMOPSO, MOPSO-CD, NSDE, GDE3, NSDE-R, and D-NSGA-II and KGB-DMOEA (for dynamic problems). The matched scenarios run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA.

**Separate tests** (2026-09-25, pymoo 0.6.2, seeds 0 to 4, the scenario's budget, 60 s cap; the hypervolume is `run.py`'s; no evaluated solution was outside the bounds):

ZDT1 (budget 25,000):

| Solver | Runs | Evaluations | Hypervolume: median (best, worst) | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 25,000 | 0.8698 (0.8699, 0.8696) | 0 |
| spea2 | 5 | 25,000 | 0.8705 (0.8706, 0.8703) | 0 |
| sms_emoa | 5 | 25,000 | 0.8716 (0.8718, 0.8715) | 0 |
| moead | 5 | 25,000 | 0.8699 (0.8705, 0.8693) | 0 |

ZDT2 (budget 25,000):

| Solver | Runs | Evaluations | Hypervolume: median (best, worst) | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 25,000 | 0.5364 (0.5366, 0.5358) | 0 |
| spea2 | 5 | 25,000 | 0.5370 (0.5374, 0.5365) | 0 |
| sms_emoa | 5 | 25,000 | 0.5382 (0.5383, 0.5380) | 0 |
| moead | 5 | 25,000 | 0.5376 (0.5380, 0.5356) | 0 |

ZDT3 (budget 25,000):

| Solver | Runs | Evaluations | Hypervolume: median (best, worst) | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 25,000 | 1.3276 (1.3278, 1.3273) | 0 |
| spea2 | 5 | 25,000 | 1.3278 (1.3280, 1.3276) | 0 |
| sms_emoa | 5 | 25,000 | 1.3288 (1.3289, 1.2455) | 0 |
| moead | 5 | 25,000 | 1.3238 (1.3251, 1.3220) | 0 |

DTLZ2, 3 objectives (budget 25,000):

| Solver | Runs | Evaluations | Hypervolume: median (best, worst) | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 25,024 | 0.6976 (0.7043, 0.6908) | 0 |
| nsga3 | 5 | 25,024 | 0.7443 (0.7447, 0.7442) | 0 |
| spea2 | 5 | 25,024 | 0.7308 (0.7343, 0.7275) | 0 |
| sms_emoa | 5 | 25,024 | 0.7544 (0.7545, 0.7542) | 0 |
| moead | 5 | 25,025 | 0.7443 (0.7445, 0.7441) | 0 |

DTLZ1, 3 objectives (budget 40,000):

| Solver | Runs | Evaluations | Hypervolume: median (best, worst) | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 40,020 | 1.3005 (1.3007, 1.2947) | 0 |
| nsga3 | 5 | 40,020 | 1.3046 (1.3046, 1.3043) | 0 |
| spea2 | 5 | 40,020 | 1.3041 (1.3045, 1.3034) | 0 |
| sms_emoa | 5 | 40,020 | 1.3046 (1.3047, 1.3046) | 0 |
| moead | 5 | 40,040 | 1.3044 (1.3045, 1.3038) | 0 |

The ZDT3 SMS-EMOA run at 1.2455 (seed 2) misses a segment of the disconnected front.

## Can't run

None: pymoo runs all 14 scenarios.

## Bugs found

- **CMA-ES with seed 0 isn't repeatable.** `CMAES._setup` passes the `minimize` seed to pycma as its `seed` option (`pymoo/algorithms/soo/nonconvex/cmaes.py`, lines 167-168), and pycma treats 0 as "seed from the clock". Two runs of `minimize(get_problem("rastrigin", n_var=10), CMAES(), ("n_evals", 2000), seed=0)` end at 2.99 and 24.87; with seed 1 they're equal. It's still so on pymoo's main branch. The adapter's seed mapping (seed + 1, above) avoids seed 0, without changing the algorithm. Not reported upstream yet.
- **Pattern search can't be seeded in 0.6.2.** `PatternSearch._next` calls `exploration_move` without the algorithm's `random_state`, so its `@default_random_state` decorator draws the coordinate order from a new, unseeded generator every step. Fixed on pymoo's main branch after 0.6.2, in [54e13ec](https://github.com/anyoptimization/pymoo/commit/54e13ecd82e69880fc758561290618a8eb0998f8) ("forward random_state across remaining stochastic call sites", following [#794](https://github.com/anyoptimization/pymoo/issues/794)), not yet released. Pattern search is left out until then.
- **The DE example's `dither="vector"` does nothing.** `DE` passes its keyword arguments to its `Variant`, which doesn't use `dither` and accepts any keyword, so the option is silently ignored (only the separate `DEX` crossover operator implements dithering). Documentation, not results: the adapter leaves the option out.
- **The docs and the code disagree:**
  - The binary page's text says "half uniform binary crossover", and its code uses `TwoPointCrossover`.
  - The permutation page's text says "edge recombination crossover", and its code uses `OrderCrossover`.

  The adapter follows the code.
