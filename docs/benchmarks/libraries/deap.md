# DEAP (Python, 1.4.4)

DEAP (Distributed Evolutionary Algorithms in Python) is a framework of building blocks rather than of finished solvers: the user creates the individual and fitness types (`creator`), registers operators from `tools` in a `Toolbox`, and writes the loop or takes one of the few in `algorithms` (`eaSimple` and others); CMA-ES is in `cma`. Its documentation is [deap.readthedocs.io](https://deap.readthedocs.io/en/master/), whose [examples pages](https://deap.readthedocs.io/en/master/examples/index.html) are built from the `examples/` folder of its repository, [DEAP/deap](https://github.com/DEAP/deap). The repository has no 1.4.4 tag; the citations below are to commit [8a96fd3](https://github.com/DEAP/deap/tree/8a96fd3a75026f7b30e835f595a5199c75634ddf) on master, "Bump version to 1.4.4". DEAP deliberately doesn't recommend operators: its [overview](https://deap.readthedocs.io/en/master/overview.html) says "Instead of suggesting unfit operators, we explicitly ask you to choose them wisely." So the methods below are the ones its examples present for each problem type, with the settings of those examples.

Adapter: [benchmarks/adapters/deap/](../../../benchmarks/adapters/deap/).
Know a better way to solve one of these problems with DEAP? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs DEAP

- **Fitness functions:** those of [problems.py](../../../benchmarks/problems.py), in plain Python on one individual, as DEAP's examples and its `deap.benchmarks` module write them, returning a tuple ([bench.py, lines 115-233](../../../benchmarks/adapters/deap/bench.py#L115-L233)). DEAP evaluates one individual per call (`toolbox.map(toolbox.evaluate, ...)`), so there's nothing to vectorize.
- **Evaluations:** every call of the fitness function counts, in the adapter's wrapper around it ([`Budget`, lines 44-105](../../../benchmarks/adapters/deap/bench.py#L44-L105)). `eaSimple` and `varAnd` don't evaluate an individual that was neither crossed nor mutated: it keeps its parent's fitness, and that saving shows in the counts.
- **First hit (rule 3.3):** the wrapper records the number and the clock of the first evaluation that reaches the target ([lines 80-91](../../../benchmarks/adapters/deap/bench.py#L80-L91)).
- **Stop:** the target and the time cap are checked between generations; the budget before each evaluation, so no run goes past it ([`full`, lines 96-99](../../../benchmarks/adapters/deap/bench.py#L96-L99)). A multi-objective run ends after the generation that reaches its budget.
- **Keeping going (rule 2.2):** the GA, DE and NSGA loops of DEAP's examples have no convergence criterion; they end only after a number of generations, a limit that's only a budget, so the adapter lifts it. The BIPOP-CMA-ES's stop criteria are part of the method as its example sets them, so they count: each ends a CMA-ES run, and the example's own BIPOP restarts start the next (see below). The adapter passes no restart seeds: the BIPOP restarts draw from numpy's generator, seeded once per run.
- **Choosing among the docs (rule 6.2):** each method and setting comes from DEAP's example for the problem type (OneMax, N-Queens, the CMA-ES examples on Rastrigin, the DE examples on a multimodal and a unimodal function); no choice fell to a tie.
- **Bounds (rule 2.4):** see each section. Every run of a continuous or multi-objective problem prints `outside`, the number of evaluated solutions outside the box, counted in the wrapper. It was 0 in every test run.
- **Time:** from before the toolbox and the initial population are created to the end of the run. The front of a multi-objective run is extracted after the clock stops.
- **One thread:** numpy's BLAS (the CMA-ES's eigendecomposition) is set to one thread before numpy is imported, by assigning the variables ([lines 17-19](../../../benchmarks/adapters/deap/bench.py#L17-L19)); DEAP evaluates in the calling thread unless `toolbox.map` is replaced.
- **Seeds:** each run seeds Python's `random` (DEAP's operators) and `numpy.random` (its CMA-ES) with the seed ([lines 695-696](../../../benchmarks/adapters/deap/bench.py#L695-L696), [672-673](../../../benchmarks/adapters/deap/bench.py#L672-L673)).
- **Solutions:** a single-objective run prints the best solution the wrapper saw evaluated; a multi-objective run the first non-dominated front of its final population with their variables ([line 679](../../../benchmarks/adapters/deap/bench.py#L679)).

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched (OneMax 100 and 1000):** the matched GA is defined from DEAP's OneMax example, so DEAP runs it as it is ([`solve_onemax`, lines 263-276](../../../benchmarks/adapters/deap/bench.py#L263-L276), [`ea_simple`, lines 241-260](../../../benchmarks/adapters/deap/bench.py#L241-L260)): [examples/ga/onemax.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/onemax.py) (docs: [One Max Problem](https://deap.readthedocs.io/en/master/examples/ga_onemax.html)) with `algorithms.eaSimple` as in [onemax_short.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/onemax_short.py) (docs: [One Max Problem: Short Version](https://deap.readthedocs.io/en/master/examples/ga_onemax_short.html)): 300 individuals, `selTournament` of 3, `cxTwoPoint` with probability 0.5, `mutFlipBit` on 20% of the individuals, generational, without elitism. The one matched difference from the example: a bit flips with probability 1 / n, not 0.05.
- **Idiomatic (OneMax 100):** the same example with its own `indpb=0.05`.

**Keeping going:** the examples' limits of 1,000 generations (onemax.py) and 40 (`ngen` in onemax_short.py) are only a budget, so they're lifted: the GA runs to the target or the budget.

**Left out:**
- [onemax_numpy.py](https://deap.readthedocs.io/en/master/examples/ga_onemax_numpy.html): the same GA and the same search with numpy arrays as individuals; the two main OneMax examples use lists, DEAP's usual individual.
- The island, multi-demic and multiprocessing OneMax examples: the same GA, distributed.
- PBIL ([examples/eda/pbil.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/eda/pbil.py)), an EDA on OneMax: it's in the examples folder only, not in the docs' examples, which present the GA for OneMax.

**Separate tests** (2026-09-25, DEAP 1.4.4, numpy 2.5.3, Python 3.13.9, seeds 0 to 4, the scenario's budget, 60 s cap; other processes ran on the machine, so times are indicative):

OneMax 100, matched (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 6,547 | 100 (100, 100) | 0 |

OneMax 1000, matched (budget 2,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 0 | - | 931 (939, 920) | 5 |

The OneMax 1000 runs reached the 60 s cap at about 53,000 evaluations: the machine ran several times slower than in the benchmark's pinned runs. Without the cap (seeds 0 to 2, a 900 s cap), for comparison, they reached the target in 3 runs of 3, with a median first hit of 104,677 evaluations.

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 6,007 | 100 (100, 100) | 0 |

## Permutation: N-Queens 32 and 64

**Methods:**
- **`ga`:** [examples/ga/nqueens.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/nqueens.py), DEAP's example for this problem: permutations (`random.sample`), `cxPartialyMatched`, `mutShuffleIndexes` with `indpb` 2 / n, `selTournament` of 3, 300 individuals, `eaSimple` with crossover probability 0.5 and mutation probability 0.2 ([`solve_nqueens`, lines 279-290](../../../benchmarks/adapters/deap/bench.py#L279-L290)). The example's fitness counts conflicting pairs; the adapter uses the reference's queens minus one per diagonal, which is 0 at the same boards.

**Keeping going:** the example's 100 generations are lifted: the GA runs to the budget.

**Left out:** the settings of [examples/ga/tsp.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/tsp.py) (the same operators, `indpb` 0.05, crossover probability 0.7): DEAP has an example for N-Queens itself.

**Separate tests** (as above):

N-Queens 32 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 38,711 | 0 (0, 0) | 0 |

N-Queens 64 (budget 1,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 68,992 | 0 (0, 0) | 0 |

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods:**
- **`cma_es`: BI-population CMA-ES,** [examples/es/cma_bipop.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_bipop.py) (docs: [Controlling the Stopping Criteria: BI-POP CMA-ES](https://deap.readthedocs.io/en/master/examples/bipop_cmaes.html)), DEAP's CMA-ES with restarts, whose example minimizes Rastrigin. The adapter runs it line for line ([`solve_bipop_cmaes`, lines 322-477](../../../benchmarks/adapters/deap/bench.py#L322-L477)): the first run and every large-population run start with λ = 4 + 3 ln n doubled at each restart, the small-population runs with the example's random λ and σ, and each run ends at the first of the example's 9 criteria (MaxIter, TolHistFun, EqualFunVals, TolX, TolUpSigma, Stagnation, ConditionCov, NoEffectAxis, NoEffectCoor). The example's domain is [−5, 5]: it starts each run at a uniform point of [−4, 4] with σ₀ = 2, "1/5th of the domain". For each problem the adapter takes the same shares of its box: a start in the inner 80% and σ₀ a fifth of the width (2.048 for Rastrigin, 13.1 for Ackley).
- **`de`: differential evolution,** [examples/de/sphere.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/sphere.py), the DE example that minimizes a multimodal function (Griewank, despite the file's name): DE/rand/1 with the example's exponential crossover (see [Bugs found](#bugs-found)), F 0.8, CR 0.8, 10 n agents, the three donors drawn by `selRandom`, a generation's children replacing their agents at its end ([`cx_exponential`, lines 487-498](../../../benchmarks/adapters/deap/bench.py#L487-L498); [`solve_de`, lines 501-561](../../../benchmarks/adapters/deap/bench.py#L501-L561)). The example starts in [−3, 3]; the adapter starts uniformly in the box. DE isn't in the docs' example pages; DEAP's README lists it among the "examples of alternative algorithms".

**Bounds:** DEAP's CMA-ES and DE are unbounded, and its documented handling of constraints is a penalty decorator ([Constraint Handling](https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html)). For a box, the adapter uses what DEAP's own box-bounded ES example does, [examples/es/cma_mo.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_mo.py): `tools.ClosestValidPenalty` with the closest point in the box (clipping) and 10⁶ times the squared distance to it ([`bounded_evaluate`, lines 298-319](../../../benchmarks/adapters/deap/bench.py#L298-L319)). A sample outside the box is evaluated at its closest point inside, and gets that value plus the penalty. The adapter's counter is around the fitness function, inside DEAP's repair: it sees the repaired point, so `outside` is 0 by construction. In seed 0 of each scenario, 1.3 to 7.9% of the CMA-ES's samples and 3.1 to 6.9% of DE's were repaired this way.

**Keeping going:** the BIPOP-CMA-ES's 9 criteria detect convergence and end each CMA-ES run, and its BIPOP scheme starts the next one: the library's restart mechanism (rule 2.2). The example stops after 10 runs (`NRESTARTS`), a limit that's only a budget, so it's lifted: the restarts go on with the example's rule for choosing the regime, less its clause that makes the tenth run a large-population one. The adapter adds one criterion: DEAP's `cma.Strategy` raises `numpy.linalg.LinAlgError` when its covariance matrix degenerates, and that also ends the run. DE has no stop criterion; the example's 200 generations are lifted.

**Left out:**
- **CMA-ES with λ = 20 n,** [examples/es/cma_minfct.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_minfct.py) (docs: [Covariance Matrix Adaptation Evolution Strategy](https://deap.readthedocs.io/en/master/examples/cmaes.html)), which minimizes Rastrigin from (5, …, 5) with σ 5. It has no stop criterion and no restarts, and DEAP documents BIPOP as its CMA-ES with restarts, so rule 6.3 leaves it out.
- **A GA with SBX and polynomial mutation:** its operators come from the multi-objective [examples/ga/nsga2.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/nsga2.py). DEAP's docs present no GA for a single-objective real-valued problem; their real-valued examples are the evolution strategies, CMA-ES, PSO and EDA.
- **(1 + λ)-CMA-ES** ([cma_1+l_minfct.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_1+l_minfct.py)), the **(μ, λ)-ES** ([Evolution Strategies Basics](https://deap.readthedocs.io/en/master/examples/es_fctmin.html)), the **(1 + 1)-ES with the one-fifth rule** ([One Fifth Rule](https://deap.readthedocs.io/en/master/examples/es_onefifth.html), also the BBOB tutorial's algorithm) and **EMNA** ([Making Your Own Strategy: A Simple EDA](https://deap.readthedocs.io/en/master/examples/eda.html)): their examples minimize the sphere, a unimodal function, and have no restarts.
- **PSO** ([Particle Swarm Optimization Basics](https://deap.readthedocs.io/en/master/examples/pso_basic.html)): its example's 5 particles and speed limits are for a function of 2 variables.
- DE's other examples: [de/basic.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/basic.py) minimizes the sphere (it runs on Rosenbrock below) and [de/dynamic.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/dynamic.py) a moving-peaks problem.

**Separate tests** (as above):

Rastrigin 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 5 | 134,822 | 0.00929 (0.00484, 0.00972) | 0 |
| de | 5 | 5 | 23,703 | 0.00813 (0.00547, 0.00927) | 0 |

Rastrigin 30 (budget 2,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 1 | 476,750 | 1.99 (0.00982, 2.98) | 4 |
| de | 5 | 3 | 238,234 | 0.00913 (0.00664, 0.482) | 2 |

Ackley 30 (budget 1,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 5 | 3,293 | 0.00974 (0.00914, 0.00987) | 0 |
| de | 5 | 5 | 239,644 | 0.00989 (0.0095, 0.00996) | 0 |

Without the 60 s cap (seeds 0 to 2, a 900 s cap), for comparison, both reached Rastrigin 30 in 3 runs of 3: `cma_es` with a median first hit of 553,299 evaluations, `de` with 236,976.

## Continuous, unimodal: Rosenbrock 10

**Methods:**
- **`cma_es`:** the same BIPOP-CMA-ES. Its first run is a CMA-ES with DEAP's default population.
- **`de`:** [examples/de/basic.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/basic.py), the DE example on a unimodal function (the sphere): DE/rand/1 with binomial crossover, F 1, CR 0.25, 300 agents, each child replacing its agent at once when it's better ([lines 547-560](../../../benchmarks/adapters/deap/bench.py#L547-L560)). The example starts in [−3, 3]; the adapter starts uniformly in the box.

**Bounds** and **keeping going:** as for the multimodal problems.

**Left out:** as for the multimodal problems; [de/sphere.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/sphere.py)'s settings are its example for a multimodal function, so the unimodal one takes basic.py's.

**Separate tests** (as above):

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 5 | 5,691 | 0.00925 (0.00791, 0.00941) | 0 |
| de | 5 | 0 | - | 0.608 (0.455, 0.91) | 0 |

DE with basic.py's settings (CR 0.25, F 1) creeps along Rosenbrock's valley and doesn't reach the target within the budget.

## Multi-objective (matched): ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Methods** ([`solve_front`, lines 569-620](../../../benchmarks/adapters/deap/bench.py#L569-L620)):
- **`nsga2`:** [examples/ga/nsga2.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/nsga2.py) with the matched settings: `selNSGA2`, `selTournamentDCD` for the parents, `cxSimulatedBinaryBounded` with η 15 on each pair with probability 0.9, `mutPolynomialBounded` with η 20 and 1 / n per variable on every child, 100 individuals (92 with 3 objectives). The example's η of the crossover is 20; the matched one is 15. As in the example, the initial population goes through `selNSGA2` once, to assign the crowding distances the first tournament uses.
- **`nsga3`** (every problem): [examples/ga/nsga3.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/nsga3.py) (docs: [Non-dominated Sorting Genetic Algorithm III (NSGA-III)](https://deap.readthedocs.io/en/master/examples/nsga3.html)): `selNSGA3` with `uniform_reference_points`, 99 divisions (100 points, 100 individuals) with 2 objectives and 12 (91 points, 92 individuals) with 3, `varAnd` with crossover and mutation probability 1, `cxSimulatedBinaryBounded` with η 30, `mutPolynomialBounded` with η 20 and 1 / n. The example's settings on DTLZ2 are the matched ones; with 100 points its population rule, `int(H + (4 - H % 4))`, would give 104, and the matched population is 100. The initial population isn't sorted before the first generation, as in the example.

**Bounds:** the bounded SBX and polynomial mutation keep every variable in [0, 1].

**Keeping going:** the examples' numbers of generations are lifted: each run uses its whole budget.

**The front:** the first non-dominated front of the final population of 100 (92), extracted with `sortNondominated` after the clock stops.

**Not run (rule 6.1):**
- **SPEA2:** DEAP has its environmental selection, `tools.selSPEA2`, but not the algorithm: no example of it, and no mating selection for it (SPEA2's binary tournament on its fitness values, which `selSPEA2` doesn't expose). Running it would mean writing the algorithm around DEAP's operator.
- **SMS-EMOA:** DEAP has no SMS-EMOA. [examples/ga/mo_rhv.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/mo_rhv.py) is a generational (μ + μ) algorithm that fills the last front by hypervolume contribution, not SMS-EMOA's one child per step.
- **MOEA/D:** DEAP has none.
- **MO-CMA-ES** ([examples/es/cma_mo.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_mo.py)): not one of the matched algorithms.

**Separate tests** (as above; hypervolume with `run.py`'s reference points; no evaluated solution was outside the bounds):

ZDT1 (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 0.8690 (0.8696, 0.8689) | 25,000 | 0 |
| nsga3 | 5 | 0.8706 (0.8706, 0.8703) | 25,000 | 0 |

ZDT2 (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 0.5363 (0.5365, 0.5357) | 25,000 | 0 |
| nsga3 | 5 | 0.5367 (0.5374, 0.5360) | 25,000 | 0 |

ZDT3 (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 1.3273 (1.3275, 1.3272) | 25,000 | 0 |
| nsga3 | 5 | 1.3258 (1.3268, 1.3245) | 25,000 | 0 |

DTLZ2, 3 objectives (budget 25,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 0.6883 (0.6973, 0.6852) | 25,024 | 0 |
| nsga3 | 5 | 0.7431 (0.7439, 0.7426) | 25,024 | 0 |

DTLZ1, 3 objectives (budget 40,000):

| Solver | Runs | Hypervolume: median (best, worst) | Median evaluations | Runs at the cap |
|---|---|---|---|---|
| nsga2 | 5 | 0.1339 (0.1362, 0.1326) | 40,020 | 0 |
| nsga3 | 5 | 0.1398 (0.1399, 0.1344) | 40,020 | 0 |

## Can't run

DEAP runs every scenario. In the multi-objective scenarios it runs NSGA-II and NSGA-III, not SPEA2, SMS-EMOA or MOEA/D, which it doesn't have as algorithms (see above).

## Bugs found

No bug in DEAP itself. Three in its examples, which the adapter keeps as the examples have them:
- **The DE example's exponential crossover is inverted.** [examples/de/sphere.py, lines 49-57](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/sphere.py#L49-L57), `cxExponential`, copies a gene from the mutant and then stops if `random.random() < cr`. Exponential crossover (Storn and Price, 1997) goes on copying while the random number is below CR. So the example's CR of 0.8 acts as a CR of 0.2: a child takes 1.25 of the mutant's genes on average, where exponential crossover with CR 0.8 gives it about 5. The runs above use the example's operator, so its DE changes about one gene per child. Low crossover rates are known to suit separable functions, such as Rastrigin and Ackley.
- In the BIPOP example, [examples/es/cma_bipop.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_bipop.py), `Strategy.update` sorts the population best first ([deap/cma.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/deap/cma.py), `population.sort(key=lambda ind: ind.fitness, reverse=True)`), but the example's comment says "At this point the population is sorted" and takes `population[-1]` as the best: its EqualFunVals criterion compares the worst sample with the k-th worst, and its Stagnation criterion follows the worst values instead of the best.
- In the same example, EqualFunVals appends 1 when the values are equal and nothing otherwise, so `sum(equalfunvalues[-N:]) / N` counts the equal generations since the start of the run, not in the last N.

The two BIPOP criteria only decide when a CMA-ES run restarts.
