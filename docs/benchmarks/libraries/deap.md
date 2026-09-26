# DEAP (Python, 1.4.4)

DEAP (Distributed Evolutionary Algorithms in Python) is a framework of building blocks: the user creates the individual and fitness types (`creator`), registers operators from `tools` in a `Toolbox`, and writes the loop or takes one from `algorithms`; CMA-ES is in `cma`. Its docs are [deap.readthedocs.io](https://deap.readthedocs.io/en/master/), whose [examples pages](https://deap.readthedocs.io/en/master/examples/index.html) come from the `examples/` folder of [DEAP/deap](https://github.com/DEAP/deap). The repository has no 1.4.4 tag; the citations are to commit [8a96fd3](https://github.com/DEAP/deap/tree/8a96fd3a75026f7b30e835f595a5199c75634ddf), "Bump version to 1.4.4".

DEAP doesn't recommend operators ("we explicitly ask you to choose them wisely", [overview](https://deap.readthedocs.io/en/master/overview.html)), so each method is the one DEAP's examples present for the problem type, with the example's settings.

Adapter: [benchmarks/adapters/deap/](../../../benchmarks/adapters/deap/).
Know a better way to solve one of these problems with DEAP? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs DEAP

- **Fitness functions:** plain Python on one individual, returning a tuple, as DEAP's examples and `deap.benchmarks` write them ([bench.py, lines 115-233](../../../benchmarks/adapters/deap/bench.py#L115-L233)). DEAP evaluates one individual per call, so there's no batch interface (rule 3.4).
- **Evaluations:** a wrapper counts every call ([`Budget`, lines 44-105](../../../benchmarks/adapters/deap/bench.py#L44-L105)) and records the first hit ([lines 80-91](../../../benchmarks/adapters/deap/bench.py#L80-L91)). `eaSimple` and `varAnd` don't evaluate an individual that was neither crossed nor mutated.
- **Stop:** the target and the time cap between generations; the budget before each evaluation, so no run passes it ([`full`, lines 96-99](../../../benchmarks/adapters/deap/bench.py#L96-L99)). A multi-objective run ends after the generation that reaches its budget.
- **Keeping going (rule 2.2):** the examples' GA, DE and NSGA loops have only generation limits, which are lifted. The BIPOP-CMA-ES's stop criteria are part of its example, so each ends a CMA-ES run and the example's BIPOP restarts start the next, drawing from numpy's generator, seeded once per run.
- **Bounds (rule 2.4):** per section.
- **One thread:** numpy's BLAS set to one thread before import ([lines 17-19](../../../benchmarks/adapters/deap/bench.py#L17-L19)); DEAP evaluates in the calling thread.
- **Seeds:** `random` (the operators) and `numpy.random` (CMA-ES) ([lines 695-696](../../../benchmarks/adapters/deap/bench.py#L695-L696), [672-673](../../../benchmarks/adapters/deap/bench.py#L672-L673)).
- **Solutions:** the best the wrapper saw; a multi-objective run prints the first non-dominated front of its final population ([line 679](../../../benchmarks/adapters/deap/bench.py#L679)).
- **Separate tests:** 2026-09-25, DEAP 1.4.4, numpy 2.5.3, Python 3.13.9, seeds 0 to 4, the scenario's budget, 60 s cap, with other processes on the machine. `outside` was 0 in every run.

## Binary: OneMax 100 and 1000

**Methods:**
- **Matched:** the matched GA is DEAP's OneMax example, run as it is ([`solve_onemax`, lines 263-276](../../../benchmarks/adapters/deap/bench.py#L263-L276), [`ea_simple`, lines 241-260](../../../benchmarks/adapters/deap/bench.py#L241-L260)): [examples/ga/onemax.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/onemax.py) ([docs](https://deap.readthedocs.io/en/master/examples/ga_onemax.html)) with `algorithms.eaSimple` as in [onemax_short.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/onemax_short.py) ([docs](https://deap.readthedocs.io/en/master/examples/ga_onemax_short.html)): 300 individuals, `selTournament` of 3, `cxTwoPoint` at 0.5, `mutFlipBit` on 20% of the individuals, generational, no elitism. One difference: a bit flips with probability 1 / n, not 0.05.
- **Idiomatic (OneMax 100):** the same example with its `indpb=0.05`.

**Keeping going:** the examples' generation limits (1,000; `ngen` 40) are lifted.

**Left out:**
- [onemax_numpy.py](https://deap.readthedocs.io/en/master/examples/ga_onemax_numpy.html): the same GA on numpy arrays; the main examples use lists.
- The island, multi-demic and multiprocessing OneMax examples: the same GA, distributed.
- PBIL ([examples/eda/pbil.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/eda/pbil.py)): only in the examples folder, not in the docs' examples.

**Separate tests:**

OneMax 100, matched (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 6,547 | 100 (100, 100) | 0 |

OneMax 1000, matched (budget 2,000,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 0 | - | 931 (939, 920) | 5 |

These runs reached the 60 s cap at about 53,000 evaluations, on a machine several times slower than the pinned runs. With a 900 s cap (seeds 0 to 2): 3 of 3 reached, median first hit 104,677 evaluations.

OneMax 100, idiomatic (budget 200,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 6,007 | 100 (100, 100) | 0 |

## Permutation: N-Queens 32 and 64

**Methods:** `ga`, [examples/ga/nqueens.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/nqueens.py): permutations (`random.sample`), `cxPartialyMatched`, `mutShuffleIndexes` with `indpb` 2 / n, `selTournament` of 3, 300 individuals, `eaSimple` with crossover 0.5 and mutation 0.2 ([`solve_nqueens`, lines 279-290](../../../benchmarks/adapters/deap/bench.py#L279-L290)). The example counts conflicting pairs; the reference fitness is 0 at the same boards.

**Keeping going:** the example's 100 generations are lifted.

**Left out:** [examples/ga/tsp.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/tsp.py)'s settings (`indpb` 0.05, crossover 0.7): DEAP has an N-Queens example.

**Separate tests:**

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
- **`cma_es`:** BI-population CMA-ES, [examples/es/cma_bipop.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_bipop.py) ([docs](https://deap.readthedocs.io/en/master/examples/bipop_cmaes.html)), DEAP's CMA-ES with restarts, on Rastrigin, run line for line ([`solve_bipop_cmaes`, lines 322-477](../../../benchmarks/adapters/deap/bench.py#L322-L477)):
  - large-population runs start with λ = 4 + 3 ln n, doubled at each restart; small-population runs use the example's random λ and σ;
  - each run ends at the first of the example's 9 criteria (MaxIter, TolHistFun, EqualFunVals, TolX, TolUpSigma, Stagnation, ConditionCov, NoEffectAxis, NoEffectCoor);
  - the example starts uniformly in [−4, 4] of its [−5, 5] domain with σ₀ = 2, "1/5th of the domain"; the adapter takes the same shares of each box (the inner 80%, σ₀ 2.048 for Rastrigin and 13.1 for Ackley).
- **`de`:** [examples/de/sphere.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/sphere.py), the DE example on a multimodal function (Griewank): DE/rand/1 with the example's exponential crossover (see [Bugs found](#bugs-found)), F 0.8, CR 0.8, 10 n agents, donors by `selRandom`, children replacing their agents at the end of the generation ([`cx_exponential`, lines 487-498](../../../benchmarks/adapters/deap/bench.py#L487-L498); [`solve_de`, lines 501-561](../../../benchmarks/adapters/deap/bench.py#L501-L561)). It starts uniformly in the box (the example: [−3, 3]). DEAP's README lists DE among the "examples of alternative algorithms".

**Bounds:** DEAP's CMA-ES and DE are unbounded, and its documented constraint handling is a penalty ([Constraint Handling](https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html)). As DEAP's box-bounded ES example ([cma_mo.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_mo.py)) does, the adapter uses `tools.ClosestValidPenalty`: a sample outside the box is evaluated at its closest point inside, plus 10⁶ times the squared distance ([`bounded_evaluate`, lines 298-319](../../../benchmarks/adapters/deap/bench.py#L298-L319)). The counter sees the repaired point, so `outside` is 0 by construction. In seed 0 of each scenario, 1.3 to 7.9% of the CMA-ES's samples and 3.1 to 6.9% of DE's were repaired.

**Keeping going:**
- `cma_es`: the 9 criteria end each run and BIPOP starts the next. The example's limit of 10 runs (`NRESTARTS`) is lifted, with its regime rule, less the clause that makes the tenth run a large one. `cma.Strategy`'s `numpy.linalg.LinAlgError`, when the covariance matrix degenerates, also ends a run.
- `de`: the example's 200 generations are lifted.

**Left out:**
- CMA-ES with λ = 20 n ([cma_minfct.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_minfct.py), [docs](https://deap.readthedocs.io/en/master/examples/cmaes.html)), on Rastrigin from (5, …, 5) with σ 5: no restarts, and DEAP documents BIPOP as its CMA-ES with restarts (rule 6.3).
- A GA with SBX and polynomial mutation (the operators of [nsga2.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/nsga2.py)): the docs present no single-objective real-valued GA.
- (1 + λ)-CMA-ES ([cma_1+l_minfct.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_1+l_minfct.py)), the (μ, λ)-ES ([docs](https://deap.readthedocs.io/en/master/examples/es_fctmin.html)), the (1 + 1)-ES with the one-fifth rule ([docs](https://deap.readthedocs.io/en/master/examples/es_onefifth.html)) and EMNA ([docs](https://deap.readthedocs.io/en/master/examples/eda.html)): examples on the sphere, without restarts.
- PSO ([docs](https://deap.readthedocs.io/en/master/examples/pso_basic.html)): its 5 particles and speed limits are for 2 variables.
- [de/basic.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/basic.py) (the sphere; it runs on Rosenbrock) and [de/dynamic.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/dynamic.py) (moving peaks).

**Separate tests:**

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

With a 900 s cap (seeds 0 to 2), both reached Rastrigin 30 in 3 of 3 runs: `cma_es` with a median first hit of 553,299 evaluations, `de` with 236,976.

## Continuous, unimodal: Rosenbrock 10

**Methods:**
- **`cma_es`:** the same BIPOP-CMA-ES.
- **`de`:** [examples/de/basic.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/basic.py), the DE example on the sphere: DE/rand/1/bin, F 1, CR 0.25, 300 agents, each child replacing its agent at once when better ([lines 547-560](../../../benchmarks/adapters/deap/bench.py#L547-L560)). It starts uniformly in the box (the example: [−3, 3]).

**Bounds**, **keeping going** and **left out:** as for the multimodal problems; [de/sphere.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/sphere.py) is the multimodal example.

**Separate tests:**

Rosenbrock 10 (budget 500,000):

| Solver | Runs | Reached | First hit: median evaluations | Best value: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| cma_es | 5 | 5 | 5,691 | 0.00925 (0.00791, 0.00941) | 0 |
| de | 5 | 0 | - | 0.608 (0.455, 0.91) | 0 |

## Multi-objective (matched): ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Methods** ([`solve_front`, lines 569-620](../../../benchmarks/adapters/deap/bench.py#L569-L620)):
- **`nsga2`:** [examples/ga/nsga2.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/nsga2.py) with the matched settings: `selNSGA2`, `selTournamentDCD` for the parents, `cxSimulatedBinaryBounded` η 15 (the example's is 20) on each pair at 0.9, `mutPolynomialBounded` η 20 at 1 / n on every child, 100 individuals (92 with 3 objectives). As in the example, the initial population goes through `selNSGA2` once, for the first tournament's crowding distances.
- **`nsga3`** (every problem): [examples/ga/nsga3.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/nsga3.py) ([docs](https://deap.readthedocs.io/en/master/examples/nsga3.html)), whose DTLZ2 settings are the matched ones: `selNSGA3` with `uniform_reference_points`, 99 divisions (100 points, 100 individuals) or 12 (91 points, 92 individuals), `varAnd` with crossover and mutation probability 1, `cxSimulatedBinaryBounded` η 30, `mutPolynomialBounded` η 20 at 1 / n. The example's population rule, `int(H + (4 - H % 4))`, would give 104 for 100 points; the matched population is 100.

**Bounds:** the bounded SBX and polynomial mutation.

**Keeping going:** the examples' generation counts are lifted.

**The front:** the first non-dominated front of the final population (`sortNondominated`).

**Left out (rule 6.1):**
- SPEA2: DEAP has its environmental selection, `tools.selSPEA2`, but not the algorithm: no example and no mating selection on SPEA2's fitness.
- SMS-EMOA: none. [mo_rhv.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/ga/mo_rhv.py) is a generational (μ + μ) algorithm with hypervolume selection, not one child per step.
- MOEA/D: none.
- MO-CMA-ES ([cma_mo.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_mo.py)): not a matched algorithm.

**Separate tests:**

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

SPEA2, SMS-EMOA and MOEA/D in the multi-objective scenarios: DEAP doesn't have them as algorithms. It runs every other scenario.

## Bugs found

None in DEAP itself. Three in its examples, kept as the examples have them:
- **The DE example's exponential crossover is inverted.** `cxExponential` ([sphere.py, lines 49-57](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/de/sphere.py#L49-L57)) copies a mutant gene and then stops if `random.random() < cr`; Storn and Price's (1997) goes on while it's below CR. So CR 0.8 acts as 0.2: a child takes 1.25 mutant genes on average instead of about 5. Effect: the DE runs above change about one gene per child.
- **The BIPOP example reads the sort order backwards.** `Strategy.update` sorts best first ([deap/cma.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/deap/cma.py)), but [cma_bipop.py](https://github.com/DEAP/deap/blob/8a96fd3a75026f7b30e835f595a5199c75634ddf/examples/es/cma_bipop.py) takes `population[-1]` as the best. So EqualFunVals compares the worst sample with the k-th worst, and Stagnation follows the worst values.
- **EqualFunVals counts from the start of the run.** It appends 1 when the values are equal and nothing otherwise, so `sum(equalfunvalues[-N:]) / N` counts the equal generations since the start, not in the last N.

Effect of the last two: only when a CMA-ES run restarts.
