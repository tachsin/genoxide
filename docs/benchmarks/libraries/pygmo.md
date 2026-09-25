# pygmo (C++ via Python, 2.19.8)

pygmo is the Python interface of pagmo, ESA's C++ library of optimization algorithms built around the island model. The 2.19.8 wheel bundles pagmo 2.19.1. The algorithms run in C++ and call a Python user-defined problem (UDP) for every fitness evaluation. Its docs are at [esa.github.io/pygmo2](https://esa.github.io/pygmo2/): the [list of algorithms](https://esa.github.io/pygmo2/overview.html#list-of-algorithms), with the problem types each one handles, and the [tutorials](https://esa.github.io/pygmo2/tutorials/tutorials.html), which are where pygmo shows which algorithm and settings to use for which problem. pagmo's C++ docs are at [esa.github.io/pagmo2](https://esa.github.io/pagmo2/).

Adapter: [benchmarks/adapters/pygmo/](../../../benchmarks/adapters/pygmo/).
Know a better way to solve one of these problems with pygmo? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs pygmo

- **Fitness functions:** a UDP class whose `fitness(x)` returns a list ([`Problem`](../../../benchmarks/adapters/pygmo/bench.py#L180)). pygmo passes `x` as a numpy array, and the fitness functions use numpy arithmetic on it, as [tutorials/coding_udp_simple](https://esa.github.io/pygmo2/tutorials/coding_udp_simple.html) shows: "It is important to remember that x is a NumPy array, so that the NumPy array arithmetic applies in the body of fitness()". OneMax is maximized, so its UDP returns minus the number of ones: pagmo always minimizes. The functions are at [bench.py#L92](../../../benchmarks/adapters/pygmo/bench.py#L92) to [#L166](../../../benchmarks/adapters/pygmo/bench.py#L163).
- **Counting:** the UDP counts every call in the adapter's own [`Counter`](../../../benchmarks/adapters/pygmo/bench.py#L46), initial populations and restarts included, and keeps the best solution. pygmo's own count (`get_fevals`) isn't used.
- **Stopping:** in a single-objective run, the counter raises an exception from inside the fitness at the target, the budget or the 60 s cap. It passes through pagmo's C++ back to the adapter, so a run ends at that evaluation and never goes past the budget. A multi-objective run has no target: it runs the generations of the budget in one call of `evolve` ([`evolve_front`](../../../benchmarks/adapters/pygmo/bench.py#L404)).
- **One thread:** no islands, archipelagos or batch fitness evaluators, which are pygmo's ways to parallelize, and numpy's BLAS with 1 thread. The CPU time stays at the wall time: 16.06 s of CPU in 16.08 s for Rastrigin 30 (seed 4, all 3 solvers).
- **Seeds:** the seed goes to `pg.population(..., seed=)` and to the algorithm's `seed`. Restart r of a run uses the seed `seed * 100000 + r` for both. The same seed gives the same evaluations and best value.

## Binary: OneMax 100 and 1000 (matched), OneMax 100 (idiomatic)

The bits are integer genes in [0, 1] (`get_nix`, as in [tutorials/coding_udp_minlp](https://esa.github.io/pygmo2/tutorials/coding_udp_minlp.html)).

**Methods:**
- **Matched** ([bench.py#L303](../../../benchmarks/adapters/pygmo/bench.py#L303)): [`sga`](https://esa.github.io/pygmo2/algorithms.html#pygmo.sga) with a population of 300, tournaments of 3, single-point crossover at 0.5 and uniform mutation at 0.4 / n per gene. The differences from the matched GA:
  - pagmo's sga has no two-point crossover, so it's single-point. Each selected parent makes one child with a random partner.
  - Its uniform mutation draws an integer gene again from its bounds, which flips a bit half the time. So the rate is doubled to 0.4 / n, which flips 0.2 bits per child, like 20% of the children with 1 / n. It mutates every child, not 20% of them.
  - Its reinsertion is elitist: the best 300 of parents and children survive. It can't be turned off ("the only reinsertion strategy provided is what we call pure elitism").
- **Idiomatic** ([bench.py#L315](../../../benchmarks/adapters/pygmo/bench.py#L315)): pygmo doesn't recommend an algorithm for binary problems. Its [list of algorithms](https://esa.github.io/pygmo2/overview.html#list-of-algorithms) flags three single-objective algorithms for integer programming ("I"), and the adapter runs all three with their documented defaults:
  - `ga`: [`sga`](https://esa.github.io/pygmo2/algorithms.html#pygmo.sga), pygmo's genetic algorithm, with its defaults: exponential crossover at 0.9, mutation at 0.02 per gene (for an integer gene, its polynomial mutation draws the gene again from its bounds), tournaments of 2.
  - `ihs`: [`ihs`](https://esa.github.io/pygmo2/algorithms.html#pygmo.ihs), improved harmony search, with its defaults.
  - `gaco`: [`gaco`](https://esa.github.io/pygmo2/algorithms.html#pygmo.gaco), extended ant colony optimization, which its docs present for "both continuous and integer variables", with its defaults.
  - Populations: pygmo's docs give none for these algorithms. sga and ihs use the population of 20 of pygmo's tutorials ([evolving_a_population](https://esa.github.io/pygmo2/tutorials/evolving_a_population.html), [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html), and pagmo's [quick start](https://esa.github.io/pagmo2/quickstart.html)). gaco uses 63: its default kernel of 63 solutions needs a population at least that large.

**Keeping going:** sga and ihs stop only after their generations, and get the generations of the whole budget, so they run to the budget by themselves. gaco also stops by itself, after 100,000 generations or evaluations without improvement (its `impstop` and `evalstop`), and then restarts from a new random population, as the [cmaes_vs_xnes tutorial](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html) does. It never stopped in these tests.

**Left out:**
- NSGA-II and MACO, also flagged for integers: they're multi-objective.
- The other single-objective algorithms: pygmo says they "will optimise the relaxed problem" of an integer problem, so their bits wouldn't be 0 or 1.

**Separate tests** (2026-09-25, pygmo 2.19.8, seeds 0 to 4, the scenario's budget and the 60 s cap; the best value is the number of ones):

OneMax 100, matched (200,000 evaluations):

| Solver | Runs | Reached | Median evaluations to the target | Best: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 11,954 | 100 (100, 100) | 0 |

OneMax 1000, matched (2,000,000 evaluations):

| Solver | Runs | Reached | Median evaluations to the target | Best: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 150,102 | 1000 (1000, 1000) | 0 |

OneMax 100, idiomatic (200,000 evaluations):

| Solver | Runs | Reached | Median evaluations to the target | Best: median (best, worst) | Runs at the cap |
|---|---|---|---|---|---|
| ga | 5 | 5 | 1,582 | 100 (100, 100) | 0 |
| ihs | 5 | 0 | - | 92 (92, 90) | 0 |
| gaco | 5 | 0 | - | 75 (76, 73) | 0 |

ihs and gaco use the whole budget without reaching all ones. With their defaults they're slow on OneMax; nothing in pygmo's docs suggests other settings for binary problems.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** ([bench.py#L332](../../../benchmarks/adapters/pygmo/bench.py#L332)):
- `sade`: [`sade`](https://esa.github.io/pygmo2/algorithms.html#pygmo.sade), self-adaptive differential evolution in its jDE variant, with its defaults: variant rand/1/exp, jDE adaptation, `ftol` and `xtol` 1e-6. Population 20. pygmo's docs use it throughout: pagmo's [quick start](https://esa.github.io/pagmo2/quickstart.html) runs it on Schwefel 30 (islands of 20), and [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html) and [evolving_a_population](https://esa.github.io/pygmo2/tutorials/evolving_a_population.html) use it with 20 individuals. In [cec2013_comp](https://esa.github.io/pygmo2/tutorials/cec2013_comp.html), "the particular instances choosen for cmaes and sade (jDE) are performing particularly well".
- `cma_es`: [`cmaes`](https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes) with the settings of [cmaes_vs_xnes](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html), which runs it on Rosenbrock, Rastrigin and Ackley: `gen=4000, ftol=1e-8, xtol=1e-10`, the other parameters at their defaults (σ0 0.5 of the bounds' width), with restarts.
  - The population comes from the same tutorial. Its figures compare three population sizes per function and dimension: Rastrigin 10: 40, 60, 100; Rastrigin 20: 100, 150, 200; Ackley 10: 10, 20, 30; Ackley 20: 20, 30, 40. The adapter takes the one with the fewest median evaluations to the target in each figure, and for a dimension the tutorial doesn't show, the nearest one it shows. That's 100 for Rastrigin 10, 200 for Rastrigin 30 (its figure for 20) and 20 for Ackley 30 (its figure for 20). The table is at [bench.py#L272](../../../benchmarks/adapters/pygmo/bench.py#L272).
  - `force_bounds=True`: the tutorial's problems accept points outside the bounds, the benchmark's don't. It's pagmo's only way to keep CMA-ES inside them: "The fitness will never be called outside the bounds but the covariance matrix adaptation mechanism will worsen".
- `simulated_annealing`: [`simulated_annealing`](https://esa.github.io/pygmo2/algorithms.html#pygmo.simulated_annealing), Corana's, one of "the two most successful algorithms" of [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html). With that tutorial's settings: `simulated_annealing(10, 0.01, 5)` (Ts 10, Tf 0.01, 5 temperature adjustments, the others at their defaults), a population of 20, from whose best it starts, and reannealing.
  - pygmo documents two other settings: the defaults (Tf 0.1, 10 temperature adjustments) and, in the [cec2013_comp](https://esa.github.io/pygmo2/tutorials/cec2013_comp.html) figures, Tf 0.1 with 300 temperature adjustments, 1 range adjustment and bins of 20. On 3 seeds of Rastrigin 10 and Ackley 30, both did worse than the tutorial's settings: Rastrigin 10 ended at 2.2 to 4.7 and 10.3 to 10.9, against 1.4 to 2.7; Ackley 30 at 4.2 to 4.5 and 3.1 to 3.2, against 0.34 to 0.37.

**Keeping going:**
- sade and CMA-ES stop by themselves: sade when its population converges (`ftol`, `xtol`), CMA-ES when its population or its step converges (`ftol`, `xtol`) or after its 4,000 generations. Then they restart from a new random population, keeping the best solution ([`Restarts`](../../../benchmarks/adapters/pygmo/bench.py#L234)). This is the procedure of [cmaes_vs_xnes](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html), "the best practice ... when algorithms have well defined exit conditions", which assembles "the results in single runs containing multiple restarts". [cec2013_comp](https://esa.github.io/pygmo2/tutorials/cec2013_comp.html) says the same of CMA-ES: a comparison "should allow for restarts as to properly make use of the allowed budget". pygmo has no IPOP or BIPOP restarts, which grow the population.
- Simulated annealing stops at the end of its cooling schedule (500 evaluations per variable). It's annealed again from its best point, as in [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html) ("since we will be using some reannealing"): each call of `evolve` on the same population starts from its best individual and puts its best back ([`Reanneal`](../../../benchmarks/adapters/pygmo/bench.py#L254)).

**Left out:**
- `pso`, which the adapter ran before: in the averaged plot of [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html), PSO stalls furthest from the optimum of the 7 algorithms.
- `sga` (the GA the adapter ran before): pygmo's tutorials don't use it for continuous problems.
- `xnes`: [cmaes_vs_xnes](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html) runs it on these functions, and concludes that "CMA-ES is, on these three problems considered, outperforming consistently xNES".
- `sea` ((N+1)-ES): the other of "the two most successful algorithms" of [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html). At most 3 methods, and in that tutorial's averaged plot simulated annealing gets there first.
- `de`: pagmo presents sade as its improvement ("The original Differential Evolution algorithm (pagmo::de) can be significantly improved introducing the idea of parameter self-adaptation", [pagmo's sade docs](https://esa.github.io/pagmo2/docs/cpp/algorithms/sade.html)). `de1220` adds the adaptation of the variant to sade, and trails it in the Schwefel tutorial's plot.
- `bee_colony`: behind sade, de1220, simulated annealing and SEA in that plot.
- `gwo`: its own docs say its update rule "is also not particulary effective and results in a rather poor performance most of times".
- `pso_gen`: the generational PSO, "suited for stochastic optimization problems".
- `gaco` and `ihs`: pygmo's tutorials don't use them on these problems.
- `mbh` (monotonic basin hopping): a meta-algorithm for local optimizers ("When a population containing a single individual is used and coupled with a local optimizer, the original method is recovered"). The tutorials use it with NLopt's SLSQP on constrained problems. It stops after a number of perturbations without improvement.
- NLopt, Ipopt and SciPy local optimizers: local methods.
- Archipelagos and islands: pygmo's way to run several populations in parallel, which one thread rules out (rule 4.3).
- pygmo's own `rastrigin` and `ackley` problems, in C++: they aren't shifted, and the adapter must count evaluations around its own fitness functions (rule 3).

**Separate tests** (2026-09-25, pygmo 2.19.8, seeds 0 to 4, the scenario's budget and the 60 s cap):

Rastrigin 10 (500,000 evaluations):

| Solver | Runs | Reached | Median evaluations to the target | Best: median (best, worst) | Runs at the cap | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 4,903 | 0.0055 (0.00096, 0.0099) | 0 | 0 |
| cma_es | 5 | 5 | 174,950 | 0.0078 (0.0041, 0.0091) | 0 | 11 |
| simulated_annealing | 5 | 0 | - | 2.03 (1.38, 2.68) | 0 | 99 |

Rastrigin 30 (2,000,000 evaluations):

| Solver | Runs | Reached | Median evaluations to the target | Best: median (best, worst) | Runs at the cap | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 15,275 | 0.0096 (0.0085, 0.0100) | 0 | 0 |
| cma_es | 5 | 3 | 617,169 | 0.0099 (0.0072, 0.995) | 0 | 12 |
| simulated_annealing | 5 | 0 | - | 12.6 (9.58, 15.5) | 0 | 133 |

Ackley 30 (1,000,000 evaluations):

| Solver | Runs | Reached | Median evaluations to the target | Best: median (best, worst) | Runs at the cap | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 11,888 | 0.0095 (0.0088, 0.0099) | 0 | 0 |
| cma_es | 5 | 5 | 3,802 | 0.0095 (0.0083, 0.0096) | 0 | 0 |
| simulated_annealing | 5 | 0 | - | 0.338 (0.330, 0.388) | 0 | 66 |

sade never needed a restart. CMA-ES restarts from local optima on Rastrigin: the 2 Rastrigin 30 runs that missed both ended at 0.995, one variable a period away from the optimum, after the whole budget and 27 restarts. Simulated annealing gets close on Ackley, but no reannealing reaches 0.01 on either function; the longest run took 8.7 s.

## Continuous, unimodal: Rosenbrock 10

**Methods** ([bench.py#L332](../../../benchmarks/adapters/pygmo/bench.py#L332)):
- `sade`: as above. [evolving_a_population](https://esa.github.io/pygmo2/tutorials/evolving_a_population.html) runs it on this very problem, Rosenbrock 10, with its defaults and a population of 20.
- `cma_es`: as above, with the settings of [cmaes_vs_xnes](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html), whose code runs Rosenbrock 10 with populations of 10, 20 and 30. The adapter takes 10, the fewest median evaluations to the target in its figure.
- `xnes`: [`xnes`](https://esa.github.io/pygmo2/algorithms.html#pygmo.xnes), exponential natural evolution strategies, the other method of the same tutorial, with the same settings (`gen=4000, ftol=1e-8, xtol=1e-10`, `force_bounds=True`, the other parameters default) and population 10, also the fewest median evaluations in its figure.

**Keeping going:** all three restart from a new random population when they stop by themselves, as above.

**Left out:**
- `de`: [using_island](https://esa.github.io/pygmo2/tutorials/using_island.html) and [coding_udi](https://esa.github.io/pygmo2/tutorials/coding_udi.html) run it on Rosenbrock 10 to show islands, and pagmo presents sade as its improvement.
- `simulated_annealing` and `sea`: pygmo's tutorials use them on Schwefel, not on a unimodal function.
- NLopt, Ipopt and SciPy local optimizers: the [nlopt tutorial](https://esa.github.io/pygmo2/tutorials/nlopt_basics.html) uses them on a constrained problem with gradients, and nothing in pygmo's docs points to them for Rosenbrock.
- The others, as above.

**Separate tests** (2026-09-25, pygmo 2.19.8, seeds 0 to 4, 500,000 evaluations and the 60 s cap):

| Solver | Runs | Reached | Median evaluations to the target | Best: median (best, worst) | Runs at the cap | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 28,228 | 0.0093 (0.0086, 0.0100) | 0 | 0 |
| cma_es | 5 | 5 | 5,453 | 0.0075 (0.0062, 0.0096) | 0 | 0 |
| xnes | 5 | 5 | 111,071 | 0.0099 (0.0095, 0.0099) | 0 | 7 |

CMA-ES and xNES run with `force_bounds`, which clips each sample to the bounds. pagmo's source warns that this "screws up the whole covariance matrix machinery and worsen performances considerably". xNES suffers from it here: it starts with a step of the bounds' whole width, and its restarts often converge with a variable stuck at a bound. In 6 of the first 7 restarts of seed 0, x₁₀ ended at 10, with values from 304 to 55,495; the other ended in the local optimum near (−1, 1, …, 1), at 3.99. Without `force_bounds`, as in the tutorial, xNES reached the target in 7,807 to 13,676 evaluations on the same 5 seeds, without a restart, and CMA-ES in 5,334 to 6,769. But then they evaluate points outside the bounds, which aren't part of the benchmark's problems. The tutorial runs pygmo's own Rosenbrock, which accepts any point.

## Multi-objective (matched): ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Methods** ([bench.py#L376](../../../benchmarks/adapters/pygmo/bench.py#L376)):
- `nsga2`: [`nsga2`](https://esa.github.io/pygmo2/algorithms.html#pygmo.nsga2) with the matched settings: 100 individuals (92 with 3 objectives), SBX with η 15 at 0.9, polynomial mutation with η 20 at 1 / n. pagmo's SBX is Deb's bounded SBX.
- `moead`: [`moead`](https://esa.github.io/pygmo2/algorithms.html#pygmo.moead) with the matched settings: 100 grid weights (91 with 3 objectives), 20 neighbours, parents from the neighbourhood with probability 0.9 (`realb`), Tchebycheff for ZDT and PBI with θ 5 (pagmo's `"bi"`) for DTLZ. Difference: pagmo's MOEA/D is the DE variant, MOEA/D-DE, with DE's CR 1 and F 0.5, then polynomial mutation with η 20; it has no SBX. At most 2 copies of a child replace neighbours (`limit`, its default).
- `nspso`: [`nspso`](https://esa.github.io/pygmo2/algorithms.html#pygmo.nspso), non-dominated sorting PSO, with the settings of [nspso_tutorial_zdt1_2](https://esa.github.io/pygmo2/tutorials/nspso_tutorial_zdt1_2.html), "the same input parameters suggested in the original paper": ω 0.001, c1 2, c2 2, χ 1, v_coeff 0.5, leader selection range 100, crowding distance. With NSGA-II's population, as the scenario sets it (the tutorial uses 200). It's not one of the matched algorithms: an idiomatic run in a matched scenario.

The front is the non-dominated part of the final population (rule 7.2), from `pg.fast_non_dominated_sorting`.

**Keeping going:** these algorithms have no stop criterion besides their generations: one call of `evolve` with the generations of the budget. If the initial population shows that these wouldn't fit in the 60 s cap, the adapter runs one generation per call and checks the time after each; it never happened here.

**Left out:**
- `moead_gen`: MOEA/D-DE that makes a generation at once, for batch evaluators and "expensive fitness functions".
- `maco`: at most 3 methods. [zdt3_maco_benchmark](https://esa.github.io/pygmo2/tutorials/zdt3_maco_benchmark.html) compares it with MOEA/D and NSGA-II.
- NSGA-III: pagmo 2.19.1 doesn't have it.

**Separate tests** (2026-09-25, pygmo 2.19.8, seeds 0 to 4, the scenario's budget; hypervolumes computed by `run.py`'s `hypervolume`):

| Scenario | Solver | Runs | Hypervolume: median (best, worst) | Evaluations | Front size: median |
|---|---|---|---|---|---|
| ZDT1 (25,000) | nsga2 | 5 | 0.8699 (0.8701, 0.8695) | 25,000 | 100 |
| | moead | 5 | 0.8473 (0.8602, 0.8400) | 25,000 | 100 |
| | nspso | 5 | 0.8620 (0.8633, 0.8597) | 25,000 | 79 |
| ZDT2 (25,000) | nsga2 | 5 | 0.5363 (0.5365, 0.5361) | 25,000 | 100 |
| | moead | 5 | 0.4669 (0.4996, 0.4131) | 25,000 | 92 |
| | nspso | 5 | 0.5295 (0.5309, 0.1100) | 25,000 | 85 |
| ZDT3 (25,000) | nsga2 | 5 | 1.3274 (1.3276, 1.3272) | 25,000 | 100 |
| | moead | 5 | 1.1730 (1.2314, 1.1324) | 25,000 | 83 |
| | nspso | 5 | 1.2894 (1.2966, 1.2876) | 25,000 | 27 |
| DTLZ2 (25,000) | nsga2 | 5 | 0.6972 (0.7016, 0.6793) | 25,024 | 92 |
| | moead | 5 | 0.6600 (0.6695, 0.6494) | 25,025 | 81 |
| | nspso | 5 | 0.4633 (0.4811, 0.4101) | 25,024 | 44 |
| DTLZ1 (40,000) | nsga2 | 5 | 1.3007 (1.3019, 1.2988) | 40,020 | 92 |
| | moead | 5 | 1.3009 (1.3029, 0.0000) | 40,040 | 91 |
| | nspso | 5 | 0.0000 (0.0000, 0.0000) | 40,020 | 17 |

No run reached the cap; the longest took 0.5 s. NSPSO doesn't converge on DTLZ1, whose distance function is multimodal: its front stays outside the reference point. One MOEA/D run of DTLZ1 (seed 0) does the same.

## Can't run

N-Queens 32 and 64: pygmo has no permutation representation. Its integer genes can't be kept a permutation, and pygmo's algorithms have no permutation operators.

## Bugs found

None in the algorithms. A documentation slip: the docstrings of `sade`, `de1220`, `cmaes` and `xnes` describe `ftol` as "stopping criteria on the x tolerance" and `xtol` as "on the f tolerance", swapped. The code checks them the right way round.
