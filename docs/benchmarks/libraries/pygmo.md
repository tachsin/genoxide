# pygmo (C++ via Python, 2.19.8)

pygmo is the Python interface of pagmo, ESA's C++ library of optimization algorithms built around the island model. The 2.19.8 wheel bundles pagmo 2.19.1. The algorithms run in C++ and call a Python user-defined problem (UDP) for every fitness evaluation. Its docs are at [esa.github.io/pygmo2](https://esa.github.io/pygmo2/): the [list of algorithms](https://esa.github.io/pygmo2/overview.html#list-of-algorithms), with the problem types each one handles, and the [tutorials](https://esa.github.io/pygmo2/tutorials/tutorials.html), which are where pygmo shows which algorithm and settings to use for which problem. pagmo's C++ docs are at [esa.github.io/pagmo2](https://esa.github.io/pagmo2/).

Adapter: [benchmarks/adapters/pygmo/](../../../benchmarks/adapters/pygmo/).
Know a better way to solve one of these problems with pygmo? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs pygmo

- **Fitness functions:** a UDP class whose `fitness(x)` returns a list ([`Problem`](../../../benchmarks/adapters/pygmo/bench.py#L182)). pygmo passes `x` as a numpy array, and the fitness functions use numpy arithmetic on it, as [tutorials/coding_udp_simple](https://esa.github.io/pygmo2/tutorials/coding_udp_simple.html) shows: "It is important to remember that x is a NumPy array, so that the NumPy array arithmetic applies in the body of fitness()". OneMax is maximized, so its UDP returns minus the number of ones: pagmo always minimizes. The functions are at [bench.py#L96](../../../benchmarks/adapters/pygmo/bench.py#L96) to [#L167](../../../benchmarks/adapters/pygmo/bench.py#L167).
- **Counting:** the UDP counts every call in the adapter's own [`Counter`](../../../benchmarks/adapters/pygmo/bench.py#L48), initial populations and restarts included, and keeps the best solution. pygmo's own count (`get_fevals`) isn't used.
- **Bounds (rule 2.4):** the UDP also counts the evaluated solutions outside the bounds, as pagmo proposed them, and every continuous and multi-objective run prints that count as `outside`. It was 0 in every run. Each algorithm's own bound handling keeps it there; the sections below say which.
- **Stopping:** in a single-objective run, the counter raises an exception from inside the fitness at the target, the budget or the 60 s cap. It passes through pagmo's C++ back to the adapter, so a run ends at that evaluation and never goes past the budget. A multi-objective run has no target: it runs the generations of the budget in one call of `evolve` ([`evolve_front`](../../../benchmarks/adapters/pygmo/bench.py#L396)).
- **Keeping going (rule 2.2):** every pygmo algorithm takes a number of generations, `gen`. That's only a budget, so it's lifted: `gen` covers the whole evaluation budget. An algorithm that stops on convergence (a tolerance, or no improvement) restarts from a new random population ([`Restarts`](../../../benchmarks/adapters/pygmo/bench.py#L240)), with the seed `seed * 1000 + restart` for the population and the algorithm. pygmo has no restart mechanism of its own; this is the procedure of the [cmaes_vs_xnes tutorial](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html), "the best practice ... when algorithms have well defined exit conditions", which assembles "the results in single runs containing multiple restarts". Simulated annealing is the exception, below.
- **One thread:** no islands, archipelagos or batch fitness evaluators, which are pygmo's ways to parallelize, and numpy's BLAS with 1 thread. `run.py check` measures a CPU time equal to the wall time (1.00).
- **Seeds:** the seed goes to `pg.population(..., seed=)` and to the algorithm's `seed`. The same seed gives the same evaluations and best value.
- **The docs decide (rule 6.2):** where pygmo's docs show several settings, the adapter takes a preference they state, else their example for the problem type, else the default. The separate tests below never picked anything.

## Binary: OneMax 100 and 1000 (matched), OneMax 100 (idiomatic)

The bits are integer genes in [0, 1] (`get_nix`, as in [tutorials/coding_udp_minlp](https://esa.github.io/pygmo2/tutorials/coding_udp_minlp.html)).

**Methods:**
- **Matched** ([bench.py#L296](../../../benchmarks/adapters/pygmo/bench.py#L296)): [`sga`](https://esa.github.io/pygmo2/algorithms.html#pygmo.sga) with a population of 300, tournaments of 3, single-point crossover at 0.5 and uniform mutation at 0.4 / n per gene. The differences from the matched GA:
  - pagmo's sga has no two-point crossover, so it's single-point. Each selected parent makes one child with a random partner.
  - Its uniform mutation draws an integer gene again from its bounds, which flips a bit half the time. So the rate is doubled to 0.4 / n, which flips 0.2 bits per child, like 20% of the children with 1 / n. It mutates every child, not 20% of them.
  - Its reinsertion is elitist: the best 300 of parents and children survive. It can't be turned off ("the only reinsertion strategy provided is what we call pure elitism").
- **Idiomatic** ([bench.py#L308](../../../benchmarks/adapters/pygmo/bench.py#L308)): pygmo doesn't recommend an algorithm for binary problems. Its [list of algorithms](https://esa.github.io/pygmo2/overview.html#list-of-algorithms) flags three single-objective algorithms for integer programming ("I"), and the adapter runs all three with their documented defaults:
  - `ga`: [`sga`](https://esa.github.io/pygmo2/algorithms.html#pygmo.sga), pygmo's genetic algorithm, with its defaults: exponential crossover at 0.9, mutation at 0.02 per gene (for an integer gene, its polynomial mutation draws the gene again from its bounds), tournaments of 2.
  - `ihs`: [`ihs`](https://esa.github.io/pygmo2/algorithms.html#pygmo.ihs), improved harmony search, with its defaults.
  - `gaco`: [`gaco`](https://esa.github.io/pygmo2/algorithms.html#pygmo.gaco), extended ant colony optimization, which its docs present for "both continuous and integer variables", with its defaults.
  - Populations: pygmo's docs give none for these algorithms. sga and ihs use the population of 20 of pygmo's tutorials ([evolving_a_population](https://esa.github.io/pygmo2/tutorials/evolving_a_population.html), [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html), and pagmo's [quick start](https://esa.github.io/pagmo2/quickstart.html)). gaco uses 63: its default kernel of 63 solutions needs a population at least that large.

**Keeping going:** sga and ihs have no convergence criterion: their `gen` covers the budget, and they run to it. gaco stops on convergence after 100,000 generations or evaluations without improvement (its `impstop` and `evalstop`), and then restarts; it never got there in these tests.

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
| gaco | 5 | 0 | - | 75 (77, 73) | 0 |

ihs and gaco use the whole budget without reaching all ones. With their defaults they're slow on OneMax; nothing in pygmo's docs suggests other settings for binary problems.

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** ([bench.py#L325](../../../benchmarks/adapters/pygmo/bench.py#L325)):
- `sade`: [`sade`](https://esa.github.io/pygmo2/algorithms.html#pygmo.sade), self-adaptive differential evolution in its jDE variant, with its defaults: variant rand/1/exp, jDE adaptation, `ftol` and `xtol` 1e-6. Population 20, as in the examples on multimodal functions: pagmo's [quick start](https://esa.github.io/pagmo2/quickstart.html) runs it on Schwefel 30 (islands of 20), and [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html) with 20 individuals. In [cec2013_comp](https://esa.github.io/pygmo2/tutorials/cec2013_comp.html), "the particular instances choosen for cmaes and sade (jDE) are performing particularly well".
- `cma_es`: [`cmaes`](https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes) as in the example of [cec2013_comp](https://esa.github.io/pygmo2/tutorials/cec2013_comp.html), pygmo's example on multimodal functions (the CEC 2013 suite): `cmaes(gen=1000, ftol=1e-9, xtol=1e-9)`, the other parameters at their defaults (σ0 0.5 of the bounds' width), and "we choose a population of 50". The same tutorial asks for restarts: a comparison "should allow for restarts as to properly make use of the allowed budget".
  - How the docs decided: [cmaes_vs_xnes](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html) also shows CMA-ES on Rastrigin and Ackley, but only in figures, with three population sizes each and no preference stated between them. Its code example is Rosenbrock's. cec2013_comp is the example in code and in words for multimodal functions.
- `simulated_annealing`: [`simulated_annealing`](https://esa.github.io/pygmo2/algorithms.html#pygmo.simulated_annealing), Corana's, one of "the two most successful algorithms" of [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html). With that tutorial's example: `simulated_annealing(10, 0.01, 5)` (Ts 10, Tf 0.01, 5 temperature adjustments, the others at their defaults), a population of 20, from whose best it starts, and reannealing.
  - How the docs decided: this is the one example in code for simulated annealing on a multimodal function. The [cec2013_comp](https://esa.github.io/pygmo2/tutorials/cec2013_comp.html) figures print other settings (Tf 0.1, 300 temperature adjustments, 1 range adjustment, bins of 20), but only inside a figure, with no code.
  - SEA is the other of the two most successful; at most 3 methods. The docs state no preference between them, and in that tutorial's own averaged plot simulated annealing gets to the optimum first.

**Bounds:**
- sade: "the correction applied to an allele when some mutation puts it outside the allowed box-bounds, is here done by creating a random number in the bounds" ([pagmo's sade docs](https://esa.github.io/pagmo2/docs/cpp/algorithms/sade.html)).
- CMA-ES: `force_bounds=True`, pagmo's only bound handling for it, which clips each sample to the bounds: "The fitness will never be called outside the bounds but the covariance matrix adaptation mechanism will worsen". The tutorials run pygmo's own problems, which accept any point, without it.
- Simulated annealing mutates a variable within the bounds (`[max(x − w, lb), min(x + w, ub)]`).

**Keeping going:**
- sade and CMA-ES: `gen` lifted; they restart when they converge: sade when its population converges (`ftol`, `xtol`), CMA-ES when its population or its step does (`ftol`, `xtol`). pygmo has no IPOP or BIPOP restarts, which grow the population.
- Simulated annealing: its cooling schedule has a fixed length (500 evaluations per variable here), and that length sets the cooling rate, (Tf / Ts)^(1 / n_T_adj) per adjustment, so it can't be lifted without changing the method. It's annealed again from its best point, as in [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html) ("since we will be using some reannealing"): each call of `evolve` on the same population starts from its best individual and puts its best back ([`Reanneal`](../../../benchmarks/adapters/pygmo/bench.py#L261)).

**Left out:**
- `pso`, which the adapter ran before: in the averaged plot of [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html), PSO stalls furthest from the optimum of the 7 algorithms.
- `sga` (the GA the adapter ran before): pygmo's tutorials don't use it for continuous problems.
- `xnes`: [cmaes_vs_xnes](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html) runs it on these functions, and concludes that "CMA-ES is, on these three problems considered, outperforming consistently xNES".
- `sea` ((N+1)-ES): see simulated annealing above.
- `de`: pagmo presents sade as its improvement ("The original Differential Evolution algorithm (pagmo::de) can be significantly improved introducing the idea of parameter self-adaptation", [pagmo's sade docs](https://esa.github.io/pagmo2/docs/cpp/algorithms/sade.html)). `de1220` adds the adaptation of the variant to sade, and trails it in the Schwefel tutorial's plot.
- `bee_colony`: behind sade, de1220, simulated annealing and SEA in that plot.
- `gwo`: its own docs say its update rule "is also not particulary effective and results in a rather poor performance most of times".
- `pso_gen`: the generational PSO, "suited for stochastic optimization problems".
- `gaco` and `ihs`: pygmo's tutorials don't use them on these problems.
- `mbh` (monotonic basin hopping): a meta-algorithm for local optimizers ("When a population containing a single individual is used and coupled with a local optimizer, the original method is recovered"). The tutorials use it with NLopt's SLSQP on constrained problems.
- NLopt, Ipopt and SciPy local optimizers: local methods.
- Archipelagos and islands: pygmo's way to run several populations in parallel, which one thread rules out (rule 4.3).
- pygmo's own `rastrigin` and `ackley` problems, in C++: they aren't shifted, and the adapter must count evaluations around its own fitness functions (rule 3).

**Separate tests** (2026-09-25, pygmo 2.19.8, seeds 0 to 4, the scenario's budget and the 60 s cap; no evaluated solution outside the bounds):

Rastrigin 10 (500,000 evaluations):

| Solver | Runs | Reached | Median evaluations to the target | Best: median (best, worst) | Runs at the cap | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 4,633 | 0.0091 (0.0038, 0.0096) | 0 | 0 |
| cma_es | 5 | 3 | 294,520 | 0.0093 (0.0075, 0.995) | 0 | 44 |
| simulated_annealing | 5 | 0 | - | 2.03 (1.38, 2.68) | 0 | 99 |

Rastrigin 30 (2,000,000 evaluations):

| Solver | Runs | Reached | Median evaluations to the target | Best: median (best, worst) | Runs at the cap | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 15,301 | 0.0091 (0.0054, 0.0099) | 0 | 0 |
| cma_es | 5 | 0 | - | 6.96 (3.98, 8.95) | 0 | 106 |
| simulated_annealing | 5 | 0 | - | 12.6 (9.58, 15.5) | 0 | 133 |

Ackley 30 (1,000,000 evaluations):

| Solver | Runs | Reached | Median evaluations to the target | Best: median (best, worst) | Runs at the cap | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 11,480 | 0.0095 (0.0089, 0.0099) | 0 | 0 |
| cma_es | 5 | 5 | 7,223 | 0.0096 (0.0092, 0.0100) | 0 | 0 |
| simulated_annealing | 5 | 0 | - | 0.338 (0.330, 0.388) | 0 | 66 |

sade never needed a restart. CMA-ES with 50 individuals converges to a local optimum of Rastrigin in each attempt and restarts: the 2 Rastrigin 10 runs that missed ended at 0.995 (one variable a period away from the optimum) after 55 and 57 restarts, and on Rastrigin 30 none of its 105 to 108 attempts per run reached the target; the longest run took 33 s. Simulated annealing gets close on Ackley, but no reannealing reaches 0.01 on either function.

## Continuous, unimodal: Rosenbrock 10

**Methods** ([bench.py#L325](../../../benchmarks/adapters/pygmo/bench.py#L325)):
- `sade`: as above. [evolving_a_population](https://esa.github.io/pygmo2/tutorials/evolving_a_population.html) is the example on this very problem, Rosenbrock 10, with its defaults and a population of 20.
- `cma_es`: the example of [cmaes_vs_xnes](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html), whose code runs this very problem, Rosenbrock 10: `cmaes(gen=4000, ftol=1e-8, xtol=1e-10)` with `popsizes = [10,20,30]`, and restarts.
  - How the docs decided: the example runs all three sizes and states no preference between them, and cmaes has no default population. The adapter takes the first listed, 10.
- `xnes`: [`xnes`](https://esa.github.io/pygmo2/algorithms.html#pygmo.xnes), exponential natural evolution strategies, the other method of the same example, with the same settings and population 10, the first listed.

**Bounds:** sade as above; CMA-ES and xNES with `force_bounds=True`, which clips each sample to the bounds.

**Keeping going:** all three have `gen` lifted and restart from a new random population when they converge (`ftol`, `xtol`).

**Left out:**
- `de`: [using_island](https://esa.github.io/pygmo2/tutorials/using_island.html) and [coding_udi](https://esa.github.io/pygmo2/tutorials/coding_udi.html) run it on Rosenbrock 10 to show islands, and pagmo presents sade as its improvement.
- `simulated_annealing` and `sea`: pygmo's tutorials use them on Schwefel, not on a unimodal function.
- NLopt, Ipopt and SciPy local optimizers: the [nlopt tutorial](https://esa.github.io/pygmo2/tutorials/nlopt_basics.html) uses them on a constrained problem with gradients, and nothing in pygmo's docs points to them for Rosenbrock.
- The others, as above.

**Separate tests** (2026-09-25, pygmo 2.19.8, seeds 0 to 4, 500,000 evaluations and the 60 s cap; no evaluated solution outside the bounds):

| Solver | Runs | Reached | Median evaluations to the target | Best: median (best, worst) | Runs at the cap | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 19,683 | 0.0092 (0.0077, 0.0100) | 0 | 0 |
| cma_es | 5 | 5 | 4,864 | 0.0089 (0.0075, 0.0094) | 0 | 0 |
| xnes | 5 | 5 | 127,517 | 0.0098 (0.0094, 0.0099) | 0 | 6 |

`force_bounds` clips each sample to the bounds, and pagmo's source warns that this "screws up the whole covariance matrix machinery and worsen performances considerably". xNES suffers from it here: it starts with a step of the bounds' whole width, and its attempts often converge with a variable stuck at a bound. In 6 of the first 7 attempts of seed 0, x₁₀ ended at 10, with values from 304 to 55,495; the other ended in the local optimum near (−1, 1, …, 1), at 3.99. For information only, as the tutorial runs it: without `force_bounds`, xNES reached the target in 6,759 to 10,886 evaluations on the same 5 seeds, without a restart, and CMA-ES in 4,807 to 6,534. But then they evaluate points outside the bounds, which rule 2.4 forbids.

## Multi-objective (matched): ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Methods** ([bench.py#L379](../../../benchmarks/adapters/pygmo/bench.py#L379)):
- `nsga2`: [`nsga2`](https://esa.github.io/pygmo2/algorithms.html#pygmo.nsga2) with the matched settings: 100 individuals (92 with 3 objectives), SBX with η 15 at 0.9, polynomial mutation with η 20 at 1 / n.

The front is the non-dominated part of the final population (rule 7.2), from `pg.fast_non_dominated_sorting`.

**Bounds:** pagmo's SBX clips the children to the bounds, and its polynomial mutation is Deb's bounded one.

**Keeping going:** NSGA-II has no stop criterion besides its generations: one call of `evolve` with the generations of the budget. If the initial population shows that these wouldn't fit in the 60 s cap, the adapter runs one generation per call and checks the time after each; it never happened here.

**Not run** (rule 6.1: the matched scenarios run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA, with the matched operators):
- `moead` and `moead_gen`: pagmo's MOEA/D is only the DE variant, MOEA/D-DE ("Multi Objective Evolutionary Algorithms by Decomposition (the DE variant)"): its operators are DE's, then polynomial mutation, and it can't use the matched SBX.
- `nspso` and `maco`: not matched algorithms.
- NSGA-III, SPEA2 and SMS-EMOA: pagmo 2.19.1 doesn't have them.

**Separate tests** (2026-09-25, pygmo 2.19.8, seeds 0 to 4, the scenario's budget; hypervolumes computed by `run.py`'s `hypervolume`; no evaluated solution outside the bounds):

| Scenario | Solver | Runs | Hypervolume: median (best, worst) | Evaluations | Front size: median |
|---|---|---|---|---|---|
| ZDT1 (25,000) | nsga2 | 5 | 0.8699 (0.8701, 0.8695) | 25,000 | 100 |
| ZDT2 (25,000) | nsga2 | 5 | 0.5363 (0.5365, 0.5361) | 25,000 | 100 |
| ZDT3 (25,000) | nsga2 | 5 | 1.3274 (1.3276, 1.3272) | 25,000 | 100 |
| DTLZ2 (25,000) | nsga2 | 5 | 0.6972 (0.7016, 0.6793) | 25,024 | 92 |
| DTLZ1 (40,000) | nsga2 | 5 | 1.3007 (1.3019, 1.2988) | 40,020 | 92 |

No run reached the cap; the longest took 0.5 s.

## Can't run

N-Queens 32 and 64: pygmo has no permutation representation. Its integer genes can't be kept a permutation, and pygmo's algorithms have no permutation operators.

## Bugs found

None in the algorithms. A documentation slip: the docstrings of `sade`, `de1220`, `cmaes` and `xnes` describe `ftol` as "stopping criteria on the x tolerance" and `xtol` as "on the f tolerance", swapped. The code checks them the right way round.
