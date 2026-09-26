# pygmo (C++ via Python, 2.19.8)

pygmo is the Python interface of pagmo, ESA's C++ library of optimization algorithms built around the island model; the 2.19.8 wheel bundles pagmo 2.19.1. The algorithms run in C++ and call a Python user-defined problem (UDP) for every evaluation. Its docs are at [esa.github.io/pygmo2](https://esa.github.io/pygmo2/): the [list of algorithms](https://esa.github.io/pygmo2/overview.html#list-of-algorithms), with the problem types each handles, and the [tutorials](https://esa.github.io/pygmo2/tutorials/tutorials.html), which show which algorithm and settings to use for which problem. pagmo's C++ docs are at [esa.github.io/pagmo2](https://esa.github.io/pagmo2/).

Adapter: [benchmarks/adapters/pygmo/](../../../benchmarks/adapters/pygmo/).
Know a better way to solve one of these problems with pygmo? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs pygmo

- **Fitness functions:** a UDP with `fitness(x)` and `batch_fitness(dvs)` ([`Problem`](../../../benchmarks/adapters/pygmo/bench.py#L250-L295)), in numpy, as [coding_udp_simple](https://esa.github.io/pygmo2/tutorials/coding_udp_simple.html) shows ([bench.py#L157-L235](../../../benchmarks/adapters/pygmo/bench.py#L157-L235)). pagmo minimizes, so OneMax returns minus the number of ones.
- **Batch evaluation (rule 3.4):** `cmaes`, `gaco` and `nsga2` get [`member_bfe`](https://esa.github.io/pygmo2/bfe.html#pygmo.member_bfe), which calls `batch_fitness` in the same thread ([`with_bfe`](../../../benchmarks/adapters/pygmo/bench.py#L298-L302)); with the same seed, each gave the same evaluations and final population with and without it. `sga`, `ihs`, `sade`, `xnes` and `simulated_annealing` take no evaluator.
- **Evaluations:** the adapter's [`Counter`](../../../benchmarks/adapters/pygmo/bench.py#L82-L126) counts every row, keeps the best and records the first hit; `get_fevals` isn't used.
- **Stop:** in a single-objective run, the counter raises an exception from inside the fitness at the target, the budget (a batch is cut there) or the 60 s cap; it passes through pagmo's C++ to the adapter. A multi-objective run evolves the budget's generations in one call ([`evolve_front`](../../../benchmarks/adapters/pygmo/bench.py#L484-L501)).
- **Keeping going (rule 2.2):** `gen` is lifted to cover the budget. An algorithm that stops on convergence restarts from a new random population with the seeds of rule 2.2 ([`Restarts`](../../../benchmarks/adapters/pygmo/bench.py#L329-L347)), the procedure of the [cmaes_vs_xnes tutorial](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html) ("the best practice ... when algorithms have well defined exit conditions"); pygmo has no restart mechanism. Simulated annealing is the exception, below.
- **Bounds (rule 2.4):** each algorithm's own, per section; counted in the UDP.
- **One thread:** no islands, archipelagos or parallel evaluators; numpy's BLAS with 1 thread. pagmo's `batch_fitness` also runs TBB worker threads (20 in a test, 1.6 times more CPU than wall time), with no pygmo setting for them. The adapter starts TBB with one CPU, by a batch evaluation of a trivial problem before any run, and then restores the process's CPUs ([`start_tbb_with_one_thread`](../../../benchmarks/adapters/pygmo/bench.py#L63-L72)).
- **Seeds:** `pg.population(..., seed=)` and the algorithm's `seed`.
- **Separate tests:** 2026-09-25, pygmo 2.19.8, seeds 0 to 4, the scenario's budget, 60 s cap, rule 5.3, with other tests on the machine. `outside` was 0 in every run.

## Binary: OneMax 100 (idiomatic)

The bits are integer genes in [0, 1] (`get_nix`, as in [coding_udp_minlp](https://esa.github.io/pygmo2/tutorials/coding_udp_minlp.html)).

**Methods** ([bench.py#L391-L416](../../../benchmarks/adapters/pygmo/bench.py#L391-L416)): pygmo recommends nothing for binary problems; its [list of algorithms](https://esa.github.io/pygmo2/overview.html#list-of-algorithms) flags three single-objective ones for integers ("I"), all run with their defaults:
- **`ga`:** [`sga`](https://esa.github.io/pygmo2/algorithms.html#pygmo.sga): exponential crossover at 0.9, mutation at 0.02 per gene (an integer gene is drawn again), tournaments of 2.
- **`ihs`:** [`ihs`](https://esa.github.io/pygmo2/algorithms.html#pygmo.ihs), improved harmony search.
- **`gaco`:** [`gaco`](https://esa.github.io/pygmo2/algorithms.html#pygmo.gaco), extended ant colony optimization ("both continuous and integer variables"), with `member_bfe`.
- Populations: 20 for sga and ihs, the tutorials' size ([evolving_a_population](https://esa.github.io/pygmo2/tutorials/evolving_a_population.html), [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html), pagmo's [quick start](https://esa.github.io/pagmo2/quickstart.html)); 63 for gaco, which its default kernel of 63 needs.

**Keeping going:** sga and ihs run to the budget. gaco stops after 100,000 generations or evaluations without improvement (`impstop`, `evalstop`) and restarts; it never got there.

**Left out:**
- NSGA-II and MACO, also flagged for integers: multi-objective.
- The other single-objective algorithms: they "will optimise the relaxed problem", so their bits wouldn't be 0 or 1.

**Separate tests** (the best value is the number of ones):

OneMax 100, idiomatic (200,000 evaluations):

| Solver | Runs | Reached | First hit: median evaluations | Best: median (best, worst) | Capped |
|---|---|---|---|---|---|
| ga | 5 | 5 | 1,582 | 100 (100, 100) | 0 |
| ihs | 5 | 0 | | 92 (92, 90) | 0 |
| gaco | 5 | 0 | | 74 (77, 73) | 0 |

## Continuous, multimodal: Rastrigin 10 and 30, Ackley 30

**Methods** ([bench.py#L418-L451](../../../benchmarks/adapters/pygmo/bench.py#L418-L451)):
- **`sade`:** [`sade`](https://esa.github.io/pygmo2/algorithms.html#pygmo.sade), jDE, with its defaults (rand/1/exp, `ftol` and `xtol` 1e-6) and population 20, as the examples on multimodal functions: pagmo's [quick start](https://esa.github.io/pagmo2/quickstart.html) (Schwefel 30, islands of 20) and [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html). [cec2013_comp](https://esa.github.io/pygmo2/tutorials/cec2013_comp.html): "the particular instances choosen for cmaes and sade (jDE) are performing particularly well".
- **`cma_es`:** [`cmaes`](https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes) as in [cmaes_vs_xnes](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html): `cmaes(gen=4000, ftol=1e-8, xtol=1e-10)`, other parameters default (σ0 0.5 of the width), restarts, `member_bfe`. Its figures run Rastrigin and Ackley with three population sizes each, no preference; the first listed runs ([`cmaes_population`](../../../benchmarks/adapters/pygmo/bench.py#L376-L386)): 40 for Rastrigin 10 (40, 60, 100); from the dimension-20 figures, the closest, 100 for Rastrigin 30 (100, 150, 200) and 20 for Ackley 30 (20, 30, 40). [cec2013_comp](https://esa.github.io/pygmo2/tutorials/cec2013_comp.html) gives "a population of 50" only at dimension 2, and "a larger population size" at 10 without a number. Both tutorials ask for restarts.
- **`simulated_annealing`:** Corana's [`simulated_annealing`](https://esa.github.io/pygmo2/algorithms.html#pygmo.simulated_annealing), one of "the two most successful algorithms" of [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html), with its example: `simulated_annealing(10, 0.01, 5)` (Ts 10, Tf 0.01, 5 temperature adjustments), a population of 20 whose best it starts from, and reannealing. It's the only example in code on a multimodal function; [cec2013_comp](https://esa.github.io/pygmo2/tutorials/cec2013_comp.html) prints other settings (Tf 0.1, 300 temperature adjustments, 1 range adjustment, bins of 20) only inside a figure.

**Bounds:**
- sade: an allele outside the bounds is replaced by "a random number in the bounds" ([sade docs](https://esa.github.io/pagmo2/docs/cpp/algorithms/sade.html)).
- CMA-ES: `force_bounds=True`, its only bound handling, which clips each sample ("the covariance matrix adaptation mechanism will worsen"). The tutorials run pygmo's own problems without it.
- Simulated annealing mutates within `[max(x − w, lb), min(x + w, ub)]`.

**Keeping going:**
- sade and CMA-ES restart when they converge (`ftol`, `xtol`). pygmo has no IPOP or BIPOP.
- Simulated annealing: its schedule length (500 evaluations per variable here) sets the cooling rate, (Tf / Ts)^(1 / n_T_adj), so it isn't lifted. It's annealed again from its best, as in [solving_schwefel_20](https://esa.github.io/pygmo2/tutorials/solving_schwefel_20.html) ("some reannealing"): each `evolve` starts from the population's best and puts its best back ([`Reanneal`](../../../benchmarks/adapters/pygmo/bench.py#L350-L367)).

**Left out:**
- `sea`, the Schwefel tutorial's other "most successful": at most 3 methods; in that tutorial's averaged plot simulated annealing gets to the optimum first.
- `pso`: in the same plot it ends furthest from the optimum of the 7 algorithms.
- `sga`, `gaco`, `ihs`: the tutorials don't use them on continuous problems.
- `xnes`: [cmaes_vs_xnes](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html) concludes that "CMA-ES is, on these three problems considered, outperforming consistently xNES".
- `de`: pagmo presents sade as its improvement ([sade docs](https://esa.github.io/pagmo2/docs/cpp/algorithms/sade.html)). `de1220` is behind sade in the Schwefel plot, and `bee_colony` behind sade, de1220, simulated annealing and SEA.
- `gwo`: its docs say it "results in a rather poor performance most of times".
- `pso_gen`: "suited for stochastic optimization problems".
- `mbh`: a meta-algorithm for local optimizers, used in the tutorials with NLopt's SLSQP on constrained problems. NLopt, Ipopt and SciPy optimizers: local methods.
- Archipelagos and islands: parallel populations (rule 4.3).
- pygmo's own `rastrigin` and `ackley`: not shifted, and evaluations must be counted around the adapter's functions (rule 3).

**Separate tests:**

Rastrigin 10 (500,000 evaluations):

| Solver | Runs | Reached | First hit: median evaluations | Best: median (best, worst) | Capped | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 4,973 | 0.0092 (0.0062, 0.0100) | 0 | 0 |
| cma_es | 5 | 0 | | 0.995 (0.995, 0.995) | 0 | 75 |
| simulated_annealing | 5 | 0 | | 2.12 (1.03, 3.04) | 0 | 99 |

Rastrigin 30 (2,000,000 evaluations):

| Solver | Runs | Reached | First hit: median evaluations | Best: median (best, worst) | Capped | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 14,423 | 0.0090 (0.0066, 0.0099) | 0 | 0 |
| cma_es | 5 | 0 | | 1.99 (1.99, 2.98) | 4 | 51 |
| simulated_annealing | 3 | 0 | | 16.3 (15.2, 20.6) | 3 | 46 |

Ackley 30 (1,000,000 evaluations):

| Solver | Runs | Reached | First hit: median evaluations | Best: median (best, worst) | Capped | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 12,276 | 0.0093 (0.0092, 0.0098) | 0 | 0 |
| cma_es | 5 | 5 | 3,909 | 0.0097 (0.0089, 0.0100) | 0 | 0 |
| simulated_annealing | 3 | 0 | | 0.328 (0.312, 0.384) | 3 | 52 |

Every CMA-ES run on Rastrigin 10 ended at 0.995, one variable a period away from the optimum. The capped runs stopped after about 1.6 million evaluations (CMA-ES, Rastrigin 30) and 700,000 to 800,000 (simulated annealing).

## Continuous, unimodal: Rosenbrock 10

**Methods** ([bench.py#L418-L451](../../../benchmarks/adapters/pygmo/bench.py#L418-L451)):
- **`sade`:** as above; [evolving_a_population](https://esa.github.io/pygmo2/tutorials/evolving_a_population.html) runs it on Rosenbrock 10 with its defaults and 20 individuals.
- **`cma_es`:** the [cmaes_vs_xnes](https://esa.github.io/pygmo2/tutorials/cmaes_vs_xnes.html) code on Rosenbrock 10: `cmaes(gen=4000, ftol=1e-8, xtol=1e-10)`, restarts, `member_bfe`, and population 10, the first of `popsizes = [10,20,30]` (no preference, no default).
- **`xnes`:** [`xnes`](https://esa.github.io/pygmo2/algorithms.html#pygmo.xnes), the same example's other method, with the same settings and population 10.

**Bounds:** sade as above; CMA-ES and xNES with `force_bounds=True`.

**Keeping going:** all three restart when they converge (`ftol`, `xtol`).

**Left out:**
- `de`: [using_island](https://esa.github.io/pygmo2/tutorials/using_island.html) and [coding_udi](https://esa.github.io/pygmo2/tutorials/coding_udi.html) run it on Rosenbrock 10 to show islands; sade is its improvement.
- `simulated_annealing`, `sea`: used on Schwefel, not on a unimodal function.
- NLopt, Ipopt and SciPy optimizers: the [nlopt tutorial](https://esa.github.io/pygmo2/tutorials/nlopt_basics.html) uses them on a constrained problem with gradients.
- The others, as above.

**Separate tests:**

| Solver | Runs | Reached | First hit: median evaluations | Best: median (best, worst) | Capped | Restarts: median |
|---|---|---|---|---|---|---|
| sade | 5 | 5 | 26,967 | 0.0097 (0.0092, 0.0100) | 0 | 0 |
| cma_es | 5 | 5 | 5,106 | 0.0080 (0.0075, 0.0098) | 0 | 0 |
| xnes | 5 | 5 | 29,152 | 0.0096 (0.0086, 0.0097) | 0 | 2 |

pagmo's source warns that `force_bounds` "screws up the whole covariance matrix machinery and worsen performances considerably". xNES starts with a step of the bounds' width, and its attempts often converge with a variable at a bound. Without `force_bounds` (5 seeds, earlier seeds `seed * 1000`), xNES reached the target in 6,759 to 10,886 evaluations and CMA-ES in 4,807 to 6,534, but they then evaluate points outside the bounds (rule 2.4).

## Multi-objective (matched): ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ1

**Methods** ([bench.py#L466-L481](../../../benchmarks/adapters/pygmo/bench.py#L466-L481)): **`nsga2`** ([docs](https://esa.github.io/pygmo2/algorithms.html#pygmo.nsga2)) with the matched settings: 100 individuals (92 with 3 objectives), pagmo's SBX η 15 at 0.9, polynomial mutation η 20 at 1 / n, `member_bfe`. It has no duplicate elimination.

**Bounds:** SBX clips; the polynomial mutation is Deb's bounded one.

**Keeping going:** one `evolve` call with the budget's generations. If these wouldn't fit in the 60 s cap, the adapter would run one generation per call; it never did.

**The front:** the non-dominated part of the final population, from `pg.fast_non_dominated_sorting`.

**Left out (rule 6.1):**
- `moead`, `moead_gen`: pagmo's MOEA/D is MOEA/D-DE, with DE's operators, so it can't use the matched SBX.
- `nspso`, `maco`: not matched algorithms.
- NSGA-III, SPEA2, SMS-EMOA: not in pagmo 2.19.1.

**Separate tests:**

| Scenario | Solver | Runs | Hypervolume: median (best, worst) | Evaluations | Front size: median | Capped |
|---|---|---|---|---|---|---|
| ZDT1 (25,000) | nsga2 | 5 | 0.8699 (0.8701, 0.8695) | 25,000 | 100 | 0 |
| ZDT2 (25,000) | nsga2 | 5 | 0.5363 (0.5365, 0.5361) | 25,000 | 100 | 0 |
| ZDT3 (25,000) | nsga2 | 5 | 1.3274 (1.3276, 1.3272) | 25,000 | 100 | 0 |
| DTLZ2 (25,000) | nsga2 | 5 | 0.6972 (0.7016, 0.6793) | 25,024 | 92 | 0 |
| DTLZ1 (40,000) | nsga2 | 5 | 0.1361 (0.1373, 0.1342) | 40,020 | 92 | 0 |

The fronts are the same as with one evaluation per call. The longest run took 2 s.

## Can't run

- OneMax 100 and 1000, matched: pagmo's only GA, [`sga`](https://esa.github.io/pygmo2/algorithms.html#pygmo.sga), has elitist reinsertion that can't be turned off ("the only reinsertion strategy provided is what we call pure elitism") and no two-point crossover (exponential, binomial, single-point or SBX).
- N-Queens 32 and 64: pygmo has no permutation representation or operators.

## Bugs found

None in the algorithms. Documentation: the docstrings of `sade`, `de1220`, `cmaes` and `xnes` describe `ftol` as "stopping criteria on the x tolerance" and `xtol` as "on the f tolerance", swapped; the code checks them the right way round.
