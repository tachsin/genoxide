# Notes on the libraries

An index for readers of the [benchmarks](../../benchmarks/README.md): what each library can't run, the bugs found, and the rule-level choices that affect many libraries. The details are on each library's page in [libraries/](libraries/). Where the [rules](rules.md) refer to "the notes", the details are there too. [results.md](results.md) has the numbers.

## What each library can't run

A library missing from a chart can't run that scenario. The benchmarks have single-objective scenarios only, for now: the multi-objective ones come back once genoxide solves these well.

| Library | Doesn't run | Why |
|---|---|---|
| [genoxide](libraries/genoxide.md#cant-run) | nothing | |
| [genoxide (Python)](libraries/genoxide_python.md#cant-run) | nothing | |
| [genetic_algorithm](libraries/genetic_algorithm.md#cant-run) | matched OneMax | no generational replacement without elitism |
| [radiate](libraries/radiate.md#cant-run) | nothing | |
| [moors](libraries/moors.md#cant-run) | matched OneMax | no tournament of 3 and no generational replacement |
| [openGA](libraries/openga.md#cant-run) | matched OneMax | no tournament selection, two-point crossover, bit flip or generational replacement |
| [pygmo](libraries/pygmo.md#cant-run) | matched OneMax; N-Queens | `sga`'s reinsertion is always elitist, and it has no two-point crossover; no permutation genome |
| [DEAP](libraries/deap.md#cant-run) | nothing | |
| [pymoo](libraries/pymoo.md#cant-run) | matched OneMax | no generational survival: its GA keeps the best of parents and children |
| [PyGAD](libraries/pygad.md#cant-run) | nothing | |
| [pycma](libraries/pycma.md#cant-run) | OneMax, N-Queens | CMA-ES optimizes real numbers |
| [Nevergrad](libraries/nevergrad.md#cant-run) | matched OneMax | no GA with the matched operators |
| [SciPy](libraries/scipy.md#cant-run) | OneMax, N-Queens | its optimizers minimize functions of real numbers |
| [Jenetics](libraries/jenetics.md#cant-run) | matched OneMax | no bit-flip mutation among its own components |
| [jMetal](libraries/jmetal.md#cant-run) | matched OneMax | no generational replacement without elitism and no two-point crossover for bits |
| [Evolutionary.jl](libraries/evolutionary_jl.md#cant-run) | nothing | |
| [Metaheuristics.jl](libraries/metaheuristics_jl.md#cant-run) | matched OneMax | no two-point crossover |

Runs that ran but didn't reach the target are in the charts: a cross for a target never reached.

## Bugs found

The benchmark runs what a library's users get, so its results show its bugs ([rule 8.4](rules.md#8-reporting)). The exceptions say "worked around" in the effect column, and the library's page shows both results where the bug changes them. No bug was found in genoxide, its Python package, pycma or SciPy.

The bugs found in the multi-objective algorithms and operators (radiate's SBX and polynomial mutation, [pkalivas/radiate#28](https://github.com/pkalivas/radiate/issues/28); moors' AGE-MOEA, [andresliszt/moo-rs#301](https://github.com/andresliszt/moo-rs/issues/301), SPEA2 and REVEA; Jenetics' SBX, [jenetics/jenetics#969](https://github.com/jenetics/jenetics/issues/969), crowding distance and `UFTournamentSelector`; Evolutionary.jl's `NSGA2`, [SciML/Evolutionary.jl#174](https://github.com/SciML/Evolutionary.jl/issues/174); Metaheuristics.jl's bounded SBX; pymoo's shared SPEA2 survival) left the tables with the multi-objective scenarios. They come back with them.

| Library | Bug | Effect on the results | Reported |
|---|---|---|---|
| [genetic_algorithm](libraries/genetic_algorithm.md#bugs-found) 0.27.3 | `call_repeatedly` with a seed repeats the same run in every repeat | none: the adapter restarts from a new seed itself | not yet |
| [radiate](libraries/radiate.md#bugs-found) 1.3.1 | the guide recommends `ShuffleCrossover` for permutations, but its children aren't permutations | none: left out | not yet |
| [radiate](libraries/radiate.md#bugs-found) 1.3.1 | the guide says `GaussianMutator` makes "small" changes, but its standard deviation is a quarter of the gene's initial range | none: not run | not yet |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | the README says `CloseDuplicatesCleaner` drops individuals within an ε-ball, but it compares ε with the squared distance | documentation only | not yet |
| [openGA](libraries/openga.md#bugs-found) f9b15e7 | the single-objective survival's rank roulette can't pick a child, and can pick the same individual again | children survive only through the elite slots, and the population collapses to copies: no idiomatic continuous run reaches its target | [Arash-codedev/openGA#30](https://github.com/Arash-codedev/openGA/issues/30) |
| [openGA](libraries/openga.md#bugs-found) | `solve_next_generation` stored `last_generation` only when it had fronts | none: fixed at the pinned commit f9b15e7 | [Arash-codedev/openGA#23](https://github.com/Arash-codedev/openGA/issues/23) |
| [pygmo](libraries/pygmo.md#bugs-found) 2.19.8 | the docstrings of `sade`, `de1220`, `cmaes` and `xnes` swap the descriptions of `ftol` and `xtol` | documentation only | not yet |
| [DEAP](libraries/deap.md#bugs-found) 1.4.4 | the DE example's exponential crossover stops copying when the random number is below CR, the inverse of Storn and Price's | its CR of 0.8 acts as 0.2: the DE runs change about one gene per child | not yet |
| [DEAP](libraries/deap.md#bugs-found) 1.4.4 | the BIPOP-CMA-ES example takes the worst sample as the best, so its EqualFunVals and Stagnation criteria follow the worst values | only when a CMA-ES run restarts | not yet |
| [DEAP](libraries/deap.md#bugs-found) 1.4.4 | the same example's EqualFunVals counts the equal generations since the start of the run, not in the last N | as above | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | `TournamentSelection(pressure=3)` draws 3 competitors, but the GA's comparison compares only the first 2 | none: no run uses a tournament of more than 2 | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | CMA-ES with seed 0 isn't repeatable: pycma reads 0 as "seed from the clock" | worked around: the adapter never passes 0 | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | pattern search draws its coordinate order from an unseeded generator | pattern search is left out (rule 5.2) | fixed on main in [54e13ec](https://github.com/anyoptimization/pymoo/commit/54e13ecd82e69880fc758561290618a8eb0998f8), after [anyoptimization/pymoo#794](https://github.com/anyoptimization/pymoo/issues/794); not released |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | the DE example's `dither="vector"` is silently ignored | documentation only | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | the binary and permutation pages name other crossovers in their text than in their code | documentation only; the adapter follows the code | not yet |
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | `sbx` always makes the child below the parents' midpoint, which pulls every crossed gene towards the lower bound | the continuous runs: Ackley 30 and Rosenbrock 10 don't reach the target | [ahmedfgad/GeneticAlgorithmPython#369](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/369) |
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | `two_points_crossover` draws only the first point; the second is always n / 2 after it | the matched OneMax runs | not yet |
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | with `allow_duplicate_genes=False`, as in its permutation example, the random mutation never changes a permutation | the N-Queens runs converge to one board and reach the time cap after about 200 evaluations | not yet |
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | the docs say `swap_mutation` swaps 2 random genes; it swaps a gene with the one half the length after it | none: not used | not yet |
| [Nevergrad](libraries/nevergrad.md#bugs-found) 1.0.12 | the metamodel that `NgIohTuned` uses crashes with NumPy 2.5 (`float()` of an array) | worked around: the adapter restores the old conversion in that module; without it, every `ngiohtuned` continuous run crashes at the metamodel's first fit; the page shows both results | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES's step size grows without bound once it has converged | no CMA-ES run reaches a target | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | `CMAESUtils.tql2` throws `ArrayIndexOutOfBoundsException` when the covariance matrix holds NaN | worked around, labelled: the adapter catches the crash and starts a new attempt; without it, the run ends, often after about 200,000 evaluations; the page shows both results | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES draws its samples from a generator seeded with the clock | worked around: the adapter seeds it; the algorithm is unchanged | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES computes χₙ with integer divisions, 2.5% too large for n = 10 | σ shrinks a little faster than intended | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | the initial permutations ignore `JMetalRandom`'s seed | worked around: the adapter draws the same uniform permutation with `JMetalRandom` | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | coral reef optimization seeds its generators with the clock, one of them inside a method | coral reef optimization is left out (rule 5.2) | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.0 | CMA-ES doesn't converge in 30 dimensions | CMA-ES reaches no multimodal target | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.0 | a GA offspring that isn't crossed is its parent, not a copy, and is mutated in place | every GA run, matched and idiomatic | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.0 | the (μ+λ)-ES can overwrite a surviving parent | small: 3 of 40,000 generations in a test | not yet |
| [Metaheuristics.jl](libraries/metaheuristics_jl.md#bugs-found) 3.5.0 | the `DE` docstring gives F = 1.0 as the default; the code's is 0.7 | documentation only | not yet |

## Rule-level choices

These apply to many libraries. The [rules](rules.md) have the full text.

- **Restarts on convergence** ([rule 2.2](rules.md#2-the-budget)). A limit that's only a budget, such as a number of generations, is lifted. A convergence criterion that's part of the method, or of the docs' example for the problem type, ends an attempt. The method then starts again, with the library's restart mechanism, else from a new random start. Some examples' criteria end attempts early and often: Jenetics' `bySteadyFitness(7)`, Evolutionary.jl's DE, pymoo's permutation GA. The matched scenarios have no convergence criterion. An attempt that makes no new evaluation for 10 generations has stalled, and restarts too, in every scenario.
- **Only inside the bounds** ([rule 2.4](rules.md#2-the-budget)). Every evaluated solution lies inside the box, through the library's own bound handling. So pygmo's CMA-ES and xNES run with `force_bounds`, which pagmo warns worsens them. DEAP's CMA-ES and DE use `ClosestValidPenalty`, as DEAP's box-bounded ES example does. SciPy's `basinhopping` is left out.
- **Evaluations are counted by the adapter** ([rule 3](rules.md#3-counting-evaluations)). A library that doesn't evaluate an unchanged copy again saves that evaluation: genoxide, DEAP, PyGAD and Jenetics. moors evaluates its parents again every generation, and those evaluations count.
- **Seeds must repeat a run** ([rule 5.2](rules.md#5-seeds-and-repeated-runs)). A method that can't be seeded is left out: jMetal's coral reef optimization and pymoo's pattern search.
- **Solvers stopped early** ([rule 5.3](rules.md#5-seeds-and-repeated-runs)). A solver whose first 3 seeds all reach the time cap without the target runs no more seeds.
- **The docs decide** ([rule 6.2](rules.md#6-which-methods-run)). The idiomatic methods and settings are the ones a library's docs prefer, else their example for the problem type, else the defaults. The separate test runs never choose. Where a recommended setting does poorly here, the page says why.
- **Matched means the library's own components** ([rule 6.1](rules.md#6-which-methods-run)). A library missing one doesn't run the scenario. It gets no component written by the adapter.
- **At most 3 methods per library and problem type** ([rule 6.4](rules.md#6-which-methods-run)): the docs' first recommendations.
