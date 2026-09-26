# Notes on the libraries

An index for readers of the [benchmarks](../../benchmarks/README.md): what each library can't run, the bugs found, and the rule-level choices that affect many libraries. The details are on each library's page in [libraries/](libraries/). Where the [rules](rules.md) refer to "the notes", the details are there too. [results.md](results.md) has the numbers.

## What each library can't run

A library missing from a chart can't run that scenario. The matched multi-objective scenarios run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA, with the library's own SBX and polynomial mutation ([rule 6.1](rules.md#6-which-methods-run)). A library's other multi-objective algorithms aren't run; its page lists them. NSGA-III runs on every multi-objective problem, ZDT included, in every library that has it.

| Library | Doesn't run | Why |
|---|---|---|
| [genoxide](libraries/genoxide.md#cant-run) | nothing | |
| [genoxide (Python)](libraries/genoxide_python.md#cant-run) | nothing | |
| [genetic_algorithm](libraries/genetic_algorithm.md#cant-run) | matched OneMax; the multi-objective scenarios | no generational replacement without elitism; it optimizes one `isize` fitness |
| [radiate](libraries/radiate.md#cant-run) | SPEA2, MOEA/D, SMS-EMOA | its multi-objective algorithms are NSGA-II and NSGA-III |
| [moors](libraries/moors.md#cant-run) | matched OneMax; the multi-objective scenarios | no tournament of 3 and no generational replacement; no polynomial mutation (and no MOEA/D or SMS-EMOA) |
| [openGA](libraries/openga.md#cant-run) | matched OneMax; the multi-objective scenarios | no tournament selection, two-point crossover, bit flip or generational replacement; no SBX or polynomial mutation for its NSGA-III, its only multi-objective algorithm |
| [pygmo](libraries/pygmo.md#cant-run) | matched OneMax; N-Queens; NSGA-III, SPEA2, MOEA/D, SMS-EMOA | `sga`'s reinsertion is always elitist, and it has no two-point crossover; no permutation genome; pagmo 2.19.1 has no NSGA-III, SPEA2 or SMS-EMOA, and its MOEA/D is MOEA/D-DE, which can't use the matched SBX |
| [DEAP](libraries/deap.md#cant-run) | SPEA2, MOEA/D, SMS-EMOA | it has SPEA2's environmental selection but not the algorithm, and no MOEA/D or SMS-EMOA |
| [pymoo](libraries/pymoo.md#cant-run) | matched OneMax | no generational survival: its GA keeps the best of parents and children |
| [PyGAD](libraries/pygad.md#cant-run) | SPEA2, MOEA/D, SMS-EMOA | its multi-objective algorithms are NSGA-II and NSGA-III |
| [pycma](libraries/pycma.md#cant-run) | OneMax, N-Queens, the multi-objective scenarios | CMA-ES optimizes one objective of real numbers |
| [Nevergrad](libraries/nevergrad.md#cant-run) | matched OneMax; the multi-objective scenarios | no GA with the matched operators; none of the five matched multi-objective algorithms |
| [SciPy](libraries/scipy.md#cant-run) | OneMax, N-Queens, the multi-objective scenarios | its optimizers minimize one objective of real numbers |
| [Jenetics](libraries/jenetics.md#cant-run) | the multi-objective scenarios | no polynomial mutation, and no NSGA-III, SPEA2, MOEA/D or SMS-EMOA |
| [jMetal](libraries/jmetal.md#cant-run) | matched OneMax | no generational replacement without elitism and no two-point crossover for bits |
| [Evolutionary.jl](libraries/evolutionary_jl.md#cant-run) | NSGA-III, SPEA2, MOEA/D, SMS-EMOA | NSGA-II is its only multi-objective algorithm |
| [Metaheuristics.jl](libraries/metaheuristics_jl.md#cant-run) | matched OneMax; MOEA/D | no two-point crossover; its MOEA/D is MOEA/D-DE, which can't use the matched SBX |

Runs that ran but didn't succeed are in the charts: a cross for a target never reached, and a hatched bar for a hypervolume far below the others.

## Bugs found

The benchmark runs what a library's users get, so its results show its bugs ([rule 8.4](rules.md#8-reporting)). The exceptions say "worked around" in the effect column, and the library's page shows both results where the bug changes them. No bug was found in genoxide, its Python package, pycma or SciPy.

| Library | Bug | Effect on the results | Reported |
|---|---|---|---|
| [genetic_algorithm](libraries/genetic_algorithm.md#bugs-found) 0.27.3 | `call_repeatedly` with a seed repeats the same run in every repeat | none: the adapter restarts from a new seed itself | not yet |
| [radiate](libraries/radiate.md#bugs-found) 1.3.1 | `SimulatedBinaryCrossover` centres the child on (a − b) / 2 instead of (a + b) / 2, and writes only one parent | every multi-objective run, with the next bug: NSGA-II's DTLZ2 hypervolume is 0.080, against 0.581 with textbook operators; on ZDT the two bugs pull the variables towards the optimum | [pkalivas/radiate#28](https://github.com/pkalivas/radiate/issues/28) |
| [radiate](libraries/radiate.md#bugs-found) 1.3.1 | `PolynomialMutator` computes the new value from a bound instead of from the gene | as above | [pkalivas/radiate#28](https://github.com/pkalivas/radiate/issues/28) |
| [radiate](libraries/radiate.md#bugs-found) 1.3.1 | the guide recommends `ShuffleCrossover` for permutations, but its children aren't permutations | none: left out | not yet |
| [radiate](libraries/radiate.md#bugs-found) 1.3.1 | the guide says `GaussianMutator` makes "small" changes, but its standard deviation is a quarter of the gene's initial range | none: not run | not yet |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | AGE-MOEA panics when its first front is degenerate | none: AGE-MOEA isn't run (rule 6.1) | [andresliszt/moo-rs#301](https://github.com/andresliszt/moo-rs/issues/301) |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | SPEA2's mating selection prefers the worst individuals | none: the multi-objective scenarios don't run. In a diagnostic, with the next bug: ZDT1 0.689, against 0.863 with both fixed | not yet; a likely cause of [andresliszt/moo-rs#161](https://github.com/andresliszt/moo-rs/issues/161) |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | SPEA2's archive truncation keeps dominated individuals | as above; on ZDT1, 1 to 45 of the final archive's 100 individuals were dominated | not yet; as above |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | `ReveaBuilder` uses one iteration count as REVEA's t_max and as the end of the run, so its runs end before the budget (a design limit) | none: REVEA isn't run (rule 6.1) | not yet |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | REVEA adapts its reference vectors only when a floating-point remainder is exactly 0 | none: REVEA isn't run (rule 6.1) | not yet |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | the README says `CloseDuplicatesCleaner` drops individuals within an ε-ball, but it compares ε with the squared distance | documentation only | not yet |
| [openGA](libraries/openga.md#bugs-found) f9b15e7 | the single-objective survival's rank roulette can't pick a child, and can pick the same individual again | children survive only through the elite slots, and the population collapses to copies: no idiomatic continuous run reaches its target | [Arash-codedev/openGA#30](https://github.com/Arash-codedev/openGA/issues/30) |
| [openGA](libraries/openga.md#bugs-found) | `solve_next_generation` stored `last_generation` only when it had fronts | none: fixed at the pinned commit f9b15e7 | [Arash-codedev/openGA#23](https://github.com/Arash-codedev/openGA/issues/23) |
| [pygmo](libraries/pygmo.md#bugs-found) 2.19.8 | the docstrings of `sade`, `de1220`, `cmaes` and `xnes` swap the descriptions of `ftol` and `xtol` | documentation only | not yet |
| [DEAP](libraries/deap.md#bugs-found) 1.4.4 | the DE example's exponential crossover stops copying when the random number is below CR, the inverse of Storn and Price's | its CR of 0.8 acts as 0.2: the DE runs change about one gene per child | not yet |
| [DEAP](libraries/deap.md#bugs-found) 1.4.4 | the BIPOP-CMA-ES example takes the worst sample as the best, so its EqualFunVals and Stagnation criteria follow the worst values | only when a CMA-ES run restarts | not yet |
| [DEAP](libraries/deap.md#bugs-found) 1.4.4 | the same example's EqualFunVals counts the equal generations since the start of the run, not in the last N | as above | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | `SPEA2`'s default survival is one object shared by every SPEA2 of a process, so a run depends on the runs before it | worked around: the adapter gives each run a new `SPEA2Survival`; the page shows both results | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | `TournamentSelection(pressure=3)` draws 3 competitors, but the GA's comparison compares only the first 2 | none: no run uses a tournament of more than 2 | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | CMA-ES with seed 0 isn't repeatable: pycma reads 0 as "seed from the clock" | worked around: the adapter never passes 0 | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | pattern search draws its coordinate order from an unseeded generator | pattern search is left out (rule 5.2) | fixed on main in [54e13ec](https://github.com/anyoptimization/pymoo/commit/54e13ecd82e69880fc758561290618a8eb0998f8), after [anyoptimization/pymoo#794](https://github.com/anyoptimization/pymoo/issues/794); not released |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | the DE example's `dither="vector"` is silently ignored | documentation only | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | the binary and permutation pages name other crossovers in their text than in their code | documentation only; the adapter follows the code | not yet |
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | `sbx` always makes the child below the parents' midpoint, which pulls every crossed gene towards the lower bound | the continuous runs: Ackley 30 and Rosenbrock 10 don't reach the target; the multi-objective runs: the fronts crowd towards f1 = 0 on ZDT and stay far from DTLZ's | [ahmedfgad/GeneticAlgorithmPython#369](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/369) |
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | `two_points_crossover` draws only the first point; the second is always n / 2 after it | the matched OneMax runs | not yet |
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | with `allow_duplicate_genes=False`, as in its permutation example, the random mutation never changes a permutation | the N-Queens runs converge to one board and reach the time cap after about 200 evaluations | not yet |
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | the docs say `swap_mutation` swaps 2 random genes; it swaps a gene with the one half the length after it | none: not used | not yet |
| [Nevergrad](libraries/nevergrad.md#bugs-found) 1.0.12 | the metamodel that `NgIohTuned` uses crashes with NumPy 2.5 (`float()` of an array) | worked around: the adapter restores the old conversion in that module; without it, every `ngiohtuned` continuous run crashes at the metamodel's first fit; the page shows both results | not yet |
| [Jenetics](libraries/jenetics.md#bugs-found) 9.1.0 | `SimulatedBinaryCrossover` centres the child on (a − b) / 2 | none: the multi-objective scenarios don't run | [jenetics/jenetics#969](https://github.com/jenetics/jenetics/issues/969) |
| [Jenetics](libraries/jenetics.md#bugs-found) 9.1.0 | with a minimizing engine and `Vec.of(...)`, as in the manual's DTLZ1 example, the crowding distance is 0 except at the extremes | none: the multi-objective scenarios don't run | not yet |
| [Jenetics](libraries/jenetics.md#bugs-found) 9.1.0 | `UFTournamentSelector` draws the same fixed pairs in every round when it samples the whole population | none: the multi-objective scenarios don't run | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES's step size grows without bound once it has converged | no CMA-ES run reaches a target | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | `CMAESUtils.tql2` throws `ArrayIndexOutOfBoundsException` when the covariance matrix holds NaN | worked around, labelled: the adapter catches the crash and starts a new attempt; without it, the run ends, often after about 200,000 evaluations; the page shows both results | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES draws its samples from a generator seeded with the clock | worked around: the adapter seeds it; the algorithm is unchanged | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES computes χₙ with integer divisions, 2.5% too large for n = 10 | σ shrinks a little faster than intended | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | the initial permutations ignore `JMetalRandom`'s seed | worked around: the adapter draws the same uniform permutation with `JMetalRandom` | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | coral reef optimization seeds its generators with the clock, one of them inside a method | coral reef optimization is left out (rule 5.2) | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.0 | `NSGA2` reorders the parents but not their objective values, ranks and crowding distances | hypervolumes of 0 on ZDT1, ZDT2, ZDT3 and DTLZ1 in the separate tests | [SciML/Evolutionary.jl#174](https://github.com/SciML/Evolutionary.jl/issues/174) |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.0 | CMA-ES doesn't converge in 30 dimensions | CMA-ES reaches no multimodal target | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.0 | a GA offspring that isn't crossed is its parent, not a copy, and is mutated in place | every GA and NSGA-II run, matched and idiomatic | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.0 | the (μ+λ)-ES can overwrite a surviving parent | small: 3 of 40,000 generations in a test | not yet |
| [Metaheuristics.jl](libraries/metaheuristics_jl.md#bugs-found) 3.5.0 | the bounded SBX computes the second child's spread from the lower bound | not measured; every multi-objective run uses it | not yet |
| [Metaheuristics.jl](libraries/metaheuristics_jl.md#bugs-found) 3.5.0 | the `DE` docstring gives F = 1.0 as the default; the code's is 0.7 | documentation only | not yet |

## Rule-level choices

These apply to many libraries. The [rules](rules.md) have the full text.

- **Restarts on convergence** ([rule 2.2](rules.md#2-the-budget)). A limit that's only a budget, such as a number of generations, is lifted. A convergence criterion that's part of the method, or of the docs' example for the problem type, ends an attempt. The method then starts again, with the library's restart mechanism, else from a new random start. Some examples' criteria end attempts early and often: Jenetics' `bySteadyFitness(7)`, Evolutionary.jl's DE, pymoo's permutation GA. The matched scenarios have no convergence criterion.
- **Only inside the bounds** ([rule 2.4](rules.md#2-the-budget)). Every evaluated solution lies inside the box, through the library's own bound handling. So pygmo's CMA-ES and xNES run with `force_bounds`, which pagmo warns worsens them. DEAP's CMA-ES and DE use `ClosestValidPenalty`, as DEAP's box-bounded ES example does. SciPy's `basinhopping` is left out.
- **Evaluations are counted by the adapter** ([rule 3](rules.md#3-counting-evaluations)). A library that doesn't evaluate an unchanged copy again saves that evaluation: genoxide, DEAP, PyGAD and Jenetics. moors evaluates its parents again every generation, and those evaluations count.
- **Seeds must repeat a run** ([rule 5.2](rules.md#5-seeds-and-repeated-runs)). A method that can't be seeded is left out: jMetal's coral reef optimization and pymoo's pattern search.
- **Solvers stopped early** ([rule 5.3](rules.md#5-seeds-and-repeated-runs)). A solver whose first 3 seeds all reach the 60 s cap without the target runs no more seeds.
- **The docs decide** ([rule 6.2](rules.md#6-which-methods-run)). The idiomatic methods and settings are the ones a library's docs prefer, else their example for the problem type, else the defaults. The separate test runs never choose. Where a recommended setting does poorly here, the page says why.
- **Matched means the library's own components** ([rule 6.1](rules.md#6-which-methods-run)). A library missing one doesn't run the scenario. It gets no component written by the adapter.
- **At most 3 methods per library and problem type** ([rule 6.4](rules.md#6-which-methods-run)): the docs' first recommendations.
- **The front is the final population** ([rule 7.2](rules.md#7-multi-objective-runs)), not an archive of every solution evaluated.
