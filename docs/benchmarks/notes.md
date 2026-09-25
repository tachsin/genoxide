# Notes on the libraries

An index for readers of the [benchmarks](../../benchmarks/README.md): what each library can't run, the bugs found in the libraries, and the rule-level choices that affect many of them. Each library's page in [libraries/](libraries/) has the details: its methods and where its docs recommend them, the methods left out and why, its differences from the matched settings, its separate test runs and its bugs. Where the [rules](rules.md) refer to "the notes" for such details, they are on the library's page. [results.md](results.md) has the numbers and a table of which library ran which scenario.

## What each library can't run

A library missing from a chart can't run that scenario. The matched multi-objective scenarios run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA ([rule 6.1](rules.md#6-which-methods-run)). A library's other multi-objective algorithms aren't run, and its page lists them.

| Library | Doesn't run | Why |
|---|---|---|
| [genoxide](libraries/genoxide.md#cant-run) | nothing | |
| [genoxide (Python)](libraries/genoxide_python.md#cant-run) | nothing | |
| [genetic_algorithm](libraries/genetic_algorithm.md#cant-run) | the multi-objective scenarios | it optimizes one `isize` fitness |
| [radiate](libraries/radiate.md#cant-run) | SPEA2, MOEA/D, SMS-EMOA | its multi-objective support is its NSGA-II and NSGA-III selectors |
| [moors](libraries/moors.md#cant-run) | MOEA/D, SMS-EMOA | it doesn't have them |
| [openGA](libraries/openga.md#cant-run) | NSGA-II, SPEA2, MOEA/D, SMS-EMOA | NSGA-III is its only multi-objective algorithm |
| [pygmo](libraries/pygmo.md#cant-run) | N-Queens; NSGA-III, SPEA2, MOEA/D, SMS-EMOA | it has no permutation genome; pagmo 2.19.1 has no NSGA-III, SPEA2 or SMS-EMOA, and its MOEA/D is MOEA/D-DE, which can't use the matched SBX |
| [DEAP](libraries/deap.md#cant-run) | SPEA2, MOEA/D, SMS-EMOA | it has SPEA2's environmental selection but not the algorithm, and no MOEA/D or SMS-EMOA |
| [pymoo](libraries/pymoo.md#cant-run) | nothing | |
| [PyGAD](libraries/pygad.md#cant-run) | SPEA2, MOEA/D, SMS-EMOA | it has only NSGA-II and NSGA-III |
| [pycma](libraries/pycma.md#cant-run) | OneMax, N-Queens, the multi-objective scenarios | CMA-ES optimizes one objective of real numbers |
| [Nevergrad](libraries/nevergrad.md#cant-run) | OneMax matched, the multi-objective scenarios | it has no GA with the matched operators, and none of the five matched multi-objective algorithms |
| [SciPy](libraries/scipy.md#cant-run) | OneMax, N-Queens, the multi-objective scenarios | its optimizers minimize one objective of real numbers |
| [Jenetics](libraries/jenetics.md#cant-run) | NSGA-III, SPEA2, MOEA/D, SMS-EMOA | it doesn't have them; its NSGA-II is built from its engine and selectors |
| [jMetal](libraries/jmetal.md#cant-run) | nothing | |
| [Evolutionary.jl](libraries/evolutionary_jl.md#cant-run) | NSGA-III, SPEA2, MOEA/D, SMS-EMOA | NSGA-II is its only multi-objective algorithm |
| [Metaheuristics.jl](libraries/metaheuristics_jl.md#cant-run) | MOEA/D | its MOEA/D is MOEA/D-DE, which can't use the matched SBX |

genoxide, its Python package, DEAP, pymoo and PyGAD run NSGA-III only on DTLZ2 and DTLZ1, the problems with 3 objectives. The runs that failed, as opposed to not running, are in the charts: a cross for a target never reached, and a hatched bar for a hypervolume far below the others.

## Bugs found

A bug in a library isn't worked around: the benchmark runs what the library's users get, and its results show the bug ([rule 8.4](rules.md#8-reporting)). The exceptions say "worked around" in the effect column, and the library's page shows both results where the bug changes them. No bug was found in genoxide, its Python package, pycma or SciPy.

| Library | Bug | Effect on the results | Reported |
|---|---|---|---|
| [genetic_algorithm](libraries/genetic_algorithm.md#bugs-found) 0.27.3 | `call_repeatedly` with a seed repeats the same run in every repeat | none: the adapter restarts from a new seed itself | not yet |
| [radiate](libraries/radiate.md#bugs-found) 1.3.1 | `SimulatedBinaryCrossover` centres the child on (a − b) / 2 instead of (a + b) / 2, and writes only one parent | every multi-objective run, with the next bug: NSGA-II's DTLZ2 hypervolume is 0.080, against 0.581 with textbook operators; on ZDT the bugs pull the variables towards the optimum, and help | [pkalivas/radiate#28](https://github.com/pkalivas/radiate/issues/28) |
| [radiate](libraries/radiate.md#bugs-found) 1.3.1 | `PolynomialMutator` computes the new value from a bound instead of from the gene | as above | [pkalivas/radiate#28](https://github.com/pkalivas/radiate/issues/28) |
| [radiate](libraries/radiate.md#bugs-found) 1.3.1 | the guide recommends `ShuffleCrossover` for permutations, but its children aren't permutations | none: left out | not yet |
| [radiate](libraries/radiate.md#bugs-found) 1.3.1 | the guide says `GaussianMutator` makes "small" changes, but its standard deviation is a quarter of the gene's initial range | none: not run | not yet |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | AGE-MOEA panics when its first front is degenerate | none now: AGE-MOEA isn't run (rule 6.1) | [andresliszt/moo-rs#301](https://github.com/andresliszt/moo-rs/issues/301) |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | SPEA2's mating selection prefers the worst individuals | lower SPEA2 hypervolumes, with the next bug: ZDT1 0.689, against 0.863 with both fixed | not yet; a likely cause of [andresliszt/moo-rs#161](https://github.com/andresliszt/moo-rs/issues/161) |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | SPEA2's archive truncation keeps dominated individuals | as above; on ZDT1, 1 to 45 of the final archive's 100 individuals are dominated | not yet; as above |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | `ReveaBuilder` uses one iteration count as REVEA's t_max and as the end of the run, so its runs end before the budget (a design limit) | none now: REVEA isn't run (rule 6.1) | not yet |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | REVEA adapts its reference vectors only when a floating-point remainder is exactly 0 | none now: REVEA isn't run (rule 6.1) | not yet |
| [moors](libraries/moors.md#bugs-found) 0.2.11 | the README says `CloseDuplicatesCleaner` drops individuals within an ε-ball, but it compares ε with the squared distance | documentation only | not yet |
| [openGA](libraries/openga.md#bugs-found) f9b15e7 | the single-objective survival's rank roulette can't pick a child, and can pick the same individual again | children survive only through the elite slots, and the population collapses to copies: no idiomatic continuous run reaches its target. Worked around in the matched OneMax runs only, where every slot is elite | [Arash-codedev/openGA#30](https://github.com/Arash-codedev/openGA/issues/30) |
| [openGA](libraries/openga.md#bugs-found) | `solve_next_generation` stored `last_generation` only when it had fronts | none: fixed at the pinned commit f9b15e7 | [Arash-codedev/openGA#23](https://github.com/Arash-codedev/openGA/issues/23) |
| [pygmo](libraries/pygmo.md#bugs-found) 2.19.8 | the docstrings of `sade`, `de1220`, `cmaes` and `xnes` swap the descriptions of `ftol` and `xtol` | documentation only | not yet |
| [DEAP](libraries/deap.md#bugs-found) 1.4.4 | the BIPOP-CMA-ES example takes the worst sample as the best, so its EqualFunVals and Stagnation criteria follow the worst values | only when a CMA-ES run restarts; every test run reached the target | not yet |
| [DEAP](libraries/deap.md#bugs-found) 1.4.4 | the same example's EqualFunVals counts the equal generations since the start of the run, not in the last N | as above | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | CMA-ES with seed 0 isn't repeatable: pycma reads 0 as "seed from the clock" | none: worked around, the adapter passes seed + 1 | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | pattern search draws its coordinate order from an unseeded generator | pattern search is left out (rule 5.2) | fixed on main in [54e13ec](https://github.com/anyoptimization/pymoo/commit/54e13ecd82e69880fc758561290618a8eb0998f8), after [anyoptimization/pymoo#794](https://github.com/anyoptimization/pymoo/issues/794); not released |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | the DE example's `dither="vector"` is silently ignored | documentation only | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | the binary and permutation pages name other crossovers in their text than in their code | documentation only; the adapter follows the code | not yet |
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | `sbx` always makes the child below the parents' midpoint, which pulls every gene towards the lower bound | the idiomatic continuous runs: Ackley 30 and Rosenbrock 10 fail, and Rastrigin needs 5 to 11 times the evaluations. Worked around in the matched multi-objective runs only, where the adapter gives PyGAD a matched SBX | [ahmedfgad/GeneticAlgorithmPython#369](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/369) |
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | with `allow_duplicate_genes=False`, as in its permutation example, the random mutation never changes a permutation | the N-Queens runs converge to one board and reach the time cap after about 200 evaluations | not yet |
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | the docs say `swap_mutation` swaps 2 random genes; it swaps a gene with the one half the length after it | none: not used | not yet |
| [Nevergrad](libraries/nevergrad.md#bugs-found) 1.0.12 | the metamodel that `NgIohTuned` uses crashes with NumPy 2.5 (`float()` of an array) | worked around: the adapter restores the old conversion in that one module; the algorithm is unchanged | not yet |
| [Jenetics](libraries/jenetics.md#bugs-found) 9.1.0 | `SimulatedBinaryCrossover` centres the child on (a − b) / 2 | NSGA-II's DTLZ1 hypervolume is 0 and its DTLZ2 0.586; with a corrected copy, 1.296 and 0.676 | [jenetics/jenetics#969](https://github.com/jenetics/jenetics/issues/969) |
| [Jenetics](libraries/jenetics.md#bugs-found) 9.1.0 | with a minimizing engine and `Vec.of(...)`, as in the manual's DTLZ1 example, the crowding distance is 0 except at the extremes | worked around: the objectives are minimized through `VecFactory`, another documented way | not yet |
| [Jenetics](libraries/jenetics.md#bugs-found) 9.1.0 | `UFTournamentSelector` draws the same fixed pairs in every round when it samples the whole population | small: with the survivors shuffled, DTLZ2 0.552 against 0.586 | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES's step size grows without bound once it has converged | no CMA-ES run reaches a target | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | `CMAESUtils.tql2` throws `ArrayIndexOutOfBoundsException` when the covariance matrix holds NaN | ends the attempt, often after about 200,000 evaluations; the adapter restarts it (rule 2.2) | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES draws its samples from a generator seeded with the clock | worked around: the adapter seeds it; the algorithm is unchanged | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES computes χₙ with integer divisions, 2.5% too large for n = 10 | σ shrinks a little faster than intended | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | the initial permutations ignore `JMetalRandom`'s seed | worked around: the adapter draws the same uniform permutation with `JMetalRandom` | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | coral reef optimization seeds its generators with the clock, one of them inside a method | coral reef optimization is left out (rule 5.2) | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.0 | `NSGA2` reorders the parents but not their objective values, ranks and crowding distances | hypervolumes of 0 on ZDT1, ZDT2, ZDT3 and DTLZ1 | [SciML/Evolutionary.jl#174](https://github.com/SciML/Evolutionary.jl/issues/174) |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.0 | CMA-ES doesn't converge in 30 dimensions | CMA-ES reaches no multimodal target | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.0 | a GA offspring that isn't crossed is its parent, not a copy, and is mutated in place | the idiomatic GAs run as users get them; worked around in the matched GA, with a wrapper that copies | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.0 | the (μ+λ)-ES can overwrite a surviving parent | small: 3 of 40,000 generations in a test | not yet |
| [Metaheuristics.jl](libraries/metaheuristics_jl.md#bugs-found) 3.5.0 | the bounded SBX computes the second child's spread from the lower bound | not measured; every multi-objective run uses it | not yet |
| [Metaheuristics.jl](libraries/metaheuristics_jl.md#bugs-found) 3.5.0 | the `DE` docstring gives F = 1.0 as the default; the code's is 0.7 | documentation only | not yet |

## Rule-level choices

These apply to many libraries and explain many of the results. The [rules](rules.md) have the full text.

- **Restarts on convergence** ([rule 2.2](rules.md#2-the-budget)). A limit that's only a budget, such as a number of generations, is lifted. A convergence criterion that is part of the method, or of the docs' example for the problem type, ends an attempt. The method then starts again, with the library's restart mechanism if it has one, otherwise from a new random start. Some examples' criteria end attempts early and often, e.g. Jenetics' `bySteadyFitness(7)`, Evolutionary.jl's DE and pymoo's permutation GA; their pages show the results without them. The matched scenarios have no convergence criterion.
- **Only inside the bounds** ([rule 2.4](rules.md#2-the-budget)). Every evaluated solution lies inside the box, through the library's own bound handling. So pygmo's CMA-ES and xNES run with `force_bounds`, which pagmo warns worsens them. DEAP's CMA-ES and DE use its closest-valid penalty. SciPy's `basinhopping` is left out.
- **Evaluations are counted by the adapter** ([rule 3](rules.md#3-counting-evaluations)). A library that doesn't evaluate an unchanged copy again saves that evaluation: genoxide, DEAP, PyGAD and Jenetics. moors evaluates its parents again every generation, and pays for it.
- **Seeds must repeat a run** ([rule 5.2](rules.md#5-seeds-and-repeated-runs)). A method that can't be seeded is left out: jMetal's coral reef optimization and pymoo's pattern search.
- **Solvers stopped early** ([rule 5.3](rules.md#5-seeds-and-repeated-runs)). A solver whose first 3 seeds all reach the 60 s cap without the target runs no more seeds.
- **The docs decide** ([rule 6.2](rules.md#6-which-methods-run)). The idiomatic methods and settings are the ones a library's docs prefer, else their example for the problem type, else the defaults. The separate test runs never choose. Some recommended settings do badly here, and their pages say why.
- **At most 3 methods per library and problem type** ([rule 6.4](rules.md#6-which-methods-run)): the docs' first recommendations.
- **The front is the final population** ([rule 7.2](rules.md#7-multi-objective-runs)), not an archive of every solution evaluated.
