# Notes on the libraries

An index for readers of the [benchmarks](../../benchmarks/README.md): which library runs which problem, how each differs from the methods' [definitions](rules.md#6-the-methods), what each can't run, and the bugs found. The details, with the sources, are on each library's page in [libraries/](libraries/). [results.md](results.md) has the numbers.

## Who runs what

The suite is matched: three problems, one method each. A library runs a problem only with its own implementation of that problem's method ([rule 6.1](rules.md#6-the-methods)).

| Library | OneMax 1000: GA | Rastrigin 30: DE/rand/1/bin | Rosenbrock 10: CMA-ES |
|---|---|---|---|
| [genoxide](libraries/genoxide.md) | ✓ | ✓ | ✓ |
| [genoxide (Python)](libraries/genoxide_python.md) | ✓ | ✓ | ✓ |
| [DEAP](libraries/deap.md) | ✓ | – no DE in the library, only in its examples | ✓ |
| [PyGAD](libraries/pygad.md) | ✓ | – only a GA | – only a GA |
| [radiate](libraries/radiate.md) | ✓ | – only a GA | – only a GA |
| [pycma](libraries/pycma.md) | – CMA-ES only | – CMA-ES only | ✓ (the reference) |
| [SciPy](libraries/scipy.md) | – no GA | ✓ | – no CMA-ES |
| [pygmo](libraries/pygmo.md) | – `sga`'s reinsertion is always elitist, and it has no two-point crossover | ✓ | ✓ |
| [pymoo](libraries/pymoo.md) | – its GA keeps the best of parents and children: no generational survival | ✓ | ✓ |
| [jMetal](libraries/jmetal.md) | – no generational replacement without elitism, no two-point crossover for bits | ✓ | ✓ |
| [Evolutionary.jl](libraries/evolutionary_jl.md) | ✓ | – its DE crosses the mutant with the base vector, not the target, and `BINX(CR)` keeps the base's gene with probability CR, with no forced gene | ✓ |
| [Metaheuristics.jl](libraries/metaheuristics_jl.md) | – no two-point crossover | ✓ | – no CMA-ES in 3.5.0 |

**Left the suite**, with none of the three methods as the definitions have them: genetic_algorithm (no generational replacement without elitism, no DE or CMA-ES), moors (no tournament of 3 or generational survival, no DE or CMA-ES), openGA (no tournament selection, two-point crossover or bit flip), Jenetics (no bit-flip mutation among its components, no DE or CMA-ES), and Nevergrad (no GA; its DE is current-to-best with its indices drawn with replacement, and steady-state; its CMA-ES starts a new pycma run by itself when pycma's `noeffectaxis` or `noeffectcoord` fires, which can't be turned off, and so has restarts). They come back with problems whose methods they have ([ROADMAP.md](../../ROADMAP.md#benchmarks)).

## How each differs from the definitions

Differences that don't change the algorithm ([rule 6.1](rules.md#6-the-methods)); each library's page has the sources.

**OneMax 1000, the GA** (DEAP's `eaSimple` is the reference):
- **genoxide, genoxide (Python):** a child identical to its parent isn't evaluated again.
- **PyGAD:** a parent is crossed with probability 0.5 with the next eligible one, one child per crossover; each gene flips with probability 0.2 / n, the same mean of 0.2 bits per child; its two-point crossover always takes n / 2 genes (a bug, below).
- **radiate:** each child is crossed with probability 0.5 with a random other child; each bit flips with probability 0.2 / n; cut points from 0..n; only changed children are evaluated.
- **Evolutionary.jl:** `flip` flips exactly one bit of a mutated child, the same mean; the tournament draws its contestants from a shuffled permutation; every child is evaluated; an uncrossed pair passes the parents themselves (a bug, below).

**Rastrigin 30, DE/rand/1/bin** (no target: a fixed budget of 300,000 evaluations; the errors at the end should agree across libraries within the seeds' spread):
- **Bounds** (the reference redraws a gene outside the box uniformly in it, as SciPy and pygmo do): genoxide sets it halfway between the target's gene and the bound; jMetal clips it; pymoo redraws it between the bound and the base's gene; Metaheuristics.jl puts it between the bound and the best solution's gene.
- **The indices:** pygmo and Metaheuristics.jl don't exclude the target from r1, r2 and r3, so about 3% of trials use it; pymoo draws each index column as a permutation of the population.
- **Replacement:** jMetal, pymoo and Metaheuristics.jl replace only on a strictly better trial; a tie has probability about 0 on Rastrigin.
- **SciPy:** a population of exactly 100 is given as an `init=` array of uniform points (its `popsize` is a multiple of n); its convergence test can't be turned off, and at `tol=0`, `atol=0` it ends a run whose 100 values are all equal, reported with `ended_by` ([rule 2.2](rules.md#2-the-budget)); the best individual is moved to the first slot each generation.
- **pymoo:** `prob_mut=0.0`, an undocumented option passed through to its DE variant, turns off the polynomial mutation it otherwise applies to 10% of trials; the forced gene is used only when no gene came from the mutant.
- **jMetal:** the population is sorted by fitness after each generation, so the targets are visited best first.

**Rosenbrock 10, CMA-ES** (pycma is the reference):
- **Learning rates for n = 10, λ = 10** (Hansen's 2016 tutorial: c_σ 0.2844, d_σ 1.2844, c_c 0.2950, c_1 0.01528, c_μ 0.02015): genoxide, pygmo, jMetal and Evolutionary.jl use the tutorial's values; pycma and pymoo (pycma inside) use c_σ 0.3196, d_σ 1.3196, c_μ 0.02355 and pycma's h_σ test; DEAP uses c_σ 0.3196, d_σ 1.3196, c_c 0.2857. All weights are 0.4563, 0.2708, 0.1622, 0.0852, 0.0255 (μ_eff 3.167), positive only.
- **Bounds** (the reference is pycma's `BoundTransform`): genoxide draws a sample outside the box again, up to 100 times, then clips it; pygmo, jMetal and Evolutionary.jl clip it; DEAP evaluates the closest point in the box plus a penalty (`ClosestValidPenalty`).
- **pycma:** `CMA_active=False`, since its default is active CMA; `maxstd_boundrange` off, its default cap of a coordinate's standard deviation at a third of the box.
- **pymoo:** pycma's `noeffectaxis` and `noeffectcoord` can't be turned off; when one fires, the run ends there, with `ended_by`.
- **pygmo:** the mean starts at the best of 10 uniform points (10 evaluations); h_σ in the 2006 form; the σ exponent capped at 0.6.
- **jMetal:** 10 uniform initial evaluations it doesn't use; h_σ's exponent 2(g + 2); χ_n = √n (a bug, below).
- **Evolutionary.jl:** its default weights are active, so the adapter passes the positive weights and, with them, the four learning rates by the library's own default formulas; the mean is the first of 5 uniform points; one type-probe evaluation.

## Bugs found

The benchmark runs what a library's users get, so its results show its bugs ([rule 8.4](rules.md#8-reporting)). A run that a library ends with an error ends there, as not reached, with `ended_by`. No bug was found in genoxide, its Python package, pycma or SciPy.

| Library | Bug | Effect on the results | Reported |
|---|---|---|---|
| [PyGAD](libraries/pygad.md#bugs-found) 3.7.0 | `two_points_crossover` draws only the first point; the second is always n / 2 after it | the OneMax runs | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.1 | a GA offspring that isn't crossed is its parent, not a copy, and is mutated in place | the OneMax runs | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.1 | CMA-ES: `eigen!` overwrites the covariance matrix before its update | with the two below, no Rosenbrock run reaches the target, and runs end at a failed eigendecomposition after about 100,000 evaluations | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.1 | CMA-ES: σ is updated with ‖p_σ‖ / n instead of ‖p_σ‖ / E‖N(0, I)‖ (h_σ too) | σ shrinks about 7% a generation until it underflows | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.1 | CMA-ES: the rank-μ update uses z zᵀ instead of y yᵀ | as above | not yet |
| [Evolutionary.jl](libraries/evolutionary_jl.md#bugs-found) 0.12.1 | DE crosses with the base vector, `BINX`'s CR is the probability of the base's gene, no forced gene, `K` documented but unused | none: its DE isn't run | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES computes C^(−1/2) with partly square-rooted eigenvalues, so σ grows without bound once the eigenvalues are below 1 | no Rosenbrock run reaches the target | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | `CMAESUtils.tql2` throws `ArrayIndexOutOfBoundsException` when the covariance matrix holds NaN | every Rosenbrock run ends there, after about 204,000 evaluations, with `ended_by` | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES draws its samples from a generator seeded with the clock | worked around: the adapter seeds it; the algorithm is unchanged | not yet |
| [jMetal](libraries/jmetal.md#bugs-found) 7.5 | CMA-ES computes χ_n with integer divisions, √n instead of 3.085 for n = 10 | σ shrinks a little faster than intended | not yet |
| [pygmo](libraries/pygmo.md#bugs-found) 2.19.8 | `de` draws r1, r2, r3 without excluding the target | about 3% of trials use the target | not yet |
| [pygmo](libraries/pygmo.md#bugs-found) 2.19.8 | the docstrings of `cmaes` (and `sade`, `de1220`, `xnes`) swap `ftol` and `xtol` | documentation only | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | `fast_fill_random` keeps positions instead of row numbers, so about 0.14% of DE trials have repeated indices or the target among them | the Rastrigin runs: the trials with r2 = r3 are a crossover without a difference vector, and halve the error at the end (69.6 as it is, 125.7 with the fix) | not yet |
| [pymoo](libraries/pymoo.md#bugs-found) 0.6.2 | DE ignores `dither`; its docs don't mention the polynomial mutation it applies by default | the adapter sets `prob_mut=0.0` | not yet |
| [Metaheuristics.jl](libraries/metaheuristics_jl.md#bugs-found) 3.5.0 | the `DE` docstring gives F = 1.0 as the default; the code's is 0.7 | none: F is set | not yet |

**Found in methods no longer in the suite**, listed on the libraries' pages: PyGAD's `sbx` pulls children towards the lower bound ([ahmedfgad/GeneticAlgorithmPython#369](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/369)), its random mutation never changes a permutation, and its docs misdescribe `swap_mutation`; radiate's guide recommends `ShuffleCrossover` for permutations, whose children aren't permutations, and misdescribes `GaussianMutator`; DEAP's DE example's exponential crossover is inverted, and its BIPOP-CMA-ES example reads the sort order backwards; pymoo's tournament of more than 2 acts as binary, its CMAES with seed 0 isn't repeatable, its pattern search can't be seeded, and its binary and permutation pages disagree with their code; Evolutionary.jl's (μ+λ)-ES can overwrite a surviving parent; jMetal's initial permutations and coral reef optimization ignore its seed; Nevergrad's `NgIohTuned` metamodel crashes with NumPy 2.5. The bugs found in the libraries that left the suite (genetic_algorithm, moors, openGA, Jenetics) and in the multi-objective algorithms come back with them.
