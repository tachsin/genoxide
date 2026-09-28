# Evolutionary.jl (Julia, 0.12.1)

Evolutionary.jl is a Julia package of evolution strategies (ES), CMA-ES, genetic algorithms (GA), differential evolution (DE), NSGA-II and genetic programming, with mutation, crossover and selection operators. Its docs are at [docs.sciml.ai/Evolutionary](https://docs.sciml.ai/Evolutionary/stable/), from `docs/src` of [SciML/Evolutionary.jl](https://github.com/SciML/Evolutionary.jl). The sources cited below (`src/...`) are those of the pinned 0.12.1, as the package manager installs it: the repository has no tag for it.

Adapter: [benchmarks/adapters/evolutionary_jl/](../../../benchmarks/adapters/evolutionary_jl/).
Know a way to set Evolutionary.jl closer to a definition? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## What it runs

The matched suite: each problem with one method, defined the same for every library ([rule 6](../rules.md#6-the-methods)), and each library's own implementation of it.

| Scenario | Method | Evolutionary.jl |
|---|---|---|
| OneMax 1000, matched | GA as DEAP's `eaSimple` ([6.2](../rules.md#6-the-methods)) | `ga`: `GA` |
| Rastrigin 30, matched | DE/rand/1/bin ([6.3](../rules.md#6-the-methods)) | can't run: see [Can't run](#cant-run) |
| Rosenbrock 10, matched | CMA-ES ([6.4](../rules.md#6-the-methods)) | `cma_es`: `CMAES`, with its bugs (rule 8.4) |

The adapter prints nothing for any other scenario ([`SCENARIOS`, lines 220-224](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L220-L224)).

## How the adapter runs Evolutionary.jl

- **Fitness functions:** in Julia ([bench.jl, lines 32-49](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L32-L49)); Evolutionary.jl minimizes, so OneMax is the negative count of ones. `values` evaluates them for rule 1.2 ([lines 248-262](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L248-L262)).
- **Evaluations:** `counted` counts every call, with the first hit and the solutions outside the bounds, including the one `EvolutionaryObjective` makes before the run to learn the value's type (`zero(f(x))`, `src/api/objective.jl`) ([`counted`, lines 94-112](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L94-L112)).
- **Stop:** the `callback` of [`Options`](https://docs.sciml.ai/Evolutionary/stable/tutorial/#General-options), called after the initial population and after every generation, ends the run at the target, the budget or the time cap, and marks the generation's end for `last_generation` ([`options`, lines 123-136](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L123-L136)).
- **No convergence criterion (rule 2.2):** `iterations = typemax(Int)` lifts the iteration limit (1,000 by default, 1,500 for CMA-ES), and `successive_f_tol = typemax(Int)` turns off the convergence test, which ends a run once the method's metric (`AbsDiff(1e-12)` for GA and CMA-ES) has held for more than `successive_f_tol` generations (`optimize`, `src/api/optimize.jl`, lines 123-127). Both are documented options.
- **Ended by the library (rule 8.4):** the CMA-ES ends a run itself when its covariance matrix can't be decomposed (`update_state!` catches the error and returns `true`, `src/cmaes.jl`, lines 156-162). That isn't worked around, and the methods have no restarts: the run ends there, not reached, and its line says `"ended_by": "eigendecomposition failed"` ([`run_once`, lines 138-149](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L138-L149)). Both methods evaluate every child, so neither stalls.
- **Time:** from the run's `Budget`, created just before `optimize` builds the initial population, to the moment the run returns, before the adapter reads anything from its result ([lines 293-298](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L293-L298)). Each method first makes the warm-up run of rule 4.2 ([line 289](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L289)).
- **Seeds (rule 5.2):** a `Xoshiro(seed)` passed as the `rng` of `Options`, which the operators and the initial population use. The same seed repeats a run, and seed 1 gives the same alone as after seed 0 (tested for both methods).
- **One thread:** [run.sh](../../../benchmarks/adapters/evolutionary_jl/run.sh) runs Julia with `--threads=1 --gcthreads=1,0` and BLAS with one thread.
- **Version:** the published results measure 0.12.1, the version the adapter pins.
- **Separate tests:** 2026-09-28, Evolutionary.jl 0.12.1, seeds 0 to 2, the scenario's budget, 60 s cap. `outside` was 0 in every run.

### Where the time goes

Audited on 2026-09-29 with Julia's sampling profiler, `@timed` (allocations, garbage collection, compilation) and seeds 0 to 2 timed three times, alternating the old and the new adapter.

- **GA:** nearly all the time is the library's GA. `TPX` swaps its segment one bit at a time through the `BitVector`s (about 60%), and `tournament` allocates its groups (about 20%). The fitness function, `-count(x)` on a `BitVector`, takes under 1%, and the adapter's counter about 3%. A run allocates 32 MiB. The genome is a `BitVector`, as in the library's tutorial (`BitVector(zeros(30))`) and its OneMax test (`BitArray(rand(rng, Bool, N))`, `test/onemax.jl`).
- **CMA-ES:** the library's `eigen!`, LAPACK's `syevr` on a 10 × 10 matrix, takes about 54%, and the rank-μ sum of `update_state!` 22%. The fitness function and the counter take under 1% together. A run allocates 428 MiB, and garbage collection takes about 10% of it.
- **Nothing to set:** `store_trace = false`, `show_trace = false` and `parallelization = :serial` are the defaults of `Options` (`src/api/types.jl`, lines 119-131). The callback makes the library build a trace record, a `Dict`, every generation (`trace!`, `src/api/utilities.jl`, lines 32-48), which is negligible. It's the documented way to stop at the target.
- **Compilation in the first timed run: fixed.** Julia compiled nothing during a timed run except once: the adapter took the method's result apart (`generations, ended_by = run(...)`) before stopping the clock. The result's type isn't known in `main`, and the warm-up doesn't use its result, so Julia compiled that on seed 0: 6 to 17 ms of its time. The clock now stops as the run returns. The runs are the same (the evaluations, best values, first hits and `ended_by` of seeds 0 to 2):

| Seeds 0, 1, 2, three times | Time (s) before | Time (s) after |
|---|---|---|
| GA | 0.040, 0.041, 0.030; 0.049, 0.027, 0.024; 0.038, 0.029, 0.026 | 0.032, 0.029, 0.025; 0.034, 0.028, 0.026; 0.033, 0.029, 0.026 |
| CMA-ES | 0.145, 0.148, 0.140; 0.150, 0.149, 0.143; 0.155, 0.146, 0.139 | 0.137, 0.144, 0.136; 0.137, 0.152, 0.141; 0.138, 0.151, 0.142 |

## OneMax 1000: the GA

**Method** ([`onemax_ga`, lines 155-176](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L155-L176)): the library's [`GA`](https://docs.sciml.ai/Evolutionary/stable/ga/) (`src/ga.jl`) with its own operators, unchanged from the published runs.

| Definition | Evolutionary.jl | Source |
|---|---|---|
| population 300, uniform random bits | `populationSize = 300`; the initial individuals from `() -> bitrand(rng, 1000)` | `initial_population`, `src/api/utilities.jl` |
| 300 tournaments of 3 | `selection = tournament(3)`: groups of 3 from a shuffled population, the best wins | `tournament`, `src/selections.jl`, lines 150-175 |
| pairs crossed with probability 0.5, two-point crossover | `crossoverRate = 0.5`, `crossover = TPX` | `recombine!`, `src/ga.jl`, lines 124-140; `TPX`, `src/recombinations.jl`, lines 83-93 |
| each child mutated with probability 0.2, bit flip | `mutationRate = 0.2`, `mutation = flip` | `mutate!`, `src/ga.jl`, lines 142-154; `flip`, `src/mutations.jl`, lines 130-138 |
| generational, no elitism | the children replace the population; `ɛ = 0` elites | `update_state!`, `src/ga.jl`, lines 83-122 |
| no convergence criterion | turned off (above) | |

**Differences** (none changes the algorithm):
- **Mutation:** `flip` flips exactly one random bit of a mutated child; the definition flips each bit with probability 1/1000, one bit on average. Evolutionary.jl has no per-bit flip for `GA`.
- **Tournament:** `tournament` takes its contestants from a random permutation of the population, 3 at a time, and draws a new permutation when it runs out: without replacement within a permutation, where the definition draws with replacement. Each contestant is still uniform.
- **Two-point crossover:** `TPX` draws two positions uniformly and swaps the genes from one to the other, both included; the definition's cut points lie between genes. The same operator, with slightly different segment lengths.
- **Evaluations:** every child is evaluated, crossed or mutated or not, where the definition keeps an unchanged copy's fitness. Those evaluations are counted (about 40% of the children here: 0.5 × 0.8).
- **Copies:** a pair that isn't crossed passes the parents themselves, not copies, and `mutate!` then changes them in place (a bug, see [Bugs found](#bugs-found)). It runs as users get it (rule 8.4).

**Keeping going:** no convergence criterion; every child is evaluated, so it never stalls.

**Separate tests:** OneMax 1000 (budget 2,000,000, cap 60 s):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| ga | 3 | 3 | 117,156 | 1,000 | 1,000 | 1,000 | 0 |

## Rosenbrock 10: CMA-ES

**Method** ([`rosenbrock_cma_es`, lines 182-218](../../../benchmarks/adapters/evolutionary_jl/bench.jl#L182-L218)): the library's [`CMAES`](https://docs.sciml.ai/Evolutionary/stable/cmaes/) (`src/cmaes.jl`), a (μ/μ_W, λ)-CMA-ES, set with its documented keyword arguments. It runs with its bugs (rule 8.4): they keep it from reaching the target (see [Bugs found](#bugs-found)).

| Definition | Evolutionary.jl | Source |
|---|---|---|
| λ = 10, μ = 5 | `lambda = 10`, `mu = 5` | constructor, `src/cmaes.jl`, lines 29-47 |
| w_i ∝ ln((λ + 1) / 2) − ln i, positive, sum 1; no negative weights | `weights` passed: 0.4563, 0.2708, 0.1622, 0.0852, 0.0255, then 0 for the other five; μ_eff = 3.1673 | lines 87 and 109-114 |
| Hansen's learning rates and damping | `c_1`, `c_c`, `c_mu`, `c_sigma` passed with the library's default formulas, which are Hansen's 2016: c_σ = 0.2844, c_c = 0.2950, c_1 = 0.01528, c_μ = 0.02015; d_σ = 1.2844, the library's (Hansen's) | lines 94-97, 125 |
| CSA, rank-one and rank-μ updates, h_σ | all of them, with the bugs below | `update_state!`, lines 137-205 |
| mean uniform in the box, σ₀ = 4.5, C₀ = I | `BoxConstraints(-5, 10, 10)`: the mean is the first of μ points drawn uniformly in the box; `sigma0 = 4.5`; C₀ = I | `optimize`, `src/api/optimize.jl`, lines 34-40; `initial_population`, `src/api/utilities.jl`, lines 103-128; `initial_state`, `src/cmaes.jl`, lines 81 and 133 |
| no restarts, no convergence criterion | none: the convergence test is turned off (above); the library can still end a run when its covariance matrix breaks down (below) | |

**Differences:**
- **Weights and learning rates:** the library's default weights (without `weights`) are active: the worst five get −0.0853, −0.2365, −0.3674, −0.4829 and −0.5862, which the definition doesn't allow. Passing `weights` is the only way to turn them off, and with weights given, the library takes other learning rates (c_c = c_σ = 1/√n, c_μ = μ_eff/n², c_1 = 2/n², lines 109-114). So the adapter passes the four learning rates too, with the formulas of the library's own defaults (lines 94-97); they are Hansen's 2016 formulas exactly (the library's c_μ, with α_cov = 2, is the tutorial's). The algorithm is the definition's, with the library's own default values.
- **Bounds (rule 2.4):** every sample is clipped to the box before it's evaluated (`apply!` → `clip!`, `src/api/constraints.jl`, lines 135 and 281-291; line 167 of `src/cmaes.jl`); the reference is pycma's `BoundTransform`. The mean update uses the clipped samples, the evolution paths and the rank-μ update the unclipped z. It acts only on genes that leave the box.
- **Initial mean:** the first of the μ = 5 uniform points of the initial population; the other four are neither used nor evaluated.
- **One more evaluation:** `EvolutionaryObjective` evaluates the initial mean once, to learn the value's type (counted).

**Keeping going:** no convergence criterion, no restarts. With the step-size bug, σ shrinks every generation: from 4.5 to 2.0 by generation 10, 0.08 by generation 50, 10⁻⁵ by 200 and 10⁻¹¹ by 500 (seed 0), while the best stops improving. After about 9,700 generations σ underflows (10⁻³⁰⁹) and the eigendecomposition fails: the library ends the run there (rule 8.4, `"ended_by": "eigendecomposition failed"`).

**Separate tests:** Rosenbrock 10 (budget 500,000, cap 60 s):

| Solver | Runs | Reached | Median first hit (evaluations) | Best value: median | best | worst | Capped |
|---|---|---|---|---|---|---|---|
| cma_es | 3 | 0 | - | 8.85 | 7.88 | 103.7 | 0 |

Every run ended at the failed eigendecomposition, after 96,901 to 102,031 evaluations (0.12 to 0.14 s). Their best values hardly changed after the first 20,000 evaluations (by less than 0.01%).

For the bug report only, not in the benchmark: the same runs, seeds 0 to 9, with a copy of `update_state!` in which some of the three CMA-ES bugs below are fixed. The step size (in the ‖p_σ‖²/n form of Hansen's update) and the overwritten matrix: 10 of 10 reached the target, in 4,958 to 6,366 evaluations (median about 6,000), like pycma. The step size alone: 5 of 10, after 263,000 to 460,000. The overwritten matrix alone, or the rank-μ update alone: none. All three: 9 of 10, in about 4,800 to 8,000; with seed 8, σ grows without bound at a corner of the box, because the library clips the samples and moves the mean with the clipped ones, while the paths use the unclipped ones. The library as it is: none, best values 7.44 to 608.6.

## Can't run

- **Matched Rastrigin 30:** Evolutionary.jl's [`DE`](https://docs.sciml.ai/Evolutionary/stable/de/) (`src/de.jl`) isn't DE/rand/1/bin as Storn and Price define it. It recombines the mutant with the base vector instead of the target: `update_state!` (lines 59-76) calls `method.recombination(mutant, base)`, so the trial takes every gene from x_r1 or from v = x_r1 + F (x_r2 − x_r3), and nothing from x_i, which only competes with it. That's another variation operator (rule 6.1). Besides:
  - `BINX(Cr)` (`src/recombinations.jl`, lines 143-157) swaps each gene with probability Cr, so the trial keeps the base's gene with probability Cr: `BINX(0.9)` takes 90% of the genes from the base, not from the mutant (measured: 0.90 of the genes of 10 trials equal the base's); and there is no forced index j_rand;
  - the base is drawn with replacement (`random`, `src/selections.jl`, lines 182-183), independently of the target and of the two difference vectors (`randexcl` excludes only the target, line 68): r1 can be i, r2 or r3;
  - the initial population is never evaluated: its fitness starts at `maxintfloat` (line 42), so every trial of the first generation replaces its target.
  The rest would match: F fixed, generational replacement when f(u) ≤ f(x_i) (lines 78-94), clipping to the box.

## Bugs found

| Bug | Effect here | Worked around | Reported |
|---|---|---|---|
| CMA-ES: `eigen!(Symmetric(state.C))` (`src/cmaes.jl`, line 157) overwrites the covariance matrix: LAPACK's `syevr` destroys its input, so the upper triangle holds the tridiagonal reduction's data, and the update of line 197 starts from that instead of C | the covariance matrix is wrong in every generation | no | [SciML/Evolutionary.jl#179](https://github.com/SciML/Evolutionary.jl/issues/179), fix proposed in [SciML/Evolutionary.jl#183](https://github.com/SciML/Evolutionary.jl/pull/183) |
| CMA-ES: the step size is updated with ‖p_σ‖/n instead of ‖p_σ‖/E‖N(0, I)‖ (or ‖p_σ‖²/n in the other form of Hansen's update): `exp(min(1, (c_σ / d_σ) * (norm(s_σ) / N - 1) / 2))` (line 199). ‖p_σ‖ ≈ √n under random selection, so σ shrinks unless the path is n/√n = √n times longer than random. The h_σ test (line 189) has the same missing square | σ shrinks by about 7% a generation until it underflows, after about 9,700 generations; then the run ends at a failed eigendecomposition (`"ended_by"`) | no | [SciML/Evolutionary.jl#179](https://github.com/SciML/Evolutionary.jl/issues/179), fix proposed in [SciML/Evolutionary.jl#183](https://github.com/SciML/Evolutionary.jl/pull/183) |
| CMA-ES: the rank-μ update (line 194) uses z_i z_iᵀ, the standard normal samples, instead of y_i y_iᵀ with y_i = B D z_i, the steps in the search space | the rank-μ update ignores the learned shape of C | no | [SciML/Evolutionary.jl#179](https://github.com/SciML/Evolutionary.jl/issues/179), fix proposed in [SciML/Evolutionary.jl#183](https://github.com/SciML/Evolutionary.jl/pull/183) |
| GA: an offspring pair that isn't crossed is the parents themselves, not copies (`recombine!`, `src/ga.jl`, line 135), and `mutate!` then changes them in place. Two offspring can be one object, and with elitism (ε > 0) an elite can be mutated | the OneMax runs, as users get them | no | [SciML/Evolutionary.jl#180](https://github.com/SciML/Evolutionary.jl/issues/180), fix proposed in [SciML/Evolutionary.jl#184](https://github.com/SciML/Evolutionary.jl/pull/184) |

Found in methods no longer in the suite:
- DE: the crossover partner, `BINX`'s probability and the unevaluated initial population ([Can't run](#cant-run)), and `K` (the "recombination scale factor", default 0.5 (F + 1)), documented in the `DE` docstring but used nowhere (`src/de.jl`, line 18). Reported: [SciML/Evolutionary.jl#181](https://github.com/SciML/Evolutionary.jl/issues/181), fix proposed in [SciML/Evolutionary.jl#185](https://github.com/SciML/Evolutionary.jl/pull/185).
- The (μ+λ)-ES can lose a surviving parent (`update_state!`, `src/es.jl`): it writes each surviving offspring into the slot of its rank, which can overwrite a surviving parent. A (15/3+100)-ES on a sphere, from 2,000 random starts, lost its best parent in 3 of 40,000 generations. Reported: [SciML/Evolutionary.jl#182](https://github.com/SciML/Evolutionary.jl/issues/182), fix proposed in [SciML/Evolutionary.jl#186](https://github.com/SciML/Evolutionary.jl/pull/186).
- The earlier finding "CMA-ES doesn't converge in 30 dimensions" is explained by the three CMA-ES bugs above; it doesn't converge in 10 dimensions either with λ = 10.
