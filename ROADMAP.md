# genoxide roadmap

## Vision

A complete, fast and reliable optimization library, in Rust and for Python: evolutionary and population-based methods, local and gradient-based methods, constrained nonlinear programming and Bayesian optimization, with the same guarantees of reproducibility, validation and speed. The plan for the methods beyond the evolutionary ones is [docs/optimization-plan.md](docs/optimization-plan.md).

### Success criteria for 1.0

- **Coverage:** the algorithms and operators of DEAP and pymoo combined; the local, gradient-based and constrained methods of SciPy's optimize; Bayesian optimization with Gaussian processes and TPE.
- **Speed:** fast on every problem of the public benchmark suite, in time to target and evaluations per second.
- **Reproducibility:** bit-identical results per seed on any thread count.
- **Docs:** the whole public API, and an example per algorithm.
- **Python:** the same capabilities.

## Design principles

1. **Validated, typed configuration:** invalid settings are builder errors; mismatched operators don't compile.
2. **Guaranteed determinism:** seeded streams per run, island and worker; no hash-order or scheduling effects; explicit ties.
3. **Ask / tell at the core:** every algorithm is an `ask() → evaluate → tell()` state machine; the run loop is a thin layer.
4. **No wasted work:** mutations always change something; invalid solutions are cached; no allocations per generation.
5. **Explicit semantics:** documented probabilities and edge cases; rates validated to [0, 1].
6. **Batteries included, costs opt-in:** statistics, hall of fame and checkpoints cost only when enabled.

### Pitfalls ruled out by design

From a review of existing libraries (e.g. genetic_algorithm, [issues #11 to #78](https://github.com/basvanwesting/genetic-algorithm/issues?q=is%3Aissue+author%3Atachsin)). Each rule has a test.

| Pitfall | genoxide rule |
|---|---|
| Wrong best index with missing fitness values | `Invalid` is a separate state |
| `HashMap` order or unstable sorts change results | Ordered containers; index tie-breaks |
| Identical parallel runs under one seed | One derived rng stream per run, island and thread |
| No selection pressure without surplus survivors | Separate parent selection and survival |
| No-op mutations or crossovers (self-swap, point 0) | Operators guarantee a change (property-tested) |
| Off-by-one elitism | Exact, property-tested elitism counts |
| Integer overflow, float steps that don't advance | Checked or saturating arithmetic; guaranteed progress |
| NaN fitness taken as a valid score | NaN is an error or `Invalid`, configurable |
| Deadlocks with one rayon thread | No blocking waits on pool threads; tested in CI |
| Docs drifting from the code | The docs' code blocks, README.md and AGENTS.md are doctests; CI runs `examples/` and compares their output |

## Architecture

- **`Genome`:** the representation: bits (bit-packed), integers, bounded reals, permutations; planned: mixed (per-gene types), trees (GP), graphs (NEAT).
- **`Fitness`:** totally ordered `f64`, single or multi-objective (`[f64; M]`, with the number of objectives fixed at compile time), optional constraint violation; batch and async evaluation.
- **Operators:** `Select`, `Crossover`, `Mutate`, generic over the genome; survival is each algorithm's scheme.
- **`Algorithm`:** ask / tell state machines (GA, ES, CMA-ES, DE, PSO, NSGA-II, …; planned: Nelder-Mead, L-BFGS-B, SQP, Bayesian optimization).
- **Derivatives (planned):** supplied gradients, finite differences, constraint and residual Jacobians, declared by the fitness function.
- **Models (planned):** Gaussian processes for Bayesian and surrogate-assisted optimization.
- **`Engine`:** termination, parallel evaluation, observers, cancellation, and a hook that changes the algorithm between generations (parameter control, re-evaluation).
- **`Observer`:** statistics, hall of fame, Pareto archive, logging, checkpoints.
- **Errors:** one typed error enum; no `&'static str` errors, no panics in library code.

## Milestones

Done means implemented, documented, tested (property tests for operators) and benchmarked where relevant.

### 0.0: Project setup ✅
- [x] Name, repository, dual MIT / Apache-2.0 license
- [x] README (homepage) and this roadmap
- [x] Crate skeleton (edition 2024, MSRV 1.86; 1.88 since #314)

### 0.1: Foundations ✅

#### Project
- [x] CI on Linux, macOS, Windows, MSRV and a single rayon thread
- [x] `#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`
- [x] Benchmark suite in [`benchmarks/`](benchmarks/)
- [x] Contributing guide, issue and PR templates
- [x] Versioning policy and automated releases (release-plz, cargo-semver-checks)

#### Core types
- [x] Error enum (no panics in library code)
- [x] Portable, seedable rng with derived streams per run / island / worker
- [x] Fitness: totally ordered `f64`, `Invalid`, NaN policy, maximize / minimize
- [x] Individual (genome, fitness, age) and Population

#### Genomes
- [x] `Genome` trait
- [x] Binary, bit-packed
- [x] Integer, bounded per gene
- [x] Real, bounded per gene
- [x] Permutation

#### Selection
- [x] Tournament
- [x] Roulette wheel and stochastic universal sampling
- [x] Rank
- [x] Truncation
- [x] Random

#### Crossover
- [x] One-point, two-point, k-point (points between genes only)
- [x] Uniform (exactly 50% per gene, or a given rate)

#### Mutation
- [x] Bit-flip (per-gene rate or exactly n genes)
- [x] Uniform (integer / real)
- [x] Swap (permutation)

#### Engine
- [x] Ask / tell core
- [x] Generational GA, steady-state GA, (μ+λ) and (μ,λ)
- [x] Elitism
- [x] Stop conditions and their combinations
- [x] Deterministic parallel evaluation (rayon)
- [x] Cancellation (abort flag)
- [x] Validating builder

#### Observers
- [x] Statistics per generation
- [x] Hall of fame (top-k unique)

#### Docs and examples
- [x] API docs for everything public, with doctests
- [x] Examples: OneMax, knapsack, N-Queens, Rastrigin
- [x] AGENTS.md (guide for AI coding assistants)

#### Performance
- [x] criterion benchmarks for the hot paths
- [x] Instruction-count benchmarks in CI (gungraun)
- [x] genoxide in the benchmark suite, first results

### 0.2: Real-valued and permutation excellence ✅
- [x] Real-valued crossover: SBX, BLX-α, arithmetic
- [x] Real-valued mutation: Gaussian, polynomial
- [x] Self-adaptive Gaussian mutation (a step size per individual)
- [x] Permutation crossover: PMX, OX1, CX, edge recombination
- [x] Permutation mutation: inversion (2-opt), insertion, scramble
- [x] Local search: hill climbing, simulated annealing
- [x] Local search: tabu search, iterated local search
- [x] Memetic (Lamarckian) GA
- [x] Constraint handling: Deb's feasibility rules, penalty functions

### 0.3: Evolution strategies and swarm ✅
- [x] CMA-ES, with IPOP / BIPOP restarts
- [x] sep-CMA-ES for high dimensions
- [x] Differential evolution: rand/1, best/1, current-to-pbest
- [x] Adaptive DE: JADE, SHADE, L-SHADE
- [x] Particle swarm: global and local topologies
- [x] (μ/ρ +, λ)-ES with self-adaptation

### 0.4: Multi-objective ✅
- [x] Non-dominated sorting (fast and log variants), crowding distance
- [x] NSGA-II
- [x] NSGA-III (reference points)
- [x] SPEA2
- [x] MOEA/D
- [x] SMS-EMOA
- [x] Pareto archive
- [x] Indicators: hypervolume, IGD / IGD+, spread
- [x] Constrained dominance

### 0.5: Scale and operations ✅
- [x] Island model: ring, fully connected and random topologies
- [x] Asynchronous steady-state evaluation
- [x] Batch evaluation hook (SIMD, GPU, remote)
- [x] GPU evaluation example
- [x] Checkpoint and resume (`serde` feature)
- [x] Runs described in TOML / JSON, with a small CLI
- [x] `tracing` integration and progress reporting

### 0.6: Python ✅
- [x] `pip install genoxide` (PyO3 / maturin) for Linux, macOS, Windows
- [x] Python fitness functions and vectorized numpy batch fitness
- [x] Pythonic builders
- [x] Examples matching the DEAP / pymoo tutorials
- Zero-copy numpy genomes: moved to 0.10.

### 0.7: Correctness ✅
Fixes from the review of 0.6.0 ([#116](https://github.com/tachsin/genoxide/issues/116)). Some change seeded results.
- [x] DE restarts: trials visible to observers, no restart every other generation on constrained problems, migrants evaluated once
- [x] Duplicate elimination without fingerprint collisions, same on 32 and 64 bits
- [x] Runs that could never end stop as stalled
- [x] Huge sizes are setting errors, not panics or hangs
- [x] The steady-state GA doesn't propose twins, and PSO's initial velocities respect `max_velocity`
- [x] Python: clearer errors, stricter settings, license files in the wheels
- [x] Benchmarks: numpy fitness in pymoo and PyGAD, timings on pinned P-cores
- [x] DE defaults: SHADE's published settings, with its random `p` per trial

### 0.8: Test problems ✅
The test problem library of [docs/problems-plan.md](docs/problems-plan.md), batches 1 to 3, and the fixes from the review of 0.7 ([#201](https://github.com/tachsin/genoxide/issues/201) to [#229](https://github.com/tachsin/genoxide/issues/229)). `multi::problems::TestProblem` became `MultiProblem`: a breaking change.
- [x] `problems`: 16 classic functions, CEC 2006's g01-g06 and 8 engineering design problems, each with its optimum and reference ([#170](https://github.com/tachsin/genoxide/pull/170), [#191](https://github.com/tachsin/genoxide/pull/191))
- [x] `multi::problems`: `MultiProblem`, constraints, and 13 classic two- and three-objective problems ([#177](https://github.com/tachsin/genoxide/pull/177))
- [x] The same problems in Python, evaluated in Rust, and `genoxide.indicators`
- [x] Examples with their output checked in CI, a full explanation and a recorded run ([#193](https://github.com/tachsin/genoxide/pull/193), [#194](https://github.com/tachsin/genoxide/pull/194), [#198](https://github.com/tachsin/genoxide/pull/198)), one per problem ([#253](https://github.com/tachsin/genoxide/pull/253))
- [x] The fixes from the review of 0.7
- [x] `genoxide::math`: the same results to the bit on every platform ([#263](https://github.com/tachsin/genoxide/pull/263))
- [x] A GA's rates and operators changed between generations, and its population re-evaluated when the fitness function changes ([#248](https://github.com/tachsin/genoxide/pull/248), [#249](https://github.com/tachsin/genoxide/pull/249))

### 0.9: More test problems ✅
- [x] Batch 4: DTLZ5-7, the binary ZDT5 and WFG1-9 ([#270](https://github.com/tachsin/genoxide/pull/270)); batch 5 in 0.9.1: CEC 2006's g07-g18 ([#277](https://github.com/tachsin/genoxide/pull/277)); each with its example
- [x] A multi-objective front has each genome once, and polynomial mutation reaches the bounds: breaking fixes ([#273](https://github.com/tachsin/genoxide/pull/273), [#276](https://github.com/tachsin/genoxide/pull/276))
- [x] Benchmarks: genoxide's releases compared by instruction counts ([#267](https://github.com/tachsin/genoxide/pull/267))
- Genetic programming and neuroevolution, first planned for 0.9: moved to 0.10.

### 0.9.2: Control and parallel breeding
On main, in the release PR ([#280](https://github.com/tachsin/genoxide/pull/280)). New features, compatible with 0.9.1's API, so a patch release; the breaking changes of #283 and #284 are the benchmark harness's, not the library's.
- [x] Parameter control: `Engine::control`, setters for DE, PSO and local search, `Islands::islands_mut` ([#292](https://github.com/tachsin/genoxide/pull/292)); in Python, `run(control=...)` ([#294](https://github.com/tachsin/genoxide/pull/294))
- [x] Re-evaluation in every single-objective algorithm but the steady-state GA, for a fitness function that changes during a run ([#292](https://github.com/tachsin/genoxide/pull/292))
- [x] Parallel breeding for the GA, DE and ES, the same results on any number of threads ([#289](https://github.com/tachsin/genoxide/pull/289), [#295](https://github.com/tachsin/genoxide/pull/295)), also in Python ([#296](https://github.com/tachsin/genoxide/pull/296))
- [x] Isolated islands, without migration ([#293](https://github.com/tachsin/genoxide/pull/293))
- [x] Faster: the Python package on bit genomes ([#291](https://github.com/tachsin/genoxide/pull/291)), the ES ([#298](https://github.com/tachsin/genoxide/pull/298))
- [x] Benchmarks: a matched suite of three problems, one method each ([#284](https://github.com/tachsin/genoxide/pull/284)), with every library bug it found reported upstream ([notes](docs/benchmarks/notes.md#bugs-found))
- [x] What a fitness function computes besides the fitness, kept by the engine for the individuals it holds ([#246](https://github.com/tachsin/genoxide/issues/246))

### 0.10: Rust 1.88
- [x] Rust 1.88, and small performance gains across the library ([#314](https://github.com/tachsin/genoxide/issues/314), [#315](https://github.com/tachsin/genoxide/pull/315))
- [x] Python: `Progress` builds its arrays only when a callback reads them ([#324](https://github.com/tachsin/genoxide/pull/324))

### 0.11: Genetic programming and neuroevolution
The plan: [docs/gp-neuroevolution-plan.md](docs/gp-neuroevolution-plan.md).
- [x] Tree GP, strongly typed
- [ ] Subtree crossover; point, subtree and hoist mutation; bloat control
- [ ] Symbolic regression examples
- [ ] NEAT (speciation, innovation numbers)
- [ ] Neuroevolution with evolution strategies
- [x] Python: the ES, islands and checkpoints, and a batch's matrix reused ([#342](https://github.com/tachsin/genoxide/pull/342))

### 0.12: Local optimization
- [ ] Linear algebra through a dependency pinned to a portable path, convergence stops, restarts, Nelder-Mead ([docs/optimization-plan.md](docs/optimization-plan.md), batch A1)
- [ ] Gradients (supplied, or by finite differences evaluated as one batch), line searches, L-BFGS-B (batch A2)

### 0.13: Bayesian optimization
- [ ] Gaussian processes; EI, log-EI, UCB and PI; batch, constrained and integer-variable Bayesian optimization, also on the asynchronous engine (batch B)

### 0.14: Constrained nonlinear programming
- [ ] SQP and the augmented Lagrangian, on the constrained test problems (batch C)

### 0.15: More local methods
- [ ] BFGS, conjugate gradient, trust region, Levenberg-Marquardt, the Adam family (batch D1)
- [ ] BOBYQA, COBYLA, pattern search, MADS, basin hopping (batch D2)

### 0.16: Advanced Bayesian optimization, surrogates and multi-fidelity
- [ ] ParEGO, EHVI, TPE, TuRBO, mixed variables (batch E)
- [ ] Surrogate-assisted evolution, multi-fidelity, DIRECT (batch F)

### 0.17: Frontier
- [ ] Quality-diversity: MAP-Elites, CMA-ME, novelty search
- [ ] LLM-guided evolution (async operators calling a language model)
- [ ] Adaptive operator selection and automatic parameter tuning

### Throughout
- [ ] The test problem library's later batches, each problem with its example ([#260](https://github.com/tachsin/genoxide/issues/260)), checked against the original papers ([#168](https://github.com/tachsin/genoxide/issues/168))
- [ ] More benchmark problems, one matched method each, once the current three are settled (see [Benchmarks](#benchmarks))

### 1.0: Stable
- [ ] API stabilization, semver guarantees, MSRV policy
- [ ] Published benchmark report
- [ ] Book (mdBook) with a guide per problem type

## Quality

- **CI:** Linux, macOS, Windows; stable and MSRV; clippy, fmt, rustdoc with `-D warnings`; a single-thread rayon job.
- **Property tests (proptest)** for every operator: validity (permutations stay permutations), bounds, no no-ops, exact rates.
- **Fuzzing** of builders and the configuration format: planned.
- **Performance:** criterion (wall time) and gungraun (exact instruction counts; fail CI on regressions) for every hot path.
- **Safety:** `#![forbid(unsafe_code)]`; any `unsafe` (SIMD, bit tricks) behind a feature, with `SAFETY` comments and Miri tests.
- **Docs:** every code block in the API docs, README.md and AGENTS.md is a doctest; CI runs `examples/` and compares each output with its `output.txt`.

## Benchmarks

[`benchmarks/`](benchmarks/) runs every library on the same problems, with identical fitness functions and budgets. Results are published for every release.

- **Now: a small matched suite.** Three problems, one method each, the same in every library, with its own implementation set to a written definition ([rules](docs/benchmarks/rules.md#6-the-methods)): a GA on OneMax 1000, DE/rand/1/bin on Rastrigin 30 and CMA-ES on Rosenbrock 10. It compares implementations of the same algorithm, not each library's pick of a method.
- **Later, after these:** more problems, one matched method each: binary (LeadingOnes, deceptive trap, NK landscapes, knapsack), permutation (N-Queens, TSPLIB, QAP, flow shop), continuous (BBOB / COCO functions in 10–100 dimensions); then multi-objective problems (ZDT, DTLZ, WFG) with a matched NSGA-II.
- **Measurements:** success rate, time and evaluations to target, the distance to the optimum at the end, evaluations per second, genoxide's own releases compared by the CPU instructions of the same runs (Callgrind, genoxide only); later peak memory and scaling with population, genome size and threads.
- **Output:** library versions, JSON, a markdown table and graphs.
- **Libraries:** those with their own implementation of a suite's method. Now: genoxide and its Python package, DEAP, PyGAD, radiate, pycma, SciPy, pygmo, pymoo, jMetal, Evolutionary.jl and Metaheuristics.jl (the [methodology](benchmarks/README.md) lists which runs which). Others come back with the problems whose methods they have:

| Language | Libraries |
|---|---|
| Python | Nevergrad, EvoX, geatpy |
| Java | Jenetics, MOEA Framework |
| C++ | openGA, ParadisEO |
| C# | GeneticSharp |
| Rust | moors, genetic_algorithm, genevo, oxigen |

## Not planned

Reverse-mode automatic differentiation, model training frameworks and general-purpose machine learning; linear, quadratic and mixed-integer programming solvers; large sparse nonlinear programming. genoxide focuses on black-box and small-to-medium dense problems.
