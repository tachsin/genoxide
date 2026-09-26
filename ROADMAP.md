# genoxide roadmap

## Vision

A complete, fast and reliable evolutionary computation library, in Rust and for Python.

### Success criteria for 1.0

- **Coverage:** the algorithms and operators of DEAP and pymoo combined.
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
| Docs drifting from the code | Every example is a doctest |

## Architecture

- **`Genome`:** the representation: bits (bit-packed), integers, bounded reals, permutations; planned: mixed (per-gene types), trees (GP), graphs (NEAT).
- **`Fitness`:** totally ordered `f64`, single or multi-objective (`[f64; N]` / `Vec<f64>`), optional constraint violation; batch and async hooks.
- **Operators:** `Select`, `Crossover`, `Mutate`, `Repair`, `Survive`, generic over the genome.
- **`Algorithm`:** ask / tell state machines (GA, ES, CMA-ES, DE, PSO, NSGA-II, …).
- **`Engine`:** termination, parallel evaluation, observers, cancellation.
- **`Observer`:** statistics, hall of fame, Pareto archive, logging, checkpoints.
- **Errors:** one typed error enum; no `&'static str` errors, no panics in library code.

## Milestones

Done means implemented, documented, tested (property tests for operators) and benchmarked where relevant.

### 0.0: Project setup ✅
- [x] Name, repository, dual MIT / Apache-2.0 license
- [x] README (homepage) and this roadmap
- [x] Crate skeleton (edition 2024, MSRV 1.86)

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
- Zero-copy numpy genomes: moved to 0.8.

### 0.7: Correctness
Fixes from the review of 0.6.0 ([#116](https://github.com/tachsin/genoxide/issues/116)). Some change seeded results.
- [x] DE restarts: trials visible to observers, no restart every other generation on constrained problems, migrants evaluated once
- [x] Duplicate elimination without fingerprint collisions, same on 32 and 64 bits
- [x] Runs that could never end stop as stalled
- [x] Huge sizes are setting errors, not panics or hangs
- [x] The steady-state GA doesn't propose twins, and PSO's initial velocities respect `max_velocity`
- [x] Python: clearer errors, stricter settings, license files in the wheels
- [x] Benchmarks: numpy fitness in pymoo and PyGAD, timings on pinned P-cores
- [x] DE defaults: SHADE's published settings, with its random `p` per trial

### 0.8: Genetic programming and neuroevolution
- [ ] Zero-copy numpy genomes in the Python package
- [ ] Tree GP, strongly typed
- [ ] Subtree crossover; point, subtree and hoist mutation; bloat control
- [ ] Symbolic regression examples
- [ ] NEAT (speciation, innovation numbers)
- [ ] Neuroevolution with evolution strategies

### 0.9: Frontier
- [ ] Quality-diversity: MAP-Elites, CMA-ME, novelty search
- [ ] LLM-guided evolution (async operators calling a language model)
- [ ] Adaptive operator selection and automatic parameter tuning

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
- **Docs:** every example is a doctest.

## Benchmarks

[`benchmarks/`](benchmarks/) runs every library on the same problems, with identical fitness functions and budgets. Results are published for every release.

- **Problems:** binary (OneMax, LeadingOnes, deceptive trap, NK landscapes, knapsack); permutation (N-Queens, TSPLIB, QAP, flow shop); continuous (BBOB / COCO functions such as Rastrigin, Rosenbrock, Ackley, in 10–100 dimensions); multi-objective (ZDT, DTLZ, WFG).
- **Measurements:** success rate, time and evaluations to target, evaluations per second, instructions per evaluation (Callgrind, (I(2N) − I(N)) / (E(2N) − E(N)): startup cancels out), peak memory, and scaling with population, genome size and threads.
- **Modes:** matched and idiomatic ([methodology](benchmarks/README.md#methodology)).
- **Output:** library versions, JSON, a markdown table and graphs.
- **Libraries:**

| Language | Libraries |
|---|---|
| Python | DEAP, pymoo, PyGAD, EvoX, Nevergrad, pycma, geatpy |
| Java | Jenetics, jMetal, MOEA Framework |
| C++ | pagmo2, openGA, ParadisEO |
| C# | GeneticSharp |
| Julia | Evolutionary.jl, Metaheuristics.jl |
| Rust | radiate, moors, genetic_algorithm, genevo, oxigen |

## Not planned

Gradient-based optimization, Bayesian optimization and general-purpose machine learning: genoxide focuses on evolutionary and population-based methods.
