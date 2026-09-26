# genoxide roadmap

## Vision

A complete, fast and reliable evolutionary computation library, in Rust and for Python.

### Success criteria for 1.0

- **Coverage:** the algorithms and operators of DEAP and pymoo combined, for single- and multi-objective optimization.
- **Speed:** fast on every problem of the public benchmark suite, in time to target and in evaluations per second.
- **Reproducibility:** bit-for-bit identical results for a given seed, on any number of threads.
- **Docs:** 100% of the public API documented, with a runnable example for every algorithm.
- **Python:** a Python package with the same capabilities.

## Design principles

1. **Validated, typed configuration.** Invalid configurations are errors when the builder builds. They never panic or get ignored later. Operators that don't fit a representation don't compile.
2. **Determinism is a guarantee.**
   - Every random decision comes from seeded rng streams: one per run, island and worker.
   - Results never depend on hash map iteration order or thread scheduling.
   - Ties are always broken explicitly.
3. **Ask / tell at the core.** Every algorithm is an `ask() → evaluate → tell()` state machine. The run loop is a thin layer on top. Python bindings, external or async evaluation, custom loops and checkpoints build on it.
4. **No wasted work.** Mutations always change something. Invalid solutions are cached like valid ones. The generation loop doesn't allocate.
5. **Explicit semantics.** Every operator documents what it does, including probabilities and edge cases. Rates are validated to [0, 1].
6. **Batteries included, costs opt-in.** Statistics, hall of fame and checkpoints are built in. You pay only for what you enable.

### Pitfalls ruled out by design

These pitfalls come from a review of existing libraries (e.g. genetic_algorithm, [issues #11 to #78](https://github.com/basvanwesting/genetic-algorithm/issues?q=is%3Aissue+author%3Atachsin)). Each rule has a test.

| Pitfall | genoxide rule |
|---|---|
| Wrong best index when some fitness values are missing | Invalid solutions are a separate state (`Invalid`), never mixed into indices |
| Results depend on `HashMap` iteration order or unstable sorts | Ordered containers; ties broken by index |
| All parallel runs are identical copies under a seed | One derived rng stream per run, island and thread |
| Survivor selection without surplus gives no selection pressure | Parent selection and survival are separate, documented stages |
| Mutations or crossovers that do nothing (self-swap, point 0) | Operators guarantee a change; tested with property tests |
| Off-by-one in elitism | Elitism counts are exact and property-tested |
| Integer overflow at type limits, float steps that don't advance | Checked or saturating arithmetic; progress guarantees in grid search |
| NaN fitness treated as a valid score | NaN is an error or `Invalid`, configurable |
| Deadlocks with a single rayon thread | No blocking waits on pool threads; CI runs with one thread |
| Docs and templates that drift from the code | Doctests for every example, including the AI-agent guide |

## Architecture

- **`Genome`:** the representation. Planned kinds:
  - bits (bit-packed)
  - integers
  - reals (bounded)
  - permutations
  - mixed (per-gene types)
  - trees (GP)
  - graphs (NEAT)
- **`Fitness`:** `f64` with a total ordering, single or multi-objective (`[f64; N]` / `Vec<f64>`), plus an optional constraint violation. Batch and async evaluation hooks.
- **Operators:** `Select`, `Crossover`, `Mutate`, `Repair` and `Survive` traits, parameterized by the genome, so mismatches don't compile.
- **`Algorithm`:** the ask / tell state machines (GA, ES, CMA-ES, DE, PSO, NSGA-II, …).
- **`Engine`:** runs an algorithm with termination, parallel evaluation, observers and cancellation.
- **`Observer`:** statistics, hall of fame or Pareto archive, logging (tracing, CSV, JSON), checkpoints.
- **Errors:** one typed error enum. No `&'static str` errors, no panics in library code.

## Milestones

An item is done when it's implemented, documented, tested (with property tests for operators) and benchmarked where relevant.

### 0.0: Project setup ✅
- [x] Name, repository, dual MIT / Apache-2.0 license
- [x] README (homepage) and this roadmap
- [x] Crate skeleton (edition 2024, MSRV 1.86)

### 0.1: Foundations ✅

#### Project
- [x] CI: build, test, clippy, fmt, rustdoc (`-D warnings`) on Linux, macOS, Windows; MSRV job; single-thread rayon job
- [x] `#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`
- [x] Benchmark suite in [`benchmarks/`](benchmarks/) (DEAP, pymoo, PyGAD, genetic_algorithm)
- [x] Contributing guide, issue and PR templates
- [x] Versioning policy and automated releases (release-plz, cargo-semver-checks, changelog from PR titles)

#### Core types
- [x] Error enum (no panics in library code)
- [x] Rng: portable, seedable generator (same results on every platform) with derived streams per run / island / worker
- [x] Fitness: `f64` with a total ordering, `Invalid` state, NaN policy, maximize / minimize
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

#### Mutation (exact per-gene rates; a picked gene always changes)
- [x] Bit-flip (per gene rate, or exactly n genes)
- [x] Uniform (integer / real)
- [x] Swap (permutation)

#### Engine
- [x] Ask / tell core
- [x] Generational GA, steady-state GA, (μ+λ) and (μ,λ)
- [x] Elitism
- [x] Termination: target, generations, evaluations, time, stagnation, custom, combinations
- [x] Parallel evaluation (rayon), deterministic regardless of thread count
- [x] Cancellation (abort flag)
- [x] Builder with validation (every invalid configuration is an error)

#### Observers
- [x] Statistics per generation (best, mean, stddev, diversity, timings)
- [x] Hall of fame (top-k unique)

#### Docs and examples
- [x] API docs for everything public, with doctests
- [x] Examples: OneMax, knapsack, N-Queens, Rastrigin
- [x] AGENTS.md (guide for AI coding assistants)

#### Performance
- [x] criterion benchmarks for the hot paths
- [x] Instruction count benchmarks in CI (gungraun)
- [x] genoxide adapter in the benchmark suite, first published results

### 0.2: Real-valued and permutation excellence ✅
- [x] Real-valued crossover: SBX, BLX-α, arithmetic
- [x] Real-valued mutation: Gaussian, polynomial
- [x] Self-adaptive Gaussian mutation (a step size per individual)
- [x] Permutation crossover: PMX, OX1, CX, edge recombination
- [x] Permutation mutation: inversion (2-opt), insertion, scramble
- [x] Local search: hill climbing (first improvement, or the best of k random neighbors), simulated annealing
- [x] Local search: tabu search, iterated local search
- [x] Memetic / Lamarckian hybrid (GA with local search on elites)
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
- [x] Island model: ring, fully connected and random topologies; deterministic and parallel
- [x] Asynchronous / steady-state evaluation for expensive fitness
- [x] Batch evaluation hook (SIMD, GPU, remote)
- [x] GPU evaluation example
- [x] Checkpoint and resume (`serde` feature)
- [x] Runs described in TOML / JSON, with a small CLI
- [x] `tracing` integration and progress reporting

### 0.6: Python ✅
- [x] `pip install genoxide` via PyO3 / maturin, wheels for Linux, macOS and Windows
- [x] Python fitness functions and vectorized numpy batch fitness
- [x] Pythonic builders
- [x] Parity examples with the DEAP / pymoo tutorials
- Zero-copy numpy genomes: moved to 0.8.

### 0.7: Correctness
Fixes from the review of 0.6.0 ([#116](https://github.com/tachsin/genoxide/issues/116)). Some change seeded results.
- [x] DE restarts: trials stay visible to observers, constrained problems don't restart every other generation, and migrants aren't evaluated twice
- [x] Duplicate elimination without fingerprint collisions, with the same results on 32-bit and 64-bit
- [x] Runs that could never end stop as stalled
- [x] Huge sizes are setting errors instead of panics or hangs
- [x] The steady-state GA doesn't propose twins, and PSO's initial velocities respect `max_velocity`
- [x] Clearer errors and stricter settings in the Python package, with license files in the wheels
- [x] Benchmark adapters with numpy fitness in pymoo and PyGAD, and timings on pinned P-cores
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
- [ ] LLM-guided evolution: async operators that call a language model
- [ ] Adaptive operator selection and automatic parameter tuning

### 1.0: Stable
- [ ] API review and stabilization, semver guarantees, MSRV policy
- [ ] Published benchmark report
- [ ] Book (mdBook) with a guide per problem type

## Quality

- **CI:** Linux, macOS and Windows; stable and MSRV; clippy, fmt and rustdoc with `-D warnings`; a single-thread rayon job.
- **Property-based tests (proptest)** for every operator:
  - validity (e.g. permutations stay permutations);
  - bounds;
  - no no-op (a picked gene changes, a count mutation changes the genome);
  - exact rates.
- **Fuzzing** of builders and the configuration file format: planned.
- **Performance:** every hot path has benchmarks:
  - criterion, for wall time;
  - gungraun (formerly iai-callgrind), for exact instruction counts. They are noise-free and fail CI on regressions.
- **Safety:** `#![forbid(unsafe_code)]` by default. Any `unsafe` for SIMD or bit tricks goes behind a feature, with a `SAFETY` comment and Miri tests.
- **Docs:** doctests for all examples, including the AI-agent guide.

## Benchmarks

The benchmark suite ([`benchmarks/`](benchmarks/)) runs every library on the same problems, with identical fitness functions and evaluation budgets. Results are published for every release.

### Problems

- **Binary:** OneMax, LeadingOnes, deceptive trap, NK landscapes, knapsack.
- **Permutation:** N-Queens, TSPLIB, QAP, flow shop.
- **Continuous:** BBOB / COCO functions (Rastrigin, Rosenbrock, Ackley, …) in 10–100 dimensions.
- **Multi-objective:** ZDT, DTLZ, WFG.

### Libraries

| Language | Libraries |
|---|---|
| Python | DEAP, pymoo, PyGAD, EvoX, Nevergrad, pycma, geatpy |
| Java | Jenetics, jMetal, MOEA Framework |
| C++ | pagmo2, openGA, ParadisEO |
| C# | GeneticSharp |
| Julia | Evolutionary.jl, Metaheuristics.jl |
| Rust | radiate, moors, genetic_algorithm, genevo, oxigen |

### Measurements

- **Success rate and time to target:** the user-facing result.
- **Evaluations to target:** search efficiency, independent of language.
- **Evaluations per second:** framework throughput.
- **Instructions per evaluation:** measured with Callgrind as (I(2N) − I(N)) / (E(2N) − E(N)). N and 2N are evaluation budgets; E counts the evaluations actually made. Startup and setup cancel out, so the count is exact and repeatable across languages.
- **Peak memory**, and scaling with population size, genome size and threads (parallel speedup).

### Two modes

- **matched:** configurations as equal as the libraries allow, to measure framework cost.
- **idiomatic:** each library's recommended setup, to measure what users get.

### Output

Every run records the library versions. It writes:
- raw JSON
- a markdown table
- graphs: time to target on a log scale, evaluations per second, instructions per evaluation, success rate, convergence curves

## Not planned

Gradient-based optimization, Bayesian optimization and general-purpose machine learning. genoxide focuses on evolutionary and population-based methods.
