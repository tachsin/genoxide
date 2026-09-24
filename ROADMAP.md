# genoxide roadmap

## Vision

The most complete, fastest and most reliable evolutionary computation library, in Rust and for Python.

### Success criteria for 1.0

- **Coverage:** the algorithm and operator coverage of DEAP and pymoo combined, for single- and multi-objective optimization.
- **Speed:** the fastest library in our public benchmark suite on every problem, measured both in time to target and in evaluations per second.
- **Reproducibility:** bit-for-bit identical results for a given seed, on any number of threads.
- **Docs:** 100% of the public API documented, with a runnable example for every algorithm.
- **Python:** a Python package with the same capabilities.

## Design principles

1. **Validated, typed configuration.** Invalid configurations are errors when the builder builds, never panics or silently ignored settings later. Operators that don't fit a representation are compile errors.
2. **Determinism is a guarantee, not a best effort.**
   - Every random decision comes from seeded rng streams, one stream per run, island and worker.
   - Results never depend on hash map iteration order or thread scheduling.
   - Ties are always broken explicitly.
3. **Ask / tell at the core.** Every algorithm is an `ask() → evaluate → tell()` state machine, and the run loop is a thin layer on top. This makes Python bindings, external or async evaluation, custom loops and checkpointing natural.
4. **No work wasted.** Mutations always change something, invalid solutions are cached like valid ones, and there are no allocations in the generation loop.
5. **Explicit semantics.** Every operator documents exactly what it does, including probabilities and edge cases. Rates are validated to [0, 1].
6. **Batteries included, costs opt-in.** Statistics, hall of fame and checkpointing are built in, but you only pay for what you enable.

### Lessons from existing libraries

A review of existing libraries found bugs that genoxide must rule out by design (for example in genetic_algorithm, [issues #11 to #78](https://github.com/basvanwesting/genetic-algorithm/issues?q=is%3Aissue+author%3Atachsin)). Each one gets a test:

| Pitfall | genoxide rule |
|---|---|
| Wrong best index when some fitness values are missing | Invalid solutions are a separate state (`Invalid`), never mixed into indices |
| Results depend on `HashMap` iteration order or unstable sorts | Ordered containers; ties broken by index |
| All parallel runs are identical copies under a seed | One derived rng stream per run, island and thread |
| Survivor selection without surplus gives no selection pressure | Parent selection and survival are separate, documented stages |
| Mutations or crossovers that do nothing (self-swap, point 0) | Operators guarantee a change; tested with property tests |
| Off-by-one in elitism | Elitism counts are exact and property-tested |
| Integer overflow at type limits, float steps that don't advance | Checked or saturating arithmetic; progress guarantees in grid search |
| NaN fitness silently treated as a valid score | NaN is an error or `Invalid`, configurable |
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
- **`Fitness`:** `f64` with a total ordering, single or multi-objective (`[f64; N]` / `Vec<f64>`), plus an optional constraint violation. Batch evaluation and async evaluation hooks.
- **Operators:** `Select`, `Crossover`, `Mutate`, `Repair` and `Survive` traits, parameterized by the genome, so mismatches don't compile.
- **`Algorithm`:** the ask / tell state machines (GA, ES, CMA-ES, DE, PSO, NSGA-II, …).
- **`Engine`:** runs an algorithm with termination, parallel evaluation, observers and cancellation.
- **`Observer`:** statistics, hall of fame or Pareto archive, logging (tracing, CSV, JSON), checkpointing.
- **Errors:** one typed error enum, no `&'static str` errors and no panics in library code.

## Milestones

Progress is tracked with the checklists below. Every item is done when it's implemented, documented, tested (including property tests for operators) and covered by the benchmarks where relevant.

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

#### Mutation (always changes the genome)
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
- [x] Instruction count benchmarks in CI (gungraun, formerly iai-callgrind)
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

### 0.3: Evolution strategies and swarm
- [x] CMA-ES, with IPOP / BIPOP restarts
- [x] sep-CMA-ES for high dimensions
- [x] Differential evolution: rand/1, best/1, current-to-pbest
- [x] Adaptive DE: JADE, SHADE, L-SHADE
- [x] Particle swarm: global and local topologies
- [x] (μ/ρ +, λ)-ES with self-adaptation

### 0.4: Multi-objective
- [ ] Non-dominated sorting (fast and log variants), crowding distance
- [ ] NSGA-II
- [ ] NSGA-III (reference points)
- [ ] SPEA2
- [ ] MOEA/D
- [ ] SMS-EMOA
- [ ] Pareto archive
- [ ] Indicators: hypervolume, IGD / IGD+, spread
- [ ] Constrained dominance

### 0.5: Scale and operations
- [ ] Island model: ring, fully connected and random topologies; deterministic and parallel
- [ ] Asynchronous / steady-state evaluation for expensive fitness
- [ ] Batch evaluation hook (SIMD, GPU, remote), with a GPU example
- [ ] Checkpoint and resume (`serde` feature)
- [ ] Runs described in TOML / JSON, with a small CLI
- [ ] `tracing` integration and progress reporting

### 0.6: Python
- [ ] `pip install genoxide` via PyO3 / maturin, wheels for Linux, macOS and Windows
- [ ] Python fitness functions and vectorized numpy batch fitness
- [ ] Zero-copy numpy genomes
- [ ] Pythonic builders
- [ ] Parity examples with the DEAP / pymoo tutorials

### 0.7: Genetic programming and neuroevolution
- [ ] Tree GP, strongly typed
- [ ] Subtree crossover; point, subtree and hoist mutation; bloat control
- [ ] Symbolic regression examples
- [ ] NEAT (speciation, innovation numbers)
- [ ] Neuroevolution with evolution strategies

### 0.8: Frontier
- [ ] Quality-diversity: MAP-Elites, CMA-ME, novelty search
- [ ] LLM-guided evolution: async operators that call a language model
- [ ] Adaptive operator selection and automatic parameter tuning

### 1.0: Stable
- [ ] API review and stabilization, semver guarantees, MSRV policy
- [ ] Published benchmark report
- [ ] Book (mdBook) with a guide per problem type

## Quality

- **CI:** Linux, macOS and Windows; stable plus MSRV; clippy, fmt and rustdoc with `-D warnings`; a single-thread rayon job.
- **Property-based tests (proptest)** for every operator: validity, e.g. permutations stay permutations; bounds; no no-op; exact rates.
- **Fuzzing** of builders and the configuration file format.
- **Performance:** from 0.1 on, every hot path has benchmarks, in two forms:
  - criterion benchmarks for wall time
  - gungraun (formerly iai-callgrind) for exact instruction counts, which are noise-free and fail CI on regressions
- **Safety:** `#![forbid(unsafe_code)]` by default. Any `unsafe` for SIMD or bit tricks goes behind a feature, with a `SAFETY` comment and Miri tests.
- **Docs:** doctests for all examples, including the AI-agent guide.

## Benchmarks

The benchmark suite ([`benchmarks/`](benchmarks/)) runs every library on the same problems, with identical fitness functions and evaluation budgets. It is the main way to show Rust's advantages, so it is published and kept up to date for every release.

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
- **Instructions per evaluation:** measured with Callgrind as (I(2N) − I(N)) / N, so interpreter startup and setup cancel out. Exact and repeatable across languages.
- **Peak memory**, and scaling with population size, genome size and threads (parallel speedup).

### Two modes

- **matched:** configurations as equal as the libraries allow, to measure framework cost.
- **idiomatic:** each library's recommended setup, to measure what users actually get.

### Output

Every run records the library versions. It writes:
- raw JSON
- a markdown table
- graphs: time to target on a log scale, evaluations per second, instructions per evaluation, success rate, convergence curves

## Not planned (for now)

Gradient-based optimization, Bayesian optimization, and general-purpose machine learning. genoxide focuses on evolutionary and population-based methods.
