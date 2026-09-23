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

### 0.1: Foundations
- **Engine:** generational, steady-state and (μ+λ)/(μ,λ) GA loops on the ask / tell core.
- **Genomes:** bit-packed binary, bounded integers, bounded reals, permutations.
- **Selection:** tournament, roulette, stochastic universal sampling, rank, truncation, random.
- **Crossover:** one-point, two-point, k-point, uniform (exact 50% per gene).
- **Mutation:** bit-flip, uniform, swap.
- **Run control:**
  - termination: target, generations, evaluations, time, stagnation, custom
  - elitism
  - seeded determinism
  - parallel evaluation (rayon, deterministic)
  - cancellation
- **Observers:** statistics collection and hall of fame.
- **Project:** typed errors; docs, examples, AGENTS.md; CI and the benchmark harness from day one.

### 0.2: Real-valued and permutation excellence
- **Real-valued:** SBX, BLX-α, arithmetic crossover; Gaussian (fixed and self-adaptive) and polynomial mutation.
- **Permutations:** PMX, OX1, CX, edge recombination; inversion (2-opt), insertion and scramble mutation.
- **Local search:** hill climbing (first/best improvement), simulated annealing, tabu search, iterated local search.
- **Hybrid:** memetic / Lamarckian hybrid (GA with local search on elites).
- **Constraints:** handling via Deb's feasibility rules and penalty functions.

### 0.3: Evolution strategies and swarm
- **CMA-ES**, with IPOP / BIPOP restarts and sep-CMA-ES for high dimensions.
- **Differential evolution:** rand/1, best/1, current-to-pbest; JADE, SHADE, L-SHADE.
- **Particle swarm:** global and local topologies.
- **(μ/ρ +, λ)-ES** with self-adaptation.

### 0.4: Multi-objective
- **Algorithms:** NSGA-II, NSGA-III, SPEA2, MOEA/D, SMS-EMOA.
- **Pareto archive:** non-dominated sorting (fast and log variants), crowding distance, reference points.
- **Indicators:** hypervolume, IGD / IGD+, spread.
- **Constraints:** constrained dominance.

### 0.5: Scale and operations
- **Island model:** migration topologies (ring, fully connected, random), deterministic and parallel.
- **Evaluation:** asynchronous / steady-state for expensive fitness; batch evaluation hook (SIMD, GPU, remote) with a GPU example.
- **Checkpoint and resume:** `serde` feature.
- **Configuration:** runs described in TOML/JSON, with a small CLI.
- **Observability:** `tracing` integration and progress reporting.

### 0.6: Python
- **Package:** `pip install genoxide` via PyO3 / maturin, with wheels for Linux, macOS and Windows.
- **Fitness:** plain Python functions or vectorized numpy batch fitness; zero-copy numpy genomes.
- **API:** Pythonic builders, and parity examples with DEAP / pymoo tutorials.

### 0.7: Genetic programming and neuroevolution
- **Tree GP:** typed (strongly typed GP), with subtree crossover, point / subtree / hoist mutation and bloat control.
- **Symbolic regression:** examples.
- **Neuroevolution:** NEAT (speciation, innovation numbers), plus an evolution-strategies route for neural networks.

### 0.8: Frontier
- **Quality-diversity:** MAP-Elites, CMA-ME, novelty search.
- **LLM-guided evolution:** async operators that call a language model to propose mutations and crossovers.
- **Tuning:** adaptive operator selection and automatic parameter tuning.

### 1.0: Stable
- **Stability:** API review and stabilization, semver guarantees and an MSRV policy.
- **Benchmarks:** a published benchmark report (see below).
- **Documentation:** a book (mdBook) with a guide per problem type.

## Quality

- **CI:** Linux, macOS and Windows; stable plus MSRV; clippy, fmt and rustdoc with `-D warnings`; a single-thread rayon job.
- **Property-based tests (proptest)** for every operator: validity, e.g. permutations stay permutations; bounds; no no-op; exact rates.
- **Fuzzing** of builders and the configuration file format.
- **Performance:** criterion benchmarks, with regression gating in CI.
- **Docs:** doctests for all examples, including the AI-agent guide.

## Benchmarks

A separate, public benchmark suite runs every library on the same problems, with the same fitness functions and budgets:

- **Binary:** OneMax, LeadingOnes, deceptive trap, NK landscapes, knapsack.
- **Permutation:** N-Queens, TSPLIB, QAP, flow shop.
- **Continuous:** BBOB / COCO functions (Rastrigin, Rosenbrock, Ackley, …) in 10–100 dimensions.
- **Multi-objective:** ZDT, DTLZ, WFG.

**Libraries:** DEAP, pymoo, PyGAD, EvoX, Nevergrad, pycma, Jenetics, pagmo, and in Rust radiate, moors and genetic_algorithm.

**Metrics:**
- success rate
- time and evaluations to target
- quality at a fixed budget
- evaluations per second
- peak memory

## Not planned (for now)

Gradient-based optimization, Bayesian optimization, and general-purpose machine learning. genoxide focuses on evolutionary and population-based methods.
