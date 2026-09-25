# genoxide

[![Crates.io](https://img.shields.io/crates/v/genoxide.svg)](https://crates.io/crates/genoxide)
[![Docs.rs](https://img.shields.io/docsrs/genoxide)](https://docs.rs/genoxide)
[![CI](https://github.com/tachsin/genoxide/actions/workflows/ci.yml/badge.svg)](https://github.com/tachsin/genoxide/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/crates/d/genoxide.svg)](https://crates.io/crates/genoxide)
[![MSRV](https://img.shields.io/crates/msrv/genoxide)](Cargo.toml)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](src/lib.rs)
[![License](https://img.shields.io/crates/l/genoxide.svg)](#license)

**Evolutionary computation for Rust: genetic algorithms, evolution strategies, differential evolution, particle swarms, local search and multi-objective optimization in one library.**

> **Alpha, pre-1.0:** the API changes between 0.x versions. See the [roadmap](ROADMAP.md), and share ideas in the issues.

```toml
[dependencies]
genoxide = "0.6"
```

## Benchmarks

genoxide and its Python package against 15 libraries: genetic_algorithm, radiate and moors (Rust), openGA and pygmo (C++), DEAP, pymoo, PyGAD, pycma, Nevergrad and SciPy (Python), Jenetics and jMetal (Java), Evolutionary.jl and Metaheuristics.jl (Julia). One thread on an Intel Core Ultra 7 265K, median of 10 runs.

![Time to target](docs/benchmarks/time_to_target.svg)

- **Fastest in 7 of the 9 single-objective problems:**

  | Problem | genoxide | Next fastest | DEAP |
  |---|---|---|---|
  | OneMax 1000 | 15 ms | 59 ms (Evolutionary.jl) | 15.8 s |
  | Rastrigin 30 | 15 ms | 70 ms (pygmo's SaDE) | 1.25 s (its CMA-ES, 8 of 10 runs) |
  | Rosenbrock 10 | 1.6 ms | 12 ms (Metaheuristics.jl's ECA) | 107 ms (its CMA-ES) |
  | Ackley 30 | 6.3 ms | 30 ms (Evolutionary.jl's ES) | 0.7 s (its CMA-ES) |

  On OneMax 100, the GAs of Evolutionary.jl and genetic_algorithm are slightly faster.
- **The cheapest evaluation:** 3,400 CPU instructions per OneMax 1000 evaluation, fitness function included. That's 2.6 times fewer than genetic_algorithm, 10 to 19 times fewer than moors, openGA and radiate, and 90 to 1,300 times fewer than the Python libraries.
- **Not always the fewest evaluations:**
  - Rastrigin 30: pygmo's SaDE needs 15,570, against genoxide's 28,280.
  - Rosenbrock: the CMA-ES of pycma and pymoo need 4,491 and 4,676, against 5,945.
  - OneMax 100: pygmo's GA needs 1,610, against 3,183.

  With an expensive fitness function, the number of evaluations is what counts.
- **From Python,** with the fitness function in Python: Rastrigin 30 in 18 ms and OneMax 1000 in 81 ms, where DEAP takes 15.8 s.
- **Multi-objective:** genoxide's SMS-EMOA gets within 0.2% of the best hypervolume on all five problems. Its NSGA-II takes 25 to 34 ms per run, against 0.55 to 0.87 s for pymoo and 1.3 to 2.1 s for DEAP.

Every library solves the same problems with the same evaluation budget, and time counts the optimization only. Time to target is the number of evaluations times the cost of one evaluation. With a cheap fitness function, the library's own cost dominates; with an expensive one, only the evaluations matter.

The methods, settings and bugs found per library are in [benchmarks/README.md](benchmarks/README.md), and the full numbers are in [docs/benchmarks/results.md](docs/benchmarks/results.md).

<details>
<summary>More charts: evaluations to target, instructions per evaluation, multi-objective runs</summary>

![Evaluations to target](docs/benchmarks/evaluations_to_target.svg)

![CPU instructions per evaluation](docs/benchmarks/instructions.svg)

![Hypervolume of the final fronts](docs/benchmarks/hypervolume.svg)

![Time of multi-objective runs](docs/benchmarks/front_time.svg)

</details>

## Why genoxide?

| | What it means |
|---|---|
| **Complete** | From a simple GA to NSGA-III, CMA-ES with restarts, L-SHADE and island models, see [what's there](#whats-there-so-far) |
| **Fast** | Compiled Rust, bit-packed binary genomes, and parallel or batch evaluation; see the [benchmarks](#benchmarks) |
| **Correct** | Invalid settings are errors from `build()`, before anything runs. An operator that doesn't fit the genome is a compile error |
| **Reproducible** | The same seed gives the same result, on any number of threads and on 32- or 64-bit machines |
| **Safe** | `#![forbid(unsafe_code)]`: memory safety from the compiler, and parallel code without data races |
| **Also for Python** | `pip install genoxide`, with numpy genomes and vectorized fitness functions; see [`python/`](python/) |
| **Batteries included** | Statistics, hall of fame, progress reports, constraints, checkpoints to resume a run, cancellation, `tracing` |

## A first look

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    // OneMax: find the 100-bit string with the most ones
    let ga = Ga::builder(Binary::new(100)?)
        .population_size(100)
        .select(Tournament::new(3)?)
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(0.01)?)
        .seed(42)
        .build()?;

    let outcome = Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
        .stop_when(Stop::target(100.0).or(Stop::generations(1_000)))
        .run()?;

    println!("best: {} after {} generations", outcome.best_fitness(), outcome.generations());
    Ok(())
}
```

### What's there so far

- **Genomes:** binary (bit-packed), integer and real (bounded per gene), permutation, and real with a self-adaptive step size
- **Selection:** tournament, roulette, stochastic universal sampling, rank, truncation, random
- **Crossover:** one-point, two-point, k-point, uniform; for real genomes SBX, blend (BLX-α), arithmetic; for permutations order (OX1), partially mapped (PMX), cycle (CX), edge recombination
- **Mutation:** bit-flip, uniform, Gaussian, polynomial, self-adaptive Gaussian; for permutations swap, inversion (2-opt), insertion, scramble
- **Schemes:** generational with elitism, steady-state, (μ+λ), (μ,λ), and memetic (Lamarckian local search on the best parents)
- **Evolution strategies:** (μ/ρ +, λ)-ES with intermediate or dominant recombination and self-adapted step sizes (one, or one per gene)
- **CMA-ES:** with IPOP and BIPOP restarts, and sep-CMA-ES for high dimensions
- **Differential evolution:** rand/1, best/1 and current-to-pbest/1 with an archive; fixed, dithered or adaptive parameters (JADE, SHADE, L-SHADE)
- **Particle swarm optimization:** global or ring topology, constriction coefficients, velocity limits
- **Multi-objective:** NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA, with constraints, duplicate elimination and a Pareto archive; indicators: hypervolume, IGD, IGD+, GD and spread; the ZDT and DTLZ test problems
- **Asynchronous evaluation:** for slow fitness functions whose time varies
- **Island model:** GA or DE islands with ring, fully connected or random migration
- **Local search:** hill climbing (first-improvement or best-of-k, with plateau moves), simulated annealing, tabu search and iterated local search, with any mutation as the neighborhood
- **Constraints:** Deb's feasibility rules (a fitness function returns a score and a constraint violation), and penalty functions
- **Engine:** stop conditions (target, generations, evaluations, time, stagnation, custom), parallel evaluation, batch evaluation (a whole generation in one call, for SIMD, GPUs or remote services), cancellation, checkpoints to resume a run (`serde` feature)
- **Observers:** statistics per generation, hall of fame, progress lines, `tracing`

### Examples

```text
cargo run --release --example one_max     # binary, statistics
cargo run --release --example knapsack    # a constraint with Deb's feasibility rules, hall of fame
cargo run --release --example n_queens    # permutation, (μ+λ)
cargo run --release --example rastrigin   # real-valued, parallel evaluation
cargo run --release --example asynchronous # a slow fitness function, asynchronous evaluation
cargo run --release --manifest-path examples/gpu/Cargo.toml  # neuroevolution on the GPU, with wgpu
```

### Without writing Rust

The `genoxide` program runs an optimization described in a TOML or JSON file. The fitness function is any program, in any language, that reads a genome per line and writes its fitness:

```text
cargo install genoxide --features cli
genoxide run sphere.toml
```

See [docs/cli.md](docs/cli.md) for the run file and the protocol.

### From Python

The Python package runs genoxide's algorithms with fitness functions in Python and numpy, one genome at a time or a whole generation in one call. See [`python/`](python/).

```python
import genoxide as gx

ga = gx.Ga(
    gx.Binary(100),
    population_size=100,
    select=gx.Tournament(3),
    crossover=gx.UniformCrossover(),
    mutation=gx.BitFlip(rate=0.01),
    seed=42,
)
result = ga.run(lambda bits: bits.sum(), target=100, generations=1_000)
```

```sh
pip install genoxide
```

Wheels for Linux, macOS and Windows, CPython 3.10 and later.

Using an AI coding assistant? Point it to [AGENTS.md](AGENTS.md): it has the decision tables, settings, templates and fixes for common errors.

## Status

| Milestone | Scope | Status |
|---|---|---|
| 0.1 Foundations | Core engine, representations, classic operators, statistics | ✅ released |
| 0.2 Real-valued & permutations | SBX, polynomial, PMX, OX, 2-opt, local search, memetic | ✅ released |
| 0.3 Evolution strategies & swarm | CMA-ES, DE (JADE, SHADE), PSO, (μ,λ) and (μ+λ)-ES | ✅ released |
| 0.4 Multi-objective | NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA, hypervolume | ✅ released |
| 0.5 Scale | Island model, checkpointing, batch/GPU and asynchronous evaluation, CLI | ✅ released |
| 0.6 Python | PyO3 bindings with numpy support | ✅ released |
| 0.7 Correctness | Fixes from an independent review of 0.6 | 🔜 next |
| 0.8 GP & neuroevolution | Typed tree GP, NEAT | planned |
| 0.9 Frontier | Quality-diversity (MAP-Elites), LLM-guided operators | planned |
| 1.0 | Stable API and published benchmark report | planned |

The details are in [ROADMAP.md](ROADMAP.md).

## Contributing

Open an issue for ideas, use cases or API feedback. See [CONTRIBUTING.md](CONTRIBUTING.md) for pull requests.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in genoxide by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
