# Features

What genoxide has, as of 0.6. The [API documentation](https://docs.rs/genoxide) has the details, and [AGENTS.md](../AGENTS.md) has decision tables and templates.

## Properties

| | |
|---|---|
| **Checked settings** | Invalid settings are errors from `build()`, before anything runs. An operator that doesn't fit the genome is a compile error. |
| **Reproducible** | The same seed gives the same result, on any number of threads and on 32- or 64-bit machines. |
| **Safe** | `#![forbid(unsafe_code)]`: memory safety from the compiler, and parallel code without data races. |
| **Fast** | Compiled Rust, bit-packed binary genomes, and parallel or batch evaluation. |

## Genomes

- Binary (bit-packed)
- Integer and real, bounded per gene
- Permutation
- Real with a self-adaptive step size

## Genetic algorithms

- **Selection:** tournament, roulette, stochastic universal sampling, rank, truncation, random.
- **Crossover:**
  - binary, integer and real genomes: one-point, two-point, k-point, uniform
  - real genomes: SBX, blend (BLX-α), arithmetic
  - permutations: order (OX1), partially mapped (PMX), cycle (CX), edge recombination
- **Mutation:**
  - bit-flip, uniform, Gaussian, polynomial, self-adaptive Gaussian
  - permutations: swap, inversion (2-opt), insertion, scramble
- **Schemes:** generational with elitism, steady-state, (μ+λ), (μ,λ), and memetic (Lamarckian local search on the best parents).

## Other single-objective methods

- **Evolution strategies:** (μ/ρ +, λ)-ES with intermediate or dominant recombination. Self-adapted step sizes: one, or one per gene.
- **CMA-ES:** IPOP and BIPOP restarts, and sep-CMA-ES for high dimensions.
- **Differential evolution:** rand/1, best/1, and current-to-pbest/1 with an archive. Fixed, dithered or adaptive parameters (JADE, SHADE, L-SHADE).
- **Particle swarm optimization:** global or ring topology, constriction coefficients, velocity limits.
- **Local search:** hill climbing (first-improvement or best-of-k, with plateau moves), simulated annealing, tabu search, iterated local search. Any mutation serves as the neighborhood.
- **Test problems** (`problems`): Sphere, the axis-parallel ellipsoid, Schwefel 1.2 and 2.26, Rastrigin, Rosenbrock, Ackley, Griewank, Levy, Zakharov, Styblinski-Tang, Michalewicz, Himmelblau, Branin, Goldstein-Price and the six-hump camel, each with its bounds, known optimum and reference, in Rust and Python.

## Multi-objective

- **Algorithms:** NSGA-II, NSGA-III, SPEA2, MOEA/D, SMS-EMOA.
- **With:** constraints, duplicate elimination, a Pareto archive.
- **Indicators:** hypervolume, IGD, IGD+, GD, spread.
- **Test problems:** ZDT and DTLZ.

## Engine

- **Stop conditions:** target, generations, evaluations, time, stagnation, custom.
- **Evaluation:**
  - parallel
  - batch: a whole generation in one call, for SIMD, GPUs or remote services
  - asynchronous: for slow fitness functions whose time varies
- **Island model:** GA or DE islands, with ring, fully connected or random migration.
- **Constraints:** Deb's feasibility rules (a fitness function returns a score and a constraint violation), and penalty functions.
- **Cancellation**, and **checkpoints** to resume a run (`serde` feature).
- **Observers:** statistics per generation, hall of fame, progress lines, `tracing`.

## Beyond Rust

- **Python:** `pip install genoxide`, with numpy genomes and vectorized fitness functions. See [python/README.md](../python/README.md).
- **Command line:** the `genoxide` program runs an optimization described in a TOML or JSON file. The fitness function is any program, in any language, that reads a genome per line and writes its fitness. See [cli.md](cli.md).

  ```text
  cargo install genoxide --features cli
  genoxide run sphere.toml
  ```

## Examples

Each example in [examples/](../examples/) is a folder with the same program in Rust and Python, and a README that cites the problem's source and its known optimum.

```text
cargo run --release --example one_max             # binary genome, GA
cargo run --release --example knapsack            # a constraint with Deb's feasibility rules
cargo run --release --example n_queens            # permutation, (μ+λ)
cargo run --release --example tsp_berlin52        # TSPLIB berlin52, simulated annealing with 2-opt moves
cargo run --release --example jobshop_ft06        # job shop ft06, permutation with repetition
cargo run --release --example rastrigin           # real-valued, CMA-ES with IPOP restarts and L-SHADE
cargo run --release --example function_suite      # CMA-ES, SHADE and PSO on twelve test functions
cargo run --release --example himmelblau          # four global minima, by restarts of a local search
cargo run --release --example pressure_vessel     # constrained mixed discrete-continuous design, SHADE
cargo run --release --example zdt1                # two objectives, NSGA-II, hypervolume
cargo run --release --example dtlz2_3obj          # three objectives, NSGA-III, hypervolume
cargo run --release --example xor_neuroevolution  # a 2-2-1 neural network's weights, CMA-ES
cargo run --release --example asynchronous        # a slow fitness function, asynchronous evaluation
cargo run --release --manifest-path examples/gpu/Cargo.toml  # neuroevolution on the GPU, with wgpu
python examples/tsp_berlin52/main.py              # the same in Python, for all but the last two
```
