# Features

What genoxide has on main; [docs.rs](https://docs.rs/genoxide) documents the latest release. The [API documentation](https://docs.rs/genoxide) has the details, and [AGENTS.md](../AGENTS.md) has decision tables and templates.

## Properties

| | |
|---|---|
| **Checked settings** | Invalid settings are errors from `build()`, before anything runs. An operator that doesn't fit the genome is a compile error. |
| **Reproducible** | The same seed gives the same result, on any number of threads, on 32- or 64-bit machines and on every operating system (with `genoxide::math` in place of `f64::sin` and the like in a fitness function). |
| **Safe** | `#![forbid(unsafe_code)]`: memory safety from the compiler, and parallel code without data races. |
| **Fast** | Compiled Rust, bit-packed binary genomes, parallel or batch evaluation, and parallel breeding. |

## Genomes

- Binary (bit-packed)
- Integer and real, bounded per gene
- Permutation
- Real with a self-adaptive step size
- Trees of a genetic program, strongly typed

## Genetic algorithms

- **Selection:** tournament, roulette, stochastic universal sampling, rank, truncation, random; against bloat in variable-size genomes, lexicographic parsimony, double tournament and Tarpeian.
- **Crossover:**
  - binary, integer and real genomes: one-point, two-point, k-point, uniform
  - real genomes: SBX, blend (BLX-α), arithmetic
  - permutations: order (OX1), partially mapped (PMX), cycle (CX), edge recombination
- **Mutation:**
  - bit-flip, uniform, Gaussian, polynomial, self-adaptive Gaussian
  - permutations: swap, inversion (2-opt), insertion, scramble
- **Schemes:** generational with elitism, steady-state, (μ+λ), (μ,λ), and memetic (Lamarckian local search on the best parents).
- **Parameter control:** a GA's rates and operators can be changed between generations, e.g. to anneal the mutation step or raise it when the search stagnates.
- **Parallel breeding:** crossover and mutation on all cores, each pair of parents on a random stream of its own, so a seeded run gives the same results on any number of threads. For large populations and operators that do real work per gene, when evaluation is fast. Differential evolution's trials and an evolution strategy's offspring can be made in parallel the same way.

## Genetic programming

- **Trees** (`gp`): programs and formulas as genomes, made of your own primitives (an enum) with typed arguments and results (Montana's strongly typed GP), terminals and ephemeral random constants (uniform, integers, a choice of values, or normal, as Keijzer's). Stored as flat arrays in prefix order, so copying, comparing, hashing and saving a tree never recurse; read and written as text, e.g. `add(x, mul(x, 0.5))`.
- **Initialization:** full, grow and ramped half-and-half (Koza's default, depths 2 to 6, with duplicates redrawn), always within the limits: Koza's depth of 17 and a size of 1024 nodes.
- **Operators:** subtree crossover with Koza's 90% bias to function nodes, whose second point keeps both children within the limits, and one-point crossover; subtree, point, hoist, shrink and constant mutation, alone or mixed by weight. A picked node always changes.
- **Bloat control:** Koza's depth limit and a size limit, and selections that favor smaller trees: lexicographic parsimony pressure, double tournament and Tarpeian.
- **Symbolic regression** (`gp::regression`): mathematical primitives (arithmetic, the inverse, the analytic quotient, Koza's protected division, logarithm and square root, `sin`, `exp` and the like, the same bits on every platform), training and test samples, and a fitness function that evaluates a tree on all the points at once in a workspace per thread: the RMSE, MSE or MAE, after Keijzer's linear scaling by default. A tree that isn't finite at a point is invalid.
- **Problems:** Koza's Boolean multiplexer and even-parity functions, and Koza's regression problems (the quartic and two more polynomials) and Nguyen-1 to 12 (polynomials, sines and cosines, logarithms, a square root, and four of two variables), each with its paper's function set and sampling.
- **Evaluation:** a stack machine for values of any type, the same on columns of all the data's points at once (several times faster, the same results to the bit), and a top-down walk for interpreters.
- Runs on the genetic algorithms, islands and the multi-objective algorithms, with checkpoints and parallel breeding.

## Neuroevolution

- **Networks** (`nn`): a multilayer perceptron and an Elman recurrent network whose weights are a `Real` genome, for CMA-ES or any real-valued method; tanh, logistic, ReLU and linear units, with or without biases; forward passes without allocation, the same bits on every platform.
- **Control tasks** (`problems::control`): the cart-pole and the double pole, with and without velocities, with Florian's corrected equations and the settings of Gomez et al. (2008), integrated by Runge-Kutta with portable `sin` and `cos`; their success criteria, and Gruau et al.'s damping fitness and generalization test. Driven by a `Policy`: a network or a closure.
- **NEAT** (`neat`): networks whose structure evolves with their weights (Stanley and Miikkulainen 2002), from minimal networks up: innovation numbers that align genes in crossover, speciation by compatibility distance, explicit fitness sharing (normalized for any objective, or the paper's raw form), species champions and stagnation, and new nodes and connections. The paper's settings by default; feed-forward and recurrent networks compiled once for evaluation without allocation, and usable as control policies. Seeded runs and checkpoints are reproducible.
- CMA-ES's step size can be bounded below (Igel 2003), for fitness functions that stop pointing at the goal near their best.

## Other single-objective methods

- **Evolution strategies:** (μ/ρ +, λ)-ES with intermediate or dominant recombination. Self-adapted step sizes: one, or one per gene.
- **CMA-ES:** IPOP and BIPOP restarts, and sep-CMA-ES for high dimensions.
- **Differential evolution:** rand/1, best/1, and current-to-pbest/1 with an archive. Fixed, dithered or adaptive parameters (JADE, SHADE, L-SHADE).
- **Particle swarm optimization:** global or ring topology, constriction coefficients, velocity limits.
- **Local search:** hill climbing (first-improvement or best-of-k, with plateau moves), simulated annealing, tabu search, iterated local search. Any mutation serves as the neighborhood.
- **Test problems** (`problems`): Sphere, the axis-parallel ellipsoid, Schwefel 1.2 and 2.26, Rastrigin, Rosenbrock, Ackley, Griewank, Levy, Zakharov, Styblinski-Tang, Michalewicz, Himmelblau, Branin, Goldstein-Price, the six-hump camel, Hartmann's functions in 3 and 6 dimensions, Shekel's with 5, 7 and 10 wells, Easom, the eggholder and Schaffer's F6, each with its bounds, known optimum (or best known, for those found numerically) and reference, in Rust and Python.
- **Constrained test problems:** CEC 2006's g01-g24 (`problems::cec2006`), and the engineering design problems (`problems::engineering`): the welded beam in two forms, the pressure vessel, the tension/compression spring, the speed reducer, the gear train (integer), the three-bar truss, the cantilever beam and the car side impact, each with its optimum or best known solution and references, in Rust and Python.

## Multi-objective

- **Algorithms:** NSGA-II, NSGA-III, SPEA2, MOEA/D, SMS-EMOA.
- **With:** constraints, duplicate elimination, a Pareto archive.
- **Indicators:** hypervolume, IGD, IGD+, GD, spread.
- **Test problems** (`multi::problems`): ZDT1-6 (ZDT5 on bit strings), DTLZ1-7, WFG1-9 (any number of objectives, checked against the authors' toolkit), Schaffer's two, Fonseca and Fleming's, Kursawe's, Poloni's and Viennet's three, and the constrained BNH, SRN, TNK, OSY and CONSTR, each with its optimal front where it's known and its reference, in Rust and Python.
- **Constrained test problems of tunable difficulty** (`multi::problems`): CTP1-8 (Deb, Pratap and Meyarivan), whose constraints make the front disconnected, a set of points or hidden behind infeasible bands, and Jain and Deb's constrained DTLZ problems C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, convex C2-DTLZ2, C3-DTLZ1 and C3-DTLZ4 for any number of objectives, with their fronts, in Rust and Python.
- **More test problems** (`multi::problems`): Deb and Jain's convex DTLZ2, scaled DTLZ1 and DTLZ2 and inverted DTLZ1, and Ma and Wang's constrained MW1-14, with fronts derived from their definitions, in Rust and Python.
- **Engineering design problems with several objectives** (`multi::problems::engineering`): the two-bar and four-bar trusses, the welded beam, the disc brake and the speed reducer (two objectives), the car side impact, the rocket injector and vehicle crashworthiness (three) and water resource planning (five), the trusses' and water resource planning's fronts derived from their definitions, in Rust and Python.

## Engine

- **Stop conditions:** target, generations, evaluations, time, stagnation, custom.
- **Evaluation:**
  - parallel
  - batch: a whole generation in one call, for SIMD, GPUs or remote services
  - asynchronous: for slow fitness functions whose time varies
- **Island model:** GA or DE islands, with ring, fully connected or random migration, or isolated islands that never migrate (e.g. different settings side by side, or independent starts) run as one algorithm.
- **Parameter control:** change an algorithm's rates, operators and coefficients between generations, e.g. an annealed mutation step or a decreasing inertia weight, from the engine's `control` hook (in Python, `run(..., control=...)`); each island on a schedule of its own.
- **A fitness function that changes during a run:** the algorithms re-evaluate what they keep, e.g. after adapting penalty weights, without comparing old and new values, in Rust and Python.
- **Extras with the fitness:** a fitness function returns what it computed along with the fitness (`Evaluated`), e.g. the terms of a penalty or a secondary measure; the engines keep it for the population and the best, for observers, the hall of fame and the outcome, without evaluating again and without changing the search.
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
cargo run --release --example himmelblau          # four global minima, by restarts of a local search
cargo run --release --example pressure_vessel     # constrained mixed discrete-continuous design, SHADE
cargo run --release --example welded_beam         # constrained design in two forms, SHADE
cargo run --release --example gear_train          # integer genome, genetic algorithm
cargo run --release --example zdt1                # two objectives, NSGA-II, hypervolume
cargo run --release --example bnh                 # two objectives and two constraints, NSGA-II, IGD+
cargo run --release --example kursawe             # a disconnected front, SPEA2 and NSGA-II
cargo run --release --example dtlz2_3obj          # three objectives, NSGA-III, hypervolume
cargo run --release --example xor_neuroevolution  # a 2-2-1 neural network's weights, CMA-ES
cargo run --release --example xor_neat            # NEAT evolves a network's structure and weights for XOR
cargo run --release --example koza_quartic        # genetic programming finds x^4 + x^3 + x^2 + x exactly
cargo run --release --example double_pole         # a network balances two poles for 100,000 steps, CMA-ES
cargo run --release --example double_pole_no_velocities  # the same with a recurrent network, no velocities
cargo run --release --example multiplexer_11      # Koza's 11-multiplexer, all 2048 cases, double tournament
cargo run --release --example abs_typed           # strongly typed GP: |x| from a comparison and a conditional
cargo run --release --example nguyen_all          # Nguyen's twelve regression problems, with and without linear scaling
cargo run --release --example accuracy_and_size   # NSGA-II on trees: the front of error against size
cargo run --release --example asynchronous        # a slow fitness function, asynchronous evaluation
cargo run --release --manifest-path examples/gpu/Cargo.toml  # neuroevolution on the GPU, with wgpu
python examples/tsp_berlin52/main.py              # the same in Python, for all but the last three
```
