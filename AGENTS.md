# genoxide guide for AI coding assistants

A reference for writing code that uses genoxide. Every Rust block is a complete program, compiled and run in CI.

genoxide is pre-1.0: the API changes between 0.x versions. Check the version in `Cargo.toml` and the [API docs](https://docs.rs/genoxide).

## The shape of every program

1. Pick a **representation**, the space of solutions: e.g. `Binary::new(100)?`.
2. Build a **`Ga`** with `Ga::builder(representation)`: population size, selection, crossover, mutation.
3. Run it with an **`Engine`**: a fitness function and at least one stop condition.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    let ga = Ga::builder(Binary::new(64)?)
        .population_size(100)
        .select(Tournament::new(3)?)
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / 64.0)?)
        .seed(42) // omit for a random seed; `ga.seed()` reports it
        .build()?;

    let outcome = Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
        .stop_when(Stop::target(64.0).or(Stop::generations(1_000)))
        .run()?;

    println!("{} after {} generations", outcome.best_fitness(), outcome.generations());
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

`use genoxide::prelude::*;` imports everything in this guide.

## Choosing the pieces

| Problem | Representation | Genome | Crossover | Mutation |
|---|---|---|---|---|
| Yes / no decisions (subset, knapsack, feature selection) | `Binary::new(len)` | `Bits` | `UniformCrossover`, `PointCrossover` | `BitFlip` |
| Whole numbers in ranges (counts, choices, schedules) | `Integer::new([lo..=hi, ...])`, `Integer::uniform(len, lo..=hi)` | `Integers` (derefs to `[i64]`) | `UniformCrossover`, `PointCrossover` | `UniformMutation` |
| Real numbers in ranges (parameters, continuous functions) | `Real::new([lo..=hi, ...])`, `Real::uniform(len, lo..=hi)` | `Reals` (derefs to `[f64]`) | `SimulatedBinaryCrossover` (η 15), `BlendCrossover` (α 0.5), `ArithmeticCrossover`, `UniformCrossover`, `PointCrossover` | `PolynomialMutation` (η 20), `GaussianMutation`, `UniformMutation` |
| Real numbers that need fine-tuning (an adaptive step size) | `AdaptiveReal::new(Real::..., initial_step)` | `AdaptiveReals` (derefs to `[f64]`; `.step()`) | `NoCrossover` for an ES, or `UniformCrossover`, `PointCrossover` | `SelfAdaptiveMutation` |
| An order of `0..n` (tours, sequencing, assignment) | `Permutation::new(n)` | `Order` (derefs to `[usize]`) | `OrderCrossover` (sequences), `EdgeRecombinationCrossover` (tours), `PartiallyMappedCrossover`, `CycleCrossover` | `InversionMutation` (tours), `SwapMutation`, `InsertionMutation`, `ScrambleMutation` |

Selection works with every representation. The usual choice is `Tournament::new(size)?` with a size of 2 to 5.

| Scheme (`.scheme(...)`) | When |
|---|---|
| `Scheme::Generational { elitism: 1 }` (default) | General purpose |
| `Scheme::SteadyState { replacements: k }` | Gradual change, replacing the `k` worst each generation |
| `Scheme::MuPlusLambda { lambda }` | Strong elitism; mutation-only search (with `NoCrossover`); plateaus |
| `Scheme::MuCommaLambda { lambda }` | Parents never survive; `lambda` ≥ population size |

## Settings

### `Ga::builder(representation)`

| Method | Required | Default | Valid |
|---|---|---|---|
| `.population_size(n)` | yes | none | ≥ 1 |
| `.select(s)` | yes | none | any `Select` |
| `.crossover(c)` | yes | none | a `Crossover` for the representation |
| `.mutate(m)` | yes | none | a `Mutate` for the representation |
| `.maximize()` / `.minimize()` / `.objective(o)` | no | maximize | |
| `.crossover_rate(p)` | no | 0.9 | 0 ≤ p ≤ 1 |
| `.mutation_rate(p)` | no | 1.0 | 0 ≤ p ≤ 1, not both rates 0 |
| `.scheme(s)` | no | generational, elitism 1 | elitism < population size; 1 ≤ replacements ≤ population size; lambda ≥ 1 for (μ+λ); lambda ≥ population size for (μ,λ) |
| `.seed(u64)` | no | random | |
| `.initial_genomes(iter)` | no | none | at most the population size, each valid for the representation |
| `.memetic(parents, neighbors)` | no | off | neighbors ≥ 1; 1 ≤ parents ≤ the surviving parents: the elitism (generational), size − replacements (steady-state), the size (μ+λ), none (μ,λ) |

`.memetic(parents, neighbors)` is Lamarckian: each of the best `parents` tries `neighbors` neighbors made by the mutation operator, and takes the best one if it's not worse.

`.build()?` validates everything. It returns `Error::MissingSetting` or `Error::InvalidSetting`, naming the setting.

### Operators

| Constructor | Valid |
|---|---|
| `Tournament::new(size)?` | size ≥ 1 |
| `Rank::new(pressure)?`, `Rank::default()` | 1 ≤ pressure ≤ 2, default 1.5 |
| `Truncation::new(fraction)?` | 0 < fraction ≤ 1 |
| `Roulette`, `StochasticUniversalSampling`, `RandomSelection` | unit structs |
| `PointCrossover::one_point()`, `two_point()`, `k_point(k)?` | k ≥ 1 |
| `UniformCrossover::new()`, `with_rate(p)?` | 0 < p < 1, default 0.5 |
| `NoCrossover` | any representation |
| `SimulatedBinaryCrossover::new(eta)?` | `Real` only; eta ≥ 0: larger means children closer to their parents; 15 to 20 is common |
| `BlendCrossover::new(alpha)?` | `Real` only; alpha ≥ 0, 0.5 is common |
| `ArithmeticCrossover::new()`, `with_weight(w)?` | `Real` only; random weight, or 0 < w < 1 and w ≠ 0.5 |
| `BitFlip::per_gene(rate)?`, `BitFlip::count(n)?` | 0 < rate ≤ 1; n ≥ 1 |
| `UniformMutation::per_gene(rate)?`, `UniformMutation::count(n)?` | 0 < rate ≤ 1; n ≥ 1 |
| `GaussianMutation::per_gene(rate, sigma)?`, `GaussianMutation::count(n, sigma)?` | sigma > 0, a fraction of each gene's range; mirrored at the bounds |
| `PolynomialMutation::per_gene(rate, eta)?`, `PolynomialMutation::count(n, eta)?` | eta ≥ 0: larger means smaller steps; 20 is common |
| `SelfAdaptiveMutation::new()`, `with_learning_rate(tau)?`, `.with_min_step(min)?` | `AdaptiveReal` only; tau > 0, 1/√n by default; min > 0, 1e-12 by default |
| `SwapMutation::new()`, `SwapMutation::count(n)?` | n ≥ 1 |
| `OrderCrossover`, `PartiallyMappedCrossover`, `CycleCrossover`, `EdgeRecombinationCrossover` | unit structs, `Permutation` only |
| `InversionMutation`, `InsertionMutation`, `ScrambleMutation` | unit structs, `Permutation` only |

- `per_gene(rate)` changes each gene independently with that probability. With a rate of `1 / length`, about a third of the children are unchanged copies. A copy inherits its parent's fitness without an evaluation.
- `count(n)` changes exactly `n` genes.
- A picked gene always gets a different value.
- Local search redraws a neighbor that didn't change.

### `Engine::new(algorithm, fitness)`

| Method | Default | Notes |
|---|---|---|
| `.stop_when(stop)` | none | required, unless an abort flag is set; several calls combine with "or" |
| `.observe(observer)` | none | pass `&mut observer` to read it after the run; `Report::new()` prints progress |
| `.on_generation(closure)` | none | `|snapshot| ...`, called after every generation, including generation 0 |
| `.parallel(true)` | off | rayon; same results as sequential; for expensive fitness functions; no effect on a `Batch` |
| `.abort_flag(Arc<AtomicBool>)` | none | stops after the current generation once set |
| `.nan_policy(NanPolicy::Error)` | `NanPolicy::Invalid` | what a NaN fitness means |

Stop conditions:

- `Stop::target(score)`, `Stop::generations(n)`, `Stop::evaluations(n)`, `Stop::time(duration)`, `Stop::stagnation(n)`, `Stop::custom(|progress| ...)`.
- Combine them with `.or(...)` and `.and(...)`.
- They are checked after every generation, including the initial population.
- `Stop::target` means "at least as good as": at most the score when minimizing.
- Some runs have only a target or an evaluation limit. They stop with `StopReason::Stalled` after `genoxide::engine::STALL_GENERATIONS` (10 000) generations in a row with no genome to evaluate. Example: a GA whose children are all copies of their parents.

## Fitness functions

- A closure `|genome: &G| -> T`, where `T` is `f64`, `Fitness` or `Option<f64>`. Or a type implementing `FitnessFunction<G>`.
- It must be deterministic: a child identical to its parent inherits the parent's fitness without an evaluation.
- Maximize is the default. For costs and errors, call `.minimize()` on the builder; don't negate scores.
- NaN becomes invalid by default, or an error with `NanPolicy::Error`.
- `None` or `Fitness::invalid()` marks a solution that can't be scored. Invalid is worse than everything else.

Constraints:

- Return `(score, violation)`. The violation is 0 for a feasible solution, otherwise how far it is from feasible.
- Build the violation by adding up `constraint::at_most(value, limit)`, `at_least` and `equal(value, target, tolerance)`.
- Deb's feasibility rules then apply everywhere: feasible beats infeasible; feasible solutions compete by score, infeasible ones by violation.
- Use `Tournament` or `Rank` selection. Roulette and SUS give infeasible solutions no weight.
- Prefer a violation to an invalid fitness: it tells the search how close a solution is.
- `Penalty::new(weight)?.fitness(objective, score, violation)` is a static penalty instead of Deb's rules. The weight needs tuning.

Batch evaluation:

- `Batch(|genomes: &[&G]| -> Vec<T>)` scores a whole generation in one call, in order: for SIMD, a GPU or a remote service.
- It works with `Engine` and `MultiEngine`.
- It is called once per generation, with an empty slice when every child inherited its fitness.
- Returning a different number of values is `Error::FitnessCount`.
- `examples/gpu` evaluates a generation of neural networks in one wgpu compute dispatch.

## Templates

### Constraints, with a hall of fame

```rust
use genoxide::prelude::*;

const WEIGHTS: [u32; 6] = [12, 7, 11, 8, 9, 5];
const VALUES: [u32; 6] = [24, 13, 23, 15, 16, 8];
const CAPACITY: u32 = 26;

// the value, and how much the weight exceeds the capacity (Deb's feasibility rules)
fn value(selection: &Bits) -> (f64, f64) {
    let (mut weight, mut value) = (0, 0);
    for (item, selected) in selection.iter().enumerate() {
        if selected {
            weight += WEIGHTS[item];
            value += VALUES[item];
        }
    }
    (
        f64::from(value),
        constraint::at_most(f64::from(weight), f64::from(CAPACITY)),
    )
}

fn main() -> genoxide::Result<()> {
    let ga = Ga::builder(Binary::new(6)?)
        .population_size(30)
        .select(Tournament::new(3)?)
        .crossover(PointCrossover::two_point())
        .mutate(BitFlip::count(1)?)
        .seed(1)
        .build()?;

    let mut hall_of_fame = HallOfFame::new(3)?;
    let outcome = Engine::new(ga, value)
        .stop_when(Stop::stagnation(50))
        .observe(&mut hall_of_fame)
        .run()?;

    assert_eq!(outcome.best_fitness(), Fitness::new(51.0));
    assert!(outcome.best_fitness().is_feasible());
    for individual in hall_of_fame.individuals() {
        println!("{} {:?}", individual.genome(), individual.fitness());
    }
    Ok(())
}
```

### Integers and reals, minimizing

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    // integers: hit a target sum with small genes
    let ga = Ga::builder(Integer::uniform(8, -10..=10)?)
        .population_size(50)
        .select(Tournament::new(3)?)
        .crossover(UniformCrossover::new())
        .mutate(UniformMutation::count(1)?)
        .minimize()
        .seed(2)
        .build()?;
    let outcome = Engine::new(ga, |genome: &Integers| {
        (genome.iter().sum::<i64>() - 42).abs() as f64
    })
    .stop_when(Stop::target(0.0).or(Stop::generations(500)))
    .run()?;
    assert_eq!(outcome.best_fitness(), Fitness::new(0.0));

    // reals: the sphere function, with per-gene bounds
    let ga = Ga::builder(Real::new([-5.0..=5.0, -1.0..=3.0, 0.0..=10.0])?)
        .population_size(50)
        .select(Tournament::new(3)?)
        .crossover(UniformCrossover::new())
        .mutate(PolynomialMutation::per_gene(0.3, 20.0)?)
        .minimize()
        .seed(3)
        .build()?;
    let outcome = Engine::new(ga, |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>())
        .stop_when(Stop::target(0.5).or(Stop::generations(500)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

### Permutations

```rust
use genoxide::prelude::*;

// distances between 5 cities on a line, at positions 0, 3, 1, 4, 2
fn tour_length(order: &Order) -> f64 {
    const POSITIONS: [f64; 5] = [0.0, 3.0, 1.0, 4.0, 2.0];
    let mut length = 0.0;
    for (i, &city) in order.iter().enumerate() {
        let next = order[(i + 1) % order.len()];
        length += (POSITIONS[city] - POSITIONS[next]).abs();
    }
    length
}

fn main() -> genoxide::Result<()> {
    let ga = Ga::builder(Permutation::new(5)?)
        .population_size(10)
        .select(Tournament::new(2)?)
        .crossover(NoCrossover)
        .mutate(SwapMutation::new())
        .scheme(Scheme::MuPlusLambda { lambda: 10 })
        .minimize()
        .seed(4)
        .build()?;
    let outcome = Engine::new(ga, tour_length)
        .stop_when(Stop::target(8.0).or(Stop::generations(500)))
        .run()?;
    assert_eq!(outcome.best_fitness(), Fitness::new(8.0));
    Ok(())
}
```

### Statistics, progress output, parallel evaluation and cancellation

```rust
use genoxide::prelude::*;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

fn main() -> genoxide::Result<()> {
    let ga = Ga::builder(Binary::new(32)?)
        .population_size(40)
        .select(Tournament::new(3)?)
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / 32.0)?)
        .seed(5)
        .build()?;

    let abort = Arc::new(AtomicBool::new(false)); // set it from another thread to stop
    let mut statistics = Statistics::new();
    let outcome = Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
        .stop_when(Stop::target(32.0))
        .stop_when(Stop::time(Duration::from_secs(10)))
        .abort_flag(Arc::clone(&abort))
        .parallel(true)
        .observe(&mut statistics)
        .observe(Report::every(Duration::from_secs(1))) // a line to stderr each second
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            if progress.generation() % 10 == 0 {
                println!("{}: {:?}", progress.generation(), progress.best());
            }
        })
        .run()?;

    let last = statistics.last().unwrap();
    println!("mean {:?}, {} distinct genomes", last.mean, last.unique);
    assert_eq!(last.generation, outcome.generations());
    Ok(())
}
```

`Report::new()` prints a progress line to stderr after the initial population, then once a second. `Report::every(duration)` and `Report::every_generations(n)?` change how often. `.to(writer)` sends the lines elsewhere. In a multi-objective run, call `report.update(snapshot.progress())` from `.on_generation`.

The `tracing` feature (`genoxide = { version = "...", features = ["tracing"] }`) makes both engines emit events with the target `genoxide`:

- an info span `run`, with the algorithm type;
- a debug event per generation: generation, evaluations, best, and the front size in a multi-objective run;
- an info event when the run finishes: the stop reason and totals.

Show them with any subscriber, e.g. `tracing-subscriber` with `RUST_LOG=genoxide=debug`.

### Evolution strategy with self-adaptation

For smooth real-valued problems that need precise answers: a (μ/ρ, λ)-ES. The step sizes evolve with each solution: large far from the optimum, small close to it. With a step size per gene (the default), it also learns how the genes are scaled.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    // (5/5_I, 35)-ES: intermediate recombination of all 5 parents, comma selection
    let es = Es::builder(Real::uniform(5, -5.0..=5.0)?)
        .parents(5) // μ
        .offspring(35) // λ, 5 to 7 times μ
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(es, |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>())
        .stop_when(Stop::target(1e-10).or(Stop::evaluations(100_000)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

| Setting | Options |
|---|---|
| `.recombination(...)` | `es::Recombination::Intermediate { rho }` (default, ρ = μ), `es::Recombination::Dominant { rho }`; ρ = 1 for none |
| `.selection(...)` | `es::Selection::Comma` (default; best for self-adaptation), `es::Selection::Plus` (elitist) |
| `.step_sizes(...)` | `es::StepSizes::PerGene` (default), `es::StepSizes::One` (faster when all genes are scaled alike) |

A GA can also run an ES, with other operators: `AdaptiveReal` genomes, `SelfAdaptiveMutation`, `NoCrossover` and `Scheme::MuCommaLambda { lambda }`. For hard problems (rotated, badly conditioned or multimodal), CMA-ES is stronger: see [CMA-ES](#cma-es).

### Differential evolution

For continuous problems on `Real` genomes, differential evolution often needs far fewer evaluations than a GA. The defaults are SHADE's published settings (Tanabe and Fukunaga, "Success-History Based Parameter Adaptation for Differential Evolution", IEEE CEC 2013):

- current-to-pbest/1, with a random `p` per trial between 2 / population and 0.2;
- an archive of the population's size;
- SHADE's adaptation of `F` and `CR`, with a memory of 100;
- a population of 100.

genoxide adds restarts when the population converges or stalls (tolerance 1e-8, patience 200 generations). This is genoxide's own choice, not SHADE's.

Set only what the problem needs:

- `.population_size(n)`: 100 by default.
- `.control(de::Control::Fixed { f, cr })`: fixed `F` and `CR`. A small `CR` (e.g. 0.1) suits separable functions; a large one (0.9) suits rotated or coupled ones.
- `.restarts(de::Restarts::Never)`: no restarts.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    let rastrigin = |x: &Reals| {
        10.0 * x.len() as f64
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (std::f64::consts::TAU * xi).cos())
                .sum::<f64>()
    };
    let de = De::builder(Real::uniform(5, -5.12..=5.12)?)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(de, rastrigin)
        .stop_when(Stop::target(0.01).or(Stop::evaluations(100_000)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

For a known evaluation budget, use L-SHADE. It shrinks a large population over the budget. It aims at the best final value, not the fewest evaluations to a target.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    let budget = 60_000;
    // L-SHADE: 18 × genes individuals, shrinking to 4 over the budget
    let de = De::l_shade(Real::uniform(6, -5.0..=5.0)?, budget).minimize().seed(1).build()?;
    let outcome = Engine::new(de, |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>())
        .stop_when(Stop::target(1e-8).or(Stop::evaluations(budget)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

| `de::Strategy` | When |
|---|---|
| `CurrentToPBestRandomP { max_p: 0.2, archive: 1.0 }` (default) | SHADE's: a random `p` per trial; the archive keeps it diverse |
| `CurrentToPBest { p: 0.1, archive: 1.0 }` | JADE's, with a fixed `p` |
| `Rand1` | Explores well |
| `Best1` | Greedy; use it with `de::Control::Dither { min_f: 0.5, max_f: 1.0, cr }`, or the population can collapse before the optimum |

### CMA-ES

The strongest general choice for continuous problems with up to a few hundred `Real` genes, especially when the genes interact (rotated or badly conditioned functions). It learns the correlations between genes.

- Nothing needs tuning. The population size follows from the number of genes; the initial step size is 0.3 of each range.
- For multimodal functions, add restarts: `cmaes::Restarts::Ipop` (a growing population) or `cmaes::Restarts::Bipop` (large and small populations in turn).
- For hundreds to thousands of genes, or separable problems, use `.covariance(cmaes::Covariance::Diagonal)` (sep-CMA-ES). Each sample costs O(n) instead of O(n²). It learns the scaling of each gene faster, but not the correlations.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    let rastrigin = |x: &Reals| {
        10.0 * x.len() as f64
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (std::f64::consts::TAU * xi).cos())
                .sum::<f64>()
    };
    let cmaes = Cmaes::builder(Real::uniform(5, -5.12..=5.12)?)
        .restarts(cmaes::Restarts::Ipop)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(cmaes, rastrigin)
        .stop_when(Stop::target(0.01).or(Stop::evaluations(200_000)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

### Particle swarm optimization

For `Real` genomes. The default constriction coefficients leave two settings: the number of particles and the topology.

- `pso::Topology::Global` (default) converges fastest.
- `pso::Topology::Ring { neighbors: 1 }` explores longer and suits multimodal functions.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    let rosenbrock = |x: &Reals| {
        x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum::<f64>()
    };
    let pso = Pso::builder(Real::uniform(5, -5.0..=10.0)?)
        .population_size(40) // 20 to 50 particles
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(pso, rosenbrock)
        .stop_when(Stop::target(0.01).or(Stop::evaluations(100_000)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

### Island model

Several populations (islands) evolve apart and exchange their best individuals every few generations. They are more diverse than one large population, and often faster on multimodal problems. Any `Ga` or `De` can be an island; give each its own seed.

```rust
use genoxide::algorithm::islands::Topology;
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    let islands = (0..4)
        .map(|seed| {
            Ga::builder(Real::uniform(10, -5.12..=5.12)?)
                .population_size(25)
                .select(Tournament::new(3)?)
                .crossover(UniformCrossover::new())
                .mutate(PolynomialMutation::per_gene(0.1, 20.0)?)
                .minimize()
                .seed(seed)
                .build()
        })
        .collect::<genoxide::Result<Vec<_>>>()?;
    let islands = Islands::builder(islands)
        .topology(Topology::Ring) // or FullyConnected, Random
        .interval(10) // generations between migrations
        .migrants(2) // copies of each island's best, replacing the worst of the next
        .build()?;
    let rastrigin = |x: &Reals| {
        10.0 * x.len() as f64
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (std::f64::consts::TAU * xi).cos())
                .sum::<f64>()
    };
    let outcome = Engine::new(islands, rastrigin)
        .stop_when(Stop::target(0.01).or(Stop::evaluations(500_000)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

- The engine evaluates the candidates of all islands together, in parallel with `.parallel(true)`.
- Breeding and migration are sequential, so a seed gives the same run with any number of threads.
- The islands must share the objective and the representation (genome length and bounds). `build` checks both.
- Each island counts only its own evaluations. Give an L-SHADE island (`De::l_shade(real, budget)`) its share of the budget, not the whole.

### Asynchronous evaluation for slow, uneven fitness functions

For evaluations that take long and vary in time: simulations, training runs, calls to other programs. A generational GA waits for the slowest evaluation of each generation. A steady-state GA with `AsyncEngine` gives each worker a new genome as soon as it's done. Build it with `build_steady()` instead of `build()`. The scheme and memetic settings don't apply.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    let ga = Ga::builder(Real::uniform(5, -5.12..=5.12)?)
        .population_size(40)
        .select(Tournament::new(3)?)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(0.2, 20.0)?)
        .minimize()
        .seed(1)
        .build_steady()?;
    let sphere = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
    let outcome = AsyncEngine::new(ga, sphere)
        .workers(8) // evaluations at a time; the number of CPUs by default
        .stop_when(Stop::target(1e-6).or(Stop::evaluations(100_000)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

- Each result replaces the worst individual when it's not worse. A genome already in the population isn't added twice.
- A generation is counted every `population_size` evaluations, for observers, checkpoints and `Stop::generations`.
- Stop conditions are checked after every result. The run returns once the evaluations in flight are done. `Stop::evaluations` ends on the limit exactly.
- With one worker, a seed gives the same run every time. With more, the order of the results depends on timing, so runs differ.
- The workers are their own threads, not rayon's. Use more workers than CPUs for fitness functions that mostly wait.

### Checkpoints: resuming a long run

With the `serde` feature (`genoxide = { version = "...", features = ["serde"] }`), `checkpoint_every` saves the algorithm every few generations and when the run stops. A resumed run gives exactly the results of the uninterrupted run. Name the algorithm's type with an alias: loading needs it.

```rust
use genoxide::checkpoint;
use genoxide::prelude::*;

type OneMax = Ga<Binary, Tournament, UniformCrossover, BitFlip>;

fn main() -> genoxide::Result<()> {
    let path = std::env::temp_dir().join("genoxide-one-max.ckpt");
    let one_max = |genome: &Bits| genome.count_ones() as f64;
    let ga: OneMax = if path.exists() {
        checkpoint::load_file(&path)? // resume
    } else {
        Ga::builder(Binary::new(200)?)
            .population_size(50)
            .select(Tournament::new(3)?)
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::per_gene(1.0 / 200.0)?)
            .seed(1)
            .build()?
    };
    let outcome = Engine::new(ga, one_max)
        .stop_when(Stop::target(200.0).or(Stop::generations(5_000)))
        .checkpoint_every(100, |ga| checkpoint::save_file(ga, &path))
        .run()?;
    println!("{} after {} generations", outcome.best_fitness(), outcome.generations());
    std::fs::remove_file(&path).ok(); // done: the next run starts afresh
    Ok(())
}
```

- `save_file` is atomic: it writes a temporary file and renames it. A crash while saving keeps the previous checkpoint.
- Generations, evaluations and stop conditions continue from the checkpoint. The clock for `Stop::time` starts again.
- `MultiEngine` has `checkpoint_every` too.
- A checkpoint loads with the genoxide version that saved it, as the same type. A corrupted, truncated or foreign file, another version or another type is `Error::Checkpoint`.
- Load only checkpoints you trust. The checksum catches accidental damage, not tampering. A crafted checkpoint can make a run panic or loop (never memory-unsafe).
- Observers aren't in a checkpoint: a resumed run's statistics and hall of fame start empty.
- Every algorithm, genome, representation and operator implements `Serialize` and `Deserialize`, as do `Statistics` and `HallOfFame`. JSON can't store NaN or infinity (invalid fitness, crowding distances): prefer `checkpoint` or a binary format.

### Multi-objective optimization

When objectives conflict (cost against quality, speed against accuracy), the result is a front of trade-offs, not a single best solution.

- The fitness function returns an array with one value per objective. Alternatives: `(values, violation)` with a constraint violation, or `Option<[f64; M]>`.
- The algorithm takes the direction of each objective. `MultiEngine` runs it.
- The outcome is the Pareto front.

```rust
use genoxide::Objective::Minimize;
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    // ZDT1: two objectives, 30 genes
    let zdt1 = |x: &Reals| {
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / 29.0;
        [x[0], g * (1.0 - (x[0] / g).sqrt())]
    };
    let nsga2 = Nsga2::builder(Real::uniform(30, 0.0..=1.0)?, [Minimize, Minimize])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0)?)
        .seed(1)
        .build()?;
    let outcome = MultiEngine::new(nsga2, zdt1)
        .stop_when(Stop::generations(200))
        .run()?;
    for [f1, f2] in outcome.front_values() {
        assert!(f2 <= 1.0 - f1.sqrt() + 0.05); // close to the optimal front
    }
    Ok(())
}
```

| Algorithm | Builder | Notes |
|---|---|---|
| `Nsga2` | `Nsga2::builder(real, objectives)` | Non-dominated sorting and crowding distance |
| `Spea2` | same as `Nsga2` | Truncation keeps the extremes and spreads the front evenly; it compares every pair of solutions, so a generation costs more |
| `SmsEmoa` | same as `Nsga2` | Keeps the solutions that add the most hypervolume; a generation costs more, especially for 3 or more objectives |
| `Moead` | `Moead::builder(real, objectives, multi::das_dennis::<M>(divisions))` | One subproblem per weight vector; Tchebycheff by default, `multi::Decomposition::Pbi { theta: 5.0 }` for 3 or more objectives; cheap per generation |
| `Nsga3` | `Nsga3::builder(real, objectives, multi::das_dennis::<3>(12))` | For 3 or more objectives: spreads the front along reference directions (91 for 3 objectives and 12 divisions). The population size defaults to their number. Use SBX with η 30 and polynomial mutation |

- The number of objectives is part of the types: returning `[f64; 3]` for 2 objectives doesn't compile.
- Constraints: feasible solutions dominate infeasible ones. Between infeasible ones, the smaller violation wins (`multi::dominates`).
- Stop conditions: generations, evaluations, time, stagnation (generations without a new non-dominated solution) or custom. `Stop::target` needs a single objective.
- `multi::non_dominated_sort` and `multi::crowding_distance` are available for your own algorithms.
- To keep every non-dominated solution of a run, not only the final front: `let mut archive = multi::ParetoArchive::new(objectives);` and `.on_generation(|snapshot| archive.update(snapshot))`.
- Test problems with known optimal fronts, usable as fitness functions: `multi::problems::{Zdt1, Zdt2, Zdt3, Zdt4, Zdt6, Dtlz1, Dtlz2, Dtlz3, Dtlz4}`. The `TestProblem` trait gives `real()` and `optimal_front(points)`.
- Measure a front with `multi::indicator`:
  - `hypervolume(&front, &reference_point, &objectives)`: larger is better. `hypervolume_contributions` gives each point's exclusive share.
  - `igd_plus`, `igd`, `gd` and `spread` against a reference front: smaller is better.

### Local search: hill climbing and simulated annealing

`LocalSearch` improves a single solution. Each step, it moves to one of `neighbors` random neighbors. It often beats a GA on permutations. Any mutation is a neighborhood; `InversionMutation` (2-opt) is the classic one for tours.

| `.acceptance(...)` | Moves to the best neighbor when |
|---|---|
| `Acceptance::Improving` | it's strictly better (stops at a local optimum) |
| `Acceptance::NotWorse` (default) | it's better or equal (drifts across plateaus) |
| `Acceptance::Annealing { initial_temperature, cooling }` | it's better, or worse by Δ with probability exp(−Δ/T), T cooling every step |
| `Acceptance::Tabu { tenure }` | it isn't one of the last `tenure` solutions (even if worse), or beats the best so far; use several neighbors |

`.restart(patience, kicks)` adds iterated local search to any of them. After `patience` steps without a new best, the search restarts from the best solution, changed by `kicks` neighbor moves.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    // 8 cities on a line; the best tour goes out and back: length 2 * 7 = 14
    let tour_length = |order: &Order| {
        let n = order.len();
        (0..n).map(|i| order[i].abs_diff(order[(i + 1) % n]) as f64).sum::<f64>()
    };
    let search = LocalSearch::builder(Permutation::new(8)?)
        .neighbor(InversionMutation)
        .neighbors(4) // best of 4 per step; 1 is first-improvement
        // Acceptance::NotWorse (the default) is hill climbing that crosses plateaus
        .acceptance(Acceptance::Annealing {
            initial_temperature: 2.0,
            cooling: 0.99,
        })
        .minimize()
        .seed(7)
        .build()?;
    let outcome = Engine::new(search, tour_length)
        .stop_when(Stop::target(14.0).or(Stop::generations(5_000)))
        .run()?;
    assert_eq!(outcome.best_fitness(), Fitness::new(14.0));
    Ok(())
}
```

### Ask / tell: evaluating outside the engine

Drive the algorithm by hand when the fitness is computed elsewhere: another process, a simulator, a remote service or async code.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    let mut ga = Ga::builder(Binary::new(16)?)
        .population_size(20)
        .select(Tournament::new(2)?)
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::count(1)?)
        .seed(6)
        .build()?;

    while ga.generation() < 50 {
        // the genomes that need a fitness (the initial population first, then offspring)
        let genomes: Vec<Bits> = ga.ask().iter().cloned().collect();
        let fitness: Vec<Fitness> = genomes
            .iter()
            .map(|genome| Fitness::new(genome.count_ones() as f64))
            .collect();
        ga.tell(&fitness)?; // same order as asked
    }
    println!("best: {:?}", ga.best().map(|best| best.genome().to_string()));
    Ok(())
}
```

### Without Rust: the `genoxide` program

For a fitness function in another language, or no code at all. Install with `cargo install genoxide --features cli`. `genoxide run run.toml` runs an optimization described in a TOML or JSON file.

- It starts `fitness.command` once per worker.
- That program reads a genome per line on stdin. It writes a line per genome on stdout: the objective values, then optionally a constraint violation.
- The result goes to stdout as JSON.

[docs/cli.md](docs/cli.md) has every setting.

```toml
[genome]
type = "real"          # binary, integer, real or permutation
length = 10
bounds = [-5.12, 5.12]

[fitness]
command = ["python3", "fitness.py"]   # or builtin = "rastrigin"
objectives = ["minimize"]

[algorithm]
type = "ga"            # ga, steady-ga, de, cmaes, pso, local-search, nsga2
population_size = 50
select = { type = "tournament", size = 3 }
crossover = { type = "simulated-binary", eta = 15.0 }
mutate = { type = "polynomial", rate = 0.1, eta = 20.0 }

[stop]
target = 0.01
evaluations = 100000

[checkpoint]           # genoxide run run.toml --resume continues from it
path = "run.ckpt"
every = 50
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| ``error[E0277]: `Unset` is not a crossover for `Binary` `` (or a selection or mutation) | That operator was not set | Call `.crossover(...)`, `.select(...)` or `.mutate(...)` before `.build()` |
| ``error[E0277]: the genes of `Order` can't be exchanged by position`` | Point or uniform crossover on a permutation would duplicate genes | Use a permutation crossover: `OrderCrossover`, `EdgeRecombinationCrossover`, `PartiallyMappedCrossover` or `CycleCrossover` |
| ``error[E0277]: `BitFlip` is not a mutation for `Integer` `` | Wrong mutation for the genome | Use the table in [Choosing the pieces](#choosing-the-pieces) |
| ``error[E0277]: `usize` is not a fitness value`` (then "the method `stop_when` exists … but its trait bounds were not satisfied") | The fitness function returns an integer | Return `f64` (`... as f64`), `Fitness` or `Option<f64>` |
| `no method named parallel` | Built without the default `parallel` feature | Enable the `parallel` feature, or drop `.parallel(true)` |
| `Error::MissingSetting { setting: "population_size" }` | No population size | `.population_size(n)` |
| ``error[E0277]: `[f64; 3]` is not a result with 2 objective values`` | The fitness function returns a different number of values than the algorithm has objectives | Return an array with one value per objective, e.g. `[f1, f2]` for `[Minimize, Minimize]` |
| `Error::MissingSetting { setting: "stop_when" }` | The engine has no stop condition | `.stop_when(Stop::generations(n))`, or an abort flag |
| `Error::InvalidSetting { setting, reason }` | A value is out of range | Read `reason`; see [Settings](#settings) |
| `Error::InvalidGenome { reason }` | An initial genome doesn't fit the representation | Match its length and bounds |
| `Error::NanFitness` | The fitness function returned NaN (a score or a violation) with `NanPolicy::Error` | Fix the fitness function, or keep the default `NanPolicy::Invalid` |
| `Error::InvalidFitness` | A negative constraint violation | A violation is 0 or more: use `constraint::at_most` and friends |
| The best solution is infeasible | No feasible solution found yet | Run longer, check the constraints can be met, or start from a feasible solution with `.initial_genomes(...)` |
| `Error::TellWithoutAsk` / `Error::FitnessCount` | Ask / tell out of step | One `tell` per `ask`, with one fitness per asked genome, in order |
| Hill climbing (`Acceptance::Improving` or `NotWorse`) stops improving | A local optimum | `.restart(patience, kicks)` (iterated local search), `Acceptance::Tabu { tenure }` with several neighbors, or `Acceptance::Annealing` with an initial temperature about the size of typical fitness differences and `cooling` close to 1 (e.g. 0.999) |
| A differential evolution stops improving far from the optimum, with a tiny population spread | The population collapsed (greedy strategy, fixed `F`) | `de::Control::Dither { min_f: 0.5, max_f: 1.0, cr }`, `de::Strategy::Rand1`, or an archive with `CurrentToPBest` |
| A CMA-ES stops improving (without restarts) | The run converged: `cmaes.converged()` says why | `.restarts(cmaes::Restarts::Ipop)` or `Bipop`; a larger `.initial_step(...)` or `.population_size(...)` |
| A particle swarm gathers around a local optimum early | The global topology spreads the best position to every particle at once | `.topology(pso::Topology::Ring { neighbors: 1 })`, or more particles |
| Real-valued search stalls in a local minimum | Steps too small to leave its basin (e.g. `GaussianMutation` with a tiny sigma) | `PolynomialMutation` with eta 20, or a larger sigma |
| `StopReason::Stalled` | For 10 000 generations, every child was a copy of a parent (e.g. a converged population with `mutation_rate(0.0)`). Nothing was evaluated, so a target or evaluation limit could never be met | A mutation rate above 0, or add `Stop::generations(n)` or `Stop::stagnation(n)` |
| The best fitness stops improving early | Too little diversity | A larger population, a smaller tournament, a higher mutation rate, or `Stop::stagnation` with restarts |
| Slow runs with a cheap fitness function | Debug build, or parallel overhead | Build with `--release`; use `.parallel(true)` only for expensive fitness functions |

## Guarantees to rely on

- **Reproducible:** the same seed and settings give the same results on every platform, with or without `.parallel(true)`, on any number of threads.
- **Deterministic ties:** the earlier individual wins.
- **The best is never lost:** `outcome.best()` is the best individual ever evaluated, even when it didn't survive.
- **No panics on invalid settings:** they are errors from constructors, `build()` and `run()`. This includes sizes above 2^24 (populations, offspring, tournaments, neighbors).
- **Documented panics only**, under `# Panics`:
  - an index out of bounds (`Bits::set`, `Order::swap`), like slices;
  - the constructors of the multi-objective test problems (`multi::problems`) with too few variables;
  - a `Batch` that returns no score for a single genome.
