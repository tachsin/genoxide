# genoxide guide for AI coding assistants

Every Rust block is a complete program, run in CI. genoxide is pre-1.0: check the version in `Cargo.toml` and the [API docs](https://docs.rs/genoxide).

## The shape of every program

A **representation** (e.g. `Binary::new(100)?`), a **`Ga`** (population size, selection, crossover, mutation) and an **`Engine`** (fitness function, stop condition). `use genoxide::prelude::*;` imports everything here.

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

## Choosing the pieces

| Problem | Representation | Genome | Crossover | Mutation |
|---|---|---|---|---|
| Yes / no choices (subsets) | `Binary::new(len)` | `Bits` | `UniformCrossover`, `PointCrossover` | `BitFlip` |
| Integers in ranges | `Integer::new([lo..=hi, ...])`, `Integer::uniform(len, lo..=hi)` | `Integers` (`[i64]`) | `UniformCrossover`, `PointCrossover` | `UniformMutation` |
| Reals in ranges | `Real::new([lo..=hi, ...])`, `Real::uniform(len, lo..=hi)` | `Reals` (`[f64]`) | `SimulatedBinaryCrossover` (η 15), `BlendCrossover` (α 0.5), `ArithmeticCrossover`, `UniformCrossover`, `PointCrossover` | `PolynomialMutation` (η 20), `GaussianMutation`, `UniformMutation` |
| Reals with an adaptive step size | `AdaptiveReal::new(Real::..., initial_step)` | `AdaptiveReals` (`[f64]`, `.step()`) | `NoCrossover` (an ES), `UniformCrossover`, `PointCrossover` | `SelfAdaptiveMutation` |
| An order of `0..n` (tours, sequencing) | `Permutation::new(n)` | `Order` (`[usize]`) | `OrderCrossover` (sequences), `EdgeRecombinationCrossover` (tours), `PartiallyMappedCrossover`, `CycleCrossover` | `InversionMutation` (tours), `SwapMutation`, `InsertionMutation`, `ScrambleMutation` |

Any selection fits any representation; usually `Tournament` of size 2 to 5.

| Scheme (`.scheme(...)`) | When |
|---|---|
| `Scheme::Generational { elitism: 1 }` (default) | General purpose |
| `Scheme::SteadyState { replacements: k }` | Replace the `k` worst each generation |
| `Scheme::MuPlusLambda { lambda }` | Strong elitism; mutation-only search (`NoCrossover`); plateaus |
| `Scheme::MuCommaLambda { lambda }` | Parents never survive; `lambda` ≥ population size |

## Settings

### `Ga::builder(representation)`

| Method | Default | Valid |
|---|---|---|
| `.population_size(n)` | required | ≥ 1 |
| `.select(s)`, `.crossover(c)`, `.mutate(m)` | required | a `Select`; a `Crossover` / `Mutate` for the representation |
| `.maximize()`, `.minimize()`, `.objective(o)` | maximize | |
| `.crossover_rate(p)` | 0.9 | 0 ≤ p ≤ 1 |
| `.mutation_rate(p)` | 1.0 | 0 ≤ p ≤ 1, not both rates 0 |
| `.scheme(s)` | generational, elitism 1 | elitism < size; 1 ≤ replacements ≤ size; lambda ≥ 1 (μ+λ) or ≥ size (μ,λ) |
| `.seed(u64)` | random | |
| `.initial_genomes(iter)` | | at most the size, each valid |
| `.memetic(parents, neighbors)` | off | neighbors ≥ 1; 1 ≤ parents ≤ surviving parents: the elitism, size − replacements, the size (μ+λ), none (μ,λ) |

`.memetic`: each of the best `parents` takes the best of `neighbors` mutated neighbors if not worse (Lamarckian). `.build()?` returns `Error::MissingSetting` or `Error::InvalidSetting`, naming the setting.

### Operators

| Constructor | Valid |
|---|---|
| `Tournament::new(size)?` | size ≥ 1 |
| `Rank::new(pressure)?`, `Rank::default()` | 1 ≤ pressure ≤ 2, default 1.5 |
| `Truncation::new(fraction)?` | 0 < fraction ≤ 1 |
| `Roulette`, `StochasticUniversalSampling`, `RandomSelection`, `NoCrossover` | unit structs |
| `PointCrossover::one_point()`, `two_point()`, `k_point(k)?` | k ≥ 1 |
| `UniformCrossover::new()`, `with_rate(p)?` | 0 < p < 1, default 0.5 |
| `SimulatedBinaryCrossover::new(eta)?` | `Real`; eta ≥ 0 (larger: children nearer the parents), 15 to 20 common |
| `BlendCrossover::new(alpha)?` | `Real`; alpha ≥ 0, 0.5 common |
| `ArithmeticCrossover::new()`, `with_weight(w)?` | `Real`; random weight, or 0 < w < 1, w ≠ 0.5 |
| `BitFlip`, `UniformMutation`: `per_gene(rate)?`, `count(n)?` | 0 < rate ≤ 1; n ≥ 1 |
| `GaussianMutation::per_gene(rate, sigma)?`, `count(n, sigma)?` | sigma > 0, a fraction of each gene's range; mirrored at the bounds |
| `PolynomialMutation::per_gene(rate, eta)?`, `count(n, eta)?` | eta ≥ 0 (larger: smaller steps), 20 common |
| `SelfAdaptiveMutation::new()`, `with_learning_rate(tau)?`, `.with_min_step(min)?` | `AdaptiveReal`; tau > 0, default 1/√n; min > 0, default 1e-12 |
| `SwapMutation::new()`, `SwapMutation::count(n)?` | n ≥ 1 |
| `OrderCrossover`, `PartiallyMappedCrossover`, `CycleCrossover`, `EdgeRecombinationCrossover`, `InversionMutation`, `InsertionMutation`, `ScrambleMutation` | unit structs, `Permutation` |

`per_gene(rate)` changes each gene with that probability (at `1 / length`, a third of the children are unevaluated copies); `count(n)` exactly `n` genes. A picked gene always changes.

### `Engine::new(algorithm, fitness)`

| Method | Notes |
|---|---|
| `.stop_when(stop)` | required without an abort flag; calls combine with "or" |
| `.observe(observer)` | `&mut observer` to read it afterwards |
| `.on_generation(\|snapshot\| ...)` | after every generation |
| `.parallel(true)` | rayon, same results; for expensive fitness functions; no effect on a `Batch` |
| `.abort_flag(Arc<AtomicBool>)` | stops after the current generation once set |
| `.nan_policy(NanPolicy::Error)` | NaN is an error, not invalid (`NanPolicy::Invalid`, default) |

Stops: `Stop::target(score)` (at least as good), `generations(n)`, `evaluations(n)`, `time(duration)`, `stagnation(n)`, `custom(|progress| ...)`, combined with `.or(...)` and `.and(...)`, checked after every generation. With only targets and evaluation limits, a run stops as `StopReason::Stalled` after `genoxide::engine::STALL_GENERATIONS` (10 000) generations with nothing to evaluate.

## Fitness functions

- A closure `|genome: &G| -> T` (`f64`, `Fitness` or `Option<f64>`) or a `FitnessFunction<G>`. Deterministic: a copy of a parent inherits its fitness.
- Maximize is the default; use `.minimize()`, don't negate.
- `None`, `Fitness::invalid()` and NaN are invalid: worse than everything.
- **Constraints:** return `(score, violation)`, 0 when feasible, adding up `constraint::at_most(value, limit)`, `at_least`, `equal(value, target, tolerance)`. Deb's rules: feasible beats infeasible, then score or violation decides. Select with `Tournament` or `Rank`: roulette and SUS give infeasible solutions no weight. `Penalty::new(weight)?.fitness(objective, score, violation)` is a static penalty instead.
- **Test problems:** `problems::{Sphere, AxisParallelEllipsoid, Schwefel1_2, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel2_26, Levy, Zakharov, StyblinskiTang, Michalewicz}::new(n)` and `problems::{Himmelblau, Branin, GoldsteinPrice, SixHumpCamel}` are fitness functions for `Engine::new(algorithm, problem)`, all minimized. The `problems::Problem` trait gives `representation()` (the bounds), `optimum()` (`value()`, `solutions()`), `reference()`; `problems::all()` lists them as `Box<dyn DynProblem>`.
- **Batch:** `Batch(|genomes: &[&G]| -> Vec<T>)` scores a generation in one call, in order (SIMD, GPU, remote), in `Engine` or `MultiEngine`; the slice can be empty. A wrong count is `Error::FitnessCount`. See `examples/gpu` (wgpu).

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

`Report::new()` writes to stderr every second; also `Report::every(duration)`, `Report::every_generations(n)?`, `.to(writer)`. Multi-objective: `report.update(snapshot.progress())` in `.on_generation`. The `tracing` feature adds a `run` span and per-generation and final events, target `genoxide`.

### Evolution strategy with self-adaptation

For smooth real-valued problems that need precise answers: step sizes evolve with each solution, one per gene by default.

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
| `.step_sizes(...)` | `es::StepSizes::PerGene` (default), `es::StepSizes::One` (genes scaled alike) |

A GA can run an ES too: `AdaptiveReal`, `SelfAdaptiveMutation`, `NoCrossover`, `Scheme::MuCommaLambda`. For hard problems (rotated, badly conditioned or multimodal), CMA-ES is stronger.

### Differential evolution

For continuous problems on `Real` genomes, differential evolution often needs far fewer evaluations than a GA. Defaults from SHADE (Tanabe and Fukunaga, CEC 2013): current-to-pbest/1 with a random `p` in [2 / size, 0.2], an archive of population size, `F` / `CR` memory of 100, 100 individuals; plus genoxide's restarts (tolerance 1e-8, patience 200). Options: `.population_size(n)`, `.control(de::Control::Fixed { f, cr })` (`CR` 0.1 for separable, 0.9 for rotated functions), `.restarts(de::Restarts::Never)`.

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

L-SHADE shrinks a population over a known budget:

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
| `CurrentToPBestRandomP { max_p: 0.2, archive: 1.0 }` (default) | SHADE's |
| `CurrentToPBest { p: 0.1, archive: 1.0 }` | JADE's, fixed `p` |
| `Rand1` | Explores well |
| `Best1` | Greedy; with `de::Control::Dither { min_f: 0.5, max_f: 1.0, cr }`, or it can collapse |

### CMA-ES

The strongest general choice for continuous problems with up to a few hundred `Real` genes, especially when the genes interact (rotated or badly conditioned functions). Nothing needs tuning (initial step: 0.3 of each range). For multimodal functions, add `cmaes::Restarts::Ipop` (growing population) or `Bipop` (large and small in turn). For thousands of genes or separable problems: `.covariance(cmaes::Covariance::Diagonal)` (sep-CMA-ES, O(n) per sample, no correlations).

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

For `Real` genomes. `pso::Topology::Global` (default) converges fastest; `pso::Topology::Ring { neighbors: 1 }` explores longer, for multimodal functions.

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

Islands evolve apart and exchange their best: more diverse than one large population, and often faster on multimodal problems. Use `Ga`s or `De`s with their own seeds, sharing objective and representation.

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

`.parallel(true)` evaluates all islands together; results don't depend on the thread count. Each island counts its own evaluations: give an L-SHADE island its share of the budget.

### Asynchronous evaluation for slow, uneven fitness functions

`AsyncEngine` hands each worker a new genome as soon as it's done. Build with `build_steady()` (no scheme, no memetic).

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

- A result replaces the worst if not worse; no duplicates.
- A generation is `population_size` evaluations; stops are checked after each result.
- Reproducible with one worker only.
- Workers are threads of their own, not rayon's: use more than CPUs when the fitness waits.

### Checkpoints: resuming a long run

`serde` feature. A resumed run matches an uninterrupted one. Loading needs the type: name it with an alias.

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

- `save_file` is atomic. `MultiEngine` has `checkpoint_every` too.
- Counters and stops continue; `Stop::time` restarts. Observers aren't saved.
- Same genoxide version and type only, else `Error::Checkpoint`. Load only trusted files: a crafted one can cause a panic or a loop.
- All algorithms, genomes and operators, `Statistics` and `HallOfFame` implement `Serialize` / `Deserialize`; JSON can't store NaN or infinity.

### Multi-objective optimization

Fitness: `[f64; M]`, `(values, violation)` or `Option<[f64; M]>`. `MultiEngine` returns the Pareto front.

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
| `Nsga2` | `Nsga2::builder(real, objectives)` | Non-dominated sorting, crowding distance |
| `Spea2`, `SmsEmoa` | same as `Nsga2` | SPEA2's truncation spreads the front evenly; SMS-EMOA keeps the largest hypervolume contributions; both cost more per generation |
| `Moead` | `Moead::builder(real, objectives, multi::das_dennis::<M>(divisions))` | A subproblem per weight vector; Tchebycheff, or `multi::Decomposition::Pbi { theta: 5.0 }` for 3+ objectives; cheap |
| `Nsga3` | `Nsga3::builder(real, objectives, multi::das_dennis::<3>(12))` | 3+ objectives; population defaults to the number of directions (91); SBX η 30 |

- Objective counts are typed: `[f64; 3]` for 2 objectives doesn't compile.
- Constrained dominance: feasible first, then the smaller violation (`multi::dominates`). `Stop::stagnation` counts generations without a new non-dominated solution; `Stop::target` isn't available.
- `multi::non_dominated_sort`, `multi::crowding_distance`; `multi::ParetoArchive::new(objectives)` with `.on_generation(|snapshot| archive.update(snapshot))` keeps every non-dominated solution.
- Test problems: `multi::problems::{Zdt1, Zdt2, Zdt3, Zdt4, Zdt6, Dtlz1, Dtlz2, Dtlz3, Dtlz4}`, with `TestProblem::real()` and `optimal_front(points)`.
- `multi::indicator`: `hypervolume(&front, &reference_point, &objectives)`, `hypervolume_contributions`; `igd_plus`, `igd`, `gd`, `spread` against a reference front.

### Local search: hill climbing and simulated annealing

`LocalSearch` improves one solution, from `neighbors` random neighbors per step. It often beats a GA on permutations. Any mutation is a neighborhood; `InversionMutation` (2-opt) suits tours.

| `.acceptance(...)` | Moves to the best neighbor when |
|---|---|
| `Acceptance::Improving` | strictly better (stops at a local optimum) |
| `Acceptance::NotWorse` (default) | better or equal (crosses plateaus) |
| `Acceptance::Annealing { initial_temperature, cooling }` | better, or worse by Δ with probability exp(−Δ/T), T cooling each step |
| `Acceptance::Tabu { tenure }` | not among the last `tenure` solutions, or better than the best; use several neighbors |

`.restart(patience, kicks)`: after `patience` steps without a new best, restart from the best, changed by `kicks` moves (iterated local search).

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

For fitness computed elsewhere (another process, async code).

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

`cargo install genoxide --features cli`; `genoxide run run.toml` (or JSON). Each worker runs `fitness.command`: a genome per stdin line in, objective values (then an optional violation) per stdout line out. The result is JSON on stdout. Settings: [docs/cli.md](docs/cli.md).

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

| Symptom | Fix |
|---|---|
| ``error[E0277]: `Unset` is not a crossover for `Binary` `` (or a selection or mutation) | Set `.crossover(...)`, `.select(...)` or `.mutate(...)` before `.build()` |
| ``error[E0277]: the genes of `Order` can't be exchanged by position`` | Use a permutation crossover: `OrderCrossover`, `EdgeRecombinationCrossover`, `PartiallyMappedCrossover` or `CycleCrossover` |
| ``error[E0277]: `BitFlip` is not a mutation for `Integer` `` | See [Choosing the pieces](#choosing-the-pieces) |
| ``error[E0277]: `usize` is not a fitness value`` (then "the method `stop_when` exists … but its trait bounds were not satisfied") | Return `f64` (`... as f64`), `Fitness` or `Option<f64>` |
| ``error[E0277]: `[f64; 3]` is not a result with 2 objective values`` | Return one value per objective, e.g. `[f1, f2]` for `[Minimize, Minimize]` |
| `no method named parallel` | Enable the default `parallel` feature, or drop `.parallel(true)` |
| `Error::MissingSetting { setting: "population_size" }` | `.population_size(n)` |
| `Error::MissingSetting { setting: "stop_when" }` | `.stop_when(Stop::generations(n))`, or an abort flag |
| `Error::InvalidSetting { setting, reason }` | Read `reason`; see [Settings](#settings) |
| `Error::InvalidGenome { reason }` | Match the initial genome's length and bounds |
| `Error::NanFitness` | Fix the NaN, or keep `NanPolicy::Invalid` |
| `Error::InvalidFitness` | A violation must be ≥ 0: use `constraint::at_most` and friends |
| `Error::TellWithoutAsk` / `Error::FitnessCount` | One `tell` per `ask`, one fitness per asked genome, in order |
| Best solution infeasible | Run longer, check the constraints, or add a feasible genome with `.initial_genomes(...)` |
| Hill climbing stops at a local optimum | `.restart(patience, kicks)`, `Acceptance::Tabu { tenure }` with several neighbors, or `Acceptance::Annealing` (initial temperature ≈ typical fitness differences, `cooling` ≈ 0.999) |
| DE collapses far from the optimum | `de::Control::Dither { min_f: 0.5, max_f: 1.0, cr }`, `de::Strategy::Rand1`, or `CurrentToPBest` with an archive |
| CMA-ES converged without restarts (`cmaes.converged()` says why) | `.restarts(cmaes::Restarts::Ipop)` or `Bipop`, or a larger `.initial_step(...)` or `.population_size(...)` |
| PSO gathers early at a local optimum | `.topology(pso::Topology::Ring { neighbors: 1 })`, or more particles |
| Real-valued GA stuck in a local minimum | `PolynomialMutation` with eta 20, or a larger `GaussianMutation` sigma |
| `StopReason::Stalled`: 10 000 generations of copies only (e.g. `mutation_rate(0.0)`) | A mutation rate above 0, or add `Stop::generations(n)` or `Stop::stagnation(n)` |
| Early stagnation (too little diversity) | A larger population, smaller tournament or higher mutation rate; or `Stop::stagnation` and restart |
| Slow with a cheap fitness function | `--release`; `.parallel(true)` only for expensive fitness |

## Guarantees to rely on

- **Reproducible:** a seed gives the same results on every platform and thread count, parallel or not.
- **Ties:** the earlier individual wins.
- **The best is kept:** `outcome.best()` is the best individual ever evaluated.
- **Errors, not panics,** for invalid settings, including sizes above 2^24. The only panics (`# Panics`): an index out of bounds (`Bits::set`, `Order::swap`), a `problems` or `multi::problems` constructor with too few dimensions or variables, and a `Batch` returning no score for a single genome.
