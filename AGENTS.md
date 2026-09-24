# genoxide guide for AI coding assistants

This guide is for AI assistants (and people) writing code that uses genoxide. Every Rust block below is a complete program, compiled and run in CI, so it matches the current API.

genoxide is pre-alpha: the API changes between 0.x versions. Check the version in `Cargo.toml` and the [API docs](https://docs.rs/genoxide) when in doubt.

## The shape of every program

1. Pick a **representation**: the space of solutions, e.g. `Binary::new(100)?`.
2. Build a **`Ga`** with `Ga::builder(representation)`, setting the population size, a selection, a crossover and a mutation.
3. Run it with an **`Engine`**, giving a fitness function and at least one stop condition.

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

`use genoxide::prelude::*;` imports everything used in this guide.

## Choosing the pieces

| Problem | Representation | Genome | Crossover | Mutation |
|---|---|---|---|---|
| Yes / no decisions (subset, knapsack, feature selection) | `Binary::new(len)` | `Bits` | `UniformCrossover`, `PointCrossover` | `BitFlip` |
| Whole numbers in ranges (counts, choices, schedules) | `Integer::new([lo..=hi, ...])`, `Integer::uniform(len, lo..=hi)` | `Integers` (derefs to `[i64]`) | `UniformCrossover`, `PointCrossover` | `UniformMutation` |
| Real numbers in ranges (parameters, continuous functions) | `Real::new([lo..=hi, ...])`, `Real::uniform(len, lo..=hi)` | `Reals` (derefs to `[f64]`) | `SimulatedBinaryCrossover` (η 15), `BlendCrossover` (α 0.5), `ArithmeticCrossover`, `UniformCrossover`, `PointCrossover` | `PolynomialMutation` (η 20), `GaussianMutation` (σ ≈ 0.03), `UniformMutation` |
| Real numbers that need precise fine-tuning (a step size that adapts) | `AdaptiveReal::new(Real::..., initial_step)` | `AdaptiveReals` (derefs to `[f64]`; `.step()`) | `NoCrossover` for an ES, or `UniformCrossover`, `PointCrossover` | `SelfAdaptiveMutation` |
| An order of `0..n` (tours, sequencing, assignment) | `Permutation::new(n)` | `Order` (derefs to `[usize]`) | `OrderCrossover` (sequences), `EdgeRecombinationCrossover` (tours), `PartiallyMappedCrossover`, `CycleCrossover` | `InversionMutation` (tours), `SwapMutation`, `InsertionMutation`, `ScrambleMutation` |

Selection works with every representation. `Tournament::new(2..=5)?` is the usual choice.

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
| `.memetic(parents, neighbors)` | no | off | neighbors ≥ 1, and 1 ≤ parents ≤ the parents that survive: the elitism of a generational scheme, size − replacements of a steady-state one, the size with (μ+λ), none with (μ,λ). The best parents each try neighbors made by the mutation operator, and take the best one if not worse (Lamarckian) |

`.build()?` validates everything and returns `Error::MissingSetting` or `Error::InvalidSetting` naming the setting.

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

Every mutation changes the genome: `per_gene` changes one random gene if none was picked.

### `Engine::new(algorithm, fitness)`

| Method | Default | Notes |
|---|---|---|
| `.stop_when(stop)` | none | required, unless an abort flag is set; several calls combine with "or" |
| `.observe(observer)` | none | pass `&mut observer` to read it after the run |
| `.on_generation(closure)` | none | `|snapshot| ...`, called after every generation, including generation 0 |
| `.parallel(true)` | off | rayon; same results as sequential; worth it for expensive fitness functions |
| `.abort_flag(Arc<AtomicBool>)` | none | stops after the current generation once set |
| `.nan_policy(NanPolicy::Error)` | `NanPolicy::Invalid` | what a NaN fitness means |

Stop conditions: `Stop::target(score)`, `Stop::generations(n)`, `Stop::evaluations(n)`, `Stop::time(duration)`, `Stop::stagnation(n)` and `Stop::custom(|progress| ...)`. Combine them with `.or(...)` and `.and(...)`. They are checked after every generation, including the initial population. `Stop::target` means "at least as good as", so at most the score when minimizing.

## Fitness functions

- A closure `|genome: &G| -> T`, where `T` is `f64`, `Fitness` or `Option<f64>`, or a type implementing `FitnessFunction<G>`.
- It must be deterministic: a child identical to its parent inherits the parent's fitness without being evaluated again.
- Constraints: return `(score, violation)`, where the violation is 0 for a feasible solution and otherwise how far it is from feasible (add up `constraint::at_most(value, limit)`, `at_least` and `equal(value, target, tolerance)`). Deb's feasibility rules then apply everywhere: feasible beats infeasible, feasible solutions compete by score, infeasible ones by violation. With constraints, use `Tournament` or `Rank` selection: roulette and SUS ignore infeasible solutions.
- `None` or `Fitness::invalid()` marks a solution that can't be scored at all. Invalid is worse than everything else. Prefer a violation for constraints: it tells the search how close a solution is.
- `Penalty::new(weight)?.fitness(objective, score, violation)` is a static penalty function instead of Deb's rules; the weight needs tuning.
- NaN becomes invalid by default, or an error with `NanPolicy::Error`.
- Maximize is the default. Call `.minimize()` on the builder for costs and errors; don't negate scores.

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

### Evolution strategy with self-adaptation

For smooth real-valued problems that need precise answers, a (μ,λ)-ES whose step size evolves with each solution: large steps far from the optimum, small ones close to it.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    let es = Ga::builder(AdaptiveReal::new(Real::uniform(5, -5.0..=5.0)?, 0.3)?)
        .population_size(10) // μ
        .select(Tournament::new(2)?)
        .crossover(NoCrossover)
        .mutate(SelfAdaptiveMutation::new())
        .scheme(Scheme::MuCommaLambda { lambda: 70 }) // λ
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(es, |x: &AdaptiveReals| x.iter().map(|xi| xi * xi).sum::<f64>())
        .stop_when(Stop::target(1e-10).or(Stop::evaluations(100_000)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    println!("step size at the end: {:e}", outcome.best_genome().step());
    Ok(())
}
```

### Differential evolution

For continuous problems on `Real` genomes, differential evolution often needs far fewer evaluations than a GA. `CR` (the crossover rate) matters most: small (e.g. 0.1) for separable functions, where each gene can be optimized on its own; large (0.9) for rotated or coupled ones.

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
        .population_size(50) // 5 to 10 times the number of genes
        .strategy(de::Strategy::CurrentToPBest { p: 0.1, archive: 1.0 })
        .control(de::Control::Fixed { f: 0.5, cr: 0.1 })
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

| `de::Strategy` | When |
|---|---|
| `Rand1` (default) | Robust; explores well |
| `CurrentToPBest { p: 0.1, archive: 1.0 }` | Faster, still diverse thanks to the archive |
| `Best1` | Fastest on easy problems; use it with `de::Control::Dither { min_f: 0.5, max_f: 1.0, cr }`, or the population can collapse before the optimum |

### Local search: hill climbing and simulated annealing

`LocalSearch` improves a single solution, moving to one of `neighbors` random neighbors per step. It often beats a GA on permutations. Any mutation is a neighborhood; `InversionMutation` (2-opt) is the classic one for tours.

| `.acceptance(...)` | Moves to the best neighbor when |
|---|---|
| `Acceptance::Improving` | it's strictly better (stops at a local optimum) |
| `Acceptance::NotWorse` (default) | it's better or equal (drifts across plateaus) |
| `Acceptance::Annealing { initial_temperature, cooling }` | it's better, or worse by Δ with probability exp(−Δ/T), T cooling every step |
| `Acceptance::Tabu { tenure }` | it isn't one of the last `tenure` solutions (even if worse), or beats the best so far; use several neighbors |

`.restart(patience, kicks)` adds iterated local search to any of them: after `patience` steps without a new best, the search restarts from the best solution changed by `kicks` neighbor moves.

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

Drive the algorithm by hand when fitness is computed elsewhere: another process, a simulator, a remote service or async code.

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

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| ``error[E0277]: `Unset` is not a crossover for `Binary` `` (or a selection or mutation) | That operator was not set | Call `.crossover(...)`, `.select(...)` or `.mutate(...)` before `.build()` |
| ``error[E0277]: the genes of `Order` can't be exchanged by position`` | Point or uniform crossover on a permutation would duplicate genes | Use a permutation crossover: `OrderCrossover`, `EdgeRecombinationCrossover`, `PartiallyMappedCrossover` or `CycleCrossover` |
| ``error[E0277]: `BitFlip` is not a mutation for `Integer` `` | Wrong mutation for the genome | Use the table in [Choosing the pieces](#choosing-the-pieces) |
| ``error[E0277]: `usize` is not a fitness value`` (then "the method `stop_when` exists … but its trait bounds were not satisfied") | The fitness function returns an integer | Return `f64` (`... as f64`), `Fitness` or `Option<f64>` |
| `no method named parallel` | Built without the default `parallel` feature | Enable the `parallel` feature, or drop `.parallel(true)` |
| `Error::MissingSetting { setting: "population_size" }` | No population size | `.population_size(n)` |
| `Error::MissingSetting { setting: "stop_when" }` | The engine has no stop condition | `.stop_when(Stop::generations(n))`, or an abort flag |
| `Error::InvalidSetting { setting, reason }` | A value is out of range | Read `reason`; see [Settings](#settings) |
| `Error::InvalidGenome { reason }` | An initial genome doesn't fit the representation | Match its length and bounds |
| `Error::NanFitness` | The fitness function returned NaN (a score or a violation) with `NanPolicy::Error` | Fix the fitness function, or keep the default `NanPolicy::Invalid` |
| `Error::InvalidFitness` | A negative constraint violation | A violation is 0 or more: use `constraint::at_most` and friends |
| The best solution is infeasible | No feasible solution found yet | Run longer, check the constraints can be met, or start from a feasible solution with `.initial_genomes(...)` |
| `Error::TellWithoutAsk` / `Error::FitnessCount` | Ask / tell out of step | One `tell` per `ask`, with one fitness per asked genome, in order |
| Hill climbing (`Acceptance::Improving` or `NotWorse`) stops improving | A local optimum | `.restart(patience, kicks)` (iterated local search), `Acceptance::Tabu { tenure }` with several neighbors, or `Acceptance::Annealing` with an initial temperature about the size of typical fitness differences and `cooling` close to 1 (e.g. 0.999) |
| A differential evolution stops improving far from the optimum, with a tiny population spread | The population collapsed (greedy strategy, fixed `F`) | `de::Control::Dither { min_f: 0.5, max_f: 1.0, cr }`, `Strategy::Rand1`, or an archive with `CurrentToPBest` |
| Real-valued search stalls in a local minimum | Steps too small to leave its basin (e.g. `GaussianMutation` with a tiny sigma) | `PolynomialMutation` with eta 20, or a larger sigma; on Rastrigin, sigma 0.03 of the range works and 0.01 stalls |
| The best fitness stops improving early | Too little diversity | A larger population, a smaller tournament, a higher mutation rate, or `Stop::stagnation` with restarts |
| Slow runs with a cheap fitness function | Debug build, or parallel overhead | Build with `--release`; use `.parallel(true)` only for expensive fitness functions |

## Guarantees to rely on

- **Reproducible:** the same seed and settings give the same results on every platform, with or without `.parallel(true)` and on any number of threads.
- **Deterministic ties:** the earlier individual wins.
- **The best is never lost:** `outcome.best()` is the best individual ever evaluated, even when it didn't survive.
- **No panics** in library code for invalid input: invalid settings are errors from `build()` and `run()`.
