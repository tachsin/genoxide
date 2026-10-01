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
| Programs and formulas: trees of typed functions (genetic programming) | `gp::Gp::builder(set)` ([template](#genetic-programming)) | `gp::Tree` (nodes in prefix order) | `gp::SubtreeCrossover`, `gp::OnePointCrossover` | `gp::SubtreeMutation`, `gp::PointMutation`, `gp::HoistMutation`, `gp::ShrinkMutation`, `gp::ConstantMutation`, a mix: `gp::Mutations` |
| A smooth function of reals, a local minimum to many digits, any number of genes | `Real::new(...)`, its box the bounds | `Reals` | none: `Lbfgsb` with the gradient ([template](#l-bfgs-b-smooth-functions-with-a-gradient)) | none |
| A neural network's weights (neuroevolution) | `nn::Mlp::new([4, 8, 1], nn::Activation::Tanh)?.representation(-1.0..=1.0)?`, `nn::Elman` (recurrent) ([template](#neuroevolution-a-networks-weights-by-cma-es)) | `Reals` | none: `Cmaes` (up to a few hundred weights), `OpenEs` (thousands and more) | none |
| A smooth function of many reals, its gradient noisy (mini-batches) or a step set by a learning-rate schedule (model fitting, up to millions of parameters) | `Real::uniform(n, lo..=hi)` ([template](#first-order-methods-adam-momentum-nesterov)) | `Reals` | none: `FirstOrder` | none |
| Smooth, with gradients: very many reals (up to millions), few inequality constraints | `Real::uniform(n, lo..=hi)` ([template](#many-variables-few-constraints-mma)) | `Reals` | none: `Mma` | none |

| Continuous problem | Method |
|---|---|
| Smooth, gradient available (or n + 1 evaluations per gradient affordable) | `Lbfgsb`: fastest to many digits, any number of genes, bounds landed on exactly |
| Smooth or not, a few genes, no gradient | `NelderMead` |
| Multimodal, rotated or badly conditioned, up to a few hundred genes | `Cmaes` (with `Restarts::Ipop`), `De`; then `Lbfgsb` from the best to polish it |
| Smooth, gradients of the score and of each constraint, very many genes and few inequality constraints | `Mma` (`Method::Gcmma` to converge from any start) |
| Smooth, solved best in stages (a smoothing, sharpness or penalty changed step by step) | `Continuation` around `FirstOrder`, `Lbfgsb`, `Mma`, `NelderMead` or `Cmaes` ([template](#continuation-stages-of-one-problem-the-state-kept)) |
| Constrained beyond the box, without gradients | `De` or `Ga` with `(score, violation)` (Deb's rules) |

Any selection fits any representation; usually `Tournament` of size 2 to 5. For trees, a selection against bloat (growth without better fitness): `DoubleTournament::new(7, 1.4)?`, `LexicographicTournament::new(7)?` (ties in fitness to the smaller), or `Tarpeian::new(select, rate)?`; size is `genome.len()`.

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
| `.parallel_breeding(true)` | off | `parallel` feature; crossover and mutation of each pair on rayon, on its own random stream: seeded results differ from off, not between thread counts; for thousands of children, operators with real work per gene, fast parallel evaluation |

`.memetic`: each of the best `parents` takes the best of `neighbors` mutated neighbors if not worse (Lamarckian). `.build()?` returns `Error::MissingSetting` or `Error::InvalidSetting`, naming the setting.

During a run (parameter control, e.g. an annealed mutation step): `ga.set_crossover_rate(p)?`, `ga.set_mutation_rate(p)?` (validated as in the builder), and `ga.select_mut()`, `ga.crossover_mut()`, `ga.mutate_mut()` to replace an operator, e.g. `*ga.mutate_mut() = GaussianMutation::per_gene(0.5, sigma)?`. A change applies from the next generation's breeding. The others: `de.set_control(...)?`, `de.set_strategy(...)?`, `pso.set_inertia(w)?` (e.g. 0.9 falling to 0.4), `pso.set_acceleration(c1, c2)?`, `search.neighbor_mut()`, `search.set_neighbors(n)?`. In an `Engine`, make the change in `.control(|algorithm, progress| ...)`; with `Islands`, through `islands.islands_mut()`.

### Operators

| Constructor | Valid |
|---|---|
| `Tournament::new(size)?` | size ≥ 1 |
| `Rank::new(pressure)?`, `Rank::default()` | 1 ≤ pressure ≤ 2, default 1.5 |
| `Truncation::new(fraction)?` | 0 < fraction ≤ 1 |
| `Roulette`, `StochasticUniversalSampling`, `RandomSelection`, `NoCrossover` | unit structs |
| `LexicographicTournament::new(size)?` | size ≥ 1; of equal fitness, the smaller genome wins |
| `DoubleTournament::new(fitness_size, parsimony)?`, `.size_first()` | size ≥ 1, 1 ≤ parsimony ≤ 2: a size tournament of two fitness-tournament winners, the smaller winning with probability parsimony / 2; 7 and 1.4 against bloat |
| `Tarpeian::new(select, rate)?` | 0 < rate ≤ 1: each call, genomes larger than the mean count as invalid with probability rate, then `select` |
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
| `gp::SubtreeCrossover::new()`, `with_internal_rate(p)?` | `Gp`; points at function nodes with probability p (0.9, Koza's), else at leaves; the second point of the same type, chosen so both children stay within the limits |
| `gp::SubtreeMutation::new()`, `with_max_depth(d)?` | `Gp`; a node chosen uniformly gets a new subtree of its type, grown to depth d (default 4) within the limits; never the same subtree |
| `gp::OnePointCrossover` | `Gp`; a point of the two trees' common region (same shape from the root), exchanged; within the limits |
| `gp::PointMutation::per_node(rate)?`, `count(n)?` | `Gp`; a node replaced by another primitive of its signature, a leaf by another terminal or a new constant; nodes without one never picked |
| `gp::HoistMutation`, `gp::ShrinkMutation` | `Gp`, unit structs; the tree replaced by one of its subtrees of its type, a function's subtree by a leaf: always smaller |
| `gp::ConstantMutation::gaussian(sigma)?` | `Gp`; sigma > 0: one constant plus normal noise, sigma a fraction of its range, mirrored at the ends |
| `gp::Mutations::builder().subtree(w).point(w).hoist(w).shrink(w).with(w, m).build()?` | `Gp`; one mutation per call by weight, another if it can't change the tree; weights ≥ 0, total > 0 |

`per_gene(rate)` changes each gene with that probability (at `1 / length`, a third of the children are unevaluated copies); `count(n)` exactly `n` genes. A picked gene always changes.

### `Engine::new(algorithm, fitness)`

| Method | Notes |
|---|---|
| `.stop_when(stop)` | required without an abort flag; calls combine with "or" |
| `.observe(observer)` | `&mut observer` to read it afterwards |
| `.on_generation(\|snapshot\| ...)` | after every generation |
| `.control(\|algorithm, progress\| ...)` | `&mut` the algorithm once per generation, after the observers and stops, before a checkpoint: parameter control, or `algorithm.reevaluate()?`; returns `Result<()>` |
| `.parallel(true)` | rayon, same results; for expensive fitness functions; no effect on a `Batch` |
| `.abort_flag(Arc<AtomicBool>)` | stops after the current generation once set |
| `.nan_policy(NanPolicy::Error)` | NaN is an error, not invalid (`NanPolicy::Invalid`, default) |

Stops: `Stop::target(score)` (at least as good), `generations(n)`, `evaluations(n)`, `time(duration)`, `stagnation(n)`, `custom(|progress| ...)`, combined with `.or(...)` and `.and(...)`, checked after every generation. With only targets and evaluation limits, a run stops as `StopReason::Stalled` after `genoxide::engine::STALL_GENERATIONS` (10 000) generations with nothing to evaluate. An algorithm that has finished (`Algorithm::is_finished`: a local method that has converged with no restart left, or a CMA-ES with `cmaes::Restarts::Stop` that has converged) stops it as `StopReason::Converged`, unless a stop condition is met in the same generation.

## Fitness functions

- A closure `|genome: &G| -> T` (`f64`, `Fitness` or `Option<f64>`) or a `FitnessFunction<G>`. Deterministic: a copy of a parent inherits its fitness.
- Maximize is the default; use `.minimize()`, don't negate.
- `None`, `Fitness::invalid()` and NaN are invalid: worse than everything.
- **Constraints:** return `(score, violation)`, 0 when feasible, adding up `constraint::at_most(value, limit)`, `at_least`, `equal(value, target, tolerance)`. Deb's rules: feasible beats infeasible, then score or violation decides. Select with `Tournament` or `Rank`: roulette and SUS give infeasible solutions no weight. `Penalty::new(weight)?.fitness(objective, score, violation)` is a static penalty instead.
- **Test problems:** `problems::{Sphere, AxisParallelEllipsoid, Schwefel1_2, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel2_26, Levy, Zakharov, StyblinskiTang, Michalewicz, Schwefel2_21, Schwefel2_22, DixonPrice, Trid, Powell}::new(n)` (`Powell` takes a multiple of 4) and `problems::{Himmelblau, Branin, GoldsteinPrice, SixHumpCamel, Hartmann3, Hartmann6, Shekel5, Shekel7, Shekel10, Easom, Eggholder, SchafferF6, Beale, Booth, Matyas, Bohachevsky1, Bohachevsky2, Bohachevsky3, ThreeHumpCamel, Langermann, ShekelFoxholes, Kowalik}` are fitness functions for `Engine::new(algorithm, problem)`, all minimized. The `problems::Problem` trait gives `representation()` (the bounds), `optimum()` (`value()`, `solutions()`), `reference()`; `problems::all()` lists them as `Box<dyn DynProblem>`. Constrained, with fitness `(score, violation)` and `constraints(&x)` (`g <= 0`, then `h = 0`): `problems::cec2006::{G01, …, G24}` (equalities met within `EQUALITY_TOLERANCE` = 1e-4; `with_tolerance(δ)` for the problems with equalities, e.g. `G03::with_tolerance(δ)`), and `problems::engineering::{WeldedBeam, WeldedBeamRagsdell, PressureVessel, TensionCompressionSpring, SpeedReducer, ThreeBarTruss, CantileverBeam, CarSideImpact}`. `PressureVessel` and `SpeedReducer` round their discrete genes when evaluated; `design(&x)` gives the rounded design. `engineering::GearTrain` has an `Integer` genome and isn't in `all()`. `Optimum::is_proven()` is false for a best known value.
- **Extras:** return `Evaluated::new(value, info)` (`value` any of the above, `info` any `Send + Sync + 'static` type, e.g. a struct with a penalty's terms) to keep what the fitness function computed. Read it by type: `outcome.best_info::<T>()`, `snapshot.info::<T>(genome)` and `snapshot.best_info::<T>()` in `.on_generation`, `hall_of_fame.info::<T>(genome)`, and in `MultiEngine` `snapshot.info` and `outcome.info(genome)` for the front; `None` for another type. Never used by the search. Kept by genome for the population, the discarded and the best (copies share it); not in checkpoints.
- **Gradients:** `Differentiable(|x: &Reals, gradient: &mut [f64]| value)`, for gradient-based methods; see [Gradients](#gradients-supplying-them).
- **Constraint values:** `Constrained::new(m, |x: &Reals, g: &mut [f64]| score)` writes the values of m constraints `gᵢ(x) <= 0`; `Constrained::differentiable(m, |x, gradient, g, jacobian| score)` also the gradient and the Jacobian (`jacobian[i * n + j]` = ∂gᵢ/∂xⱼ). Either is `(score, Σ max(0, gᵢ))` for any algorithm, and gives the values one by one to those that use them (`Mma`). The CEC 2006 problems with inequalities only and the engineering problems give their values (`problem.provides().inequalities`), not their gradients.
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

### Genetic programming

Trees of the user's primitives (an enum), strongly typed: declare types, functions (argument types, return type), terminals and ephemeral random constants, then match on the enum in the fitness function. `Gp::builder(set)`: `.max_depth(17)`, `.max_size(1024)`, `.init(gp::Init::RampedHalfAndHalf { depths: 2..=6 })` (also `Full`, `Grow`), the defaults. `set.parse("add(x, 1.0)")?` and `tree.display(&set)` read and write trees; `gp.validate(&tree)` checks types and limits. Evaluate with `tree.evaluate(&set, &mut stack, |op, args: &[f64]| ..., |ty, constant| ...)` per point, `tree.evaluate_columns(&set, &mut gp::Columns::new(points), |op, args, out| ...)` on all points at once (several times faster for data, the same bits), or walk `tree.root(&set)` top-down. Seed a population with `gp.ramped_half_and_half(n, &mut rng)?` (Koza's even division, no duplicates). Python has genoxide's built-in primitives, those of `gp::regression` and `gp::boolean`, and your own, evaluated by numpy (below).

```rust
use genoxide::gp::{Constants, Gp, Mutations, PrimitiveSet, SubtreeCrossover, Tree};
use genoxide::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Add,
    Sub,
    Mul,
    X,
}

fn main() -> genoxide::Result<()> {
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real"); // one type: untyped GP
    set.function("add", Op::Add, [real, real], real)
        .function("sub", Op::Sub, [real, real], real)
        .function("mul", Op::Mul, [real, real], real)
        .terminal("x", Op::X, real)
        .constants(real, Constants::integers(-2..=2)?);
    let gp = Gp::builder(set.build(real)?).build()?;
    let set = gp.primitives().clone();

    // find x³ − 2x from 21 points
    let xs: Vec<f64> = (0..=20).map(|i| f64::from(i) / 10.0 - 1.0).collect();
    let error = |tree: &Tree| {
        let mut stack = Vec::new();
        let mut sum = 0.0;
        for &x in &xs {
            let y = tree.evaluate(
                &set,
                &mut stack,
                |op, args: &[f64]| match op {
                    Op::Add => args[0] + args[1],
                    Op::Sub => args[0] - args[1],
                    Op::Mul => args[0] * args[1],
                    Op::X => x,
                },
                |_, constant| constant,
            );
            sum += (y - (x * x * x - 2.0 * x)).abs();
        }
        sum
    };
    let initial = gp.ramped_half_and_half(500, &mut StreamRng::seed_from_u64(1))?;
    let ga = Ga::builder(gp)
        .population_size(500)
        .initial_genomes(initial)
        .select(DoubleTournament::new(7, 1.4)?) // against bloat
        .crossover(SubtreeCrossover::new())
        .mutate(Mutations::builder().subtree(0.5).point(0.5).build()?)
        .mutation_rate(0.3)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(ga, error)
        // not 0: the formula in another order of operations can differ by rounding
        .stop_when(Stop::target(1e-9).or(Stop::generations(200)))
        .run()?;
    println!("{}", outcome.best_genome().display(&set));
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

- Types: `set.new_type("bool")`, `set.function("less", Op::Less, [real, real], boolean)`; every tree genoxide makes is typed. `build` rejects a set whose types can't make a tree.
- The primitive set is data, not closures, so a `Gp<Op>` checkpoints when `Op` derives `Serialize` and `Deserialize`.
- For many points, `evaluate_columns` with a `thread_local!` `Columns` per thread; `gp::regression` does this for symbolic regression (below).
- Bloat (trees growing without better fitness): `DoubleTournament`, `LexicographicTournament` or `Tarpeian` as the selection, and hoist and shrink in the mutations; Koza's depth limit of 17 always holds.
- Symbolic regression: `gp::regression::primitives([Math::Add, Math::Mul, Math::Aq, Math::Sin], ["x", "y"], constants)?` makes a set of `gp::regression::Math` (also `Div` (IEEE), `Inv`, `ProtectedDiv`, `ProtectedLog`, `Exp`, `Sqrt`, ...), `constants` e.g. `Some(gp::Constants::normal(0.0, 5.0)?)` (Keijzer's; also `uniform`, `integers`, `choice`), `Dataset::new(Sample::new(columns, targets)?).with_test(sample)?` the data, and `Regression::new(set, dataset)?` the fitness function: RMSE (`.metric(Metric::Mae)`) after linear scaling (`.linear_scaling(false)` for the tree itself), `None` where a value isn't finite; `regression.display(&tree)` prints the scaled expression, `regression.error(&tree, test)` the test error. Problems: `gp::regression::problems::{Koza1, Koza2, Koza3, Nguyen1, …, Nguyen12}` (`primitives()`, `dataset()`, `regression()`), fitness functions themselves; see `examples/koza_quartic` and `examples/nguyen_1`. Error against size as two objectives: `Nsga2::builder(gp, [Minimize, Minimize])` with `|tree| Some([regression.evaluate(tree)?, tree.len() as f64])`; see `examples/accuracy_and_size`.
- Python: `gx.gp.regression.primitives(["add", "mul", "aq", "sin"], ["x", "y"], gx.gp.Constants.normal(0.0, 5.0))` (functions by name, `gx.gp.regression.FUNCTIONS`), `gx.gp.Gp(set, max_depth=17, max_size=1024, init=gx.gp.RampedHalfAndHalf((2, 6)))` the genome (`ramped_half_and_half(n, seed)` for `gx.Ga(initial_genomes=...)`, `parse`, `validate`), `gx.gp.Tree` the fitness function's argument (`len`, `depth`, `str`, `evaluate(x)` on a point per row), `gx.gp.SubtreeCrossover()`, `OnePointCrossover()`, `SubtreeMutation()`, `PointMutation(rate=... | count=...)`, `HoistMutation()`, `ShrinkMutation()`, `ConstantMutation(sigma)`, `Mutations([(weight, mutation), ...])`, and `gx.DoubleTournament(7, 1.4)`, `gx.LexicographicTournament(size)`, `gx.Tarpeian(select, rate)`. `gx.gp.regression.Regression(set, Dataset(Sample(x, y), test), metric=..., linear_scaling=...)`, the problems `gx.gp.regression.problems.Koza1()` ... `Nguyen12()` (`regression(linear_scaling=False)`), `gx.gp.boolean.Multiplexer(3)`, `EvenParity(n)` and `gx.gp.WithSize(fitness)` (error and size, for `gx.Nsga2`) are evaluated in Rust; a Python function of a `Tree` works too. Your own primitives: `b = gx.gp.PrimitiveSetBuilder()`, `real = b.new_type("real")` (a type is its name), `b.function("less", [real, real], boolean)`, `b.terminal("x", real)`, `b.constants(real, gx.gp.Constants.integers(-2, 2))`, `b.build(real)`; evaluate with `tree.evaluate({"x": xs}, {"less": np.less, "if": np.where, ...})`, one call per node on numpy columns (constants are floats that broadcast), the same with numbers and plain functions per point, or interpret `tree.nodes()` (`kind`, `name`, `arity`, `type`, `value`, prefix order) yourself; see `examples/abs_typed`. The set is data: it checkpoints, the functions come with the fitness function on resume. Trees run with `Ga`, `Islands` of `Ga`s and `Nsga2` of 2 objectives.
- Boolean problems: `gp::boolean::Multiplexer::new(3)?` (Koza's 11-multiplexer), `EvenParity::new(n)?`, fitness functions (cases wrong, minimized) for trees of their `primitives()` (`gp::boolean::Logic`); see `examples/multiplexer_11`. A typed set (`less` returning a Boolean, `if` taking one): `examples/abs_typed`.

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

### Neuroevolution: a network's weights by CMA-ES

`nn::Mlp::new(layers, activation)?` (`.output_activation(a)`, `.bias(false)`; weights unit by unit: each unit's input weights in order, then its bias) and `nn::Elman::new(inputs, hidden, outputs, activation)?` (a hidden layer that also sees its previous outputs; `reset()` between episodes) have `parameters()` weights, `representation(bounds)?` for the genome, and `with(&weights)?` for a network whose `forward(&input, &mut output)` doesn't allocate. Activations: `Identity`, `Tanh`, `Sigmoid`, `Relu`, through `genoxide::math`. `problems::control::{CartPole, DoublePole}` are pole-balancing tasks (Florian's corrected equations, Gomez et al. 2008's settings) driven by a `Policy`: the networks, or a closure `|observation: &[f64], action: &mut [f64]|`. `task.run(&mut policy, steps)` gives the steps balanced from the start (100,000, `SUCCESS_STEPS`, solves it); `DoublePole::without_velocities()` observes `x`, `θ₁`, `θ₂` only and has Gruau et al.'s `damping_fitness` and `generalization` test (`solved` applies both criteria). No biases suit these symmetric tasks. Python: `gx.nn.Mlp(layers, "tanh", output_activation=..., bias=...)`, `gx.nn.Elman(inputs, hidden, outputs, ...)` (`parameters`, `representation((lo, hi))`, `forward(weights, X)` and `policy(weights)` in Rust), `gx.problems.control.{CartPole, DoublePole}` (`run(policy, steps)`, `solved`, `damping_fitness`, `generalization`; a policy is a network's `policy(weights)` or a slow Python callable `(observation, action)`), and `Balance(task, network, fitness="steps" | "damping")`, a fitness evaluated in Rust, maximized, on a `Real` of `network.parameters` genes.

```rust
use genoxide::nn::{Activation, Mlp};
use genoxide::prelude::*;
use genoxide::problems::control::{DoublePole, SUCCESS_STEPS};

fn main() -> genoxide::Result<()> {
    // a 6-6-1 network without biases balances two poles for 100,000 steps
    let mlp = Mlp::new([6, 6, 1], Activation::Tanh)?
        .output_activation(Activation::Tanh)
        .bias(false);
    let task = DoublePole::new();
    let steps = |weights: &Reals| -> Option<f64> {
        let mut network = mlp.with(weights).ok()?;
        Some(f64::from(task.run(&mut network, SUCCESS_STEPS)))
    };
    let cmaes = Cmaes::builder(mlp.representation(-1.0..=1.0)?)
        .restarts(cmaes::Restarts::Ipop)
        .seed(1)
        .build()?;
    let outcome = Engine::new(cmaes, steps)
        .stop_when(Stop::target(f64::from(SUCCESS_STEPS)).or(Stop::evaluations(100_000)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    let mut network = mlp.with(outcome.best_genome())?;
    assert!(task.solved(&mut network));
    Ok(())
}
```


### Neuroevolution: thousands of weights by OpenAI's evolution strategy

`OpenEs::builder(real)` (Salimans et al. 2017): mirrored samples `mean ± σ ε`, centered ranks, and an optimizer step along the gradient estimate, O(genes) per sample. `.population_size(n)` required, even; `.sigma(0.02)` and the learning rate are fractions of each gene's range; `.optimizer(open_es::Optimizer::adam(0.01))` (default) or `Optimizer::sgd(rate, momentum)`; `.weight_decay(d)` (0); `.evaluate_mean(true)` asks for the mean too each generation (often the best network); `.initial_mean(genome)` (random by default: small weights, e.g. from `Real::uniform(n, -0.25..=0.25)?`, suit networks); `.parallel_breeding(true)` for thousands of genes. `open_es.mean()`; `set_sigma`, `set_learning_rate` in `.control`. Adam steps about the learning rate per gene even near the optimum: lower it for precise answers. See `examples/two_spirals`. Python: `gx.OpenEs(real, population_size=..., sigma=..., optimizer=gx.Adam(lr) | gx.Sgd(lr, momentum), weight_decay=..., evaluate_mean=..., initial_mean=..., parallel_breeding=...)`, `gx.RunningOpenEs` (`sigma`, `learning_rate`) in `control`; `gx.Real(...).random_genome(seed)` is Rust's `random_genome` of `StreamRng::seed_from_u64(seed)`, and `gx.math` genoxide's portable math.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    let sphere = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
    let open_es = OpenEs::builder(Real::uniform(100, -5.0..=5.0)?)
        .population_size(50)
        .sigma(0.01)
        .optimizer(open_es::Optimizer::adam(0.003))
        .evaluate_mean(true)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(open_es, sphere)
        .stop_when(Stop::target(0.1).or(Stop::generations(2_000)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

### Neuroevolution: a network's structure by NEAT

`neat::Neat::builder(inputs, outputs)` evolves `neat::Network`s (a bias input is added) from minimal networks up, with the NEAT paper's settings: population 150, speciation (`.compatibility(c1, c2, c3, threshold)`, 1, 1, 0.4, 3), `.structural_mutation(add_node, add_connection)` (0.03, 0.05), `.weight_mutation(rate, replace)` (0.8, 0.1), stagnation 15. Maximized by default; `.sharing(neat::Sharing::Raw)` is the paper's fitness sharing (maximized, non-negative scores), `Normalized` (default) any objective. Networks stay feed-forward (`.feed_forward(false)` for recurrent ones); evaluate with `network.feed_forward()?` and `activate(&input, &mut output)`, no allocation, or a recurrent one with `network.recurrent()?` (one time step per `activate`, `reset()` between episodes). Both are `problems::control::Policy`s; NEAT's sigmoid outputs are in (0, 1), so map them to a force with `2.0 * output - 1.0` (see `examples/double_pole_no_velocities`). `neat.species()`, `network.hidden()`, `network.enabled()`. Python: `gx.Neat(inputs, outputs, population_size=..., compatibility=(c1, c2, c3, t), structural_mutation=..., weight_mutation=..., sharing="raw" | "normalized", feed_forward=..., ...)`; the fitness function gets a `gx.neat.Network` (`hidden()`, `enabled()`, `nodes()`, `connections()`), whose `feed_forward()` and `recurrent()` evaluators have `activate(input)` and are control policies run in Rust, `policy(scale=2.0, offset=-1.0)` for a force from the sigmoid; `gx.RunningNeat.species` in `control`; `task.episode(policy, steps)` gives an episode's states.

```rust
use genoxide::neat::{Neat, Network, Sharing};
use genoxide::prelude::*;

const CASES: [([f64; 2], f64); 4] =
    [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)];

fn main() -> genoxide::Result<()> {
    // the paper's XOR fitness: (4 - sum of |error|)^2, 16 for a perfect network
    let xor = |network: &Network| {
        let mut evaluator = network.feed_forward().expect("feed-forward");
        let mut output = [0.0];
        let error: f64 = CASES
            .iter()
            .map(|(input, target)| {
                evaluator.activate(input, &mut output);
                (output[0] - target).abs()
            })
            .sum();
        (4.0 - error).powi(2)
    };
    let neat = Neat::builder(2, 1).sharing(Sharing::Raw).seed(1).build()?;
    let outcome = Engine::new(neat, xor)
        .stop_when(Stop::target(15.0).or(Stop::generations(500)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    assert!(outcome.best_genome().hidden() >= 1); // XOR needs a hidden node
    Ok(())
}
```
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
| `.parallel_breeding(true)` | Off by default; `parallel` feature. Each offspring made on rayon, on its own random stream: seeded results differ from off, not between thread counts. Making offspring (a normal number and an `exp` per gene, a `ln` per parent and gene with intermediate recombination) is most of a generation with a fast fitness function |

A GA can run an ES too: `AdaptiveReal`, `SelfAdaptiveMutation`, `NoCrossover`, `Scheme::MuCommaLambda`. For hard problems (rotated, badly conditioned or multimodal), CMA-ES is stronger.

### Differential evolution

For continuous problems on `Real` genomes, differential evolution often needs far fewer evaluations than a GA. Defaults from SHADE (Tanabe and Fukunaga, CEC 2013): current-to-pbest/1 with a random `p` in [2 / size, 0.2], an archive of population size, `F` / `CR` memory of 100, 100 individuals; plus genoxide's restarts, `de::Restarts::OnStagnation { tolerance: 1e-12, patience: 200 }`: all but the best replaced once every gene's spread is within 1e-12 of its range (and the scores' within 1e-12 of the best), or after 200 generations without a better best. Options: `.population_size(n)`, `.control(de::Control::Fixed { f, cr })` (`CR` 0.1 for separable, 0.9 for rotated functions), `.restarts(de::Restarts::Never)`, `.parallel_breeding(true)` (`parallel` feature: each trial built on rayon, on its own random stream, as a GA's breeding; for hundreds of individuals or genes and a fast fitness function evaluated in parallel).

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

The strongest general choice for continuous problems with up to a few hundred `Real` genes, especially when the genes interact (rotated or badly conditioned functions). Nothing needs tuning (initial step: 0.3 of each range). For multimodal functions, add `cmaes::Restarts::Ipop` (growing population) or `Bipop` (large and small in turn). For thousands of genes or separable problems: `.covariance(cmaes::Covariance::Diagonal)` (sep-CMA-ES, O(n) per sample, no correlations). `.min_step(fraction)` (0 to the initial step, 0 by default) bounds the step size below, so the search keeps exploring when the fitness stops pointing at the goal (Igel 2003, on control tasks scored on short episodes). Without restarts, a converged run goes on sampling around its point until a stop condition (`Restarts::Never`, the default); `cmaes::Restarts::Stop` ends it there instead, as `StopReason::Converged`, like Hansen's reference code. That saves most of a budget on smooth problems without constraints; on flat or quantized fitness (plateaus) and with constraints, the criteria can fire while the best would still improve, so keep `Never` there, or stop with a target.

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
        .topology(Topology::Ring) // or FullyConnected, Random, or Isolated: no migration
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

Fitness: `[f64; M]`, `(values, violation)` or `Option<[f64; M]>`. `MultiEngine` returns the Pareto front: each genome once (the first of its copies), like every generation's `snapshot.front()`.

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
- Constrained dominance: feasible first, then the smaller violation (`multi::dominates`). `Stop::stagnation` counts generations whose front has no solution that the previous front didn't dominate or equal: with a front larger than the population, it may never fire, so add `Stop::generations`; `Stop::target` isn't available.
- `multi::non_dominated_sort`, `multi::crowding_distance`; `multi::ParetoArchive::new(objectives)` with `.on_generation(|snapshot| archive.update(snapshot))` keeps every non-dominated solution.
- Test problems: `multi::problems::{Zdt1, Zdt2, Zdt3, Zdt4, Zdt6}::new(n)`, `Zdt5` (a `Binary` genome of 80 bits, not in `all()`), `{Dtlz1, …, Dtlz7}::<M>::new(n)` (the technical report's numbering), `{Wfg1, …, Wfg9}::<M>::new(k, l)` (k position parameters, a multiple of M − 1; l distance parameters, even for WFG2 and WFG3; `default()`: k = 4 for 2 objectives, 2(M − 1) for more, l = 20), `{FonsecaFleming, Kursawe}::new(n)`, `Schaffer1`, `Schaffer2`, `Poloni`, `Viennet1`/`2`/`3` (3 objectives), and the constrained `Bnh`, `Srn`, `Tnk`, `Osy`, `Constr` (fitness `([f64; 2], violation)`), all minimized, for `MultiEngine::new(algorithm, problem)`. The `multi::problems::MultiProblem<M>` trait gives `representation()`, `optimal_front(points)` (`Option`: `None` for KUR, POL, VNT2, VNT3, DTLZ5, DTLZ6 with 4 or more objectives, and WFG3 with 3 or more), `ideal_point()`, `nadir_point()`, `constraints(&x)` (`g <= 0`), `reference()`; `multi::problems::all::<M>()` lists them as `Box<dyn DynMultiProblem<M>>`.
- Constrained test problems of tunable difficulty: `multi::problems::{Ctp1, …, Ctp8}` (two objectives, two variables, disconnected fronts, fronts of separate points, infeasible bands and tunnels) and `{C1Dtlz1, C1Dtlz3, C2Dtlz2, ConvexC2Dtlz2, C3Dtlz1, C3Dtlz4}::<M>::new(n)` (Jain and Deb's constrained DTLZ; `C1Dtlz3` and `ConvexC2Dtlz2` take the paper's radius for 3, 5, 8, 10 and 15 objectives, else `with_radius(n, r)`), all with `optimal_front`, and in `all()`.
- More test problems, with their fronts: Deb and Jain's `{ConvexDtlz2, ScaledDtlz1, ScaledDtlz2, InvertedDtlz1}::<M>::new(n)` (scaled: the paper's factor for M, or `.with_factor(s)`), and Ma and Wang's constrained `Mw1`, `Mw2`, `Mw3`, `Mw5`, `Mw6`, `Mw7`, `Mw9`, …, `Mw13` (2 objectives, `new(n)`, n = 15 by default) and `{Mw4, Mw8, Mw14}::<M>::new(n)` (M + 12 by default), all in `all::<M>()`.
- Advanced constrained suites, with their fronts, all in `all::<M>()`: the DTLZ report's `{Dtlz8, Dtlz9}::<M>::new(n)` (10M by default; DTLZ8 from 3 objectives), Li et al.'s DC-DTLZ `{Dc1Dtlz1, Dc1Dtlz3, Dc2Dtlz1, Dc2Dtlz3, Dc3Dtlz1, Dc3Dtlz3}::<M>::new(n)` (`with_parameters(n, a, b)`), and Fan et al.'s `DasCmop1` to `DasCmop9` (2 objectives to `DasCmop6`, 3 from `DasCmop7`; `new(n, difficulty)`, `with_difficulty(d)`, 30 variables and the paper's figure 6 triplet by default), whose `Difficulty::new(η, ζ, γ)` or `Difficulty::standard(1..=16)` (the paper's table 3) sets the constraints' difficulty; their fronts are sampled per triplet.
- Engineering designs with several objectives, `multi::problems::engineering::{TwoBarTruss, WeldedBeam, DiscBrake, SpeedReducer, FourBarTruss}` (2 objectives), `{CarSideImpact, RocketInjector, VehicleCrashworthiness, MarineDesign}` (3) and `WaterResourcePlanning` (5), in `all::<M>()`: the trusses' and `WaterResourcePlanning`'s fronts are derived; the others' `optimal_front` is `None`, with an `ideal_point()` (and `nadir_point()` for 2 objectives). `DiscBrake` and `SpeedReducer` round their integer gene; `design(&x)` gives the rounded design. Python: `gx.problems.multi_engineering`.
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

### Nelder-Mead: a local method without derivatives

`NelderMead` on `Real` genomes: a simplex of n + 1 points that reflects, expands, contracts and shrinks to a minimum (Lagarias et al. 1998's steps, Gao and Han's coefficients). For low dimensions (up to about 10 genes), non-smooth or noisy functions, a few hundred evaluations, and polishing what a global method found (`.initial_genome(outcome.best_genome().clone())`). It converges to the minimum of the basin it starts in, then stops as `StopReason::Converged`; restarts find other minima. It only compares values, so invalid and constrained fitness work as everywhere.

| Setting | Default |
|---|---|
| `.coefficients(nelder_mead::Coefficients::...)` | `Adaptive` (Gao and Han), `Standard` (1, 2, 1/2, 1/2), `Custom { reflection, expansion, contraction, shrink }` |
| `.initial_step(fraction)` | 0.1 of each gene's range, 0 < fraction ≤ 1 |
| `.initial_step_absolute(distance)` | instead of a fraction: the same distance in every gene, for a wide box around an unbounded problem (e.g. ±1e10) |
| `.tolerance(fraction)` | 1e-9: converged when every vertex is this close to the best in each gene, relative to the initial step (1e-10 of the range by default); below 1 |
| `.restarts(local::Restarts::Random { times })` | `Never`; each restart from a random point |
| `.speculative(true)` | off: the reflection, expansion and both contractions in one round, the same path in fewer rounds, for `.parallel(true)` |
| `.initial_genome(genome)` | random |

A generation is a round of evaluations: n + 1 for a new simplex, 1 trial point (4 speculative), or n for a shrink. Points outside the bounds are mirrored back in, so the simplex can't flatten on a bound; a minimum on a bound costs more evaluations than one inside. `nelder_mead.converged()`, `size()`, `iterations()`, `restart_count()`.

```rust
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    // Rosenbrock's valley, from the classic start
    let rosenbrock = |x: &Reals| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2);
    let nelder_mead = NelderMead::builder(Real::uniform(2, -5.0..=5.0)?)
        .initial_genome(Reals::from(vec![-1.2, 1.0])) // random by default
        .restarts(local::Restarts::Random { times: 3 }) // then 3 runs from random points
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(nelder_mead, rosenbrock)
        .stop_when(Stop::evaluations(10_000)) // a stop condition is still required
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!(outcome.best_fitness().score().unwrap() < 1e-15);
    Ok(())
}
```

### L-BFGS-B: smooth functions with a gradient

`Lbfgsb` on `Real` genomes (Byrd, Lu, Nocedal and Zhu 1995): the limited-memory BFGS method with the box of the genome as bounds. Each iteration finds the generalized Cauchy point (the first minimum of a quadratic model along steepest descent bent at the bounds), minimizes the model over the genes left free (projected into the box, Morales and Nocedal 2011) and searches along the step with Moré-Thuente. The strongest local method for smooth functions, from a few genes to millions (O(m · n) memory and work per iteration, no n × n matrix, no allocation after the first iteration); a minimum on a bound is landed on exactly. It converges to the minimum of the basin it starts in and stops as `StopReason::Converged`; `lbfgsb.converged()` says why. It uses the score only (a violation is ignored by the search).

| Setting | Default |
|---|---|
| `.gradients(gradient::Gradients::...)` | `Auto`: the fitness function's gradient if it provides one (`Differentiable`, the smooth test problems), else forward differences (n more points per gradient, in the same round; up to 10⁴ genes); `Supplied`, `Forward { step }`, `Central { step }` |
| `.memory(m)` | 10 correction pairs, at least 1; 3 to 20 usual |
| `.gradient_tolerance(t)` | 1e-5: converged when the projected gradient's largest component is at most this (absolute; forward differences rarely meet much less than 1e-7 of the function's scale) |
| `.function_tolerance(t)` | 1e7 ε ≈ 2.2e-9: converged when a step lowers f by at most t · max(\|f\|, 1); 0 to go on while a step lowers f |
| `.max_line_search(n)` | 20 trials per line search |
| `.restarts(local::Restarts::Random { times })` | `Never` |
| `.initial_genome(genome)` | random; e.g. `outcome.best_genome().clone()` of a global method, to polish it |

`lbfgsb.gradients()` (the source resolved for the run), `gradient_evaluations()`, `stencil_evaluations()` (the cost of finite differences, part of `evaluations()`), `iterations()`, `projected_gradient()`, `pairs()`, `set_memory(m)?` (in `.control`). `reevaluate()` drops the pairs (`reevaluate_keeping_pairs()` keeps them). A round (generation) is one trial point with its stencil; near a minimum, one iteration.

```rust
use genoxide::gradient::{Differentiable, Gradients};
use genoxide::prelude::*;
use genoxide::problems::{Problem, Rosenbrock};

fn main() -> genoxide::Result<()> {
    // a test problem with its analytic gradient: Rosenbrock in 100 dimensions
    let problem = Rosenbrock::new(100);
    let start: Reals = (0..100).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
    let lbfgsb = Lbfgsb::builder(problem.representation())
        .initial_genome(start)
        .minimize()
        .build()?;
    let mut engine = Engine::new(lbfgsb, problem).stop_when(Stop::evaluations(10_000));
    let outcome = engine.run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!(outcome.best_fitness().score().unwrap() < 1e-8);
    // one evaluation per gradient: the problem supplied it
    assert_eq!(engine.algorithm().gradients(), Gradients::Supplied);

    // your own function and gradient, whose minimum (3, ..., 3) is outside the box
    let shifted = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        for (g, xi) in gradient.iter_mut().zip(x.iter()) {
            *g = 2.0 * (xi - 3.0);
        }
        x.iter().map(|xi| (xi - 3.0) * (xi - 3.0)).sum::<f64>()
    });
    let lbfgsb = Lbfgsb::builder(Real::uniform(5, -2.0..=2.0)?).minimize().seed(1).build()?;
    let outcome = Engine::new(lbfgsb, shifted).stop_when(Stop::evaluations(1_000)).run()?;
    assert_eq!(outcome.best_genome()[..], [2.0; 5]); // exactly on the bound
    Ok(())
}
```

Python: `gx.Lbfgsb(real, memory=10, gradients="auto", ...)`, `lbfgsb.run(f, gradient=g)` or `run(f, gradient=True)` with `f` returning `(value, gradient)` (2-D with `batch=True`); a problem of `gx.problems` gives its gradient in Rust; `gx.RunningLbfgsb` in `control`. The `genoxide` program: `type = "lbfgsb"`, and `gradient = true` in `[fitness]` for a program that writes the value then the gradient.

### Gradients: supplying them

How fitness functions give gradients to gradient-based methods ([L-BFGS-B](#l-bfgs-b-smooth-functions-with-a-gradient), [first-order methods](#first-order-methods-adam-momentum-nesterov), [MMA](#many-variables-few-constraints-mma)). A gradient is of the score as returned (`∂score/∂xᵢ`), whatever the objective: no sign change when maximizing.

- `Differentiable(|x: &Reals, gradient: &mut [f64]| value)` writes the gradient (zeroed, one value per gene) and returns the value; any algorithm takes it as a plain fitness function. `Batch(Differentiable(|xs: &[&Reals], gradients: &mut [f64]| values))`: flat, row-major, a row per genome.
- A `FitnessFunction` declares it with `fn provides(&self) -> Provided { Provided::GRADIENT }` and writes it in `fn evaluate_with(&self, x, extras: &mut Extras<'_>)` when `extras.gradient()` is `Some` (`genoxide::engine::{Extras, Provided}`); the value must be `evaluate`'s, to the bit.
- The smooth test problems supply theirs: every classic function in `problems` but `Eggholder`, `Schwefel2_21` and `Schwefel2_22`; `problem.provides().gradient`.
- `gradient::check(&function, &x)?` compares a supplied gradient with central differences: `.largest()` about 1e-10 when right, `.worst_gene()`.
- An algorithm's `gradient::Gradients` setting: `Auto` (default: supplied if provided, else forward differences, n evaluations per gradient, up to `gradient::AUTO_LIMIT` = 10⁴ genes), `Supplied` (an error at the start of a run without one), `Forward { step: None }`, `Central { step: None }` (2n per gradient, more accurate). Finite differences count towards `Stop::evaluations`.
- A NaN in a gradient follows the `NanPolicy`: invalid fitness, or `Error::NanFitness`.
- Writing an algorithm that uses gradients: `prepare(provided)` (call `gradients.resolve(provided, &real)?`), `wants()` (`Wanted::GRADIENT` for supplied ones), `tell_evaluations(&evaluations)` (`evaluations.gradient(i)`); for finite differences, ask the points of a `gradient::Stencil` with the current point in one round, and `stencil.gradient(f_x, &values, &mut gradient)?`. Constraint values the same way: `provided.inequalities` (their number), `Wanted::GRADIENT.with_inequalities().with_constraint_jacobian()`, `evaluations.inequalities(i)` and `evaluations.constraint_jacobian(i)` (row-major, m × n).

```rust
use genoxide::gradient::{self, Differentiable};
use genoxide::prelude::*;
use genoxide::problems::Rosenbrock;

fn main() -> genoxide::Result<()> {
    let rosenbrock = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        let (a, b) = (x[1] - x[0] * x[0], 1.0 - x[0]);
        gradient[0] = -400.0 * x[0] * a - 2.0 * b;
        gradient[1] = 200.0 * a;
        100.0 * a * a + b * b
    });
    // a hand-written gradient, against central differences
    assert!(gradient::check(&rosenbrock, &Reals::from(vec![-1.2, 1.0]))?.largest() < 1e-8);
    // the test problems' own
    assert!(Rosenbrock::new(10).provides().gradient);
    // a plain fitness function for any other algorithm
    let cmaes = Cmaes::builder(Real::uniform(2, -5.0..=5.0)?).minimize().seed(1).build()?;
    let outcome = Engine::new(cmaes, rosenbrock)
        .stop_when(Stop::target(1e-10).or(Stop::evaluations(10_000)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

### First-order methods: Adam, momentum, Nesterov

`FirstOrder::builder(real)` steps along the gradient by a rule, without a line search: for smooth problems with many parameters, up to millions (O(n) memory and work per step, no allocation after the first). One gradient per generation (supplied, or finite differences in the same round, up to `gradient::AUTO_LIMIT` genes with `Auto`); points projected onto the bounds; it stops as `StopReason::Converged` when the projected gradient's largest component is within `.gradient_tolerance(1e-6)` or a step moves no gene by more than `.step_tolerance(1e-12)` relative to max(1, |x|) (`first_order.converged()` says which). Adam with a constant learning rate hovers near the minimum: lower it over the run in `.control` (a schedule) for precise answers.

| `.step(first_order::Step::...)` | Rule |
|---|---|
| `adam(lr)` (default, `adam(0.001)`), `Adam { learning_rate, beta1, beta2, epsilon }` | Kingma and Ba's Algorithm 1 with bias correction; steps of about `lr` per gene, whatever the gradient's scale |
| `adamw(lr, weight_decay)` | Loshchilov and Hutter's decoupled weight decay: `x -= η (lr m̂ / (√v̂ + ε) + λ x)`, not L2 in the gradient |
| `momentum(lr, mu)`, `nesterov(lr, mu)` | Polyak's heavy ball; Nesterov's accelerated gradient in Sutskever et al.'s form (the look-ahead point is evaluated); `0 <= mu < 1` |
| `gradient(lr)` | `x -= lr g` |

Also `.gradients(gradient::Gradients::...)`, `.restarts(local::Restarts::Random { times })` (memory reset), `.initial_genome(...)`. During a run: `set_learning_rate(lr)?`, `set_multiplier(eta)?` (the schedule multiplier, which also scales AdamW's decay), `reevaluate()?` (keeps the velocity and Adam's averages); `gradient()`, `gradient_norm()`, `iterations()`, `steps()` (Adam's t), `gradients()` (the resolved source). An invalid point (fitness, or a gradient that isn't finite) is stepped back from, halfway to the last valid one. The rules are scale-sensitive: scale the genes alike. For a smooth deterministic function, `Lbfgsb` or Nesterov usually needs fewer steps; Adam suits noisy (mini-batch) gradients. See `examples/adam`. Python: `gx.FirstOrder(real, step="adam", learning_rate=..., momentum=..., weight_decay=..., gradients="auto", ...)` with `run(f, gradient=g)` or `run(f_and_g, gradient=True)`; `gx.RunningFirstOrder` (`learning_rate`, `multiplier`) in `control`. CLI: `type = "first-order"`, with the gradient protocol (`fitness.gradient = true`) or finite differences.

```rust
use genoxide::algorithm::first_order::Step;
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    // least squares: 1,000 parameters w, residuals rᵢ = wᵢ − i / 1000, loss Σ rᵢ² with its gradient
    let loss = Differentiable(|w: &Reals, gradient: &mut [f64]| {
        let mut value = 0.0;
        for (i, (g, &wi)) in gradient.iter_mut().zip(w.iter()).enumerate() {
            let r = wi - i as f64 / 1000.0;
            *g = 2.0 * r;
            value += r * r;
        }
        value
    });
    let adam = FirstOrder::builder(Real::uniform(1_000, -5.0..=5.0)?)
        .step(Step::adam(0.05))
        .initial_genome(Reals::from(vec![0.0; 1_000])) // random by default
        .minimize()
        .seed(1)
        .build()?;
    let mut engine = Engine::new(adam, loss)
        .stop_when(Stop::generations(10_000))
        // a schedule: the learning rate halved every 100 steps
        .control(|adam: &mut FirstOrder, progress| {
            adam.set_learning_rate(0.05 * 0.5f64.powi((progress.generation() / 100) as i32))
        });
    let outcome = engine.run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!(outcome.best_fitness().score().unwrap() < 1e-9);
    Ok(())
}
```

### Many variables, few constraints: MMA

`Mma` on `Real` genomes: Svanberg's method of moving asymptotes, for smooth problems with very many variables (up to millions) and few inequality constraints `gᵢ(x) <= 0` (up to a few hundred), from the gradient of the score and of every constraint (`Constrained::differentiable`, or `Differentiable` without constraints). Each iteration is one evaluation: convex, separable approximations around the current point, with two asymptotes per gene that move with the iterates, solved through their dual in the m multipliers: O(n·m) memory and work, no n × n matrix, nothing allocated after the first iteration. It stops as `StopReason::Converged` when the KKT residual or the step is within its tolerance; `mma.converged()` says which (`mma::Convergence::Kkt`, `Step`). A run that converges to a point infeasible by rounding ends with a restoration step onto the feasible side of its active constraints. Equality constraints aren't supported.

| Setting | Default |
|---|---|
| `.method(mma::Method::Gcmma)` | `Mma`: one evaluation per iteration. `Gcmma` adds inner iterations (each an evaluation) until the approximations are conservative: it converges from any start, and where MMA cycles (around a minimum inside the bounds where the objective's gradient vanishes) |
| `.constraint_cost(c)` | 1000: the cost of the artificial variable that relaxes each constraint; must exceed the multipliers at the solution, or the result is infeasible with a multiplier at `c` (`mma.multipliers()`) |
| `.asymptote_initial(0.5)`, `.asymptote_decrease(0.7)`, `.asymptote_increase(1.2)`, `.move_limit(0.5)` | Svanberg's: fractions of each gene's range, and the factors for oscillating and steady genes |
| `.kkt_tolerance(t)`, `.step_tolerance(t)` | 1e-9 (the KKT residual, in the score's units), 1e-10 (of each range) |
| `.restoration(false)` | on |
| `.parallel_sums(true)` | off: the dual's sums on rayon's threads, the same bits; from about 10⁵ genes |
| `.initial_genome(genome)` | random |

Scale the genes to ranges of about 0.1 to 100 and the score to about 1 to 100, as Svanberg advises. `mma.multipliers()`, `kkt_residual()`, `iterations()`, `inner_iterations()`, `lower_asymptotes()`, `upper_asymptotes()`. See `examples/mma` (a million variables, about 2.5 s).

```rust
use genoxide::constraint::Constrained;
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    // minimize Σ cⱼ/xⱼ subject to Σ xⱼ <= n: the minimum is at xⱼ = n √cⱼ / Σ √cₖ
    let n = 1_000;
    let c: Vec<f64> = (0..n).map(|j| 1.0 + (j % 9) as f64).collect();
    let problem = Constrained::differentiable(
        1,
        |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
            let (mut value, mut sum) = (0.0, 0.0);
            for j in 0..n {
                value += c[j] / x[j];
                gradient[j] = -c[j] / (x[j] * x[j]);
                jacobian[j] = 1.0; // row i is constraint i's gradient: here ∂g₀/∂xⱼ
                sum += x[j];
            }
            g[0] = sum - n as f64; // feasible at 0 or below
            value
        },
    );
    let mma = Mma::builder(Real::uniform(n, 0.01..=10.0)?)
        .initial_genome(Reals::from(vec![0.5; n]))
        .minimize()
        .build()?;
    let mut engine = Engine::new(mma, problem).stop_when(Stop::evaluations(500));
    let outcome = engine.run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!(outcome.best_fitness().is_feasible());
    let roots: f64 = c.iter().map(|c| c.sqrt()).sum();
    let multiplier = engine.algorithm().multipliers()[0]; // (Σ √cₖ / n)²
    assert!((multiplier - (roots / n as f64).powi(2)).abs() < 1e-6);
    Ok(())
}
```

### Continuation: stages of one problem, the state kept

`Continuation` runs a local method through stages of one problem: a smooth version first, then sharper ones (a smoothing that shrinks, a p-norm's p that grows, a penalty weight raised), each from the last stage's result. A stage ends when the method has converged (`is_finished`), or after `.generations(n)`; then the `on_stage` closure sets the next stage's parameters in state shared with the fitness function (an `Arc<AtomicU64>` of an `f64`'s bits), the method's point is evaluated again on the changed function, and it goes on with its state. The run stops as `StopReason::Converged` after the last stage, not at the first stage's convergence.

| `.keep(...)` | Kept between stages |
|---|---|
| `continuation::Keep::State` (default) | `FirstOrder`: the velocity, Adam's averages and step count; `Mma`: the asymptotes and the iterates they're updated from; `NelderMead`: the simplex; `Cmaes`: the distribution; `Lbfgsb`: nothing (its pairs describe the old function), unless built with `.keep_pairs(true)` |
| `continuation::Keep::Point` | only the point: the state starts again as at a run's start |

- `on_stage(|stage, algorithm| ...)` is called with the current stage when a run starts (the first, or the one a checkpoint resumes in) and with each next stage as it begins: set everything from the index alone. It gets the method mutably, e.g. for a learning rate per stage.
- `.on_stage_finished(|stage, algorithm| ...)`, and `continuation.stages()` after the run: each stage's `index()`, `generations()`, `evaluations()`, `best()` (by its own function) and `end()` (`StageEnd::Finished` or `Generations`).
- Checkpoints hold the stage, not the closures: after `checkpoint::load_file`, `set_on_stage(...)` (and `set_on_stage_finished`); a run without one fails with `Error::MissingSetting { setting: "on_stage" }`.
- Python: `gx.Continuation(method, stages=4, on_stage=lambda index: ..., generations=None, keep="state", on_stage_finished=None)` around `FirstOrder`, `Lbfgsb` or `Mma`; `result.stages`. See `examples/continuation`.

```rust
use genoxide::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

fn main() -> genoxide::Result<()> {
    // Σ √((xᵢ − cᵢ)² + ε²), a smoothed Σ |xᵢ − cᵢ|, sharper as ε shrinks
    const EPSILON: [f64; 3] = [1.0, 0.1, 0.01];
    let c = [0.3, -1.2, 2.0];
    let epsilon = Arc::new(AtomicU64::new(EPSILON[0].to_bits()));
    let shared = Arc::clone(&epsilon);
    let smoothed = Differentiable(move |x: &Reals, gradient: &mut [f64]| {
        let e = f64::from_bits(shared.load(Ordering::Relaxed));
        let mut value = 0.0;
        for i in 0..x.len() {
            let root = ((x[i] - c[i]) * (x[i] - c[i]) + e * e).sqrt();
            gradient[i] = (x[i] - c[i]) / root;
            value += root;
        }
        value
    });
    let adam = FirstOrder::builder(Real::uniform(3, -5.0..=5.0)?)
        .step(first_order::Step::adam(0.01))
        .gradient_tolerance(1e-9)
        .minimize()
        .seed(1)
        .build()?;
    let continuation = Continuation::builder(adam)
        .stages(EPSILON.len())
        .on_stage(move |stage, _| {
            epsilon.store(EPSILON[stage].to_bits(), Ordering::Relaxed);
            Ok(())
        })
        .build()?;
    let mut engine = Engine::new(continuation, smoothed).stop_when(Stop::generations(100_000));
    let outcome = engine.run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    for stage in engine.algorithm().stages() {
        println!("stage {}: {} steps", stage.index(), stage.generations());
    }
    for (x, c) in outcome.best_genome().iter().zip(c) {
        assert!((x - c).abs() < 1e-6);
    }
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

When the fitness function changes during a run (adaptive penalty weights, a retrained surrogate), call `ga.reevaluate()?` between a tell and the next ask (in an `Engine`, from `.control`): the next ask gives the whole population, its tell scores it again without starting a generation, and `best()` is then the best by the new function. The single-objective algorithms other than `SteadyGa` have `reevaluate()` (the `Reevaluate` trait): DE, ES and CMA-ES score their population again, PSO its positions and personal bests, local search its current solution and best, Nelder-Mead its simplex, L-BFGS-B its current point (dropping its correction pairs), and `Islands` every island.

### Without Rust: the `genoxide` program

`cargo install genoxide --features cli`; `genoxide run run.toml` (or JSON). Each worker runs `fitness.command`: a genome per stdin line in, objective values (then an optional violation) per stdout line out; with `gradient = true` and `constraints = m` in `[fitness]`, the score, its gradient, the m constraint values and their Jacobian (for `mma`). The result is JSON on stdout. Settings: [docs/cli.md](docs/cli.md).

```toml
[genome]
type = "real"          # binary, integer, real or permutation
length = 10
bounds = [-5.12, 5.12]

[fitness]
command = ["python3", "fitness.py"]   # or builtin = "rastrigin"
objectives = ["minimize"]

[algorithm]
type = "ga"            # ga, steady-ga, de, cmaes, pso, local-search, nelder-mead, lbfgsb, first-order, mma, nsga2
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
| `no method named parallel` (or `parallel_breeding`) | Enable the default `parallel` feature, or drop `.parallel(true)` |
| `Error::MissingSetting { setting: "population_size" }` | `.population_size(n)` |
| `Error::MissingSetting { setting: "stop_when" }` | `.stop_when(Stop::generations(n))`, or an abort flag |
| `Error::InvalidSetting { setting, reason }` | Read `reason`; see [Settings](#settings) |
| `Error::InvalidGenome { reason }` | Match the initial genome's length and bounds |
| ``error[E0282]: type annotations needed for `&[_]` `` in `tree.evaluate` | Name the value type: `\|op, args: &[f64]\| ...` |
| `Error::NanFitness` | Fix the NaN, or keep `NanPolicy::Invalid` |
| `Error::InvalidFitness` | A violation must be ≥ 0: use `constraint::at_most` and friends |
| `Error::TellWithoutAsk` / `Error::FitnessCount` | One `tell` per `ask`, one fitness per asked genome, in order |
| Best solution infeasible | Run longer, check the constraints, or add a feasible genome with `.initial_genomes(...)` |
| Adam hovers near the minimum and doesn't converge | Lower the learning rate over the run in `.control` (`set_learning_rate`), e.g. halved every few hundred steps |
| `FirstOrder` diverges to the bounds, or its loss grows | A smaller learning rate (gradient descent and momentum need `lr` below 2 / the largest curvature), or Adam, whose steps don't scale with the gradient |
| MMA ends infeasible, with a multiplier at the constraint cost | A `.constraint_cost(c)` above the multipliers (or the constraints scaled down), or a feasible `.initial_genome(...)` |
| MMA cycles without converging (stops by `Stop::evaluations`) | `.method(mma::Method::Gcmma)`; genes scaled to ranges of 0.1 to 100 |
| Nelder-Mead converges to a local minimum, or crawls in many genes | `.restarts(local::Restarts::Random { times })`; above about 10 genes, CMA-ES first, then Nelder-Mead from its best (`.initial_genome(...)`) |
| L-BFGS-B ends with `converged()` = `Some(Criterion::LineSearch)` far from a minimum | Check the gradient with `gradient::check(&f, &x)?`; with forward differences, a `gradient_tolerance` above their accuracy (about 1e-7 of f's scale), or `Gradients::Central` |
| L-BFGS-B takes many evaluations per iteration | Forward differences cost n + 1 per round: supply the gradient (`Differentiable`, `FitnessFunction::provides`) |
| L-BFGS-B ends in a local minimum | `.restarts(local::Restarts::Random { times })`, or a global method first (`Cmaes`, `De`) and `.initial_genome(best)` |
| Hill climbing stops at a local optimum | `.restart(patience, kicks)`, `Acceptance::Tabu { tenure }` with several neighbors, or `Acceptance::Annealing` (initial temperature ≈ typical fitness differences, `cooling` ≈ 0.999) |
| DE collapses far from the optimum | `de::Control::Dither { min_f: 0.5, max_f: 1.0, cr }`, `de::Strategy::Rand1`, or `CurrentToPBest` with an archive |
| CMA-ES converged without restarts (`cmaes.converged()` says why) | `.restarts(cmaes::Restarts::Ipop)` or `Bipop`, or a larger `.initial_step(...)` or `.population_size(...)`; to end the run there instead of sampling on, `cmaes::Restarts::Stop` |
| PSO gathers early at a local optimum | `.topology(pso::Topology::Ring { neighbors: 1 })`, or more particles |
| Trees grow large without getting better (bloat) | `DoubleTournament::new(7, 1.4)?` as the selection, hoist and shrink in `gp::Mutations`; a smaller `.max_size(...)` |
| Real-valued GA stuck in a local minimum | `PolynomialMutation` with eta 20, or a larger `GaussianMutation` sigma |
| `StopReason::Stalled`: 10 000 generations of copies only (e.g. `mutation_rate(0.0)`) | A mutation rate above 0, or add `Stop::generations(n)` or `Stop::stagnation(n)` |
| Early stagnation (too little diversity) | A larger population, smaller tournament or higher mutation rate; or `Stop::stagnation` and restart |
| Slow with a cheap fitness function | `--release`; `.parallel(true)` only for expensive fitness; `.parallel_breeding(true)` (`Ga`, `De`, `Es`) when breeding takes much of a generation |

## Guarantees to rely on

- **Reproducible:** a seed gives the same results on every platform and thread count, parallel or not. The exception is a fitness function that calls the platform's `sin`, `cos`, `exp` and the like (`f64::sin`, numpy): their last bit can differ between operating systems, and long runs drift apart. `genoxide::math::{sin, cos, tan, exp, ln, powf, powi, atan2, ...}` are the same to the bit everywhere, at native speed; `problems` and `multi::problems` use them.
- **Ties:** the earlier individual wins.
- **The best is kept:** `outcome.best()` is the best individual ever evaluated.
- **Errors, not panics,** for invalid settings, including sizes above 2^24. The only panics (`# Panics`): an index out of bounds (`Bits::set`, `Order::swap`), a `problems` or `multi::problems` constructor with too few dimensions or variables (or a radius that isn't above 0, or none from the paper for `C1Dtlz3::new` and `ConvexC2Dtlz2::new`, a DC-DTLZ `a` that isn't above 0 or `b` outside (−1, 1), or a `Difficulty` level outside [0, 1]), a `Batch` returning no score for a single genome, a gradient or constraint-values slice of the wrong length for a test problem's `evaluate_with`, and an input, output or observation slice of the wrong length for an `nn` network or a `control` task.
