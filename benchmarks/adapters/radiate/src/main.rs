//! Benchmark adapter for radiate (https://crates.io/crates/radiate).
//!
//! Usage: ga_bench_radiate <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//! Prints one JSON line per solver per seed, see ../../README.md for the fields.
//!
//! radiate has one search method, its `GeneticEngine`: a GA whose generation keeps
//! `population_size * (1 - offspring_fraction)` survivors (survivor selector) and breeds
//! `population_size * offspring_fraction` offspring (offspring selector, then crossover and
//! mutation). Only the individuals whose genome changed are evaluated again. Multi-objective
//! problems use the same engine with the NSGA-II or NSGA-III selectors.
//!
//! Single-threaded: radiate is built without its `rayon` feature and no executor is set, so the
//! engine uses its default `Executor::Serial` for the fitness, the species and the events.
//! Seeded: every run is inside `random_provider::scoped_seed(seed, ..)`, which reseeds the
//! thread-local generator radiate draws all its random numbers from (`random_provider::seed`
//! only takes effect before the first draw of a thread, so it can't reseed the second run).
//! Fitness: through `raw_fitness_fn`, radiate's documented way to evaluate the genotype without
//! decoding it (docs/source/fitness.md, "Raw Fitness"); the values are computed in f64 like in
//! the other adapters, radiate keeps them as an f32 `Score`.

use radiate::prelude::*;
use std::f64::consts::{E, PI};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Fitness functions, identical to the ones in the other adapters
// ---------------------------------------------------------------------------------------------

fn onemax(genes: &[BitGene]) -> f64 {
    genes.iter().filter(|gene| *gene.allele()).count() as f64
}

/// Number of diagonal conflicts, O(n) (as NQueens.fitness in adapters/pymoo/bench.py)
fn nqueens(genes: &[PermutationGene<usize>]) -> f64 {
    let size = genes.len();
    let mut left_diagonal = vec![0usize; 2 * size - 1];
    let mut right_diagonal = vec![0usize; 2 * size - 1];
    for (i, gene) in genes.iter().enumerate() {
        let column = *gene.allele();
        left_diagonal[i + column] += 1;
        right_diagonal[size - 1 - i + column] += 1;
    }
    left_diagonal
        .iter()
        .chain(&right_diagonal)
        .map(|&count| count.saturating_sub(1))
        .sum::<usize>() as f64
}

// Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift
// towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
fn shift(i: usize) -> f64 {
    2.0 * ((37 * i + 11) % 101) as f64 / 101.0 - 1.0
}

fn rastrigin(x: &[f64]) -> f64 {
    10.0 * x.len() as f64
        + x.iter()
            .enumerate()
            .map(|(i, v)| {
                let v = v - shift(i);
                v * v - 10.0 * (2.0 * PI * v).cos()
            })
            .sum::<f64>()
}

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let shifted = || x.iter().enumerate().map(|(i, v)| v - shift(i));
    let squares = shifted().map(|v| v * v).sum::<f64>() / n;
    let cosines = shifted().map(|v| (2.0 * PI * v).cos()).sum::<f64>() / n;
    -20.0 * (-0.2 * squares.sqrt()).exp() - cosines.exp() + 20.0 + E
}

fn zdt_g(x: &[f64]) -> f64 {
    1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64
}

fn zdt1(x: &[f64]) -> Vec<f64> {
    let g = zdt_g(x);
    vec![x[0], g * (1.0 - (x[0] / g).sqrt())]
}

fn zdt2(x: &[f64]) -> Vec<f64> {
    let g = zdt_g(x);
    vec![x[0], g * (1.0 - (x[0] / g).powi(2))]
}

fn zdt3(x: &[f64]) -> Vec<f64> {
    let g = zdt_g(x);
    let r = x[0] / g;
    vec![x[0], g * (1.0 - r.sqrt() - r * (10.0 * PI * x[0]).sin())]
}

/// 3 objectives, 12 variables (k = 10)
fn dtlz2(x: &[f64]) -> Vec<f64> {
    let g = x[2..].iter().map(|v| (v - 0.5).powi(2)).sum::<f64>();
    let (a, b) = (x[0] * PI / 2.0, x[1] * PI / 2.0);
    vec![
        (1.0 + g) * a.cos() * b.cos(),
        (1.0 + g) * a.cos() * b.sin(),
        (1.0 + g) * a.sin(),
    ]
}

/// 3 objectives, 7 variables (k = 5)
fn dtlz1(x: &[f64]) -> Vec<f64> {
    let g = 100.0
        * (5.0
            + x[2..]
                .iter()
                .map(|v| (v - 0.5).powi(2) - (20.0 * PI * (v - 0.5)).cos())
                .sum::<f64>());
    vec![
        0.5 * x[0] * x[1] * (1.0 + g),
        0.5 * x[0] * (1.0 - x[1]) * (1.0 + g),
        0.5 * (1.0 - x[0]) * (1.0 + g),
    ]
}

fn alleles(genotype: &Genotype<FloatChromosome<f64>>) -> Vec<f64> {
    genotype[0].as_slice().iter().map(|gene| *gene.allele()).collect()
}

// ---------------------------------------------------------------------------------------------
// The budget: counts the evaluations and keeps the best value, in the fitness function
// ---------------------------------------------------------------------------------------------

struct Budget {
    evaluations: AtomicUsize,
    // f64 bits of the best value so far
    best: AtomicU64,
    minimize: bool,
    target: Option<f64>,
    max_evaluations: usize,
    deadline: Instant,
}

impl Budget {
    fn new(args: &Args, minimize: bool, target: Option<f64>) -> Arc<Self> {
        let worst = if minimize { f64::INFINITY } else { f64::NEG_INFINITY };
        Arc::new(Self {
            evaluations: AtomicUsize::new(0),
            best: AtomicU64::new(worst.to_bits()),
            minimize,
            target,
            max_evaluations: args.max_evaluations,
            deadline: Instant::now() + Duration::from_secs_f64(args.max_seconds),
        })
    }

    fn record(&self, value: f64) -> f64 {
        self.count();
        let best = self.best();
        if (self.minimize && value < best) || (!self.minimize && value > best) {
            self.best.store(value.to_bits(), Ordering::Relaxed);
        }
        value
    }

    fn count(&self) {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
    }

    fn evaluations(&self) -> usize {
        self.evaluations.load(Ordering::Relaxed)
    }

    fn best(&self) -> f64 {
        f64::from_bits(self.best.load(Ordering::Relaxed))
    }

    fn reached(&self) -> bool {
        match self.target {
            Some(target) if self.minimize => self.best() <= target,
            Some(target) => self.best() >= target,
            None => false,
        }
    }

    // checked by the engine after every generation
    fn done(&self) -> bool {
        self.reached()
            || self.evaluations() >= self.max_evaluations
            || Instant::now() >= self.deadline
    }
}

// ---------------------------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------------------------

struct Args {
    problem: String,
    size: usize,
    mode: String,
    seed_from: u64,
    seed_to: u64,
    max_evaluations: usize,
    max_seconds: f64,
}

/// Builds the engine (the random initial population) and runs it until the budget is done,
/// checked after every generation. Returns the last generation and the seconds from before the
/// build to the end of the run.
fn run_engine<C, T>(
    budget: &Arc<Budget>,
    build: impl FnOnce() -> GeneticEngine<C, T>,
) -> (Generation<C, T>, f64)
where
    C: Chromosome + Clone + PartialEq + 'static,
    T: Clone + Send + Sync + 'static,
{
    let start = Instant::now();
    let engine = build();
    let stop = Arc::clone(budget);
    let generation = engine
        .iter()
        .until(move |_: GenerationView<C, T>| stop.done())
        .last()
        .expect("radiate engine failed");
    (generation, start.elapsed().as_secs_f64())
}

fn print_single(args: &Args, seed: u64, budget: &Budget, generations: usize, time_s: f64) {
    let target = budget.target.expect("single-objective runs have a target");
    println!(
        "{{\"library\":\"radiate\",\"solver\":\"ga\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{generations},\"evaluations\":{},\"best\":{:?},\"target\":{target:?},\"success\":{}}}",
        args.problem,
        args.size,
        args.mode,
        budget.evaluations(),
        budget.best(),
        budget.reached(),
    );
}

fn run_onemax(args: &Args, seed: u64) {
    let size = args.size;
    let budget = Budget::new(args, false, Some(size as f64));
    let fitness = Arc::clone(&budget);
    let (generation, time_s) = random_provider::scoped_seed(seed, || {
        run_engine(&budget, || {
            let builder = GeneticEngine::builder()
                .codec(BitCodec::vector(size))
                .raw_fitness_fn(move |genotype: &Genotype<BitChromosome>| {
                    fitness.record(onemax(genotype[0].as_slice()))
                });
            match args.mode.as_str() {
                // as DEAP's eaSimple: population 300, tournament of 3 (with replacement, like
                // selTournament), two-point crossover 0.5, bit flip 1/size on 20% of the children,
                // no elitism. Differences:
                // - offspring_fraction 1.0: every generation is 300 selected and altered copies,
                //   no survivors, as eaSimple. max_age off: radiate by default replaces
                //   individuals older than 20 generations with random ones, eaSimple doesn't.
                // - crossover: radiate visits every child and with probability 0.5 crosses it
                //   with a random other child (both change), DEAP crosses the disjoint pairs
                //   (0,1), (2,3), ... with probability 0.5: the same expected 150 crossovers per
                //   generation, but a child can be crossed more than once. radiate's cut points
                //   are drawn from 0..size (a cut at 0 is a no-op), DEAP's from 1..size.
                // - mutation: radiate has no per-individual mutation probability, so BitFlip
                //   flips each bit with probability 0.2 / size: the same expected 0.2 flipped
                //   bits per child, spread over more children (18% instead of 12.6% mutated).
                // - evaluations: both evaluate only the changed children; DEAP also re-evaluates
                //   a child it chose to mutate when no bit happened to flip.
                "matched" => builder
                    .population_size(300)
                    .offspring_fraction(1.0)
                    .max_age(usize::MAX)
                    .offspring_selector(TournamentSelector::new(3))
                    .alter(alters!(
                        MultiPointCrossover::new(0.5, 2),
                        BitFlipMutator::new(0.2 / size as f32)
                    ))
                    .build(),
                // the README's "Hello, Radiate!" example (examples/rust/hello-world), a count of
                // matching genes like OneMax: Boltzmann offspring selection with temperature 4,
                // and the engine defaults (docs/source/engine/index.md, "Engine Defaults"):
                // population 100, offspring fraction 0.8, tournament of 3 for the survivors,
                // UniformCrossover(0.5) and UniformMutator(0.1), max age 20
                _ => builder
                    .offspring_selector(BoltzmannSelector::new(4.0))
                    .build(),
            }
        })
    });
    print_single(args, seed, &budget, generation.index(), time_s);
}

fn run_nqueens(args: &Args, seed: u64) {
    let size = args.size;
    let budget = Budget::new(args, true, Some(0.0));
    let fitness = Arc::clone(&budget);
    // radiate's permutation example (examples/rust/TSP): PermutationCodec, population 250,
    // PMXCrossover(0.4) and SwapMutator(0.05), minimizing, the other engine defaults (roulette
    // offspring selection, tournament of 3 for the survivors, offspring fraction 0.8, max age
    // 20). radiate's own N-Queens example (examples/rust/nqueens) uses an integer genome with
    // row conflicts in the fitness, not a permutation, so it doesn't apply here.
    let (generation, time_s) = random_provider::scoped_seed(seed, || {
        run_engine(&budget, || {
            GeneticEngine::builder()
                .codec(PermutationCodec::new((0..size).collect()))
                .raw_fitness_fn(move |genotype: &Genotype<PermutationChromosome<usize>>| {
                    fitness.record(nqueens(genotype[0].as_slice()))
                })
                .minimizing()
                .population_size(250)
                .alter(alters!(PMXCrossover::new(0.4), SwapMutator::new(0.05)))
                .build()
        })
    });
    print_single(args, seed, &budget, generation.index(), time_s);
}

/// Rastrigin, Rosenbrock and Ackley: `size` reals in `range`
fn run_real(args: &Args, seed: u64, function: fn(&[f64]) -> f64, range: std::ops::Range<f64>) {
    const TARGET: f64 = 0.01;
    let size = args.size;
    let budget = Budget::new(args, true, Some(TARGET));
    let fitness = Arc::clone(&budget);
    let (generation, time_s) = random_provider::scoped_seed(seed, || {
        run_engine(&budget, || {
            let builder = GeneticEngine::builder()
                .codec(FloatCodec::vector(size, range.clone()))
                .raw_fitness_fn(move |genotype: &Genotype<FloatChromosome<f64>>| {
                    fitness.record(function(&alleles(genotype)))
                })
                .minimizing();
            match args.problem.as_str() {
                // radiate's Rosenbrock example (examples/rust/rosenbrock): Boltzmann offspring
                // selection with temperature 4, MeanCrossover(0.75), ArithmeticMutator(0.1), the
                // other engine defaults (population 100, fraction 0.8, tournament of 3, max age 20)
                "rosenbrock" => builder
                    .offspring_selector(BoltzmannSelector::new(4.0))
                    .alter(alters!(
                        MeanCrossover::new(0.75),
                        ArithmeticMutator::new(0.1)
                    ))
                    .build(),
                // radiate's Rastrigin example (examples/rust/rastrigin, also on the user guide's
                // examples page): population 500, UniformCrossover(0.5), ArithmeticMutator(0.01),
                // the other engine defaults (roulette offspring selection, tournament of 3,
                // fraction 0.8, max age 20). radiate has no Ackley example; its Rastrigin example
                // is the one for a multimodal real function, so Ackley uses it too.
                _ => builder
                    .population_size(500)
                    .alter(alters!(
                        UniformCrossover::new(0.5),
                        ArithmeticMutator::new(0.01)
                    ))
                    .build(),
            }
        })
    });
    print_single(args, seed, &budget, generation.index(), time_s);
}

// ---------------------------------------------------------------------------------------------
// Multi-objective
// ---------------------------------------------------------------------------------------------

/// The distinct non-dominated points (to minimize) among `points`
fn non_dominated(points: &[Vec<f64>]) -> Vec<&Vec<f64>> {
    let dominates = |a: &Vec<f64>, b: &Vec<f64>| {
        a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
    };
    let mut front: Vec<&Vec<f64>> = Vec::new();
    for point in points {
        if !points.iter().any(|other| dominates(other, point)) && !front.contains(&point) {
            front.push(point);
        }
    }
    front
}

fn print_front(
    args: &Args,
    seed: u64,
    solver: &str,
    budget: &Budget,
    generations: usize,
    time_s: f64,
    front: &[Vec<f64>],
) {
    let front: Vec<String> = front
        .iter()
        .map(|values| {
            let values: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
            format!("[{}]", values.join(","))
        })
        .collect();
    println!(
        "{{\"library\":\"radiate\",\"solver\":\"{solver}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{generations},\"evaluations\":{},\"front\":[{}]}}",
        args.problem,
        args.size,
        args.mode,
        budget.evaluations(),
        front.join(","),
    );
}

type Objectives = fn(&[f64]) -> Vec<f64>;

/// The matched settings: population 100 (92 for DTLZ), SBX with η 15 at 0.9 and polynomial
/// mutation with η 20 at 1 / n; NSGA-III with 91 Das-Dennis directions (12 divisions) and SBX
/// with η 30 at 1.
///
/// NSGA-II and NSGA-III select the next population from parents plus offspring. In radiate's
/// engine that is `population_size(2 * mu)` with `offspring_fraction(0.5)`: every generation keeps
/// mu survivors of the 2 mu individuals (parents and their offspring) by the NSGA-II / NSGA-III
/// selector, and breeds mu offspring. The final front is the non-dominated set of the mu the
/// survivor selector keeps of the last 2 mu, i.e. NSGA-II's (NSGA-III's) population after the
/// last generation. radiate's own Pareto archive (`front_size`, by default 800 to 900 of all the
/// non-dominated individuals seen) is not reported: the other libraries report their population.
///
/// Differences from the textbook algorithms, all radiate's own:
/// - the mating pool is drawn from all 2 mu individuals (parents and offspring of the previous
///   generation), not from the mu survivors
/// - NSGA-II: the crowding distance is computed over the whole population, not per front
/// - `max_age` is off: radiate by default replaces individuals older than 20 generations with
///   random ones
/// - radiate's SimulatedBinaryCrossover changes one of the two parents (the other is re-evaluated
///   unchanged), crosses each variable with probability 0.5, and centers the child on
///   (p1 - p2) / 2 where SBX has (p1 + p2) / 2, clamped to the bounds
/// - radiate's PolynomialMutator puts the mutated variable at lower + δ (upper - lower) (or
///   upper - ...) where polynomial mutation has x + δ (upper - lower)
fn run_front(args: &Args, seed: u64) {
    let (function, variables, objectives, mu): (Objectives, usize, usize, usize) =
        match args.problem.as_str() {
            "zdt1" => (zdt1, args.size, 2, 100),
            "zdt2" => (zdt2, args.size, 2, 100),
            "zdt3" => (zdt3, args.size, 2, 100),
            // size: the number of objectives, 3 (the variables are fixed at 12 and 7)
            "dtlz2" => (dtlz2, 12, 3, 92),
            "dtlz1" => (dtlz1, 7, 3, 92),
            _ => unreachable!(),
        };
    if objectives == 3 && args.size != 3 {
        eprintln!("radiate adapter: DTLZ is implemented for 3 objectives");
        return;
    }
    let solvers: &[&str] = if objectives == 2 {
        &["nsga2"]
    } else {
        &["nsga2", "nsga3"]
    };
    let rate = 1.0 / variables as f32;

    for &solver in solvers {
        let budget = Budget::new(args, true, None);
        let fitness = Arc::clone(&budget);
        let (front, generations, time_s) = random_provider::scoped_seed(seed, || {
            let start = Instant::now();
            let (generation, _) = run_engine(&budget, || {
                let builder = GeneticEngine::builder()
                    .codec(FloatCodec::vector(variables, 0.0_f64..1.0))
                    .raw_fitness_fn(move |genotype: &Genotype<FloatChromosome<f64>>| {
                        fitness.count();
                        function(&alleles(genotype))
                            .into_iter()
                            .map(|v| v as f32)
                            .collect::<Vec<f32>>()
                    })
                    .multi_objective(vec![Optimize::Minimize; objectives])
                    .population_size(2 * mu)
                    .offspring_fraction(0.5)
                    .max_age(usize::MAX);
                match solver {
                    // NSGA-II: binary tournament on rank and crowding distance for the mating
                    // pool, rank and crowding distance for the survivors
                    "nsga2" => builder
                        .offspring_selector(TournamentNSGA2Selector::new())
                        .survivor_selector(NSGA2Selector::new())
                        .alter(alters!(
                            SimulatedBinaryCrossover::new(0.9, 15.0),
                            PolynomialMutator::new(rate, 20.0)
                        ))
                        .build(),
                    // NSGA-III: random mating, reference-direction niching for the survivors
                    _ => builder
                        .offspring_selector(RandomSelector::new())
                        .survivor_selector(NSGA3Selector::new(12))
                        .alter(alters!(
                            SimulatedBinaryCrossover::new(1.0, 30.0),
                            PolynomialMutator::new(rate, 20.0)
                        ))
                        .build(),
                }
            });
            // the survivors of the last generation, as NSGA-II (NSGA-III) selects them
            let population: &[Phenotype<FloatChromosome<f64>>] = generation.population().as_ref();
            let survivors = if solver == "nsga2" {
                NSGA2Selector::new().select(population, generation.objective(), mu)
            } else {
                NSGA3Selector::new(12).select(population, generation.objective(), mu)
            };
            // their objectives in f64 (radiate keeps f32 scores); not counted as evaluations
            let points: Vec<Vec<f64>> = survivors
                .iter()
                .map(|&i| function(&alleles(population[i].genotype())))
                .collect();
            let front: Vec<Vec<f64>> = non_dominated(&points).into_iter().cloned().collect();
            (front, generation.index(), start.elapsed().as_secs_f64())
        });
        print_front(args, seed, solver, &budget, generations, time_s, &front);
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() != 7 {
        eprintln!(
            "usage: ga_bench_radiate <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>"
        );
        std::process::exit(2);
    }
    let args = Args {
        problem: raw[0].clone(),
        size: raw[1].parse().expect("size"),
        mode: raw[2].clone(),
        seed_from: raw[3].parse().expect("seed_from"),
        seed_to: raw[4].parse().expect("seed_to"),
        max_evaluations: raw[5].parse().expect("max_evaluations"),
        max_seconds: raw[6].parse().expect("max_seconds"),
    };
    for seed in args.seed_from..=args.seed_to {
        match args.problem.as_str() {
            "onemax" => run_onemax(&args, seed),
            "nqueens" => run_nqueens(&args, seed),
            "rastrigin" => run_real(&args, seed, rastrigin, -5.12..5.12),
            "rosenbrock" => run_real(&args, seed, rosenbrock, -5.0..10.0),
            "ackley" => run_real(&args, seed, ackley, -32.768..32.768),
            "zdt1" | "zdt2" | "zdt3" | "dtlz1" | "dtlz2" => run_front(&args, seed),
            // unsupported: print nothing
            other => {
                eprintln!("radiate adapter: unsupported problem {other}");
                return;
            }
        }
    }
}

#[cfg(test)]
mod shift_tests {
    use super::*;

    #[test]
    fn shifted_functions() {
        // 0 at the shift, and the values of a Python reference at a fixed point
        let s: Vec<f64> = (0..10).map(shift).collect();
        let x: Vec<f64> = (0..10).map(|i| 0.5 * (i % 7) as f64 - 1.5).collect();
        assert!(rastrigin(&s).abs() < 1e-12);
        assert!(ackley(&s).abs() < 1e-12);
        assert!((rastrigin(&x) - 87.78147018265213).abs() < 1e-9);
        assert!((ackley(&x) - 5.149902035382837).abs() < 1e-9);
    }
}
