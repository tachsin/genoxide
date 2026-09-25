//! Benchmark adapter for radiate (https://crates.io/crates/radiate), version 1.3.1.
//!
//! Usage:
//!   ga_bench_radiate <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//!   ga_bench_radiate values <problem> <size>
//! The first prints one JSON line per solver per seed (see ../../README.md, "Adding a library"),
//! the second reads one JSON solution per line and prints its value (or objectives) per line.
//!
//! What runs where, and where radiate recommends it: docs/benchmarks/libraries/radiate.md.
//!
//! radiate has one search method, its `GeneticEngine`: a GA whose generation keeps
//! `population_size * (1 - offspring_fraction)` survivors (survivor selector) and breeds
//! `population_size * offspring_fraction` offspring (offspring selector, then crossover and
//! mutation). Only the individuals whose genome changed are evaluated again. By default it
//! replaces every individual older than `max_age` = 20 generations with a random one
//! (radiate-engines-1.3.1/src/builder/mod.rs, `max_age: 20`; steps/filter.rs). Multi-objective
//! problems use the same engine with the NSGA-II or NSGA-III selectors.
//!
//! Every run ends only at the target, the budget or the time cap (rule 2.1): the adapter's
//! `until` limit is the engine's only stop criterion, checked after every generation, and radiate
//! has no stop criterion of its own (an engine without a limit runs forever,
//! docs/source/engine/index.md, "Common Pitfalls"), neither a budget nor a convergence test, so
//! no run needs a restart (rule 2.2).
//! Bounds (rule 2.4): `FloatCodec::vector` draws the genes in the problem's range and sets it as
//! their bounds, and radiate's float alterers write through `FloatGene::set_allele` or
//! `safe_clamp`, which clamp to them. The fitness wrapper counts, without clipping, every
//! evaluated solution outside the bounds and the run prints it as `outside`.
//!
//! Single-threaded: radiate is built without its `rayon` feature and no executor is set, so the
//! engine uses its default `Executor::Serial` for the fitness, the species and the events.
//! Seeded: every run is inside `random_provider::scoped_seed(seed, ..)`, which reseeds the
//! thread-local generator radiate draws all its random numbers from (`random_provider::seed`
//! only reseeds the global generator new threads start from, so it can't reseed a second run in
//! the same thread; radiate-core-1.3.1/src/domain/random_provider.rs).
//! Fitness: through `raw_fitness_fn`, radiate's documented way to evaluate the genotype without
//! decoding it (docs/source/fitness.md, "Raw Fitness"). The values are computed in f64 like in the
//! other adapters; radiate keeps them as an f32 `Score`, so the fitness function also keeps the
//! best value in f64 and its solution, and the run reports those.

use radiate::prelude::*;
use std::f64::consts::{E, PI};
use std::io::BufRead;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Fitness functions, the same as benchmarks/problems.py
// ---------------------------------------------------------------------------------------------

fn onemax(bits: impl Iterator<Item = bool>) -> f64 {
    bits.filter(|&bit| bit).count() as f64
}

/// Diagonal conflicts of queens at (i, columns[i]): for each diagonal, its queens minus one
fn nqueens(columns: impl ExactSizeIterator<Item = usize>) -> f64 {
    let size = columns.len();
    let mut left = vec![0usize; 2 * size - 1];
    let mut right = vec![0usize; 2 * size - 1];
    for (i, column) in columns.enumerate() {
        left[i + column] += 1;
        right[size - 1 - i + column] += 1;
    }
    left.iter()
        .chain(&right)
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

fn zdt1(x: &[f64], _: usize) -> Vec<f64> {
    let g = zdt_g(x);
    vec![x[0], g * (1.0 - (x[0] / g).sqrt())]
}

fn zdt2(x: &[f64], _: usize) -> Vec<f64> {
    let g = zdt_g(x);
    vec![x[0], g * (1.0 - (x[0] / g).powi(2))]
}

fn zdt3(x: &[f64], _: usize) -> Vec<f64> {
    let g = zdt_g(x);
    let r = x[0] / g;
    vec![x[0], g * (1.0 - r.sqrt() - r * (10.0 * PI * x[0]).sin())]
}

/// `objectives` objectives of `objectives + 9` variables (k = 10)
fn dtlz2(x: &[f64], objectives: usize) -> Vec<f64> {
    let g = x[objectives - 1..].iter().map(|v| (v - 0.5).powi(2)).sum::<f64>();
    (0..objectives)
        .map(|m| {
            let mut f = 1.0 + g;
            for v in &x[..objectives - 1 - m] {
                f *= (v * PI / 2.0).cos();
            }
            if m > 0 {
                f *= (x[objectives - 1 - m] * PI / 2.0).sin();
            }
            f
        })
        .collect()
}

/// `objectives` objectives of `objectives + 4` variables (k = 5)
fn dtlz1(x: &[f64], objectives: usize) -> Vec<f64> {
    let tail = &x[objectives - 1..];
    let g = 100.0
        * (tail.len() as f64
            + tail
                .iter()
                .map(|v| (v - 0.5).powi(2) - (20.0 * PI * (v - 0.5)).cos())
                .sum::<f64>());
    (0..objectives)
        .map(|m| {
            let mut f = 0.5 * (1.0 + g);
            for v in &x[..objectives - 1 - m] {
                f *= v;
            }
            if m > 0 {
                f *= 1.0 - x[objectives - 1 - m];
            }
            f
        })
        .collect()
}

type Objectives = fn(&[f64], usize) -> Vec<f64>;
type Function = fn(&[f64]) -> f64;

/// The objectives, the number of variables and the number of objectives of a front problem
fn front_problem(problem: &str, size: usize) -> Option<(Objectives, usize, usize)> {
    Some(match problem {
        "zdt1" => (zdt1 as Objectives, size, 2),
        "zdt2" => (zdt2, size, 2),
        "zdt3" => (zdt3, size, 2),
        "dtlz2" => (dtlz2, size + 9, size),
        "dtlz1" => (dtlz1, size + 4, size),
        _ => return None,
    })
}

fn real_problem(problem: &str) -> Option<(Function, std::ops::Range<f64>)> {
    Some(match problem {
        "rastrigin" => (rastrigin as Function, -5.12..5.12),
        "rosenbrock" => (rosenbrock, -5.0..10.0),
        "ackley" => (ackley, -32.768..32.768),
        _ => return None,
    })
}

fn alleles(genotype: &Genotype<FloatChromosome<f64>>) -> Vec<f64> {
    genotype[0].as_slice().iter().map(|gene| *gene.allele()).collect()
}

// ---------------------------------------------------------------------------------------------
// The budget: counts the evaluations and keeps the best value and its solution, in the fitness
// function
// ---------------------------------------------------------------------------------------------

struct Budget<S> {
    evaluations: AtomicUsize,
    // f64 bits of the best value so far, and its solution
    best: AtomicU64,
    solution: Mutex<Vec<S>>,
    // evaluated solutions outside the problem's bounds, as radiate proposed them (rule 2.4)
    outside: AtomicUsize,
    minimize: bool,
    target: f64,
    max_evaluations: usize,
    deadline: Instant,
}

impl<S> Budget<S> {
    fn new(args: &Args, start: Instant, minimize: bool, target: f64) -> Arc<Self> {
        let worst = if minimize { f64::INFINITY } else { f64::NEG_INFINITY };
        Arc::new(Self {
            evaluations: AtomicUsize::new(0),
            best: AtomicU64::new(worst.to_bits()),
            solution: Mutex::new(Vec::new()),
            outside: AtomicUsize::new(0),
            minimize,
            target,
            max_evaluations: args.max_evaluations,
            deadline: start + Duration::from_secs_f64(args.max_seconds),
        })
    }

    /// Counts one evaluation of `value`; keeps it and its solution if it's the best so far
    fn record(&self, value: f64, solution: impl FnOnce() -> Vec<S>) -> f64 {
        self.count();
        let best = self.best();
        if (self.minimize && value < best) || (!self.minimize && value > best) {
            self.best.store(value.to_bits(), Ordering::Relaxed);
            *self.solution.lock().unwrap() = solution();
        }
        value
    }

    /// Counts `x` if it's outside [lower, upper] in any variable (not clipped: as evaluated)
    fn check_bounds(&self, x: &[f64], lower: f64, upper: f64) {
        if x.iter().any(|v| !(lower..=upper).contains(v)) {
            self.outside.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn outside(&self) -> usize {
        self.outside.load(Ordering::Relaxed)
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

    // problems.reached
    fn reached(&self) -> bool {
        if self.minimize {
            self.best() <= self.target
        } else {
            self.best() >= self.target
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
/// checked after every generation. Returns the last generation.
fn run_engine<C, T, S>(budget: &Arc<Budget<S>>, engine: GeneticEngine<C, T>) -> Generation<C, T>
where
    C: Chromosome + Clone + PartialEq + 'static,
    T: Clone + Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    let stop = Arc::clone(budget);
    engine
        .iter()
        .until(move |_: GenerationView<C, T>| stop.done())
        .last()
        .expect("radiate engine failed")
}

fn print_single<S>(
    args: &Args,
    seed: u64,
    solver: &str,
    budget: &Budget<S>,
    generations: usize,
    time_s: f64,
    format: impl Fn(&S) -> String,
) {
    let solution: Vec<String> = budget.solution.lock().unwrap().iter().map(format).collect();
    // rule 2.4: the continuous problems report the evaluated solutions outside the bounds
    let outside = match real_problem(&args.problem) {
        Some(_) => format!(",\"outside\":{}", budget.outside()),
        None => String::new(),
    };
    println!(
        "{{\"library\":\"radiate\",\"solver\":\"{solver}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{generations},\"evaluations\":{},\"best\":{:?},\"target\":{:?},\"success\":{},\"solution\":[{}]{outside}}}",
        args.problem,
        args.size,
        args.mode,
        budget.evaluations(),
        budget.best(),
        budget.target,
        budget.reached(),
        solution.join(","),
    );
}

/// Runs `build` (which gets the budget for its fitness function) with the seed, and prints the run
fn run_single<C, T, S>(
    args: &Args,
    seed: u64,
    solver: &str,
    minimize: bool,
    target: f64,
    build: impl FnOnce(Arc<Budget<S>>) -> GeneticEngine<C, T>,
    format: impl Fn(&S) -> String,
) where
    C: Chromosome + Clone + PartialEq + 'static,
    T: Clone + Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    let (budget, generations, time_s) = random_provider::scoped_seed(seed, || {
        // the clock starts before the engine creates the initial population
        let start = Instant::now();
        let budget = Budget::new(args, start, minimize, target);
        let generation = run_engine(&budget, build(Arc::clone(&budget)));
        (budget, generation.index(), start.elapsed().as_secs_f64())
    });
    print_single(args, seed, solver, &budget, generations, time_s, format);
}

// The idiomatic methods; the page docs/benchmarks/libraries/radiate.md links each source.
// Each problem type runs radiate's own example for it as it is (`ga`), and the recipe of radiate's
// guide for its genome type, one solver per crossover the recipe names. The guide's "Best
// Practices" (docs/source/alters/index.md, lines 30-38, in the v1.3.1 repository):
//   "Start with conservative rates (0.01 for mutation, 0.5-0.8 for crossover)"
//   "For continuous problems: Use Gaussian or Arithmetic mutators with Blend/Intermediate crossover"
//   "For permutation problems: Use Swap/Scramble mutators with PMX or Shuffle crossover"
//   "For binary problems: Use Uniform mutator with Multi-point or Uniform crossover"
// Where the recipe leaves a choice, rule 6.2 decides: a preference the docs state, else radiate's
// example for the problem type, else the default. So a recipe solver is the problem type's example
// (its population and selectors) with the recipe's alterers:
// - mutation rate 0.01, the stated starting rate;
// - crossover rate: the stated range is 0.5-0.8; within it, the example's own crossover rate
//   (hello-world and Rastrigin 0.5, Rosenbrock 0.75); the TSP example's 0.4 is outside the range, so
//   the permutation recipe takes the engine default crossover's 0.5 (engine/index.md, "Engine
//   Defaults": UniformCrossover(0.5));
// - the mutator, where the recipe names two: the one of the example (Swap in the TSP example,
//   Arithmetic in the Rastrigin and Rosenbrock examples);
// - MultiPointCrossover with 2 points (every radiate example of it), Blend and Intermediate with
//   alpha 0.5 (the guide's snippets, docs/source/src/rust/alters/crossovers.rs, lines 6 and 11).
const IDIOMATIC_BINARY: [&str; 3] = ["ga", "ga_uniform", "ga_multipoint"];
const IDIOMATIC_PERMUTATION: [&str; 2] = ["ga", "ga_pmx"];
const IDIOMATIC_REAL: [&str; 3] = ["ga", "ga_blend", "ga_intermediate"];

// ---------------------------------------------------------------------------------------------
// OneMax
// ---------------------------------------------------------------------------------------------

fn run_onemax(args: &Args, seed: u64) {
    let size = args.size;
    let solvers: &[&str] = if args.mode == "matched" { &["ga"] } else { &IDIOMATIC_BINARY };
    for &solver in solvers {
        let build = |budget: Arc<Budget<bool>>| {
            let builder = GeneticEngine::builder()
                .codec(BitCodec::vector(size))
                .raw_fitness_fn(move |genotype: &Genotype<BitChromosome>| {
                    let genes = genotype[0].as_slice();
                    budget.record(onemax(genes.iter().map(|gene| *gene.allele())), || {
                        genes.iter().map(|gene| *gene.allele()).collect()
                    })
                });
            if args.mode == "matched" {
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
                return builder
                    .population_size(300)
                    .offspring_fraction(1.0)
                    .max_age(usize::MAX)
                    .offspring_selector(TournamentSelector::new(3))
                    .alter(alters!(
                        MultiPointCrossover::new(0.5, 2),
                        BitFlipMutator::new(0.2 / size as f32)
                    ))
                    .build();
            }
            // the README's "Hello, Radiate!" example (examples/rust/hello-world, lines 8-19), a
            // count of matching genes like OneMax: Boltzmann offspring selection with temperature
            // 4, the other engine defaults (population 100, tournament of 3 for the survivors)
            let builder = builder.offspring_selector(BoltzmannSelector::new(4.0));
            match solver {
                // the example as it is, with the default alterers UniformCrossover(0.5) and
                // UniformMutator(0.1), which redraws each bit with probability 0.1 (flips it with
                // 0.05)
                "ga" => builder.build(),
                // the binary recipe with Uniform crossover, at the example's 0.5
                "ga_uniform" => builder
                    .alter(alters!(UniformCrossover::new(0.5), UniformMutator::new(0.01)))
                    .build(),
                // the binary recipe with Multi-point crossover, 2 points, at the example's 0.5
                _ => builder
                    .alter(alters!(MultiPointCrossover::new(0.5, 2), UniformMutator::new(0.01)))
                    .build(),
            }
        };
        let format = |bit: &bool| if *bit { "1" } else { "0" }.to_string();
        run_single(args, seed, solver, false, size as f64, build, format);
    }
}

// ---------------------------------------------------------------------------------------------
// N-Queens
// ---------------------------------------------------------------------------------------------

fn run_nqueens(args: &Args, seed: u64) {
    let size = args.size;
    for solver in IDIOMATIC_PERMUTATION {
        let build = |budget: Arc<Budget<usize>>| {
            // radiate's permutation example (examples/rust/TSP, lines 13-19): PermutationCodec,
            // population 250, minimizing, the other engine defaults (roulette offspring
            // selection, tournament of 3 for the survivors). radiate's own N-Queens example
            // (examples/rust/nqueens) evolves integers with row conflicts in the fitness, not a
            // permutation, so it doesn't apply here.
            let builder = GeneticEngine::builder()
                .codec(PermutationCodec::new((0..size).collect()))
                .raw_fitness_fn(move |genotype: &Genotype<PermutationChromosome<usize>>| {
                    let genes = genotype[0].as_slice();
                    budget.record(nqueens(genes.iter().map(|gene| *gene.allele())), || {
                        genes.iter().map(|gene| *gene.allele()).collect()
                    })
                })
                .minimizing()
                .population_size(250);
            match solver {
                // the example as it is: PMXCrossover(0.4) and SwapMutator(0.05)
                "ga" => builder
                    .alter(alters!(PMXCrossover::new(0.4), SwapMutator::new(0.05)))
                    .build(),
                // the permutation recipe: PMX at 0.5 and Swap (the example's mutator) at 0.01.
                // Shuffle crossover is left out: it swaps genes between the parents position by
                // position, so its children aren't permutations, and the engine replaces them
                // with random ones (radiate-engines-1.3.1/src/steps/filter.rs)
                _ => builder
                    .alter(alters!(PMXCrossover::new(0.5), SwapMutator::new(0.01)))
                    .build(),
            }
        };
        run_single(args, seed, solver, true, 0.0, build, |v: &usize| v.to_string());
    }
}

// ---------------------------------------------------------------------------------------------
// Rastrigin, Rosenbrock and Ackley
// ---------------------------------------------------------------------------------------------

fn run_real(args: &Args, seed: u64) {
    let size = args.size;
    let (function, range) = real_problem(&args.problem).unwrap();
    let unimodal = args.problem == "rosenbrock";
    for solver in IDIOMATIC_REAL {
        let range = range.clone();
        let (lower, upper) = (range.start, range.end);
        let build = |budget: Arc<Budget<f64>>| {
            // the bounds: FloatCodec::vector draws the genes in the range and sets it as their
            // bounds; every alterer writes through FloatGene::set_allele, which clamps to them
            // (radiate-core-1.3.1/src/genome/chromosomes/float.rs, lines 82-85)
            let builder = GeneticEngine::builder()
                .codec(FloatCodec::vector(size, range))
                .raw_fitness_fn(move |genotype: &Genotype<FloatChromosome<f64>>| {
                    let x = alleles(genotype);
                    budget.check_bounds(&x, lower, upper);
                    budget.record(function(&x), || x)
                })
                .minimizing();
            let builder = if unimodal {
                // radiate's Rosenbrock example (examples/rust/rosenbrock, lines 13-20): Boltzmann
                // offspring selection with temperature 4, the other engine defaults (population
                // 100, tournament of 3 for the survivors)
                builder.offspring_selector(BoltzmannSelector::new(4.0))
            } else {
                // radiate's Rastrigin example (examples/rust/rastrigin, lines 10-17, also on the
                // guide's examples page): population 500, the other engine defaults (roulette
                // offspring selection, tournament of 3). radiate has no Ackley example; the
                // Rastrigin example is its example of a multimodal real function, so Ackley uses
                // it too.
                builder.population_size(500)
            };
            // the crossover rate of the example: Rosenbrock's MeanCrossover(0.75), Rastrigin's
            // UniformCrossover(0.5)
            let rate = if unimodal { 0.75 } else { 0.5 };
            match solver {
                // the examples as they are: MeanCrossover(0.75) and ArithmeticMutator(0.1)
                // (Rosenbrock), UniformCrossover(0.5) and ArithmeticMutator(0.01) (Rastrigin)
                "ga" if unimodal => builder
                    .alter(alters!(MeanCrossover::new(0.75), ArithmeticMutator::new(0.1)))
                    .build(),
                "ga" => builder
                    .alter(alters!(UniformCrossover::new(0.5), ArithmeticMutator::new(0.01)))
                    .build(),
                // the continuous recipe with Blend crossover and Arithmetic mutation (the
                // examples' mutator) at 0.01
                "ga_blend" => builder
                    .alter(alters!(BlendCrossover::new(rate, 0.5), ArithmeticMutator::new(0.01)))
                    .build(),
                // the continuous recipe with Intermediate crossover and Arithmetic mutation at
                // 0.01
                _ => builder
                    .alter(alters!(
                        IntermediateCrossover::new(rate, 0.5),
                        ArithmeticMutator::new(0.01)
                    ))
                    .build(),
            }
        };
        run_single(args, seed, solver, true, 0.01, build, |v: &f64| format!("{v:?}"));
    }
}

// ---------------------------------------------------------------------------------------------
// Multi-objective
// ---------------------------------------------------------------------------------------------

/// The distinct non-dominated points (to minimize) among `points`: their indices
fn non_dominated(points: &[Vec<f64>]) -> Vec<usize> {
    let dominates = |a: &Vec<f64>, b: &Vec<f64>| {
        a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
    };
    let mut front: Vec<usize> = Vec::new();
    for (i, point) in points.iter().enumerate() {
        if !points.iter().any(|other| dominates(other, point))
            && !front.iter().any(|&j| &points[j] == point)
        {
            front.push(i);
        }
    }
    front
}

/// A JSON array of numbers at full precision
fn json_row(row: &[f64]) -> String {
    let values: Vec<String> = row.iter().map(|v| format!("{v:?}")).collect();
    format!("[{}]", values.join(","))
}

fn json_rows(rows: &[Vec<f64>]) -> String {
    let rows: Vec<String> = rows.iter().map(|row| json_row(row)).collect();
    format!("[{}]", rows.join(","))
}

/// The matched settings (README, "Scenarios"): NSGA-II with 100 individuals (92 with 3
/// objectives), SBX with η 15 at 0.9 and polynomial mutation with η 20 at 1 / n; NSGA-III with
/// Das-Dennis reference directions (99 divisions with 2 objectives: 100 directions, population
/// 100; 12 with 3: 91 directions, population 92), SBX with η 30 at 1, the same mutation.
///
/// NSGA-II and NSGA-III select the next population from parents plus offspring. In radiate's
/// engine that is `population_size(2 * mu)` with `offspring_fraction(0.5)`: every generation keeps
/// mu survivors of the 2 mu individuals (parents and their offspring) by the NSGA-II / NSGA-III
/// selector, and breeds mu offspring. The front reported is the non-dominated set of the mu the
/// survivor selector keeps of the last 2 mu, i.e. NSGA-II's (NSGA-III's) population after the
/// last generation (rule 7.2). radiate's own Pareto archive (`front_size`, by default 800 to 900
/// of all the non-dominated individuals seen) is not reported: the other libraries report their
/// population.
///
/// Differences from the textbook algorithms, all radiate's own:
/// - the mating pool is drawn from all 2 mu individuals (parents and offspring of the previous
///   generation), not from the mu survivors
/// - NSGA-II: the crowding distance is computed over the whole population, not per front
///   (radiate-core-1.3.1/src/objectives/pareto.rs, `crowding_distance`)
/// - NSGA-III: the objectives are normalized by the population's ideal and nadir points, without
///   the extreme-point hyperplane (radiate-selectors-1.3.1/src/nsga3.rs, `ObjectiveBounds`)
/// - `max_age` is off: radiate by default replaces individuals older than 20 generations with
///   random ones
/// - scores are f32 (radiate's `Score`); the front is recomputed in f64
///
/// Bugs, not worked around (rule 8.4, pkalivas/radiate#28):
/// - radiate's SimulatedBinaryCrossover changes one of the two parents (the other is re-evaluated
///   unchanged), crosses each variable with probability 0.5, and centres the child on
///   (p1 - p2) / 2 where SBX has (p1 + p2) / 2, clamped to the bounds
///   (radiate-alters-1.3.1/src/crossovers/simulated_binary.rs, lines 50-68)
/// - radiate's PolynomialMutator puts the mutated variable at lower + δ (upper - lower) where
///   polynomial mutation has x + δ (upper - lower), so it lands next to a bound
///   (radiate-alters-1.3.1/src/mutators/polynomial.rs, line 58)
fn run_front(args: &Args, seed: u64) {
    let (function, variables, objectives) = front_problem(&args.problem, args.size).unwrap();
    // (population, NSGA-III divisions)
    let (mu, divisions) = match objectives {
        2 => (100, 99),
        3 => (92, 12),
        _ => {
            eprintln!("radiate adapter: the multi-objective settings are for 2 and 3 objectives");
            return;
        }
    };
    let rate = 1.0 / variables as f32;

    for solver in ["nsga2", "nsga3"] {
        let (counter, generations, time_s, points, solutions) =
            random_provider::scoped_seed(seed, || {
                let start = Instant::now();
                let counter: Arc<Budget<()>> = Budget::new(args, start, true, f64::NEG_INFINITY);
                let fitness = Arc::clone(&counter);
                let builder = GeneticEngine::builder()
                    .codec(FloatCodec::vector(variables, 0.0_f64..1.0))
                    .raw_fitness_fn(move |genotype: &Genotype<FloatChromosome<f64>>| {
                        fitness.count();
                        let x = alleles(genotype);
                        fitness.check_bounds(&x, 0.0, 1.0);
                        function(&x, objectives)
                            .into_iter()
                            .map(|v| v as f32)
                            .collect::<Vec<f32>>()
                    })
                    .multi_objective(vec![Optimize::Minimize; objectives])
                    .population_size(2 * mu)
                    .offspring_fraction(0.5)
                    .max_age(usize::MAX);
                let engine = match solver {
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
                        .survivor_selector(NSGA3Selector::new(divisions))
                        .alter(alters!(
                            SimulatedBinaryCrossover::new(1.0, 30.0),
                            PolynomialMutator::new(rate, 20.0)
                        ))
                        .build(),
                };
                let generation = run_engine(&counter, engine);
                // the survivors of the last generation, as NSGA-II (NSGA-III) selects them
                let population: &[Phenotype<FloatChromosome<f64>>] =
                    generation.population().as_ref();
                let survivors = if solver == "nsga2" {
                    NSGA2Selector::new().select(population, generation.objective(), mu)
                } else {
                    NSGA3Selector::new(divisions).select(population, generation.objective(), mu)
                };
                let time_s = start.elapsed().as_secs_f64();
                // their objectives in f64 (radiate keeps f32 scores); not counted as evaluations
                let solutions: Vec<Vec<f64>> = survivors
                    .iter()
                    .map(|&i| alleles(population[i].genotype()))
                    .collect();
                let points: Vec<Vec<f64>> =
                    solutions.iter().map(|x| function(x, objectives)).collect();
                (counter, generation.index(), time_s, points, solutions)
            });
        let front = non_dominated(&points);
        let front_points: Vec<Vec<f64>> = front.iter().map(|&i| points[i].clone()).collect();
        let front_solutions: Vec<Vec<f64>> = front.iter().map(|&i| solutions[i].clone()).collect();
        println!(
            "{{\"library\":\"radiate\",\"solver\":\"{solver}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{generations},\"evaluations\":{},\"outside\":{},\"front\":{},\"solutions\":{}}}",
            args.problem,
            args.size,
            args.mode,
            counter.evaluations(),
            counter.outside(),
            json_rows(&front_points),
            json_rows(&front_solutions),
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The `values` command: the adapter's own fitness functions at the given solutions (rule 1.2)
// ---------------------------------------------------------------------------------------------

fn parse_numbers(line: &str) -> Vec<f64> {
    line.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| match item {
            "true" => 1.0,
            "false" => 0.0,
            number => number.parse().expect("a number"),
        })
        .collect()
}

fn values(problem: &str, size: usize) {
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        let x = parse_numbers(&line);
        if let Some((function, variables, objectives)) = front_problem(problem, size) {
            assert_eq!(x.len(), variables, "{problem} {size} has {variables} variables");
            println!("{}", json_row(&function(&x, objectives)));
            continue;
        }
        let value = match problem {
            "onemax" => onemax(x.iter().map(|&v| v != 0.0)),
            "nqueens" => nqueens(x.iter().map(|&v| v as usize)),
            other => real_problem(other).expect("a known problem").0(&x),
        };
        println!("{value:?}");
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() == 3 && raw[0] == "values" {
        values(&raw[1], raw[2].parse().expect("size"));
        return;
    }
    if raw.len() != 7 {
        eprintln!(
            "usage: ga_bench_radiate <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>\n       ga_bench_radiate values <problem> <size>"
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
            "rastrigin" | "rosenbrock" | "ackley" => run_real(&args, seed),
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
mod tests {
    use super::*;

    #[test]
    fn fitness_values() {
        // 0 at the optimum, and the values of problems.py at fixed points
        let s: Vec<f64> = (0..10).map(shift).collect();
        let x: Vec<f64> = (0..10).map(|i| 0.5 * (i % 7) as f64 - 1.5).collect();
        assert!(rastrigin(&s).abs() < 1e-12);
        assert!(ackley(&s).abs() < 1e-12);
        assert!(rosenbrock(&[1.0; 10]).abs() < 1e-12);
        assert!((rastrigin(&x) - 87.78147018265213).abs() < 1e-9);
        assert!((ackley(&x) - 5.149902035382837).abs() < 1e-9);
        assert_eq!(onemax([true, false, true].into_iter()), 2.0);
        // all 8 queens on one diagonal: 7 conflicts
        assert_eq!(nqueens(0..8), 7.0);
        let mut zdt = vec![0.0; 30];
        zdt[0] = 0.25;
        assert_eq!(zdt1(&zdt, 2), vec![0.25, 0.5]);
        // on the optimal fronts: DTLZ2's objectives on the unit sphere, DTLZ1's summing to 0.5
        let mut x2 = vec![0.5; 12];
        x2[0] = 0.3;
        x2[1] = 0.6;
        assert!((dtlz2(&x2, 3).iter().map(|f| f * f).sum::<f64>() - 1.0).abs() < 1e-12);
        let mut x1 = vec![0.5; 7];
        x1[0] = 0.3;
        x1[1] = 0.6;
        assert!((dtlz1(&x1, 3).iter().sum::<f64>() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn parse_solutions() {
        assert_eq!(parse_numbers("[1, 0,true]"), vec![1.0, 0.0, 1.0]);
        assert_eq!(parse_numbers("[-0.5,1e-7]"), vec![-0.5, 1e-7]);
        assert_eq!(json_row(&[0.1, -2.0]), "[0.1,-2.0]");
    }
}
