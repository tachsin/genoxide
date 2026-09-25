//! Benchmark adapter for the genetic_algorithm crate (https://crates.io/crates/genetic_algorithm).
//!
//! Usage: ga_bench_genetic_algorithm <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//!        ga_bench_genetic_algorithm values <problem> <size>   (JSON solutions on stdin, one value per line)
//! Prints one JSON line per solver per seed, see ../../README.md for the fields.
//!
//! The methods, their settings and where the library recommends them are on the library's page,
//! docs/benchmarks/libraries/genetic_algorithm.md. The citations in the comments below are to the
//! files of the crate 0.27.3 (README.md, AGENTS.md, AGENTS_TEMPLATES.md, examples/).
//!
//! How the runs follow the benchmark rules (docs/benchmarks/rules.md):
//! - Evaluations are counted in the fitness functions, every call (rule 3).
//! - A run ends at the target (the strategy's `with_target_fitness_score`), or when the budget or
//!   the time is used up (the abort flag, which Evolve and HillClimb check once per generation).
//! - Rule 2.2: `with_max_stale_generations` detects convergence (generations without
//!   improvement), so it ends an attempt and the method starts again. The library's restart
//!   mechanism is `call_repeatedly(n)` (AGENTS.md, "Choosing a call variant"), but with
//!   `with_rng_seed_from_u64` every repeat gets the same seed and repeats the same run
//!   (strategy/evolve/builder.rs and strategy/hill_climb/builder.rs: every repeat is
//!   `self.clone().try_into()`, whose rng is `SmallRng::seed_from_u64(seed)`). So `restarts`
//!   below does what `call_repeatedly` does, one run after the other until a run is conclusive
//!   (target or abort), keeping the best, from a new random start with the seed
//!   `seed * 1000 + restart`. No run uses a limit that's only a budget (`with_max_generations`).
//! - Rule 2.4: the RangeGenotype keeps every gene inside its allele range itself (random values
//!   drawn from the range, mutations clamped to it: genotype/range.rs). The fitness counts the
//!   evaluated solutions outside the bounds, as the library proposed them, and each continuous
//!   run prints the count as `outside`.
//! - Single-threaded: no `with_par_fitness`, no `call_par_*`. The only other thread is the timer,
//!   which sleeps until the time cap.
//! - Seeded with `with_rng_seed_from_u64`.

use genetic_algorithm::strategy::evolve::prelude::*;
use genetic_algorithm::strategy::hill_climb::prelude::*;
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Shared evaluation budget: counts fitness evaluations and sets the abort flag when the budget
/// (evaluations or seconds) is used up.
#[derive(Clone, Debug)]
struct Budget {
    evaluations: Arc<AtomicUsize>,
    // evaluated solutions outside the problem's bounds (rule 2.4)
    outside: Arc<AtomicUsize>,
    max_evaluations: usize,
    abort_flag: Arc<AtomicBool>,
}
impl Budget {
    fn new(max_evaluations: usize) -> Self {
        Self {
            evaluations: Arc::new(AtomicUsize::new(0)),
            outside: Arc::new(AtomicUsize::new(0)),
            max_evaluations,
            abort_flag: Arc::new(AtomicBool::new(false)),
        }
    }
    fn count(&self) {
        if self.evaluations.fetch_add(1, Ordering::Relaxed) + 1 >= self.max_evaluations {
            self.abort_flag.store(true, Ordering::Relaxed);
        }
    }
    fn evaluations(&self) -> usize {
        self.evaluations.load(Ordering::Relaxed)
    }
    fn aborted(&self) -> bool {
        self.abort_flag.load(Ordering::Relaxed)
    }
    /// Sets the abort flag after max_seconds, stopped by dropping the returned sender
    fn start_timer(&self, max_seconds: f64) -> (mpsc::Sender<()>, std::thread::JoinHandle<()>) {
        let (sender, receiver) = mpsc::channel::<()>();
        let abort_flag = self.abort_flag.clone();
        let handle = std::thread::spawn(move || {
            if let Err(mpsc::RecvTimeoutError::Timeout) =
                receiver.recv_timeout(Duration::from_secs_f64(max_seconds))
            {
                abort_flag.store(true, Ordering::Relaxed);
            }
        });
        (sender, handle)
    }
}

// ---------------------------------------------------------------------------------------------
// Fitness functions, identical to benchmarks/problems.py
// ---------------------------------------------------------------------------------------------

fn onemax_value(bits: &[bool]) -> usize {
    bits.iter().filter(|&&bit| bit).count()
}

/// Diagonal conflicts of queens at (i, order[i]): for each diagonal, its queens minus one, O(n)
fn nqueens_value(order: &[usize]) -> usize {
    let size = order.len();
    let mut left_diagonal = vec![0usize; 2 * size - 1];
    let mut right_diagonal = vec![0usize; 2 * size - 1];
    for (i, &column) in order.iter().enumerate() {
        left_diagonal[i + column] += 1;
        right_diagonal[size - 1 - i + column] += 1;
    }
    left_diagonal
        .iter()
        .chain(right_diagonal.iter())
        .map(|&count| count.saturating_sub(1))
        .sum()
}

// Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift
// towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
fn shift(i: usize) -> f64 {
    2.0 * ((37 * i + 11) % 101) as f64 / 101.0 - 1.0
}

fn rastrigin_value(x: &[f64]) -> f64 {
    10.0 * x.len() as f64
        + x.iter()
            .enumerate()
            .map(|(i, x)| {
                let x = x - shift(i);
                x * x - 10.0 * (2.0 * std::f64::consts::PI * x).cos()
            })
            .sum::<f64>()
}

fn rosenbrock_value(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|pair| 100.0 * (pair[1] - pair[0] * pair[0]).powi(2) + (1.0 - pair[0]).powi(2))
        .sum()
}

fn ackley_value(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let shifted = || x.iter().enumerate().map(|(i, x)| x - shift(i));
    let squares = shifted().map(|x| x * x).sum::<f64>() / n;
    let cosines = shifted()
        .map(|x| (2.0 * std::f64::consts::PI * x).cos())
        .sum::<f64>()
        / n;
    -20.0 * (-0.2 * squares.sqrt()).exp() - cosines.exp() + 20.0 + std::f64::consts::E
}

#[derive(Clone, Debug)]
struct OneMax(Budget);
impl Fitness for OneMax {
    type Genotype = BinaryGenotype;
    fn calculate_for_chromosome(
        &mut self,
        chromosome: &FitnessChromosome<Self>,
        _genotype: &FitnessGenotype<Self>,
    ) -> Option<FitnessValue> {
        self.0.count();
        Some(onemax_value(&chromosome.genes) as FitnessValue)
    }
}

#[derive(Clone, Debug)]
struct NQueens(Budget);
impl Fitness for NQueens {
    // AGENTS.md "Which Genotype?": UniqueGenotype for a permutation (N-Queens, TSP); the
    // examples use u8 alleles for the columns
    type Genotype = UniqueGenotype<u8>;
    fn calculate_for_chromosome(
        &mut self,
        chromosome: &FitnessChromosome<Self>,
        _genotype: &FitnessGenotype<Self>,
    ) -> Option<FitnessValue> {
        self.0.count();
        let order: Vec<usize> = chromosome.genes.iter().map(|&gene| gene as usize).collect();
        Some(nqueens_value(&order) as FitnessValue)
    }
}

// FitnessValue is isize, so a real value is scaled by a precision (AGENTS.md "Critical:
// FitnessValue is isize"); 1e-5 as in AGENTS.md and AGENTS_TEMPLATES.md "Continuous Optimization".
// The value is rounded up, not truncated as `fitness_value` does, so the library's target of
// 0.01 / 1e-5 = 1000 units means a value of at most 0.01: truncation would count 0.010009 as
// reached. Only the target test depends on it.
const PRECISION: f64 = 1e-5;
const REAL_TARGET: f64 = 0.01;
fn scaled(value: f64) -> FitnessValue {
    (value / PRECISION).ceil() as FitnessValue
}

#[derive(Clone, Debug)]
struct Real {
    budget: Budget,
    function: fn(&[f64]) -> f64,
    low: f64,
    high: f64,
}
impl Fitness for Real {
    type Genotype = RangeGenotype<f64>;
    fn calculate_for_chromosome(
        &mut self,
        chromosome: &FitnessChromosome<Self>,
        _genotype: &FitnessGenotype<Self>,
    ) -> Option<FitnessValue> {
        self.budget.count();
        if chromosome.genes.iter().any(|&x| x < self.low || x > self.high) {
            self.budget.outside.fetch_add(1, Ordering::Relaxed);
        }
        Some(scaled((self.function)(&chromosome.genes)))
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

/// The outcome of one run: the best solution found and its value, computed in f64 by the
/// fitness function above (not the library's scaled score)
struct Outcome {
    solver: &'static str,
    generations: usize,
    best: String,
    target: String,
    success: bool,
    solution: String,
    // a continuous problem: the run reports `outside` (rule 2.4)
    bounded: bool,
}

fn print_result(args: &Args, seed: u64, outcome: &Outcome, time_s: f64, budget: &Budget) {
    let outside = if outcome.bounded {
        format!(",\"outside\":{}", budget.outside.load(Ordering::Relaxed))
    } else {
        String::new()
    };
    println!(
        "{{\"library\":\"genetic_algorithm\",\"solver\":\"{}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{},\"time_s\":{:.6},\"generations\":{},\"evaluations\":{},\"best\":{},\"target\":{},\"success\":{},\"solution\":{}{}}}",
        outcome.solver,
        args.problem,
        args.size,
        args.mode,
        seed,
        time_s,
        outcome.generations,
        budget.evaluations(),
        outcome.best,
        outcome.target,
        outcome.success,
        outcome.solution,
        outside,
    );
}

/// Times one run, from before the initial population to the end
fn run<F: FnOnce(&Budget) -> Outcome>(args: &Args, seed: u64, solve: F) {
    let budget = Budget::new(args.max_evaluations);
    let (timer, timer_handle) = budget.start_timer(args.max_seconds);
    let now = Instant::now();
    let outcome = solve(&budget);
    let time_s = now.elapsed().as_secs_f64();
    drop(timer);
    timer_handle.join().unwrap();
    print_result(args, seed, &outcome, time_s, &budget);
}

/// `call_repeatedly` with a seed per run (see the top of the file): `once(seed)` runs the
/// strategy and returns (its best genes, whether it is conclusive, its generations). Runs until
/// one is conclusive (the target reached or the run aborted), and returns the best genes by
/// `value` (minimized) and the generations of all runs.
fn restarts<T>(
    seed: u64,
    mut once: impl FnMut(u64) -> (Option<Vec<T>>, bool, usize),
    value: impl Fn(&[T]) -> f64,
) -> (Vec<T>, f64, usize) {
    let mut best: Option<(Vec<T>, f64)> = None;
    let mut generations = 0;
    for restart in 0.. {
        let (genes, conclusive, run_generations) = once(seed * 1000 + restart);
        generations += run_generations;
        if let Some(genes) = genes {
            let run_value = value(&genes);
            if best.as_ref().is_none_or(|(_, best_value)| run_value < *best_value) {
                best = Some((genes, run_value));
            }
        }
        if conclusive {
            break;
        }
    }
    let (genes, best_value) = best.expect("a run evaluates at least one chromosome");
    (genes, best_value, generations)
}

fn json_list<T: std::fmt::Debug>(values: impl Iterator<Item = T>) -> String {
    let values: Vec<String> = values.map(|v| format!("{v:?}")).collect();
    format!("[{}]", values.join(","))
}

fn onemax(args: &Args, seed: u64) {
    let target = args.size;
    run(args, seed, |budget| {
        let genotype = BinaryGenotype::builder()
            .with_genes_size(args.size)
            .build()
            .unwrap();
        // the operator types are generic parameters of the builder, so build it per arm. The
        // only ending condition is the target (with_max_stale_generations isn't set), so a run
        // ends at the target or through the abort flag
        macro_rules! builder {
            () => {
                Evolve::builder()
                    .with_genotype(genotype.clone())
                    .with_fitness(OneMax(budget.clone()))
                    .with_target_fitness_score(target as FitnessValue)
                    .with_abort_flag(budget.abort_flag.clone())
                    .with_rng_seed_from_u64(seed)
            };
        }
        let (genes, generations) = match args.mode.as_str() {
            // matched, as DEAP eaSimple: population 300, tournament of 3, two-point crossover
            // with probability 0.5, mutation of ~1 bit on 20% of the children, no elitism.
            // Differences: DEAP selects parents by tournament with replacement; this library
            // selects the survivors from parents + offspring by tournament without replacement,
            // so the selection pressure comes from the surplus. A replacement_rate of 1.0 (only
            // offspring survive) would select 300 out of 300 offspring, i.e. no selection at all,
            // hence 0.5. MutateSingleGene(0.2) mutates exactly one bit of 20% of the children,
            // where a bit flip at 1 / size flips ~1 bit.
            "matched" => {
                let evolve = builder!()
                    .with_target_population_size(300)
                    .with_select(SelectTournament::new(0.5, 0.0, 3))
                    .with_crossover(CrossoverMultiPoint::new(1.0, 0.5, 2, false))
                    .with_mutate(MutateSingleGene::new(0.2))
                    .call()
                    .unwrap();
                (evolve.best_genes(), evolve.state.current_generation)
            }
            // idiomatic: AGENTS.md "If unsure, start here", for binary genotypes:
            // SelectTournament(0.5, 0.02, 4), CrossoverUniform(0.7, 0.8), MutateSingleGene(0.2);
            // population 100 as in README.md "Quick Usage", which is this problem (100 genes,
            // count the true values, target 100)
            _ => {
                let evolve = builder!()
                    .with_target_population_size(100)
                    .with_select(SelectTournament::new(0.5, 0.02, 4))
                    .with_crossover(CrossoverUniform::new(0.7, 0.8))
                    .with_mutate(MutateSingleGene::new(0.2))
                    .call()
                    .unwrap();
                (evolve.best_genes(), evolve.state.current_generation)
            }
        };
        let genes = genes.expect("the initial population is evaluated");
        let best = onemax_value(&genes);
        Outcome {
            solver: "evolve",
            generations,
            best: best.to_string(),
            target: target.to_string(),
            success: best >= target,
            solution: json_list(genes.iter().map(|&bit| bit as u8)),
            bounded: false,
        }
    });
}

fn nqueens_genotype(size: usize) -> UniqueGenotype<u8> {
    assert!(size <= 256, "nqueens size must be <= 256 for u8 genes");
    UniqueGenotype::builder()
        .with_allele_list((0..size).map(|v| v as u8).collect())
        .build()
        .unwrap()
}

fn nqueens(args: &Args, seed: u64) {
    // HillClimb, the strategy README.md recommends for permutation problems ("When to use which
    // strategy?"), as examples/hill_climb_nqueens.rs (the 64-queens board): the Stochastic
    // variant (AGENTS.md "Which HillClimb Variant?": large genome, plateau traversal; "Use
    // Stochastic with call_repeatedly for genomes >20 genes"), max_stale_generations(10000) and
    // with_replace_on_equal_fitness(true) ("crucial for this problem"). It ends after 10000
    // generations without improvement, so it restarts, as call_repeatedly would.
    run(args, seed, |budget| {
        let order = |genes: &[u8]| genes.iter().map(|&gene| gene as usize).collect::<Vec<_>>();
        let (genes, best, generations) = restarts(
            seed,
            |run_seed| {
                let hill_climb = HillClimb::builder()
                    .with_genotype(nqueens_genotype(args.size))
                    .with_variant(HillClimbVariant::Stochastic)
                    .with_max_stale_generations(10000)
                    .with_fitness(NQueens(budget.clone()))
                    .with_fitness_ordering(FitnessOrdering::Minimize)
                    .with_target_fitness_score(0)
                    .with_replace_on_equal_fitness(true)
                    .with_abort_flag(budget.abort_flag.clone())
                    .with_rng_seed_from_u64(run_seed)
                    .call()
                    .unwrap();
                let conclusive = hill_climb.best_fitness_score() == Some(0) || budget.aborted();
                (
                    hill_climb.best_genes(),
                    conclusive,
                    hill_climb.state.current_generation,
                )
            },
            |genes| nqueens_value(&order(genes)) as f64,
        );
        Outcome {
            solver: "hill_climb",
            generations,
            best: (best as usize).to_string(),
            target: "0".to_string(),
            success: best == 0.0,
            solution: json_list(genes.iter()),
            bounded: false,
        }
    });
}

/// (lower bound, upper bound, function) of a real-valued problem
fn real_problem(problem: &str) -> (f64, f64, fn(&[f64]) -> f64) {
    match problem {
        "rastrigin" => (-5.12, 5.12, rastrigin_value),
        "rosenbrock" => (-5.0, 10.0, rosenbrock_value),
        _ => (-32.768, 32.768, ackley_value),
    }
}

fn real_outcome(solver: &'static str, genes: Vec<f64>, best: f64, generations: usize) -> Outcome {
    Outcome {
        solver,
        generations,
        best: format!("{best:?}"),
        target: format!("{REAL_TARGET:?}"),
        success: best <= REAL_TARGET,
        solution: json_list(genes.iter()),
        bounded: true,
    }
}

/// Evolve on Rastrigin, Rosenbrock and Ackley
fn real_evolve(args: &Args, seed: u64) {
    let (low, high, function) = real_problem(&args.problem);
    let width = high - low;
    run(args, seed, |budget| {
        let (genes, best, generations) = restarts(
            seed,
            |run_seed| {
                // Rule 6.2's order (a stated preference, then the example for the problem type,
                // then the default); the page explains each step.
                // The example: examples/evolve_range_float.rs, the library's Evolve example for a
                // real function: population 100, SelectTournament(0.5, 0.02, 4),
                // MutateMultiGene(2, 0.2), precision 1e-5.
                // Stated preferences, which come first:
                // - its comments call StepScaled(vec![0.1, 0.01, 0.001, 0.0001]) (on its range
                //   0..=1; the same shares of the range here) the "best approach for this problem,
                //   converges fast, but needs low max_stale_generations to trigger next scale",
                //   and give the low value, .with_max_stale_generations(100), commented out next
                //   to the 100_000 it runs with: so StepScaled with 100;
                // - AGENTS.md "Which Crossover?" recommends CrossoverUniform or
                //   CrossoverSinglePoint for a RangeGenotype, where the example has
                //   CrossoverMultiPoint(0.7, 0.8, 9, false): so CrossoverUniform(0.7, 0.8), the
                //   rates of the example and of AGENTS.md's presets.
                // The step advances after max_stale_generations without improvement (AGENTS.md
                // "Scale advancement"), and the attempt ends in the last one.
                let genotype = RangeGenotype::<f64>::builder()
                    .with_genes_size(args.size)
                    .with_allele_range(low..=high)
                    .with_mutation_type(MutationType::StepScaled(
                        [0.1, 0.01, 0.001, 0.0001]
                            .iter()
                            .map(|share| share * width)
                            .collect(),
                    ))
                    .build()
                    .unwrap();
                let evolve = Evolve::builder()
                    .with_genotype(genotype)
                    .with_target_population_size(100)
                    .with_max_stale_generations(100)
                    .with_fitness(Real {
                        budget: budget.clone(),
                        function,
                        low,
                        high,
                    })
                    .with_fitness_ordering(FitnessOrdering::Minimize)
                    .with_target_fitness_score(scaled(REAL_TARGET))
                    .with_select(SelectTournament::new(0.5, 0.02, 4))
                    .with_crossover(CrossoverUniform::new(0.7, 0.8))
                    .with_mutate(MutateMultiGene::new(2, 0.2))
                    .with_abort_flag(budget.abort_flag.clone())
                    .with_rng_seed_from_u64(run_seed)
                    .call()
                    .unwrap();
                let conclusive = evolve
                    .best_fitness_score()
                    .is_some_and(|score| score <= scaled(REAL_TARGET))
                    || budget.aborted();
                (
                    evolve.best_genes(),
                    conclusive,
                    evolve.state.current_generation,
                )
            },
            function,
        );
        real_outcome("evolve", genes, best, generations)
    });
}

/// HillClimb on Rosenbrock (continuous, unimodal)
fn real_hill_climb(args: &Args, seed: u64) {
    let (low, high, function) = real_problem(&args.problem);
    let width = high - low;
    run(args, seed, |budget| {
        let (genes, best, generations) = restarts(
            seed,
            |run_seed| {
                // HillClimb for a "Convex search space, few local optima" (README.md "When to
                // use which strategy?"), SteepestAscent for a small genome (AGENTS.md "Which
                // HillClimb Variant?"), as examples/hill_climb_range.rs: StepScaled(vec![0.1,
                // 0.01, 0.001, 0.0001, 0.00001]) on the range 0..=1 (here the same steps as
                // shares of the range) and max_stale_generations(1), which moves to the next
                // step after a generation without improvement and ends the run in the last.
                // AGENTS.md "Exact local optimum needed: SteepestAscent + call_repeatedly(n)".
                let genotype = RangeGenotype::<f64>::builder()
                    .with_genes_size(args.size)
                    .with_allele_range(low..=high)
                    .with_mutation_type(MutationType::StepScaled(
                        [0.1, 0.01, 0.001, 0.0001, 0.00001]
                            .iter()
                            .map(|share| share * width)
                            .collect(),
                    ))
                    .build()
                    .unwrap();
                let hill_climb = HillClimb::builder()
                    .with_genotype(genotype)
                    .with_variant(HillClimbVariant::SteepestAscent)
                    .with_max_stale_generations(1)
                    .with_fitness(Real {
                        budget: budget.clone(),
                        function,
                        low,
                        high,
                    })
                    .with_fitness_ordering(FitnessOrdering::Minimize)
                    .with_target_fitness_score(scaled(REAL_TARGET))
                    .with_abort_flag(budget.abort_flag.clone())
                    .with_rng_seed_from_u64(run_seed)
                    .call()
                    .unwrap();
                let conclusive = hill_climb
                    .best_fitness_score()
                    .is_some_and(|score| score <= scaled(REAL_TARGET))
                    || budget.aborted();
                (
                    hill_climb.best_genes(),
                    conclusive,
                    hill_climb.state.current_generation,
                )
            },
            function,
        );
        real_outcome("hill_climb", genes, best, generations)
    });
}

// ---------------------------------------------------------------------------------------------
// values: the adapter's fitness functions on the solutions read from stdin (rule 1.2)
// ---------------------------------------------------------------------------------------------

/// One JSON array of numbers per line
fn parse_solution(line: &str) -> Vec<f64> {
    let inner = line.trim().trim_start_matches('[').trim_end_matches(']');
    inner
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

fn values(problem: &str) {
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        let x = parse_solution(&line);
        match problem {
            "onemax" => {
                let bits: Vec<bool> = x.iter().map(|&v| v != 0.0).collect();
                println!("{}", onemax_value(&bits));
            }
            "nqueens" => {
                let order: Vec<usize> = x.iter().map(|&v| v as usize).collect();
                println!("{}", nqueens_value(&order));
            }
            "rastrigin" | "rosenbrock" | "ackley" => {
                println!("{:?}", (real_problem(problem).2)(&x));
            }
            other => {
                eprintln!("genetic_algorithm adapter: no values for {other}");
                std::process::exit(2);
            }
        }
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() == 3 && raw[0] == "values" {
        values(&raw[1]);
        return;
    }
    if raw.len() != 7 {
        eprintln!("usage: ga_bench_genetic_algorithm <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>");
        eprintln!("       ga_bench_genetic_algorithm values <problem> <size>");
        std::process::exit(2);
    }
    let args = Args {
        problem: raw[0].clone(),
        size: raw[1].parse().unwrap(),
        mode: raw[2].clone(),
        seed_from: raw[3].parse().unwrap(),
        seed_to: raw[4].parse().unwrap(),
        max_evaluations: raw[5].parse().unwrap(),
        max_seconds: raw[6].parse().unwrap(),
    };
    for seed in args.seed_from..=args.seed_to {
        match args.problem.as_str() {
            "onemax" => onemax(&args, seed),
            "nqueens" => nqueens(&args, seed),
            "rastrigin" | "ackley" => real_evolve(&args, seed),
            "rosenbrock" => {
                real_evolve(&args, seed);
                real_hill_climb(&args, seed);
            }
            // no multi-objective strategy: print nothing
            "zdt1" | "zdt2" | "zdt3" | "dtlz1" | "dtlz2" => {}
            other => {
                eprintln!("unknown problem {}", other);
                std::process::exit(2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitness_values() {
        // 0 at the shift, and the values of the Python reference at a fixed point
        let s: Vec<f64> = (0..10).map(shift).collect();
        let x: Vec<f64> = (0..10).map(|i| 0.5 * (i % 7) as f64 - 1.5).collect();
        assert!(rastrigin_value(&s).abs() < 1e-12);
        assert!(ackley_value(&s).abs() < 1e-12);
        assert!((rastrigin_value(&x) - 87.78147018265213).abs() < 1e-9);
        assert!((ackley_value(&x) - 5.149902035382837).abs() < 1e-9);
        assert_eq!(rosenbrock_value(&[1.0; 10]), 0.0);
        assert_eq!(nqueens_value(&[3, 1, 6, 2, 5, 7, 4, 0]), 0);
        assert_eq!(nqueens_value(&[0, 1, 2, 3, 4, 5, 6, 7]), 7);
        assert_eq!(parse_solution("[1, 0.5, -2e-3]"), vec![1.0, 0.5, -0.002]);
    }

    #[test]
    fn scaled_target() {
        assert!(scaled(0.01) <= scaled(REAL_TARGET));
        assert!(scaled(0.010001) > scaled(REAL_TARGET));
    }
}
