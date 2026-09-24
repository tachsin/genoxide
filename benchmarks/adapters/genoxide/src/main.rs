//! Benchmark adapter for genoxide.
//!
//! Usage: ga_bench_genoxide <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//! Prints one JSON line per solver per seed, see ../../README.md for the fields.

use genoxide::prelude::*;
use std::f64::consts::PI;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Fitness functions, identical to the ones in the other adapters
// ---------------------------------------------------------------------------------------------

fn onemax(genome: &Bits) -> f64 {
    genome.count_ones() as f64
}

/// Number of diagonal conflicts, O(n) (same as DEAP's examples/ga/nqueens.py)
fn nqueens(genome: &Order) -> f64 {
    let size = genome.len();
    let mut left_diagonal = vec![0usize; 2 * size - 1];
    let mut right_diagonal = vec![0usize; 2 * size - 1];
    for (i, &gene) in genome.iter().enumerate() {
        left_diagonal[i + gene] += 1;
        right_diagonal[size - 1 - i + gene] += 1;
    }
    left_diagonal
        .iter()
        .chain(&right_diagonal)
        .map(|&count| count.saturating_sub(1))
        .sum::<usize>() as f64
}

fn rastrigin(genome: &Reals) -> f64 {
    10.0 * genome.len() as f64
        + genome
            .iter()
            .map(|x| x * x - 10.0 * (2.0 * PI * x).cos())
            .sum::<f64>()
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
    max_evaluations: u64,
    max_seconds: f64,
}

impl Args {
    // the budget: the target, the evaluations or the time, whichever comes first
    fn stop(&self, target: f64) -> Stop {
        Stop::target(target)
            .or(Stop::evaluations(self.max_evaluations))
            .or(Stop::time(Duration::from_secs_f64(self.max_seconds)))
    }
}

fn print_result<G: Genome>(
    args: &Args,
    seed: u64,
    solver: &str,
    outcome: &Outcome<G>,
    time_s: f64,
    target: f64,
    success: bool,
) {
    println!(
        "{{\"library\":\"genoxide\",\"solver\":\"{solver}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{},\"evaluations\":{},\"best\":{},\"target\":{target},\"success\":{success}}}",
        args.problem,
        args.size,
        args.mode,
        outcome.generations(),
        outcome.evaluations(),
        outcome.best_fitness().score().unwrap_or(f64::NAN),
    );
}

// times building the algorithm (the random initial population) and running it
fn timed<T>(run: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let result = run();
    (result, start.elapsed().as_secs_f64())
}

fn run_onemax(args: &Args, seed: u64) -> Result<()> {
    let size = args.size;
    let target = size as f64;
    let (outcome, time_s) = timed(|| {
        let builder = Ga::builder(Binary::new(size)?)
            .select(Tournament::new(3)?)
            .crossover(PointCrossover::two_point())
            .seed(seed);
        let ga = match args.mode.as_str() {
            // as DEAP eaSimple: population 300, tournament 3, two-point crossover with probability
            // 0.5, bit-flip with probability 1 / size on 20% of the children, no elitism
            "matched" => builder
                .population_size(300)
                .crossover_rate(0.5)
                .mutation_rate(0.2)
                .mutate(BitFlip::per_gene(1.0 / size as f64)?)
                .scheme(Scheme::Generational { elitism: 0 })
                .build()?,
            // the binary template of AGENTS.md
            _ => builder
                .population_size(100)
                .mutate(BitFlip::per_gene(1.0 / size as f64)?)
                .build()?,
        };
        Engine::new(ga, onemax).stop_when(args.stop(target)).run()
    });
    let outcome = outcome?;
    let success = outcome.best_fitness().score() >= Some(target);
    print_result(args, seed, "ga", &outcome, time_s, target, success);
    Ok(())
}

fn run_nqueens(args: &Args, seed: u64) -> Result<()> {
    // the permutation template of AGENTS.md: (μ+λ) with swap mutation
    let (outcome, time_s) = timed(|| {
        let ga = Ga::builder(Permutation::new(args.size)?)
            .population_size(20)
            .select(Tournament::new(2)?)
            .crossover(NoCrossover)
            .mutate(SwapMutation::new())
            .scheme(Scheme::MuPlusLambda { lambda: 20 })
            .minimize()
            .seed(seed)
            .build()?;
        Engine::new(ga, nqueens).stop_when(args.stop(0.0)).run()
    });
    let outcome = outcome?;
    let success = outcome.best_fitness().score() == Some(0.0);
    print_result(args, seed, "ga", &outcome, time_s, 0.0, success);

    // the local search template of AGENTS.md, like genetic_algorithm's stochastic hill climbing:
    // one neighbor per step, and moves to equal neighbors
    let (outcome, time_s) = timed(|| {
        let search = LocalSearch::builder(Permutation::new(args.size)?)
            .neighbor(SwapMutation::new())
            .acceptance(Acceptance::NotWorse)
            .minimize()
            .seed(seed)
            .build()?;
        Engine::new(search, nqueens).stop_when(args.stop(0.0)).run()
    });
    let outcome = outcome?;
    let success = outcome.best_fitness().score() == Some(0.0);
    print_result(args, seed, "local_search", &outcome, time_s, 0.0, success);
    Ok(())
}

const RASTRIGIN_TARGET: f64 = 0.01;

fn run_rastrigin(args: &Args, seed: u64) -> Result<()> {
    // the settings of examples/rastrigin.rs: polynomial mutation at the usual rate of 1 / length,
    // single-threaded like every adapter
    let (outcome, time_s) = timed(|| {
        let ga = Ga::builder(Real::uniform(args.size, -5.12..=5.12)?)
            .population_size(100)
            .select(Tournament::new(3)?)
            .crossover(UniformCrossover::new())
            .mutate(PolynomialMutation::per_gene(1.0 / args.size as f64, 20.0)?)
            .scheme(Scheme::Generational { elitism: 2 })
            .minimize()
            .seed(seed)
            .build()?;
        Engine::new(ga, rastrigin)
            .stop_when(args.stop(RASTRIGIN_TARGET))
            .run()
    });
    let outcome = outcome?;
    let success = outcome
        .best_fitness()
        .score()
        .is_some_and(|best| best <= RASTRIGIN_TARGET);
    print_result(args, seed, "ga", &outcome, time_s, RASTRIGIN_TARGET, success);
    Ok(())
}

fn main() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() != 7 {
        eprintln!(
            "usage: ga_bench_genoxide <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>"
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
            "onemax" => run_onemax(&args, seed)?,
            "nqueens" => run_nqueens(&args, seed)?,
            "rastrigin" => run_rastrigin(&args, seed)?,
            other => {
                eprintln!("unknown problem {other}");
                std::process::exit(2);
            }
        }
    }
    Ok(())
}
