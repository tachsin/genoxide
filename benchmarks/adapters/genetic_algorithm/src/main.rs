//! Benchmark adapter for the genetic_algorithm crate.
//!
//! Usage: ga_bench_genetic_algorithm <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//! Prints one JSON line per solver per seed, see ../../README.md for the fields.

use genetic_algorithm::strategy::evolve::prelude::*;
use genetic_algorithm::strategy::hill_climb::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Shared evaluation budget: counts fitness evaluations and sets the abort flag when the budget
/// (evaluations or seconds) is used up.
#[derive(Clone, Debug)]
struct Budget {
    evaluations: Arc<AtomicUsize>,
    max_evaluations: usize,
    abort_flag: Arc<AtomicBool>,
}
impl Budget {
    fn new(max_evaluations: usize) -> Self {
        Self {
            evaluations: Arc::new(AtomicUsize::new(0)),
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
// Fitness functions, identical to the ones in the DEAP adapter
// ---------------------------------------------------------------------------------------------

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
        Some(chromosome.genes.iter().filter(|&&gene| gene).count() as FitnessValue)
    }
}

/// Number of diagonal conflicts, O(n) (same as DEAP's examples/ga/nqueens.py)
#[derive(Clone, Debug)]
struct NQueens(Budget);
impl Fitness for NQueens {
    type Genotype = UniqueGenotype<u8>;
    fn calculate_for_chromosome(
        &mut self,
        chromosome: &FitnessChromosome<Self>,
        _genotype: &FitnessGenotype<Self>,
    ) -> Option<FitnessValue> {
        self.0.count();
        let size = chromosome.genes.len();
        let mut left_diagonal = vec![0usize; 2 * size - 1];
        let mut right_diagonal = vec![0usize; 2 * size - 1];
        for (i, &gene) in chromosome.genes.iter().enumerate() {
            left_diagonal[i + gene as usize] += 1;
            right_diagonal[size - 1 - i + gene as usize] += 1;
        }
        let conflicts: usize = left_diagonal
            .iter()
            .chain(right_diagonal.iter())
            .map(|&count| count.saturating_sub(1))
            .sum();
        Some(conflicts as FitnessValue)
    }
}

const RASTRIGIN_PRECISION: f64 = 1e-6;
// Rastrigin, Rosenbrock and Ackley
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
struct Rastrigin(Budget, fn(&[f64]) -> f64);
impl Fitness for Rastrigin {
    type Genotype = RangeGenotype<f64>;
    fn calculate_for_chromosome(
        &mut self,
        chromosome: &FitnessChromosome<Self>,
        _genotype: &FitnessGenotype<Self>,
    ) -> Option<FitnessValue> {
        self.0.count();
        Some(fitness_value((self.1)(&chromosome.genes), RASTRIGIN_PRECISION))
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

struct Outcome {
    solver: &'static str,
    generations: usize,
    best: f64,
    target: f64,
    success: bool,
}

fn print_result(args: &Args, seed: u64, outcome: &Outcome, time_s: f64, evaluations: usize) {
    println!(
        "{{\"library\":\"genetic_algorithm\",\"solver\":\"{}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{},\"time_s\":{:.6},\"generations\":{},\"evaluations\":{},\"best\":{},\"target\":{},\"success\":{}}}",
        outcome.solver,
        args.problem,
        args.size,
        args.mode,
        seed,
        time_s,
        outcome.generations,
        evaluations,
        outcome.best,
        outcome.target,
        outcome.success
    );
}

fn run<F: FnOnce(&Budget) -> Outcome>(args: &Args, seed: u64, solve: F) {
    let budget = Budget::new(args.max_evaluations);
    let (timer, timer_handle) = budget.start_timer(args.max_seconds);
    let now = Instant::now();
    let outcome = solve(&budget);
    let time_s = now.elapsed().as_secs_f64();
    drop(timer);
    timer_handle.join().unwrap();
    print_result(args, seed, &outcome, time_s, budget.evaluations());
}

fn onemax(args: &Args, seed: u64) {
    let target = args.size as isize;
    run(args, seed, |budget| {
        let genotype = BinaryGenotype::builder()
            .with_genes_size(args.size)
            .build()
            .unwrap();
        // the operator types are generic parameters of the builder, so build it per arm
        macro_rules! builder {
            () => {
                Evolve::builder()
                    .with_genotype(genotype.clone())
                    .with_fitness(OneMax(budget.clone()))
                    .with_target_fitness_score(target)
                    .with_abort_flag(budget.abort_flag.clone())
                    .with_rng_seed_from_u64(seed)
            };
        }
        // (best fitness score, generations)
        let (best, generations) = match args.mode.as_str() {
            // as close as possible to DEAP eaSimple: population 300, tournament 3, two point
            // crossover with probability 0.5, mutation probability 0.2 of ~1 bit, no elitism.
            // Difference: DEAP selects parents by tournament with replacement, this library selects
            // survivors from parents + offspring by tournament without replacement, so selection
            // pressure comes from the surplus. replacement_rate 1.0 (only offspring survive) would
            // select 300 out of 300 offspring, i.e. no selection at all, hence 0.5.
            "matched" => {
                let evolve = builder!()
                    .with_target_population_size(300)
                    .with_select(SelectTournament::new(0.5, 0.0, 3))
                    .with_crossover(CrossoverMultiPoint::new(1.0, 0.5, 2, false))
                    .with_mutate(MutateSingleGene::new(0.2))
                    .call()
                    .unwrap();
                (evolve.best_fitness_score(), evolve.state.current_generation)
            }
            // the "If unsure, start here" binary preset of AGENTS.md (population defaults to 100)
            _ => {
                let evolve = builder!()
                    .with_select(SelectTournament::new(0.5, 0.02, 4))
                    .with_crossover(CrossoverUniform::new(0.7, 0.8))
                    .with_mutate(MutateSingleGene::new(0.2))
                    .call()
                    .unwrap();
                (evolve.best_fitness_score(), evolve.state.current_generation)
            }
        };
        let best = best.unwrap_or(0);
        Outcome {
            solver: "evolve",
            generations,
            best: best as f64,
            target: target as f64,
            success: best >= target,
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
    // the "If unsure, start here" unique preset of AGENTS.md (population defaults to 100)
    run(args, seed, |budget| {
        let evolve = Evolve::builder()
            .with_genotype(nqueens_genotype(args.size))
            .with_fitness(NQueens(budget.clone()))
            .with_fitness_ordering(FitnessOrdering::Minimize)
            .with_target_fitness_score(0)
            .with_select(SelectTournament::new(0.5, 0.02, 4))
            .with_crossover(CrossoverClone::new(0.7))
            .with_mutate(MutateSingleGene::new(0.8))
            .with_abort_flag(budget.abort_flag.clone())
            .with_rng_seed_from_u64(seed)
            .call()
            .unwrap();
        let best = evolve.best_fitness_score().unwrap_or(isize::MAX);
        Outcome {
            solver: "evolve",
            generations: evolve.state.current_generation,
            best: best as f64,
            target: 0.0,
            success: best == 0,
        }
    });
    // AGENTS.md recommends HillClimb for permutation problems (see examples/hill_climb_nqueens.rs)
    run(args, seed, |budget| {
        let hill_climb = HillClimb::builder()
            .with_genotype(nqueens_genotype(args.size))
            .with_variant(HillClimbVariant::Stochastic)
            .with_fitness(NQueens(budget.clone()))
            .with_fitness_ordering(FitnessOrdering::Minimize)
            .with_target_fitness_score(0)
            .with_replace_on_equal_fitness(true)
            .with_abort_flag(budget.abort_flag.clone())
            .with_rng_seed_from_u64(seed)
            .call()
            .unwrap();
        let best = hill_climb.best_fitness_score().unwrap_or(isize::MAX);
        Outcome {
            solver: "hill_climb",
            generations: hill_climb.state.current_generation,
            best: best as f64,
            target: 0.0,
            success: best == 0,
        }
    });
}

const RASTRIGIN_TARGET: f64 = 0.01;
fn rastrigin(args: &Args, seed: u64) {
    let (low, high, function): (f64, f64, fn(&[f64]) -> f64) = match args.problem.as_str() {
        "rastrigin" => (-5.12, 5.12, rastrigin_value),
        "rosenbrock" => (-5.0, 10.0, rosenbrock_value),
        _ => (-32.768, 32.768, ackley_value),
    };
    run(args, seed, |budget| {
        let genotype = RangeGenotype::<f64>::builder()
            .with_genes_size(args.size)
            .with_allele_range(low..=high)
            // full range first, then narrowing bandwidths, advancing when stale (AGENTS.md)
            .with_mutation_type(MutationType::RangeScaled(vec![
                high - low,
                (high - low) / 2.0,
                1.0,
                0.1,
                0.01,
                0.001,
            ]))
            .build()
            .unwrap();
        let evolve = Evolve::builder()
            .with_genotype(genotype)
            .with_fitness(Rastrigin(budget.clone(), function))
            .with_fitness_ordering(FitnessOrdering::Minimize)
            .with_target_fitness_score(fitness_value(RASTRIGIN_TARGET, RASTRIGIN_PRECISION))
            .with_max_stale_generations(50)
            .with_select(SelectTournament::new(0.5, 0.02, 4))
            .with_crossover(CrossoverUniform::new(0.7, 0.8))
            .with_mutate(MutateMultiGene::new((args.size / 5).max(1), 0.8))
            .with_abort_flag(budget.abort_flag.clone())
            .with_rng_seed_from_u64(seed)
            .call()
            .unwrap();
        let best = evolve
            .best_fitness_score()
            .map(|score| score as f64 * RASTRIGIN_PRECISION)
            .unwrap_or(f64::MAX);
        Outcome {
            solver: "evolve",
            generations: evolve.state.current_generation,
            best,
            target: RASTRIGIN_TARGET,
            success: best <= RASTRIGIN_TARGET,
        }
    });
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() != 7 {
        eprintln!("usage: ga_bench_genetic_algorithm <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>");
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
            "rastrigin" | "rosenbrock" | "ackley" => rastrigin(&args, seed),
            "zdt1" | "zdt2" | "zdt3" | "dtlz1" | "dtlz2" => {}
            other => {
                eprintln!("unknown problem {}", other);
                std::process::exit(2);
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
        assert!(rastrigin_value(&s).abs() < 1e-12);
        assert!(ackley_value(&s).abs() < 1e-12);
        assert!((rastrigin_value(&x) - 87.78147018265213).abs() < 1e-9);
        assert!((ackley_value(&x) - 5.149902035382837).abs() < 1e-9);
    }
}
