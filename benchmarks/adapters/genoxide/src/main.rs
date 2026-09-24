//! Benchmark adapter for genoxide.
//!
//! Usage: ga_bench_genoxide <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//! Prints one JSON line per solver per seed, see ../../README.md for the fields.

use genoxide::Objective::Minimize;
use genoxide::multi::problems::{Dtlz1, Dtlz2, TestProblem, Zdt1, Zdt2, Zdt3};
use genoxide::multi::{Decomposition, Moead, Nsga3, SmsEmoa, Spea2, das_dennis};
use genoxide::prelude::*;
use std::f64::consts::{E, PI};
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

// Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift
// towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
fn shift(i: usize) -> f64 {
    2.0 * ((37 * i + 11) % 101) as f64 / 101.0 - 1.0
}

fn rastrigin(genome: &Reals) -> f64 {
    10.0 * genome.len() as f64
        + genome
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let x = x - shift(i);
                x * x - 10.0 * (2.0 * PI * x).cos()
            })
            .sum::<f64>()
}

fn rosenbrock(genome: &Reals) -> f64 {
    genome
        .windows(2)
        .map(|pair| 100.0 * (pair[1] - pair[0] * pair[0]).powi(2) + (1.0 - pair[0]).powi(2))
        .sum()
}

fn ackley(genome: &Reals) -> f64 {
    let n = genome.len() as f64;
    let shifted = || genome.iter().enumerate().map(|(i, x)| x - shift(i));
    let squares = shifted().map(|x| x * x).sum::<f64>() / n;
    let cosines = shifted().map(|x| (2.0 * PI * x).cos()).sum::<f64>() / n;
    -20.0 * (-0.2 * squares.sqrt()).exp() - cosines.exp() + 20.0 + E
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

// the real-valued problems: Rastrigin, Rosenbrock and Ackley, with their bounds
fn run_real(args: &Args, seed: u64) -> Result<()> {
    let (bounds, fitness): (std::ops::RangeInclusive<f64>, fn(&Reals) -> f64) =
        match args.problem.as_str() {
            "rastrigin" => (-5.12..=5.12, rastrigin),
            "rosenbrock" => (-5.0..=10.0, rosenbrock),
            "ackley" => (-32.768..=32.768, ackley),
            other => unreachable!("unknown problem {other}"),
        };
    let real = || Real::uniform(args.size, bounds.clone());
    let rastrigin = fitness;
    let report = |solver: &str, outcome: Result<Outcome<Reals>>, time_s: f64| -> Result<()> {
        let outcome = outcome?;
        let success = outcome
            .best_fitness()
            .score()
            .is_some_and(|best| best <= RASTRIGIN_TARGET);
        print_result(
            args,
            seed,
            solver,
            &outcome,
            time_s,
            RASTRIGIN_TARGET,
            success,
        );
        Ok(())
    };

    // the settings of examples/rastrigin.rs: polynomial mutation at the usual rate of 1 / length,
    // single-threaded like every adapter
    let (outcome, time_s) = timed(|| {
        let ga = Ga::builder(real()?)
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
    report("ga", outcome, time_s)?;

    // differential evolution without tuning, as AGENTS.md suggests when F and CR are unknown:
    // SHADE with current-to-pbest/1 and an archive, and the usual population of 100
    let (outcome, time_s) = timed(|| {
        let de = De::builder(real()?)
            .population_size(100)
            .strategy(de::Strategy::CurrentToPBest {
                p: 0.1,
                archive: 1.0,
            })
            .control(de::Control::Shade { memory: 6 })
            .minimize()
            .seed(seed)
            .build()?;
        Engine::new(de, rastrigin)
            .stop_when(args.stop(RASTRIGIN_TARGET))
            .run()
    });
    report("de", outcome, time_s)?;

    // the CMA-ES template of AGENTS.md: the defaults, with IPOP restarts for a multimodal function
    let (outcome, time_s) = timed(|| {
        let cmaes = Cmaes::builder(real()?)
            .restarts(cmaes::Restarts::Ipop)
            .minimize()
            .seed(seed)
            .build()?;
        Engine::new(cmaes, rastrigin)
            .stop_when(args.stop(RASTRIGIN_TARGET))
            .run()
    });
    report("cma_es", outcome, time_s)
}

// multi-objective runs have a budget and no target: they print their final front, and run.py
// computes its hypervolume the same way for every library
fn print_front<G: Genome, const M: usize>(
    args: &Args,
    seed: u64,
    solver: &str,
    outcome: &genoxide::multi::MultiOutcome<G, M>,
    time_s: f64,
) {
    let front: Vec<String> = outcome
        .front_values()
        .iter()
        .map(|values| {
            let values: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
            format!("[{}]", values.join(","))
        })
        .collect();
    println!(
        "{{\"library\":\"genoxide\",\"solver\":\"{solver}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{},\"evaluations\":{},\"front\":[{}]}}",
        args.problem,
        args.size,
        args.mode,
        outcome.generations(),
        outcome.evaluations(),
        front.join(","),
    );
}

// the matched settings of every library: SBX with η 15 at 0.9 and polynomial mutation with η 20
// at 1 / n; MOEA/D and NSGA-III with their usual SBX (η 20 and 30 at 1)
fn run_front_problem<P, const M: usize>(
    args: &Args,
    seed: u64,
    problem: P,
    population: usize,
    divisions: usize,
    solvers: &[&str],
) -> Result<()>
where
    P: TestProblem<M> + Copy,
{
    let real = problem.real();
    let rate = 1.0 / real.bounds().len() as f64;
    let stop = || {
        Stop::evaluations(args.max_evaluations)
            .or(Stop::time(Duration::from_secs_f64(args.max_seconds)))
    };
    let objectives = [Minimize; M];
    for &solver in solvers {
        let start = Instant::now();
        let outcome = match solver {
            "nsga2" => {
                let algorithm = Nsga2::builder(real.clone(), objectives)
                    .population_size(population)
                    .crossover(SimulatedBinaryCrossover::new(15.0)?)
                    .mutate(PolynomialMutation::per_gene(rate, 20.0)?)
                    .seed(seed)
                    .build()?;
                MultiEngine::new(algorithm, problem)
                    .stop_when(stop())
                    .run()?
            }
            "spea2" => {
                let algorithm = Spea2::builder(real.clone(), objectives)
                    .population_size(population)
                    .crossover(SimulatedBinaryCrossover::new(15.0)?)
                    .mutate(PolynomialMutation::per_gene(rate, 20.0)?)
                    .seed(seed)
                    .build()?;
                MultiEngine::new(algorithm, problem)
                    .stop_when(stop())
                    .run()?
            }
            "sms_emoa" => {
                let algorithm = SmsEmoa::builder(real.clone(), objectives)
                    .population_size(population)
                    .crossover(SimulatedBinaryCrossover::new(15.0)?)
                    .mutate(PolynomialMutation::per_gene(rate, 20.0)?)
                    .seed(seed)
                    .build()?;
                MultiEngine::new(algorithm, problem)
                    .stop_when(stop())
                    .run()?
            }
            "moead" => {
                let decomposition = if M == 2 {
                    Decomposition::Tchebycheff
                } else {
                    Decomposition::Pbi { theta: 5.0 }
                };
                let algorithm =
                    Moead::builder(real.clone(), objectives, das_dennis::<M>(divisions))
                        .decomposition(decomposition)
                        .crossover(SimulatedBinaryCrossover::new(20.0)?)
                        .mutate(PolynomialMutation::per_gene(rate, 20.0)?)
                        .seed(seed)
                        .build()?;
                MultiEngine::new(algorithm, problem)
                    .stop_when(stop())
                    .run()?
            }
            "nsga3" => {
                let algorithm =
                    Nsga3::builder(real.clone(), objectives, das_dennis::<M>(divisions))
                        .population_size(population)
                        .crossover(SimulatedBinaryCrossover::new(30.0)?)
                        .mutate(PolynomialMutation::per_gene(rate, 20.0)?)
                        .seed(seed)
                        .build()?;
                MultiEngine::new(algorithm, problem)
                    .stop_when(stop())
                    .run()?
            }
            other => unreachable!("unknown solver {other}"),
        };
        print_front(args, seed, solver, &outcome, start.elapsed().as_secs_f64());
    }
    Ok(())
}

fn run_front(args: &Args, seed: u64) -> Result<()> {
    let two = ["nsga2", "spea2", "sms_emoa", "moead"];
    match args.problem.as_str() {
        "zdt1" => run_front_problem(args, seed, Zdt1::new(args.size), 100, 99, &two),
        "zdt2" => run_front_problem(args, seed, Zdt2::new(args.size), 100, 99, &two),
        "zdt3" => run_front_problem(args, seed, Zdt3::new(args.size), 100, 99, &two),
        "dtlz1" => run_front_problem(
            args,
            seed,
            Dtlz1::<3>::default(),
            92,
            12,
            &["nsga2", "nsga3", "spea2", "sms_emoa", "moead"],
        ),
        // size: the number of objectives
        "dtlz2" => run_front_problem(
            args,
            seed,
            Dtlz2::<3>::default(),
            92,
            12,
            &["nsga2", "nsga3", "spea2", "sms_emoa", "moead"],
        ),
        other => unreachable!("unknown problem {other}"),
    }
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
            "rastrigin" | "rosenbrock" | "ackley" => run_real(&args, seed)?,
            "zdt1" | "zdt2" | "zdt3" | "dtlz1" | "dtlz2" => run_front(&args, seed)?,
            other => {
                eprintln!("unknown problem {other}");
                std::process::exit(2);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod shift_tests {
    use super::*;

    #[test]
    fn shifted_functions() {
        // 0 at the shift, and the values of a Python reference at a fixed point
        let s: Vec<f64> = (0..10).map(shift).collect();
        let x: Vec<f64> = (0..10).map(|i| 0.5 * (i % 7) as f64 - 1.5).collect();
        assert!(rastrigin(&Reals::from(s.clone())).abs() < 1e-12);
        assert!(ackley(&Reals::from(s.clone())).abs() < 1e-12);
        assert!((rastrigin(&Reals::from(x.clone())) - 87.78147018265213).abs() < 1e-9);
        assert!((ackley(&Reals::from(x.clone())) - 5.149902035382837).abs() < 1e-9);
    }
}
