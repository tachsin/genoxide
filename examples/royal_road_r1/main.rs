//! Royal road R1: maximize 8 blocks of 8 ones, each scoring only when complete, with
//! random-mutation hill climbing, which Mitchell, Holland and Forrest (1994) found faster on it
//! than their genetic algorithm.
//!
//! Random-mutation hill climbing (RMHC) flips one bit, chosen at random, and keeps the change if
//! it's no worse: genoxide's `LocalSearch` with `BitFlip::count(1)`. A run from seed 1 prints the
//! evaluations at which each block is completed; then 200 runs, as in the paper's Table 1, give
//! the mean and median evaluations to the optimum, 64. As a comparison, a genetic algorithm with
//! the paper's population, crossover and mutation (and tournament selection) from 50 seeds. The
//! function is genoxide's `problems::binary::RoyalRoad::r1`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example royal_road_r1
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::binary::RoyalRoad;

const BITS: usize = 64;
// the runs of RMHC, as in the paper's Table 1, and of the genetic algorithm
const RUNS: u64 = 200;
const GA_RUNS: u64 = 50;
// the most evaluations of a run, the paper's
const BUDGET: u64 = 256_000;

// random-mutation hill climbing from `seed`
fn rmhc(problem: &RoyalRoad, seed: u64) -> Result<LocalSearch<Binary, BitFlip>> {
    LocalSearch::builder(problem.representation())
        .neighbor(BitFlip::count(1)?)
        .seed(seed)
        .build()
}

// the mean and the median of `values`
fn mean_and_median(mut values: Vec<f64>) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    let median = if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    };
    (values.iter().sum::<f64>() / values.len() as f64, median)
}

fn main() -> Result<()> {
    let problem = RoyalRoad::r1();
    let optimum = problem.optimum().expect("known").value();
    println!("Royal road R1: 8 blocks of 8 bits, maximum {optimum}");
    println!("random-mutation hill climbing from seed 1");
    println!("R1  evaluations");
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(BITS, optimum);
    let mut last = 0.0;
    let outcome = Engine::new(rmhc(&problem, 1)?, problem)
        .stop_when(Stop::target(optimum).or(Stop::evaluations(BUDGET)))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            let best = progress.best().and_then(Fitness::score).unwrap_or(0.0);
            // the evaluations at which a block is completed
            if best > last {
                println!("{best:>2}  {:>11}", progress.evaluations());
                last = best;
            }
        })
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;
    println!(
        "{} after {} evaluations",
        outcome.best_fitness(),
        outcome.evaluations()
    );

    // RMHC from seeds 1 to 200, and the genetic algorithm from seeds 1 to 50
    let mut evaluations = Vec::new();
    for seed in 1..=RUNS {
        let outcome = Engine::new(rmhc(&problem, seed)?, problem)
            .stop_when(Stop::target(optimum).or(Stop::evaluations(BUDGET)))
            .run()?;
        if outcome.stop_reason() == StopReason::Target {
            evaluations.push(outcome.evaluations() as f64);
        }
    }
    let reached = evaluations.len();
    let (mean, median) = mean_and_median(evaluations);
    println!(
        "\nRMHC, seeds 1 to {RUNS}: {reached} reach {optimum}; mean {mean:.0}, median {median:.0} \
         evaluations"
    );
    println!("  (the paper's 200 runs: mean 6179, median 5775)");
    let mut evaluations = Vec::new();
    for seed in 1..=GA_RUNS {
        let ga = Ga::builder(problem.representation())
            .population_size(128)
            .select(Tournament::new(2)?)
            .crossover(PointCrossover::one_point())
            .crossover_rate(0.7)
            .mutate(BitFlip::per_gene(0.005)?)
            .seed(seed)
            .build()?;
        let outcome = Engine::new(ga, problem)
            .stop_when(Stop::target(optimum).or(Stop::evaluations(BUDGET)))
            .run()?;
        if outcome.stop_reason() == StopReason::Target {
            evaluations.push(outcome.evaluations() as f64);
        }
    }
    let reached = evaluations.len();
    let (mean, median) = mean_and_median(evaluations);
    println!(
        "GA, seeds 1 to {GA_RUNS}: {reached} reach {optimum}; mean {mean:.0}, median {median:.0} \
         evaluations"
    );
    println!("  (the paper's GA, 200 runs: mean 61334, median 54208)");
    trace.write();
    Ok(())
}
