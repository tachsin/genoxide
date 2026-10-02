//! Royal road R2: maximize 8 blocks of 8 ones, and the pairs, quadruples and whole string above
//! them, each scoring its number of bits when complete, with a genetic algorithm with the
//! settings of Mitchell, Forrest and Holland (1992).
//!
//! The GA has their population of 128, single-point crossover at a rate of 0.7 and a mutation
//! probability of 0.005 per bit, with tournament selection of size 2 in place of their
//! fitness-proportionate selection with sigma scaling. A run from seed 1 prints the generations at
//! which its best improves; then runs from seeds 1 to 50, as in their Table 1, give the
//! generations to the optimum, 256. As a comparison, random-mutation hill climbing, one bit at a time. The function
//! is genoxide's `problems::binary::RoyalRoad::r2`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example royal_road_r2
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::binary::RoyalRoad;

const BITS: usize = 64;
// the runs of the GA, as in the paper's Table 1, and of hill climbing
const RUNS: u64 = 50;
const CLIMBS: u64 = 200;
// the most evaluations of a run, the paper's 2000 generations of 128
const BUDGET: u64 = 256_000;

// the genetic algorithm from `seed`
fn ga(problem: &RoyalRoad, seed: u64) -> Result<Ga<Binary, Tournament, PointCrossover, BitFlip>> {
    Ga::builder(problem.representation())
        .population_size(128)
        .select(Tournament::new(2)?)
        .crossover(PointCrossover::one_point())
        .crossover_rate(0.7)
        .mutate(BitFlip::per_gene(0.005)?)
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
    let problem = RoyalRoad::r2();
    let optimum = problem.optimum().expect("known").value();
    println!("Royal road R2: 8 blocks of 8 bits and the levels above, maximum {optimum}");
    println!("a GA with single-point crossover, from seed 1");
    println!("generation  best");
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(BITS, optimum);
    let mut last = -1.0;
    let outcome = Engine::new(ga(&problem, 1)?, problem)
        .stop_when(Stop::target(optimum).or(Stop::evaluations(BUDGET)))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            let best = progress.best().and_then(Fitness::score).unwrap_or(0.0);
            // the generations at which the best improves
            if best > last {
                println!("{:>10}  {best:>4}", progress.generation());
                last = best;
            }
        })
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;
    println!(
        "{} after {} generations and {} evaluations",
        outcome.best_fitness(),
        outcome.generations(),
        outcome.evaluations()
    );

    // the GA from seeds 1 to 50, and hill climbing from seeds 1 to 200
    let mut generations = Vec::new();
    for seed in 1..=RUNS {
        let outcome = Engine::new(ga(&problem, seed)?, problem)
            .stop_when(Stop::target(optimum).or(Stop::evaluations(BUDGET)))
            .run()?;
        if outcome.stop_reason() == StopReason::Target {
            generations.push(outcome.generations() as f64);
        }
    }
    let reached = generations.len();
    let (mean, median) = mean_and_median(generations);
    println!(
        "\nGA, seeds 1 to {RUNS}: {reached} reach {optimum}; mean {mean:.0}, median {median:.0} \
         generations"
    );
    println!("  (the paper's GA, 50 runs: mean 590, median 542)");
    let mut evaluations = Vec::new();
    for seed in 1..=CLIMBS {
        let search = LocalSearch::builder(problem.representation())
            .neighbor(BitFlip::count(1)?)
            .seed(seed)
            .build()?;
        let outcome = Engine::new(search, problem)
            .stop_when(Stop::target(optimum).or(Stop::evaluations(BUDGET)))
            .run()?;
        if outcome.stop_reason() == StopReason::Target {
            evaluations.push(outcome.evaluations() as f64);
        }
    }
    let reached = evaluations.len();
    let (mean, median) = mean_and_median(evaluations);
    println!(
        "hill climbing, seeds 1 to {CLIMBS}: {reached} reach {optimum}; mean {mean:.0}, median \
         {median:.0} evaluations"
    );
    trace.write();
    Ok(())
}
