//! 0/1 knapsack: choose items with the highest total value that fit in the knapsack, on an
//! instance of 50 items drawn from Pisinger's uncorrelated class.
//!
//! Shows a constraint with Deb's feasibility rules: the fitness is the value and how far the
//! weight exceeds the capacity, so overweight selections still guide the search towards the
//! feasible ones. The instance is genoxide's `problems::binary::Knapsack`, generated from seed 1,
//! whose optimum dynamic programming finds. A run from seed 1 stops at it; then runs from seeds 1
//! to 20 count how often the GA reaches it, and, as a contrast, how often it reaches the optimum
//! of an instance of Pisinger's strongly correlated class, which is harder.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example knapsack
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::binary::{Knapsack, KnapsackClass};

const ITEMS: usize = 50;
const SEEDS: u64 = 20;

// the genetic algorithm from `seed`
fn ga(knapsack: &Knapsack, seed: u64) -> Result<Ga<Binary, Tournament, UniformCrossover, BitFlip>> {
    Ga::builder(knapsack.representation())
        .population_size(200)
        .select(Tournament::new(3)?)
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / ITEMS as f64)?)
        .seed(seed)
        .build()
}

// stops at the optimum that dynamic programming finds, or once the search stalls
fn stop(optimum: f64) -> Stop {
    Stop::target(optimum)
        .or(Stop::stagnation(200))
        .or(Stop::generations(2_000))
}

// the runs from seeds 1 to 20 that reach the optimum, and their median evaluations
fn seeds(knapsack: &Knapsack, optimum: f64) -> Result<(usize, f64)> {
    let mut evaluations = Vec::new();
    for seed in 1..=SEEDS {
        let outcome = Engine::new(ga(knapsack, seed)?, knapsack.clone())
            .stop_when(stop(optimum))
            .run()?;
        if outcome.stop_reason() == StopReason::Target {
            evaluations.push(outcome.evaluations() as f64);
        }
    }
    evaluations.sort_by(f64::total_cmp);
    let (count, middle) = (evaluations.len(), evaluations.len() / 2);
    let median = match count {
        0 => f64::NAN,
        n if n % 2 == 1 => evaluations[middle],
        _ => (evaluations[middle - 1] + evaluations[middle]) / 2.0,
    };
    Ok((count, median))
}

fn main() -> Result<()> {
    let knapsack = Knapsack::generator(KnapsackClass::Uncorrelated, ITEMS)
        .seed(1)
        .generate()?;
    let optimum = knapsack.optimum().expect("small enough").value();
    let total: u64 = knapsack.weights().iter().sum();
    println!(
        "{ITEMS} uncorrelated items (R = 1000, seed 1): total weight {total}, capacity {}",
        knapsack.capacity()
    );

    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(&knapsack, optimum);
    let outcome = Engine::new(ga(&knapsack, 1)?, knapsack.clone())
        .stop_when(stop(optimum))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;
    let best = outcome.best_genome();
    let items: Vec<usize> = (0..ITEMS)
        .filter(|&item| best.get(item) == Some(true))
        .collect();
    let (weight, value) = knapsack.totals(best);
    println!("items {items:?}");
    println!("value {value}, weight {weight} of {}", knapsack.capacity());
    println!(
        "after {} evaluations; the optimum, by dynamic programming, is {optimum}",
        outcome.evaluations()
    );

    let (reached, median) = seeds(&knapsack, optimum)?;
    println!(
        "\nseeds 1 to {SEEDS}: {reached} reach {optimum}, after a median of {median:.1} evaluations"
    );
    let strong = Knapsack::generator(KnapsackClass::StronglyCorrelated, ITEMS)
        .seed(1)
        .generate()?;
    let strong_optimum = strong.optimum().expect("small enough").value();
    let (reached, _) = seeds(&strong, strong_optimum)?;
    println!(
        "contrast, {ITEMS} strongly correlated items (seed 1): {reached} of {SEEDS} reach its \
         optimum, {strong_optimum}"
    );
    trace.write();
    Ok(())
}
