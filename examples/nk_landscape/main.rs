//! NK landscape: maximize a landscape of Kauffman and Weinberger's NK model, N = 20 bits each
//! interacting with K = 4 others chosen at random, with iterated local search, and check it
//! against the optimum found by evaluating all 2^20 strings.
//!
//! The landscape is drawn from seed 1 with genoxide's portable random numbers. Iterated local
//! search flips one bit at a time, keeping changes that are no worse, and after 100 steps without
//! a better best restarts from the best, changed by 5 random flips. A run from seed 1 prints its
//! improvements; then, on the landscapes of seeds 1 to 5, runs from seeds 1 to 20 count how often
//! it reaches the optimum, against a genetic algorithm as a contrast. The landscapes are genoxide's
//! `problems::binary::NkLandscape`, whose `optimum` searches every string.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example nk_landscape
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::binary::{Neighborhood, NkLandscape};

const N: usize = 20;
const K: usize = 4;
const LANDSCAPES: u64 = 5;
const SEEDS: u64 = 20;
// the most evaluations of a run of iterated local search, and generations of the GA
const BUDGET: u64 = 500_000;
const GENERATIONS: u64 = 200;

// iterated local search from `seed`
fn ils(landscape: &NkLandscape, seed: u64) -> Result<LocalSearch<Binary, BitFlip>> {
    LocalSearch::builder(landscape.representation())
        .neighbor(BitFlip::count(1)?)
        .restart(100, 5)
        .seed(seed)
        .build()
}

// the median of `values`
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

fn main() -> Result<()> {
    let landscape = NkLandscape::new(N, K, Neighborhood::Random, 1)?;
    let optimum = landscape.optimum().expect("small enough");
    println!("NK landscape: N = {N}, K = {K}, random neighbors, drawn from seed 1");
    println!(
        "the optimum, over all 2^{N} strings: {:.6} at {}",
        optimum.value(),
        optimum.solutions()[0]
    );
    println!("iterated local search from seed 1");
    println!("evaluations  best");
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(N, optimum.value());
    let mut last = 0.0;
    let outcome = Engine::new(ils(&landscape, 1)?, landscape.clone())
        .stop_when(Stop::target(optimum.value()).or(Stop::evaluations(BUDGET)))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            let best = progress.best().and_then(Fitness::score).unwrap_or(0.0);
            if best > last {
                println!("{:>11}  {best:.6}", progress.evaluations());
                last = best;
            }
        })
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;
    println!(
        "{:.6} after {} evaluations, at {}",
        outcome.best_fitness().score().expect("valid"),
        outcome.evaluations(),
        outcome.best_genome()
    );

    // iterated local search and, as a contrast, a GA on landscapes 1 to 5, from seeds 1 to 20
    println!("\nlandscape  optimum   ILS reaches  median evaluations  GA reaches");
    for landscape_seed in 1..=LANDSCAPES {
        let landscape = NkLandscape::new(N, K, Neighborhood::Random, landscape_seed)?;
        let optimum = landscape.optimum().expect("small enough").value();
        let mut evaluations = Vec::new();
        let mut ga_reached = 0;
        for seed in 1..=SEEDS {
            let outcome = Engine::new(ils(&landscape, seed)?, landscape.clone())
                .stop_when(Stop::target(optimum).or(Stop::evaluations(BUDGET)))
                .run()?;
            if outcome.stop_reason() == StopReason::Target {
                evaluations.push(outcome.evaluations() as f64);
            }
            let ga = Ga::builder(landscape.representation())
                .population_size(500)
                .select(Tournament::new(2)?)
                .crossover(PointCrossover::two_point())
                .mutate(BitFlip::per_gene(1.0 / N as f64)?)
                .seed(seed)
                .build()?;
            let outcome = Engine::new(ga, landscape.clone())
                .stop_when(Stop::target(optimum).or(Stop::generations(GENERATIONS)))
                .run()?;
            ga_reached += usize::from(outcome.stop_reason() == StopReason::Target);
        }
        println!(
            "{landscape_seed:>9}  {optimum:.6}  {:>8}/{SEEDS}  {:>18.1}  {ga_reached:>7}/{SEEDS}",
            evaluations.len(),
            median(evaluations)
        );
    }
    trace.write();
    Ok(())
}
